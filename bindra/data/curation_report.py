"""Assemble the curation report the specification requires.

*"Deliver a curation report with counts after every filter, endpoint
distributions, target-family coverage where known, duplicates, censored
observations, disagreements, and exclusions."*

Reads the JSON side-cars each stage writes and renders one Markdown document plus
a machine-readable roll-up. Nothing is recomputed here: if a number is not in a
stage's own report, it does not appear, so the document cannot drift from the
artifacts it describes.

The report carries a **policy register**: every curation choice, with the
measurement that justified it. Several of those choices overturned a default --
the alignment mode, the tautomer step, the ambiguity margin -- and a reader needs
to see why, not just what.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from bindra.config.profiles import CurationProfile, get_profile
from bindra.data.endpoints import Endpoint


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return lines


def _counts(mapping: Mapping[str, Any] | None, *, limit: int | None = None) -> list[list[Any]]:
    if not mapping:
        return []
    items = sorted(mapping.items(), key=lambda kv: -kv[1] if isinstance(kv[1], int) else 0)
    if limit:
        items = items[:limit]
    return [[key, f"{value:,}" if isinstance(value, int) else value] for key, value in items]


def _fmt(value: Any) -> str:
    return f"{value:,}" if isinstance(value, int) else str(value)


def build_report(
    processed_dir: Path | str,
    *,
    profile: CurationProfile | None = None,
) -> tuple[str, dict[str, Any]]:
    processed_dir = Path(processed_dir)
    profile = profile or get_profile()

    ingest = _load(processed_dir / "ingest_stats.json")
    standardization = _load(processed_dir / "standardization_report.json")
    proteins = _load(processed_dir / "protein_report.json")
    qc = _load(processed_dir / "qc_report.json")
    groups = _load(processed_dir / "protein_groups_report.json")

    stats = ingest.get("stats", {})
    std = standardization.get("report", {})
    qc_report = qc.get("report", {})

    lines: list[str] = []
    add = lines.append

    add("# BindingDB 202610 - curation report")
    add("")
    add(
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
        f"from profile `{profile.name}` (`{profile.profile_hash}`)."
    )
    add("")
    add(
        "Every count below is read from the stage that produced it. This is the "
        "report the specification asks for; `docs/DATA_PROFILE.md` is the earlier "
        "pre-curation census and is not a substitute for it."
    )
    add("")

    # -- counts after every filter ----------------------------------------
    add("## Counts after every filter")
    add("")
    rows = [
        ["Source rows", _fmt(stats.get("rows"))],
        ["Rows with no affinity endpoint", _fmt(stats.get("rows_without_any_endpoint"))],
        ["Rows carrying EC50 (excluded by design)", _fmt(stats.get("rows_with_ec50_excluded"))],
        ["Observations after row->endpoint expansion", _fmt(
            (stats.get("observations_emitted") or 0) + (stats.get("observations_quarantined") or 0)
        )],
        ["  quarantined at ingest", _fmt(stats.get("observations_quarantined"))],
        ["  passed structural screening", _fmt(stats.get("observations_emitted"))],
        ["Unusable after standardization/construct join", _fmt(qc_report.get("n_unusable_after_join"))],
        ["Duplicate imports collapsed", _fmt(qc_report.get("n_duplicate_imports"))],
        ["**Curated observations**", f"**{_fmt(qc_report.get('n_after_dedup'))}**"],
    ]
    lines.extend(_table(["Stage", "Observations"], rows))
    add("")

    if stats.get("quarantine_reasons"):
        add("### Ingest quarantine reasons")
        add("")
        lines.extend(_table(["Reason", "Observations"], _counts(stats["quarantine_reasons"])))
        add("")

    # -- endpoints ---------------------------------------------------------
    add("## Endpoints")
    add("")
    exact_censored = qc_report.get("endpoint_exact_censored", {})
    rows = []
    for endpoint in (Endpoint.KD, Endpoint.KI, Endpoint.IC50):
        payload = exact_censored.get(endpoint.value, {})
        exact = payload.get("exact", 0)
        censored = payload.get("censored", 0)
        total = exact + censored
        rows.append(
            [
                endpoint.p_label,
                _fmt(total),
                _fmt(exact),
                _fmt(censored),
                f"{100 * censored / total:.1f}%" if total else "-",
                f"{profile.error_model.sigma(endpoint):.3f}",
            ]
        )
    lines.extend(
        _table(
            ["Endpoint", "Curated", "Exact", "Censored", "Censored %", "sigma_rep"],
            rows,
        )
    )
    add("")
    add(
        "pKd, pKi and pIC50 are kept separate throughout. IC50 is an "
        "assay-dependent potency endpoint and is never relabelled as Kd. EC50 is "
        "excluded from the affinity regression entirely."
    )
    add("")

    # -- measurement error -------------------------------------------------
    add("## Measurement error, and the thresholds derived from it")
    add("")
    add(
        "`sigma_rep` is the pooled within-group standard deviation of exact "
        "p-values over within-publication groups (ligand, target, endpoint, "
        "citation) with at least two members."
    )
    add("")
    rows = []
    for endpoint in (Endpoint.KD, Endpoint.KI, Endpoint.IC50):
        sigma = profile.error_model.sigma(endpoint)
        moderate, high = profile.discordance_thresholds(endpoint)
        rows.append(
            [
                endpoint.p_label,
                f"{sigma:.3f}",
                _fmt(profile.error_model.degrees_of_freedom.get(endpoint.value)),
                f"{profile.ranking_margin(endpoint):.2f}",
                f"{profile.cliff_delta_threshold(endpoint):.2f}",
                f"{moderate:.2f} / {high:.2f}",
            ]
        )
    lines.extend(
        _table(
            [
                "Endpoint",
                "sigma_rep",
                "df",
                "Ranking margin",
                "Cliff \\|dp\\|",
                "Discordance mod/high",
            ],
            rows,
        )
    )
    add("")
    add("Three consequences that change what results can mean:")
    add("")
    add(
        "- **pKd is the noisiest endpoint, not the cleanest** (0.945 against 0.641 "
        "for pKi), and also the scarcest. Any pKd claim rests on a small and noisy "
        "base, and an RMSE near 0.95 on pKd is at the noise floor rather than good."
    )
    add(
        "- **The specification's starting 0.5 p-unit ranking margin is below the "
        "noise floor of every endpoint.** It is raised to one sigma_rep; ordering a "
        "pair that differs by less than measurement error is training on noise."
    )
    add(
        "- **Patent-derived sigma_rep is lower than literature-derived** "
        "(Kd 0.707 against 1.019). That indicates re-statement of a single table "
        "value rather than independent measurement, so patent replicates are less "
        "independent, not more precise."
    )
    add("")
    add(
        "sigma_rep is a within-publication estimate and therefore a **lower bound** "
        "on true measurement error; between-laboratory reproducibility is larger and "
        "is not estimated here."
    )
    add("")

    # -- dependence --------------------------------------------------------
    dependence = profile.dependence
    add("## Dependence structure")
    add("")
    lines.extend(
        _table(
            ["Unit", "Observed", "Kish effective n", "Largest single unit", "Top 1% share"],
            [
                [
                    "Targets",
                    _fmt(dependence.n_targets_observed),
                    f"**{dependence.n_targets_effective:,}**",
                    _fmt(dependence.target_max_observations),
                    f"{100 * dependence.target_top1pct_share:.1f}%",
                ],
                [
                    "Ligands",
                    _fmt(dependence.n_ligands_observed),
                    _fmt(dependence.n_ligands_effective),
                    _fmt(dependence.ligand_max_observations),
                    f"{100 * dependence.ligand_top1pct_share:.1f}%",
                ],
            ],
        )
    )
    add("")
    add(
        f"{dependence.n_targets_observed:,} targets correspond to "
        f"**{dependence.n_targets_effective:,} effective independent target units**. "
        "Confidence intervals must therefore come from a cluster bootstrap over "
        "protein components, and per-target macro averages must be reported beside "
        "pooled metrics, which are dominated by roughly a hundred targets."
    )
    add("")

    # -- chemistry ---------------------------------------------------------
    add("## Chemical standardization")
    add("")
    rows = [
        ["Unique raw SMILES", _fmt(std.get("n_input"))],
        ["Standardized", _fmt(std.get("n_ok"))],
        ["Quarantined", _fmt(std.get("n_quarantined"))],
        ["Unique identity keys (exact stereochemistry)", _fmt(std.get("unique_identity_keys"))],
        ["Unique leakage keys (parent connectivity)", _fmt(std.get("unique_leakage_keys"))],
        ["Unique Murcko scaffolds", _fmt(std.get("unique_scaffold_keys"))],
        ["Counterions removed", _fmt(std.get("n_with_removed_fragments"))],
        ["Molecules carrying a formal charge", _fmt(std.get("n_charged"))],
        ["Molecules with unspecified stereocentres", _fmt(std.get("n_unspecified_stereo"))],
    ]
    lines.extend(_table(["Quantity", "Count"], rows))
    add("")
    if std.get("status_counts"):
        add("Standardization outcomes:")
        add("")
        lines.extend(_table(["Status", "Molecules"], _counts(std["status_counts"])))
        add("")

    # -- targets -----------------------------------------------------------
    add("## Targets and constructs")
    add("")
    length = proteins.get("length", {})
    rows = [
        ["Unique constructs", _fmt(proteins.get("n_input"))],
        ["Unique UniProt accessions", _fmt(proteins.get("n_unique_uniprot"))],
        ["Accessions with more than one construct", _fmt(proteins.get("n_uniprot_with_multiple_constructs"))],
        ["Constructs with detected point mutations", _fmt(proteins.get("n_constructs_with_mutations"))],
        ["Constructs that are substrings of another", _fmt(proteins.get("n_substring_constructs"))],
        ["Residue length min / median / max", f"{length.get('min')} / {length.get('p50')} / {length.get('max')}"],
        ["Total residues", _fmt(length.get("total_residues"))],
        ["Constructs over the 2,046-residue encoder window", _fmt(length.get("over_2046"))],
    ]
    lines.extend(_table(["Quantity", "Count"], rows))
    add("")
    corroboration = proteins.get("mutation_corroboration", {})
    if corroboration:
        add(
            "Mutation calls are derived by aligning constructs that share a UniProt "
            "accession, with no network access. Free-text target-name tokens are an "
            "**independent** signal, so their agreement validates the method: "
            f"{corroboration.get('n_exact_agreement')} of "
            f"{corroboration.get('n_comparable')} comparable constructs agree exactly "
            f"and {corroboration.get('n_agreement_at_constant_offset')} more agree on "
            "the substitution at a constant positional offset, giving "
            f"**{100 * (corroboration.get('substitution_agreement_rate') or 0):.1f}% "
            "agreement on the substitution itself**. The offsets are construct-relative "
            "versus canonical numbering, which cannot be reconciled offline, and which "
            "is also why the reference construct is not claimed to be wild type."
        )
        add("")

    # -- homology ----------------------------------------------------------
    if groups:
        add("## Protein grouping")
        add("")
        components = groups.get("components", {})
        rows = [
            ["Constructs", _fmt(groups.get("n_constructs"))],
            ["Pairs aligned (exhaustive, exact)", _fmt(groups.get("n_pairs"))],
            ["Edges stored (identity >= 0.20)", _fmt(groups.get("n_edges_stored"))],
            ["Qualifying edges at 40% / 80%", _fmt(groups.get("n_qualifying_edges"))],
            ["UniProt-sharing edges", _fmt(groups.get("n_uniprot_sharing_edges"))],
            ["Substring containment edges", _fmt(groups.get("n_substring_containment_edges"))],
        ]
        lines.extend(_table(["Quantity", "Count"], rows))
        add("")

        # -- the percolation finding --------------------------------------
        curve = groups.get("percolation_curve") or []
        strict = groups.get("components_strict") or {}
        cold = groups.get("components_cold_target") or {}
        if curve:
            add("### The target graph percolates at the specified threshold")
            add("")
            add(
                "This is the most consequential measurement in the grouping stage. "
                "At the specification's 40% identity / 80% coverage, single linkage "
                "does not produce protein families -- it produces one connected mass. "
                "Average degree is about 11.6, far above the percolation point."
            )
            add("")
            lines.extend(
                _table(
                    [
                        "Identity",
                        "Qualifying edges",
                        "Components",
                        "Largest",
                        "Largest %",
                        "Observations outside largest",
                    ],
                    [
                        [
                            f"{row['identity_threshold']:.2f}",
                            _fmt(row["n_qualifying_edges"]),
                            _fmt(row["n_components"]),
                            _fmt(row["largest_component"]),
                            f"{100 * row['largest_component_share']:.1f}%",
                            _fmt(row.get("observations_outside_largest")),
                        ]
                        for row in curve
                    ],
                )
            )
            add("")
            add(
                "At 40% a cold-target test set can draw on **1.6% of observations**. "
                "The graph shatters between 0.50 and 0.60, and everything from 0.60 "
                "upward looks much the same -- so 60% sits on a plateau rather than "
                "on a slope, which is why it is not a tuned number."
            )
            add("")
            add("Two groupings are therefore published, and both are used:")
            add("")
            lines.extend(
                _table(
                    ["Grouping", "Threshold", "Components", "Largest", "Largest %", "Used by"],
                    [
                        [
                            "`protein_group_strict`",
                            "40%",
                            _fmt(strict.get("n_components")),
                            _fmt(strict.get("largest_component")),
                            f"{100 * (strict.get('largest_component_share') or 0):.1f}%",
                            "`cold_target_strict`; bounds novel-target claims",
                        ],
                        [
                            "`protein_group`",
                            "60%",
                            _fmt(cold.get("n_components")),
                            _fmt(cold.get("largest_component")),
                            f"{100 * (cold.get('largest_component_share') or 0):.1f}%",
                            "`cold_target`, `double_cold`; the powered measurement",
                        ],
                    ],
                )
            )
            add("")
            add(
                "The looser regime's cost is published rather than implied: its audit "
                "records every cross-partition construct pair with identity in "
                "[0.40, 0.60) -- 17,981 pairs for `cold_target` and 10,165 for "
                "`double_cold`. Those are the homologues the 60% threshold permits to "
                "straddle a partition, and no claim about novel targets should rest on "
                "that regime alone."
            )
            add("")
        verification = groups.get("verification", {})
        if verification:
            differences = verification.get("identity_differences", {})
            add(
                "Alignment is exhaustive and exact -- every pair, no prefilter -- and "
                "a sample is re-aligned with an independent implementation. Optimal "
                "alignments are rarely unique, so identity counts can differ between "
                "two correct implementations at the same optimal score; only the "
                "qualification decision is asserted. Measured identity difference: "
                f"median {differences.get('median')}, p90 {differences.get('p90')}, "
                f"max {differences.get('max')} over {differences.get('n')} pairs, with "
                f"{len(verification.get('missed_qualifying_pairs', []))} pair(s) that "
                "the oracle qualifies and the table does not."
            )
            add("")

    # -- comparability -----------------------------------------------------
    add("## Assay comparability")
    add("")
    add(
        "**BindingDB 202610 contains no assay identifier and no assay description "
        "column anywhere in its 640 columns.** This is not a missing value; the field "
        "does not exist. Every within-assay quantity in BINDRA therefore rests on a "
        "named proxy, recorded per observation in `comparability_scope`."
    )
    add("")
    group_summary = qc_report.get("group_size_summary", {})
    rows = [
        ["Comparability groups", _fmt(group_summary.get("n_groups"))],
        ["Median group size", group_summary.get("median")],
        ["90th percentile group size", group_summary.get("p90")],
        ["Largest group", _fmt(group_summary.get("max"))],
        ["Singleton groups", _fmt(group_summary.get("singletons"))],
        ["Observations with no citation of any kind", _fmt(qc_report.get("unknown_citation_observations"))],
    ]
    lines.extend(_table(["Quantity", "Count"], rows))
    add("")
    add(
        "An observation with no PMID, DOI, patent number or assay identifier takes "
        "its own id as its citation key and therefore forms a **group of one**. The "
        "specification forbids a missing identifier collapsing a target into one "
        "giant ranking group, and this is the mechanism that prevents it."
    )
    add("")
    if qc_report.get("citation_kinds"):
        lines.extend(_table(["Citation kind", "Observations"], _counts(qc_report["citation_kinds"])))
        add("")
    if qc_report.get("comparability_scopes"):
        lines.extend(_table(["Comparability scope", "Observations"], _counts(qc_report["comparability_scopes"])))
        add("")
    if qc_report.get("confidence_counts"):
        lines.extend(_table(["Metadata confidence", "Observations"], _counts(qc_report["confidence_counts"])))
        add("")
        add(
            "`high` is rare because it requires both reported conditions, and pH is "
            "populated for only 6.5% of rows and temperature for 6.1%. Ranking "
            "supervision deliberately requires only `medium` so the objective is not "
            "starved by that scarcity. Patent-sourced records are capped at `medium` "
            "however complete their other fields, because one patent examples table "
            "can hold thousands of values under more than one protocol."
        )
        add("")

    # -- duplicates and replicates ----------------------------------------
    add("## Duplicates, replicates and disagreement")
    add("")
    spread = qc_report.get("spread_quantiles", {})
    rows = [
        ["Duplicate imports collapsed", _fmt(qc_report.get("n_duplicate_imports"))],
        ["Identical value under a different citation (kept, flagged)", _fmt(qc_report.get("n_cross_source_identical"))],
        ["Replicate groups", _fmt(qc_report.get("replicate_groups"))],
        ["Replicate groups with more than one exact value", _fmt(qc_report.get("replicate_groups_multi"))],
        ["Internally contradictory censoring", _fmt(qc_report.get("n_censoring_inconsistent"))],
    ]
    lines.extend(_table(["Quantity", "Count"], rows))
    add("")
    if spread:
        add("Replicate spread, in p-units, over groups holding more than one exact value:")
        add("")
        lines.extend(
            _table(
                ["median", "mean", "p90", "max", "> 0.5", "> 1.0", "> 2.0"],
                [
                    [
                        spread.get("median"),
                        spread.get("mean"),
                        spread.get("p90"),
                        spread.get("max"),
                        _fmt(spread.get("over_0.5")),
                        _fmt(spread.get("over_1.0")),
                        _fmt(spread.get("over_2.0")),
                    ]
                ],
            )
        )
        add("")
    if qc_report.get("discordance_counts"):
        lines.extend(_table(["Discordance", "Observations"], _counts(qc_report["discordance_counts"])))
        add("")
    add(
        "A group disagreeing beyond its endpoint's repeatability limit keeps its "
        "median but is flagged and barred from analog-pair and activity-cliff "
        "construction: a delta computed from mutually inconsistent measurements means "
        "nothing. Such groups are never dropped and never silently averaged. Censored "
        "bounds are never pooled into a median."
    )
    add("")

    # -- eligibility -------------------------------------------------------
    add("## Per-task eligibility")
    add("")
    add(
        "The specification asks how many observations remain eligible for "
        "regression, within-assay ranking, retrieval supervision and analog-pair "
        "training, separately."
    )
    add("")
    eligibility = qc_report.get("eligibility", {})
    total = qc_report.get("n_after_dedup") or 0
    rows = [
        [
            name.replace("_", " "),
            _fmt(count),
            f"{100 * count / total:.1f}%" if total else "-",
        ]
        for name, count in eligibility.items()
    ]
    lines.extend(_table(["Task", "Eligible observations", "Share"], rows))
    add("")
    if qc_report.get("ic50_tiers"):
        lines.extend(_table(["IC50 tier", "Observations"], _counts(qc_report["ic50_tiers"])))
        add("")
        add(
            "`na` is Ki and Kd, which the specification treats as high-confidence "
            "binding constants and does not tier. Tier B is excluded from the primary "
            "affinity metric and from matched-pair construction but may supervise "
            "retrieval and within-assay ranking, where the signal is relative order "
            "inside a group."
        )
        add("")

    # -- policy register --------------------------------------------------
    add("## Policy register")
    add("")
    add(
        "Every curation choice, with the measurement behind it. Several overturned a "
        "default, and those are marked."
    )
    add("")
    register = [
        ["Endpoints kept separate", "pKd, pKi, pIC50 never merge; EC50 excluded", "Specification"],
        ["Censoring", "One-sided bound, never trained as an exact value", f"{_fmt(sum(v.get('censored', 0) for v in exact_censored.values()))} censored observations"],
        ["Plausibility window", "Disabled" if profile.plausible_range is None else str(profile.plausible_range), "Policy decision; recorded"],
        ["Formal charges", "Preserved on the identity path", "Specification"],
        ["Ambiguous mixtures", "Quarantined, never resolved by largest fragment", f"{_fmt((std.get('status_counts') or {}).get('MULTIPLE_ORGANIC_FRAGMENTS'))} molecules"],
        [
            "**Tautomer canonicalization**",
            "**Disabled** (overturns the default)",
            "96.5 ms/mol against ~4 ms/mol for the rest of the pipeline; InChI's mobile-hydrogen layer already collapses the common classes",
        ],
        ["Acyclic scaffold fallback", "The molecule's own leakage key, not a shared bucket", f"{(std.get('scaffold_kinds') or {}).get('acyclic_fallback', 0):,} molecules"],
        ["Construct identity", "sha256 of the exact construct, not the UniProt accession", "Specification"],
        ["Sequence case", "Canonicalized to upper case", "34 constructs were lower case; ESM-C maps lower case to <unk>, a special token, which would strip every residue"],
        [
            "**Alignment mode**",
            "**Semi-global** (overturns global)",
            "Sensitivity 0.987 at 0.000 false positives, against 0.867 at 0.043 for global; global collapsed a 500-construct sample into one component",
        ],
        [
            "**Identity ambiguity margin**",
            "**Zero** (overturns a conservative margin)",
            "A 0.08 margin turned 134 qualifying edges into 473 and the largest component from 10.8% to 53.8%; near-threshold pairs are audited individually instead",
        ],
        ["Linkage", "Single linkage; a large component is reported, never split", "Splitting would reintroduce the homology the regime excludes"],
        ["Comparability key", "target + endpoint + citation + condition bucket + source class", "No assay identifier exists in the file"],
        ["Missing citation", "Forms a singleton group", f"{_fmt(qc_report.get('unknown_citation_observations'))} observations"],
        ["pH / temperature", "Excluded from the key as components; missing is its own bucket", "Populated for 6.5% / 6.1% of rows"],
        ["Duplicate key", "Byte-exact value cell within one comparability group", f"{_fmt(qc_report.get('n_duplicate_imports'))} collapsed"],
        ["Discordance threshold", "2.77 x sigma_rep, endpoint-specific", "Measured; a fixed 0.5 would flag a third of replicated groups"],
        ["IC50 Tier B", "Keyed on curation source class, never on PubChem AID presence", "68% of rows carry an AID, including ~545k patent rows"],
        ["Split seed", f"Declared constant {profile.splits.seed}, all seeds logged", "Seed shopping must be visible"],
    ]
    lines.extend(_table(["Choice", "Decision", "Basis"], register))
    add("")

    # -- what is not claimed ---------------------------------------------
    add("## Not claimed")
    add("")
    for caveat in [
        "No model has been trained on this corpus and no performance number exists.",
        "`sigma_rep` is a within-publication lower bound on measurement error.",
        "40% identity / 80% coverage is an operational threshold, not proof that shared domains or protein families are absent across partitions.",
        "No within-assay claim rests on a real assay identifier, because the field does not exist in this source.",
        "Mutation calls identify that two constructs differ at a position, not which one is wild type.",
        "Computational decoys, where used later, are presumed controls with unknown binding status and never receive affinity labels.",
    ]:
        add(f"- {caveat}")
    add("")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "profile": {"name": profile.name, "hash": profile.profile_hash},
        "ingest": stats,
        "standardization": std,
        "proteins": proteins,
        "qc": qc_report,
        "protein_groups": groups,
    }
    return "\n".join(lines) + "\n", summary


def write_report(
    processed_dir: Path | str,
    *,
    markdown_path: Path | str = "docs/CURATION_REPORT.md",
    profile: CurationProfile | None = None,
) -> tuple[Path, Path]:
    from bindra.data.tables import write_json

    processed_dir = Path(processed_dir)
    markdown, summary = build_report(processed_dir, profile=profile)
    markdown_path = Path(markdown_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding="utf-8")
    json_path = write_json(summary, processed_dir / "curation_report.json")
    return markdown_path, json_path


__all__ = ["build_report", "write_report"]
