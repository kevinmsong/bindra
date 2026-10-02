"""The ``finalize`` half of ``bindra lock-benchmarks``.

Kept in its own module because it depends on the splits, while ``candidates`` must
run before them. Importing the split-dependent code from the pre-split command
would make the ordering constraint invisible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OK = "  ok   "
FAIL = " FAIL  "
INFO = " info  "

SPLIT_COLUMNS = (
    "observation_id",
    "pair_key",
    "identity_key",
    "leakage_key",
    "scaffold_key",
    "sequence_sha256",
    "endpoint",
    "p_value",
    "is_exact",
    "is_censored",
    "lower_bound",
    "upper_bound",
    "comparability_group_id",
    # The target-independent panel key the selectivity panel compares on.
    "citation_key",
    "assay_condition_bucket",
    "curation_source_class",
    "ic50_tier",
    "first_evidence_date",
    "date_in_bindingdb",
)


def _corpus(processed: Path):
    """The curated corpus, verified against the hash curation recorded."""
    import pandas as pd

    from bindra.data.tables import ShardedTableRef

    payload = json.loads((processed / "qc_report.json").read_text(encoding="utf-8"))
    reference = ShardedTableRef.from_dict(payload["artifacts"]["observations"])
    reference.verify()
    frame = pd.concat(
        list(reference.iter_frames(columns=list(SPLIT_COLUMNS))), ignore_index=True
    )
    print(f"[{OK}] corpus verified ({reference.rows:,} observations)")
    return frame, payload


def _inputs(processed: Path, payload: dict[str, Any]) -> dict[str, Any]:
    from bindra.config.profiles import get_profile

    profile = get_profile()
    relation_meta = processed / "protein_relation" / "relation_table.json"
    relation_sha = None
    if relation_meta.exists():
        relation_sha = json.loads(relation_meta.read_text(encoding="utf-8")).get("sha256")
    return {
        "snapshot_release": profile.snapshot_release,
        "curation_profile_hash": profile.profile_hash,
        "observations_combined_sha256": payload["artifacts"]["observations"][
            "combined_sha256"
        ],
        "observations_row_count": payload["artifacts"]["observations"]["rows"],
        "source_classes_sha256": payload.get("source_classes_sha256"),
        "protein_relation_sha256": relation_sha,
        "error_model": profile.error_model.to_dict()["sigma_rep"],
    }


def _test_ids(splits: Path, regime: str) -> set[str]:
    from bindra.data.splits import Partition, read_split

    directory = splits / regime
    if not (directory / "manifest.json").exists():
        return set()
    _, assignment = read_split(directory)
    return {o for o, p in assignment.items() if p == Partition.TEST.value}


def finalize(args) -> int:
    processed = Path(args.processed)
    out = Path(args.out)
    splits = Path(args.splits)
    frame, payload = _corpus(processed)
    inputs = _inputs(processed, payload)
    status = 0

    # -- activity cliff ---------------------------------------------------
    cliff_dir = out / "activity_cliff"
    candidates_path = cliff_dir / "candidates.parquet"
    if candidates_path.exists():
        import pandas as pd

        from bindra.eval.benchmarks.activity_cliff import CliffDefinition, lock_cliff_panel

        candidates = pd.read_parquet(candidates_path)
        definition_payload = json.loads(
            (cliff_dir / "candidates_report.json").read_text(encoding="utf-8")
        )["definition"]
        definition = CliffDefinition(**definition_payload)
        held = _test_ids(splits, "cold_drug")
        if not held:
            print(f"[{INFO}] no cold_drug split yet; cliff panel not finalized")
        else:
            manifest = lock_cliff_panel(
                candidates,
                definition=definition,
                inputs=inputs,
                directory=cliff_dir / "panel",
                test_observations=held,
                overwrite=args.overwrite,
            )
            print(
                f"[{OK}] activity_cliff locked: {manifest.counts['queries']:,} pairs "
                f"({manifest.attrition.get('dropped_not_in_test', 0):,} dropped as not in test)"
            )
    else:
        print(f"[{INFO}] no cliff candidates at {candidates_path}; run `candidates` first")

    # -- temporal ---------------------------------------------------------
    from bindra.eval.benchmarks.temporal import build_temporal_panels, lock_temporal_panels

    panels, report = build_temporal_panels(frame)
    lock_temporal_panels(
        panels, report, inputs=inputs, directory=out, overwrite=args.overwrite
    )
    for name, counts in report["counts"].items():
        print(f"[{OK}] {name} locked: {counts['n_observations']:,} observations")

    # -- selectivity ------------------------------------------------------
    import pandas as pd

    from bindra.eval.benchmarks.selectivity import (
        SelectivityDefinition,
        build_selectivity_triples,
        load_seed_pairs,
        lock_selectivity_panel,
        related_target_pairs,
    )

    constructs_path = processed / "target_constructs.parquet"
    constructs = pd.read_parquet(
        constructs_path, columns=["sequence_sha256", "construct_id", "uniprot_primary", "status"]
    )
    constructs = constructs[constructs["status"] == "OK"]
    seed_pairs, seed_sha = load_seed_pairs()
    relation = None
    relation_dir = processed / "protein_relation"
    if (relation_dir / "relation_table.json").exists():
        from bindra.data.homology import ProteinRelationTable

        relation = ProteinRelationTable.load(relation_dir)
    else:
        print(f"[{INFO}] no relation table; selectivity uses the curated seed list only")

    related, related_report = related_target_pairs(
        constructs, relation=relation, seed_pairs=seed_pairs
    )
    definition = SelectivityDefinition(seed_pairs_sha256=seed_sha)
    exact, censored, unknowns, sel_report = build_selectivity_triples(
        frame, related, definition=definition, report=related_report
    )
    print(
        f"[{OK}] related target pairs: {related_report.n_related_pairs_alignment:,} from "
        f"alignment, {related_report.n_related_pairs_seed:,} from the curated list "
        f"({related_report.n_seed_pairs_matched}/{related_report.n_seed_pairs_listed} listed pairs matched)"
    )
    if len(exact) or len(censored):
        lock_selectivity_panel(
            exact,
            censored,
            unknowns,
            definition=definition,
            inputs=inputs,
            directory=out,
            report=sel_report,
            overwrite=args.overwrite,
        )
        print(
            f"[{OK}] selectivity locked: {len(exact):,} exact triples, "
            f"{len(censored):,} interval triples, {len(unknowns):,} unknowns"
        )
    else:
        print(f"[{INFO}] no selectivity triples at tier 1; nothing locked")

    # -- fishing ----------------------------------------------------------
    if args.skip_fishing:
        print(f"[{INFO}] fishing panel skipped by request")
        return status

    groups_path = processed / "protein_groups.parquet"
    if not groups_path.exists():
        print(f"[{INFO}] no protein groups; fishing panel needs homology-aware exclusion")
        return status

    from bindra.data.properties import build_property_store
    from bindra.eval.benchmarks.fishing import (
        FishingDefinition,
        build_fishing_challenges,
        lock_fishing_panel,
    )

    print(f"[{INFO}] building molecular properties (cached after the first run)...")
    properties = build_property_store(processed, workers=args.workers)
    print(f"[{OK}] properties: {properties.to_meta()}")

    protein_groups = pd.read_parquet(
        groups_path, columns=["sequence_sha256", "protein_group"]
    )
    fishing_definition = FishingDefinition()
    challenges, fishing_report = build_fishing_challenges(
        frame,
        protein_groups=protein_groups,
        properties=properties,
        definition=fishing_definition,
        max_targets=args.max_targets or None,
    )
    if len(challenges):
        lock_fishing_panel(
            challenges,
            definition=fishing_definition,
            inputs=inputs,
            directory=out,
            report=fishing_report,
            overwrite=args.overwrite,
        )
        print(
            f"[{OK}] fishing locked: {fishing_report.n_targets_locked:,} targets, "
            f"{fishing_report.n_actives:,} actives, "
            f"{fishing_report.n_measured_inactives:,} measured inactives, "
            f"{fishing_report.n_decoys:,} presumed-control decoys"
        )
        if fishing_report.decoy_shortfall:
            print(
                f"[{INFO}] decoy shortfall {fishing_report.decoy_shortfall:,}: the pool "
                "could not fill the quota inside the declared tolerances, which is "
                "reported rather than fixed by loosening the match"
            )
    else:
        print(f"[{INFO}] no target met the minimum active count; nothing locked")
    return status


__all__ = ["finalize"]
