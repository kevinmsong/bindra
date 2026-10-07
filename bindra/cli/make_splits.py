"""``bindra make-splits`` -- build and audit the immutable split manifests.

Builds all five regimes over the curated corpus, audits each against the data
rather than against group labels, and writes one manifest per regime. Manifests
are immutable: the command refuses to overwrite an existing regime directory.

The audit is the point of this command as much as the split is. Every gating check
recomputes the relation it tests -- pair integrity from ``pair_key``, chemical
disjointness from the standardized keys, protein disjointness from the exhaustive
alignment table -- because comparing group ids would pass even if the grouping
itself were wrong.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import json
from pathlib import Path
import sys

OK = "  ok   "
FAIL = " FAIL  "
INFO = " info  "

#: Columns the split and audit stages need from the curated table.
SPLIT_COLUMNS = (
    "observation_id",
    "pair_key",
    "identity_key",
    "leakage_key",
    "scaffold_key",
    "sequence_sha256",
    "endpoint",
    "is_exact",
    "is_censored",
    "first_evidence_date",
    "date_in_bindingdb",
)


def _load_corpus(processed: Path):
    """Load the curated corpus, verified against the hash the curation recorded.

    A directory listing is not a dataset. ``curate`` clears ``observations/`` and
    rewrites it shard by shard, so a glob run mid-rewrite returns a *partial*
    corpus that looks entirely healthy -- which is how an earlier check here read
    686,202 of 2,739,920 observations and reported plausible split sizes for a
    quarter of the data. Going through the recorded ``ShardedTableRef`` makes a
    partial or altered corpus an error instead of a silent subset.
    """
    import pandas as pd

    from bindra.data.tables import ShardedTableRef, TableError

    report_path = processed / "qc_report.json"
    if not report_path.exists():
        raise SystemExit(
            f"{report_path} is missing. Run `bindra prepare-data curate` first."
        )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    reference = ShardedTableRef.from_dict(payload["artifacts"]["observations"])
    try:
        reference.verify()
    except TableError as exc:
        raise SystemExit(
            f"the curated observation table does not match qc_report.json: {exc}\n"
            "If a curation run is still in progress, wait for it to finish; "
            "otherwise re-run `bindra prepare-data curate`."
        ) from exc

    frame = pd.concat(
        list(reference.iter_frames(columns=list(SPLIT_COLUMNS))), ignore_index=True
    )
    if len(frame) != reference.rows:
        raise SystemExit(
            f"read {len(frame):,} rows but the manifest records {reference.rows:,}"
        )
    print(f"[{OK}] corpus verified against qc_report.json ({reference.rows:,} rows)")

    groups_path = processed / "protein_groups.parquet"
    if not groups_path.exists():
        raise SystemExit(
            f"{groups_path} is missing. Run `bindra prepare-data homology` first; "
            "cold-target and double-cold splits cannot be built without the "
            "exhaustive alignment table."
        )
    columns = ["sequence_sha256", "protein_group"]
    available = set(pd.read_parquet(groups_path).columns)
    if "protein_group_strict" in available:
        columns.append("protein_group_strict")
    protein = pd.read_parquet(groups_path, columns=columns)
    frame = frame.merge(protein, on="sequence_sha256", how="left")

    missing = int(frame["protein_group"].isna().sum())
    if missing:
        print(
            f"[{INFO}] {missing:,} observations have no protein group "
            "(construct failed validation); they are dropped from every regime"
        )
        frame = frame[frame["protein_group"].notna()]
    frame["protein_group"] = frame["protein_group"].astype("int64")
    if "protein_group_strict" in frame.columns:
        frame["protein_group_strict"] = frame["protein_group_strict"].astype("int64")
    return frame


def _drug_groups(frame, processed: Path):
    """Attach drug-group labels, preferring the cached ones.

    ``lock-benchmarks candidates`` writes ``drug_groups.parquet`` and freezes the
    cliff series against those exact labels. Recomputing here could relabel a
    group, which would preassign one set and split on another -- so the cache is
    authoritative when present.
    """
    import pandas as pd

    from bindra.data.splits import drug_groups

    cached = processed / "drug_groups.parquet"
    if cached.exists():
        table = pd.read_parquet(cached)
        labels = pd.Series(
            table["drug_group"].to_numpy(), index=table["identity_key"].to_numpy()
        )
        print(f"[{OK}] drug groups from {cached.name} ({labels.nunique():,} groups)")
    else:
        ligands = pd.read_parquet(
            processed / "ligands", columns=["identity_key", "leakage_key", "scaffold_key"]
        )
        ligands = ligands.dropna(subset=["identity_key"]).drop_duplicates("identity_key")
        labels = drug_groups(ligands)
    frame = frame.copy()
    frame["drug_group"] = frame["identity_key"].map(labels)
    unmapped = int(frame["drug_group"].isna().sum())
    if unmapped:
        print(f"[{INFO}] {unmapped:,} observations have no drug group; dropped")
        frame = frame[frame["drug_group"].notna()]
    frame["drug_group"] = frame["drug_group"].astype("int64")
    return frame


def _required_groups_hash(benchmarks: Path | str) -> str | None:
    path = Path(benchmarks) / "activity_cliff" / "required_test_groups.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("set_sha256")


def _relation(processed: Path):
    """Load the relation table, for the homology audit. None if absent."""
    from bindra.data.homology import ProteinRelationTable

    directory = processed / "protein_relation"
    if not (directory / "relation_table.json").exists():
        return None, None
    table = ProteinRelationTable.load(directory)
    import pandas as pd

    constructs = pd.read_parquet(
        processed / "protein_groups.parquet", columns=["construct_id", "sequence_sha256"]
    )
    position = {cid: index for index, cid in enumerate(table.sequence_ids)}
    index = {
        row.sequence_sha256: position[row.construct_id]
        for row in constructs.itertuples(index=False)
        if row.construct_id in position
    }
    return table, index


def _sealed_panel_observations(benchmarks: Path) -> set[str]:
    """Observation ids in the three affinity-shaped locked panels.

    Read from each panel's ``queries.parquet``, which is the unsealed half --
    no label is opened and no access log is touched.
    """
    import pandas as pd

    panels = (
        "panel_strict_future",
        "panel_repeated_measurement",
        "panel_reimport_control",
    )
    reserved: set[str] = set()
    for name in panels:
        path = benchmarks / name / "queries.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        column = next(
            (c for c in frame.columns if "observation" in c.lower()), frame.columns[0]
        )
        reserved |= set(frame[column].astype(str))
    return reserved


def cmd_build(args) -> int:
    from bindra.config.profiles import get_profile
    from bindra.data.splits import (
        Partition,
        Regime,
        SplitManifest,
        audit_split,
        make_double_cold_split,
        make_grouped_split,
        make_temporal_split,
        partition_counts,
        write_split,
    )

    processed = Path(args.processed)
    out = Path(args.out)
    profile = get_profile()
    policy = profile.splits

    frame = _load_corpus(processed)
    frame = _drug_groups(frame, processed)
    print(
        f"[{OK}] corpus {len(frame):,} observations, "
        f"{frame['pair_key'].nunique():,} pairs, "
        f"{frame['drug_group'].nunique():,} drug groups, "
        f"{frame['protein_group'].nunique():,} protein groups"
    )

    relation, construct_index = _relation(processed)
    if relation is None:
        print(f"[{INFO}] no relation table; the homology audit will be skipped")

    dataset = json.loads((processed / "qc_report.json").read_text(encoding="utf-8"))
    inputs = {
        "snapshot_release": profile.snapshot_release,
        "curation_profile_hash": profile.profile_hash,
        "observations_combined_sha256": dataset["artifacts"]["observations"]["combined_sha256"],
        "observations_row_count": dataset["artifacts"]["observations"]["rows"],
        "source_classes_sha256": dataset.get("source_classes_sha256"),
        "protein_relation_sha256": getattr(relation, "sha256", None),
        "protein_identity_threshold": profile.proteins.identity_threshold,
        "protein_cold_target_identity_threshold": (
            profile.proteins.cold_target_identity_threshold
        ),
        "protein_coverage_threshold": profile.proteins.coverage_threshold,
        "alignment_mode": profile.proteins.alignment_mode,
    }

    # Drug groups the activity-cliff panel requires in test, frozen before this
    # command ran. Preassigning them is what keeps the panel from being selected
    # after the fact; an empty set simply means `lock-benchmarks candidates` has
    # not run yet, which is reported rather than assumed harmless.
    from bindra.cli.lock_benchmarks import load_required_test_groups

    required = load_required_test_groups(args.benchmarks)
    preassign_drug = {group: Partition.TEST.value for group in required}
    if required:
        present = set(frame["drug_group"].unique())
        usable = {g: v for g, v in preassign_drug.items() if g in present}
        print(
            f"[{OK}] preassigning {len(usable):,} cliff-panel drug groups to test "
            f"({len(required) - len(usable):,} of the frozen set are not in this corpus)"
        )
        preassign_drug = usable
        inputs["preassigned_test_groups_sha256"] = _required_groups_hash(args.benchmarks)
        inputs["n_preassigned_test_groups"] = len(usable)
    else:
        print(
            f"[{INFO}] no frozen cliff series found under {args.benchmarks}; "
            "cold-drug and double-cold will not preassign anything. Run "
            "`bindra lock-benchmarks candidates` first if the cliff panel is wanted."
        )

    # Two cold-target regimes, deliberately. At the specification's 40% identity
    # threshold single linkage puts 93% of constructs in one component covering
    # 98.4% of observations, so the strict regime can only test on 1.6% of the
    # data. It is built anyway because it bounds any claim about novel targets;
    # the 60% regime is the one with enough data to measure anything, and its
    # audit reports the 40-60% band it permits to straddle.
    regimes = {
        Regime.RANDOM_PAIR.value: ("pair_key", None),
        Regime.COLD_DRUG.value: ("drug_group", preassign_drug or None),
        Regime.COLD_TARGET.value: ("protein_group", None),
        "cold_target_strict": ("protein_group_strict", None),
    }

    requested = args.regime or [
        Regime.RANDOM_PAIR.value,
        Regime.COLD_DRUG.value,
        Regime.COLD_TARGET.value,
        "cold_target_strict",
        Regime.DOUBLE_COLD.value,
        Regime.TEMPORAL.value,
    ]

    # Observation ids belonging to the three affinity-shaped sealed panels.
    reserved_ids = _sealed_panel_observations(Path(args.benchmarks))
    if reserved_ids:
        print(
            f"[{OK}] {len(reserved_ids):,} sealed-panel observations will be held "
            "out of every non-temporal regime"
        )
        inputs["reserved_panel_observations"] = len(reserved_ids)
    else:
        print(
            f"[{INFO}] no sealed panels found under {args.benchmarks}; nothing "
            "reserved, and only a temporal model will be able to score them"
        )

    failures = 0
    for regime in requested:
        print()
        print(f"[{INFO}] {regime}")
        attrition: dict = {}
        if regime in regimes:
            column, preassigned = regimes[regime]
            assignment = make_grouped_split(
                frame, column, policy=policy, preassigned=preassigned
            )
        elif regime == Regime.DOUBLE_COLD.value:
            # Level 1 splits test off, so the cliff series are preassigned there.
            assignment, attrition = make_double_cold_split(
                frame,
                policy=policy,
                preassigned_drug={g: "test" for g in preassign_drug} or None,
            )
        elif regime == Regime.TEMPORAL.value:
            assignment, attrition = make_temporal_split(frame, policy=policy)
        else:
            print(f"[{FAIL}] unknown regime {regime!r}")
            return 2

        # The sealed panels are held out of EVERY regime, not just the temporal
        # one that defines them. Without this a panel is scorable only by a
        # temporally-trained model, because every other regime draws its training
        # set at random across dates and swallows most of the panel: measured at
        # 60.2%/100.0%/67.7% for random_pair and 66.4%/68.8%/67.6% for
        # cold_target. A benchmark that only one regime can use is not a
        # benchmark for the others, and the generalization regimes are exactly
        # the ones whose panel numbers would be most worth having.
        #
        # Reserved into quarantine rather than into test: these observations
        # belong to a locked panel with its own access log, and routing them into
        # a regime's test partition would let an ordinary `--partition test` run
        # read labels the panel lock is there to ration.
        if regime != Regime.TEMPORAL.value and reserved_ids:
            moved = 0
            for observation_id in list(assignment):
                if (
                    observation_id in reserved_ids
                    and assignment[observation_id] != Partition.QUARANTINE.value
                ):
                    assignment[observation_id] = Partition.QUARANTINE.value
                    moved += 1
            if moved:
                print(
                    f"    reserved {moved:,} sealed-panel observations into "
                    "quarantine, so this regime can be scored on the panels"
                )

        # Each regime is audited at its OWN grouping threshold; auditing the 60%
        # regime at 40% would flag every homologue the regime deliberately permits.
        strict = regime == "cold_target_strict"
        audit = audit_split(
            frame,
            assignment,
            Regime.COLD_TARGET.value if strict else regime,
            relation=relation,
            construct_index=construct_index,
            policy=policy,
            protein_identity_threshold=(
                profile.proteins.identity_threshold
                if strict
                else profile.proteins.cold_target_identity_threshold
            ),
            protein_disclosure_floor=None if strict else profile.proteins.identity_threshold,
        )

        counts = partition_counts(frame, assignment)
        quarantined = counts.pop("quarantine", {"n_observations": 0})
        for partition in sorted(counts):
            payload = counts[partition]
            print(
                f"    {partition:<11} {payload['n_observations']:>10,} obs  "
                f"{payload['n_pairs']:>9,} pairs  "
                f"{payload['n_ligands']:>9,} ligands  {payload['n_targets']:>6,} targets"
            )
        if quarantined["n_observations"]:
            print(f"    {'quarantine':<11} {quarantined['n_observations']:>10,} obs")

        for check in audit["checks"]:
            mark = OK if check.get("passed", True) else FAIL
            print(
                f"    [{mark}] {check['check']:<26} checked {check['n_checked']:>10,}  "
                f"violations {check['n_violations']:,}"
            )
            if not check.get("passed", True) and check.get("examples"):
                print(f"              e.g. {check['examples'][:3]}")

        if not audit["all_gating_checks_passed"]:
            failures += 1
            print(f"    [{FAIL}] gating checks failed; manifest not written")
            if not args.write_failing:
                continue

        manifest = SplitManifest(
            regime=regime,
            seed=policy.seed,
            fractions=dict(policy.fractions),
            counts={**counts, "quarantine": quarantined},
            quarantine={"n_observations": quarantined["n_observations"], "attrition": attrition},
            inputs=inputs,
            policy={
                "assignment": policy.assignment,
                "double_cold_independent": policy.double_cold_independent,
                "double_cold_cascade": policy.double_cold_cascade,
                "temporal_cutoff": policy.temporal_cutoff,
                "temporal_nested": policy.temporal_nested,
                "temporal_train_cutoff": policy.temporal_train_cutoff,
                "identity_ambiguity_margin": profile.proteins.identity_ambiguity_margin,
            },
            audit=audit,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        directory = write_split(
            manifest, assignment, out / regime, overwrite=args.overwrite
        )
        _append_seed_log(out, regime, policy.seed, manifest.split_id)
        print(f"    [{OK}] wrote {directory / 'manifest.json'}  split_id {manifest.split_id}")

    print()
    if failures:
        print(f"make-splits: {failures} regime(s) failed their gating checks.")
        return 1
    print("make-splits: all regimes built and audited.")
    return 0


def _append_seed_log(out: Path, regime: str, seed: int, split_id: str) -> None:
    """Record every seed ever used, so seed shopping is visible in version control."""
    out.mkdir(parents=True, exist_ok=True)
    entry = {
        "at": datetime.now(timezone.utc).isoformat(),
        "regime": regime,
        "seed": seed,
        "split_id": split_id,
    }
    with open(out / "seed_log.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def cmd_verify(args) -> int:
    from bindra.data.splits import read_split, verify_manifest

    processed = Path(args.processed)
    dataset = json.loads((processed / "qc_report.json").read_text(encoding="utf-8"))
    current = dataset["artifacts"]["observations"]["combined_sha256"]

    directories = sorted(p for p in Path(args.out).iterdir() if (p / "manifest.json").exists())
    if not directories:
        print(f"[{FAIL}] no manifests under {args.out}")
        return 1

    status = 0
    for directory in directories:
        try:
            manifest = verify_manifest(directory, observations_sha256=current)
            _, assignment = read_split(directory)
            print(
                f"[{OK}] {manifest.regime:<13} {len(assignment):>10,} ids  "
                f"split_id {manifest.split_id}  "
                f"gating {'pass' if manifest.audit.get('all_gating_checks_passed') else 'FAIL'}"
            )
        except Exception as exc:  # noqa: BLE001 - diagnostic surface
            print(f"[{FAIL}] {directory.name}: {type(exc).__name__}: {exc}")
            status = 1
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bindra make-splits",
        description="Build and audit the immutable split manifests.",
    )
    sub = parser.add_subparsers(dest="action")

    build = sub.add_parser("build", help="build and audit split manifests")
    build.add_argument("--processed", default="data/processed/202610")
    build.add_argument("--out", default="data/splits/202610")
    build.add_argument(
        "--benchmarks",
        default="data/benchmarks/202610",
        help="where the frozen cliff series live (required_test_groups.json)",
    )
    build.add_argument(
        "--regime",
        action="append",
        default=None,
        help="build only this regime (repeatable); default is all five",
    )
    build.add_argument("--overwrite", action="store_true", help="replace an existing manifest")
    build.add_argument(
        "--write-failing",
        action="store_true",
        help="write a manifest even when gating checks fail (for diagnosis only)",
    )
    build.set_defaults(func=cmd_build)

    verify = sub.add_parser("verify", help="re-verify written manifests")
    verify.add_argument("--processed", default="data/processed/202610")
    verify.add_argument("--out", default="data/splits/202610")
    verify.set_defaults(func=cmd_verify)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        # Bare `bindra make-splits` means build, which is what a reader expects.
        args = parser.parse_args(["build", *(argv or [])])
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
