"""``bindra lock-benchmarks`` -- freeze the prespecified evaluation panels.

Two subcommands, and the order between them is load-bearing.

``candidates`` runs first, **before any split exists**. It finds activity-cliff
pairs over the whole curated corpus and freezes the set of drug groups whose
members must end up in the test partition. ``bindra make-splits`` then reads that
frozen set and preassigns those groups to test.

``finalize`` runs after the splits, and locks every panel against them.

Inverting that order would be post-hoc selection: keeping whichever cliff pairs
happened to land in test lets the split decide the panel, and risks an empty panel
when none do.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

OK = "  ok   "
FAIL = " FAIL  "
INFO = " info  "


def _drug_groups(processed: Path):
    """Drug-group labels, cached so every consumer uses identical labels.

    The preassignment and the split must agree exactly. Recomputing in two places
    invites a silent divergence -- a relabelled group would preassign one set and
    split on another -- so the labels are written once and reused.
    """
    import pandas as pd

    from bindra.data.splits import drug_groups
    from bindra.data.tables import write_table

    cached = processed / "drug_groups.parquet"
    if cached.exists():
        frame = pd.read_parquet(cached)
        print(f"[{OK}] reusing {cached} ({len(frame):,} molecules)")
        return dict(zip(frame["identity_key"], frame["drug_group"]))

    ligands = pd.read_parquet(
        processed / "ligands", columns=["identity_key", "leakage_key", "scaffold_key"]
    )
    ligands = ligands.dropna(subset=["identity_key"]).drop_duplicates("identity_key")
    labels = drug_groups(ligands)
    frame = pd.DataFrame(
        {"identity_key": labels.index, "drug_group": labels.to_numpy()}
    )
    reference = write_table(frame, cached)
    print(
        f"[{OK}] drug groups: {len(frame):,} molecules, "
        f"{frame['drug_group'].nunique():,} groups -> {reference.path}"
    )
    return dict(zip(frame["identity_key"], frame["drug_group"]))


def cmd_candidates(args) -> int:
    from rdkit import Chem

    from bindra.config.profiles import get_profile
    from bindra.data.tables import write_json, write_table
    from bindra.eval.benchmarks.activity_cliff import (
        CliffDefinition,
        build_cliff_candidates,
        required_test_drug_groups,
    )
    from bindra.eval.benchmarks.lock import definition_hash

    processed = Path(args.processed)
    out = Path(args.out) / "activity_cliff"
    out.mkdir(parents=True, exist_ok=True)
    profile = get_profile()

    definition = CliffDefinition(rdkit_version=Chem.rdBase.rdkitVersion)

    # Build the shared fingerprint store first. Over the full corpus fingerprint
    # computation dominates the scan, and the same store is reused by the fishing
    # panel and the split similarity audit.
    from bindra.data.similarity import build_fingerprint_store

    print(f"[{INFO}] building the shared fingerprint store (cached after first run)...")
    fingerprints = build_fingerprint_store(
        processed,
        radius=definition.radius,
        n_bits=definition.n_bits,
        use_chirality=definition.use_chirality,
        workers=args.workers,
    )
    print(f"[{OK}] fingerprints: {fingerprints.to_meta()}")

    print(f"[{INFO}] cliff thresholds (endpoint-specific, from measured sigma_rep):")
    for endpoint in definition.endpoints:
        print(
            f"           {endpoint:<5} |dp| >= {profile.cliff_delta_threshold(endpoint):.3f}"
            f"   (sigma_rep {profile.error_model.sigma(endpoint):.3f})"
        )

    candidates, report = build_cliff_candidates(
        processed,
        definition=definition,
        profile=profile,
        max_groups=args.max_groups or None,
        fingerprints=fingerprints,
    )
    print()
    print(f"[{OK}] scanned {report.n_groups_considered:,} (group, endpoint) blocks")
    print(f"[{OK}] cliffs {report.n_cliffs:,}   non-cliff controls {report.n_noncliffs:,}")
    print(f"[{OK}] cliff kinds {report.cliff_kinds}")
    print(f"[{INFO}] rejected {report.rejected}")

    if not len(candidates):
        print(f"[{FAIL}] no candidates; nothing to freeze")
        return 1

    groups = _drug_groups(processed)
    required = required_test_drug_groups(candidates, groups)
    payload = {
        "panel": "activity_cliff",
        "definition_sha256": definition_hash(definition.to_dict()),
        "n_required_groups": len(required),
        "required_test_drug_groups": sorted(int(g) for g in required),
        "note": [
            "Frozen BEFORE any split exists. bindra make-splits preassigns these "
            "drug groups to the test partition.",
            "Both cliff members and their whole chemical series are held out: "
            "holding out only the two molecules would leave their series in "
            "training, and the specification forbids moving one member into "
            "training to make the other easier.",
        ],
    }
    blob = json.dumps(payload, indent=2, sort_keys=True)
    payload["set_sha256"] = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    (out / "required_test_groups.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    reference = write_table(candidates, out / "candidates")
    write_json(
        {
            "report": report.to_dict(),
            "definition": definition.to_dict(),
            "candidates": reference.to_dict(),
        },
        out / "candidates_report.json",
    )
    print()
    print(
        f"[{OK}] froze {len(required):,} drug groups that must land in test "
        f"-> {out / 'required_test_groups.json'}"
    )
    print(f"[{OK}] cached {len(candidates):,} candidate pairs -> {reference.path}")
    print()
    print("Next: bindra make-splits build   (it reads required_test_groups.json)")
    return 0


def _finalize(args) -> int:
    from bindra.cli.lock_benchmarks_finalize import finalize

    return finalize(args)


def load_required_test_groups(benchmarks: Path | str) -> frozenset[int]:
    """Drug groups the cliff panel needs in test, or empty if not yet frozen."""
    path = Path(benchmarks) / "activity_cliff" / "required_test_groups.json"
    if not path.exists():
        return frozenset()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return frozenset(int(g) for g in payload["required_test_drug_groups"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bindra lock-benchmarks",
        description="Freeze the prespecified evaluation panels.",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    candidates = sub.add_parser(
        "candidates",
        help="find cliff pairs and freeze the drug groups that must be in test",
    )
    candidates.add_argument("--processed", default="data/processed/202610")
    candidates.add_argument("--out", default="data/benchmarks/202610")
    candidates.add_argument(
        "--max-groups", type=int, default=0, help="limit blocks scanned (smoke test)"
    )
    candidates.add_argument("--workers", type=int, default=14)
    candidates.set_defaults(func=cmd_candidates)

    final = sub.add_parser(
        "finalize", help="lock every panel against the built splits"
    )
    final.add_argument("--processed", default="data/processed/202610")
    final.add_argument("--out", default="data/benchmarks/202610")
    final.add_argument("--splits", default="data/splits/202610")
    final.add_argument("--workers", type=int, default=14)
    final.add_argument(
        "--max-targets", type=int, default=0, help="limit fishing targets (smoke test)"
    )
    final.add_argument("--skip-fishing", action="store_true")
    final.add_argument("--overwrite", action="store_true")
    final.set_defaults(func=_finalize)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
