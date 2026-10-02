"""Immutable split manifests: random-pair, cold-drug, cold-target, double-cold, temporal.

Four partitions, not three: calibration is separate from tuning validation,
because the specification requires conformal calibration to be fitted on data
that never informed hyperparameter selection.

Three design points carry most of the weight here.

**Groups are bin-packed by weight, not shuffled.** Group sizes on this corpus are
extremely skewed -- a single protein component or chemical series can carry a
large share of all observations -- so assigning groups by a plain shuffle misses
the target fractions badly. Groups are sorted by a seeded hash and then packed by
observation count.

**Double-cold assigns the two axes independently and intersects.** Grouping by a
combined protein-x-drug cluster id is a different and weaker algorithm, and there
is a test built to pass under that implementation and fail under this one. The
naive fractions also do not work: at 70/10/10/10 on both axes the retained share
is sum(p_i*q_i) ~= 0.52 and the test partition receives about 1% of interactions,
so a two-level cascade splits test off first and then re-splits the training block.

**The audit verifies relationships, it does not trust labels.** Every check
recomputes the forbidden relation from the published table or the relation table
rather than comparing group ids, because a group-id comparison would pass even if
the grouping itself were wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from bindra.config.profiles import CurationProfile, SplitPolicy, canonical_json, get_profile
from bindra.data.homology import UnionFind

SPLIT_SCHEMA_VERSION = "split-v1"


class Regime(StrEnum):
    RANDOM_PAIR = "random_pair"
    COLD_DRUG = "cold_drug"
    COLD_TARGET = "cold_target"
    DOUBLE_COLD = "double_cold"
    TEMPORAL = "temporal"


class Partition(StrEnum):
    TRAIN = "train"
    VAL = "val"
    CAL = "cal"
    TEST = "test"
    #: Pairs with evidence on both sides of the temporal cutoff. They cannot
    #: satisfy pair integrity and temporal purity at once, so the specification
    #: requires them in a separately labelled panel rather than silently in train
    #: or test.
    REPEATED_MEASUREMENT = "repeated_measurement"
    QUARANTINE = "quarantine"


class SplitError(RuntimeError):
    """A split could not be built or verified."""


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


def drug_groups(ligands: pd.DataFrame) -> pd.Series:
    """Connected components over shared leakage key *or* shared Murcko scaffold.

    Both edges are needed. The leakage key fuses salts, stereoisomers, tautomers
    and protonation states of one compound; the scaffold key fuses a chemical
    series. A cold-drug split has to separate series, not just compounds, or a
    "novel" test molecule is one methyl group from a training molecule.

    Unioning the scaffold edge also covers the tautomer classes InChI's
    mobile-hydrogen layer does not collapse: keto-enol forms of a ring-containing
    molecule share a scaffold and so land in one group regardless.
    """
    index = {key: position for position, key in enumerate(ligands["identity_key"])}
    union = UnionFind(len(index))

    for column in ("leakage_key", "scaffold_key"):
        if column not in ligands:
            continue
        for _, members in ligands.groupby(column, dropna=True)["identity_key"]:
            positions = [index[key] for key in members]
            for other in positions[1:]:
                union.union(positions[0], other)

    roots = {}
    labels = []
    for key in ligands["identity_key"]:
        root = union.find(index[key])
        labels.append(roots.setdefault(root, len(roots)))
    return pd.Series(labels, index=ligands["identity_key"].values, name="drug_group")


def _seeded_order(keys: Sequence[Any], seed: int) -> list[Any]:
    """Deterministic order from a seed, independent of input order.

    Hashing the key rather than shuffling a list means the order does not change
    when upstream code happens to emit groups in a different sequence, which is
    what makes a manifest reproducible from its recorded seed alone.
    """
    salt = str(seed).encode("utf-8")
    return sorted(
        keys,
        key=lambda key: hashlib.sha256(salt + str(key).encode("utf-8")).hexdigest(),
    )


def assign_groups(
    weights: Mapping[Any, int],
    fractions: Mapping[str, float],
    *,
    seed: int,
    preassigned: Mapping[Any, str] | None = None,
) -> dict[Any, str]:
    """Pack groups into partitions by observation weight.

    Largest-remainder packing over a seeded ordering: walk the groups in seeded
    order and place each into whichever partition is furthest below its target
    share. Deterministic, and far closer to the requested fractions than a
    shuffle when group sizes span several orders of magnitude.
    """
    preassigned = dict(preassigned or {})
    total = sum(weights.values())
    if total == 0:
        return dict(preassigned)

    targets = {name: share * total for name, share in fractions.items()}
    current = {name: 0.0 for name in fractions}
    assignment: dict[Any, str] = {}

    for key, partition in preassigned.items():
        if key in weights:
            assignment[key] = partition
            current[partition] = current.get(partition, 0.0) + weights[key]

    # Largest group first (longest-processing-time packing), with the seeded order
    # as the tie-break. Ordering matters a great deal here: a single protein
    # component or chemical series can hold a large share of all observations, and
    # a group is indivisible. Placed late, such a group lands wherever there is
    # room and overshoots a small partition badly; placed first, it goes to the
    # partition with the largest absolute deficit -- normally train -- and the
    # many small groups then fine-tune the remainder.
    #
    # A group bigger than a partition's entire target still cannot be
    # accommodated; the realized counts are recorded in the manifest so the
    # discrepancy is visible rather than implied.
    ordered = sorted(
        _seeded_order([k for k in weights if k not in assignment], seed),
        key=lambda key: -weights[key],
    )
    for key in ordered:
        deficits = {name: targets[name] - current[name] for name in fractions}
        chosen = max(deficits, key=lambda name: (deficits[name], name))
        assignment[key] = chosen
        current[chosen] += weights[key]
    return assignment


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass
class SplitManifest:
    regime: str
    seed: int
    fractions: Mapping[str, float]
    counts: Mapping[str, Mapping[str, int]]
    quarantine: Mapping[str, Any]
    inputs: Mapping[str, Any]
    policy: Mapping[str, Any]
    audit: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = ""
    schema_version: str = SPLIT_SCHEMA_VERSION
    split_id: str = ""
    manifest_sha256: str = ""
    id_files: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "regime": self.regime,
            "created_at": self.created_at,
            "seed": self.seed,
            "fractions": dict(self.fractions),
            "counts": {k: dict(v) for k, v in self.counts.items()},
            "quarantine": dict(self.quarantine),
            "inputs": dict(self.inputs),
            "policy": dict(self.policy),
            "audit": dict(self.audit),
            "id_files": {k: dict(v) for k, v in self.id_files.items()},
            "split_id": self.split_id,
        }

    def compute_hash(self) -> str:
        return hashlib.sha256(canonical_json(self.payload()).encode("utf-8")).hexdigest()


def _write_ids(path: Path, ids: Iterable[str]) -> dict[str, Any]:
    ordered = sorted(ids)
    blob = ("\n".join(ordered) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    # GzipFile, not gzip.open: only GzipFile accepts mtime, and a zero mtime is
    # what makes the compressed bytes reproducible rather than time-stamped.
    with gzip.GzipFile(filename=str(path), mode="wb", mtime=0) as handle:
        handle.write(blob)
    return {
        "path": path.name,
        "n": len(ordered),
        # Hash the *content*, not the gzip container, so a different zlib build
        # cannot make an identical split look like a changed one.
        "sha256": hashlib.sha256(blob).hexdigest(),
    }


def write_split(
    manifest: SplitManifest,
    assignment: Mapping[str, str],
    directory: Path | str,
    *,
    overwrite: bool = False,
) -> Path:
    """Write id files and the manifest. Refuses to overwrite by default."""
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise SplitError(
            f"{directory} already holds a split. Manifests are immutable; write to a "
            "new directory or pass overwrite=True deliberately."
        )
    directory.mkdir(parents=True, exist_ok=True)

    buckets: dict[str, list[str]] = {}
    for observation_id, partition in assignment.items():
        buckets.setdefault(partition, []).append(observation_id)

    manifest.id_files = {
        partition: _write_ids(directory / "ids" / f"{partition}.txt.gz", ids)
        for partition, ids in sorted(buckets.items())
    }
    manifest.split_id = f"{manifest.regime}-{manifest.compute_hash()[:8]}"
    manifest.manifest_sha256 = manifest.compute_hash()

    payload = manifest.payload()
    payload["manifest_sha256"] = manifest.manifest_sha256
    (directory / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return directory


def read_split(directory: Path | str) -> tuple[SplitManifest, dict[str, str]]:
    """Load a manifest and its id files, verifying every recorded hash."""
    directory = Path(directory)
    payload = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    recorded = payload.pop("manifest_sha256", None)

    manifest = SplitManifest(
        regime=payload["regime"],
        seed=payload["seed"],
        fractions=payload["fractions"],
        counts=payload["counts"],
        quarantine=payload["quarantine"],
        inputs=payload["inputs"],
        policy=payload["policy"],
        audit=payload.get("audit", {}),
        created_at=payload.get("created_at", ""),
        schema_version=payload["schema_version"],
        split_id=payload.get("split_id", ""),
        id_files=payload.get("id_files", {}),
    )
    if recorded is not None and manifest.compute_hash() != recorded:
        raise SplitError(
            f"{directory / 'manifest.json'} has been edited: its recorded hash does "
            "not match its content."
        )
    manifest.manifest_sha256 = recorded or manifest.compute_hash()

    assignment: dict[str, str] = {}
    for partition, info in manifest.id_files.items():
        path = directory / "ids" / info["path"]
        with gzip.open(path, "rb") as handle:
            blob = handle.read()
        if hashlib.sha256(blob).hexdigest() != info["sha256"]:
            raise SplitError(f"{path} has changed since the manifest was written")
        for line in blob.decode("utf-8").splitlines():
            if line:
                assignment[line] = partition
    return manifest, assignment


def verify_manifest(
    directory: Path | str,
    *,
    observations_sha256: str | None = None,
) -> SplitManifest:
    """Re-verify a split, including that the curated table has not moved."""
    manifest, _ = read_split(directory)
    if observations_sha256 is not None:
        recorded = manifest.inputs.get("observations_combined_sha256")
        if recorded and recorded != observations_sha256:
            raise SplitError(
                "the curated observation table has changed since this split was "
                f"written (manifest {recorded[:16]}..., current "
                f"{observations_sha256[:16]}...). The split no longer describes the "
                "data it was built from."
            )
    return manifest


# ---------------------------------------------------------------------------
# Regime builders
# ---------------------------------------------------------------------------


def _weights(frame: pd.DataFrame, key: str) -> dict[Any, int]:
    return frame.groupby(key, dropna=False)["observation_id"].size().to_dict()


def partition_counts(
    frame: pd.DataFrame, assignment: Mapping[str, str]
) -> dict[str, dict[str, int]]:
    """Observations, pairs, ligands and targets per partition, for the manifest."""
    labelled = frame.assign(partition=frame["observation_id"].map(assignment))
    out: dict[str, dict[str, int]] = {}
    for partition, group in labelled.groupby("partition", dropna=True):
        out[str(partition)] = {
            "n_observations": int(len(group)),
            "n_pairs": int(group["pair_key"].nunique()),
            "n_ligands": int(group["identity_key"].nunique()),
            "n_targets": int(group["sequence_sha256"].nunique()),
        }
    return out


def make_grouped_split(
    frame: pd.DataFrame,
    group_column: str,
    *,
    policy: SplitPolicy,
    preassigned: Mapping[Any, str] | None = None,
) -> dict[str, str]:
    """Assign observations by whole groups of ``group_column``."""
    assignment = assign_groups(
        _weights(frame, group_column),
        dict(policy.fractions),
        seed=policy.seed,
        preassigned=preassigned,
    )
    return {
        str(observation): assignment[group]
        for observation, group in zip(frame["observation_id"], frame[group_column])
        if group in assignment
    }


def make_double_cold_split(
    frame: pd.DataFrame,
    *,
    policy: SplitPolicy,
    preassigned_drug: Mapping[Any, str] | None = None,
    preassigned_protein: Mapping[Any, str] | None = None,
) -> tuple[dict[str, str], dict[str, int]]:
    """Independent drug and protein assignments, intersected.

    Grouping by a combined protein-x-drug cluster id would be a different and
    weaker algorithm: it only guarantees that the *combination* is unseen, not
    that either member is.

    The naive fractions do not work. At 70/10/10/10 on both axes the retained
    share is ``sum(p_i * q_i) ~= 0.52`` and the test partition receives roughly 1%
    of interactions. The cascade therefore splits test off first at 80/20 on each
    axis, then re-splits the training block by a *further* independent assignment,
    so validation and calibration come from inside the double-cold training block
    rather than being borrowed from a different regime.
    """
    attrition: dict[str, int] = {"input": int(len(frame))}

    level1 = {"train_block": policy.double_cold_level1["train_block"], "test": policy.double_cold_level1["test"]}
    drug_level1 = assign_groups(
        _weights(frame, "drug_group"), level1, seed=policy.seed, preassigned=preassigned_drug
    )
    protein_level1 = assign_groups(
        _weights(frame, "protein_group"),
        level1,
        seed=policy.seed + 1,
        preassigned=preassigned_protein,
    )

    assignment: dict[str, str] = {}
    train_block_rows: list[int] = []
    for position, (observation, drug, protein) in enumerate(
        zip(frame["observation_id"], frame["drug_group"], frame["protein_group"])
    ):
        left = drug_level1.get(drug)
        right = protein_level1.get(protein)
        if left is None or right is None or left != right:
            assignment[str(observation)] = Partition.QUARANTINE.value
            continue
        if left == "test":
            assignment[str(observation)] = Partition.TEST.value
        else:
            train_block_rows.append(position)

    attrition["level1_quarantined"] = sum(
        1 for value in assignment.values() if value == Partition.QUARANTINE.value
    )
    attrition["level1_test"] = sum(
        1 for value in assignment.values() if value == Partition.TEST.value
    )

    block = frame.iloc[train_block_rows]
    attrition["level1_train_block"] = int(len(block))

    level2 = dict(policy.double_cold_level2)
    drug_level2 = assign_groups(_weights(block, "drug_group"), level2, seed=policy.seed + 2)
    protein_level2 = assign_groups(
        _weights(block, "protein_group"), level2, seed=policy.seed + 3
    )
    for observation, drug, protein in zip(
        block["observation_id"], block["drug_group"], block["protein_group"]
    ):
        left = drug_level2.get(drug)
        right = protein_level2.get(protein)
        if left is None or right is None or left != right:
            assignment[str(observation)] = Partition.QUARANTINE.value
        else:
            assignment[str(observation)] = left

    attrition["level2_quarantined"] = (
        sum(1 for value in assignment.values() if value == Partition.QUARANTINE.value)
        - attrition["level1_quarantined"]
    )
    attrition["retained"] = sum(
        1 for value in assignment.values() if value != Partition.QUARANTINE.value
    )
    return assignment, attrition


def make_temporal_split(
    frame: pd.DataFrame,
    *,
    policy: SplitPolicy,
) -> tuple[dict[str, str], dict[str, int]]:
    """Nested temporal split on first evidence per pair.

    First evidence is the earliest publication date over **all** of a pair's
    observations, across every endpoint and source, so a pair cannot be "new"
    merely because one of its measurements is. ``Date in BindingDB`` is never used:
    a recent import of an old measurement is not new evidence.

    Pairs with any unknown-date observation are excluded entirely rather than
    assumed old or new.
    """
    attrition: dict[str, int] = {"input": int(len(frame))}

    dated = frame.copy()
    dated["evidence"] = pd.to_datetime(dated["first_evidence_date"], errors="coerce")
    unknown_pairs = set(dated.loc[dated["evidence"].isna(), "pair_key"])
    attrition["pairs_with_unknown_date"] = len(unknown_pairs)

    # Both ends of each pair's evidence window. Using only the minimum is the
    # trap: it puts every observation of an old pair into train, including ones
    # published after the cutoff, which is a future label in the training set.
    # The audit caught exactly that on the real corpus.
    span = dated.groupby("pair_key")["evidence"].agg(["min", "max"])

    train_cutoff = pd.Timestamp(policy.temporal_train_cutoff)
    cutoff = pd.Timestamp(policy.temporal_cutoff)
    val_start, val_end = (pd.Timestamp(d) for d in policy.temporal_val_window)
    cal_start, cal_end = (pd.Timestamp(d) for d in policy.temporal_cal_window)

    def bucket(pair: Any) -> str:
        if pair in unknown_pairs:
            return Partition.QUARANTINE.value
        try:
            first, last = span.loc[pair, "min"], span.loc[pair, "max"]
        except KeyError:
            return Partition.QUARANTINE.value
        if pd.isna(first) or pd.isna(last):
            return Partition.QUARANTINE.value

        # A pair that straddles the cutoff cannot be in train (it carries future
        # evidence) nor in the strict future panel (it carries prior evidence).
        if first <= cutoff < last:
            return Partition.REPEATED_MEASUREMENT.value

        # Assigned on the whole window, so pair integrity holds and no partition
        # receives an observation from outside its own period.
        if last <= train_cutoff:
            return Partition.TRAIN.value
        if val_start <= first and last <= val_end:
            return Partition.VAL.value
        if cal_start <= first and last <= cal_end:
            return Partition.CAL.value
        if first > cutoff:
            return Partition.TEST.value
        # Inside 2023 but spanning the H1/H2 boundary: real, and not assignable to
        # either tuning partition without splitting the pair.
        return Partition.QUARANTINE.value

    assignment = {
        str(observation): bucket(pair)
        for observation, pair in zip(frame["observation_id"], frame["pair_key"])
    }
    for partition in Partition:
        attrition[partition.value] = sum(
            1 for value in assignment.values() if value == partition.value
        )
    return assignment, attrition


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def audit_split(
    frame: pd.DataFrame,
    assignment: Mapping[str, str],
    regime: str,
    *,
    relation=None,
    construct_index: Mapping[str, int] | None = None,
    policy: SplitPolicy | None = None,
    protein_identity_threshold: float | None = None,
    protein_disclosure_floor: float | None = None,
) -> dict[str, Any]:
    """Verify the forbidden cross-partition relationships from the data.

    Every check **recomputes** its relation from the published table or the
    relation table. Comparing group ids would pass even if the grouping itself
    were wrong, which is exactly the failure mode the specification warns about
    when it says cluster labels are insufficient evidence of separation.
    """
    policy = policy or get_profile().splits
    labelled = frame.assign(partition=frame["observation_id"].map(assignment))
    # The repeated-measurement panel is excluded alongside quarantine: it is a
    # labelled side panel, not one of the partitions a model is split across, so
    # it must not count as a pair straddling a boundary.
    excluded = {Partition.QUARANTINE.value, Partition.REPEATED_MEASUREMENT.value}
    live = labelled[~labelled["partition"].isin(excluded)]
    checks: list[dict[str, Any]] = []

    def record(name: str, gating: bool, n_checked: int, violations: list[Any]) -> None:
        checks.append(
            {
                "check": name,
                "gating": gating,
                "n_checked": int(n_checked),
                "n_violations": len(violations),
                "examples": [str(v) for v in violations[:5]],
                "passed": (not violations) if gating else True,
            }
        )

    def straddles(column: str) -> list[Any]:
        counts = live.groupby(column)["partition"].nunique()
        return list(counts[counts > 1].index)

    # 1 pair integrity, in every regime
    record("pair_integrity", True, live["pair_key"].nunique(), straddles("pair_key"))

    if regime in (Regime.COLD_DRUG.value, Regime.DOUBLE_COLD.value):
        record("exact_identity_disjoint", True, live["identity_key"].nunique(), straddles("identity_key"))
        record("leakage_key_disjoint", True, live["leakage_key"].nunique(), straddles("leakage_key"))
        if "scaffold_key" in live:
            record("scaffold_disjoint", True, live["scaffold_key"].nunique(), straddles("scaffold_key"))

    if regime in (Regime.COLD_TARGET.value, Regime.DOUBLE_COLD.value):
        record("target_sequence_disjoint", True, live["sequence_sha256"].nunique(), straddles("sequence_sha256"))
        if relation is not None and construct_index is not None:
            checks.append(
                _audit_homology(
                    live,
                    relation,
                    construct_index,
                    policy,
                    identity_threshold=protein_identity_threshold,
                    disclosure_floor=protein_disclosure_floor,
                )
            )

    # Set algebra, in every regime
    partitions = {}
    for partition, group in labelled.groupby("partition"):
        partitions[str(partition)] = set(group["observation_id"])
    overlaps: list[Any] = []
    names = sorted(partitions)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            shared = partitions[left] & partitions[right]
            if shared:
                overlaps.append(f"{left}&{right}:{len(shared)}")
    covered = set().union(*partitions.values()) if partitions else set()
    missing = set(frame["observation_id"]) - covered
    record("set_algebra", True, len(frame), overlaps + ([f"unassigned:{len(missing)}"] if missing else []))

    if regime == Regime.TEMPORAL.value:
        checks.append(_audit_temporal(labelled, policy))

    # Reported, not gating
    balance = {
        str(partition): {
            "n_observations": int(len(group)),
            "exact": int(group["is_exact"].fillna(False).sum()) if "is_exact" in group else None,
            "censored": int(group["is_censored"].fillna(False).sum()) if "is_censored" in group else None,
        }
        for partition, group in live.groupby("partition")
    }

    gating = [c for c in checks if c.get("gating")]
    return {
        "regime": regime,
        "checks": checks,
        "balance": balance,
        "all_gating_checks_passed": all(c.get("passed", True) for c in gating),
    }


def _audit_homology(
    live: pd.DataFrame,
    relation,
    construct_index: Mapping[str, int],
    policy: SplitPolicy,
    *,
    identity_threshold: float | None = None,
    disclosure_floor: float | None = None,
) -> dict[str, Any]:
    """For every cross-partition sequence pair: was it aligned, and does it qualify?

    A pair that was *never aligned* counts as a violation in its own right. Without
    that, "no edge recorded" would be indistinguishable from "verified dissimilar",
    and the check would be vacuous.

    ``identity_threshold`` is the regime's own grouping threshold, which the gating
    check uses. ``disclosure_floor`` names a lower bound whose band is **reported
    rather than gated**: for the 60% cold-target regime the 40-60% band is exactly
    the leakage the looser threshold admits, and burying it would make the regime
    look stricter than it is.
    """
    by_partition: dict[str, set[int]] = {}
    for partition, group in live.groupby("partition"):
        indices = {
            construct_index[h]
            for h in group["sequence_sha256"].unique()
            if h in construct_index
        }
        by_partition[str(partition)] = indices

    threshold = (
        relation.policy.identity_threshold
        if identity_threshold is None
        else identity_threshold
    )
    qualifying = set(relation.qualifying_pairs_at(threshold))
    disclosed = (
        set(relation.qualifying_pairs_at(disclosure_floor)) - qualifying
        if disclosure_floor is not None and disclosure_floor < threshold
        else set()
    )
    violations: list[str] = []
    not_computed: list[str] = []
    disclosure_hits = 0
    checked = 0

    names = sorted(by_partition)
    for position, left in enumerate(names):
        for right in names[position + 1 :]:
            for i in by_partition[left]:
                for j in by_partition[right]:
                    if i == j:
                        continue
                    checked += 1
                    low, high = min(i, j), max(i, j)
                    if not relation.was_computed(low, high):
                        not_computed.append(f"{low}-{high}")
                    elif (low, high) in qualifying:
                        violations.append(f"{left}/{right}:{low}-{high}")
                    elif (low, high) in disclosed:
                        disclosure_hits += 1

    payload = {
        "check": "protein_homology",
        "gating": True,
        "identity_threshold": threshold,
        "n_checked": checked,
        "n_violations": len(violations) + len(not_computed),
        "n_qualifying_across_partitions": len(violations),
        "n_never_aligned": len(not_computed),
        "examples": violations[:5] + not_computed[:5],
        "passed": not violations and not not_computed,
        "note": (
            "A pair that was never aligned is itself a violation: otherwise an "
            "absent edge could mean 'verified dissimilar' or 'never looked'."
        ),
    }
    if disclosure_floor is not None and disclosure_floor < threshold:
        payload["disclosure"] = {
            "band": [disclosure_floor, threshold],
            "n_cross_partition_pairs_in_band": disclosure_hits,
            "gating": False,
            "note": (
                f"Cross-partition construct pairs with identity in "
                f"[{disclosure_floor}, {threshold}). These are the homologues this "
                "regime's looser threshold permits to straddle a partition. "
                "Reported, not gated: the strict regime exists to bound claims "
                "about novel targets, and this number is what the looser regime "
                "costs."
            ),
        }
    return payload


def _audit_temporal(labelled: pd.DataFrame, policy: SplitPolicy) -> dict[str, Any]:
    """Old measurements re-imported after the cutoff must never be in **test**.

    The specification's concern is that *"a recent database import of an old
    measurement is not new evidence"*, so such a record must not be counted as a
    future observation. It may legitimately sit in train, validation or
    calibration -- under the nested temporal split those are all "available by
    their designated cutoffs" -- so demanding train specifically would be wrong.

    This is a positive control on real data rather than a synthetic fixture: the
    class is populous in this release.
    """
    if "date_in_bindingdb" not in labelled:
        return {"check": "temporal_integrity", "gating": False, "n_checked": 0,
                "n_violations": 0, "examples": [], "passed": True,
                "note": "import date not present in the table"}

    published = pd.to_datetime(labelled["first_evidence_date"], errors="coerce")
    imported = pd.to_datetime(labelled["date_in_bindingdb"], errors="coerce")
    cutoff = pd.Timestamp(policy.temporal_cutoff)
    reimports = labelled[(published <= cutoff) & (imported > cutoff)]
    # Anywhere but test: pre-cutoff evidence is not a future measurement.
    misplaced = reimports[reimports["partition"] == Partition.TEST.value]

    # Any tuning partition, not just train: validation and calibration must also
    # respect their designated cutoffs.
    informs_model = [Partition.TRAIN.value, Partition.VAL.value, Partition.CAL.value]
    future_in_train = labelled[
        (published > cutoff) & (labelled["partition"].isin(informs_model))
    ]
    violations = [
        *(f"reimport_in_test:{o}" for o in misplaced["observation_id"].head(5)),
        *(f"future_label_informs_model:{o}" for o in future_in_train["observation_id"].head(5)),
    ]
    return {
        "check": "temporal_integrity",
        "gating": True,
        "n_checked": int(len(reimports) + len(future_in_train)),
        # Every check reports n_violations so the summary can be read uniformly.
        "n_violations": int(len(misplaced) + len(future_in_train)),
        "n_reimports": int(len(reimports)),
        "n_reimports_misplaced": int(len(misplaced)),
        "n_future_labels_in_fitting_partitions": int(len(future_in_train)),
        "n_future_labels_in_train": int(len(future_in_train)),
        "examples": violations,
        "passed": not violations,
    }


__all__ = [
    "SPLIT_SCHEMA_VERSION",
    "audit_split",
    "partition_counts",
    "make_double_cold_split",
    "make_grouped_split",
    "make_temporal_split",
    "Partition",
    "Regime",
    "SplitError",
    "SplitManifest",
    "assign_groups",
    "drug_groups",
    "read_split",
    "verify_manifest",
    "write_split",
]
