"""Locked activity-cliff panel.

A documented extension of MoleculeACE rather than a reimplementation, because this
corpus forces two departures and both are recorded in the definition:

**The magnitude threshold is endpoint-specific, not 1 p-unit.** Measured replicate
noise on this snapshot is sigma_rep = 0.945 for pKd, 0.641 for pKi and 0.671 for
pIC50, so a flat 1 p-unit "cliff" in pKd is only ~1.06 sigma -- inside measurement
error. The threshold is ``max(1.0, 2 * sigma_rep)``: never looser than the
literature convention, never inside the noise.

**Both members must come from the same comparability group.** With no assay
identifier anywhere in BindingDB, the tightest available proxy for "measured the
same way" is the same target, endpoint, citation, condition bucket and source
class. That makes the primary panel small, and the size is reported rather than
quietly relaxed; tiers 2 and 3 are built and reported separately.

A pair also has to clear a signal-to-noise gate: both members' own replicate
spreads must be below sigma_rep, and the difference must exceed twice the pooled
spread of the pair. Without it, a "cliff" can be two noisy replicate sets of
otherwise-similar compounds.

**Holdout ordering matters and is not negotiable.** Candidates are generated over
the whole curated corpus first, the drug groups they need are frozen and hashed,
and only then are the cold-drug and double-cold splits built with those groups
preassigned to test. Building splits first and keeping whichever pairs happen to
land in test would be post-hoc selection, and would risk an empty panel.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from bindra.config.profiles import CurationProfile, get_profile
from bindra.data.endpoints import Endpoint
from bindra.data.similarity import (
    FingerprintStore,
    morgan_fingerprints,
    nearest_neighbour,
    tanimoto_matrix,
    unpack,
)

#: Groups larger than this use top-k neighbour search instead of all-pairs. The
#: result is the same; the cost is not. One comparability group in this release
#: holds 2,675 observations, which would be 3.6M pairs on its own.
ALL_PAIRS_LIMIT = 600


@dataclass(frozen=True)
class CliffDefinition:
    """Every pinned choice. Hashed into the manifest, so it cannot drift."""

    fingerprint: str = "morgan"
    radius: int = 2
    n_bits: int = 2048
    use_chirality: bool = False
    similarity: str = "tanimoto"
    tanimoto_threshold: float = 0.90
    #: An alternative route in: identical generic Murcko framework at a lower
    #: similarity. Catches series related by a ring-system swap that ECFP4 scores
    #: below 0.90.
    generic_scaffold_threshold: float = 0.80
    endpoints: tuple[str, ...] = (Endpoint.KD.value, Endpoint.KI.value, Endpoint.IC50.value)
    ic50_requires_tier_a: bool = True
    #: |delta p| >= max(delta_floor, delta_sigma_multiple * sigma_rep[endpoint])
    delta_sigma_multiple: float = 2.0
    delta_floor: float = 1.0
    #: Matched controls: similar pairs that are *not* cliffs.
    noncliff_delta_max: float = 0.50
    require_exact_both: bool = True
    #: Signal-to-noise gate.
    max_member_replicate_spread_sigma: float = 1.0
    min_delta_over_pooled_spread: float = 2.0
    comparability_tier: int = 1
    series_unit: str = "drug_group"
    preassign_series_to_test: bool = True
    rdkit_version: str | None = None

    #: Similarity at or above this counts as "the fingerprint cannot tell these
    #: apart". Measured on this corpus, 365 of 747 sampled cliffs sit here, and
    #: they are **two different phenomena** that must not be pooled:
    #:
    #: * 325 share a leakage key -- stereoisomers, where ECFP4 without chirality
    #:   is blind by construction.
    #: * 40 do not -- homologues differing in chain length, ring size or linker
    #:   length, which ECFP4 at radius 2 also cannot see.
    #:
    #: Both are legitimate activity cliffs and both are unsolvable for a model
    #: whose ligand representation is ECFP4-equivalent, so ``cliff_kind`` is a
    #: prespecified diagnostic category and metrics are reported per kind.
    fingerprint_identical_threshold: float = 0.999

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CliffReport:
    n_groups_considered: int = 0
    n_groups_all_pairs: int = 0
    n_groups_topk: int = 0
    n_groups_too_small: int = 0
    n_candidate_pairs: int = 0
    n_similar_pairs: int = 0
    n_cliffs: int = 0
    n_noncliffs: int = 0
    cliff_kinds: dict[str, int] = field(default_factory=dict)
    rejected: dict[str, int] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    per_endpoint: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cliff_kind(
    similarity: float,
    leakage_a: Any,
    leakage_b: Any,
    definition: CliffDefinition,
) -> str:
    """Which mechanism makes this pair a cliff.

    A prespecified diagnostic category, not a post-hoc explanation. The
    distinction matters because the first two kinds are invisible to an
    ECFP4-equivalent ligand representation, so pooling them with ordinary
    substitution cliffs would blame a model for a question its features cannot
    answer -- and would hide the fact that some cliffs are unanswerable.
    """
    if similarity >= definition.fingerprint_identical_threshold:
        if leakage_a is not None and leakage_a == leakage_b:
            # Same parent skeleton: stereoisomers, tautomers or charge variants.
            return "stereochemical"
        # Different molecules the fingerprint cannot separate: chain length,
        # ring size, linker length.
        return "fingerprint_blind"
    return "substitution"


def _observation_table(processed: Path, definition: CliffDefinition) -> pd.DataFrame:
    """Exact, regression-eligible observations with the fields a pair needs."""
    import glob

    columns = [
        "observation_id",
        "identity_key",
        "leakage_key",
        "scaffold_key",
        "generic_scaffold_key",
        "sequence_sha256",
        "endpoint",
        "comparability_group_id",
        "p_value",
        "is_exact",
        "ic50_tier",
        "primary_regression_eligible",
        "replicate_p_range",
        "replicate_discordance",
    ]
    shards = sorted(glob.glob(str(processed / "observations" / "*.parquet")))
    available = set(pd.read_parquet(shards[0]).columns)
    columns = [c for c in columns if c in available]
    frame = pd.concat(
        [pd.read_parquet(path, columns=columns) for path in shards], ignore_index=True
    )

    frame = frame[frame["is_exact"].fillna(False)]
    frame = frame[frame["endpoint"].isin(definition.endpoints)]
    frame = frame[frame["primary_regression_eligible"].fillna(False)]
    if definition.ic50_requires_tier_a:
        frame = frame[~((frame["endpoint"] == Endpoint.IC50.value) & (frame["ic50_tier"] == "B"))]
    # A beyond-repeatability group cannot support a delta.
    frame = frame[frame["replicate_discordance"] != "high"]
    return frame.reset_index(drop=True)


def _smiles_for(processed: Path, identity_keys: Iterable[str]) -> dict[str, str]:
    wanted = set(identity_keys)
    ligands = pd.read_parquet(
        processed / "ligands", columns=["identity_key", "standardized_smiles"]
    )
    ligands = ligands[ligands["identity_key"].isin(wanted)].drop_duplicates("identity_key")
    return dict(zip(ligands["identity_key"], ligands["standardized_smiles"]))


def _similar_pairs_in_group(
    keys: list[str],
    packed: np.ndarray,
    threshold: float,
    *,
    top_k: int = 25,
) -> list[tuple[int, int, float]]:
    """Index pairs above ``threshold``, by all-pairs or top-k as size dictates."""
    n = len(keys)
    if n < 2:
        return []
    if n <= ALL_PAIRS_LIMIT:
        matrix = tanimoto_matrix(unpack(packed), unpack(packed))
        rows, columns = np.where(np.triu(matrix >= threshold, k=1))
        return [(int(i), int(j), float(matrix[i, j])) for i, j in zip(rows, columns)]

    result = nearest_neighbour(
        packed, packed, top_k=min(top_k, n), rescore_top=min(400, n)
    )
    found: set[tuple[int, int]] = set()
    out: list[tuple[int, int, float]] = []
    for i in range(n):
        for position in range(result.similarity.shape[1]):
            j = int(result.index[i, position])
            value = float(result.similarity[i, position])
            if i == j or value < threshold:
                continue
            pair = (min(i, j), max(i, j))
            if pair in found:
                continue
            found.add(pair)
            out.append((pair[0], pair[1], value))
    return out


def build_cliff_candidates(
    processed_dir: Path | str,
    *,
    definition: CliffDefinition | None = None,
    profile: CurationProfile | None = None,
    max_groups: int | None = None,
    fingerprints: "FingerprintStore | None" = None,
) -> tuple[pd.DataFrame, CliffReport]:
    """Find cliff and matched non-cliff pairs over the whole curated corpus.

    Split-agnostic on purpose: the panel's series are preassigned to test
    afterwards, which is what keeps the selection from being post-hoc.

    Pass ``fingerprints`` (a prebuilt :class:`~bindra.data.similarity.FingerprintStore`)
    to skip fingerprint computation entirely. Over the full corpus that is the
    dominant cost, and the store is shared with the fishing panel and the split
    similarity audit, so computing it once is strictly better than three times.
    """
    from rdkit import Chem

    processed_dir = Path(processed_dir)
    profile = profile or get_profile()
    definition = definition or CliffDefinition(rdkit_version=Chem.rdBase.rdkitVersion)
    report = CliffReport()

    # Validated before any I/O: a parameter mismatch is cheap to detect and
    # expensive to discover after loading 2.7M observations. A store built at a
    # different radius or width would silently redefine the similarity this
    # definition names.
    if fingerprints is not None and (
        fingerprints.radius != definition.radius
        or fingerprints.n_bits != definition.n_bits
        or fingerprints.use_chirality != definition.use_chirality
    ):
        raise ValueError(
            "the supplied fingerprint store was built with "
            f"radius={fingerprints.radius}, n_bits={fingerprints.n_bits}, "
            f"use_chirality={fingerprints.use_chirality}, which does not match this "
            f"panel definition (radius={definition.radius}, "
            f"n_bits={definition.n_bits}, use_chirality={definition.use_chirality}). "
            "Similarity would not mean what the definition says it means."
        )

    thresholds = {
        endpoint: profile.cliff_delta_threshold(endpoint) for endpoint in definition.endpoints
    }
    sigmas = {endpoint: profile.error_model.sigma(endpoint) for endpoint in definition.endpoints}
    report.thresholds = {k: round(v, 4) for k, v in thresholds.items()}

    frame = _observation_table(processed_dir, definition)

    if fingerprints is not None:

        def fingerprints_for(keys: list[str]) -> tuple[list[str], np.ndarray]:
            return fingerprints.subset(keys)

    else:
        smiles = _smiles_for(processed_dir, frame["identity_key"].unique())
        fingerprint_cache: dict[str, np.ndarray] = {}

        def fingerprints_for(keys: list[str]) -> tuple[list[str], np.ndarray]:
            missing = [k for k in keys if k not in fingerprint_cache]
            if missing:
                packed, valid = morgan_fingerprints(
                    [smiles.get(k, "") for k in missing],
                    radius=definition.radius,
                    n_bits=definition.n_bits,
                    use_chirality=definition.use_chirality,
                )
                for key, row, ok in zip(missing, packed, valid):
                    if ok:
                        fingerprint_cache[key] = row
            usable = [k for k in keys if k in fingerprint_cache]
            if not usable:
                return [], np.zeros((0, definition.n_bits // 8), dtype=np.uint8)
            return usable, np.stack([fingerprint_cache[k] for k in usable])

    rejected: dict[str, int] = {}

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    records: list[dict[str, Any]] = []
    grouped = frame.groupby(["comparability_group_id", "endpoint"], sort=False)
    for index, ((group_id, endpoint), block) in enumerate(grouped):
        if max_groups is not None and index >= max_groups:
            break
        report.n_groups_considered += 1

        # One observation per distinct molecule; the replicate median already
        # collapsed genuine repeats inside this group.
        block = block.drop_duplicates("identity_key")
        if len(block) < 2:
            report.n_groups_too_small += 1
            continue

        keys = block["identity_key"].tolist()
        usable, packed = fingerprints_for(keys)
        if len(usable) < 2:
            report.n_groups_too_small += 1
            continue
        if len(usable) <= ALL_PAIRS_LIMIT:
            report.n_groups_all_pairs += 1
        else:
            report.n_groups_topk += 1

        rows = block.set_index("identity_key").loc[usable]
        values = rows["p_value"].to_numpy(dtype=float)
        spreads = rows["replicate_p_range"].fillna(0.0).to_numpy(dtype=float)
        generic = rows["generic_scaffold_key"].tolist() if "generic_scaffold_key" in rows else [None] * len(usable)
        leakage = rows["leakage_key"].tolist() if "leakage_key" in rows else [None] * len(usable)

        lower_threshold = min(definition.tanimoto_threshold, definition.generic_scaffold_threshold)
        pairs = _similar_pairs_in_group(usable, packed, lower_threshold)
        report.n_candidate_pairs += len(pairs)

        sigma = sigmas[endpoint]
        cliff_threshold = thresholds[endpoint]

        for i, j, similarity in pairs:
            same_framework = (
                generic[i] is not None and generic[i] == generic[j]
            )
            if similarity < definition.tanimoto_threshold and not (
                same_framework and similarity >= definition.generic_scaffold_threshold
            ):
                reject("similarity_below_threshold")
                continue
            report.n_similar_pairs += 1

            delta = float(values[j] - values[i])
            magnitude = abs(delta)
            pooled = float(max(spreads[i], spreads[j]))

            if spreads[i] >= definition.max_member_replicate_spread_sigma * sigma or (
                spreads[j] >= definition.max_member_replicate_spread_sigma * sigma
            ):
                reject("member_replicate_spread_too_wide")
                continue

            if magnitude >= cliff_threshold:
                if pooled > 0 and magnitude < definition.min_delta_over_pooled_spread * pooled:
                    reject("delta_within_pooled_spread")
                    continue
                label = "cliff"
                report.n_cliffs += 1
                kind = _cliff_kind(
                    similarity, leakage[i], leakage[j], definition
                )
                report.cliff_kinds[kind] = report.cliff_kinds.get(kind, 0) + 1
            elif magnitude <= definition.noncliff_delta_max:
                label = "noncliff_control"
                kind = _cliff_kind(similarity, leakage[i], leakage[j], definition)
                report.n_noncliffs += 1
            else:
                reject("delta_in_dead_band")
                continue

            records.append(
                {
                    "pair_id": f"{rows.index[i]}|{rows.index[j]}|{endpoint}|{group_id}",
                    "label": label,
                    "cliff_kind": kind,
                    "endpoint": endpoint,
                    "comparability_group_id": group_id,
                    "sequence_sha256": rows["sequence_sha256"].iloc[i],
                    "identity_key_a": rows.index[i],
                    "identity_key_b": rows.index[j],
                    "observation_a": rows["observation_id"].iloc[i],
                    "observation_b": rows["observation_id"].iloc[j],
                    "drug_group_a": None,
                    "drug_group_b": None,
                    "tanimoto": round(similarity, 4),
                    "same_generic_framework": bool(same_framework),
                    "p_value_a": float(values[i]),
                    "p_value_b": float(values[j]),
                    "delta_p": round(delta, 4),
                    "abs_delta_p": round(magnitude, 4),
                    "pooled_replicate_spread": round(pooled, 4),
                    "cliff_threshold": round(cliff_threshold, 4),
                    "sigma_rep": round(sigma, 4),
                }
            )

    report.rejected = rejected
    candidates = pd.DataFrame(records)
    if len(candidates):
        for endpoint in definition.endpoints:
            subset = candidates[candidates["endpoint"] == endpoint]
            report.per_endpoint[endpoint] = {
                "cliffs": int((subset["label"] == "cliff").sum()),
                "noncliff_controls": int((subset["label"] == "noncliff_control").sum()),
            }
    return candidates, report


def required_test_drug_groups(
    candidates: pd.DataFrame,
    drug_group_of: Mapping[str, int],
) -> frozenset[int]:
    """Drug groups that must be preassigned to test.

    Both cliff members **and their whole chemical series** are held out. Holding
    out only the two molecules would leave their series in training, and the
    specification forbids moving one member into training to make the other
    easier.
    """
    cliffs = candidates[candidates["label"] == "cliff"]
    groups: set[int] = set()
    for column in ("identity_key_a", "identity_key_b"):
        for key in cliffs[column]:
            group = drug_group_of.get(key)
            if group is not None:
                groups.add(int(group))
    return frozenset(groups)


def lock_cliff_panel(
    candidates: pd.DataFrame,
    *,
    definition: CliffDefinition,
    inputs: Mapping[str, Any],
    directory: Path | str,
    test_observations: Iterable[str] | None = None,
    report: CliffReport | None = None,
    overwrite: bool = False,
):
    """Freeze the panel, keeping queries open and labels sealed.

    ``test_observations`` finalizes the panel to pairs whose members actually
    landed in test; the attrition is recorded rather than hidden.
    """
    from bindra.eval.benchmarks.lock import lock

    directory = Path(directory)
    attrition: dict[str, int] = {"candidates": int(len(candidates))}

    final = candidates
    if test_observations is not None:
        held = set(test_observations)
        in_test = final["observation_a"].isin(held) & final["observation_b"].isin(held)
        attrition["dropped_not_in_test"] = int((~in_test).sum())
        final = final[in_test]
    attrition["locked"] = int(len(final))

    query_columns = [
        "pair_id",
        "cliff_kind",
        "endpoint",
        "sequence_sha256",
        "identity_key_a",
        "identity_key_b",
        "observation_a",
        "observation_b",
        "comparability_group_id",
        "tanimoto",
        "same_generic_framework",
    ]
    label_columns = [
        "pair_id",
        "label",
        "p_value_a",
        "p_value_b",
        "delta_p",
        "abs_delta_p",
        "pooled_replicate_spread",
        "cliff_threshold",
        "sigma_rep",
    ]

    return lock(
        "activity_cliff",
        definition,
        inputs,
        directory,
        queries=final[[c for c in query_columns if c in final.columns]],
        labels=final[[c for c in label_columns if c in final.columns]],
        attrition={**attrition, **({} if report is None else {"rejected_" + k: v for k, v in report.rejected.items()})},
        notes=[
            "cliff_kind is a prespecified diagnostic category. 'stereochemical' "
            "and 'fingerprint_blind' pairs are indistinguishable to an "
            "ECFP4-equivalent ligand representation, so metrics must be reported "
            "per kind rather than pooled.",
            "Cliff magnitude is endpoint-specific: max(1.0, 2 * sigma_rep). A flat "
            "1 p-unit threshold would sit inside measurement noise for pKd "
            "(sigma_rep 0.945).",
            "Both members share one comparability group (tier 1). BindingDB has no "
            "assay identifier, so that is the tightest available proxy, and the "
            "panel is small as a result.",
            "Both members are exact: a delta computed from a censored bound is not a "
            "measured difference.",
            "Both cliff members and their whole drug group are held out of training.",
            "Labels are sealed; they exist to score frozen predictions.",
        ],
        overwrite=overwrite,
    )


__all__ = [
    "ALL_PAIRS_LIMIT",
    "CliffDefinition",
    "CliffReport",
    "build_cliff_candidates",
    "lock_cliff_panel",
    "required_test_drug_groups",
]
