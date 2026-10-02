"""Locked drug-fishing and analog-fishing challenge sets.

Per-target challenge sets with **two control tiers**, kept separate because they
license different claims:

*Preferred — measured.* Compounds experimentally measured against that target and
found weak or inactive, on the same endpoint, from exact labels. These are real
negatives, and enrichment against them is a real result.

*Supplementary — presumed.* Property-matched molecules drawn from the curated
corpus that were never measured against the target **or any construct in its
protein group**. They are labelled ``presumed_control`` with
``binding_status="unknown"`` and they never receive an affinity label. A
computational decoy is not an experimental nonbinder, and an unmeasured library
member is unlabelled, not a known false positive.

The homologue-aware exclusion is the part that is easy to get wrong. Excluding
molecules measured against the target's exact sequence hash is not enough: the same
protein appears in this corpus as many constructs (mutants, domain fragments,
tagged versions), so a compound measured against one of them is not a safe decoy
for another. The exclusion therefore uses the protein groups from the exhaustive
alignment, not string matching.

DUD-E-style decoy construction is known to flatter 2D similarity methods, so the
manifest *requires* fingerprint and property baselines to be reported alongside any
enrichment number, and target-only fishing is reported separately from
seed-assisted fishing: a known binder supplied as a seed is real chemical
information, not zero-shot discovery.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import itertools
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from bindra.config.profiles import CurationProfile, get_profile
from bindra.data.properties import PROPERTY_COLUMNS, PropertyStore


@dataclass(frozen=True)
class FishingDefinition:
    """Pinned challenge-set policy. Hashed into the manifest."""

    #: Prespecified activity thresholds on exact labels, same endpoint.
    #: pX >= 7 is 100 nM; pX <= 5 is 10 uM.
    active_p_min: float = 7.0
    inactive_p_max: float = 5.0
    endpoints: tuple[str, ...] = ("Kd", "Ki", "IC50")
    ic50_requires_tier_a: bool = True
    require_exact: bool = True
    #: A target needs this many actives to be worth scoring at all.
    min_actives_per_target: int = 10
    #: Actives kept per (target, endpoint), sampled with ``seed``.
    #:
    #: The cap exists because this corpus is not DUD-E-shaped. 3,721 target-endpoint
    #: groups clear the minimum, holding 786,222 actives between them; at the DUD-E
    #: convention of 50 decoys each that is 39 million decoy rows, which is a data
    #: dump rather than a challenge set. Capping actives instead of dropping targets
    #: keeps full target coverage, which is what per-target macro estimates need --
    #: and the effective sample size here is 572 target units, so breadth across
    #: targets buys more than depth within one.
    max_actives_per_target: int = 25
    #: Decoys per active, reduced from the DUD-E 50 for the same reason. Recorded
    #: so the number behind any enrichment figure is explicit.
    decoys_per_active: int = 25
    #: Property-matching tolerances. "Property-matched" is meaningless without
    #: naming the axes and the windows, so both are pinned here.
    match_columns: tuple[str, ...] = (
        "molecular_weight",
        "clogp",
        "tpsa",
        "formal_charge",
        "rotatable_bonds",
    )
    match_tolerances: tuple[float, ...] = (25.0, 1.0, 20.0, 0.0, 2.0)
    #: Exclude candidate decoys measured against any construct in the target's
    #: protein group, not merely the same sequence hash.
    exclude_by_protein_group: bool = True
    seed: int = 20_261_001

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FishingReport:
    n_targets_considered: int = 0
    n_actives_before_cap: int = 0
    n_targets_capped: int = 0
    n_targets_locked: int = 0
    n_actives: int = 0
    n_measured_inactives: int = 0
    n_decoys: int = 0
    targets_without_enough_actives: int = 0
    targets_without_measured_inactives: int = 0
    decoy_shortfall: int = 0
    per_endpoint: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _eligible(observations: pd.DataFrame, definition: FishingDefinition) -> pd.DataFrame:
    frame = observations[observations["endpoint"].isin(definition.endpoints)]
    if definition.ic50_requires_tier_a:
        frame = frame[~((frame["endpoint"] == "IC50") & (frame["ic50_tier"] == "B"))]
    if definition.require_exact:
        frame = frame[frame["is_exact"].fillna(False)]
    return frame


def build_fishing_challenges(
    observations: pd.DataFrame,
    *,
    protein_groups: pd.DataFrame,
    properties: PropertyStore,
    definition: FishingDefinition | None = None,
    profile: CurationProfile | None = None,
    max_targets: int | None = None,
) -> tuple[pd.DataFrame, FishingReport]:
    """One row per (target, endpoint, candidate) with its tier and label status.

    ``protein_groups`` maps ``sequence_sha256`` to ``protein_group`` and comes from
    the exhaustive alignment, so decoy exclusion is homologue-aware.
    """
    definition = definition or FishingDefinition()
    profile = profile or get_profile()
    report = FishingReport()
    rng = np.random.default_rng(definition.seed)

    frame = _eligible(observations, definition)
    group_of = dict(zip(protein_groups["sequence_sha256"], protein_groups["protein_group"]))
    frame = frame.assign(protein_group=frame["sequence_sha256"].map(group_of))
    frame = frame[frame["protein_group"].notna()]

    # Which molecules have been measured against each protein GROUP. This is the
    # exclusion set: a compound measured against one construct of a protein is not
    # a safe decoy for another construct of the same protein.
    measured_in_group: dict[Any, set[str]] = {}
    for group, block in frame.groupby("protein_group"):
        measured_in_group[group] = set(block["identity_key"])

    property_index = properties.index
    columns = [properties.columns.index(c) for c in definition.match_columns]
    tolerances = np.asarray(definition.match_tolerances, dtype=np.float32)

    # Candidate decoy pool: every molecule with usable properties.
    pool_keys = [k for k in properties.identity_keys if properties.valid[property_index[k]]]
    pool_positions = np.asarray([property_index[k] for k in pool_keys])
    pool_values = np.asarray(properties.values)[pool_positions][:, columns]
    pool_keys_array = np.asarray(pool_keys, dtype=object)
    bins = _PropertyBins(pool_values, pool_keys_array, columns, tolerances)

    rows: list[dict[str, Any]] = []
    grouped = list(frame.groupby(["sequence_sha256", "endpoint"], sort=False))
    for index, ((target, endpoint), block) in enumerate(grouped):
        if max_targets is not None and index >= max_targets:
            break
        report.n_targets_considered += 1

        actives = block[block["p_value"] >= definition.active_p_min]
        actives = actives.drop_duplicates("identity_key")
        if len(actives) < definition.min_actives_per_target:
            report.targets_without_enough_actives += 1
            continue
        report.n_actives_before_cap += int(len(actives))
        if len(actives) > definition.max_actives_per_target:
            # Seeded sample, not the top-N by potency: taking the strongest
            # binders would make the panel easier than the corpus it came from.
            actives = actives.sample(
                n=definition.max_actives_per_target, random_state=definition.seed
            )
            report.n_targets_capped += 1

        inactives = block[block["p_value"] <= definition.inactive_p_max]
        inactives = inactives.drop_duplicates("identity_key")
        if len(inactives) > definition.max_actives_per_target:
            inactives = inactives.sample(
                n=definition.max_actives_per_target, random_state=definition.seed
            )
        if not len(inactives):
            report.targets_without_measured_inactives += 1

        group = block["protein_group"].iloc[0]
        excluded = measured_in_group.get(group, set()) if definition.exclude_by_protein_group else set(block["identity_key"])

        report.n_targets_locked += 1
        report.per_endpoint[endpoint] = report.per_endpoint.get(endpoint, 0) + 1

        for row in actives.itertuples(index=False):
            rows.append(
                {
                    "target": target,
                    "endpoint": endpoint,
                    "identity_key": row.identity_key,
                    "tier": "measured_active",
                    "binding_status": "active",
                    "p_value": float(row.p_value),
                    "observation_id": row.observation_id,
                }
            )
        for row in inactives.itertuples(index=False):
            rows.append(
                {
                    "target": target,
                    "endpoint": endpoint,
                    "identity_key": row.identity_key,
                    "tier": "measured_inactive",
                    "binding_status": "weak_or_inactive",
                    "p_value": float(row.p_value),
                    "observation_id": row.observation_id,
                }
            )

        wanted = definition.decoys_per_active * len(actives)
        decoys = _match_decoys(
            actives["identity_key"].tolist(),
            property_index,
            properties,
            columns,
            tolerances,
            pool_values,
            pool_keys_array,
            bins,
            excluded=excluded,
            wanted=wanted,
            rng=rng,
        )
        if len(decoys) < wanted:
            report.decoy_shortfall += wanted - len(decoys)
        for key in decoys:
            rows.append(
                {
                    "target": target,
                    "endpoint": endpoint,
                    "identity_key": key,
                    "tier": "property_matched_decoy",
                    # Never "inactive": nobody measured it.
                    "binding_status": "unknown",
                    "p_value": None,
                    "observation_id": None,
                }
            )

    challenges = pd.DataFrame(rows)
    if len(challenges):
        counts = challenges["tier"].value_counts()
        report.n_actives = int(counts.get("measured_active", 0))
        report.n_measured_inactives = int(counts.get("measured_inactive", 0))
        report.n_decoys = int(counts.get("property_matched_decoy", 0))
    return challenges, report


class _PropertyBins:
    """A spatial index over the decoy pool, built once.

    The naive matcher compared every active against all 1,433,489 pool molecules
    on five axes: ~30 ms per active, and with 786,222 actives that is 6.5 hours.
    Binning each continuous axis at its own tolerance width means a candidate can
    only lie in the reference's bin or an adjacent one, so a lookup touches 3^k
    small buckets instead of the whole pool. Exact-match axes (tolerance 0, e.g.
    formal charge) become part of the key directly.
    """

    def __init__(
        self,
        values: np.ndarray,
        keys: np.ndarray,
        columns: Sequence[int],
        tolerances: np.ndarray,
    ) -> None:
        self.columns = list(columns)
        self.tolerances = tolerances
        self.keys = keys
        self.values = values
        # Axes with a zero tolerance are exact-match; the rest are binned.
        self.exact = [i for i, t in enumerate(tolerances) if t == 0]
        self.binned = [i for i, t in enumerate(tolerances) if t > 0]
        self.widths = np.array(
            [tolerances[i] if tolerances[i] > 0 else 1.0 for i in range(len(tolerances))]
        )

        self.buckets: dict[tuple, list[int]] = {}
        quantised = np.floor_divide(values, self.widths).astype(np.int64)
        for row in range(values.shape[0]):
            self.buckets.setdefault(self._key(quantised[row], values[row]), []).append(row)

    def _key(self, quantised: np.ndarray, raw: np.ndarray) -> tuple:
        return (
            tuple(int(quantised[i]) for i in self.binned)
            + tuple(float(raw[i]) for i in self.exact)
        )

    def _buckets_near(self, reference: np.ndarray) -> list[list[int]]:
        quantised = np.floor_divide(reference, self.widths).astype(np.int64)
        offsets = [(-1, 0, 1)] * len(self.binned)
        found: list[list[int]] = []
        for shift in itertools.product(*offsets) if offsets else [()]:
            key = (
                tuple(int(quantised[axis]) + delta for axis, delta in zip(self.binned, shift))
                + tuple(float(reference[axis]) for axis in self.exact)
            )
            bucket = self.buckets.get(key)
            if bucket:
                found.append(bucket)
        return found

    def sample(self, reference: np.ndarray, n: int, rng: np.random.Generator) -> list[int]:
        """Up to ``n`` pool rows inside the tolerance window, sampled not enumerated.

        The window around a typical active holds ~19,000 molecules on this corpus,
        and only a couple of dozen decoys are wanted, so materialising the window
        was the remaining cost once the full-pool scan was gone. Drawing at random
        from the neighbouring buckets and verifying each draw against the exact
        window gives the same distribution for a few microseconds.
        """
        buckets = self._buckets_near(reference)
        if not buckets:
            return []
        sizes = np.array([len(b) for b in buckets], dtype=np.float64)
        weights = sizes / sizes.sum()
        picked: list[int] = []
        seen: set[int] = set()
        # Bounded attempts, so a sparse window cannot spin.
        for _ in range(n * 20):
            if len(picked) >= n:
                break
            bucket = buckets[int(rng.choice(len(buckets), p=weights))]
            row = int(bucket[int(rng.integers(0, len(bucket)))])
            if row in seen:
                continue
            seen.add(row)
            if np.all(np.abs(self.values[row] - reference) <= self.tolerances):
                picked.append(row)
        return picked

    def candidates(self, reference: np.ndarray) -> np.ndarray:
        """Pool rows that could lie inside the tolerance window of ``reference``."""
        quantised = np.floor_divide(reference, self.widths).astype(np.int64)
        offsets = [(-1, 0, 1)] * len(self.binned)
        rows: list[int] = []
        for shift in itertools.product(*offsets) if offsets else [()]:
            key = (
                tuple(int(quantised[axis]) + delta for axis, delta in zip(self.binned, shift))
                + tuple(float(reference[axis]) for axis in self.exact)
            )
            bucket = self.buckets.get(key)
            if bucket:
                rows.extend(bucket)
        if not rows:
            return np.zeros(0, dtype=np.int64)
        index = np.asarray(rows, dtype=np.int64)
        # Adjacent bins are a superset, so the exact window is still enforced.
        within = np.all(
            np.abs(self.values[index] - reference) <= self.tolerances, axis=1
        )
        return index[within]


def _match_decoys(
    active_keys: Sequence[str],
    property_index: Mapping[str, int],
    properties: PropertyStore,
    columns: Sequence[int],
    tolerances: np.ndarray,
    pool_values: np.ndarray,
    pool_keys: np.ndarray,
    bins: "_PropertyBins",
    *,
    excluded: set[str],
    wanted: int,
    rng: np.random.Generator,
) -> list[str]:
    """Property-matched molecules never measured against the target's group."""
    chosen: list[str] = []
    taken: set[str] = set()
    values = np.asarray(properties.values)
    index_columns = list(columns)

    # Spread the quota across actives rather than exhausting it on the first one,
    # so the decoy set mirrors the actives' property distribution instead of one
    # active's neighbourhood.
    per_active = -(-wanted // max(len(active_keys), 1))

    for key in active_keys:
        position = property_index.get(key)
        if position is None or not properties.valid[position]:
            continue
        reference = values[position][index_columns]
        # Over-draw a little, because some draws will be excluded or already taken.
        rows = bins.sample(reference, per_active * 4, rng)
        if not rows:
            continue
        candidates = pool_keys[np.asarray(rows, dtype=np.int64)]

        added = 0
        for candidate in candidates:
            if len(chosen) >= wanted:
                return chosen
            if candidate in excluded or candidate in taken:
                continue
            taken.add(candidate)
            chosen.append(candidate)
            added += 1
            if added >= per_active:
                break
    return chosen


def lock_fishing_panel(
    challenges: pd.DataFrame,
    *,
    definition: FishingDefinition,
    inputs: Mapping[str, Any],
    directory: Path | str,
    report: FishingReport | None = None,
    overwrite: bool = False,
):
    """Lock the challenge set: candidate lists open, measured labels sealed.

    Decoys live in ``unknowns`` rather than ``labels``, so there is no table in
    which a decoy could acquire an affinity value.
    """
    from bindra.eval.benchmarks.lock import lock

    measured = challenges[challenges["tier"] != "property_matched_decoy"]
    decoys = challenges[challenges["tier"] == "property_matched_decoy"]

    query_columns = ["target", "endpoint", "identity_key", "tier"]
    return lock(
        "fishing",
        definition,
        dict(inputs),
        Path(directory) / "fishing",
        queries=challenges[query_columns],
        labels=measured[["target", "endpoint", "identity_key", "binding_status", "p_value"]],
        unknowns=decoys[["target", "endpoint", "identity_key", "tier", "binding_status"]],
        attrition=(report.to_dict() if report else None),
        notes=[
            "Two control tiers, deliberately separate. measured_inactive are "
            "experimentally measured weak/inactive compounds and are real "
            "negatives. property_matched_decoy are PRESUMED controls with unknown "
            "binding status; they carry no affinity label and appear only in the "
            "unknowns table.",
            "Decoy exclusion is homologue-aware: a candidate measured against any "
            "construct in the target's protein group is excluded, because the same "
            "protein appears here as many constructs.",
            "DUD-E-style decoys favour 2D similarity methods, so any enrichment "
            "number must be reported alongside fingerprint-similarity and "
            "property-only baselines.",
            "Report target-only fishing separately from seed-assisted fishing: a "
            "known binder given as a seed is chemical information, not zero-shot "
            "discovery.",
            "Metrics are computed over the FULL locked pool, counting actives missed "
            "by any retrieval or shortlisting stage.",
        ],
        overwrite=overwrite,
    )


__all__ = [
    "FishingDefinition",
    "FishingReport",
    "build_fishing_challenges",
    "lock_fishing_panel",
]
