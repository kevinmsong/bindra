"""Matched analog pairs and measured delta_y, train partitions only.

A pair is two molecules measured **against the same target construct, on the
same endpoint, in a compatible assay context**, whose measured potencies differ.
Four rules decide what counts, and each exists because breaking it produces a
dataset that trains well and means nothing:

1.  **Same construct, endpoint and assay context.** A pKi against one construct
    minus a pIC50 against another is not a potency change; it is two unrelated
    numbers subtracted.
2.  **Exact labels only.** A censored record has no measured value. Subtracting
    a ">10 uM" threshold as if it were an affinity invents a delta whose sign is
    determined by the detection limit of the assay.
3.  **Train partitions only.** A pair is built from two observations, and if
    either lands in val or test the delta carries a held-out label into
    training.
4.  **Large series are capped.** One optimisation campaign can contribute
    thousands of pairs from a few dozen molecules, and uncapped it dominates the
    objective with one chemotype.

Owed by the specification:
    Upgrade A: compatible target construct, endpoint and assay context;
    exact labels only; do not subtract censoring thresholds as if they were
    measured affinities; cap large-series influence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bindra.training.losses.affinity import noise_floor_table


class AnalogPairError(ValueError):
    """Pairs were requested from data that cannot support them."""


@dataclass(frozen=True)
class PairPolicy:
    """What counts as a usable analog pair."""

    #: Multiplier on the endpoint's replicate-noise floor. A delta smaller than
    #: this is indistinguishable from repeating the same measurement twice.
    margin_scale: float = 1.0
    #: Hard cap on pairs drawn from any one (construct, endpoint, assay) group.
    max_pairs_per_group: int = 256
    #: Groups below this many molecules cannot form a meaningful series.
    min_group_size: int = 2
    #: Partitions a pair's observations may come from.
    allowed_partitions: tuple[str, ...] = ("train",)

    def to_dict(self) -> dict:
        return {
            "margin_scale": self.margin_scale,
            "max_pairs_per_group": self.max_pairs_per_group,
            "min_group_size": self.min_group_size,
            "allowed_partitions": list(self.allowed_partitions),
            "exact_labels_only": True,
            "censored_thresholds_subtracted": False,
        }


@dataclass(frozen=True)
class AnalogPairs:
    """The built pairs, plus what was discarded and why."""

    left: np.ndarray
    right: np.ndarray
    delta: np.ndarray
    group_id: np.ndarray
    endpoint: np.ndarray
    attrition: dict

    def __len__(self) -> int:
        return int(self.left.shape[0])

    def to_dict(self) -> dict:
        return {"n_pairs": len(self), "attrition": self.attrition}


def build_pairs(
    frame,
    *,
    partition_of: "dict[str, str] | None" = None,
    policy: PairPolicy | None = None,
    seed: int = 20261001,
) -> AnalogPairs:
    """Build matched pairs from a curated observation frame.

    ``frame`` needs ``observation_id``, ``sequence_sha256``, ``endpoint``,
    ``p_value``, ``is_exact`` and ``assay_condition_bucket``. ``partition_of``
    maps observation id to partition; without it nothing can be checked and the
    call is refused rather than producing pairs that may contain held-out rows.
    """
    policy = policy or PairPolicy()
    required = {
        "observation_id", "sequence_sha256", "endpoint",
        "p_value", "is_exact", "assay_condition_bucket",
    }
    missing = required - set(frame.columns)
    if missing:
        raise AnalogPairError(f"frame is missing {sorted(missing)}")
    if partition_of is None:
        raise AnalogPairError(
            "partition_of is required: a pair built from a val or test "
            "observation carries a held-out label into training, and that "
            "cannot be checked after the fact"
        )

    attrition = {"input_rows": int(len(frame))}

    # Rule 2: exact only. Censored rows are dropped, never differenced.
    exact = frame[frame["is_exact"].astype(bool)]
    attrition["dropped_censored"] = int(len(frame) - len(exact))

    # Rule 3: train partitions only.
    allowed = set(policy.allowed_partitions)
    keep = exact["observation_id"].map(lambda o: partition_of.get(o) in allowed)
    in_partition = exact[keep.to_numpy(dtype=bool)]
    attrition["dropped_wrong_partition"] = int(len(exact) - len(in_partition))

    floors = noise_floor_table()
    rng = np.random.default_rng(seed)
    left, right, delta, groups, endpoints = [], [], [], [], []
    dropped_below_margin = capped = 0

    # Rule 1: the grouping IS the compatibility rule.
    grouped = in_partition.groupby(
        ["sequence_sha256", "endpoint", "assay_condition_bucket"], sort=False
    )
    for (_, endpoint, _), block in grouped:
        if len(block) < policy.min_group_size:
            continue
        values = block["p_value"].to_numpy(dtype=float)
        ids = block["observation_id"].to_numpy()
        margin = floors.get(str(endpoint), 0.0) * policy.margin_scale
        i, j = np.triu_indices(len(values), k=1)
        gap = values[j] - values[i]
        usable = np.abs(gap) > margin
        dropped_below_margin += int((~usable).sum())
        i, j, gap = i[usable], j[usable], gap[usable]
        # Rule 4: cap the series.
        if i.size > policy.max_pairs_per_group:
            capped += int(i.size - policy.max_pairs_per_group)
            choice = rng.choice(i.size, policy.max_pairs_per_group, replace=False)
            i, j, gap = i[choice], j[choice], gap[choice]
        if not i.size:
            continue
        left.append(ids[i])
        right.append(ids[j])
        delta.append(gap)
        groups.append(np.full(i.size, f"{block.iloc[0]['sequence_sha256'][:12]}|{endpoint}"))
        endpoints.append(np.full(i.size, str(endpoint)))

    attrition["dropped_below_noise_floor"] = dropped_below_margin
    attrition["dropped_by_series_cap"] = capped
    empty = np.array([], dtype=object)
    return AnalogPairs(
        left=np.concatenate(left) if left else empty,
        right=np.concatenate(right) if right else empty,
        delta=np.concatenate(delta) if delta else np.array([], dtype=float),
        group_id=np.concatenate(groups) if groups else empty,
        endpoint=np.concatenate(endpoints) if endpoints else empty,
        attrition=attrition,
    )


def assert_no_held_out(pairs: AnalogPairs, partition_of: "dict[str, str]", allowed=("train",)) -> None:
    """Every observation in every pair must be in an allowed partition."""
    allowed_set = set(allowed)
    for side in (pairs.left, pairs.right):
        for observation in side:
            where = partition_of.get(str(observation))
            if where not in allowed_set:
                raise AnalogPairError(
                    f"observation {observation} is in partition {where!r}; a pair "
                    "built from it would carry a held-out label into training"
                )


__all__ = [
    "AnalogPairError",
    "AnalogPairs",
    "PairPolicy",
    "assert_no_held_out",
    "build_pairs",
]
