"""Ordered arithmetic aggregation for the declared three-member Davis ensemble.

Inputs are exported member predictions, not encoder features or checkpoints.
The returned member SD is descriptive and is not a calibrated prediction interval.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


DAVIS_SEEDS = (20261001, 20261002, 20261003)


@dataclass(frozen=True)
class MemberPredictions:
    """One member's pKd values and their original, ordered observation IDs."""

    seed: int
    pair_ids: Sequence[str]
    scores: Sequence[float] | np.ndarray
    label_scale: str = "pKd"


def _seed(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("Member seeds must be integer identifiers")
    return int(value)


def aggregate_davis(
    members: Sequence[MemberPredictions],
    *,
    expected_seeds: Sequence[int] = DAVIS_SEEDS,
) -> pd.DataFrame:
    """Return aligned member predictions, arithmetic mean pKd, and sample SD.

    Exactly three distinct seeds are required. The default seed set identifies
    the reported Davis ensemble; an explicit alternative supports separate runs.
    Member columns are sorted by seed, and rows preserve the supplied common
    observation order. Differing row orders are rejected, never silently aligned.
    Use three members of the same declared architecture and provide the IDs that
    originally accompanied each member's prediction array.
    """
    expected = tuple(sorted(_seed(seed) for seed in expected_seeds))
    if len(expected) != 3 or len(set(expected)) != 3:
        raise ValueError("Expected exactly three distinct seed identifiers")
    members = tuple(members)
    if len(members) != 3:
        raise ValueError("The Davis ensemble requires exactly three members")
    if not all(isinstance(member, MemberPredictions) for member in members):
        raise ValueError("Members must be MemberPredictions records")

    by_seed = {}
    pair_ids = None
    for member in members:
        seed = _seed(member.seed)
        if seed in by_seed:
            raise ValueError("Member seed identifiers must be distinct")
        if member.label_scale != "pKd":
            raise ValueError("Davis aggregation requires trained pKd predictions")
        ids = tuple(member.pair_ids)
        if not ids or any(not isinstance(value, str) or not value for value in ids):
            raise ValueError("Pair IDs must be nonempty strings")
        if len(set(ids)) != len(ids):
            raise ValueError("Pair IDs must be unique within each member")
        if pair_ids is None:
            pair_ids = ids
        elif ids != pair_ids:
            raise ValueError("Member pair IDs and ordering must match exactly")

        raw = np.asarray(member.scores)
        if raw.dtype.kind not in "fiu":
            raise ValueError("Member scores must be real numeric predictions")
        values = np.array(raw, dtype=np.float64, copy=True)
        if values.ndim != 1 or values.shape[0] != len(ids):
            raise ValueError("Member scores must be one-dimensional and aligned with pair IDs")
        if not np.isfinite(values).all():
            raise ValueError("Member scores must be finite")
        by_seed[seed] = values

    if set(by_seed) != set(expected):
        raise ValueError("Member seeds do not match the declared ensemble")
    matrix = np.stack([by_seed[seed] for seed in expected])
    with np.errstate(over="raise", invalid="raise"):
        try:
            mean = matrix.mean(axis=0)
            sample_sd = matrix.std(axis=0, ddof=1)
        except FloatingPointError as error:
            raise ValueError("Member values produced nonfinite aggregate statistics") from error
    result = {"pair_id": list(pair_ids)}
    result.update({f"member_{seed}_pKd": by_seed[seed] for seed in expected})
    result["mean_pKd"] = mean
    result["member_SD_pKd"] = sample_sd
    return pd.DataFrame(result, copy=True)


__all__ = ["DAVIS_SEEDS", "MemberPredictions", "aggregate_davis"]
