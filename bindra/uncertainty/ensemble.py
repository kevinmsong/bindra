"""Independently seeded ensemble inference.

Owed by the specification:
    Priority 4: three independently trained models; dropout passes are not
    an ensemble.

That clause is a *check*, not a comment. Three forward passes with dropout
left on will produce a spread, and that spread will look like uncertainty on
a plot, so the cheapest way to appear calibrated is to do the wrong thing.
:func:`assert_independent` refuses it: members must be distinct files, carry
distinct seeds in their provenance, and hold weights that actually differ.

What the spread means is a separate question, and a weaker one than the
arithmetic suggests. Disagreement between independently seeded runs is a
**heuristic** signal. It is not a calibrated interval and carries no coverage
guarantee; that is what :mod:`bindra.uncertainty.conformal` is for, and it is
why the two are separate modules rather than one.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

_SPEC_CLAUSE = (
    "Priority 4: three independently trained models; dropout passes are not an ensemble."
)

#: The specification asks for three. Two gives a standard deviation that is
#: just a half-difference, which is a worse estimator than it looks.
RECOMMENDED_MEMBERS = 3


class EnsembleError(RuntimeError):
    """The members do not constitute an independently trained ensemble."""


@dataclass(frozen=True)
class EnsembleMember:
    """One trained checkpoint, with what makes it independent of the others."""

    path: Path
    seed: int | None
    weight_digest: str
    epoch: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "seed": self.seed,
            "weight_digest": self.weight_digest,
            "epoch": self.epoch,
        }


@dataclass(frozen=True)
class EnsemblePrediction:
    """Mean and spread in p-units, per observation."""

    mean: np.ndarray
    std: np.ndarray
    members: tuple[EnsembleMember, ...]
    #: [n_members, n] raw per-member predictions, kept so a caller can inspect
    #: a disagreement rather than only its summary.
    per_member: np.ndarray

    @property
    def n_members(self) -> int:
        return len(self.members)

    def to_dict(self) -> dict[str, object]:
        return {
            "n_members": self.n_members,
            "members": [m.to_dict() for m in self.members],
            "mean_spread": float(np.mean(self.std)) if self.std.size else 0.0,
            "note": (
                "Ensemble disagreement is a heuristic uncertainty signal, not a "
                "calibrated interval and not a guarantee of correctness."
            ),
        }


def _digest(state: dict) -> str:
    """Content hash of a state dict, so identical weights are detectable."""
    hasher = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if torch.is_tensor(value):
            hasher.update(key.encode("utf-8"))
            hasher.update(value.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()[:16]


def describe_member(path: Path | str) -> EnsembleMember:
    """Read a checkpoint's identity without building the model."""
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload.get("config") or {}
    seed = config.get("seed") if isinstance(config, dict) else None
    return EnsembleMember(
        path=path,
        seed=None if seed is None else int(seed),
        weight_digest=_digest(payload.get("model", {})),
        epoch=payload.get("epoch"),
    )


def assert_independent(members: Sequence[EnsembleMember]) -> None:
    """Refuse anything that is not genuinely several trained models.

    Three failure modes, in increasing order of how plausible the result looks:
    one checkpoint listed twice, several checkpoints from one run at different
    epochs, and -- the one the specification names -- repeated passes over a
    single model dressed up as members. All three show up as duplicate weight
    digests or duplicate seeds.
    """
    if len(members) < 2:
        raise EnsembleError(
            f"an ensemble needs at least 2 members, got {len(members)}. "
            f"Specification requirement: {_SPEC_CLAUSE}"
        )

    digests = [m.weight_digest for m in members]
    if len(set(digests)) != len(digests):
        raise EnsembleError(
            "two or more members hold identical weights. Repeated passes over "
            "one model are not an ensemble, however they are averaged. "
            f"Specification requirement: {_SPEC_CLAUSE}"
        )

    seeds = [m.seed for m in members if m.seed is not None]
    if len(set(seeds)) != len(seeds):
        duplicated = sorted({s for s in seeds if seeds.count(s) > 1})
        raise EnsembleError(
            f"members share seed(s) {duplicated}; they are not independently "
            "trained. Train with different --seed values, which changes both "
            f"initialisation and data order. Specification requirement: {_SPEC_CLAUSE}"
        )


def combine(
    per_member: np.ndarray, members: Sequence[EnsembleMember]
) -> EnsemblePrediction:
    """Mean and sample standard deviation across members, per observation.

    ``ddof=1`` because these are a sample of training runs, not the population
    of them. With three members the difference from ``ddof=0`` is 22%, which is
    large enough to matter to anything thresholded on the spread.
    """
    array = np.asarray(per_member, dtype=np.float64)
    if array.ndim != 2:
        raise EnsembleError(
            f"per_member must be [n_members, n], got {array.shape}"
        )
    if array.shape[0] != len(members):
        raise EnsembleError(
            f"{array.shape[0]} prediction rows against {len(members)} members"
        )
    assert_independent(members)
    return EnsemblePrediction(
        mean=array.mean(axis=0),
        std=array.std(axis=0, ddof=1),
        members=tuple(members),
        per_member=array,
    )


__all__ = [
    "RECOMMENDED_MEMBERS",
    "EnsembleError",
    "EnsembleMember",
    "EnsemblePrediction",
    "assert_independent",
    "combine",
    "describe_member",
]
