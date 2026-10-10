"""Listwise objective over measured candidates.

**An unmeasured pair is not a nonbinder.** The standard in-batch contrastive
loss treats every other ligand in the batch as a negative for every protein,
which on this corpus is simply false: BindingDB is a record of what was
measured, and the overwhelming majority of protein-ligand pairs were never
tested. Training on that assumption teaches the model that the absence of a
measurement is evidence of non-binding, which is the opposite of true -- people
measure pairs they expect to bind.

**A weaker measured ligand is not automatically a nonbinder either.** A 10 uM
binder is a real binder. So negatives are graded rather than binary: the loss
only asks that a stronger measured ligand outranks a weaker measured one, and
only when the gap between them exceeds that endpoint's replicate-noise floor.
Pairs closer than the floor carry no information about ordering and are dropped.

Owed by the specification:
    Priority 2: do not treat every unmeasured pair as a nonbinder; a weaker
    measured ligand is not automatically a nonbinder.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from bindra.training.losses.affinity import noise_floor_table


class RetrievalLossError(ValueError):
    """The listwise objective was asked for something it cannot score."""


@dataclass(frozen=True)
class RetrievalLossConfig:
    """What counts as a usable comparison."""

    #: Multiplier on the endpoint's replicate-noise floor. A pair whose
    #: potency gap is under this is not a reliable ordering and is dropped.
    margin_scale: float = 1.0
    #: Cap on comparisons drawn per query, so one heavily-measured target
    #: cannot dominate the gradient.
    max_pairs_per_query: int = 64
    #: Include pairs where the weaker side is censored ("> 10 uM"). The bound
    #: still orders them when the stronger side is below it.
    include_censored: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "margin_scale": self.margin_scale,
            "max_pairs_per_query": self.max_pairs_per_query,
            "include_censored": self.include_censored,
            "negatives": "measured-and-weaker only; unmeasured pairs are never negatives",
        }


@dataclass(frozen=True)
class RetrievalLossOutput:
    total: torch.Tensor
    n_pairs: int
    n_dropped_below_margin: int
    n_queries: int

    def to_dict(self) -> dict[str, object]:
        return {
            "total": float(self.total.detach()),
            "n_pairs": self.n_pairs,
            "n_dropped_below_margin": self.n_dropped_below_margin,
            "n_queries": self.n_queries,
        }


def listwise_loss(
    scores: torch.Tensor,  # [Q, C] similarity of each query to each candidate
    potency: torch.Tensor,  # [Q, C] measured pX, NaN where unmeasured
    endpoint_index: torch.Tensor,  # [Q, C] long
    *,
    config: RetrievalLossConfig | None = None,
    is_censored: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> RetrievalLossOutput:
    """Softplus ranking loss over measured pairs only.

    ``potency`` carries NaN for a pair that was never measured, and those
    positions take no part in any comparison -- not as positives and, crucially,
    not as negatives.
    """
    config = config or RetrievalLossConfig()
    if scores.shape != potency.shape:
        raise RetrievalLossError(
            f"scores {tuple(scores.shape)} and potency {tuple(potency.shape)} differ"
        )
    floors = noise_floor_table()
    floor_values = torch.tensor(
        [floors[name] for name in sorted(floors)], dtype=torch.float32, device=scores.device
    )
    # Map the endpoint index onto its floor through the same ordering the
    # affinity head uses.
    from bindra.heads.affinity import ENDPOINT_ORDER

    ordered = torch.tensor(
        [floors[str(getattr(e, "value", e))] for e in ENDPOINT_ORDER],
        dtype=torch.float32,
        device=scores.device,
    )
    del floor_values

    measured = ~torch.isnan(potency)
    if is_censored is not None and not config.include_censored:
        measured = measured & ~is_censored

    total = scores.new_zeros(())
    n_pairs = n_dropped = n_queries = 0
    for query in range(scores.shape[0]):
        keep = measured[query]
        if int(keep.sum()) < 2:
            continue
        index = torch.nonzero(keep, as_tuple=False).squeeze(-1)
        values, candidate_scores = potency[query][index], scores[query][index]
        margins = ordered[endpoint_index[query][index]] * config.margin_scale

        # All ordered pairs (i stronger than j), then filter by the margin.
        gap = values.unsqueeze(1) - values.unsqueeze(0)
        pair_margin = torch.maximum(margins.unsqueeze(1), margins.unsqueeze(0))
        usable = gap > pair_margin
        n_dropped += int(((gap > 0) & ~usable).sum())
        pairs = torch.nonzero(usable, as_tuple=False)
        if not pairs.numel():
            continue
        if pairs.shape[0] > config.max_pairs_per_query:
            choice = torch.randperm(pairs.shape[0], generator=generator, device=pairs.device)
            pairs = pairs[choice[: config.max_pairs_per_query]]
        stronger, weaker = pairs[:, 0], pairs[:, 1]
        # Softplus rather than hinge: a smooth gradient everywhere, so a pair
        # that is already ordered still contributes a little signal.
        total = total + torch.nn.functional.softplus(
            candidate_scores[weaker] - candidate_scores[stronger]
        ).mean()
        n_pairs += int(pairs.shape[0])
        n_queries += 1

    if n_queries:
        total = total / n_queries
    return RetrievalLossOutput(total, n_pairs, n_dropped, n_queries)


def assert_no_unmeasured_negatives(potency: torch.Tensor, used: torch.Tensor) -> None:
    """Every position that entered the loss must have been measured."""
    if bool((used & torch.isnan(potency)).any()):
        raise RetrievalLossError(
            "an unmeasured pair was used as a negative; absence of a "
            "measurement is not evidence of non-binding"
        )


__all__ = [
    "RetrievalLossConfig",
    "RetrievalLossError",
    "RetrievalLossOutput",
    "assert_no_unmeasured_negatives",
    "listwise_loss",
]
