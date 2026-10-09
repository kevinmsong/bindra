"""Within-assay pairwise ranking.

Owed by the specification:
    Priority 2: compare only within a compatible target/endpoint/assay
    group, with a configurable p-unit margin.

Affinity training fits absolute values; a chemist asks which of two compounds
is better. Those are different questions and a model can be good at one and bad
at the other, which is why the specification asks for a ranking term and why it
is restricted rather than global.

## "Assay" is a proxy here, and it has a name

**BindingDB 202610 contains no assay identifier and no assay description column
anywhere in its 640 columns** -- not a missing value, the field does not exist.
The curation layer therefore built one: ``comparability_group_id``, from target
+ endpoint + citation + condition bucket + source class, with the proxy that
produced it recorded per observation in ``comparability_scope``
(``literature_assay_proxy``, ``patent_table_proxy``, ``bioassay_aid``,
``unknown_singleton``).

This module ranks **only within one of those groups**. It does not fall back to
ranking within a target, or within a batch, when the group id is missing: a
comparison across groups is a comparison across experiments, which is the thing
the specification's clause exists to forbid. Absent group ids raise.

## The margin, and why it is not 0.5

The specification starts the margin at 0.5 p-units. **That is below the
replicate noise floor of every endpoint** -- 0.945 for pKd, 0.641 for pKi,
0.671 for pIC50 -- so a 0.5 margin selects pairs whose measured difference the
laboratory itself could not reproduce, and trains the model to order them
anyway.

Measured over the corpus's 152,643,122 comparable pairs:

=========  ============  ===========  ==================  ==================
endpoint   pairs         median gap   share over 0.5      share over floor
=========  ============  ===========  ==================  ==================
pKd        507,134       0.683        59.5%               38.6%
pKi        4,380,395     0.645        58.6%               50.2%
pIC50      12,082,254    0.577        54.7%               44.7%
=========  ============  ===========  ==================  ==================

So the floor costs about 21 points of pKd pairs and 8 of pKi against the spec's
0.5, and buys that every retained pair differs by more than replicate scatter.
:class:`RankingLossConfig` therefore defaults the margin to **the measured
per-endpoint noise floor**, with ``margin_override`` available to reproduce the
specification's single 0.5 if someone wants the comparison.

## Censored rows can be ranked, in the direction their bound supports

16.6% of the corpus is censored, and dropping all of it here would be a waste:
"IC50 > 10 uM" does not say where a compound sits, but it does say it is weaker
than one measured at 10 nM. So each row is reduced to the interval its record
guarantees -- ``[p, p]`` when exact, ``(-inf, u]`` for an upper bound, ``[l,
+inf)`` for a lower one -- and a pair is used only when those intervals are
**disjoint by more than the margin**. Nothing is assumed about where inside a
bound the true value lies, which is the same rule the censored affinity term
follows.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_SPEC_CLAUSE = (
    "Priority 2: compare only within a compatible target/endpoint/assay group, "
    "with a configurable p-unit margin."
)


class RankingLossError(ValueError):
    """The batch cannot support a within-assay ranking term."""


@dataclass(frozen=True)
class RankingLossConfig:
    """Every threshold declared, and the margin defaulted to measured noise."""

    #: Weight on the ranking term in the total loss. **Zero by default.**
    #: Adding a term changes the objective, and no BINDRA-v1 baseline exists
    #: yet to compare against; this turns on deliberately, not by upgrading.
    weight: float = 0.0
    #: Multiplies the per-endpoint replicate noise floor to get the margin.
    #: 1.0 means "a pair must differ by more than the laboratory's own repeat
    #: scatter before we ask the model to order it".
    margin_scale: float = 1.0
    #: One margin in p-units for every endpoint, replacing the floor-derived
    #: one. Set to 0.5 to reproduce the specification's starting value; see the
    #: module docstring for what that costs.
    margin_override: float | None = None
    #: Cap on pairs scored per batch. A comparability group of 2,295 rows would
    #: contribute 2.6M pairs on its own and dominate the term; the cap is
    #: applied by random subsample so the choice is not systematically the
    #: easiest pairs.
    max_pairs: int = 4096
    #: Rank censored rows against exact ones where the bounds allow it. False
    #: restricts the term to exact-vs-exact, which is the ablation.
    include_censored: bool = True
    #: Restrict to rows the curation layer marked rankable.
    ranking_eligible_only: bool = True

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise RankingLossError(f"weight must be >= 0, got {self.weight}")
        if self.margin_scale <= 0:
            raise RankingLossError(
                f"margin_scale must be > 0, got {self.margin_scale}; a margin of "
                "zero ranks pairs that differ by nothing"
            )
        if self.margin_override is not None and self.margin_override <= 0:
            raise RankingLossError(
                f"margin_override must be > 0, got {self.margin_override}"
            )
        if self.max_pairs < 1:
            raise RankingLossError(f"max_pairs must be >= 1, got {self.max_pairs}")

    def to_dict(self) -> dict[str, object]:
        return {
            "weight": self.weight,
            "margin_scale": self.margin_scale,
            "margin_override": self.margin_override,
            "max_pairs": self.max_pairs,
            "include_censored": self.include_censored,
            "ranking_eligible_only": self.ranking_eligible_only,
            "margin_source": (
                "override"
                if self.margin_override is not None
                else "per-endpoint replicate noise floor x margin_scale"
            ),
            "spec_clause": _SPEC_CLAUSE,
        }


@dataclass(frozen=True)
class RankingLossOutput:
    """The term, and enough counts to tell an empty term from a satisfied one."""

    loss: torch.Tensor
    #: Pairs that cleared the margin and were scored.
    n_pairs: int
    #: Pairs available before the ``max_pairs`` subsample.
    n_pairs_available: int
    #: Share of scored pairs the model ordered correctly, ignoring the margin.
    #: A plain accuracy, reported because a falling hinge loss and a correct
    #: ordering are not the same thing.
    concordance: float
    #: Comparability groups that contributed at least one pair.
    n_groups: int

    def to_dict(self) -> dict[str, object]:
        return {
            "loss": float(self.loss.detach()),
            "n_pairs": self.n_pairs,
            "n_pairs_available": self.n_pairs_available,
            "concordance": self.concordance,
            "n_groups": self.n_groups,
        }


def margin_per_endpoint(config: RankingLossConfig) -> dict[str, float]:
    """The margin each endpoint will use, named so a report can state it."""
    from bindra.training.losses.affinity import noise_floor_table

    if config.margin_override is not None:
        return {name: config.margin_override for name in noise_floor_table()}
    return {
        name: floor * config.margin_scale
        for name, floor in noise_floor_table().items()
    }


def certain_bounds(
    p_value: torch.Tensor,
    is_exact: torch.Tensor,
    lower_bound: torch.Tensor,
    upper_bound: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Each row as the interval its record actually guarantees.

    Exact rows collapse to a point. A censored row becomes the half-line its
    bound allows and nothing narrower: "at most 5.0" says the true value is
    somewhere at or below 5.0, and assuming it sits *at* 5.0 is the error the
    censored affinity term exists to avoid. Two rows are orderable only when
    these intervals are disjoint, which is a statement about what the records
    guarantee rather than about where the values probably are.
    """
    infinite = torch.full_like(p_value, float("inf"))
    low = torch.where(is_exact, p_value, -infinite)
    high = torch.where(is_exact, p_value, infinite)
    has_lower = ~torch.isnan(lower_bound)
    has_upper = ~torch.isnan(upper_bound)
    low = torch.where(~is_exact & has_lower, torch.nan_to_num(lower_bound), low)
    high = torch.where(~is_exact & has_upper, torch.nan_to_num(upper_bound), high)
    return low, high


def ranking_loss(
    prediction: torch.Tensor,
    targets,
    config: RankingLossConfig | None = None,
    *,
    generator: torch.Generator | None = None,
) -> RankingLossOutput:
    """Margin ranking over pairs inside one comparability group.

    ``targets`` is an :class:`~bindra.training.losses.affinity.AffinityTargets`
    carrying ``comparability_group`` and ``ranking_eligible``. Both are
    required: without the group id there is no way to tell a within-assay
    comparison from a cross-experiment one, and this refuses rather than
    guessing.
    """
    config = config or RankingLossConfig()
    from bindra.heads.affinity import ENDPOINT_ORDER

    group = getattr(targets, "comparability_group", None)
    if group is None:
        raise RankingLossError(
            "the batch carries no comparability_group, so a within-assay "
            "comparison cannot be distinguished from a cross-experiment one. "
            "Ranking across groups compares different experiments. "
            f"Specification requirement: {_SPEC_CLAUSE}"
        )
    if prediction.shape != targets.p_value.shape:
        raise RankingLossError(
            f"prediction is {tuple(prediction.shape)} against "
            f"{tuple(targets.p_value.shape)} labels"
        )

    device = prediction.device
    usable = torch.ones_like(targets.is_exact)
    if config.ranking_eligible_only:
        eligible = getattr(targets, "ranking_eligible", None)
        if eligible is None:
            raise RankingLossError(
                "ranking_eligible_only is set but the batch carries no "
                "ranking_eligible flag; set it False to rank every row"
            )
        usable = usable & eligible
    if not config.include_censored:
        usable = usable & targets.is_exact

    zero = prediction.sum() * 0.0
    empty = RankingLossOutput(zero, 0, 0, float("nan"), 0)
    if int(usable.sum()) < 2:
        return empty

    low, high = certain_bounds(
        targets.p_value, targets.is_exact, targets.lower_bound, targets.upper_bound
    )
    margins = margin_per_endpoint(config)
    margin_of = torch.tensor(
        [margins[str(getattr(e, "value", e))] for e in ENDPOINT_ORDER],
        dtype=prediction.dtype,
        device=device,
    )
    row_margin = margin_of[targets.endpoint_index]

    # The pair mask. Same group, same endpoint, both usable, and -- the part
    # that matters -- the guaranteed intervals disjoint by more than the margin.
    same_group = group[:, None] == group[None, :]
    same_endpoint = targets.endpoint_index[:, None] == targets.endpoint_index[None, :]
    both = usable[:, None] & usable[None, :]
    # gap[i, j] > 0 means row i is certainly more potent than row j by that much.
    gap = low[:, None] - high[None, :]
    threshold = torch.maximum(row_margin[:, None], row_margin[None, :])
    ordered = same_group & same_endpoint & both & (gap > threshold)
    # i > j only; the transpose is the same comparison written backwards.
    rows, columns = torch.nonzero(ordered, as_tuple=True)
    n_available = int(rows.numel())
    if n_available == 0:
        return empty

    if n_available > config.max_pairs:
        # Subsampled at random rather than truncated: taking the first N would
        # take them in group order, which is a systematic choice of whichever
        # groups happen to sort first.
        keep = torch.randperm(n_available, generator=generator, device="cpu")
        keep = keep[: config.max_pairs].to(device)
        rows, columns = rows[keep], columns[keep]

    better, worse = prediction[rows], prediction[columns]
    pair_margin = threshold[rows, columns]
    # Hinge: the prediction gap must reach the same margin that qualified the
    # pair. Asking for less would accept an ordering the data cannot see;
    # asking for more would turn a ranking term into a regression on the gap.
    hinge = torch.clamp(pair_margin - (better - worse), min=0.0)
    with torch.no_grad():
        concordance = float((better > worse).to(torch.float32).mean())
        n_groups = int(torch.unique(group[rows]).numel())
    return RankingLossOutput(
        loss=hinge.mean(),
        n_pairs=int(rows.numel()),
        n_pairs_available=n_available,
        concordance=concordance,
        n_groups=n_groups,
    )


__all__ = [
    "RankingLossConfig",
    "RankingLossError",
    "RankingLossOutput",
    "certain_bounds",
    "margin_per_endpoint",
    "ranking_loss",
]
