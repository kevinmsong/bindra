"""Huber for exact labels plus one-sided penalties for bounds.

Owed by the specification:
    Priority 1: do not train against a bound as if it were exact; report
    exact-label metrics separately from bound violations.

## What a censored row actually says

The curated corpus stores censoring in **p-units**, already resolved: a row with
``relation_p == "<"`` carries an ``upper_bound`` and means "the true pX is at most
this", and ``">"`` carries a ``lower_bound`` meaning "at least this". Exactly one
bound is set per censored row, never both and never neither (verified on the
corpus). The reversal that happens under the negative log -- "IC50 greater than
10 micromolar" becoming an *upper* bound on pIC50 -- is handled upstream, so this
module can take the bounds at face value. It still checks, because getting the
direction backwards would train the model to predict the opposite of the truth
while the loss curve looked normal.

In this shard, 6,902 of 41,554 rows are censored (16.6%): 6,065 right-censored in
concentration (upper bound in pX, i.e. weak or inactive) and 837 left-censored
(lower bound, i.e. very potent). Treating those 16.6% as exact point labels would
pull predictions toward the detection limits of the assays rather than toward
affinities, and the weak end is where the fishing and selectivity panels live.

## The penalty

A censored row constrains the prediction to a half-line, and a prediction anywhere
inside it is **fully consistent with the measurement**. So the penalty is zero
inside and grows only outside:

* upper bound ``b``: penalty on ``max(0, pred - b)``
* lower bound ``b``: penalty on ``max(0, b - pred)``

The violation is then passed through the same Huber as exact residuals, so a small
violation is quadratic and a large one linear. Using a bare squared hinge instead
would let one grossly mispredicted inactive dominate a batch.

``censored_mode="tobit"`` offers the statistically principled alternative: the
censored-Gaussian log-likelihood ``-log Phi((b - pred) / sigma)``, using the
*measured* sigma_rep per endpoint. It is not the default, and the reason is
recorded rather than hidden -- ``ErrorModel``'s own docstring establishes that
sigma_rep is a **lower bound** on true measurement error, because it is pooled
within publication. A Tobit term built on it is therefore over-confident by an
unknown factor, which turns a modelling assumption into a silent one. The hinge
assumes no distribution at all.

## Tier B

The curated IC50 tiering sends Tier B records to **auxiliary objectives only**.
That is enforced here structurally: Tier B rows are excluded from the primary term
and accumulated into a separately weighted, separately reported auxiliary term.
They cannot reach the primary loss by any weighting, only by being relabelled
upstream.

## Reporting

Exact-label error and bound violations are different quantities and are never
summed into one reported number. :class:`AffinityLossOutput` carries exact RMSE
and MAE, the fraction of censored rows whose prediction falls outside the allowed
half-line, and the mean violation magnitude -- separately, as the specification
requires.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch

from bindra.config.profiles import ErrorModel
from bindra.data.endpoints import Endpoint
from bindra.heads.affinity import ENDPOINT_ORDER, AffinityPrediction


class AffinityLossError(ValueError):
    """Targets were inconsistent, or censoring was described impossibly."""


@dataclass(frozen=True)
class AffinityTargets:
    """One batch of curated labels. Bounds are in p-units.

    ``is_primary`` is the curation layer's verdict (``primary_regression_eligible``
    and not IC50 Tier B). Rows with ``is_primary=False`` are *not* discarded -- they
    carry real information -- but they only ever reach the auxiliary term.
    """

    endpoint_index: torch.Tensor  # [B] long, into ENDPOINT_ORDER
    p_value: torch.Tensor  # [B] float: the exact value, or the bound
    is_exact: torch.Tensor  # [B] bool
    lower_bound: torch.Tensor  # [B] float, NaN where absent
    upper_bound: torch.Tensor  # [B] float, NaN where absent
    is_primary: torch.Tensor  # [B] bool

    def __post_init__(self) -> None:
        size = self.endpoint_index.shape
        if len(size) != 1:
            raise AffinityLossError(
                f"endpoint_index must be [B], got {tuple(size)}"
            )
        for name in ("p_value", "is_exact", "lower_bound", "upper_bound", "is_primary"):
            tensor = getattr(self, name)
            if tensor.shape != size:
                raise AffinityLossError(
                    f"{name} must be {tuple(size)}, got {tuple(tensor.shape)}"
                )
        for name in ("is_exact", "is_primary"):
            if getattr(self, name).dtype is not torch.bool:
                raise AffinityLossError(
                    f"{name} must be bool, got {getattr(self, name).dtype}"
                )

        censored = ~self.is_exact
        has_lower = ~torch.isnan(self.lower_bound)
        has_upper = ~torch.isnan(self.upper_bound)
        # The checks that would catch a bound written into the wrong field, which
        # trains the model toward the opposite of the measurement.
        if bool((censored & ~(has_lower | has_upper)).any()):
            raise AffinityLossError(
                "a censored row has neither a lower nor an upper bound; it states "
                "nothing and must be quarantined upstream, not trained on"
            )
        if bool((censored & has_lower & has_upper).any()):
            raise AffinityLossError(
                "a censored row has both bounds; the corpus stores exactly one "
                "per row, so this is an interval that needs its own handling"
            )
        if bool((self.is_exact & (has_lower | has_upper)).any()):
            raise AffinityLossError(
                "an exact row carries a censoring bound; exact and censored are "
                "mutually exclusive in the curated corpus"
            )

    @property
    def batch_size(self) -> int:
        return int(self.endpoint_index.shape[0])

    @classmethod
    def from_frame(cls, frame, *, device: torch.device | str | None = None) -> "AffinityTargets":
        """Build from a curated observations frame.

        Expects the corpus's own column names. ``is_primary`` combines
        ``primary_regression_eligible`` with the Tier B exclusion, so a caller
        cannot forget the second half.
        """
        from bindra.heads.affinity import endpoint_indices

        import numpy as np

        def column(name: str) -> torch.Tensor:
            return torch.from_numpy(
                np.array(frame[name].to_numpy(), dtype="float32", copy=True)
            )

        # np.asarray on a pandas column can hand back a read-only view, which
        # torch.as_tensor adopts and then warns about; copy once instead.
        def flag(values) -> torch.Tensor:
            return torch.from_numpy(np.array(values, dtype=bool, copy=True))

        eligible = flag(frame["primary_regression_eligible"].to_numpy())
        tier_b = flag((frame["ic50_tier"].astype(str) == "B").to_numpy())
        targets = cls(
            endpoint_index=endpoint_indices(list(frame["endpoint"].astype(str))),
            p_value=column("p_value"),
            is_exact=flag(frame["is_exact"].to_numpy()),
            lower_bound=column("lower_bound"),
            upper_bound=column("upper_bound"),
            is_primary=eligible & ~tier_b,
        )
        if device is None:
            return targets
        return targets.to(device)

    def to(self, device: torch.device | str) -> "AffinityTargets":
        return AffinityTargets(
            endpoint_index=self.endpoint_index.to(device),
            p_value=self.p_value.to(device),
            is_exact=self.is_exact.to(device),
            lower_bound=self.lower_bound.to(device),
            upper_bound=self.upper_bound.to(device),
            is_primary=self.is_primary.to(device),
        )


@dataclass(frozen=True)
class AffinityLossConfig:
    """Every weight and threshold, declared.

    ``huber_delta_scale`` multiplies the *measured* sigma_rep per endpoint. At 1.0
    the quadratic region is exactly the replicate noise floor: residuals inside the
    noise are treated as noise, and only residuals the data can distinguish from
    noise are penalised linearly. That is why the delta is per-endpoint -- 0.945
    for pKd against 0.641 for pKi -- rather than a single tuned constant.
    """

    huber_delta_scale: float = 1.0
    censored_weight: float = 1.0
    #: Tier B and other non-primary rows. Below 1.0 on purpose: they are weaker
    #: evidence, and the curation decision was "auxiliary objectives only".
    auxiliary_weight: float = 0.25
    censored_mode: str = "hinge"
    #: Multiplier on sigma_rep when censored_mode="tobit". See the module docstring
    #: for why sigma_rep understates the true error.
    tobit_sigma_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.censored_mode not in ("hinge", "tobit"):
            raise AffinityLossError(
                f"censored_mode must be 'hinge' or 'tobit', got {self.censored_mode!r}"
            )
        for name in ("huber_delta_scale", "tobit_sigma_scale"):
            if getattr(self, name) <= 0:
                raise AffinityLossError(f"{name} must be > 0, got {getattr(self, name)}")
        for name in ("censored_weight", "auxiliary_weight"):
            if getattr(self, name) < 0:
                raise AffinityLossError(f"{name} must be >= 0, got {getattr(self, name)}")

    def to_dict(self) -> dict[str, object]:
        return {
            "huber_delta_scale": self.huber_delta_scale,
            "censored_weight": self.censored_weight,
            "auxiliary_weight": self.auxiliary_weight,
            "censored_mode": self.censored_mode,
            "tobit_sigma_scale": self.tobit_sigma_scale,
        }


@dataclass(frozen=True)
class AffinityLossOutput:
    """The loss plus the diagnostics that must not be folded into it.

    ``exact_rmse`` and ``bound_violation_rate`` answer different questions and are
    reported separately, as the specification requires. A single "accuracy" number
    covering both would let a model trade bound violations for point accuracy
    invisibly.
    """

    total: torch.Tensor
    exact_loss: torch.Tensor
    censored_loss: torch.Tensor
    auxiliary_loss: torch.Tensor
    n_exact: int
    n_censored: int
    n_auxiliary: int
    #: RMSE over exact *primary* rows only, in p-units.
    exact_rmse: torch.Tensor
    exact_mae: torch.Tensor
    #: Fraction of censored primary rows predicted outside their allowed half-line.
    bound_violation_rate: torch.Tensor
    #: Mean violation magnitude over *violating* rows, in p-units; zero if none.
    mean_bound_violation: torch.Tensor
    per_endpoint: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "total": float(self.total),
            "exact_loss": float(self.exact_loss),
            "censored_loss": float(self.censored_loss),
            "auxiliary_loss": float(self.auxiliary_loss),
            "n_exact": self.n_exact,
            "n_censored": self.n_censored,
            "n_auxiliary": self.n_auxiliary,
            "exact_rmse": float(self.exact_rmse),
            "exact_mae": float(self.exact_mae),
            "bound_violation_rate": float(self.bound_violation_rate),
            "mean_bound_violation": float(self.mean_bound_violation),
            "per_endpoint": {k: dict(v) for k, v in self.per_endpoint.items()},
        }


def endpoint_sigma(
    error_model: ErrorModel, *, device: torch.device | str | None = None
) -> torch.Tensor:
    """Measured sigma_rep as a ``[3]`` tensor in :data:`ENDPOINT_ORDER`."""
    return torch.tensor(
        [error_model.sigma(endpoint) for endpoint in ENDPOINT_ORDER],
        dtype=torch.float32,
        device=device,
    )


def _huber(residual: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Elementwise Huber with a per-element delta.

    ``torch.nn.functional.huber_loss`` takes a scalar delta, and the delta here is
    per-endpoint, so it is written out.
    """
    absolute = residual.abs()
    quadratic = 0.5 * residual.pow(2) / delta
    linear = absolute - 0.5 * delta
    return torch.where(absolute <= delta, quadratic, linear)


def censoring_violation(
    prediction: torch.Tensor,
    lower_bound: torch.Tensor,
    upper_bound: torch.Tensor,
) -> torch.Tensor:
    """How far a prediction falls outside the half-line a bound allows.

    Zero inside the allowed region: a prediction consistent with "at most 5.0" is
    consistent, and must not be penalised for being far below 5.0. That asymmetry
    is the whole point of not treating a bound as exact.

    NaN bounds mean "no constraint on this side" and contribute zero, so this is
    safe to call on a whole batch including exact rows.
    """
    has_upper = ~torch.isnan(upper_bound)
    has_lower = ~torch.isnan(lower_bound)
    above = torch.where(
        has_upper,
        (prediction - torch.nan_to_num(upper_bound)).clamp_min(0.0),
        torch.zeros_like(prediction),
    )
    below = torch.where(
        has_lower,
        (torch.nan_to_num(lower_bound) - prediction).clamp_min(0.0),
        torch.zeros_like(prediction),
    )
    # Exactly one bound is set per censored row, so these never both fire.
    return above + below


def _tobit_negative_log_likelihood(
    prediction: torch.Tensor,
    lower_bound: torch.Tensor,
    upper_bound: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """``-log P(true value satisfies the bound | prediction, sigma)``.

    For an upper bound ``b`` this is ``-log Phi((b - pred) / sigma)``. Clamped away
    from zero probability: a prediction many sigma above the bound otherwise gives
    ``-log 0 = inf`` and destroys the batch.
    """
    normal = torch.distributions.Normal(
        torch.zeros((), device=prediction.device), torch.ones((), device=prediction.device)
    )
    has_upper = ~torch.isnan(upper_bound)
    standardized = torch.where(
        has_upper,
        (torch.nan_to_num(upper_bound) - prediction) / sigma,
        (prediction - torch.nan_to_num(lower_bound)) / sigma,
    )
    log_probability = normal.cdf(standardized).clamp_min(1e-12).log()
    return -log_probability


def affinity_loss(
    prediction: AffinityPrediction,
    targets: AffinityTargets,
    *,
    config: AffinityLossConfig | None = None,
    error_model: ErrorModel | None = None,
) -> AffinityLossOutput:
    """Compute the affinity loss, keeping exact and censored evidence distinct.

    Each term is averaged over the rows that *contribute to it*, not over the
    batch. Averaging over the batch would make the exact term shrink whenever a
    batch happened to hold more censored rows, so the effective learning rate on
    exact labels would depend on batch composition.
    """
    config = config or AffinityLossConfig()
    error_model = error_model or ErrorModel()
    values = prediction.observed(targets.endpoint_index)
    sigma = endpoint_sigma(error_model, device=values.device)[targets.endpoint_index]
    delta = config.huber_delta_scale * sigma

    primary = targets.is_primary
    exact = targets.is_exact & primary
    censored = (~targets.is_exact) & primary
    auxiliary = ~primary

    zero = values.new_zeros(())

    # -- exact rows -------------------------------------------------------
    if bool(exact.any()):
        residual = values[exact] - targets.p_value[exact]
        exact_loss = _huber(residual, delta[exact]).mean()
        exact_rmse = residual.pow(2).mean().sqrt()
        exact_mae = residual.abs().mean()
    else:
        exact_loss, exact_rmse, exact_mae = zero, zero, zero

    # -- censored rows ----------------------------------------------------
    violation = censoring_violation(values, targets.lower_bound, targets.upper_bound)
    if bool(censored.any()):
        if config.censored_mode == "hinge":
            censored_loss = _huber(violation[censored], delta[censored]).mean()
        else:
            censored_loss = _tobit_negative_log_likelihood(
                values[censored],
                targets.lower_bound[censored],
                targets.upper_bound[censored],
                config.tobit_sigma_scale * sigma[censored],
            ).mean()
        violating = violation[censored] > 0
        bound_violation_rate = violating.to(values.dtype).mean()
        mean_bound_violation = (
            violation[censored][violating].mean() if bool(violating.any()) else zero
        )
    else:
        censored_loss, bound_violation_rate, mean_bound_violation = zero, zero, zero

    # -- auxiliary rows (Tier B and anything else not primary) ------------
    # Structurally separate: no weighting can route these into the primary term.
    if bool(auxiliary.any()):
        aux_exact = auxiliary & targets.is_exact
        aux_censored = auxiliary & ~targets.is_exact
        parts: list[torch.Tensor] = []
        if bool(aux_exact.any()):
            parts.append(
                _huber(values[aux_exact] - targets.p_value[aux_exact], delta[aux_exact])
            )
        if bool(aux_censored.any()):
            parts.append(_huber(violation[aux_censored], delta[aux_censored]))
        auxiliary_loss = torch.cat(parts).mean() if parts else zero
    else:
        auxiliary_loss = zero

    total = (
        exact_loss
        + config.censored_weight * censored_loss
        + config.auxiliary_weight * auxiliary_loss
    )

    return AffinityLossOutput(
        total=total,
        exact_loss=exact_loss,
        censored_loss=censored_loss,
        auxiliary_loss=auxiliary_loss,
        n_exact=int(exact.sum()),
        n_censored=int(censored.sum()),
        n_auxiliary=int(auxiliary.sum()),
        exact_rmse=exact_rmse,
        exact_mae=exact_mae,
        bound_violation_rate=bound_violation_rate,
        mean_bound_violation=mean_bound_violation,
        per_endpoint=_per_endpoint_report(values, targets, exact, censored, violation),
    )


def _per_endpoint_report(
    values: torch.Tensor,
    targets: AffinityTargets,
    exact: torch.Tensor,
    censored: torch.Tensor,
    violation: torch.Tensor,
) -> dict[str, dict[str, float]]:
    """Per-endpoint diagnostics, because the noise floors differ.

    A pooled RMSE mixes a 0.945-noise endpoint with a 0.641-noise one, so a pooled
    number cannot be compared to any single noise floor.
    """
    # Detached: this is a report, and keeping graph references alive in a dict of
    # floats is how a training loop ends up holding every batch's graph.
    values = values.detach()
    violation = violation.detach()
    report: dict[str, dict[str, float]] = {}
    for column, endpoint in enumerate(ENDPOINT_ORDER):
        selected = targets.endpoint_index == column
        endpoint_exact = selected & exact
        endpoint_censored = selected & censored
        entry: dict[str, float] = {
            "n_exact": int(endpoint_exact.sum()),
            "n_censored": int(endpoint_censored.sum()),
        }
        if bool(endpoint_exact.any()):
            residual = values[endpoint_exact] - targets.p_value[endpoint_exact]
            entry["rmse"] = float(residual.pow(2).mean().sqrt())
            entry["mae"] = float(residual.abs().mean())
        if bool(endpoint_censored.any()):
            entry["bound_violation_rate"] = float(
                (violation[endpoint_censored] > 0).to(values.dtype).mean()
            )
        report[endpoint.value] = entry
    return report


def noise_floor_table(error_model: ErrorModel | None = None) -> dict[str, float]:
    """The RMSE at which a model is at the measured noise floor, not skilful.

    Published next to any RMSE this loss reports. sigma_rep is a *lower* bound on
    measurement error, so these are the most generous possible reading.
    """
    error_model = error_model or ErrorModel()
    return {endpoint.value: error_model.sigma(endpoint) for endpoint in ENDPOINT_ORDER}


__all__ = [
    "AffinityLossConfig",
    "AffinityLossError",
    "AffinityLossOutput",
    "AffinityTargets",
    "affinity_loss",
    "censoring_violation",
    "endpoint_sigma",
    "noise_floor_table",
]
