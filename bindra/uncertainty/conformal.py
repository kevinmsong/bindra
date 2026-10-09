"""Split-conformal calibration, endpoint-specific.

Owed by the specification:
    Priority 4: return an insufficient-calibration status rather than a
    fabricated finite interval when the order statistic is unsupported.

Split conformal on exact labels, per the specification's own formulation::

    score_i = abs(y_i - ensemble_mean_i)
    k       = ceil((n_calibration + 1) * (1 - alpha))
    q       = kth smallest calibration score
    interval= [mean - q, mean + q]

**The order statistic can fail to exist, and that is the point of this
module.** When ``k > n`` -- few calibration points, or a tight alpha -- there
is no kth smallest score. The tempting move is to fall back to the largest
score, or to a Gaussian multiple of the spread, and return a finite interval
anyway. That interval would look exactly like a valid one and carry none of
the guarantee. :class:`InsufficientCalibration` is returned instead, and it is
a distinct type rather than a wide interval so that a caller cannot plot it by
accident.

Calibrated **per endpoint**, because the noise floors differ by a factor of
1.5 (pKd 0.945, pKi 0.641, pIC50 0.671) and one pooled quantile would be too
wide for IC50 and too narrow for Kd.

## What the guarantee rests on

Exchangeability between the calibration set and whatever it is applied to.
That assumption is doing real work here and it is not safe everywhere: the
corpus has correlated assays, 41.6% patent-derived records, and a Kish
effective sample size of 572 target units against 10,474 observed targets. On
a novel target -- the DIP2A case the specification names -- coverage is not
guaranteed and must not be claimed. :func:`coverage` exists so the assumption
can be checked rather than asserted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_SPEC_CLAUSE = (
    "Priority 4: return an insufficient-calibration status rather than a fabricated "
    "finite interval when the order statistic is unsupported."
)

#: 90% intervals, the specification's starting point.
DEFAULT_ALPHA = 0.10


class ConformalError(ValueError):
    """The calibration inputs are not usable."""


@dataclass(frozen=True)
class InsufficientCalibration:
    """No valid quantile exists. Deliberately not an interval.

    A separate type, not a very wide interval, so that code which plots or
    thresholds on a width cannot consume this without noticing.
    """

    endpoint: str
    n_calibration: int
    alpha: float
    required: int

    @property
    def reason(self) -> str:
        return (
            f"{self.endpoint}: need the {self.required}th smallest of "
            f"{self.n_calibration} calibration scores for alpha={self.alpha}, "
            f"which does not exist. At least {self.required} exact calibration "
            "labels are required for this endpoint."
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "endpoint": self.endpoint,
            "status": "insufficient_calibration",
            "n_calibration": self.n_calibration,
            "alpha": self.alpha,
            "required": self.required,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ConformalQuantile:
    """A calibrated half-width in p-units for one endpoint."""

    endpoint: str
    q: float
    alpha: float
    n_calibration: int
    #: Index used, 1-based, as the specification writes it.
    k: int

    def interval(self, mean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``[mean - q, mean + q]``, in p-units."""
        mean = np.asarray(mean, dtype=np.float64)
        return mean - self.q, mean + self.q

    def to_nanomolar(self, mean: float) -> tuple[float, float]:
        """A p-unit interval as nM, with the endpoints reversed.

        The specification calls this out because the inverse logarithm is
        order-reversing: the *upper* p-value is the *lower* concentration.
        Converting each end independently and keeping the order produces an
        interval whose bounds are the wrong way round.
        """
        low_p, high_p = mean - self.q, mean + self.q
        return 10.0 ** (9.0 - high_p), 10.0 ** (9.0 - low_p)

    def to_dict(self) -> dict[str, object]:
        return {
            "endpoint": self.endpoint,
            "status": "calibrated",
            "q": self.q,
            "alpha": self.alpha,
            "n_calibration": self.n_calibration,
            "k": self.k,
            "width_p_units": 2 * self.q,
        }


def required_k(n_calibration: int, alpha: float = DEFAULT_ALPHA) -> int:
    """``ceil((n + 1)(1 - alpha))``, the specification's order statistic."""
    if not 0.0 < alpha < 1.0:
        raise ConformalError(f"alpha must be in (0, 1), got {alpha}")
    return math.ceil((n_calibration + 1) * (1.0 - alpha))


def calibrate(
    truth: np.ndarray,
    predicted: np.ndarray,
    *,
    endpoint: str,
    alpha: float = DEFAULT_ALPHA,
) -> ConformalQuantile | InsufficientCalibration:
    """Split-conformal quantile for one endpoint, or the insufficient status."""
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if truth.shape != predicted.shape:
        raise ConformalError(
            f"{endpoint}: {truth.size} labels against {predicted.size} predictions"
        )

    scores = np.abs(truth - predicted)
    n = int(scores.size)
    k = required_k(n, alpha)
    if n == 0 or k > n:
        return InsufficientCalibration(
            endpoint=endpoint, n_calibration=n, alpha=alpha, required=k
        )
    return ConformalQuantile(
        endpoint=endpoint,
        q=float(np.sort(scores)[k - 1]),
        alpha=alpha,
        n_calibration=n,
        k=k,
    )


def coverage(
    truth: np.ndarray, predicted: np.ndarray, quantile: ConformalQuantile
) -> float:
    """Empirical share of held-out labels inside the interval.

    Reported rather than assumed. Conformal's guarantee is marginal and rests
    on exchangeability; measuring coverage on a held-out split is how that
    assumption gets checked for a given regime.
    """
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if truth.size == 0:
        return float("nan")
    inside = np.abs(truth - predicted) <= quantile.q
    return float(inside.mean())


__all__ = [
    "DEFAULT_ALPHA",
    "ConformalError",
    "ConformalQuantile",
    "InsufficientCalibration",
    "calibrate",
    "coverage",
    "required_k",
]
