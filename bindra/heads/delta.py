"""Analog delta prediction: difference baseline and antisymmetric head.

Two ways to predict how much more potent analog B is than analog A:

**The difference baseline** subtracts the two single-molecule predictions. It is
free, it is already antisymmetric and already zero for A=A, and it is what the
dedicated head has to beat to justify existing. Reporting the head without it is
how a delta model gets credit for arithmetic the affinity head already did.

**The dedicated head** reads both pair representations. Two structural
properties are enforced by construction rather than learned, because a model
that predicts a nonzero change between a molecule and itself is wrong in a way
no amount of training data fixes:

``f(a, a) = 0``       built from an antisymmetrised difference of features
``f(a, b) = -f(b, a)``  the same construction gives this for free

Owed by the specification:
    Upgrade A: enforce zero change for A=A and antisymmetry under reversal;
    the dedicated head is an ablation, not the default.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from bindra.heads.affinity import ENDPOINT_ORDER


class DeltaHeadError(ValueError):
    """The delta head was built or fed inconsistently."""


@dataclass(frozen=True)
class DeltaHeadConfig:
    """Widths, and which mode is the default."""

    pair_dim: int
    hidden_dim: int = 256
    dropout: float = 0.1
    n_endpoints: int = len(ENDPOINT_ORDER)
    #: The baseline is the default on purpose. The head is the ablation.
    use_dedicated_head: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "pair_dim": self.pair_dim,
            "hidden_dim": self.hidden_dim,
            "n_endpoints": self.n_endpoints,
            "use_dedicated_head": self.use_dedicated_head,
            "default": "difference_baseline" if not self.use_dedicated_head else "dedicated",
        }


def difference_baseline(
    prediction_a: torch.Tensor, prediction_b: torch.Tensor
) -> torch.Tensor:
    """``y_b - y_a``. Exactly antisymmetric and exactly zero for a=b.

    The number the dedicated head must beat.
    """
    return prediction_b - prediction_a


class DeltaHead(nn.Module):
    """Antisymmetric by construction, not by regularisation.

    The input is ``(z_b - z_a)`` concatenated with the symmetric part
    ``(z_a + z_b) / 2``, and the symmetric part is passed through a **gate** on
    the antisymmetric signal rather than added to it. Writing it this way means:

    - ``a == b`` gives a zero difference, so the gated product is exactly zero.
    - swapping ``a`` and ``b`` flips the difference and leaves the sum alone,
      so the output flips sign exactly.

    Both hold at every parameter value, so neither can be trained away.
    """

    def __init__(self, config: DeltaHeadConfig) -> None:
        super().__init__()
        self.config = config
        d, h = config.pair_dim, config.hidden_dim
        self.difference = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, h), nn.GELU(), nn.Dropout(config.dropout)
        )
        # The context path has no bias: a bias would survive a zero difference
        # and break f(a, a) = 0.
        self.context = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, h, bias=False), nn.GELU()
        )
        self.out = nn.Linear(h, config.n_endpoints, bias=False)

    def forward(self, pair_a: torch.Tensor, pair_b: torch.Tensor) -> torch.Tensor:
        if pair_a.shape != pair_b.shape:
            raise DeltaHeadError(
                f"pair shapes differ: {tuple(pair_a.shape)} vs {tuple(pair_b.shape)}"
            )
        antisymmetric = self.difference(pair_b - pair_a) - self.difference(pair_a - pair_b)
        gate = self.context((pair_a + pair_b) * 0.5)
        return self.out(antisymmetric * gate)


def assert_delta_properties(head: DeltaHead, pair_dim: int, *, atol: float = 1e-5) -> None:
    """Check both structural properties numerically. Cheap; run it."""
    head = head.eval()
    torch.manual_seed(0)
    a, b = torch.randn(8, pair_dim), torch.randn(8, pair_dim)
    with torch.no_grad():
        same = head(a, a)
        forward, reverse = head(a, b), head(b, a)
    if not torch.allclose(same, torch.zeros_like(same), atol=atol):
        raise DeltaHeadError(
            f"f(a, a) is not zero (max {same.abs().max():.3e}); the head predicts "
            "a potency change between a molecule and itself"
        )
    if not torch.allclose(forward, -reverse, atol=atol):
        raise DeltaHeadError(
            f"f(a, b) != -f(b, a) (max {(forward + reverse).abs().max():.3e}); "
            "the predicted change depends on which analog was listed first"
        )


__all__ = [
    "DeltaHead",
    "DeltaHeadConfig",
    "DeltaHeadError",
    "assert_delta_properties",
    "difference_baseline",
]
