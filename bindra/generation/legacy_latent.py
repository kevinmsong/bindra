"""Quarantined legacy Gaussian DDPM over atom feature vectors.

**This is not analog generation and must never be reported as it.** It runs a
Gaussian diffusion over continuous atom feature vectors, holds bond structure
fixed, and has no molecular decoder: nothing it produces is a molecule. It is
retained only as a labelled ablation, to show what the discrete process in
:mod:`bindra.generation.digress` is being compared against.

Two things it cannot do, by construction rather than by omission:

- **It cannot change connectivity.** Bonds are fixed input, so every sample has
  the seed's exact graph. "Analogs" from it differ only in a latent vector.
- **It cannot be decoded.** There is no mapping from its continuous output back
  to atom types, so validity, uniqueness and scaffold preservation are all
  undefined for it. :func:`decode` says so rather than returning something.

Calling it requires passing ``acknowledge_ablation_only=True``. That is
deliberate friction: the failure this module guards against is someone reaching
for the nearest thing named "generation" and reporting its loss curve.

Owed by the specification:
    Priority 3: retained only as a labelled ablation. It has no molecular
    decoder and holds bond structure fixed, so it cannot be presented as
    analog generation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


class LegacyLatentError(RuntimeError):
    """The legacy model was asked for something it cannot do."""


#: Stated in one place so a report can quote it verbatim.
LIMITATIONS: tuple[str, ...] = (
    "Gaussian noise over continuous features, not a categorical process",
    "bond structure is held fixed; connectivity never changes",
    "no molecular decoder, so no SMILES and no validity rate",
    "scaffold preservation is trivially 100% and therefore meaningless",
)


@dataclass(frozen=True)
class LegacyLatentConfig:
    feature_dim: int
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02


class LegacyLatentDiffusion:
    """A Gaussian DDPM over ``[B, N, feature_dim]``. Ablation only."""

    def __init__(self, config: LegacyLatentConfig, *, acknowledge_ablation_only: bool = False) -> None:
        if not acknowledge_ablation_only:
            raise LegacyLatentError(
                "LegacyLatentDiffusion is retained as a labelled ablation and "
                "cannot generate molecules. Pass acknowledge_ablation_only=True "
                "if that is understood. For analog generation use "
                "bindra.generation.digress.\nLimitations: " + "; ".join(LIMITATIONS)
            )
        self.config = config
        betas = torch.linspace(config.beta_start, config.beta_end, config.timesteps)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None):
        noise = torch.randn_like(x0) if noise is None else noise
        a = self.alpha_bar.to(x0.device)[t].view(-1, *([1] * (x0.dim() - 1)))
        return a.sqrt() * x0 + (1 - a).sqrt() * noise, noise

    @staticmethod
    def decode(*_args, **_kwargs):
        """There is no decoder. Saying so beats returning something plausible."""
        raise LegacyLatentError(
            "the legacy latent model has no molecular decoder: its output is a "
            "continuous feature tensor, not a molecule. Validity, uniqueness "
            "and novelty are undefined for it."
        )

    @staticmethod
    def describe() -> dict:
        return {
            "status": "quarantined ablation",
            "is_analog_generation": False,
            "limitations": list(LIMITATIONS),
            "use_instead": "bindra.generation.digress",
        }


__all__ = ["LIMITATIONS", "LegacyLatentConfig", "LegacyLatentDiffusion", "LegacyLatentError"]
