"""Categorical forward noising over atom and bond categories.

A real discrete process, not a Gaussian scheduler applied to categorical graphs.
Atom types and bond types are *categories*: there is no sense in which carbon is
between nitrogen and oxygen, so adding Gaussian noise to a one-hot and rounding
is not a corruption of the molecule, it is a corruption of an arbitrary ordering.

The forward process is therefore a Markov chain on categories. At each step a
category either stays put or is resampled from a fixed base distribution, with
transition matrix

    Q_t = alpha_t * I + (1 - alpha_t) * 1 m^T

where ``m`` is the base distribution. Composing these gives the closed form used
for training, so ``q_t`` is a single call rather than ``t`` sequential ones:

    Q_bar_t = alpha_bar_t * I + (1 - alpha_bar_t) * 1 m^T

**The base distribution is the training-set marginal, not uniform.** Uniform
noise over atom types makes the model spend its capacity rediscovering that
molecules are mostly carbon. Starting from the marginal means the fully-noised
state is already an element-plausible graph and the reverse process only has to
learn structure.

Owed by the specification:
    Priority 3: a real discrete process, not a Gaussian scheduler applied to
    categorical graphs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


class NoiseScheduleError(ValueError):
    """The schedule was built or queried inconsistently."""


def cosine_alpha_bar(timesteps: int, *, s: float = 0.008) -> torch.Tensor:
    """``alpha_bar_t`` on a cosine schedule, ``[timesteps + 1]`` from 1 to ~0.

    The cosine shape spends more steps near the clean end than a linear one,
    which is where the hard decisions are: by the time half the bonds are random
    the graph carries little information worth resolving slowly.
    """
    if timesteps < 1:
        raise NoiseScheduleError(f"timesteps must be >= 1, got {timesteps}")
    steps = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps
    values = torch.cos((steps + s) / (1.0 + s) * torch.pi * 0.5) ** 2
    values = values / values[0]
    return values.clamp(min=1e-6, max=1.0).to(torch.float32)


@dataclass(frozen=True)
class CategoricalMarginals:
    """Base distributions the noise decays toward, one per stream."""

    atom: torch.Tensor  # [n_atom_types]
    bond: torch.Tensor  # [n_bond_types]

    def __post_init__(self) -> None:
        for name in ("atom", "bond"):
            vector = getattr(self, name)
            if vector.dim() != 1 or vector.numel() < 2:
                raise NoiseScheduleError(f"{name} marginal must be 1-D with >= 2 entries")
            if not torch.isclose(vector.sum(), torch.ones(()), atol=1e-4):
                raise NoiseScheduleError(
                    f"{name} marginal must sum to 1, got {float(vector.sum()):.6f}"
                )
            if bool((vector < 0).any()):
                raise NoiseScheduleError(f"{name} marginal has a negative entry")

    @classmethod
    def uniform(cls, n_atom_types: int, n_bond_types: int) -> "CategoricalMarginals":
        """Equal mass everywhere. A baseline to ablate the marginal against."""
        return cls(
            atom=torch.full((n_atom_types,), 1.0 / n_atom_types),
            bond=torch.full((n_bond_types,), 1.0 / n_bond_types),
        )

    @classmethod
    def from_counts(
        cls, atom_counts: torch.Tensor, bond_counts: torch.Tensor, *, smoothing: float = 1.0
    ) -> "CategoricalMarginals":
        """Empirical marginals, Laplace-smoothed so no category is impossible.

        A zero-probability category can never be sampled, and a reverse step
        that needs it has no way back -- so an element absent from the training
        split would be unrecoverable rather than merely rare.
        """
        atom = (atom_counts.float() + smoothing)
        bond = (bond_counts.float() + smoothing)
        return cls(atom=atom / atom.sum(), bond=bond / bond.sum())


class DiscreteGraphNoise:
    """The forward process: corrupt a clean graph to step ``t``.

    Call :meth:`q_sample` for training (one jump to any ``t``) and
    :meth:`posterior` for the reverse step the sampler needs.
    """

    def __init__(
        self,
        marginals: CategoricalMarginals,
        *,
        timesteps: int = 500,
        schedule: torch.Tensor | None = None,
    ) -> None:
        self.marginals = marginals
        self.timesteps = timesteps
        self.alpha_bar = cosine_alpha_bar(timesteps) if schedule is None else schedule
        if self.alpha_bar.numel() != timesteps + 1:
            raise NoiseScheduleError(
                f"schedule must hold {timesteps + 1} values, got {self.alpha_bar.numel()}"
            )

    # -- the closed form -------------------------------------------------

    def _q_bar(self, probabilities: torch.Tensor, marginal: torch.Tensor, t: torch.Tensor):
        """``x Q_bar_t`` for a batch of distributions, broadcasting over shape."""
        alpha_bar = self.alpha_bar.to(probabilities.device)[t]
        while alpha_bar.dim() < probabilities.dim():
            alpha_bar = alpha_bar.unsqueeze(-1)
        base = marginal.to(probabilities.device).expand_as(probabilities)
        return alpha_bar * probabilities + (1.0 - alpha_bar) * base

    def q_probabilities(
        self, atom_onehot: torch.Tensor, bond_onehot: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``q(x_t | x_0)`` for atoms and bonds, as category distributions."""
        return (
            self._q_bar(atom_onehot, self.marginals.atom, t),
            self._q_bar(bond_onehot, self.marginals.bond, t),
        )

    def q_sample(
        self,
        atom_onehot: torch.Tensor,  # [B, N, A]
        bond_onehot: torch.Tensor,  # [B, N, N, E]
        t: torch.Tensor,  # [B]
        *,
        generator: torch.Generator | None = None,
        keep: torch.Tensor | None = None,  # [B, N] True = never corrupt
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw ``x_t`` in one jump. Returns one-hot atoms and bonds.

        ``keep`` is the scaffold mask. A kept atom is returned unchanged, and so
        is any bond with a kept atom at *both* ends -- the scaffold's internal
        bonds. Bonds from the scaffold out to the decorated part are corrupted,
        because those are exactly what generation is for.
        """
        atom_probabilities, bond_probabilities = self.q_probabilities(
            atom_onehot, bond_onehot, t
        )
        atoms = _sample_categorical(atom_probabilities, generator)
        bonds = _sample_symmetric(bond_probabilities, generator)
        if keep is not None:
            atoms = torch.where(keep.unsqueeze(-1), atom_onehot, atoms)
            internal = keep.unsqueeze(1) & keep.unsqueeze(2)
            bonds = torch.where(internal.unsqueeze(-1), bond_onehot, bonds)
        return atoms, bonds

    def posterior(
        self,
        x_t: torch.Tensor,
        x_0_probabilities: torch.Tensor,
        marginal: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """``q(x_{t-1} | x_t, x_0_hat)``, up to normalisation.

        The usual discrete-diffusion posterior: the single-step likelihood of
        landing on ``x_t`` from each candidate, times that candidate's
        probability under the predicted clean distribution.
        """
        device = x_t.device
        alpha_bar_t = self.alpha_bar.to(device)[t]
        alpha_bar_prev = self.alpha_bar.to(device)[(t - 1).clamp(min=0)]
        alpha_t = (alpha_bar_t / alpha_bar_prev.clamp(min=1e-8)).clamp(max=1.0)
        while alpha_t.dim() < x_t.dim():
            alpha_t = alpha_t.unsqueeze(-1)
            alpha_bar_prev = alpha_bar_prev.unsqueeze(-1)
        base = marginal.to(device).expand_as(x_t)
        # P(x_t | candidate) for every candidate category.
        likelihood = alpha_t * x_t + (1.0 - alpha_t) * (x_t * base).sum(-1, keepdim=True)
        prior = alpha_bar_prev * x_0_probabilities + (1.0 - alpha_bar_prev) * base
        unnormalised = likelihood * prior
        return unnormalised / unnormalised.sum(-1, keepdim=True).clamp(min=1e-12)


def _sample_categorical(
    probabilities: torch.Tensor, generator: torch.Generator | None
) -> torch.Tensor:
    flat = probabilities.reshape(-1, probabilities.shape[-1]).clamp(min=1e-12)
    drawn = torch.multinomial(flat, 1, generator=generator).squeeze(-1)
    return torch.nn.functional.one_hot(drawn, probabilities.shape[-1]).to(
        probabilities.dtype
    ).reshape(probabilities.shape)


def _sample_symmetric(
    probabilities: torch.Tensor, generator: torch.Generator | None
) -> torch.Tensor:
    """Sample a bond tensor that stays symmetric and has no self-bonds.

    A bond is one fact about a pair, not two. Sampling ``[i, j]`` and ``[j, i]``
    independently produces a graph where a bond exists in one direction only,
    which RDKit cannot decode and which no amount of downstream repair makes
    meaningful. The upper triangle is drawn and mirrored.
    """
    sampled = _sample_categorical(probabilities, generator)
    n = sampled.shape[1]
    upper = torch.triu(torch.ones(n, n, dtype=torch.bool, device=sampled.device), diagonal=1)
    mirrored = torch.where(upper.unsqueeze(0).unsqueeze(-1), sampled, sampled.transpose(1, 2))
    # Diagonal is "no bond", category 0, always.
    eye = torch.eye(n, dtype=torch.bool, device=sampled.device).unsqueeze(0).unsqueeze(-1)
    no_bond = torch.zeros_like(mirrored)
    no_bond[..., 0] = 1.0
    return torch.where(eye, no_bond, mirrored)


__all__ = [
    "CategoricalMarginals",
    "DiscreteGraphNoise",
    "NoiseScheduleError",
    "cosine_alpha_bar",
]
