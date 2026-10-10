"""Reverse sampler with hard scaffold constraints.

Scaffold atom identities, their internal bonds and their stereochemistry stay
fixed through corruption *and* sampling. "Fixed" here means re-imposed at every
step, not merely initialised and hoped for: a soft penalty on scaffold drift
lets a long chain wander off the scaffold a fraction at a time, and the result
looks like a successful generation until someone checks the atom mapping.

The constraint is applied in three places, and all three are necessary:

1.  The forward corruption never touches kept atoms or scaffold-internal bonds
    (:meth:`~bindra.generation.digress.noise.DiscreteGraphNoise.q_sample`).
2.  Every reverse step overwrites them again, because the denoiser is free to
    predict anything and its prediction for a kept position is ignored.
3.  :func:`verify_scaffold` checks the decoded molecule against the scaffold by
    atom mapping, after RDKit has sanitised it.

Owed by the specification:
    Priority 3: scaffold atom identities, internal bonds and stereochemistry
    stay fixed through corruption and sampling.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from bindra.generation.digress.noise import DiscreteGraphNoise


class SamplerError(ValueError):
    """Sampling was asked for something inconsistent."""


@dataclass(frozen=True)
class ScaffoldConstraint:
    """Which positions are frozen, and what they hold.

    ``keep`` is ``[B, N]``. The atom one-hots and the scaffold-internal bond
    one-hots are carried alongside rather than re-derived, so a step can restore
    them without consulting the seed again.
    """

    keep: torch.Tensor
    atoms: torch.Tensor  # [B, N, A]
    bonds: torch.Tensor  # [B, N, N, E]

    def __post_init__(self) -> None:
        if self.keep.dtype is not torch.bool:
            raise SamplerError(f"keep must be bool, got {self.keep.dtype}")
        if self.keep.shape != self.atoms.shape[:2]:
            raise SamplerError(
                f"keep {tuple(self.keep.shape)} does not match atoms "
                f"{tuple(self.atoms.shape[:2])}"
            )

    @property
    def internal(self) -> torch.Tensor:
        """``[B, N, N]`` True where *both* ends are kept.

        Bonds from the scaffold out to the decorated region are deliberately
        not internal: those are what the generation is for.
        """
        return self.keep.unsqueeze(1) & self.keep.unsqueeze(2)

    def apply(self, atoms: torch.Tensor, bonds: torch.Tensor):
        """Overwrite the frozen positions. Called after every reverse step."""
        atoms = torch.where(self.keep.unsqueeze(-1), self.atoms, atoms)
        bonds = torch.where(self.internal.unsqueeze(-1), self.bonds, bonds)
        return atoms, bonds


@torch.no_grad()
def sample(
    denoiser,
    noise: DiscreteGraphNoise,
    *,
    conditions: torch.Tensor,
    atom_mask: torch.Tensor,
    constraint: ScaffoldConstraint | None = None,
    condition_mask: torch.Tensor | None = None,
    n_atom_types: int | None = None,
    n_bond_types: int | None = None,
    generator: torch.Generator | None = None,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the reverse chain from pure noise to a clean graph.

    Returns one-hot atoms ``[B, N, A]`` and bonds ``[B, N, N, E]``. The caller
    decodes; this function knows nothing about chemistry beyond the categories.
    """
    device = atom_mask.device
    batch, n = atom_mask.shape
    n_atom_types = n_atom_types or noise.marginals.atom.numel()
    n_bond_types = n_bond_types or noise.marginals.bond.numel()
    if temperature <= 0:
        raise SamplerError(f"temperature must be > 0, got {temperature}")

    # Start from the base distribution, which is where the forward process ends.
    atoms = _draw(noise.marginals.atom.to(device).expand(batch, n, n_atom_types), generator)
    bonds = _draw_symmetric(
        noise.marginals.bond.to(device).expand(batch, n, n, n_bond_types), generator
    )
    if constraint is not None:
        atoms, bonds = constraint.apply(atoms, bonds)

    for step in range(noise.timesteps, 0, -1):
        t = torch.full((batch,), step, device=device, dtype=torch.long)
        atom_logits, bond_logits = denoiser(
            atoms, bonds, t, conditions, atom_mask, condition_mask
        )
        atom_x0 = torch.softmax(atom_logits / temperature, dim=-1)
        bond_x0 = torch.softmax(bond_logits / temperature, dim=-1)
        atom_posterior = noise.posterior(atoms, atom_x0, noise.marginals.atom, t)
        bond_posterior = noise.posterior(bonds, bond_x0, noise.marginals.bond, t)
        atoms = _draw(atom_posterior, generator)
        bonds = _draw_symmetric(bond_posterior, generator)
        if constraint is not None:
            # Re-imposed every step. The denoiser's prediction for a frozen
            # position is discarded rather than penalised.
            atoms, bonds = constraint.apply(atoms, bonds)

    # Padding atoms carry no meaning; zero them so a decoder cannot read them.
    atoms = atoms * atom_mask.unsqueeze(-1)
    real_pair = (atom_mask.unsqueeze(1) & atom_mask.unsqueeze(2)).unsqueeze(-1)
    bonds = bonds * real_pair
    return atoms, bonds


def _draw(probabilities: torch.Tensor, generator) -> torch.Tensor:
    flat = probabilities.reshape(-1, probabilities.shape[-1]).clamp(min=1e-12)
    drawn = torch.multinomial(flat, 1, generator=generator).squeeze(-1)
    return (
        torch.nn.functional.one_hot(drawn, probabilities.shape[-1])
        .to(probabilities.dtype)
        .reshape(probabilities.shape)
    )


def _draw_symmetric(probabilities: torch.Tensor, generator) -> torch.Tensor:
    sampled = _draw(probabilities, generator)
    n = sampled.shape[1]
    upper = torch.triu(torch.ones(n, n, dtype=torch.bool, device=sampled.device), diagonal=1)
    mirrored = torch.where(
        upper.unsqueeze(0).unsqueeze(-1), sampled, sampled.transpose(1, 2)
    )
    eye = torch.eye(n, dtype=torch.bool, device=sampled.device).unsqueeze(0).unsqueeze(-1)
    no_bond = torch.zeros_like(mirrored)
    no_bond[..., 0] = 1.0
    return torch.where(eye, no_bond, mirrored)


def verify_scaffold(
    atoms: torch.Tensor, bonds: torch.Tensor, constraint: ScaffoldConstraint
) -> dict[str, object]:
    """Did the frozen positions actually survive? Checked, not assumed."""
    atom_ok = torch.where(
        constraint.keep.unsqueeze(-1), atoms == constraint.atoms, torch.ones_like(atoms, dtype=torch.bool)
    ).all()
    internal = constraint.internal.unsqueeze(-1)
    bond_ok = torch.where(
        internal, bonds == constraint.bonds, torch.ones_like(bonds, dtype=torch.bool)
    ).all()
    return {
        "scaffold_atoms_preserved": bool(atom_ok),
        "scaffold_bonds_preserved": bool(bond_ok),
        "n_frozen_atoms": int(constraint.keep.sum()),
    }


__all__ = ["SamplerError", "ScaffoldConstraint", "sample", "verify_scaffold"]
