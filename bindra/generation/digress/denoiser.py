"""Conditional denoiser over noisy graph tensors.

Predicts the *clean* atom and bond categories from a corrupted graph, the
timestep, and the conditions. Target and seed conditioning enter by
cross-attention, never by concatenation into the node features: an appended
condition can be ignored by learning a zero weight on it, and a model that
ignores its target is a generator of plausible molecules rather than of
molecules for that target.

**The clean product is never a condition.** It is the thing being predicted. The
only inputs are the noisy graph, ``t``, and conditions derived from the target,
the seed and the scaffold. :func:`assert_no_product_leakage` states that as a
runtime check rather than a comment, because the failure is invisible in the
loss -- a model handed the answer trains beautifully and generates nothing.

Owed by the specification:
    Priority 3: target and seed conditioning enter via cross-attention; the
    clean product representation must never be a condition.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class DenoiserError(ValueError):
    """The denoiser was built or fed inconsistently."""


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of the diffusion step, ``[B, dim]``."""
    half = dim // 2
    frequencies = torch.exp(
        torch.arange(half, device=t.device, dtype=torch.float32)
        * (-torch.log(torch.tensor(10_000.0, device=t.device)) / max(half - 1, 1))
    )
    angles = t.float().unsqueeze(-1) * frequencies.unsqueeze(0)
    embedded = torch.cat((angles.sin(), angles.cos()), dim=-1)
    if embedded.shape[-1] < dim:  # odd dim
        embedded = torch.cat((embedded, embedded[..., :1]), dim=-1)
    return embedded


@dataclass(frozen=True)
class DenoiserConfig:
    """Widths in, categories out."""

    n_atom_types: int
    n_bond_types: int
    condition_dim: int
    model_dim: int = 256
    n_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.1

    def __post_init__(self) -> None:
        for name in ("n_atom_types", "n_bond_types", "model_dim", "n_layers"):
            if getattr(self, name) < 1:
                raise DenoiserError(f"{name} must be >= 1")
        if self.model_dim % self.n_heads:
            raise DenoiserError(
                f"model_dim {self.model_dim} must divide by n_heads {self.n_heads}"
            )


class _Block(nn.Module):
    """Self-attention over atoms, then cross-attention to the conditions.

    Bond state enters as an additive attention bias rather than as a separate
    token stream. A bond is a statement about a *pair* of atoms, which is what
    an attention logit already is, so biasing the logit keeps the bond and the
    attention it should influence in the same place.
    """

    def __init__(self, config: DenoiserConfig) -> None:
        super().__init__()
        d, h = config.model_dim, config.n_heads
        self.n_heads = h
        self.norm1 = nn.LayerNorm(d)
        self.self_attention = nn.MultiheadAttention(d, h, dropout=config.dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.cross_attention = nn.MultiheadAttention(d, h, dropout=config.dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(d)
        self.feedforward = nn.Sequential(
            nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(2 * d, d)
        )
        self.bond_bias = nn.Linear(config.n_bond_types, h)

    def forward(self, atoms, bonds, conditions, atom_mask, condition_mask):
        # [B, N, N, H] -> [B * H, N, N], the layout MultiheadAttention wants.
        bias = self.bond_bias(bonds).permute(0, 3, 1, 2)
        bias = bias.reshape(-1, bias.shape[-2], bias.shape[-1])
        # Folded into the float bias rather than passed as a separate bool
        # key_padding_mask: torch deprecates mixing a float attn_mask with a
        # bool padding mask, and silently canonicalising one to the other is
        # how a padded atom quietly becomes attendable.
        padded = (~atom_mask).unsqueeze(1).unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        bias = bias.masked_fill(
            padded.reshape(-1, 1, atom_mask.shape[-1]), float("-inf")
        )
        normed = self.norm1(atoms)
        attended, _ = self.self_attention(
            normed, normed, normed, attn_mask=bias, need_weights=False
        )
        # A fully-padded row attends to nothing and comes back NaN. Those rows
        # are discarded downstream, but a NaN propagates through the whole
        # batch's gradients, so it is zeroed at the source.
        attended = torch.nan_to_num(attended) * atom_mask.unsqueeze(-1)
        atoms = atoms + attended
        normed = self.norm2(atoms)
        attended, _ = self.cross_attention(
            normed, conditions, conditions,
            key_padding_mask=None if condition_mask is None else ~condition_mask,
            need_weights=False,
        )
        atoms = atoms + attended
        return atoms + self.feedforward(self.norm3(atoms))


class GraphDenoiser(nn.Module):
    """``(noisy graph, t, conditions) -> logits over clean categories``."""

    def __init__(self, config: DenoiserConfig) -> None:
        super().__init__()
        self.config = config
        d = config.model_dim
        self.atom_in = nn.Linear(config.n_atom_types, d)
        self.bond_in = nn.Linear(config.n_bond_types, d)
        self.time_in = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.condition_in = nn.Linear(config.condition_dim, d)
        self.blocks = nn.ModuleList(_Block(config) for _ in range(config.n_layers))
        self.atom_out = nn.Linear(d, config.n_atom_types)
        # A bond logit is read off the pair of atom states, symmetrised below.
        self.bond_out = nn.Sequential(
            nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, config.n_bond_types)
        )

    def forward(
        self,
        atoms: torch.Tensor,  # [B, N, A] one-hot or distribution
        bonds: torch.Tensor,  # [B, N, N, E]
        t: torch.Tensor,  # [B]
        conditions: torch.Tensor,  # [B, C, condition_dim]
        atom_mask: torch.Tensor,  # [B, N] bool
        condition_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if atoms.shape[-1] != self.config.n_atom_types:
            raise DenoiserError(
                f"expected {self.config.n_atom_types} atom categories, got {atoms.shape[-1]}"
            )
        if bonds.shape[-1] != self.config.n_bond_types:
            raise DenoiserError(
                f"expected {self.config.n_bond_types} bond categories, got {bonds.shape[-1]}"
            )
        hidden = self.atom_in(atoms)
        hidden = hidden + self.time_in(
            timestep_embedding(t, self.config.model_dim)
        ).unsqueeze(1)
        context = self.condition_in(conditions)
        for block in self.blocks:
            hidden = block(hidden, bonds, context, atom_mask, condition_mask)

        atom_logits = self.atom_out(hidden)
        n = hidden.shape[1]
        left = hidden.unsqueeze(2).expand(-1, -1, n, -1)
        right = hidden.unsqueeze(1).expand(-1, n, -1, -1)
        # Symmetrised explicitly: the bond i-j and the bond j-i are one fact,
        # and an asymmetric prediction would make the sampler's two halves
        # disagree about the same bond.
        pair = torch.cat((left, right), dim=-1)
        flipped = torch.cat((right, left), dim=-1)
        bond_logits = 0.5 * (self.bond_out(pair) + self.bond_out(flipped))
        return atom_logits, bond_logits


def assert_no_product_leakage(conditions: dict[str, object]) -> None:
    """Refuse a condition set that contains the answer.

    Conditioning on anything derived from the clean product makes the training
    loss collapse and the sampler useless, and nothing in the loss curve says
    so. Named keys rather than a type check, because the leak arrives as a
    well-meaning extra feature.
    """
    forbidden = (
        "product",
        "product_graph",
        "clean",
        "x_0",
        "target_molecule",
        "answer",
    )
    present = sorted(key for key in conditions if any(bad in key.lower() for bad in forbidden))
    if present:
        raise DenoiserError(
            f"conditions {present} look like the clean product the denoiser is "
            "predicting. Conditioning on it trains a model that cannot generate."
        )


__all__ = [
    "DenoiserConfig",
    "DenoiserError",
    "GraphDenoiser",
    "assert_no_product_leakage",
    "timestep_embedding",
]
