"""Bidirectional protein-ligand cross-attention with retained weights.

Owed by the specification:
    Priority 2: protein-to-ligand and ligand-to-protein are not
    interchangeable transposes; keep both, with documented axes/masks.

## Why the two directions are not transposes

It is tempting to compute one ``[L, T]`` score matrix and read it both ways:
softmax over ``T`` to update residues, softmax over ``L`` to update atoms. That
is wrong twice over.

* **The projections differ.** ``P2L`` asks "which atoms matter *to this
  residue*", so residues supply queries and atoms supply keys and values. ``L2P``
  asks the converse. Sharing one score matrix forces ``W_q`` for residues to also
  serve as ``W_k`` for residues, which is a constraint nobody intends and which
  no amount of training can undo.
* **The normalisation axis differs, so the masks differ.** A ``P2L`` row is a
  distribution over atoms and must be masked by the *ligand* mask; a ``L2P`` row
  is a distribution over residues and must be masked by the *protein* mask. A
  single matrix invites applying one mask to both, which leaks padding into the
  attention -- and padding is not a small share here, because protein lengths in
  this corpus span two orders of magnitude.

So this module keeps two complete attention blocks. They are not symmetric and
the output of one is not the transpose of the other.

## Axes, stated once

Throughout: ``B`` batch, ``L`` padded residue count, ``T`` padded ligand token
count, ``D`` model width, ``H`` heads.

* ``protein_features`` ``[B, L, D]``, ``protein_mask`` ``[B, L]`` bool, True = real.
* ``ligand_features`` ``[B, T, D]``, ``ligand_mask`` ``[B, T]`` bool, True = real.
* ``p2l_attention`` ``[B, H, L, T]`` -- row ``(b, h, l)`` sums to 1 over atoms.
* ``l2p_attention`` ``[B, H, T, L]`` -- row ``(b, h, t)`` sums to 1 over residues.

Masks are **True for real positions**, matching ``ProteinFeatures.residue_mask``
and ``LigandFeatures.token_mask``, and the opposite of ``torch.nn.functional``'s
``key_padding_mask`` convention. Mixing the two silently attends to padding only,
so the convention is asserted in the tests rather than left to recollection.

## The all-padded row

A row whose every key is masked has no valid distribution. Softmax over all
``-inf`` yields NaN, which propagates through the whole batch and shows up far
from here. Such rows are given a zero output and their attention is reported as
zero, which is wrong-but-inert rather than destructive; the rows themselves are
masked out of every loss by the query-side mask anyway. This is reachable
whenever a batch contains a molecule or protein that failed featurization.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class CrossAttentionError(ValueError):
    """Shapes, dtypes or masks were inconsistent with the documented axes."""


@dataclass(frozen=True)
class CrossAttentionConfig:
    model_dim: int = 256
    num_heads: int = 8
    num_layers: int = 2
    dropout: float = 0.1
    attention_dropout: float = 0.0
    #: Width of the per-direction feed-forward block, as a multiple of model_dim.
    feedforward_multiplier: int = 4

    def __post_init__(self) -> None:
        if self.model_dim % self.num_heads:
            raise CrossAttentionError(
                f"model_dim {self.model_dim} is not divisible by num_heads {self.num_heads}"
            )
        if self.num_layers < 1:
            raise CrossAttentionError(f"num_layers must be >= 1, got {self.num_layers}")

    @property
    def head_dim(self) -> int:
        return self.model_dim // self.num_heads

    def to_dict(self) -> dict[str, object]:
        return {
            "model_dim": self.model_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "attention_dropout": self.attention_dropout,
            "feedforward_multiplier": self.feedforward_multiplier,
        }


class DirectedCrossAttention(nn.Module):
    """One direction of cross-attention: queries from ``x``, keys/values from ``y``.

    Instantiated twice per layer with the roles swapped. Deliberately unaware of
    which side is protein and which is ligand, so neither direction can
    accidentally inherit the other's projections.
    """

    def __init__(self, config: CrossAttentionConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.model_dim
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        queries: torch.Tensor,  # [B, Lq, D]
        keys: torch.Tensor,  # [B, Lk, D]
        key_mask: torch.Tensor,  # [B, Lk] bool, True = real
        *,
        need_attention: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Returns updated queries ``[B, Lq, D]`` and attention ``[B, H, Lq, Lk]``.

        With ``need_attention=False`` the map is **not materialised** and ``None``
        is returned in its place. That is the training path, and it is not a
        micro-optimisation: the explicit path builds five ``[B, H, Lq, Lk]``
        tensors per direction (scores, masked, where, softmax, where) and retains
        them for backward. At the batching policy's residue budget that is
        hundreds of MiB per layer per direction, which is why a 5.7M-parameter
        model peaked at 9.89 GiB.

        iBAM needs the maps, so it asks for them. Training does not, so it does
        not pay for them. ``tests/test_fusion.py`` asserts the two paths agree.
        """
        batch, num_queries, _ = queries.shape
        num_keys = keys.shape[1]
        heads, head_dim = self.config.num_heads, self.config.head_dim

        q = self.query(queries).view(batch, num_queries, heads, head_dim).transpose(1, 2)
        k = self.key(keys).view(batch, num_keys, heads, head_dim).transpose(1, 2)
        v = self.value(keys).view(batch, num_keys, heads, head_dim).transpose(1, 2)

        # [B, 1, 1, Lk] broadcasts over heads and queries: this is a key-side mask.
        allowed = key_mask[:, None, None, :]
        # A row with no real keys has no valid distribution. Both paths below give
        # it a zero output; the mask is widened first so neither softmaxes over
        # all -inf, which is NaN and would poison the whole batch.
        has_keys = key_mask.any(dim=1)  # [B]
        usable = allowed | ~has_keys[:, None, None, None]

        weights: torch.Tensor | None = None
        if need_attention:
            scores = (q @ k.transpose(-2, -1)) / (head_dim**0.5)
            scores = scores.masked_fill(~usable, float("-inf"))
            weights = scores.softmax(dim=-1)
            weights = torch.where(
                has_keys[:, None, None, None] & allowed, weights, torch.zeros_like(weights)
            )
            context = self.attention_dropout(weights) @ v
        else:
            context = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=usable,
                dropout_p=self.config.attention_dropout if self.training else 0.0,
            )
            # Rows with no real keys attended uniformly over padding above; their
            # output carries no information and is zeroed, matching the other path.
            context = context * has_keys[:, None, None, None]

        context = context.transpose(1, 2).reshape(batch, num_queries, heads * head_dim)
        updated = self.norm(queries + self.dropout(self.output(context)))
        return updated, weights


class CrossAttentionLayer(nn.Module):
    """Both directions plus a feed-forward block on each side.

    The two directions read the *same* inputs and are applied in parallel, not in
    sequence. Sequential application would make the second direction attend to
    features the first had already rewritten, which makes the returned attention
    maps uninterpretable: ``l2p_attention`` would describe a protein that no
    longer exists by the time the layer finishes.
    """

    def __init__(self, config: CrossAttentionConfig) -> None:
        super().__init__()
        self.protein_to_ligand = DirectedCrossAttention(config)
        self.ligand_to_protein = DirectedCrossAttention(config)
        self.protein_feedforward = _feedforward(config)
        self.ligand_feedforward = _feedforward(config)
        self.protein_norm = nn.LayerNorm(config.model_dim)
        self.ligand_norm = nn.LayerNorm(config.model_dim)

    def forward(
        self,
        protein: torch.Tensor,
        ligand: torch.Tensor,
        protein_mask: torch.Tensor,
        ligand_mask: torch.Tensor,
        *,
        need_attention: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        # Both read the inputs; neither sees the other's output.
        protein_updated, p2l = self.protein_to_ligand(
            protein, ligand, ligand_mask, need_attention=need_attention
        )
        ligand_updated, l2p = self.ligand_to_protein(
            ligand, protein, protein_mask, need_attention=need_attention
        )

        protein_out = self.protein_norm(
            protein_updated + self.protein_feedforward(protein_updated)
        )
        ligand_out = self.ligand_norm(
            ligand_updated + self.ligand_feedforward(ligand_updated)
        )
        # Padded query positions carry no meaning; zero them so nothing downstream
        # has to remember to re-mask before pooling.
        protein_out = protein_out * protein_mask.unsqueeze(-1)
        ligand_out = ligand_out * ligand_mask.unsqueeze(-1)
        return protein_out, ligand_out, p2l, l2p


def _feedforward(config: CrossAttentionConfig) -> nn.Module:
    inner = config.model_dim * config.feedforward_multiplier
    return nn.Sequential(
        nn.Linear(config.model_dim, inner),
        nn.GELU(),
        nn.Dropout(config.dropout),
        nn.Linear(inner, config.model_dim),
        nn.Dropout(config.dropout),
    )


@dataclass(frozen=True)
class FusionOutput:
    """Fused streams plus the attention from every layer, retained for iBAM.

    Attention is kept **per layer and per head**, unaveraged. Averaging before
    export is lossy in exactly the dimension iBAM inspects, and once averaged the
    information cannot be recovered from a checkpoint.
    """

    protein: torch.Tensor  # [B, L, D]
    ligand: torch.Tensor  # [B, T, D]
    protein_mask: torch.Tensor  # [B, L] bool
    ligand_mask: torch.Tensor  # [B, T] bool
    p2l_attention: tuple[torch.Tensor, ...]  # each [B, H, L, T]
    l2p_attention: tuple[torch.Tensor, ...]  # each [B, H, T, L]

    @property
    def num_layers(self) -> int:
        return len(self.p2l_attention)

    def contact_map(self, *, layer: int = -1, reduction: str = "mean") -> torch.Tensor:
        """A ``[B, L, T]`` residue-by-atom view, explicitly a *derived aggregation*.

        Combines the two directions, which point different ways, so the result is
        a symmetrised summary and not an attention distribution: its rows do not
        sum to 1. Anything presented as a contact prediction must say that this
        is what it is.
        """
        p2l = self.p2l_attention[layer].mean(dim=1)  # [B, L, T]
        l2p = self.l2p_attention[layer].mean(dim=1).transpose(1, 2)  # [B, L, T]
        if reduction == "mean":
            combined = 0.5 * (p2l + l2p)
        elif reduction == "min":
            # Agreement of both directions: stricter, and the honest choice when
            # the map is used to claim a contact.
            combined = torch.minimum(p2l, l2p)
        elif reduction == "p2l":
            combined = p2l
        elif reduction == "l2p":
            combined = l2p
        else:
            raise CrossAttentionError(f"unknown reduction {reduction!r}")
        pair_mask = self.protein_mask.unsqueeze(-1) & self.ligand_mask.unsqueeze(1)
        return combined * pair_mask


class CrossAttentionStack(nn.Module):
    """``num_layers`` bidirectional layers, retaining all attention maps."""

    def __init__(self, config: CrossAttentionConfig | None = None) -> None:
        super().__init__()
        self.config = config or CrossAttentionConfig()
        self.layers = nn.ModuleList(
            CrossAttentionLayer(self.config) for _ in range(self.config.num_layers)
        )

    def forward(
        self,
        protein: torch.Tensor,
        ligand: torch.Tensor,
        protein_mask: torch.Tensor,
        ligand_mask: torch.Tensor,
        *,
        need_attention: bool = True,
    ) -> FusionOutput:
        """``need_attention=False`` skips materialising the maps. See
        :meth:`DirectedCrossAttention.forward` for why that matters."""
        _validate(protein, ligand, protein_mask, ligand_mask, self.config.model_dim)
        p2l_maps: list[torch.Tensor] = []
        l2p_maps: list[torch.Tensor] = []
        for layer in self.layers:
            protein, ligand, p2l, l2p = layer(
                protein, ligand, protein_mask, ligand_mask, need_attention=need_attention
            )
            if p2l is not None:
                p2l_maps.append(p2l)
            if l2p is not None:
                l2p_maps.append(l2p)
        return FusionOutput(
            protein=protein,
            ligand=ligand,
            protein_mask=protein_mask,
            ligand_mask=ligand_mask,
            p2l_attention=tuple(p2l_maps),
            l2p_attention=tuple(l2p_maps),
        )


def _validate(
    protein: torch.Tensor,
    ligand: torch.Tensor,
    protein_mask: torch.Tensor,
    ligand_mask: torch.Tensor,
    model_dim: int,
) -> None:
    """Fail loudly on the mistakes that otherwise train quietly."""
    if protein.ndim != 3 or ligand.ndim != 3:
        raise CrossAttentionError(
            f"expected [B, L, D] and [B, T, D], got {tuple(protein.shape)} and "
            f"{tuple(ligand.shape)}"
        )
    if protein.shape[0] != ligand.shape[0]:
        raise CrossAttentionError(
            f"batch mismatch: protein {protein.shape[0]} vs ligand {ligand.shape[0]}"
        )
    for name, tensor in (("protein", protein), ("ligand", ligand)):
        if tensor.shape[-1] != model_dim:
            raise CrossAttentionError(
                f"{name} width {tensor.shape[-1]} != model_dim {model_dim}; "
                "features must pass through SharedAdapters first"
            )
    for name, mask, tensor in (
        ("protein_mask", protein_mask, protein),
        ("ligand_mask", ligand_mask, ligand),
    ):
        if mask.dtype is not torch.bool:
            raise CrossAttentionError(
                f"{name} must be bool with True = real position, got {mask.dtype}"
            )
        if mask.shape != tensor.shape[:2]:
            raise CrossAttentionError(
                f"{name} must be {tensor.shape[:2]}, got {tuple(mask.shape)}"
            )


__all__ = [
    "CrossAttentionConfig",
    "CrossAttentionError",
    "CrossAttentionLayer",
    "CrossAttentionStack",
    "DirectedCrossAttention",
    "FusionOutput",
]
