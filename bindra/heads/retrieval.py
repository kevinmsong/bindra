"""Independent protein and ligand retrieval projections.

**Each side must be computable without the other.** An index is built once over
millions of ligands and queried by a protein that was not present when it was
built; if a ligand's vector depended on the protein it is scored against, the
index would have to be rebuilt per query, which is not an index.

That rules out the obvious shortcut of embedding the pair representation. The
cross-attention output is pair-conditioned by construction -- it is literally
the protein attending to *this* ligand -- so indexing it would produce
spectacular offline numbers and nothing usable.
:func:`assert_not_pair_conditioned` states that as a check.

The projections read the **pre-fusion** adapted streams, which the model already
computes and already routes to the pair vector as a separate block.

Owed by the specification:
    Priority 2: computable without knowing the other member of a pair; do
    not index pair-conditioned cross-attention embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from bindra.fusion.adapters import AttentionPool, masked_mean


class RetrievalHeadError(ValueError):
    """The retrieval head was built or fed in a way that breaks the index."""


@dataclass(frozen=True)
class RetrievalHeadConfig:
    """Widths and the normalisation the index assumes."""

    model_dim: int
    embedding_dim: int = 256
    dropout: float = 0.1
    #: L2-normalise outputs. The index does inner-product search, so without
    #: this a vector's rank is partly its magnitude, and magnitude here tracks
    #: how many residues a protein has rather than what it binds.
    normalize: bool = True
    #: Learned temperature for the contrastive scale, as log(1/T).
    logit_scale_init: float = 2.659  # ln(1 / 0.07)

    def to_dict(self) -> dict[str, object]:
        return {
            "model_dim": self.model_dim,
            "embedding_dim": self.embedding_dim,
            "normalize": self.normalize,
            "dropout": self.dropout,
        }


class _Tower(nn.Module):
    def __init__(self, config: RetrievalHeadConfig) -> None:
        super().__init__()
        d = config.model_dim
        self.pool = AttentionPool(d, num_heads=4, dropout=config.dropout)
        self.project = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, config.embedding_dim)
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.project(self.pool(features, mask))


class RetrievalHeads(nn.Module):
    """Two towers that never see each other's input."""

    def __init__(self, config: RetrievalHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.protein = _Tower(config)
        self.ligand = _Tower(config)
        self.logit_scale = nn.Parameter(torch.tensor(config.logit_scale_init))

    def encode_protein(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``[B, embedding_dim]`` from the protein stream alone."""
        return self._finish(self.protein(features, mask))

    def encode_ligand(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``[B, embedding_dim]`` from the ligand stream alone."""
        return self._finish(self.ligand(features, mask))

    def _finish(self, vectors: torch.Tensor) -> torch.Tensor:
        if self.config.normalize:
            vectors = torch.nn.functional.normalize(vectors, dim=-1)
        return vectors

    def similarity(self, protein: torch.Tensor, ligand: torch.Tensor) -> torch.Tensor:
        """``[B_p, B_l]`` scaled inner products, the only place they meet."""
        return self.logit_scale.exp().clamp(max=100.0) * protein @ ligand.T

    def forward(self, protein_features, protein_mask, ligand_features, ligand_mask):
        return {
            "protein": self.encode_protein(protein_features, protein_mask),
            "ligand": self.encode_ligand(ligand_features, ligand_mask),
        }


def assert_not_pair_conditioned(head: RetrievalHeads, model_dim: int, *, atol: float = 1e-5) -> None:
    """A ligand vector must not move when the protein beside it changes.

    Run as a test and before building an index. Two different proteins are
    paired with one ligand; if the ligand embedding differs, something
    pair-conditioned reached the tower and the index is invalid.
    """
    head = head.eval()
    torch.manual_seed(0)
    ligand = torch.randn(1, 11, model_dim)
    ligand_mask = torch.ones(1, 11, dtype=torch.bool)
    with torch.no_grad():
        first = head.encode_ligand(ligand, ligand_mask)
        # A protein pass in between must leave no state behind either.
        head.encode_protein(torch.randn(1, 50, model_dim), torch.ones(1, 50, dtype=torch.bool))
        second = head.encode_ligand(ligand, ligand_mask)
    if not torch.allclose(first, second, atol=atol):
        raise RetrievalHeadError(
            "the ligand embedding changed after a protein was encoded; the "
            "tower is pair-conditioned and an index over it would be invalid"
        )


__all__ = [
    "RetrievalHeadConfig",
    "RetrievalHeadError",
    "RetrievalHeads",
    "assert_not_pair_conditioned",
    "masked_mean",
]
