"""The bond map: which bonds built which other bonds' representation.

Cross-attention already answers *which bonds does this residue look at*. This
answers the question underneath it -- *what built the representation of that
bond* -- and the two compose into something neither gives alone. If a residue
attends to a scaffold bond whose embedding is dominated by a distal
substituent, the editable thing is the substituent, not the bond.

## What a cell means

``bond_map[i, j]`` is how much triplet ``i`` drew from triplet ``j`` while
forming its representation. Two properties shape how it may be read, and both
have precedents in this package:

* **Rows sum to 1**, because the attention is a softmax over each destination's
  incoming edges. Row sums therefore rank nothing -- the same trap
  ``per_residue`` falls into on the protein side. The rankable direction is the
  **column** sum, :func:`hub_scores`, which says which bonds feed the molecule.
* **It is not symmetric.** ``i -> j`` and ``j -> i`` are separate numbers, and
  the asymmetry is information: a bond that feeds many and draws from few is
  doing structural work.

Edges are **not adjacency**. KPGT connects triplets up to ``max_length + 1``
atoms apart and featurises the path between them, so aspirin's 13 bonds carry
185 edges -- 12.3 per node. A cell is "these bonds are within reach through
this path", not "these bonds touch".

## What it is not

Not bond order, strength or reactivity. Not a chemical interaction. It is
representation flow inside a learned encoder, and it carries the same
disclaimer as every other attention surface here.

## The control that comes first

The attention logit is ``node_attn + dist_attn + path_attn``, and ``dist_attn``
is an explicit learned function of path length. So the map can be tested
against its own distance prior before anything is read from it:
:func:`distance_control` reports how much of the attention is explained by path
length alone. A map that is well predicted by distance has learned the
molecule's topology and nothing chemical, and any interpretation built on it is
unfounded. That answer is cheap, and for an encoder with no pre-training it is
the likely one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class BondMap:
    """One molecule's bond-to-bond attention, dense and physical-only."""

    #: [n_bonds, n_bonds], row i = what bond i drew from. Rows sum to 1 over
    #: the edges present; absent pairs are 0 because no path connects them.
    matrix: np.ndarray
    #: [n_bonds] column sums. The rankable direction.
    hub: np.ndarray
    #: [n_bonds] tuples of atom indices, so a row can be drawn on a molecule.
    #: Empty tuples when the caller did not supply the mapping.
    bond_to_atoms: tuple[tuple[int, ...], ...]
    #: [n_bonds] mass each bond drew from the knowledge nodes rather than from
    #: another bond. ``1 - matrix.sum(axis=1)``, kept rather than renormalised.
    knowledge_mass: np.ndarray
    #: Per-layer, per-head, unaveraged, for a reader who wants the detail the
    #: mean throws away.
    by_layer_head: np.ndarray  # [n_layers, n_heads, n_bonds, n_bonds]
    #: Share of attention variance explained by path length alone. High means
    #: the map is topology, not chemistry.
    distance_r2: float
    disclaimer: str = (
        "Attention over a learned encoder's line graph. NOT bond order, bond "
        "strength, reactivity, or a chemical interaction."
    )

    @property
    def n_bonds(self) -> int:
        return int(self.matrix.shape[0])


def _dense(
    weights: torch.Tensor,
    edge_index: torch.Tensor,
    keep: torch.Tensor,
    n_nodes: int,
) -> np.ndarray:
    """Scatter [E, heads] edge weights into a dense [heads, N, N] matrix."""
    heads = weights.shape[1]
    out = torch.zeros(heads, n_nodes, n_nodes, dtype=torch.float32)
    source, destination = edge_index[0], edge_index[1]
    live = keep[source] & keep[destination]
    s, d, w = source[live], destination[live], weights[live].float()
    for head in range(heads):
        out[head].index_put_((d, s), w[:, head], accumulate=True)
    return out.detach().numpy()


def build_bond_map(
    attention: tuple[torch.Tensor, ...],
    distance_bias: torch.Tensor,
    batch,
    *,
    index: int = 0,
    bond_to_atoms: tuple[tuple[int, ...], ...] | None = None,
) -> BondMap:
    """Assemble one molecule's bond map from the arm's per-layer attention.

    ``batch`` is the ``KPGTBatch`` the attention came from. Only physical
    triplets are kept: a knowledge node is not a bond and cannot be drawn, the
    same rule the fusion token axis applies through ``physical_mask``.

    Dropping those edges is why **rows sum to less than 1**. The softmax
    normalised over every incoming edge including the knowledge nodes, and the
    missing mass is what each bond drew from the molecule's global descriptors
    rather than from another bond. It is left missing rather than renormalised
    away, for the same reason ``nonphysical_mass`` is reported on the protein
    side: the quantity is real and a reader should see it.

    ``bond_to_atoms`` comes from ``LigandFeatures.token_to_atom`` -- the
    verified mapping -- restricted to physical tokens. ``KPGTBatch`` does not
    carry it, because collation concatenates graphs and drops the per-molecule
    ``triplet_to_atoms``, so a caller that wants drawable rows must pass it.
    Without it the map is still correct, just not plottable on a molecule.
    """
    graph_index = batch.graph_index
    rows = (graph_index == index).nonzero(as_tuple=True)[0]
    if rows.numel() == 0:
        raise ValueError(f"no triplets for molecule {index} in this batch")
    physical = batch.vavn <= 0
    keep = torch.zeros_like(physical)
    keep[rows] = True
    keep &= physical

    # Reindex the kept nodes to 0..n-1 so the matrix is the molecule's own.
    local = torch.full((graph_index.shape[0],), -1, dtype=torch.long)
    kept = keep.nonzero(as_tuple=True)[0]
    local[kept] = torch.arange(kept.numel())
    n = int(kept.numel())

    source, destination = batch.edge_index[0], batch.edge_index[1]
    live = keep[source] & keep[destination]
    edges = torch.stack((local[source[live]], local[destination[live]]))

    stacked = []
    for layer in attention:
        stacked.append(_dense(layer[live], edges, torch.ones(n, dtype=torch.bool), n))
    by_layer_head = np.stack(stacked)  # [layers, heads, n, n]
    matrix = by_layer_head.mean(axis=(0, 1))

    return BondMap(
        matrix=matrix,
        hub=matrix.sum(axis=0),
        knowledge_mass=np.clip(1.0 - matrix.sum(axis=1), 0.0, None),
        bond_to_atoms=(
            tuple(bond_to_atoms[:n]) if bond_to_atoms is not None
            else tuple(() for _ in range(n))
        ),
        by_layer_head=by_layer_head,
        distance_r2=distance_control(attention, distance_bias, live),
    )


def distance_control(
    attention: tuple[torch.Tensor, ...],
    distance_bias: torch.Tensor,
    live: torch.Tensor | None = None,
) -> float:
    """How much of the attention path length alone explains, as an R^2.

    Near 1 means the map is the molecule's topology restated and carries no
    chemical information. Run this before reading anything else off the map.
    """
    weights = torch.cat([layer.detach().reshape(-1) for layer in attention]).float()
    bias = distance_bias.detach().reshape(-1).float().repeat(len(attention))
    if live is not None:
        mask = live.repeat(len(attention) * (weights.numel() // live.numel() // len(attention) or 1))
        if mask.numel() == weights.numel():
            weights, bias = weights[mask], bias[mask]
    if weights.numel() < 2:
        return float("nan")
    w = weights - weights.mean()
    b = bias - bias.mean()
    denominator = float((w.norm() * b.norm()).clamp_min(1e-12))
    correlation = float((w * b).sum() / denominator)
    return correlation**2


__all__ = ["BondMap", "build_bond_map", "distance_control"]
