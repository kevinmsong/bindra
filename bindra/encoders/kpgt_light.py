"""LiGhT, KPGT's graph transformer, ported to pure PyTorch with no DGL.

DGL's final release cannot load under any torch built for this GPU's compute
capability (see ``envs/kpgt/README.md``), so the reference implementation cannot
run here. The architecture can: KPGT uses DGL as a graph *container* and for four
message-passing primitives, all of which have direct dense/segment equivalents.

| Reference call | Replacement |
|---|---|
| ``fn.u_dot_v('Q', 'K')`` | ``(q[src] * k[dst]).sum(-1)`` |
| ``edge_softmax(g, a)`` (``norm_by='dst'``) | :func:`~bindra.encoders.ligand.segment_softmax` over ``dst`` |
| ``update_all(copy_e('he'), sum)`` | :func:`~bindra.encoders.ligand.segment_sum` over ``dst`` |
| ``dgl.readout_nodes(op='mean')`` | :func:`~bindra.encoders.ligand.segment_mean` over the graph index |

This is a **transcription, not a reimplementation**. Every layer is ordered and
parameterised exactly as the reference, so the published state dict loads
unchanged, and ``tests/test_kpgt_port.py`` checks staged intermediates --
embeddings, path features, each transformer layer, the readout -- against tensors
dumped from the reference running on CPU. A port verified only at the output
tells you nothing about where it diverged.

Several reference behaviours look like bugs and are reproduced deliberately;
they are marked ``# reference quirk`` below. Changing any of them would silently
invalidate the pretrained weights.

## What a "triplet" is, and why that is good news for iBAM

LiGhT does not operate on atoms. Each node is a **triplet** -- an ordered
(atom, bond, atom) unit, one per bond -- plus one node per unbonded atom and
``n_virtual_nodes`` knowledge nodes carrying the fingerprint and the descriptor
vector. So a token is natively a *bond*, which is exactly the residue x bond view
the specification asks iBAM to produce, with a verified two-atom mapping rather
than an invented one.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from bindra.encoders.ligand import segment_mean, segment_softmax, segment_sum

#: Marks a triplet whose second atom does not exist (an unbonded atom).
VIRTUAL_ATOM_INDICATOR = -1
#: Reference placeholder written into the feature vectors of virtual entries.
VIRTUAL_FEATURE_PLACEHOLDER = -1
#: Reference sentinel for an absent path step, before it is clamped to -1.
VIRTUAL_PATH_INDICATOR = -1e6

#: The published "base" configuration, transcribed from ``src/model_config.py``.
BASE_CONFIG: dict[str, object] = {
    "d_node_feats": 137,
    "d_edge_feats": 14,
    "d_g_feats": 768,
    "d_hpath_ratio": 12,
    "n_mol_layers": 12,
    "path_length": 5,
    "n_heads": 12,
    "n_ffn_dense_layers": 2,
    "input_drop": 0.0,
    "attn_drop": 0.1,
    "feat_drop": 0.1,
}


class KPGTPortError(ValueError):
    """The batch tensors did not describe a consistent triplet graph."""


@dataclass(frozen=True)
class KPGTBatch:
    """A batch of triplet graphs, concatenated. The DGL graph's replacement.

    Node axis is the concatenation of every molecule's triplet nodes, so
    ``edge_index`` and ``path`` both index into it. The reference achieves the
    same with ``dgl.batch`` plus ``preprocess_batch_light``, which shifts the path
    indices by the cumulative node counts; here the shift happens at collation.
    """

    begin_end: torch.Tensor  # [N, 2, d_node_feats]
    edge_feats: torch.Tensor  # [N, d_edge_feats]  (per triplet NODE, not per edge)
    vavn: torch.Tensor  # [N] long: 0 real, -1 virtual atom, >=1 knowledge node
    edge_index: torch.Tensor  # [2, E] long
    path: torch.Tensor  # [E, path_length] long, indices into the node axis
    virtual_path: torch.Tensor  # [E] bool
    self_loop: torch.Tensor  # [E] bool
    graph_index: torch.Tensor  # [N] long: which molecule each node belongs to
    num_graphs: int

    def __post_init__(self) -> None:
        num_nodes = self.begin_end.shape[0]
        if self.edge_feats.shape[0] != num_nodes or self.vavn.shape[0] != num_nodes:
            raise KPGTPortError(
                f"node tensors disagree: begin_end {num_nodes}, edge_feats "
                f"{self.edge_feats.shape[0]}, vavn {self.vavn.shape[0]}"
            )
        if self.graph_index.shape[0] != num_nodes:
            raise KPGTPortError(
                f"graph_index must be [{num_nodes}], got {tuple(self.graph_index.shape)}"
            )
        num_edges = self.edge_index.shape[1]
        for name, tensor in (
            ("path", self.path),
            ("virtual_path", self.virtual_path),
            ("self_loop", self.self_loop),
        ):
            if tensor.shape[0] != num_edges:
                raise KPGTPortError(
                    f"{name} must have {num_edges} rows, got {tensor.shape[0]}"
                )
        if num_edges and int(self.edge_index.max()) >= num_nodes:
            raise KPGTPortError(
                f"edge_index references node {int(self.edge_index.max())} but there "
                f"are {num_nodes} nodes; the batch offsets were not applied"
            )

    @property
    def num_nodes(self) -> int:
        return int(self.begin_end.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def to(self, device: torch.device | str) -> "KPGTBatch":
        return KPGTBatch(
            begin_end=self.begin_end.to(device),
            edge_feats=self.edge_feats.to(device),
            vavn=self.vavn.to(device),
            edge_index=self.edge_index.to(device),
            path=self.path.to(device),
            virtual_path=self.virtual_path.to(device),
            self_loop=self.self_loop.to(device),
            graph_index=self.graph_index.to(device),
            num_graphs=self.num_graphs,
        )


class MLP(nn.Module):
    """Transcribed from ``light.MLP``, including its layer-count convention.

    ``n_dense_layers`` counts the input and output projections, so
    ``n_dense_layers=2`` means no hidden layer at all. That reading is not
    obvious and determines the parameter names, so it is preserved rather than
    normalised.
    """

    def __init__(
        self,
        d_in_feats: int,
        d_out_feats: int,
        n_dense_layers: int,
        activation: nn.Module,
        d_hidden_feats: int | None = None,
    ) -> None:
        super().__init__()
        self.n_dense_layers = n_dense_layers
        self.d_hidden_feats = d_out_feats if d_hidden_feats is None else d_hidden_feats
        self.dense_layer_list = nn.ModuleList()
        self.in_proj = nn.Linear(d_in_feats, self.d_hidden_feats)
        for _ in range(n_dense_layers - 2):
            self.dense_layer_list.append(nn.Linear(self.d_hidden_feats, self.d_hidden_feats))
        self.out_proj = nn.Linear(self.d_hidden_feats, d_out_feats)
        self.act = activation

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        feats = self.act(self.in_proj(feats))
        for layer in self.dense_layer_list:
            feats = self.act(layer(feats))
        return self.out_proj(feats)


class Residual(nn.Module):
    """Transcribed from ``light.Residual``.

    Note the asymmetry: the incoming ``y`` is projected and added to ``x``, then
    the *result* is normalised and fed through the feed-forward block. This is
    neither pre-norm nor post-norm in the usual sense, and matching it matters
    for weight compatibility.
    """

    def __init__(
        self,
        d_in_feats: int,
        d_out_feats: int,
        n_ffn_dense_layers: int,
        feat_drop: float,
        activation: nn.Module,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_in_feats)
        self.in_proj = nn.Linear(d_in_feats, d_out_feats)
        self.ffn = MLP(
            d_out_feats,
            d_out_feats,
            n_ffn_dense_layers,
            activation,
            d_hidden_feats=d_out_feats * 4,
        )
        self.feat_dropout = nn.Dropout(feat_drop)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x + self.feat_dropout(self.in_proj(y))
        y = self.norm(x)
        y = self.ffn(y)
        y = self.feat_dropout(y)
        return x + y


class TripletTransformer(nn.Module):
    """One attention layer over the triplet graph."""

    def __init__(
        self,
        d_feats: int,
        d_hpath_ratio: int,
        path_length: int,
        n_heads: int,
        n_ffn_dense_layers: int,
        feat_drop: float = 0.0,
        attn_drop: float = 0.0,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        activation = activation if activation is not None else nn.GELU()
        self.d_feats = d_feats
        self.d_trip_path = d_feats // d_hpath_ratio
        self.path_length = path_length
        self.n_heads = n_heads
        # reference quirk: scaled by the FULL model width, not the head width.
        # The conventional factor is head_dim ** -0.5; using d_feats ** -0.5 makes
        # the logits smaller by sqrt(n_heads). Reproduced because the pretrained
        # weights were fitted under it.
        self.scale = d_feats ** (-0.5)

        self.attention_norm = nn.LayerNorm(d_feats)
        self.qkv = nn.Linear(d_feats, d_feats * 3)
        self.node_out_layer = Residual(
            d_feats, d_feats, n_ffn_dense_layers, feat_drop, activation
        )
        self.feat_dropout = nn.Dropout(p=feat_drop)
        self.attn_dropout = nn.Dropout(p=attn_drop)
        self.act = activation

    def forward(
        self,
        batch: KPGTBatch,
        triplet_h: torch.Tensor,
        dist_attn: torch.Tensor,
        path_attn: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = triplet_h.shape[0]
        head_dim = self.d_feats // self.n_heads
        source, destination = batch.edge_index[0], batch.edge_index[1]

        normed = self.attention_norm(triplet_h)
        qkv = self.qkv(normed).reshape(-1, 3, self.n_heads, head_dim).permute(1, 0, 2, 3)
        query, key, value = qkv[0] * self.scale, qkv[1], qkv[2]

        # u_dot_v('Q', 'K'): Q sits on the source, K on the destination.
        node_attn = (query[source] * key[destination]).sum(dim=-1)  # [E, n_heads]
        logits = node_attn + dist_attn + path_attn
        weights = self.attn_dropout(segment_softmax(logits, destination, num_nodes))

        # pretrans_edges copies the source node's value onto the edge.
        messages = value.reshape(num_nodes, self.n_heads, head_dim)[source]
        messages = (messages * weights.unsqueeze(-1)).reshape(-1, self.d_feats)
        aggregated = segment_sum(messages, destination, num_nodes)
        return self.node_out_layer(triplet_h, aggregated)


class LiGhT(nn.Module):
    """The transformer stack plus its distance and path attention biases."""

    def __init__(
        self,
        d_g_feats: int,
        d_hpath_ratio: int,
        path_length: int,
        n_mol_layers: int = 2,
        n_heads: int = 4,
        n_ffn_dense_layers: int = 4,
        feat_drop: float = 0.0,
        attn_drop: float = 0.0,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        activation = activation if activation is not None else nn.GELU()
        self.n_mol_layers = n_mol_layers
        self.n_heads = n_heads
        self.path_length = path_length
        self.d_g_feats = d_g_feats
        self.d_trip_path = d_g_feats // d_hpath_ratio

        self.mask_emb = nn.Embedding(1, d_g_feats)
        self.path_len_emb = nn.Embedding(path_length + 1, d_g_feats)
        self.virtual_path_emb = nn.Embedding(1, d_g_feats)
        self.self_loop_emb = nn.Embedding(1, d_g_feats)
        self.dist_attn_layer = nn.Sequential(
            nn.Linear(d_g_feats, d_g_feats), activation, nn.Linear(d_g_feats, n_heads)
        )
        self.trip_fortrans = nn.ModuleList(
            MLP(d_g_feats, self.d_trip_path, 2, activation) for _ in range(path_length)
        )
        self.path_attn_layer = nn.Sequential(
            nn.Linear(self.d_trip_path, self.d_trip_path),
            activation,
            nn.Linear(self.d_trip_path, n_heads),
        )
        self.mol_T_layers = nn.ModuleList(
            TripletTransformer(
                d_g_feats,
                d_hpath_ratio,
                path_length,
                n_heads,
                n_ffn_dense_layers,
                feat_drop,
                attn_drop,
                activation,
            )
            for _ in range(n_mol_layers)
        )
        self.feat_dropout = nn.Dropout(p=feat_drop)
        self.attn_dropout = nn.Dropout(p=attn_drop)
        self.act = activation

    def _featurize_path(self, batch: KPGTBatch, path_indices: torch.Tensor) -> torch.Tensor:
        """Embed each edge by how many real steps its path has.

        Virtual-node edges and self-loops get their own learned embedding,
        overwriting the length embedding entirely.
        """
        valid = (path_indices >= 0).to(torch.int32)
        lengths = valid.sum(dim=-1)
        path_feats = self.path_len_emb(lengths).clone()
        path_feats[batch.virtual_path] = self.virtual_path_emb.weight
        path_feats[batch.self_loop] = self.self_loop_emb.weight
        return path_feats

    def _init_path(self, triplet_h: torch.Tensor, path_indices: torch.Tensor) -> torch.Tensor:
        """Average the projected states of the triplets along each path.

        The reference appends a zero row and lets index ``-1`` select it, which is
        how absent path steps contribute nothing. Transcribed exactly, including
        that trick, because an alternative masking scheme would differ in the
        denominator.
        """
        # reference quirk: the sentinel -1e6 is clamped to -1 in place, so both
        # sentinels end up selecting the appended zero row.
        clamped = path_indices.clone()
        clamped[clamped < -99] = -1

        steps = []
        for index in range(self.path_length):
            projected = self.trip_fortrans[index](triplet_h)
            padded = torch.cat(
                (projected, projected.new_zeros(1, self.d_trip_path)), dim=0
            )
            steps.append(padded[clamped[:, index]])
        stacked = torch.stack(steps, dim=-1)
        valid = (clamped >= 0).to(torch.int32)
        path_size = valid.sum(dim=-1, keepdim=True)
        return stacked.sum(dim=-1) / path_size

    def forward(self, batch: KPGTBatch, triplet_h: torch.Tensor) -> torch.Tensor:
        """Run the full stack.

        There is deliberately no "run the frozen prefix under no_grad" option.
        One was written and then removed: with the frozen layers' parameters
        marked ``requires_grad=False``, autograd already declines to retain
        their activations, because nothing in that subgraph requires gradient.
        Measured on a batch of 128 real molecules (68,693 triplet edges) with
        the last layer unfrozen, an explicit split and no split both peak at
        1,670 MiB against 1,671. The parameter was dead machinery resting on a
        wrong belief about where the memory went.

        Peak memory does scale with how much is unfrozen -- 1.1 GiB fully
        frozen, 1.7 with one layer, 12.7 with all twelve -- but that happens on
        its own.
        """
        dist_attn = self.dist_attn_layer(self._featurize_path(batch, batch.path))
        path_attn = self.path_attn_layer(self._init_path(triplet_h, batch.path))
        for layer in self.mol_T_layers:
            triplet_h = layer(batch, triplet_h, dist_attn, path_attn)
        return triplet_h


class AtomEmbedding(nn.Module):
    """Embeds the two atoms of a triplet and sums them."""

    def __init__(self, d_atom_feats: int, d_g_feats: int, input_drop: float) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_atom_feats, d_g_feats)
        self.virtual_atom_emb = nn.Embedding(1, d_g_feats)
        self.input_dropout = nn.Dropout(input_drop)

    def forward(self, pair_node_feats: torch.Tensor, indicators: torch.Tensor) -> torch.Tensor:
        pair_node_h = self.in_proj(pair_node_feats).clone()
        # An unbonded atom's absent partner gets a learned embedding in slot 1.
        pair_node_h[indicators == VIRTUAL_ATOM_INDICATOR, 1, :] = self.virtual_atom_emb.weight
        return self.input_dropout(pair_node_h).sum(dim=-2)


class BondEmbedding(nn.Module):
    """Embeds the bond of a triplet."""

    def __init__(self, d_bond_feats: int, d_g_feats: int, input_drop: float) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_bond_feats, d_g_feats)
        self.virutal_bond_emb = nn.Embedding(1, d_g_feats)  # reference spelling
        self.input_dropout = nn.Dropout(input_drop)

    def forward(self, edge_feats: torch.Tensor, indicators: torch.Tensor) -> torch.Tensor:
        edge_h = self.in_proj(edge_feats).clone()
        # reference quirk: only virtual *atoms* (-1) get the learned bond
        # embedding. Knowledge nodes (>=1) carry placeholder -1 features that are
        # pushed through the linear layer as if they were a real bond.
        edge_h[indicators == VIRTUAL_ATOM_INDICATOR] = self.virutal_bond_emb.weight
        return self.input_dropout(edge_h)


class TripletEmbedding(nn.Module):
    """Combines atom and bond embeddings, then overwrites the knowledge nodes."""

    def __init__(
        self,
        d_g_feats: int,
        d_fp_feats: int,
        d_md_feats: int,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        activation = activation if activation is not None else nn.GELU()
        self.in_proj = MLP(d_g_feats * 2, d_g_feats, 2, activation)
        self.fp_proj = MLP(d_fp_feats, d_g_feats, 2, activation)
        self.md_proj = MLP(d_md_feats, d_g_feats, 2, activation)

    def forward(
        self,
        node_h: torch.Tensor,
        edge_h: torch.Tensor,
        fp: torch.Tensor,
        md: torch.Tensor,
        indicators: torch.Tensor,
    ) -> torch.Tensor:
        triplet_h = self.in_proj(torch.cat((node_h, edge_h), dim=-1)).clone()
        # Knowledge node 1 carries the fingerprint, node 2 the descriptors. Their
        # rows appear once per molecule, in batch order.
        triplet_h[indicators == 1] = self.fp_proj(fp)
        triplet_h[indicators == 2] = self.md_proj(md)
        return triplet_h


class LiGhTPredictor(nn.Module):
    """The full KPGT encoder. Parameter names match the published checkpoint.

    ``generate_fps`` is the method the ligand encoder uses: it returns the
    ``3 * d_g_feats`` molecule representation, and :meth:`encode` additionally
    returns the per-triplet states that iBAM needs.
    """

    def __init__(
        self,
        d_node_feats: int = 40,
        d_edge_feats: int = 12,
        d_g_feats: int = 128,
        d_fp_feats: int = 512,
        d_md_feats: int = 200,
        d_hpath_ratio: int = 1,
        n_mol_layers: int = 2,
        path_length: int = 5,
        n_heads: int = 4,
        n_ffn_dense_layers: int = 2,
        input_drop: float = 0.0,
        feat_drop: float = 0.0,
        attn_drop: float = 0.0,
        activation: nn.Module | None = None,
        n_node_types: int = 1,
        readout_mode: str = "mean",
    ) -> None:
        super().__init__()
        activation = activation if activation is not None else nn.GELU()
        self.d_g_feats = d_g_feats
        self.readout_mode = readout_mode
        self.node_emb = AtomEmbedding(d_node_feats, d_g_feats, input_drop)
        self.edge_emb = BondEmbedding(d_edge_feats, d_g_feats, input_drop)
        self.triplet_emb = TripletEmbedding(d_g_feats, d_fp_feats, d_md_feats, activation)
        self.mask_emb = nn.Embedding(1, d_g_feats)
        self.model = LiGhT(
            d_g_feats,
            d_hpath_ratio,
            path_length,
            n_mol_layers,
            n_heads,
            n_ffn_dense_layers,
            feat_drop,
            attn_drop,
            activation,
        )
        # Pretraining heads. Kept so the published state dict loads strictly;
        # unused for feature extraction.
        self.node_predictor = nn.Sequential(
            nn.Linear(d_g_feats, d_g_feats), activation, nn.Linear(d_g_feats, n_node_types)
        )
        self.fp_predictor = nn.Sequential(
            nn.Linear(d_g_feats, d_g_feats), activation, nn.Linear(d_g_feats, d_fp_feats)
        )
        self.md_predictor = nn.Sequential(
            nn.Linear(d_g_feats, d_g_feats), activation, nn.Linear(d_g_feats, d_md_feats)
        )

    def triplet_states(
        self, batch: KPGTBatch, fp: torch.Tensor, md: torch.Tensor
    ) -> torch.Tensor:
        """Per-triplet representations: the token states iBAM attributes over."""
        indicators = batch.vavn
        node_h = self.node_emb(batch.begin_end, indicators)
        edge_h = self.edge_emb(batch.edge_feats, indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)
        return self.model(batch, triplet_h)

    def readout(self, batch: KPGTBatch, triplet_h: torch.Tensor) -> torch.Tensor:
        """``[B, 3 * d_g_feats]``: fingerprint node, descriptor node, mean of the rest.

        The reference deletes the knowledge nodes from the graph before averaging.
        Masking them out of a segment mean is the same operation without mutating
        anything, which matters because the reference's ``remove_nodes`` edits the
        graph in place and cannot be called twice.
        """
        indicators = batch.vavn
        real = indicators <= 0
        mean = segment_mean(
            triplet_h[real], batch.graph_index[real], batch.num_graphs
        )
        return torch.cat(
            (triplet_h[indicators == 1], triplet_h[indicators == 2], mean), dim=-1
        )

    def generate_fps(
        self, batch: KPGTBatch, fp: torch.Tensor, md: torch.Tensor
    ) -> torch.Tensor:
        return self.readout(batch, self.triplet_states(batch, fp, md))

    def forward(
        self, batch: KPGTBatch, fp: torch.Tensor, md: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(triplet states, molecule representation)``."""
        triplet_h = self.triplet_states(batch, fp, md)
        return triplet_h, self.readout(batch, triplet_h)


__all__ = [
    "BASE_CONFIG",
    "VIRTUAL_ATOM_INDICATOR",
    "VIRTUAL_PATH_INDICATOR",
    "AtomEmbedding",
    "BondEmbedding",
    "KPGTBatch",
    "KPGTPortError",
    "LiGhT",
    "LiGhTPredictor",
    "MLP",
    "Residual",
    "TripletEmbedding",
    "TripletTransformer",
]
