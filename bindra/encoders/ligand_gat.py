"""Legacy 78-feature GAT ligand encoder: ablation 1, and the first working arm.

Owed by the specification:
    Ablation 1: existing ESMC + GAT model retrained on the same curated data.

It is also, for now, the *only* ligand encoder that can run on this GPU: DGL,
which KPGT is written against, cannot load under any torch built for compute
capability 12.0 (see ``envs/kpgt/README.md``). So this module both discharges the
ablation and unblocks everything downstream of the ligand seam.

Design notes that matter for correctness rather than taste:

* **GATv2, not GAT.** The original GAT computes ``a^T [Wh_i || Wh_j]``, which
  factorises into a per-source and a per-destination term; the ranking of
  sources is then the same for every destination ("static attention"). GATv2
  moves the nonlinearity before the projection and removes that limitation. For
  a molecular graph the distinction is real: static attention cannot express
  "this carbonyl matters to the amide nitrogen but not to the ring".
* **Edge features enter the logit, not just the message.** Bond order and
  conjugation should be able to change *how much* a neighbour counts, not only
  what it contributes.
* **Self-loops are added.** Without them a single-atom molecule (methane appears
  in the corpus) has no incoming edge and its representation would be the zero
  vector regardless of its features. The legacy dense adjacency added self-loops
  too, so this also matches the ablation's reference.
* **Attention is returned per layer and per head, unaveraged.** Averaging over
  heads before export destroys the thing iBAM is meant to inspect.

The forward pass is written against the concatenated (not padded) batch from
``ligand_graph.collate_graphs``: molecule sizes in this corpus span three orders
of magnitude, so padding to the batch maximum would waste most of the work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn

from bindra.encoders.ligand import (
    LigandEncoderError,
    register_ligand_encoder,
    segment_max,
    segment_mean,
    segment_softmax,
    segment_sum,
)
from bindra.encoders.ligand_arm import (
    FusionTokens,
    LigandArmError,
    pad_by_destination,
)
from bindra.encoders.ligand_graph import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    FEATURE_SCHEMA_VERSION,
    BatchedGraphs,
    MolecularGraph,
    collate_graphs,
    featurize_molecule,
)
from bindra.features.types import EncoderMetadata, LigandFeatures, TokenType

#: Bumped when the architecture changes in a way that invalidates weights.
ARCHITECTURE_VERSION = "gatv2-edge-selfloop-v1"


@dataclass(frozen=True)
class GATConfig:
    """Shape of the encoder. Every default is declared here, never inline.

    ``hidden_dim`` must divide evenly by ``num_heads``; the constructor checks
    rather than silently truncating, because a truncating integer division loses
    channels in a way that only shows up as slightly worse metrics.
    """

    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    attention_dropout: float = 0.0
    negative_slope: float = 0.2
    #: Mean and max pooling concatenated, so ``graph_dim == 2 * hidden_dim``.
    readout: str = "mean_max"

    def __post_init__(self) -> None:
        if self.hidden_dim % self.num_heads:
            raise ValueError(
                f"hidden_dim {self.hidden_dim} is not divisible by num_heads "
                f"{self.num_heads}"
            )
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if self.readout not in ("mean_max", "mean", "max", "sum"):
            raise ValueError(f"unknown readout {self.readout!r}")

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    @property
    def token_dim(self) -> int:
        """Width of one token state. For this arm a token is one atom.

        Named the same on every arm's config so ``BindraConfig`` can size its
        adapters without knowing which arm it holds; KPGT calls the same
        quantity ``d_g_feats``.
        """
        return self.hidden_dim

    @property
    def graph_dim(self) -> int:
        return 2 * self.hidden_dim if self.readout == "mean_max" else self.hidden_dim

    def to_dict(self) -> dict[str, object]:
        return {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "attention_dropout": self.attention_dropout,
            "negative_slope": self.negative_slope,
            "readout": self.readout,
            "architecture_version": ARCHITECTURE_VERSION,
        }


class GATv2Layer(nn.Module):
    """One GATv2 message-passing layer with edge-conditioned attention."""

    def __init__(self, config: GATConfig, *, edge_dim: int) -> None:
        super().__init__()
        self.config = config
        heads, head_dim = config.num_heads, config.head_dim
        self.source = nn.Linear(config.hidden_dim, heads * head_dim, bias=False)
        self.destination = nn.Linear(config.hidden_dim, heads * head_dim, bias=False)
        self.edge = nn.Linear(edge_dim, heads * head_dim, bias=False)
        self.value = nn.Linear(config.hidden_dim, heads * head_dim, bias=False)
        # The attention vector 'a', one per head.
        self.attention = nn.Parameter(torch.empty(heads, head_dim))
        self.activation = nn.LeakyReLU(config.negative_slope)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.output = nn.Linear(heads * head_dim, config.hidden_dim)
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for linear in (self.source, self.destination, self.edge, self.value, self.output):
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
        nn.init.xavier_uniform_(self.attention)

    def forward(
        self,
        states: torch.Tensor,  # [N, hidden]
        edge_index: torch.Tensor,  # [2, E]
        edge_features: torch.Tensor,  # [E, edge_dim]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the updated states and the per-edge attention ``[E, heads]``."""
        heads, head_dim = self.config.num_heads, self.config.head_dim
        num_atoms = states.shape[0]
        src, dst = edge_index[0], edge_index[1]

        source = self.source(states).view(num_atoms, heads, head_dim)[src]
        destination = self.destination(states).view(num_atoms, heads, head_dim)[dst]
        edge = self.edge(edge_features).view(-1, heads, head_dim)

        # GATv2: the nonlinearity comes *before* the projection onto 'a', which
        # is what makes the attention dynamic rather than static.
        logits = (self.activation(source + destination + edge) * self.attention).sum(-1)
        weights = segment_softmax(logits, dst, num_atoms)
        dropped = self.attention_dropout(weights)

        messages = self.value(states).view(num_atoms, heads, head_dim)[src]
        aggregated = segment_sum(messages * dropped.unsqueeze(-1), dst, num_atoms)

        updated = self.output(aggregated.reshape(num_atoms, heads * head_dim))
        # Pre-norm residual: the residual path is what keeps a 4-layer stack from
        # over-smoothing every atom in a small molecule toward one vector.
        return self.norm(states + self.dropout(updated)), weights


class GATLigandEncoderModule(nn.Module):
    """The trainable part. Separated from the encoder so it can be saved alone."""

    def __init__(self, config: GATConfig | None = None) -> None:
        super().__init__()
        self.config = config or GATConfig()
        self.input_projection = nn.Linear(ATOM_FEATURE_DIM, self.config.hidden_dim)
        self.input_norm = nn.LayerNorm(self.config.hidden_dim)
        # +1 edge-feature channel flags the added self-loops, so the model can
        # tell "bonded to itself" from "single bond with all-zero features".
        self.layers = nn.ModuleList(
            GATv2Layer(self.config, edge_dim=BOND_FEATURE_DIM + 1)
            for _ in range(self.config.num_layers)
        )

    @property
    def token_dim(self) -> int:
        return self.config.hidden_dim

    @property
    def graph_dim(self) -> int:
        return self.config.graph_dim

    def forward(
        self, batch: BatchedGraphs, *, return_attention: bool = False
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        device = self.input_projection.weight.device
        if batch.atom_features.device != device:
            batch = batch.to(device)
        edge_index, edge_features = _with_self_loops(batch)

        states = self.input_norm(self.input_projection(batch.atom_features))
        attention: list[torch.Tensor] = []
        for layer in self.layers:
            states, weights = layer(states, edge_index, edge_features)
            if return_attention:
                attention.append(weights)

        graph = self._readout(states, batch.batch_index, batch.num_graphs)
        result: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "token_states": states,
            "graph_states": graph,
            "edge_index": edge_index,
        }
        if return_attention:
            result["attention"] = attention
        return result

    def collate_smiles(self, smiles: Sequence[str]) -> BatchedGraphs:
        """Featurize and collate into this arm's batch type.

        ``collate_graphs`` computes the padding map in numpy, so the batch that
        comes back already carries ``token_destination`` and ``max_atoms`` and
        :meth:`encode_for_fusion` needs no device-to-host stall for the width.
        """
        return collate_graphs([featurize_molecule(text) for text in smiles])

    def encode_for_fusion(
        self, batch: BatchedGraphs, *, return_attention: bool = False
    ) -> FusionTokens:
        """Dense, masked atom tokens plus the molecule readout.

        Every token this arm emits is one real atom, so the mask marks exactly
        the collated atoms and nothing is filtered out. The KPGT arm is where
        that stops being true.
        """
        if batch.token_destination is None or not batch.max_atoms:
            raise LigandArmError(
                "this batch carries no token_destination; build it with "
                "collate_graphs so the padding map is computed on the CPU"
            )
        encoded = self.forward(batch, return_attention=return_attention)
        tokens, mask = pad_by_destination(
            encoded["token_states"],  # type: ignore[arg-type]
            batch.token_destination,
            num_graphs=batch.num_graphs,
            width=batch.max_atoms,
        )
        return FusionTokens(
            tokens=tokens,
            token_mask=mask,
            graph_states=encoded["graph_states"],  # type: ignore[arg-type]
            attention=tuple(encoded.get("attention", ())),  # type: ignore[arg-type]
        )

    def _readout(
        self, states: torch.Tensor, batch_index: torch.Tensor, num_graphs: int
    ) -> torch.Tensor:
        if self.config.readout == "mean":
            return segment_mean(states, batch_index, num_graphs)
        if self.config.readout == "max":
            return segment_max(states, batch_index, num_graphs)
        if self.config.readout == "sum":
            return segment_sum(states, batch_index, num_graphs)
        return torch.cat(
            (
                segment_mean(states, batch_index, num_graphs),
                segment_max(states, batch_index, num_graphs),
            ),
            dim=-1,
        )


def _with_self_loops(batch: BatchedGraphs) -> tuple[torch.Tensor, torch.Tensor]:
    """Append one self-loop per atom, flagged in a dedicated edge channel.

    Returned rather than stored on the batch so that the batch stays a faithful
    description of the molecule and the self-loops stay an encoder decision.
    """
    device = batch.atom_features.device
    num_atoms = batch.total_atoms
    loops = torch.arange(num_atoms, device=device)
    edge_index = torch.cat(
        (batch.edge_index, torch.stack((loops, loops))), dim=1
    )
    bonds = batch.bond_features
    real = torch.cat((bonds, bonds.new_zeros(bonds.shape[0], 1)), dim=1)
    loop_features = real.new_zeros(num_atoms, BOND_FEATURE_DIM + 1)
    loop_features[:, -1] = 1.0
    return edge_index, torch.cat((real, loop_features), dim=0)


class GATLigandEncoder:
    """Encoder-shaped wrapper: SMILES in, :class:`LigandFeatures` out.

    Randomly initialised unless weights are loaded. That is correct for an
    ablation arm that is *trained* rather than pretrained, but it means
    :meth:`metadata` reports ``checkpoint="untrained"`` until weights exist --
    and that string is part of the feature cache key, so untrained activations
    can never be mistaken for trained ones.
    """

    def __init__(
        self,
        config: GATConfig | None = None,
        *,
        device: torch.device | str | None = None,
        module: GATLigandEncoderModule | None = None,
        checkpoint: str = "untrained",
    ) -> None:
        self.config = config or (module.config if module is not None else GATConfig())
        self.module = module or GATLigandEncoderModule(self.config)
        self.checkpoint = checkpoint
        if device is not None:
            self.module.to(device)

    @property
    def device(self) -> torch.device:
        return self.module.input_projection.weight.device

    @property
    def token_dim(self) -> int:
        return self.module.token_dim

    @property
    def graph_dim(self) -> int:
        return self.module.graph_dim

    def metadata(self) -> EncoderMetadata:
        return EncoderMetadata(
            name="ligand_gat",
            checkpoint=self.checkpoint,
            output_dim=self.token_dim,
            precision=str(self.module.input_projection.weight.dtype).removeprefix("torch."),
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            revision=ARCHITECTURE_VERSION,
            preprocessing_policy="legacy78",
            extra={"graph_dim": self.graph_dim, **self.config.to_dict()},
        )

    def encode(
        self,
        smiles: str,
        *,
        atom_index_maps: Mapping[str, tuple[int, ...]] | None = None,
    ) -> LigandFeatures:
        return self.encode_batch(
            [smiles],
            atom_index_maps=None if atom_index_maps is None else [atom_index_maps],
        )[0]

    @torch.no_grad()
    def encode_batch(
        self,
        smiles: Sequence[str],
        *,
        atom_index_maps: Sequence[Mapping[str, tuple[int, ...]]] | None = None,
        return_attention: bool = False,
    ) -> list[LigandFeatures]:
        if not smiles:
            return []
        if atom_index_maps is not None and len(atom_index_maps) != len(smiles):
            raise LigandEncoderError(
                f"got {len(atom_index_maps)} atom index maps for {len(smiles)} molecules"
            )
        graphs = [featurize_molecule(text) for text in smiles]
        batch = collate_graphs(graphs).to(self.device)
        was_training = self.module.training
        self.module.eval()
        try:
            output = self.module(batch, return_attention=return_attention)
        finally:
            self.module.train(was_training)

        token_states = output["token_states"]
        graph_states = output["graph_states"]
        metadata = self.metadata()
        features: list[LigandFeatures] = []
        for index, graph in enumerate(graphs):
            rows = batch.graph_slice(index)
            maps = dict(atom_index_maps[index]) if atom_index_maps else {}
            features.append(
                LigandFeatures(
                    token_features=token_states[rows].cpu(),
                    token_mask=torch.ones(graph.num_atoms, dtype=torch.bool),
                    # For this encoder one token *is* one atom. Stated, not assumed.
                    token_types=(TokenType.ATOM,) * graph.num_atoms,
                    graph_features=graph_states[index].cpu(),
                    token_to_atom=tuple((atom,) for atom in range(graph.num_atoms)),
                    token_to_bond=(None,) * graph.num_atoms,
                    atom_index_maps=maps,
                    molecule_id=graph.smiles,
                    encoder_metadata=metadata,
                    num_atoms=graph.num_atoms,
                )
            )
        return features

    def attention_by_atom(
        self, smiles: str
    ) -> tuple[LigandFeatures, list[torch.Tensor]]:
        """Encode and return the per-layer, per-head, per-edge attention.

        Unaveraged over heads on purpose: averaging before export destroys what
        iBAM exists to inspect. Edge ``e`` of layer ``l`` carries weight
        ``attention[l][e, head]`` and runs ``edge_index[0, e] -> edge_index[1, e]``
        with the self-loops appended after the real bonds.
        """
        graphs = [featurize_molecule(smiles)]
        batch = collate_graphs(graphs).to(self.device)
        was_training = self.module.training
        self.module.eval()
        try:
            with torch.no_grad():
                output = self.module(batch, return_attention=True)
        finally:
            self.module.train(was_training)
        features = self.encode_batch([smiles])[0]
        return features, [weights.cpu() for weights in output["attention"]]


@register_ligand_encoder("gat")
def _build_gat(**kwargs: object) -> GATLigandEncoder:
    """Factory for ``build_ligand_encoder("gat", ...)``."""
    device = kwargs.pop("device", None)
    checkpoint = str(kwargs.pop("checkpoint", "untrained"))
    config = kwargs.pop("config", None)
    if config is None and kwargs:
        config = GATConfig(**kwargs)  # type: ignore[arg-type]
    elif kwargs:
        raise LigandEncoderError(f"unexpected arguments alongside config: {sorted(kwargs)}")
    return GATLigandEncoder(
        config,  # type: ignore[arg-type]
        device=device,  # type: ignore[arg-type]
        checkpoint=checkpoint,
    )


__all__ = [
    "ARCHITECTURE_VERSION",
    "GATConfig",
    "GATLigandEncoder",
    "GATLigandEncoderModule",
    "GATv2Layer",
]
