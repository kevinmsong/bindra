"""The assembled model: cached protein features and SMILES in, affinity out.

This is the seam the specification's end-to-end requirement runs through --
"from target sequence and SMILES through affinity prediction plus iBAM" -- and the
object the tiny-data overfit and shared-gradient tests exercise.

## What is trained and what is not

The protein side is **frozen**. ESM-C features come from the populated cache
(10,820 constructs, 14.023 GiB, verified bit-identical on re-encode), so this
module takes ``[B, L, dp]`` residue features rather than sequences. That is a
deliberate asymmetry: fine-tuning a 600M-parameter protein encoder against 572
effective target units would fit the encoder to the targets, not the interaction.

The ligand side **is** trained. Molecules are the axis with real sample size
(1,436,561 distinct), and the GAT arm is randomly initialised by construction.

So gradients flow through: ligand encoder, adapters, cross-attention, heads. They
do not flow into ESM-C, which is not part of this module at all.

## The pooled pair representation

Cross-attention leaves two sequences; the heads need one vector. The pooling
concatenates four things, and each is there for a reason:

* masked mean over fused residues -- what the protein became given this ligand
* masked mean over fused ligand tokens -- the converse
* the adapted ligand graph readout -- whole-molecule properties that no single
  atom carries (size, logP-like bulk), which a mean over atoms attenuates
* the adapted protein mean *before* fusion -- so a prediction can still be made
  when the fused signal is weak, and so the model is not forced to route
  everything through attention

Every pooling step is masked. A plain mean would divide by the padded length, making
every result depend on which other proteins shared the batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch
from torch import nn

from bindra.encoders.ligand_gat import GATConfig, GATLigandEncoderModule
from bindra.encoders.ligand_graph import (
    BatchedGraphs,
    MolecularGraph,
    collate_graphs,
    featurize_molecule,
)
from bindra.features.types import EncoderMetadata
from bindra.fusion.adapters import AdapterConfig, SharedAdapters, masked_mean
from bindra.fusion.cross_attention import (
    CrossAttentionConfig,
    CrossAttentionStack,
    FusionOutput,
)
from bindra.heads.affinity import AffinityHeadConfig, AffinityHeads, AffinityPrediction


class BindraModelError(ValueError):
    """Inputs inconsistent with the model's declared widths or batch size."""


def pad_token_states(
    states: torch.Tensor, batch: BatchedGraphs
) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn concatenated per-atom states into padded ``[B, T, D]`` plus a mask.

    The ligand encoder works on a concatenated batch, because molecule sizes span
    three orders of magnitude and padding there would waste most of the work.
    Cross-attention needs a dense axis, so the padding happens here, once, at the
    boundary -- and the mask it returns is what keeps the padding out of the
    attention.

    Written as two scatters rather than a loop over molecules. The loop version
    was measured at **190 microseconds per molecule** -- 49 ms for a batch of 256,
    more than the entire rest of the forward pass -- because
    ``BatchedGraphs.graph_slice`` calls ``int()`` on a CUDA tensor, forcing a
    device-to-host synchronisation twice per molecule. Nothing about the work is
    expensive; it was all launch latency and stalls.
    """
    num_graphs = batch.num_graphs
    if num_graphs == 0:
        return states.new_zeros(0, 0, states.shape[-1]), torch.zeros(
            0, 0, dtype=torch.bool, device=states.device
        )
    if batch.token_destination is None or not batch.max_atoms:
        raise ValueError(
            "this batch carries no token_destination; build it with "
            "collate_graphs so the padding map is computed on the CPU"
        )
    width = batch.max_atoms
    destination = batch.token_destination

    padded = states.new_zeros(num_graphs * width, states.shape[-1])
    padded.index_copy_(0, destination, states)
    mask = torch.zeros(num_graphs * width, dtype=torch.bool, device=states.device)
    mask.index_fill_(0, destination, True)
    return padded.view(num_graphs, width, -1), mask.view(num_graphs, width)


@dataclass(frozen=True)
class BindraConfig:
    """Composed configuration. Sub-configs stay their own types.

    ``protein_dim`` is the only width that must be supplied, because it comes from
    whichever protein encoder populated the cache. Everything else is derived, so
    a mismatch between the ligand encoder's width and the adapter's input is not
    expressible.
    """

    protein_dim: int
    ligand: GATConfig = field(default_factory=GATConfig)
    model_dim: int = 256
    cross_attention_layers: int = 2
    cross_attention_heads: int = 8
    dropout: float = 0.1
    head: AffinityHeadConfig = field(default_factory=AffinityHeadConfig)

    def adapter_config(self) -> AdapterConfig:
        return AdapterConfig(
            protein_dim=self.protein_dim,
            ligand_dim=self.ligand.hidden_dim,
            ligand_graph_dim=self.ligand.graph_dim,
            model_dim=self.model_dim,
            dropout=self.dropout,
        )

    def cross_attention_config(self) -> CrossAttentionConfig:
        return CrossAttentionConfig(
            model_dim=self.model_dim,
            num_heads=self.cross_attention_heads,
            num_layers=self.cross_attention_layers,
            dropout=self.dropout,
        )

    @property
    def pair_dim(self) -> int:
        """Width of the pooled pair representation: four ``model_dim`` blocks."""
        return 4 * self.model_dim

    def to_dict(self) -> dict[str, object]:
        return {
            "protein_dim": self.protein_dim,
            "model_dim": self.model_dim,
            "cross_attention_layers": self.cross_attention_layers,
            "cross_attention_heads": self.cross_attention_heads,
            "dropout": self.dropout,
            "ligand": self.ligand.to_dict(),
            "head": self.head.to_dict(),
            "pair_dim": self.pair_dim,
        }


@dataclass(frozen=True)
class BindraOutput:
    """Predictions plus everything iBAM needs, and nothing it would have to guess.

    ``fusion`` carries the unaveraged per-layer per-head attention. ``ligand_batch``
    is retained because mapping an attention weight back to an atom requires the
    collation offsets, and recomputing them later risks a different answer.
    """

    prediction: AffinityPrediction
    fusion: FusionOutput
    pair_representation: torch.Tensor  # [B, 4 * model_dim]
    ligand_batch: BatchedGraphs
    ligand_attention: tuple[torch.Tensor, ...] = ()

    @property
    def batch_size(self) -> int:
        return self.prediction.batch_size


class BindraModel(nn.Module):
    """Ligand encoder, shared adapters, bidirectional fusion, endpoint heads."""

    def __init__(self, config: BindraConfig) -> None:
        super().__init__()
        self.config = config
        self.ligand_encoder = GATLigandEncoderModule(config.ligand)
        self.adapters = SharedAdapters(config.adapter_config())
        self.fusion = CrossAttentionStack(config.cross_attention_config())
        self.heads = AffinityHeads(config.pair_dim, config.head)

    @classmethod
    def from_protein_metadata(
        cls, metadata: EncoderMetadata, **kwargs: object
    ) -> "BindraModel":
        """Build with the protein width read off the cache's own encoder metadata.

        Nothing hardcodes 1152: the width comes from whatever produced the cached
        features, so a different protein encoder changes the model shape rather
        than silently mis-slicing.
        """
        return cls(BindraConfig(protein_dim=metadata.output_dim, **kwargs))  # type: ignore[arg-type]

    @property
    def shared_parameter_prefixes(self) -> tuple[str, ...]:
        """Prefixes whose parameters every enabled task must reach.

        Used by the specification's nonzero-shared-gradient check. Naming them here
        rather than in the test means the model declares what "shared" means.
        """
        return ("ligand_encoder.", "adapters.", "fusion.")

    def forward(
        self,
        protein_features: torch.Tensor,  # [B, L, protein_dim]
        protein_mask: torch.Tensor,  # [B, L] bool, True = real residue
        ligand_batch: BatchedGraphs,
        *,
        return_ligand_attention: bool = False,
        need_attention: bool = False,
    ) -> BindraOutput:
        """``need_attention`` defaults to **False**: the training path does not
        need the cross-attention maps, and materialising them dominates memory.
        iBAM asks for them explicitly."""
        self._validate(protein_features, protein_mask, ligand_batch)
        device = protein_features.device
        if ligand_batch.atom_features.device != device:
            ligand_batch = ligand_batch.to(device)

        encoded = self.ligand_encoder(
            ligand_batch, return_attention=return_ligand_attention
        )
        ligand_tokens, ligand_mask = pad_token_states(encoded["token_states"], ligand_batch)

        adapted = self.adapters(
            protein_features, ligand_tokens, encoded["graph_states"]
        )
        fused = self.fusion(
            adapted["protein"],
            adapted["ligand"],
            protein_mask,
            ligand_mask,
            need_attention=need_attention,
        )

        pair = torch.cat(
            (
                masked_mean(fused.protein, protein_mask),
                masked_mean(fused.ligand, ligand_mask),
                adapted["ligand_graph"],
                # The pre-fusion protein mean: a path that does not depend on
                # attention having found anything.
                masked_mean(adapted["protein"], protein_mask),
            ),
            dim=-1,
        )
        return BindraOutput(
            prediction=self.heads(pair),
            fusion=fused,
            pair_representation=pair,
            ligand_batch=ligand_batch,
            ligand_attention=tuple(encoded.get("attention", ())),  # type: ignore[arg-type]
        )

    def _validate(
        self,
        protein_features: torch.Tensor,
        protein_mask: torch.Tensor,
        ligand_batch: BatchedGraphs,
    ) -> None:
        if protein_features.ndim != 3:
            raise BindraModelError(
                f"protein_features must be [B, L, {self.config.protein_dim}], got "
                f"{tuple(protein_features.shape)}"
            )
        if protein_features.shape[-1] != self.config.protein_dim:
            raise BindraModelError(
                f"protein_features width {protein_features.shape[-1]} != configured "
                f"protein_dim {self.config.protein_dim}; the cache was written by a "
                "different encoder than this model was built for"
            )
        if protein_mask.dtype is not torch.bool:
            raise BindraModelError(
                f"protein_mask must be bool with True = real residue, got "
                f"{protein_mask.dtype}"
            )
        if protein_mask.shape != protein_features.shape[:2]:
            raise BindraModelError(
                f"protein_mask must be {protein_features.shape[:2]}, got "
                f"{tuple(protein_mask.shape)}"
            )
        if protein_features.shape[0] != ligand_batch.num_graphs:
            raise BindraModelError(
                f"batch mismatch: {protein_features.shape[0]} proteins against "
                f"{ligand_batch.num_graphs} molecules. Pairs must be aligned "
                "row-for-row; a silent mismatch here trains on wrong pairs"
            )

    def predict(
        self,
        protein_features: torch.Tensor,
        protein_mask: torch.Tensor,
        smiles: Sequence[str],
    ) -> BindraOutput:
        """Convenience path from SMILES, for inference and for small tests."""
        graphs: list[MolecularGraph] = [featurize_molecule(text) for text in smiles]
        batch = collate_graphs(graphs).to(protein_features.device)
        return self(protein_features, protein_mask, batch, need_attention=True)


__all__ = [
    "BindraConfig",
    "BindraModel",
    "BindraModelError",
    "BindraOutput",
    "pad_token_states",
]
