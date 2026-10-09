"""BINDRA-v1: cached protein features and SMILES in, affinity out.

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

from bindra.encoders.ligand_arm import (
    FusionTokens,
    LigandArm,
    LigandArmError,
    build_ligand_arm,
    pad_by_destination,
)
from bindra.encoders.ligand_kpgt import KPGTConfig
from bindra.features.types import EncoderMetadata
from bindra.fusion.adapters import (
    AdapterConfig,
    KnowledgeFiLM,
    SharedAdapters,
    masked_mean,
)
from bindra.fusion.cross_attention import (
    CrossAttentionConfig,
    CrossAttentionStack,
    FusionOutput,
)
from bindra.heads.affinity import AffinityHeadConfig, AffinityHeads, AffinityPrediction

#: Name of this architecture, recorded in every provenance block so a results
#: table says which model produced it. Bump it when the shape changes in a way
#: that makes two runs incomparable.
ARCHITECTURE_NAME = "BINDRA-v1"


class BindraModelError(ValueError):
    """Inputs inconsistent with the model's declared widths or batch size."""


@dataclass(frozen=True)
class BindraConfig:
    """Composed configuration. Sub-configs stay their own types.

    ``protein_dim`` is the only width that must be supplied, because it comes from
    whichever protein encoder populated the cache. Everything else is derived, so
    a mismatch between the ligand encoder's width and the adapter's input is not
    expressible.

    ``ligand`` selects the **arm**. LiGhT is the only one, and the widths are
    read through ``token_dim``/``graph_dim`` so this class never learns an
    arm's internal field names -- which is what lets a second arm be added
    without the adapters, the fusion stack or iBAM knowing it exists.
    """

    protein_dim: int
    ligand: object = field(default_factory=KPGTConfig.mini)
    model_dim: int = 256
    cross_attention_layers: int = 2
    cross_attention_heads: int = 8
    dropout: float = 0.1
    #: Width of each fusion feed-forward block, as a multiple of ``model_dim``.
    #:
    #: Exposed because the fusion stack is where this model's parameters
    #: actually are, and its feed-forwards are most of that. At the default 4
    #: and two layers, fusion is **4,931,840 of 9,324,195** trainable
    #: parameters -- 52.9%, against 23.9% for the ligand arm. An ablation
    #: between two arms is therefore read through a shared stack twice their
    #: size, which attenuates whatever difference exists.
    #:
    #: Measured alternatives, holding layers at 2: a multiplier of 2 gives
    #: 3,290,880 (42.8% fusion, 29.1% arm) and 1 gives 2,470,400 (36.0% / 32.5%).
    #:
    #: **The default stays 4**, because changing it changes the model that
    #: produced the recorded 1.0809 and that number is the current baseline.
    #: Lowering it keeps an arm-versus-arm comparison valid -- both arms see the
    #: same stack -- but costs a fresh GAT run to restore a comparable target.
    feedforward_multiplier: int = 4
    #: Give each part of the ligand graph readout its own projection and its own
    #: block of the pair vector, instead of compressing the concatenation into
    #: one. See ``AdapterConfig.graph_components``. Widens ``pair_dim``.
    split_graph_readout: bool = True
    #: Let the knowledge vector scale and shift the fused streams, instead of
    #: only being concatenated alongside them. See
    #: :class:`~bindra.fusion.adapters.KnowledgeFiLM`. Identity at
    #: initialisation, so turning it off is a clean ablation rather than a
    #: different model.
    film_conditioning: bool = True
    head: AffinityHeadConfig = field(default_factory=AffinityHeadConfig)

    def __post_init__(self) -> None:
        for name in ("token_dim", "graph_dim", "to_dict"):
            if not hasattr(self.ligand, name):
                raise BindraModelError(
                    f"ligand config {type(self.ligand).__name__} has no {name!r}; "
                    "an arm config must expose token_dim, graph_dim and to_dict"
                )

    def adapter_config(self) -> AdapterConfig:
        components = None
        if self.split_graph_readout:
            components = getattr(self.ligand, "graph_components", None)
            if components is None:
                components = (self.ligand.graph_dim,)  # type: ignore[attr-defined]
        return AdapterConfig(
            protein_dim=self.protein_dim,
            ligand_dim=self.ligand.token_dim,  # type: ignore[attr-defined]
            ligand_graph_dim=self.ligand.graph_dim,  # type: ignore[attr-defined]
            model_dim=self.model_dim,
            dropout=self.dropout,
            graph_components=components,
        )

    def cross_attention_config(self) -> CrossAttentionConfig:
        return CrossAttentionConfig(
            model_dim=self.model_dim,
            num_heads=self.cross_attention_heads,
            num_layers=self.cross_attention_layers,
            dropout=self.dropout,
            feedforward_multiplier=self.feedforward_multiplier,
        )

    @property
    def pair_dim(self) -> int:
        """Width of the pooled pair representation.

        Three fixed blocks -- fused protein, fused ligand, pre-fusion protein --
        plus one per graph-readout component. With ``split_graph_readout`` off
        that is the historical four.
        """
        return (3 + self.adapter_config().graph_blocks) * self.model_dim

    def to_dict(self) -> dict[str, object]:
        return {
            "architecture": ARCHITECTURE_NAME,
            "protein_dim": self.protein_dim,
            "model_dim": self.model_dim,
            "cross_attention_layers": self.cross_attention_layers,
            "cross_attention_heads": self.cross_attention_heads,
            "feedforward_multiplier": self.feedforward_multiplier,
            "split_graph_readout": self.split_graph_readout,
            "film_conditioning": self.film_conditioning,
            "dropout": self.dropout,
            "ligand": self.ligand.to_dict(),  # type: ignore[attr-defined]
            "ligand_arm": type(self.ligand).__name__,
            "head": self.head.to_dict(),
            "pair_dim": self.pair_dim,
        }


@dataclass(frozen=True)
class BindraOutput:
    """Predictions plus everything iBAM needs, and nothing it would have to guess.

    ``fusion`` carries the unaveraged per-layer per-head attention. ``ligand_batch``
    is retained because mapping an attention weight back to an atom requires the
    collation offsets, and recomputing them later risks a different answer. Its
    type is whatever the installed arm collates -- ``BatchedGraphs`` for GAT, a
    ``KPGTLigandBatch`` for KPGT -- so a consumer must ask the arm rather than
    assume atoms.

    ``ligand_tokens`` carries the arm's own densified tokens and their mask.
    iBAM needs the mask to know which columns of a cross-attention map are real,
    and for the KPGT arm the token axis is already filtered: knowledge nodes
    were dropped before fusion, so a column index there is not a triplet index.
    """

    prediction: AffinityPrediction
    fusion: FusionOutput
    pair_representation: torch.Tensor  # [B, 4 * model_dim]
    ligand_batch: object
    ligand_tokens: FusionTokens | None = None
    ligand_attention: tuple[torch.Tensor, ...] = ()

    @property
    def batch_size(self) -> int:
        return self.prediction.batch_size


class BindraModel(nn.Module):
    """Ligand encoder, shared adapters, bidirectional fusion, endpoint heads."""

    def __init__(self, config: BindraConfig) -> None:
        super().__init__()
        self.config = config
        # Registered under this exact name, with the arm module itself and no
        # wrapper level, because the trained checkpoints hold keys like
        # ``ligand_encoder.input_projection.weight``. Interposing a wrapper here
        # would rename every one of them and silently stop those checkpoints
        # loading.
        self.ligand_encoder: LigandArm = build_ligand_arm(config.ligand)
        self.adapters = SharedAdapters(config.adapter_config())
        self.fusion = CrossAttentionStack(config.cross_attention_config())
        self.heads = AffinityHeads(config.pair_dim, config.head)
        # Constructed LAST, deliberately. Every module above draws from the RNG
        # stream, so building the conditioner earlier would shift fusion's and
        # the heads' initialisation and `film_conditioning=False` would differ
        # from True by more than the conditioner -- the ablation would measure
        # a different random init at the same time. Built here, the two models
        # are bit-identical until FiLM itself learns something.
        #
        # Conditioning reads the fingerprint and descriptor blocks, so it needs
        # a split readout to have separated them in the first place.
        self.film = (
            KnowledgeFiLM(config.adapter_config())
            if config.film_conditioning and config.adapter_config().graph_blocks >= 2
            else None
        )
        if self.ligand_encoder.token_dim != config.ligand.token_dim:  # type: ignore[attr-defined]
            raise BindraModelError(
                f"arm reports token_dim {self.ligand_encoder.token_dim} but its "
                f"config says {config.ligand.token_dim}"  # type: ignore[attr-defined]
            )

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
        if _ligand_device(ligand_batch) != device:
            ligand_batch = ligand_batch.to(device)

        encoded = self.ligand_encoder.encode_for_fusion(
            ligand_batch, return_attention=return_ligand_attention
        )
        ligand_tokens, ligand_mask = encoded.tokens, encoded.token_mask

        adapted = self.adapters(
            protein_features, ligand_tokens, encoded.graph_states
        )
        conditioned_protein, conditioned_ligand = adapted["protein"], adapted["ligand"]
        if self.film is not None:
            conditioned_protein, conditioned_ligand = self.film(
                adapted["protein"], adapted["ligand"], adapted["ligand_graph"]
            )
        fused = self.fusion(
            conditioned_protein,
            conditioned_ligand,
            protein_mask,
            ligand_mask,
            need_attention=need_attention,
        )

        pair = torch.cat(
            (
                masked_mean(fused.protein, protein_mask),
                masked_mean(fused.ligand, ligand_mask),
                # One block per readout component when the arm declares a split.
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
            ligand_tokens=encoded,
            ligand_attention=encoded.attention,
        )

    def _validate(
        self,
        protein_features: torch.Tensor,
        protein_mask: torch.Tensor,
        ligand_batch: object,
    ) -> None:
        if not hasattr(ligand_batch, "num_graphs"):
            raise BindraModelError(
                f"ligand_batch {type(ligand_batch).__name__} has no num_graphs; "
                "build it with the installed arm's collate_smiles"
            )
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
        if protein_features.shape[0] != ligand_batch.num_graphs:  # type: ignore[attr-defined]
            raise BindraModelError(
                f"batch mismatch: {protein_features.shape[0]} proteins against "
                f"{ligand_batch.num_graphs} molecules. Pairs must be aligned "  # type: ignore[attr-defined]
                "row-for-row; a silent mismatch here trains on wrong pairs"
            )

    def predict(
        self,
        protein_features: torch.Tensor,
        protein_mask: torch.Tensor,
        smiles: Sequence[str],
    ) -> BindraOutput:
        """Convenience path from SMILES, for inference and for small tests.

        Featurization goes through the arm, because the batch type is the thing
        that differs between arms and only the consumer can build one correctly.
        For the KPGT arm this computes ~200 RDKit descriptors per molecule and
        is correspondingly slow -- fine for inference and small tests, which is
        all this path is for.
        """
        batch = self.ligand_encoder.collate_smiles(smiles)
        return self(
            protein_features,
            protein_mask,
            batch.to(protein_features.device),  # type: ignore[attr-defined]
            need_attention=True,
        )


def _ligand_device(batch: object) -> torch.device:
    """Where a ligand batch currently lives, whichever arm collated it.

    ``BatchedGraphs`` exposes its tensors directly and has no ``device``
    property; ``KPGTLigandBatch`` has one. Asked in that order so the GAT path
    keeps working unchanged.
    """
    features = getattr(batch, "atom_features", None)
    if features is not None:
        return features.device
    device = getattr(batch, "device", None)
    if device is not None:
        return device
    raise LigandArmError(
        f"cannot determine the device of a {type(batch).__name__}; a ligand "
        "batch must expose atom_features or device"
    )


__all__ = [
    "ARCHITECTURE_NAME",
    "BindraConfig",
    "BindraModel",
    "BindraModelError",
    "BindraOutput",
]
