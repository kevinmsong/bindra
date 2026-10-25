"""Controlled downstream BINDRA models over immutable pretrained features.

No protein or ligand encoder is constructed here. Cache builders must use the
original ESM-C and five-epoch LiGhT weights, in evaluation mode, rather than any
Davis-adapted member. Normal ``forward`` detaches inputs at that cache boundary;
``adapt(detach=False)`` and ``forward_adapted`` expose explicit attribution seams.

The E/F/G pair order is exactly the production downstream path: fused protein,
fused ligand, fingerprint, descriptors, physical-token mean, pre-fusion protein.
The physical mean is the *encoder's graph readout*, not a recomputed token mean.
Pooled C/D controls retain protein, ligand tokens, and all three graph blocks.
These are research models with one declared score scale, not production bundles.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Literal

import torch
from torch import nn

from bindra.fusion.adapters import AdapterConfig, AttentionPool, KnowledgeFiLM, SharedAdapters
from bindra.fusion.cross_attention import CrossAttentionConfig, CrossAttentionStack, FusionOutput


Arm = Literal["A", "B", "C", "D", "E", "F", "G", "C-matched"]
ARMS = ("A", "B", "C", "D", "E", "F", "G", "C-matched")
LABEL_SCALES = ("pKd", "KIBA_score")
MODEL_SCHEMA = "bindra-frozen-encoder-followup-model-v1"


@dataclass(frozen=True)
class FollowupModelConfig:
    arm: Arm = "E"
    label_scale: str = "pKd"
    protein_dim: int = 1152
    ligand_dim: int = 96
    graph_components: tuple[int, ...] = (96, 96, 96)
    model_dim: int = 192
    cross_attention_layers: int = 2
    cross_attention_heads: int = 8
    feedforward_multiplier: int = 3
    pool_heads: int = 2
    film_bottleneck: int = 192
    score_hidden: int = 256
    dropout: float = 0.1

    def __post_init__(self) -> None:
        # A JSON round trip turns the immutable tuple into a list.
        object.__setattr__(self, "graph_components", tuple(self.graph_components))
        if self.arm not in ARMS:
            raise ValueError(f"unknown follow-up arm {self.arm!r}")
        if self.label_scale not in LABEL_SCALES:
            raise ValueError(f"label_scale must be one of {LABEL_SCALES}, got {self.label_scale!r}")
        if len(self.graph_components) != 3 or any(x < 1 for x in self.graph_components):
            raise ValueError("graph_components must contain FP, descriptor and physical-mean widths")
        for name in ("protein_dim", "ligand_dim", "model_dim", "cross_attention_layers",
                     "cross_attention_heads", "feedforward_multiplier", "pool_heads",
                     "film_bottleneck", "score_hidden"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.model_dim % self.cross_attention_heads:
            raise ValueError("model_dim must be divisible by cross_attention_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def graph_dim(self) -> int:
        return sum(self.graph_components)

    @property
    def uses_protein(self) -> bool:
        return self.arm != "B"

    @property
    def uses_ligand(self) -> bool:
        return self.arm != "A"

    @property
    def uses_cross_attention(self) -> bool:
        return self.arm in ("E", "F", "G")

    @property
    def pair_blocks(self) -> int:
        return {"A": 1, "B": 4, "C": 5, "D": 5, "E": 6, "F": 6,
                "G": 6, "C-matched": 5}[self.arm]


@dataclass(frozen=True)
class CachedFeatures:
    protein_features: torch.Tensor | None = None
    protein_mask: torch.Tensor | None = None
    ligand_tokens: torch.Tensor | None = None
    ligand_mask: torch.Tensor | None = None
    physical_mask: torch.Tensor | None = None
    graph_states: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "CachedFeatures":
        return replace(self, **{name: value.to(device) if value is not None else None
                               for name, value in vars(self).items()})


@dataclass(frozen=True)
class AdaptedFeatures:
    protein: torch.Tensor | None
    ligand: torch.Tensor | None
    ligand_graph: torch.Tensor | None
    protein_mask: torch.Tensor | None
    ligand_mask: torch.Tensor | None
    physical_mask: torch.Tensor | None


@dataclass(frozen=True)
class FollowupOutput:
    values: torch.Tensor  # [B], in label_scale; never unused endpoint columns.
    label_scale: str
    pair_representation: torch.Tensor
    fusion: FusionOutput | None = None
    group_gates: torch.Tensor | None = None


def trainable_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class FollowupModel(nn.Module):
    """A declared architectural arm trained only downstream of frozen caches.

    ``C-matched`` allocates its extra capacity to a fully used MLP layer before
    the common 256-unit regression layer. Its target is max(E,G) for the exact
    same config; matching never depends on outcomes or unused dummy parameters.
    """

    def __init__(self, config: FollowupModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or FollowupModelConfig()
        cfg = self.config
        self.capacity_target: int | None = None
        if cfg.arm == "C-matched":
            # The capacity calculation must not perturb this run's RNG stream.
            with torch.random.fork_rng(devices=[]):
                self.capacity_target = max(trainable_count(FollowupModel(replace(cfg, arm=arm)))
                                           for arm in ("E", "G"))
        adapter_cfg = AdapterConfig(
            protein_dim=cfg.protein_dim, ligand_dim=cfg.ligand_dim,
            ligand_graph_dim=cfg.graph_dim, graph_components=cfg.graph_components,
            model_dim=cfg.model_dim, dropout=cfg.dropout,
            film_bottleneck=cfg.film_bottleneck,
        )
        self.adapters = SharedAdapters(adapter_cfg)
        # Exclude inactive modality adapters from trainable capacity/optimization.
        if not cfg.uses_protein:
            self.adapters.protein.requires_grad_(False)
        if not cfg.uses_ligand:
            self.adapters.ligand.requires_grad_(False)
            self.adapters.ligand_graph_parts.requires_grad_(False)
        self.fusion = (CrossAttentionStack(CrossAttentionConfig(
            model_dim=cfg.model_dim, num_heads=cfg.cross_attention_heads,
            num_layers=cfg.cross_attention_layers, dropout=cfg.dropout,
            feedforward_multiplier=cfg.feedforward_multiplier,
        )) if cfg.uses_cross_attention else None)
        self.protein_pool = (AttentionPool(cfg.model_dim, num_heads=cfg.pool_heads,
                                           dropout=cfg.dropout) if cfg.uses_protein else None)
        self.ligand_pool = (AttentionPool(cfg.model_dim, num_heads=cfg.pool_heads,
                                          dropout=cfg.dropout) if cfg.uses_ligand else None)
        width = cfg.pair_blocks * cfg.model_dim
        self.matched_hidden: int | None = None
        if self.capacity_target is None:
            self.score = nn.Sequential(nn.Linear(width, cfg.score_hidden), nn.GELU(),
                                       nn.Dropout(cfg.dropout), nn.Linear(cfg.score_hidden, 1))
        else:
            # Count before the score head; h*(input+1+score_hidden)+2*score_hidden+1.
            base = trainable_count(self)
            fixed = 2 * cfg.score_hidden + 1
            self.matched_hidden = max(1, round((self.capacity_target - base - fixed) /
                                                (width + 1 + cfg.score_hidden)))
            self.score = nn.Sequential(
                nn.Linear(width, self.matched_hidden), nn.GELU(), nn.Dropout(cfg.dropout),
                nn.Linear(self.matched_hidden, cfg.score_hidden), nn.GELU(),
                nn.Dropout(cfg.dropout), nn.Linear(cfg.score_hidden, 1),
            )
            if abs(trainable_count(self) / self.capacity_target - 1) > 0.05:
                raise ValueError("could not capacity-match pooled control within 5%")
        # Construct FiLM after all shared parameters, so F's removal does not
        # change the score initialization when E and F use the same seed.
        self.film = (KnowledgeFiLM(adapter_cfg) if cfg.arm in ("E", "G") else None)
        # Appended last: E/G have identical shared initialization for a fixed RNG.
        self.group_gate = nn.Linear(width, 3) if cfg.arm == "G" else None
        self.pooled_gate = nn.Linear(width, 2) if cfg.arm == "D" else None
        for gate in (self.group_gate, self.pooled_gate):
            if gate is not None:
                nn.init.zeros_(gate.weight)
                nn.init.zeros_(gate.bias)

    @property
    def trainable_parameters(self) -> int:
        return trainable_count(self)

    def parameter_report(self) -> dict[str, object]:
        return {"schema": MODEL_SCHEMA, "model_config": self.config.to_dict(),
                "trainable_parameters": self.trainable_parameters,
                "registered_parameters": sum(p.numel() for p in self.parameters()),
                "encoder_parameters_registered": 0,
                "capacity_target": self.capacity_target, "matched_hidden": self.matched_hidden,
                "components": {name: trainable_count(module)
                               for name, module in self.named_children()},
                "pair_blocks": self.config.pair_blocks,
                "gates": "2*sigmoid, identity-initialized" if self.group_gate is not None
                         or self.pooled_gate is not None else None}

    def adapt(self, cached: CachedFeatures, *, detach: bool = True,
              validate_inputs: bool = True) -> AdaptedFeatures:
        """Project cached features; attribution can explicitly retain input grads."""
        self._validate_cached(cached, check_values=validate_inputs)
        boundary = lambda x: x.detach() if detach else x
        protein = (self.adapters.protein(boundary(cached.protein_features))
                   if self.config.uses_protein else None)
        ligand = (self.adapters.ligand(boundary(cached.ligand_tokens))
                  if self.config.uses_ligand else None)
        graph = (self.adapters._adapt_graph(boundary(cached.graph_states))
                 if self.config.uses_ligand else None)
        return AdaptedFeatures(protein, ligand, graph, cached.protein_mask,
                               cached.ligand_mask, cached.physical_mask)

    def forward(self, cached: CachedFeatures, *, need_attention: bool = False,
                validate_inputs: bool = True) -> FollowupOutput:
        """Predict; trusted QC'd collations may skip device-reducing mask checks.

        Shapes/dtypes/devices remain checked with ``validate_inputs=False``.
        Only nonempty-mask and physical-subset value reductions are skipped.
        """
        return self.forward_adapted(self.adapt(cached, validate_inputs=validate_inputs),
                                    need_attention=need_attention, validate_inputs=False)

    def prediction(self, cached: CachedFeatures) -> torch.Tensor:
        return self(cached).values

    def forward_adapted(self, adapted: AdaptedFeatures, *, need_attention: bool = False,
                        validate_inputs: bool = True
                        ) -> FollowupOutput:
        """Differentiable path through all interaction and bypass readouts.

        Retain masks while replacing tensors with ``dataclasses.replace`` to
        interpolate or perturb raw/adapted features without changing axes.
        """
        cfg = self.config
        protein, ligand, graph = adapted.protein, adapted.ligand, adapted.ligand_graph
        self._validate_adapted(adapted, check_values=validate_inputs)
        fused = None
        gates = None
        if cfg.arm == "A":
            pair = self.protein_pool(protein, adapted.protein_mask)
        elif cfg.arm == "B":
            l = self.ligand_pool(ligand, adapted.ligand_mask)
            pair = torch.cat((l, graph), dim=-1)
        elif cfg.uses_cross_attention:
            conditioned_p, conditioned_l = protein, ligand
            if self.film is not None:
                conditioned_p, conditioned_l = self.film(protein, ligand, graph)
            fused = self.fusion(conditioned_p, conditioned_l, adapted.protein_mask,
                                adapted.ligand_mask, need_attention=need_attention)
            interaction = torch.cat((self.protein_pool(fused.protein, adapted.protein_mask),
                                     self.ligand_pool(fused.ligand, adapted.ligand_mask)), dim=-1)
            p = self.protein_pool(protein, adapted.protein_mask)
            pair = torch.cat((interaction, graph, p), dim=-1)
            if self.group_gate is not None:
                gates = 2 * torch.sigmoid(self.group_gate(pair))
                pair = torch.cat((interaction * gates[:, 0:1], graph * gates[:, 1:2],
                                  p * gates[:, 2:3]), dim=-1)
        else:
            p = self.protein_pool(protein, adapted.protein_mask)
            l = self.ligand_pool(ligand, adapted.ligand_mask)
            ligand_readouts = torch.cat((l, graph), dim=-1)
            pair = torch.cat((p, ligand_readouts), dim=-1)
            if self.pooled_gate is not None:
                gates = 2 * torch.sigmoid(self.pooled_gate(pair))
                pair = torch.cat((p * gates[:, 0:1], ligand_readouts * gates[:, 1:2]), dim=-1)
        return FollowupOutput(self.score(pair).squeeze(-1), cfg.label_scale, pair, fused, gates)

    def _validate_cached(self, cached: CachedFeatures, *, check_values: bool = True) -> None:
        cfg = self.config
        batch = None
        device = None
        if cfg.uses_protein:
            self._stream("protein_features", cached.protein_features, cached.protein_mask,
                         cfg.protein_dim, check_values=check_values)
            batch, device = cached.protein_features.shape[0], cached.protein_features.device
        if cfg.uses_ligand:
            self._stream("ligand_tokens", cached.ligand_tokens, cached.ligand_mask,
                         cfg.ligand_dim, check_values=check_values)
            if batch is not None and cached.ligand_tokens.shape[0] != batch:
                raise ValueError("protein/ligand batch ordering mismatch")
            if device is not None and cached.ligand_tokens.device != device:
                raise ValueError("protein and ligand tensors must share a device")
            b = cached.ligand_tokens.shape[0]
            if cached.graph_states is None or cached.graph_states.shape != (b, cfg.graph_dim):
                raise ValueError(f"graph_states must be [B,{cfg.graph_dim}]")
            if cached.graph_states.device != cached.ligand_tokens.device:
                raise ValueError("graph_states and tokens must share a device")
            self._physical(cached.physical_mask, cached.ligand_mask, check_values=check_values)

    @staticmethod
    def _stream(name: str, features: torch.Tensor | None, mask: torch.Tensor | None,
                width: int, *, check_values: bool = True) -> None:
        if features is None or features.ndim != 3 or features.shape[-1] != width:
            raise ValueError(f"{name} must be [B,N,{width}]")
        if not features.is_floating_point():
            raise ValueError(f"{name} must contain floating-point encoder features")
        if mask is None or mask.dtype != torch.bool or mask.shape != features.shape[:2]:
            raise ValueError(f"{name} requires an aligned boolean mask")
        if mask.device != features.device:
            raise ValueError(f"{name} and its mask must share a device")
        if features.shape[0] == 0 or (check_values and not bool(mask.any(dim=-1).all())):
            raise ValueError(f"{name} requires at least one real position per row")

    @staticmethod
    def _physical(physical: torch.Tensor | None, mask: torch.Tensor, *,
                  check_values: bool = True) -> None:
        if physical is None or physical.dtype != torch.bool or physical.shape != mask.shape:
            raise ValueError("physical_mask must be an aligned boolean ligand-token mask")
        if physical.device != mask.device or (check_values and bool((physical & ~mask).any())):
            raise ValueError("physical_mask must be a device-aligned subset of ligand_mask")

    def _validate_adapted(self, adapted: AdaptedFeatures, *, check_values: bool = True) -> None:
        cfg = self.config
        if cfg.uses_protein:
            self._stream("adapted protein", adapted.protein, adapted.protein_mask,
                         cfg.model_dim, check_values=check_values)
        if cfg.uses_ligand:
            self._stream("adapted ligand", adapted.ligand, adapted.ligand_mask,
                         cfg.model_dim, check_values=check_values)
            self._physical(adapted.physical_mask, adapted.ligand_mask, check_values=check_values)
            if adapted.ligand_graph is None or adapted.ligand_graph.shape != (
                    adapted.ligand.shape[0], 3 * cfg.model_dim):
                raise ValueError("adapted graph readout must contain three model-width blocks")
        if cfg.uses_protein and cfg.uses_ligand:
            if adapted.protein.shape[0] != adapted.ligand.shape[0]:
                raise ValueError("adapted protein/ligand batch mismatch")


__all__ = ["ARMS", "MODEL_SCHEMA", "LABEL_SCALES", "FollowupModelConfig", "CachedFeatures",
           "AdaptedFeatures", "FollowupOutput", "FollowupModel", "trainable_count"]
