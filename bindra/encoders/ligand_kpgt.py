"""KPGT molecular encoder, behind the ligand seam, with no DGL.

Owed by the specification:
    Priority 2: use the author's preprocessing and pretrained pathway;
    expose token states AND the graph readout; do not assume one token
    equals one atom.

All three clauses are discharged here.

**The author's preprocessing** is ported in
:mod:`bindra.encoders.kpgt_graph` (triplet graph, 137-dim atom and 14-dim bond
vectors, 512-bit RDKit path fingerprint) and
:mod:`bindra.encoders.kpgt_descriptors` (the 200 normalized descriptors,
vendored because the pretrained weights were fitted against that exact
implementation). Both are verified against the reference running in the
CPU-only DGL environment -- the graph bit-identically, the descriptors to 1e-15.

**The pretrained pathway** is :mod:`bindra.encoders.kpgt_light`, a transcription
of LiGhT whose state dict loads the published checkpoint with ``strict=True``.

**One token is not one atom**, and here that is not a formality. A KPGT token is
a *triplet*: for a bonded pair it is a bond and maps to **two** atoms; for an
unbonded atom it maps to one; for the two knowledge nodes it maps to none and
must never be drawn. :class:`~bindra.features.types.LigandFeatures` carries that
distinction, and ``TokenType.is_physical`` is what stops iBAM painting a
fingerprint node onto a depiction.

This makes KPGT a better fit for the specification's required residue x bond
heatmap than the GAT arm: the bond view is native rather than derived, and the
residue x atom view divides a real two-atom token instead of approximating one.

## Weights

``checkpoint="untrained"`` by default, and that string is part of every feature
cache key, so untrained activations can never be confused with pretrained ones.
:meth:`KPGTLigandEncoder.from_pretrained` loads the published ``base.pth``.

That file is **not** downloaded automatically: it is hosted on figshare behind a
JavaScript challenge that no scripted fetch clears, and no mirror exists. See
``docs/STATUS.md`` for the link. Until it is present, this encoder is
architecturally complete and numerically verified but not pretrained, and no
claim about KPGT accuracy is supported.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from bindra.encoders.kpgt_descriptors import N_DESCRIPTORS, RDKit2DNormalized
from bindra.encoders.kpgt_graph import (
    FEATURE_SCHEMA_VERSION,
    KPGTGraph,
    KPGTGraphError,
    KPGTPolicy,
    collate_kpgt,
    featurize_kpgt,
    rdk_fingerprint,
)
from bindra.encoders.kpgt_light import BASE_CONFIG, LiGhTPredictor
from bindra.encoders.ligand import LigandEncoderError, register_ligand_encoder
from bindra.features.types import EncoderMetadata, LigandFeatures, TokenType

ARCHITECTURE_VERSION = "kpgt-light-base-v1"

#: Parameters that exist only for KPGT's own pre-training objectives. No
#: forward path through this encoder reaches them, so they stay frozen in every
#: regime, ``from_scratch`` included: marking them trainable would allocate
#: optimizer state that never updates and inflate every reported "trainable
#: parameters" count with weights that cannot move.
PRETRAINING_ONLY = (
    "fp_predictor",
    "md_predictor",
    "node_predictor",
    "mask_emb",
    "model.mask_emb",
)


def apply_kpgt_freezing(module: torch.nn.Module, config: "KPGTConfig") -> None:
    """Set ``requires_grad`` on a LiGhT predictor to match ``config``.

    **One function, called from both places.** The encoder (extraction) and the
    arm (training) each used to own a copy of this policy, and they drifted the
    moment ``from_scratch`` existed: constructing an encoder around an arm's
    module -- which is exactly what building provenance metadata does --
    silently re-applied the partial-fine-tuning rule and re-pinned the
    embeddings of a from-scratch model. It trained, and reported a loss curve,
    with 452,896 parameters that were supposed to be learning held at their
    random initialisation. Duplicated policy is what made that possible, so
    there is now only one copy.

    Two regimes, and the difference is not a matter of degree.
    ``n_trainable_layers`` is partial fine-tuning: the trailing blocks move and
    everything beneath them stays put, which is right only when what stays put
    was pretrained. ``from_scratch`` trains all of it except
    :data:`PRETRAINING_ONLY`, because a permanently frozen *random* embedding is
    not a conservative choice but an uncorrectable one.
    """
    for parameter in module.parameters():
        parameter.requires_grad_(False)

    if config.from_scratch:
        for name, parameter in module.named_parameters():
            if not name.startswith(PRETRAINING_ONLY):
                parameter.requires_grad_(True)
        return

    count = config.n_trainable_layers
    if count:
        for layer in module.model.mol_T_layers[-count:]:  # type: ignore[union-attr]
            for parameter in layer.parameters():
                parameter.requires_grad_(True)


@dataclass(frozen=True)
class KPGTConfig:
    """Model shape. Defaults are the published ``base`` configuration."""

    d_g_feats: int = 768
    n_mol_layers: int = 12
    n_heads: int = 12
    d_hpath_ratio: int = 12
    path_length: int = 5
    n_ffn_dense_layers: int = 2
    input_drop: float = 0.0
    feat_drop: float = 0.0
    attn_drop: float = 0.0
    #: How many trailing ``TripletTransformer`` layers stay trainable.
    #:
    #: **Default 0: frozen.** Partial fine-tuning is implemented and tested
    #: (set this to 1 or more), but it is not the default, and the reason is
    #: sequencing rather than preference: there is no baseline yet to say
    #: whether it helps. Frozen keeps the readout a deterministic function of
    #: the SMILES, which is what makes the cache valid and lets an epoch cost
    #: minutes instead of hours. Unfreezing is an ablation to measure against a
    #: baseline, not a starting configuration.
    #:
    #: It has one hard consequence. A cached molecule vector is valid only
    #: because it is a deterministic function of the SMILES; the moment any
    #: layer is trainable that stops being true, and
    #: :meth:`KPGTLigandEncoder.assert_cacheable` refuses to write a cache.
    #: Measured on the RTX 5060 Ti over real curated molecules:
    #:
    #: ===================  ===========  =========  ===========
    #: ``n_trainable``      throughput   per epoch  peak VRAM
    #: ===================  ===========  =========  ===========
    #: 0 (cached)           n/a          ~0         read only
    #: 0 (live)             750 mol/s    1.01 h     1.1 GiB
    #: **1 (default)**      667 mol/s    1.14 h     1.7 GiB
    #: 2                    602 mol/s    1.27 h     2.7 GiB
    #: 12 (full)            309 mol/s    2.46 h     12.7 GiB
    #: ===================  ===========  =========  ===========
    #:
    #: Set to 0 to restore caching. The count participates in the cache key, so
    #: the regimes can never share cached vectors.
    n_trainable_layers: int = 0

    #: Train the whole encoder from random initialisation, embeddings included.
    #:
    #: **This is a different regime, not a bigger** ``n_trainable_layers``.
    #: Partial fine-tuning deliberately keeps the input embeddings and the
    #: path/distance attention biases frozen, because unfreezing an embedding
    #: under frozen layers moves the representation out from under them. That
    #: reasoning assumes the frozen weights are *pretrained*. With no
    #: checkpoint they are random, and freezing them permanently is not
    #: conservative — it is a fixed random input embedding the model can never
    #: correct. Measured: even at the maximum ``n_trainable_layers=12``,
    #: 6,818,977 parameters stay frozen, of which ``triplet_emb`` alone is
    #: 3,500,544.
    #:
    #: So ``from_scratch`` trains everything except :data:`PRETRAINING_ONLY`,
    #: which no forward path through this arm ever calls.
    #:
    #: **What survives is LiGhT, and that is what it gets called.** The authors
    #: describe KPGT as two separable things: "a novel graph transformer
    #: architecture, LiGhT, and a knowledge-guided pre-training strategy". This
    #: regime has the first and not the second, so
    #: :meth:`KPGTLigandEncoder.metadata` reports ``ligand_light_mini`` rather
    #: than any name containing KPGT — a name that kept the acronym and negated
    #: one letter would ask every reader to hold a correction in mind, and a
    #: results table would still get filed against published KPGT numbers.
    #:
    #: The knowledge inputs are **not** lost, which is why "LiGhT" is accurate
    #: rather than a hedge: the 512-bit fingerprint and 200 descriptors still
    #: condition every forward pass and still occupy two nodes per molecule.
    #: They are inputs to the architecture. Only the objective that
    #: *reconstructed* them is gone.
    from_scratch: bool = False

    #: Let the protein cross-attend to the two knowledge nodes.
    #:
    #: They are flagged non-physical, so iBAM still refuses to draw them. The
    #: alternative -- dropping them before fusion -- leaves the static
    #: ``graph_states`` block as the only route for whole-molecule properties,
    #: with no way for attention to weigh them differently per residue. Set
    #: False to restore that drop; the pair is the ablation.
    knowledge_tokens_in_fusion: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.n_trainable_layers <= self.n_mol_layers:
            raise ValueError(
                f"n_trainable_layers must be in [0, {self.n_mol_layers}], "
                f"got {self.n_trainable_layers}"
            )
        if self.from_scratch and self.n_trainable_layers != self.n_mol_layers:
            raise ValueError(
                "from_scratch trains every layer, so n_trainable_layers must "
                f"equal n_mol_layers ({self.n_mol_layers}), got "
                f"{self.n_trainable_layers}. Setting one without the other "
                "would silently train less than the name promises"
            )
        if self.d_g_feats % self.n_heads:
            raise ValueError(
                f"d_g_feats {self.d_g_feats} is not divisible by n_heads "
                f"{self.n_heads}"
            )
        if self.d_g_feats // self.d_hpath_ratio < 1:
            raise ValueError(
                f"d_hpath_ratio {self.d_hpath_ratio} exceeds d_g_feats "
                f"{self.d_g_feats}, leaving the triplet path width at 0"
            )

    @classmethod
    def mini(cls, **overrides: object) -> "KPGTConfig":
        """**LiGhT-mini**: a small line-graph transformer trained from scratch.

        Reported as ``ligand_light_mini``. LiGhT is the authors' own name for
        KPGT's architecture, as distinct from its pre-training strategy, and the
        architecture is all this configuration has. It keeps the line graph, the
        bond-level triplet tokens and the two knowledge nodes; it has no
        pre-trained weights and supports no claim about KPGT.

        **2,232,352 trainable parameters** against the GAT arm's 2,101,120 — 6%
        apart, so a comparison between the two arms is about bond-level versus
        atom-level tokens and the knowledge inputs, not about capacity. Whole
        models: 9,324,195 trainable against v2-mid's 9,254,787, within 0.8%.
        That is the only reason these particular numbers were chosen; they are
        not tuned, and no sweep has been run.

        It is small because the published ``base.pth`` is not obtainable here
        (see ``docs/STATUS.md``). A 91.9M-parameter transformer learning from
        random initialisation on this corpus is not a serious proposition; a
        2.2M one, matched to a baseline that works, is.
        """
        defaults = dict(
            d_g_feats=128,
            n_mol_layers=4,
            n_heads=8,
            d_hpath_ratio=4,
            path_length=5,
            n_trainable_layers=4,
            from_scratch=True,
        )
        defaults.update(overrides)
        return cls(**defaults)  # type: ignore[arg-type]

    @property
    def is_frozen(self) -> bool:
        return self.n_trainable_layers == 0

    def to_model_kwargs(self) -> dict[str, object]:
        return {
            "d_node_feats": 137,
            "d_edge_feats": 14,
            "d_g_feats": self.d_g_feats,
            "d_fp_feats": 512,
            "d_md_feats": N_DESCRIPTORS,
            "d_hpath_ratio": self.d_hpath_ratio,
            "n_mol_layers": self.n_mol_layers,
            "path_length": self.path_length,
            "n_heads": self.n_heads,
            "n_ffn_dense_layers": self.n_ffn_dense_layers,
            "input_drop": self.input_drop,
            "feat_drop": self.feat_drop,
            "attn_drop": self.attn_drop,
            "n_node_types": 1,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "d_g_feats": self.d_g_feats,
            "n_mol_layers": self.n_mol_layers,
            "n_heads": self.n_heads,
            "d_hpath_ratio": self.d_hpath_ratio,
            "path_length": self.path_length,
            "n_trainable_layers": self.n_trainable_layers,
            "from_scratch": self.from_scratch,
            "architecture_version": ARCHITECTURE_VERSION,
        }

    @property
    def token_dim(self) -> int:
        """Width of one triplet state.

        Named the same on every arm's config so ``BindraConfig`` can size its
        adapters without knowing which arm it holds; the GAT arm calls the same
        quantity ``hidden_dim``.
        """
        return self.d_g_feats

    @property
    def graph_dim(self) -> int:
        """Fingerprint node, descriptor node and the mean of the rest."""
        return 3 * self.d_g_feats

    @property
    def graph_components(self) -> tuple[int, ...]:
        """Those three parts, as widths. Mirrored on the arm module.

        Declared on the config as well because ``BindraConfig`` sizes its
        adapters before any module exists.
        """
        return (self.d_g_feats, self.d_g_feats, self.d_g_feats)


class KPGTLigandEncoder:
    """SMILES in, :class:`LigandFeatures` out, with bond-level tokens."""

    def __init__(
        self,
        config: KPGTConfig | None = None,
        *,
        device: torch.device | str | None = None,
        module: LiGhTPredictor | None = None,
        policy: KPGTPolicy | None = None,
        checkpoint: str = "untrained",
    ) -> None:
        self.config = config or KPGTConfig()
        self.policy = policy or KPGTPolicy()
        self.module = module or LiGhTPredictor(**self.config.to_model_kwargs())
        self.checkpoint = checkpoint
        self._descriptors = RDKit2DNormalized()
        self.apply_freezing()
        if device is not None:
            self.module.to(device)

    def apply_freezing(self) -> None:
        """Set ``requires_grad`` to match ``config.n_trainable_layers``.

        Everything is frozen except the trailing ``n_trainable_layers`` triplet
        transformer blocks. The embeddings, the path and distance attention
        biases and the pretraining heads stay frozen in every partial regime:
        unfreezing an input embedding while the layers above it are frozen moves
        the representation out from under them, which is the usual way partial
        fine-tuning goes wrong.

        Called from ``__init__`` so an encoder is never in a state its config
        does not describe.

        Delegates to :func:`apply_kpgt_freezing`, which is also what the
        training arm calls. When these were two copies they drifted, and an
        encoder built around an arm's module re-pinned that arm's embeddings.
        """
        apply_kpgt_freezing(self.module, self.config)

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        """The parameters an optimizer should receive. Empty when fully frozen."""
        return [p for p in self.module.parameters() if p.requires_grad]

    @property
    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def assert_cacheable(self) -> None:
        """Refuse to cache the readout of an encoder that is being trained.

        A cached vector is only valid because it is a deterministic function of
        the SMILES. The moment any KPGT parameter is trainable that stops being
        true, and a cache written mid-training would be silently wrong for every
        step after the one that produced it -- no error, no shape mismatch,
        just stale numbers that look exactly like fresh ones.
        """
        if not self.config.is_frozen:
            raise LigandEncoderError(
                f"this encoder has {self.config.n_trainable_layers} trainable "
                f"layer(s) ({self.n_trainable:,} parameters), so its readout is "
                "not a function of the SMILES alone and must not be cached. "
                "Either set n_trainable_layers=0 to cache, or compute ligand "
                "features live each step (~2.5 h per epoch on the curated corpus)."
            )

    @classmethod
    def from_pretrained(
        cls,
        path: Path | str,
        *,
        config: KPGTConfig | None = None,
        device: torch.device | str | None = None,
        **kwargs: object,
    ) -> "KPGTLigandEncoder":
        """Load the published checkpoint.

        Loads with ``strict=True``: a silently partial load would leave randomly
        initialised layers inside a model reported as pretrained, which is the
        one failure here that produces plausible numbers.
        """
        path = Path(path)
        if not path.exists():
            raise LigandEncoderError(
                f"KPGT checkpoint not found at {path}. It is not downloaded "
                "automatically; see docs/STATUS.md for where to get base.pth"
            )
        config = config or KPGTConfig()
        module = LiGhTPredictor(**config.to_model_kwargs())
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
        # The published checkpoint is saved from a DistributedDataParallel run.
        state = {key.removeprefix("module."): value for key, value in state.items()}
        module.load_state_dict(state, strict=True)
        import hashlib

        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        return cls(
            config,
            device=device,
            module=module,
            checkpoint=f"{path.name}@{digest}",
            **kwargs,  # type: ignore[arg-type]
        )

    @property
    def device(self) -> torch.device:
        return next(self.module.parameters()).device

    @property
    def token_dim(self) -> int:
        return self.config.d_g_feats

    @property
    def graph_dim(self) -> int:
        return self.config.graph_dim

    def metadata(self) -> EncoderMetadata:
        # A from-scratch model keeps KPGT's architecture and its knowledge
        # inputs but none of its pre-training, which is the part the name
        # refers to. Reported under a different name so no results table can
        # read as a KPGT result; the name reaches every cache key and every
        # provenance record.
        return EncoderMetadata(
            name="ligand_light_mini" if self.config.from_scratch else "ligand_kpgt",
            checkpoint=self.checkpoint,
            output_dim=self.token_dim,
            precision=str(next(self.module.parameters()).dtype).removeprefix("torch."),
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            revision=ARCHITECTURE_VERSION,
            preprocessing_policy="kpgt-triplet",
            extra={
                "graph_dim": self.graph_dim,
                **self.config.to_dict(),
                **self.policy.to_dict(),
            },
        )

    def knowledge_inputs(self, smiles: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """The fingerprint and descriptor vectors the two knowledge nodes take.

        Descriptor computation dominates the cost of this encoder -- roughly 200
        RDKit descriptors per molecule, far more than the graph build -- so a
        caller encoding the corpus should cache these rather than recompute them
        every epoch. They are deterministic functions of the SMILES.
        """
        fingerprints = np.stack([rdk_fingerprint(s, self.policy) for s in smiles])
        descriptors = np.stack(
            [np.asarray(self._descriptors.process(s)[1:], dtype=np.float32) for s in smiles]
        )
        if descriptors.shape[1] != N_DESCRIPTORS:
            raise LigandEncoderError(
                f"descriptor generator produced {descriptors.shape[1]} values, "
                f"expected {N_DESCRIPTORS}"
            )
        return (
            torch.from_numpy(fingerprints).to(torch.float32),
            torch.from_numpy(np.nan_to_num(descriptors)).to(torch.float32),
        )

    def encode_tensors(
        self, smiles: Sequence[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(triplet states, molecule readout)`` with the autograd graph intact.

        The training path. :meth:`encode_batch` is the *extraction* path and runs
        under ``no_grad``, which is right for populating a cache and wrong for
        the trailing layers when ``n_trainable_layers > 0`` -- gradients would
        silently never arrive and the unfrozen layer would simply not move.

        Nothing special is done about the frozen layers. Autograd already
        declines to retain their activations, since none of their parameters
        requires gradient; an explicit ``no_grad`` split was written, measured
        against this, found to change peak memory by 1 MiB in 1,670, and removed.
        """
        graphs = [featurize_kpgt(text, self.policy) for text in smiles]
        batch = collate_kpgt(graphs).to(self.device)
        fingerprints, descriptors = self.knowledge_inputs(smiles)
        return self.module(
            batch,
            fingerprints.to(self.device),
            descriptors.to(self.device),
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
    ) -> list[LigandFeatures]:
        if not smiles:
            return []
        if atom_index_maps is not None and len(atom_index_maps) != len(smiles):
            raise LigandEncoderError(
                f"got {len(atom_index_maps)} atom index maps for {len(smiles)} molecules"
            )
        graphs = [featurize_kpgt(text, self.policy) for text in smiles]
        batch = collate_kpgt(graphs).to(self.device)
        fingerprints, descriptors = self.knowledge_inputs(smiles)

        was_training = self.module.training
        self.module.eval()
        try:
            states, readout = self.module(
                batch, fingerprints.to(self.device), descriptors.to(self.device)
            )
        finally:
            self.module.train(was_training)

        metadata = self.metadata()
        offsets = np.concatenate(([0], np.cumsum([g.num_nodes for g in graphs])))
        features: list[LigandFeatures] = []
        for index, graph in enumerate(graphs):
            rows = slice(int(offsets[index]), int(offsets[index + 1]))
            features.append(
                LigandFeatures(
                    token_features=states[rows].cpu(),
                    token_mask=torch.ones(graph.num_nodes, dtype=torch.bool),
                    token_types=_token_types(graph),
                    graph_features=readout[index].cpu(),
                    token_to_atom=graph.triplet_to_atoms,
                    token_to_bond=_token_to_bond(graph),
                    atom_index_maps=dict(atom_index_maps[index]) if atom_index_maps else {},
                    molecule_id=graph.smiles,
                    encoder_metadata=metadata,
                    num_atoms=graph.num_atoms,
                )
            )
        return features


def _token_types(graph: KPGTGraph) -> tuple[TokenType, ...]:
    """Role of each triplet node, from the reference's own ``vavn`` indicator.

    0 is a bond, -1 an unbonded atom, >= 1 a knowledge node. Read off the graph
    rather than inferred from the mapping, so the two cannot disagree.
    """
    types: list[TokenType] = []
    for role in graph.vavn.tolist():
        if role == 0:
            types.append(TokenType.BOND)
        elif role < 0:
            types.append(TokenType.ATOM)
        else:
            types.append(TokenType.KNOWLEDGE)
    return tuple(types)


def _token_to_bond(graph: KPGTGraph) -> tuple[int | None, ...]:
    """Bond index per token; ``None`` for atom and knowledge tokens.

    Bond triplets come first and in RDKit bond order, which is what makes this a
    running counter rather than a lookup.
    """
    mapping: list[int | None] = []
    bond_index = 0
    for role in graph.vavn.tolist():
        if role == 0:
            mapping.append(bond_index)
            bond_index += 1
        else:
            mapping.append(None)
    return tuple(mapping)


@register_ligand_encoder("kpgt")
def _build_kpgt(**kwargs: object) -> KPGTLigandEncoder:
    """Factory for ``build_ligand_encoder("kpgt", ...)``."""
    weights = kwargs.pop("weights", None)
    device = kwargs.pop("device", None)
    config = kwargs.pop("config", None)
    policy = kwargs.pop("policy", None)
    checkpoint = str(kwargs.pop("checkpoint", "untrained"))
    if config is None and kwargs:
        config = KPGTConfig(**kwargs)  # type: ignore[arg-type]
    elif kwargs:
        raise LigandEncoderError(f"unexpected arguments alongside config: {sorted(kwargs)}")
    if weights is not None:
        return KPGTLigandEncoder.from_pretrained(
            weights,  # type: ignore[arg-type]
            config=config,  # type: ignore[arg-type]
            device=device,  # type: ignore[arg-type]
        )
    return KPGTLigandEncoder(
        config,  # type: ignore[arg-type]
        device=device,  # type: ignore[arg-type]
        policy=policy,  # type: ignore[arg-type]
        checkpoint=checkpoint,
    )


__all__ = [
    "ARCHITECTURE_VERSION",
    "KPGTConfig",
    "KPGTGraphError",
    "KPGTLigandEncoder",
]
