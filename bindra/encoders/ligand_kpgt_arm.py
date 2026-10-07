"""The KPGT ligand arm, as the trained model consumes it.

:class:`~bindra.encoders.ligand_kpgt.KPGTLigandEncoder` is the *extraction* path:
it takes SMILES and returns :class:`~bindra.features.types.LigandFeatures`, and
``encode_batch`` runs under ``no_grad``. That is right for populating a cache and
wrong for training. This module is the training path --
:class:`~torch.nn.Module`, collated batch in, autograd graph intact -- so that
:class:`~bindra.model.BindraModel` can hold either arm.

## Three things this arm must reconcile

**A token is not an atom.** A KPGT token is a triplet: a bond (two atoms), an
unbonded atom (one), or a knowledge node (none). The knowledge nodes are
excluded from the fusion token axis and kept in the readout, for the reason
``ligand_arm`` states: a fingerprint vector with a row in the attention map is a
map whose mass cannot be drawn on a molecule. Virtual-atom triplets
(``vavn == -1``) are kept -- they carry one real atom, and the reference's own
readout counts them through ``indicators <= 0``.

**The batch carries two extra per-molecule tensors.** LiGhT's forward is
``(batch, fp, md)``, not ``(batch)``. :class:`KPGTLigandBatch` bundles all three
so the arm presents the same one-argument interface as the GAT arm, and so a
batch cannot be built with its fingerprints and descriptors out of step with its
graphs.

**The descriptors are expensive.** Roughly 200 RDKit descriptors per molecule
dominate this encoder's cost, and they are a deterministic function of the
SMILES, so recomputing them every epoch is pure waste.
:meth:`KPGTLigandBatch.from_smiles` computes them, which makes it correct and
slow; a training pipeline should cache them and call the constructor directly.
Nothing here caches, because the cache format is a decision the training path
owns -- and this is recorded rather than silently accepted, since it is the
difference between a viable epoch and an unviable one.

## Weights

The published ``base.pth`` is not present (figshare, behind a JavaScript
challenge no scripted fetch clears). Constructing this arm therefore gives
**randomly initialised** KPGT weights, and the architecture being verified says
nothing about accuracy. ``KPGTConfig.n_trainable_layers`` defaults to 0, so by
default this arm contributes no trainable parameters at all -- which makes a
randomly initialised frozen arm a fixed random projection. That configuration is
expressible and is almost certainly not what anyone wants; it is left to the
caller rather than silently overridden, and :meth:`frozen_and_untrained` reports
it so a run can refuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from bindra.encoders.kpgt_graph import (
    DEFAULT_POLICY,
    KPGTGraphError,
    KPGTPolicy,
    collate_kpgt,
    featurize_kpgt,
)
from bindra.encoders.kpgt_light import KPGTBatch, LiGhTPredictor
from bindra.encoders.ligand_arm import (
    FusionTokens,
    LigandArmError,
    pad_by_destination,
    physical_token_destination,
)
from bindra.encoders.ligand_kpgt import (
    PRETRAINING_ONLY,
    KPGTConfig,
    apply_kpgt_freezing,
)


@dataclass(frozen=True)
class KPGTLigandBatch:
    """A collated triplet batch plus the two knowledge-node inputs.

    Bundled rather than passed as three arguments so that a graph batch can
    never be paired with another batch's fingerprints. The row counts are
    checked here, because a silent mismatch would train on the wrong molecule's
    descriptors and still produce finite numbers.
    """

    graphs: KPGTBatch
    fingerprints: torch.Tensor  # [B, 512]
    descriptors: torch.Tensor  # [B, 200]

    def __post_init__(self) -> None:
        expected = self.graphs.num_graphs
        for name, tensor in (
            ("fingerprints", self.fingerprints),
            ("descriptors", self.descriptors),
        ):
            if tensor.ndim != 2:
                raise LigandArmError(
                    f"{name} must be [B, D], got {tuple(tensor.shape)}"
                )
            if tensor.shape[0] != expected:
                raise LigandArmError(
                    f"{name} has {tensor.shape[0]} rows against "
                    f"{expected} molecules; the knowledge inputs are out of "
                    "step with the graphs"
                )

    @property
    def num_graphs(self) -> int:
        return self.graphs.num_graphs

    @property
    def total_tokens(self) -> int:
        """Triplet nodes in the batch, knowledge nodes included.

        Named ``tokens`` rather than ``atoms``: for this arm a token is a bond,
        an unbonded atom, or a knowledge node, and none of those is an atom
        count. The training logger prints it as a batch-shape diagnostic.
        """
        return self.graphs.num_nodes

    @property
    def device(self) -> torch.device:
        return self.fingerprints.device

    def to(self, device: torch.device | str) -> "KPGTLigandBatch":
        return KPGTLigandBatch(
            graphs=self.graphs.to(device),
            fingerprints=self.fingerprints.to(device),
            descriptors=self.descriptors.to(device),
        )

    @classmethod
    def from_smiles(
        cls,
        smiles: Sequence[str],
        *,
        policy: KPGTPolicy = DEFAULT_POLICY,
    ) -> "KPGTLigandBatch":
        """Build from SMILES. Correct, and slow enough to matter.

        The ~200 descriptors per molecule dominate; see the module docstring.
        Importing the descriptor generator here rather than at module scope
        keeps the import cost off any path that does not build a batch.
        """
        import numpy as np

        from bindra.encoders.kpgt_descriptors import N_DESCRIPTORS, RDKit2DNormalized
        from bindra.encoders.kpgt_graph import rdk_fingerprint

        if not smiles:
            raise KPGTGraphError("cannot collate an empty batch")
        graphs = collate_kpgt([featurize_kpgt(text, policy) for text in smiles])
        generator = RDKit2DNormalized()
        fingerprints = np.stack([rdk_fingerprint(text, policy) for text in smiles])
        descriptors = np.stack(
            [
                np.asarray(generator.process(text)[1:], dtype=np.float32)
                for text in smiles
            ]
        )
        if descriptors.shape[1] != N_DESCRIPTORS:
            raise LigandArmError(
                f"descriptor generator produced {descriptors.shape[1]} values, "
                f"expected {N_DESCRIPTORS}"
            )
        return cls(
            graphs=graphs,
            fingerprints=torch.from_numpy(fingerprints).to(torch.float32),
            descriptors=torch.from_numpy(
                np.nan_to_num(descriptors)
            ).to(torch.float32),
        )


class KPGTLigandArmModule(nn.Module):
    """LiGhT behind the arm contract.

    The predictor is held as ``self.module`` so its parameter names stay exactly
    the published checkpoint's, which is what lets ``from_pretrained`` load with
    ``strict=True``. State-dict keys here are therefore
    ``ligand_encoder.module.*``, distinct from the GAT arm's
    ``ligand_encoder.*`` -- no trained checkpoint exists for this arm, so there
    is nothing to stay compatible with.
    """

    def __init__(self, config: KPGTConfig | None = None) -> None:
        super().__init__()
        self.config = config or KPGTConfig()
        self.module = LiGhTPredictor(**self.config.to_model_kwargs())  # type: ignore[arg-type]
        #: Set only by :meth:`load_pretrained`. Not a parameter and not in the
        #: state dict, so a resumed run must re-state where its weights came
        #: from rather than inherit the claim.
        self.pretrained_checkpoint: str | None = None
        self.apply_freezing()

    #: The single definition lives beside the freezing policy it belongs to.
    #: The two ``mask_emb`` entries in it are the masked-node objective's
    #: embedding; they were found by backpropagating a real batch and listing
    #: the trainable parameters that received no gradient under any molecule --
    #: which is also how the batch-*dependent* ones were ruled out:
    #: ``virtual_atom_emb`` fires only for a molecule with an unbonded atom (a
    #: salt), and ``trip_fortrans.N`` only for paths of length N.
    PRETRAINING_ONLY = PRETRAINING_ONLY

    def apply_freezing(self) -> None:
        """Set ``requires_grad`` to match the configured regime.

        Applied at construction, not only after a checkpoint load: an arm that
        is only frozen on the load path is unfrozen whenever a model is built
        from config, which is every training run that does not resume.

        Delegates to :func:`~bindra.encoders.ligand_kpgt.apply_kpgt_freezing`,
        the single copy of this policy. See that function for why it is single.
        """
        apply_kpgt_freezing(self.module, self.config)

    @property
    def token_dim(self) -> int:
        return self.config.token_dim

    @property
    def graph_dim(self) -> int:
        return self.config.graph_dim

    @property
    def graph_components(self) -> tuple[int, ...]:
        """Fingerprint node, descriptor node, mean of the rest — three equal parts."""
        width = self.config.d_g_feats
        return (width, width, width)

    @property
    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.module.parameters() if p.requires_grad)

    def frozen_and_untrained(self) -> bool:
        """True when this arm is a fixed random projection.

        Both conditions together -- no trainable parameters and no pretrained
        weights -- mean the ligand side cannot learn and was never taught. A
        caller that cares should refuse rather than train a model whose ligand
        arm is noise. A ``from_scratch`` arm is never this: it is untrained by
        construction but every weight moves.
        """
        if self.config.from_scratch:
            return False
        return self.config.is_frozen and self.pretrained_checkpoint is None

    def partially_frozen_at_random(self) -> int:
        """Parameters frozen at *random* initialisation. Zero is the good case.

        Distinct from :meth:`frozen_and_untrained`, which is the all-or-nothing
        version. This is the quieter failure: partial fine-tuning with no
        checkpoint loaded leaves most of the encoder pinned to its random
        initialisation while the trailing blocks learn to compensate. It trains
        and it reports a loss curve, so nothing visibly breaks.
        """
        if self.pretrained_checkpoint is not None or self.config.from_scratch:
            return 0
        return sum(
            parameter.numel()
            for name, parameter in self.module.named_parameters()
            if not parameter.requires_grad
            and not name.startswith(self.PRETRAINING_ONLY)
        )

    def load_pretrained(self, path: Path | str) -> str:
        """Load the published ``base.pth`` into this arm, strictly.

        ``strict=True`` for the reason the extraction path gives: a silently
        partial load leaves randomly initialised layers inside a model reported
        as pretrained, which is the one failure mode here that still produces
        plausible numbers. Freezing is re-applied afterwards, because loading a
        state dict does not restore ``requires_grad``.
        """
        import hashlib

        path = Path(path)
        if not path.exists():
            raise LigandArmError(
                f"KPGT checkpoint not found at {path}. It is not downloaded "
                "automatically; see docs/STATUS.md for where to get base.pth"
            )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = (
            payload.get("model_state_dict", payload)
            if isinstance(payload, dict)
            else payload
        )
        # The published checkpoint is saved from a DistributedDataParallel run.
        state = {key.removeprefix("module."): value for key, value in state.items()}
        self.module.load_state_dict(state, strict=True)
        self.apply_freezing()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        self.pretrained_checkpoint = f"{path.name}@{digest}"
        return self.pretrained_checkpoint

    def collate_smiles(self, smiles: Sequence[str]) -> KPGTLigandBatch:
        return KPGTLigandBatch.from_smiles(smiles)

    def encode_for_fusion(
        self, batch: KPGTLigandBatch, *, return_attention: bool = False
    ) -> FusionTokens:
        """Every triplet token, densified, plus the three-part readout.

        **Knowledge nodes stay on the token axis.** They are real tokens that
        the protein may attend to, and they are flagged non-physical so iBAM
        never paints a fingerprint onto a molecule. Dropping them before fusion
        was the earlier behaviour; it solved the drawing problem at the cost of
        the protein's only *conditional* path to whole-molecule properties,
        leaving a static concatenated block as the sole route. Set
        ``KPGTConfig.knowledge_tokens_in_fusion = False`` to restore the drop,
        which is the ablation that tells you whether this mattered.

        ``return_attention`` is accepted for interface parity and ignored: LiGhT
        does not expose its per-layer attention through this port, and returning
        an empty tuple is honest where returning zeros would not be. iBAM's
        residue x token view comes from the cross-attention maps, which the
        fusion stack does expose.
        """
        if not isinstance(batch, KPGTLigandBatch):
            raise LigandArmError(
                f"the KPGT arm needs a KPGTLigandBatch, got "
                f"{type(batch).__name__}; the GAT arm's BatchedGraphs carries "
                "no fingerprint or descriptor inputs"
            )
        device = next(self.module.parameters()).device
        if batch.device != device:
            batch = batch.to(device)

        triplet_states, readout = self.module(
            batch.graphs, batch.fingerprints, batch.descriptors
        )

        # vavn: 0 real bond/atom, -1 virtual atom, >=1 knowledge node.
        is_physical = batch.graphs.vavn <= 0
        keep = (
            torch.ones_like(is_physical)
            if self.config.knowledge_tokens_in_fusion
            else is_physical
        )
        destination, width = physical_token_destination(
            batch.graphs.graph_index[keep], batch.graphs.num_graphs
        )
        tokens, mask = pad_by_destination(
            triplet_states[keep],
            destination,
            num_graphs=batch.graphs.num_graphs,
            width=width,
        )
        physical_mask = None
        if self.config.knowledge_tokens_in_fusion:
            # Same scatter, carrying the physical flag instead of the states, so
            # the two masks cannot disagree about which slot holds what.
            flags, _ = pad_by_destination(
                is_physical[keep].unsqueeze(-1).to(tokens.dtype),
                destination,
                num_graphs=batch.graphs.num_graphs,
                width=width,
            )
            physical_mask = flags.squeeze(-1) > 0
        return FusionTokens(
            tokens=tokens,
            token_mask=mask,
            graph_states=readout,
            physical_mask=physical_mask,
            attention=(),
        )


__all__ = ["KPGTLigandArmModule", "KPGTLigandBatch"]
