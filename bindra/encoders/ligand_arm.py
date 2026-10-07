"""The ligand arm seam *inside* the trained model.

``bindra/encoders/ligand.py`` already defines a seam, but it is the wrong one for
this job. That protocol is about **extraction**: ``encode``/``encode_batch`` take
SMILES and return :class:`~bindra.features.types.LigandFeatures` under
``no_grad``, which is what populates a cache. :class:`~bindra.model.BindraModel`
needs the **training** interface instead -- an :class:`~torch.nn.Module` that
takes an already-collated batch and returns states with the autograd graph
intact. Those are different contracts, and conflating them is how a cache ends
up holding activations from a trainable encoder.

So this module adds the module-level contract, and it exists because the two
arms genuinely do not line up:

======================  ==========================  ==========================
                        GAT                         KPGT
======================  ==========================  ==========================
batch type              ``BatchedGraphs``           ``KPGTLigandBatch``
one token is            one atom                    one bond (two atoms), or
                                                    one unbonded atom, or a
                                                    knowledge node (no atoms)
extra per-molecule      none                        512-bit fingerprint and
inputs                                              200 normalized descriptors
readout                 mean ++ max over atoms      fp node ++ md node ++ mean
======================  ==========================  ==========================

The specification's own words for this were that the arms "take different batch
types, so this is a real change, not a parameter". The resolution here is that
each arm owns its own padding, because only the arm knows which of its tokens
are real.

## Why padding moved into the arm

:func:`bindra.model.pad_token_states` padded ``BatchedGraphs`` on the assumption
that every token is a real atom. That holds for the GAT arm and fails for KPGT,
whose batch carries two **knowledge nodes** per molecule. Padding them in would
put a fingerprint vector on the cross-attention token axis, where the protein
would attend to it and produce a map whose mass sits on something that cannot be
drawn on a molecule -- precisely what
:attr:`~bindra.features.types.TokenType.is_physical` exists to prevent.

**Decision: knowledge nodes are excluded from the fusion token axis and retained
in the graph readout.** They are not discarded -- KPGT's readout concatenates
both of them explicitly, so whole-molecule fingerprint and descriptor
information still reaches the pair representation through ``graph_states``, which
is the path that carries properties no single token holds. What they do not get
is a row in the attention map. Virtual-atom triplets (``vavn == -1``) *are*
physical: they map to one real atom, and the reference's own readout counts them
via ``indicators <= 0``.

## The one synchronisation

``pad_by_destination`` needs a width, and a width is a host-side integer. The
GAT path gets it free: ``collate_graphs`` computes ``max_atoms`` in numpy on the
CPU, where the data already is. An arm that must derive it from a device tensor
pays **one** device-to-host stall per batch. That is the right order of
magnitude -- the defect this codebase already paid for was 190 microseconds per
*molecule*, 49 ms at batch 256 -- but it is not free, so arms precompute the
width at collation wherever they can and :func:`physical_token_destination`
exists for the fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

import torch


class LigandArmError(ValueError):
    """A batch did not match the arm it was handed to."""


@dataclass(frozen=True)
class FusionTokens:
    """What the fusion stack needs from a ligand arm, and nothing more.

    ``tokens`` is already dense and already masked, so
    :class:`~bindra.model.BindraModel` does not need to know whether a token was
    an atom, a bond or a triplet. ``attention`` is the arm's own internal
    attention, kept separate from the cross-attention maps because iBAM reports
    them separately.
    """

    tokens: torch.Tensor  # [B, T, token_dim]
    token_mask: torch.Tensor  # [B, T] bool, True = real physical token
    graph_states: torch.Tensor  # [B, graph_dim]
    attention: tuple[torch.Tensor, ...] = ()

    def __post_init__(self) -> None:
        if self.tokens.ndim != 3:
            raise LigandArmError(
                f"tokens must be [B, T, token_dim], got {tuple(self.tokens.shape)}"
            )
        if self.token_mask.shape != self.tokens.shape[:2]:
            raise LigandArmError(
                f"token_mask must be {self.tokens.shape[:2]}, got "
                f"{tuple(self.token_mask.shape)}"
            )
        if self.token_mask.dtype is not torch.bool:
            raise LigandArmError(
                f"token_mask must be bool with True = real token, got "
                f"{self.token_mask.dtype}"
            )
        if self.graph_states.ndim != 2:
            raise LigandArmError(
                f"graph_states must be [B, graph_dim], got "
                f"{tuple(self.graph_states.shape)}"
            )
        if self.graph_states.shape[0] != self.tokens.shape[0]:
            raise LigandArmError(
                f"graph_states has {self.graph_states.shape[0]} rows against "
                f"{self.tokens.shape[0]} token rows"
            )

    @property
    def batch_size(self) -> int:
        return int(self.tokens.shape[0])


@runtime_checkable
class LigandArm(Protocol):
    """The trainable ligand module, as :class:`~bindra.model.BindraModel` sees it.

    Implementations are :class:`~torch.nn.Module` subclasses. The protocol is
    ``runtime_checkable`` so a mis-wired arm fails at construction with a clear
    message rather than at the first forward pass with a shape error.
    """

    @property
    def token_dim(self) -> int:
        """Width of each token state."""

    @property
    def graph_dim(self) -> int:
        """Width of the whole-molecule readout."""

    def encode_for_fusion(
        self, batch: Any, *, return_attention: bool = False
    ) -> FusionTokens:
        """Encode a collated batch into dense, masked tokens plus a readout."""

    def collate_smiles(self, smiles: Sequence[str]) -> Any:
        """Featurize and collate SMILES into this arm's own batch type.

        Required on the arm rather than left to the caller because the batch type
        *is* the thing that differs, so the only object that can build one
        correctly is the arm that consumes it.
        """


def pad_by_destination(
    states: torch.Tensor,
    destination: torch.Tensor,
    *,
    num_graphs: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter concatenated token states into ``[B, width, D]`` plus a mask.

    ``destination`` gives each state's slot in the flattened ``B * width`` axis.
    Two scatters, no loop over molecules: the loop version cost 190 microseconds
    per molecule because slicing a batch calls ``int()`` on a CUDA tensor, which
    stalls the pipeline twice each time.
    """
    if num_graphs == 0:
        return (
            states.new_zeros(0, 0, states.shape[-1]),
            torch.zeros(0, 0, dtype=torch.bool, device=states.device),
        )
    if destination.numel() != states.shape[0]:
        raise LigandArmError(
            f"destination has {destination.numel()} entries for "
            f"{states.shape[0]} token states"
        )
    padded = states.new_zeros(num_graphs * width, states.shape[-1])
    padded.index_copy_(0, destination, states)
    mask = torch.zeros(num_graphs * width, dtype=torch.bool, device=states.device)
    mask.index_fill_(0, destination, True)
    return padded.view(num_graphs, width, -1), mask.view(num_graphs, width)


def physical_token_destination(
    graph_index: torch.Tensor, num_graphs: int
) -> tuple[torch.Tensor, int]:
    """Destination slots and width for already-filtered physical tokens.

    ``graph_index`` names the molecule each surviving token belongs to, and must
    be **sorted**, which collation guarantees by concatenating molecules in
    order. Position within a molecule is then ``arange`` minus that molecule's
    start offset, with no sort and no gather.

    Costs **one** device-to-host synchronisation, for the width. Arms that can
    compute the width on the CPU at collation time should do that instead and
    pass it to :func:`pad_by_destination` directly.
    """
    if graph_index.ndim != 1:
        raise LigandArmError(
            f"graph_index must be [N], got {tuple(graph_index.shape)}"
        )
    if graph_index.numel() == 0:
        return graph_index.new_zeros(0), 0
    counts = torch.bincount(graph_index, minlength=num_graphs)
    starts = torch.cumsum(counts, 0) - counts
    within = torch.arange(graph_index.numel(), device=graph_index.device) - starts[
        graph_index
    ]
    width = int(counts.max())  # the one sync
    return graph_index * width + within, width


def build_ligand_arm(config: Any) -> LigandArm:
    """Construct the arm its config names, without naming either arm here.

    Dispatch is on the config type, so adding an arm does not mean editing
    :class:`~bindra.model.BindraModel`. KPGT is imported lazily: it pulls RDKit
    descriptor machinery and the pure-PyTorch LiGhT port, and a broken optional
    arm must not stop the GAT arm from building.
    """
    from bindra.encoders.ligand_gat import GATConfig, GATLigandEncoderModule

    if isinstance(config, GATConfig):
        return GATLigandEncoderModule(config)

    from bindra.encoders.ligand_kpgt import KPGTConfig

    if isinstance(config, KPGTConfig):
        from bindra.encoders.ligand_kpgt_arm import KPGTLigandArmModule

        return KPGTLigandArmModule(config)

    raise LigandArmError(
        f"no ligand arm is registered for config type {type(config).__name__}; "
        "expected GATConfig or KPGTConfig"
    )


__all__ = [
    "FusionTokens",
    "LigandArm",
    "LigandArmError",
    "build_ligand_arm",
    "pad_by_destination",
    "physical_token_destination",
]
