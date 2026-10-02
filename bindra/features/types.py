"""The encoder output contracts every downstream module depends on.

These are the seams named in the specification. They are defined once, here,
so that the cross-attention stack, the iBAM exporter, the retrieval projections
and the diffusion conditioner all agree on what an encoder produced and on how
to map a feature back to a residue, atom or bond.

Two invariants matter more than the rest:

* **Dimensions are discovered, never assumed.** ``feature_dim`` is read off the
  real tensor. Nothing here hardcodes 1152 or 768.
* **One ligand token is not one atom.** KPGT operates on molecular line graphs
  with knowledge nodes, so a token may be a bond, a knowledge node or padding.
  ``token_types`` plus ``token_to_atom``/``token_to_bond`` make that explicit;
  any residue x atom view is a *derived aggregation* and must say so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import hashlib
from typing import Any, Mapping

import torch


class TokenType(StrEnum):
    """What a ligand token actually represents.

    Drawing a knowledge or virtual node as if it were a physical atom would be
    a straightforward misrepresentation, so the type travels with the features.
    """

    ATOM = "atom"
    BOND = "bond"
    KNOWLEDGE = "knowledge"
    VIRTUAL = "virtual"
    PADDING = "padding"
    UNKNOWN = "unknown"

    @property
    def is_physical(self) -> bool:
        """Can this token be drawn onto a molecular depiction?"""
        return self in (TokenType.ATOM, TokenType.BOND)


def sequence_digest(text: str) -> str:
    """Stable content hash used for cache keys and provenance."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EncoderMetadata:
    """Everything needed to decide whether cached features are still valid.

    The legacy project keyed its cache on ``sha1(smiles)[:16]`` alone, with no
    encoder identity in the key. Changing encoder or precision therefore reused
    stale features silently. Every field here participates in the cache key.
    """

    name: str
    checkpoint: str
    output_dim: int
    precision: str
    feature_schema_version: str
    revision: str | None = None
    weights_checksum: str | None = None
    preprocessing_policy: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def cache_key(self, standardized_input: str) -> str:
        """Content-addressed key over the input *and* the producing encoder."""
        parts = [
            standardized_input,
            self.name,
            self.checkpoint,
            str(self.output_dim),
            self.precision,
            self.feature_schema_version,
            self.revision or "",
            self.weights_checksum or "",
            self.preprocessing_policy or "",
        ]
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "checkpoint": self.checkpoint,
            "output_dim": self.output_dim,
            "precision": self.precision,
            "feature_schema_version": self.feature_schema_version,
            "revision": self.revision,
            "weights_checksum": self.weights_checksum,
            "preprocessing_policy": self.preprocessing_policy,
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class ProteinFeatures:
    """Residue-level representations for one protein sequence.

    ``residue_indices`` holds the position of each row in the *original* input
    sequence. It is not decorative: long proteins are encoded in overlapping
    windows, and without explicit numbering a stitched representation cannot be
    mapped back onto the sequence for iBAM display or contact comparison.
    """

    residue_features: torch.Tensor  # [L, dp]
    residue_mask: torch.Tensor  # [L] bool
    residue_indices: torch.Tensor  # [L] long
    sequence_hash: str
    encoder_metadata: EncoderMetadata
    sequence_length: int

    def __post_init__(self) -> None:
        if self.residue_features.ndim != 2:
            raise ValueError(
                f"residue_features must be [L, dp], got {tuple(self.residue_features.shape)}"
            )
        length = self.residue_features.shape[0]
        for name, tensor in (
            ("residue_mask", self.residue_mask),
            ("residue_indices", self.residue_indices),
        ):
            if tensor.shape != (length,):
                raise ValueError(
                    f"{name} must be [{length}], got {tuple(tensor.shape)}"
                )
        if self.residue_mask.dtype is not torch.bool:
            raise ValueError(f"residue_mask must be bool, got {self.residue_mask.dtype}")
        if self.residue_features.shape[1] != self.encoder_metadata.output_dim:
            raise ValueError(
                f"residue_features dim {self.residue_features.shape[1]} disagrees with "
                f"encoder metadata output_dim {self.encoder_metadata.output_dim}"
            )

    @property
    def feature_dim(self) -> int:
        return int(self.residue_features.shape[1])

    @property
    def num_residues(self) -> int:
        return int(self.residue_mask.sum().item())

    @property
    def covers_full_sequence(self) -> bool:
        """Did every residue of the input get a representation?

        False means the encoder truncated or dropped positions, which must be
        reported rather than quietly accepted.
        """
        return self.num_residues == self.sequence_length


@dataclass(frozen=True)
class LigandFeatures:
    """Token-level and graph-level representations for one molecule.

    ``token_to_atom`` maps each token to the atom indices it covers: one index
    for an atom token, two for a bond token, and none for knowledge, virtual or
    padding tokens. Atom indices refer to the *standardized* molecule;
    ``atom_index_maps`` relates those back to the original input ordering.
    """

    token_features: torch.Tensor  # [T, dl]
    token_mask: torch.Tensor  # [T] bool
    token_types: tuple[TokenType, ...]  # length T
    graph_features: torch.Tensor  # [dg]
    token_to_atom: tuple[tuple[int, ...], ...]  # length T
    token_to_bond: tuple[int | None, ...]  # length T
    atom_index_maps: Mapping[str, tuple[int, ...]]
    molecule_id: str
    encoder_metadata: EncoderMetadata
    num_atoms: int

    def __post_init__(self) -> None:
        if self.token_features.ndim != 2:
            raise ValueError(
                f"token_features must be [T, dl], got {tuple(self.token_features.shape)}"
            )
        num_tokens = self.token_features.shape[0]
        if self.token_mask.shape != (num_tokens,):
            raise ValueError(
                f"token_mask must be [{num_tokens}], got {tuple(self.token_mask.shape)}"
            )
        if self.token_mask.dtype is not torch.bool:
            raise ValueError(f"token_mask must be bool, got {self.token_mask.dtype}")
        for name, sequence in (
            ("token_types", self.token_types),
            ("token_to_atom", self.token_to_atom),
            ("token_to_bond", self.token_to_bond),
        ):
            if len(sequence) != num_tokens:
                raise ValueError(
                    f"{name} must have {num_tokens} entries, got {len(sequence)}"
                )
        if self.graph_features.ndim != 1:
            raise ValueError(
                f"graph_features must be [dg], got {tuple(self.graph_features.shape)}"
            )
        self._validate_atom_references()

    def _validate_atom_references(self) -> None:
        """Every referenced atom must exist, and arities must match the type."""
        for position, (token_type, atoms) in enumerate(zip(self.token_types, self.token_to_atom)):
            for atom in atoms:
                if not 0 <= atom < self.num_atoms:
                    raise ValueError(
                        f"token {position} ({token_type}) references atom {atom}, "
                        f"but the molecule has {self.num_atoms} atoms"
                    )
            if token_type is TokenType.ATOM and len(atoms) != 1:
                raise ValueError(
                    f"token {position} is an atom token but maps to {len(atoms)} atoms"
                )
            if token_type is TokenType.BOND and len(atoms) != 2:
                raise ValueError(
                    f"token {position} is a bond token but maps to {len(atoms)} atoms"
                )
            if not token_type.is_physical and atoms:
                raise ValueError(
                    f"token {position} is {token_type} and must not map to atoms, got {atoms}"
                )

    @property
    def feature_dim(self) -> int:
        return int(self.token_features.shape[1])

    @property
    def num_tokens(self) -> int:
        return int(self.token_mask.sum().item())

    def physical_token_positions(self) -> tuple[int, ...]:
        """Positions of tokens that can legitimately be drawn on a molecule."""
        return tuple(
            position
            for position, token_type in enumerate(self.token_types)
            if token_type.is_physical and bool(self.token_mask[position])
        )


__all__ = [
    "EncoderMetadata",
    "LigandFeatures",
    "ProteinFeatures",
    "TokenType",
    "sequence_digest",
]
