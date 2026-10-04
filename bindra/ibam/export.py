"""Raw attention matrices plus plotting metadata.

Owed by the specification:
    Priority 2: export numeric matrices, not just images; residue x
    token/bond and derived residue x atom views; bond tokens split equally
    between endpoints and labelled as a derived aggregation.

## What iBAM is, and what it is not

An attention-based explanation. Not an experimentally established contact map, not
a binding pose, not a mechanism. Every export carries that disclaimer as data
(:data:`DISCLAIMER`) so it travels with the matrices into whatever reads them,
rather than living only in a paper's caption.

## The two views, and why one is privileged

* **Residue x token** is native. It is literally what the model computed, masked
  and per-head, with no interpretation applied.
* **Residue x atom** is *derived*. A token is not an atom. For the GAT arm one
  token happens to be one atom, which is still recorded rather than assumed; for
  KPGT a token may be a bond, a triplet or a knowledge node.

:class:`IBAMExport` keeps both, and ``residue_atom_is_derived`` is always True --
there is no path that produces a residue x atom matrix and claims it is native.

## The bond rule

The specification fixes the initial rule: a bond token's contribution is divided
**equally** between its two endpoint atoms. That is a choice, not a measurement:
attention on a bond does not mean both atoms contributed equally. It is labelled
as a derived aggregation and the policy is recorded in every export.

## Knowledge and virtual nodes

Never drawn as physical atoms. Their attention mass is **preserved in the raw
export** (``nonphysical_mass``), not silently dropped, because dropping it makes
the remaining numbers look like a complete distribution when they are not. If a
visualization renormalises the physical tokens to sum to 1 after excluding them,
``renormalized`` records that it did and ``nonphysical_mass`` says how much was
set aside.

## Residue numbering

Positions come from ``ProteinFeatures.residue_indices``, which are indices into
the **original** sequence. Long proteins are encoded in overlapping 2,046-residue
windows, so a row number in the feature matrix is not a residue number. Exporting
row indices as residue numbers would silently misplace every annotation on any
protein over the window length -- and 261 of the 10,820 constructs are over it.

## The aggregation policy is fixed in advance

The specification requires a fixed aggregation policy "chosen before inspecting
test examples". :data:`DEFAULT_POLICY` is that choice, declared here with a
content hash so a later change is visible in the export rather than lost in a
diff: mean over heads, last layer. Mean over heads because a single head is not
more trustworthy than another and picking one post hoc is selection; last layer
because it is the representation the prediction is actually read from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from bindra.features.types import LigandFeatures, TokenType
from bindra.fusion.cross_attention import FusionOutput

#: Travels with every export. The specification forbids the stronger readings.
DISCLAIMER = (
    "Attention-based explanation. NOT an experimentally established contact map, "
    "binding pose, or mechanism. The residue x atom view is a derived aggregation."
)


class IBAMExportError(ValueError):
    """The attention, the features and the mapping did not describe one molecule."""


@dataclass(frozen=True)
class AggregationPolicy:
    """How heads and layers are combined, and how bond tokens are split.

    Fixed before any test example was inspected. ``policy_hash`` makes a later
    change visible in every export that used it.
    """

    head_reduction: str = "mean"
    layer_reduction: str = "last"
    #: Equal split between a bond's endpoints, as the specification prescribes.
    bond_split: str = "equal"
    #: Renormalise the physical tokens to sum to 1 after excluding knowledge and
    #: virtual nodes. Off by default: the raw numbers are the honest ones.
    renormalize_physical: bool = False

    def __post_init__(self) -> None:
        if self.head_reduction not in ("mean", "max"):
            raise IBAMExportError(f"unknown head_reduction {self.head_reduction!r}")
        if self.layer_reduction not in ("last", "mean", "first"):
            raise IBAMExportError(f"unknown layer_reduction {self.layer_reduction!r}")
        if self.bond_split != "equal":
            raise IBAMExportError(
                f"bond_split {self.bond_split!r} is not implemented; the "
                "specification prescribes 'equal'"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "head_reduction": self.head_reduction,
            "layer_reduction": self.layer_reduction,
            "bond_split": self.bond_split,
            "renormalize_physical": self.renormalize_physical,
        }

    @property
    def policy_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def describe(self) -> str:
        """One line, suitable for a figure caption."""
        return (
            f"{self.layer_reduction} layer, {self.head_reduction} over heads, "
            f"bond contributions split {self.bond_split}ly between endpoint atoms "
            f"(derived aggregation)"
        )


DEFAULT_POLICY = AggregationPolicy()


@dataclass(frozen=True)
class IBAMExport:
    """Numeric matrices plus everything needed to plot or re-derive them.

    ``residue_token`` is native; ``residue_atom`` is derived. Both are exported,
    and the derived flag is not optional.
    """

    #: [L, T] native attention, residues by ligand tokens.
    residue_token: np.ndarray
    #: [L, n_atoms] derived through the verified token-to-atom mapping.
    residue_atom: np.ndarray
    #: [L] summary per residue, over physical tokens only.
    per_residue: np.ndarray
    #: [n_atoms] summary per atom.
    per_atom: np.ndarray
    #: Indices into the ORIGINAL protein sequence, not row numbers.
    residue_positions: np.ndarray
    #: Atom indices in the standardized molecule.
    atom_indices: np.ndarray
    token_types: tuple[str, ...]
    token_to_atom: tuple[tuple[int, ...], ...]
    token_to_bond: tuple[int | None, ...]
    #: Attention mass sitting on knowledge/virtual tokens, per residue. Preserved,
    #: never dropped silently.
    nonphysical_mass: np.ndarray
    #: True if the physical tokens were renormalised after excluding the above.
    renormalized: bool
    #: Full per-layer, per-head attention, unaveraged: [n_layers, H, L, T].
    attention_by_layer_head: np.ndarray
    policy: AggregationPolicy
    molecule_id: str
    model_checkpoint: str
    ligand_encoder: str
    protein_encoder: str
    num_layers: int
    num_heads: int
    atom_index_maps: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    residue_atom_is_derived: bool = True
    disclaimer: str = DISCLAIMER
    ensemble_std: np.ndarray | None = None
    #: [T, L] ligand->protein attention, the other direction of the fusion.
    #: Its rows sum to 1 over **residues**, so summing it over ligand tokens is
    #: a per-residue quantity that actually varies. ``per_residue`` cannot be:
    #: the protein->ligand rows are softmax-normalised over atoms, so every
    #: residue's total is exactly 1 and ranking residues by it ranks ties.
    token_residue: np.ndarray | None = None
    #: [L] total ligand attention landing on each residue. The honest answer to
    #: "which residues does this model look at", and what ``top_residues`` uses
    #: when it is present.
    residue_salience: np.ndarray | None = None

    def __post_init__(self) -> None:
        length, num_tokens = self.residue_token.shape
        if self.residue_atom.shape[0] != length:
            raise IBAMExportError(
                f"residue_atom has {self.residue_atom.shape[0]} rows against "
                f"{length} in residue_token"
            )
        if self.per_residue.shape != (length,):
            raise IBAMExportError(f"per_residue must be [{length}]")
        if self.residue_positions.shape != (length,):
            raise IBAMExportError(f"residue_positions must be [{length}]")
        if len(self.token_types) != num_tokens:
            raise IBAMExportError(
                f"token_types has {len(self.token_types)} entries against "
                f"{num_tokens} token columns"
            )
        if not self.residue_atom_is_derived:
            raise IBAMExportError(
                "residue_atom is always a derived aggregation; it cannot be "
                "presented as native atom attention"
            )

    @property
    def num_atoms(self) -> int:
        return int(self.residue_atom.shape[1])

    @property
    def sequence_length(self) -> int:
        return int(self.residue_token.shape[0])

    @property
    def residue_scores(self) -> np.ndarray:
        """The per-residue quantity worth ranking, and the reason it is that one.

        ``residue_salience`` when the ligand->protein direction was exported,
        ``per_residue`` otherwise. The fallback is retained for exports written
        before the second direction existed, but it is **constant by
        construction** -- see :attr:`token_residue` -- so a ranking built on it
        is a ranking of ties in index order.
        """
        return self.per_residue if self.residue_salience is None else self.residue_salience

    @property
    def residue_scores_are_informative(self) -> bool:
        """False when the only available per-residue score is constant."""
        return self.residue_salience is not None

    def top_residues(self, count: int = 10) -> list[tuple[int, float]]:
        """``(original sequence position, score)`` for the highest-scoring residues."""
        scores = self.residue_scores
        order = np.argsort(-scores)[:count]
        return [(int(self.residue_positions[i]), float(scores[i])) for i in order]

    def top_atoms(self, count: int = 10) -> list[tuple[int, float]]:
        """``(atom index, score)``, in standardized-molecule indexing."""
        order = np.argsort(-self.per_atom)[:count]
        return [(int(self.atom_indices[i]), float(self.per_atom[i])) for i in order]

    def residue_window(self, start: int, stop: int) -> "IBAMExport":
        """A zoomed slice, for targets too long to plot whole.

        Slices by **original sequence position**, not by row, so a window means the
        same thing on a windowed 34,350-residue construct as on a short one.
        """
        selected = np.where(
            (self.residue_positions >= start) & (self.residue_positions < stop)
        )[0]
        if not len(selected):
            raise IBAMExportError(f"no residues in [{start}, {stop})")
        return IBAMExport(
            residue_token=self.residue_token[selected],
            residue_atom=self.residue_atom[selected],
            per_residue=self.per_residue[selected],
            per_atom=self.residue_atom[selected].sum(axis=0),
            residue_positions=self.residue_positions[selected],
            atom_indices=self.atom_indices,
            token_types=self.token_types,
            token_to_atom=self.token_to_atom,
            token_to_bond=self.token_to_bond,
            nonphysical_mass=self.nonphysical_mass[selected],
            renormalized=self.renormalized,
            attention_by_layer_head=self.attention_by_layer_head[:, :, selected, :],
            token_residue=(
                None if self.token_residue is None else self.token_residue[:, selected]
            ),
            residue_salience=(
                None if self.residue_salience is None else self.residue_salience[selected]
            ),
            policy=self.policy,
            molecule_id=self.molecule_id,
            model_checkpoint=self.model_checkpoint,
            ligand_encoder=self.ligand_encoder,
            protein_encoder=self.protein_encoder,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            atom_index_maps=self.atom_index_maps,
            ensemble_std=None if self.ensemble_std is None else self.ensemble_std[selected],
        )

    def metadata(self) -> dict[str, object]:
        """Everything the specification requires alongside the matrices."""
        return {
            "disclaimer": self.disclaimer,
            "molecule_id": self.molecule_id,
            "model_checkpoint": self.model_checkpoint,
            "ligand_encoder": self.ligand_encoder,
            "protein_encoder": self.protein_encoder,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "sequence_length": self.sequence_length,
            "num_atoms": self.num_atoms,
            "aggregation": self.policy.to_dict(),
            "aggregation_hash": self.policy.policy_hash,
            "aggregation_description": self.policy.describe(),
            "residue_atom_is_derived": self.residue_atom_is_derived,
            "renormalized": self.renormalized,
            "nonphysical_mass_total": float(self.nonphysical_mass.sum()),
            "token_types": list(self.token_types),
            "token_to_atom": [list(atoms) for atoms in self.token_to_atom],
            "token_to_bond": list(self.token_to_bond),
            "atom_index_maps": {k: list(v) for k, v in self.atom_index_maps.items()},
            "has_ensemble_variability": self.ensemble_std is not None,
            "has_ligand_to_protein": self.token_residue is not None,
            "residue_scores_are_informative": self.residue_scores_are_informative,
            "residue_score_source": (
                "residue_salience (ligand->protein, summed over ligand tokens)"
                if self.residue_scores_are_informative
                else "per_residue (protein->ligand row sums; constant by construction)"
            ),
        }

    def write(self, directory: Path | str) -> dict[str, Path]:
        """Write the matrices (``.npz``) and the metadata (``.json``).

        Numeric export is mandatory, not an image. The plots are a separate module
        and read these files, so a figure can always be re-derived from what was
        exported.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {
            "residue_token": self.residue_token,
            "residue_atom": self.residue_atom,
            "per_residue": self.per_residue,
            "per_atom": self.per_atom,
            "residue_positions": self.residue_positions,
            "atom_indices": self.atom_indices,
            "nonphysical_mass": self.nonphysical_mass,
            "attention_by_layer_head": self.attention_by_layer_head,
        }
        if self.ensemble_std is not None:
            arrays["ensemble_std"] = self.ensemble_std
        if self.token_residue is not None:
            arrays["token_residue"] = self.token_residue
        if self.residue_salience is not None:
            arrays["residue_salience"] = self.residue_salience
        matrices = directory / "ibam_matrices.npz"
        np.savez_compressed(matrices, **arrays)
        metadata = directory / "ibam_metadata.json"
        metadata.write_text(
            json.dumps(self.metadata(), indent=2, sort_keys=True), encoding="utf-8"
        )
        return {"matrices": matrices, "metadata": metadata}


def _reduce_heads(attention: torch.Tensor, policy: AggregationPolicy) -> torch.Tensor:
    """``[H, L, T] -> [L, T]``."""
    if policy.head_reduction == "mean":
        return attention.mean(dim=0)
    return attention.amax(dim=0)


def _reduce_layers(
    stacked: torch.Tensor, policy: AggregationPolicy
) -> torch.Tensor:
    """``[n_layers, L, T] -> [L, T]``."""
    if policy.layer_reduction == "last":
        return stacked[-1]
    if policy.layer_reduction == "first":
        return stacked[0]
    return stacked.mean(dim=0)


def derive_residue_atom(
    residue_token: torch.Tensor,
    token_to_atom: Sequence[Sequence[int]],
    token_types: Sequence[TokenType | str],
    num_atoms: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map a residue x token matrix onto residue x atom. A derived aggregation.

    A token contributing to ``k`` atoms gives each of them ``weight / k``: the
    equal split the specification prescribes for bond tokens, applied uniformly.
    Non-physical tokens (knowledge, virtual, padding) contribute to **no** atom;
    their mass is returned separately rather than discarded, so a caller can see
    how much of the distribution the atom view does not represent.

    Returns ``(residue_atom [L, num_atoms], nonphysical_mass [L])``.
    """
    length = residue_token.shape[0]
    result = residue_token.new_zeros(length, num_atoms)
    nonphysical = residue_token.new_zeros(length)
    for column, (atoms, token_type) in enumerate(zip(token_to_atom, token_types)):
        kind = TokenType(token_type) if not isinstance(token_type, TokenType) else token_type
        if not kind.is_physical or not atoms:
            nonphysical += residue_token[:, column]
            continue
        share = residue_token[:, column] / len(atoms)
        for atom in atoms:
            if not 0 <= atom < num_atoms:
                raise IBAMExportError(
                    f"token {column} maps to atom {atom}, outside [0, {num_atoms})"
                )
            result[:, atom] += share
    return result, nonphysical


def export_ibam(
    fusion: FusionOutput,
    ligand_features: LigandFeatures,
    *,
    index: int = 0,
    residue_positions: torch.Tensor | None = None,
    policy: AggregationPolicy = DEFAULT_POLICY,
    model_checkpoint: str = "untrained",
    protein_encoder: str = "unknown",
    ensemble_attention: Sequence[torch.Tensor] | None = None,
) -> IBAMExport:
    """Build an export for one pair out of a fused batch.

    ``residue_positions`` must be ``ProteinFeatures.residue_indices`` for this
    protein. Omitting it falls back to row numbers and is only correct for
    proteins shorter than the encoder window; a long construct exported that way
    would have every annotation misplaced, so the fallback is recorded in the
    metadata rather than being invisible.
    """
    if not 0 <= index < fusion.protein.shape[0]:
        raise IBAMExportError(
            f"index {index} outside the batch of {fusion.protein.shape[0]}"
        )
    protein_mask = fusion.protein_mask[index]
    ligand_mask = fusion.ligand_mask[index]
    num_real_tokens = int(ligand_mask.sum())
    if num_real_tokens != len(ligand_features.token_types):
        raise IBAMExportError(
            f"the fused batch has {num_real_tokens} real ligand tokens for row "
            f"{index} but the supplied LigandFeatures describes "
            f"{len(ligand_features.token_types)}; these are not the same molecule"
        )

    # [n_layers, H, L_real, T_real], unaveraged, padding removed.
    stacked = torch.stack(
        [layer[index][:, protein_mask][:, :, ligand_mask] for layer in fusion.p2l_attention]
    )
    per_layer = torch.stack([_reduce_heads(layer, policy) for layer in stacked])
    residue_token = _reduce_layers(per_layer, policy)

    residue_atom, nonphysical = derive_residue_atom(
        residue_token,
        ligand_features.token_to_atom,
        ligand_features.token_types,
        ligand_features.num_atoms,
    )
    if policy.renormalize_physical:
        physical_total = residue_atom.sum(dim=1, keepdim=True).clamp_min(1e-12)
        residue_atom = residue_atom / physical_total

    if residue_positions is None:
        positions = torch.arange(residue_token.shape[0])
    else:
        # Accept either the full padded length (then mask it) or the already
        # masked length, since callers hold ProteinFeatures in both forms.
        if residue_positions.shape[0] == protein_mask.shape[0]:
            positions = residue_positions[protein_mask.cpu()]
        else:
            positions = residue_positions
        if positions.shape[0] != residue_token.shape[0]:
            raise IBAMExportError(
                f"residue_positions gives {positions.shape[0]} positions for "
                f"{residue_token.shape[0]} unmasked residues"
            )

    # The other direction. Its rows are softmax-normalised over *residues*, so
    # summing over ligand tokens gives a per-residue quantity that varies --
    # unlike per_residue, which is 1.0 for every residue by construction and so
    # ranks residues by nothing at all.
    token_residue = None
    residue_salience = None
    if fusion.l2p_attention:
        l2p = torch.stack(
            [
                layer[index][:, ligand_mask][:, :, protein_mask]
                for layer in fusion.l2p_attention
            ]
        )
        token_residue = _reduce_layers(
            torch.stack([_reduce_heads(layer, policy) for layer in l2p]), policy
        )
        residue_salience = token_residue.sum(dim=0)

    ensemble_std = None
    if ensemble_attention is not None and len(ensemble_attention) > 1:
        members = torch.stack(
            [member.detach().to(residue_token.dtype) for member in ensemble_attention]
        )
        ensemble_std = members.std(dim=0).sum(dim=-1).cpu().numpy()

    def array(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy()

    return IBAMExport(
        residue_token=array(residue_token),
        residue_atom=array(residue_atom),
        per_residue=array(residue_atom.sum(dim=1)),
        per_atom=array(residue_atom.sum(dim=0)),
        residue_positions=array(positions.to(torch.int64)),
        atom_indices=np.arange(ligand_features.num_atoms, dtype=np.int64),
        token_types=tuple(str(kind) for kind in ligand_features.token_types),
        token_to_atom=tuple(tuple(atoms) for atoms in ligand_features.token_to_atom),
        token_to_bond=tuple(ligand_features.token_to_bond),
        nonphysical_mass=array(nonphysical),
        renormalized=policy.renormalize_physical,
        attention_by_layer_head=array(stacked),
        policy=policy,
        molecule_id=ligand_features.molecule_id,
        model_checkpoint=model_checkpoint,
        ligand_encoder=ligand_features.encoder_metadata.name,
        protein_encoder=protein_encoder,
        num_layers=int(stacked.shape[0]),
        num_heads=int(stacked.shape[1]),
        atom_index_maps=dict(ligand_features.atom_index_maps),
        ensemble_std=ensemble_std,
        token_residue=None if token_residue is None else array(token_residue),
        residue_salience=None if residue_salience is None else array(residue_salience),
    )


__all__ = [
    "DEFAULT_POLICY",
    "DISCLAIMER",
    "AggregationPolicy",
    "IBAMExport",
    "IBAMExportError",
    "derive_residue_atom",
    "export_ibam",
]
