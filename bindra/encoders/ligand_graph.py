"""Molecular graph featurization: RDKit molecule to tensors, with no DGL.

This reproduces the legacy 78-dimensional atom / 12-dimensional bond layout
**exactly**, because the specification's ablation 1 ("existing ESMC + GAT model
retrained on the same curated data") is only a controlled comparison if the
ligand features are the same ones the legacy model saw. `tests/test_ligand_graph.py`
asserts bit-equality against the legacy
``DeepDTA-iBAM/models/rmse_model.py::smiles_to_graph`` on real corpus molecules.

Two things are deliberately different from the legacy function:

* **The graph is sparse.** The legacy function returns a dense ``[N, N, 12]``
  bond tensor, which is quadratic in atom count: the corpus's largest molecule
  would need a 1.9 GB bond tensor per copy. Here bonds live in ``[E, 12]`` with
  an ``edge_index``, and :meth:`MolecularGraph.dense_views` reconstructs the
  legacy tensors on demand for the equivalence test.
* **Deprecated accessors are replaced.** ``GetImplicitValence()`` warns on
  RDKit 2026.03; ``GetValence(Chem.ValenceType.IMPLICIT)`` is used instead. A
  test asserts the two agree atom-by-atom, so this is a substitution, not a
  change.

DGL is not imported here and must not be. DGL's final release (2.2.1) cannot
load under a torch built for this GPU's compute capability; see
``envs/kpgt/README.md``.

## Known defects in the legacy layout, preserved on purpose

These are faithfully reproduced so the ablation stays controlled, and recorded
here so no one mistakes them for intended chemistry. ``FeaturizationQuirk``
counts how often each one actually fires, so their effect on the corpus is a
measured number rather than a worry.

1. ``UNKNOWN_SYMBOL`` -- the symbol one-hot covers only C, N, O, S, F, P, Cl,
   Br, I. Any other element (B, Si, Se, metals, ...) gets an **all-zero symbol
   block**, not an "other" bin, so boron and platinum are indistinguishable in
   that block.
2. ``GASTEIGER_FALLBACK`` / ``GASTEIGER_NAN`` -- the charge block has three
   failure paths and the legacy code treats them inconsistently. An *absent or
   unparseable* property sets the *slightly positive* bin (slot 72), encoding a
   failed computation as a real, specific charge state. A *NaN* leaves all four
   slots zero. An *infinity* bins as though it were a genuine extreme charge.
   Both quirks are counted separately so the split is visible.
3. ``SMALL_RING_COLLAPSE`` -- the bond ring-size block maps sizes 3, 4 and 5 all
   onto slot 6, with 6 on slot 7 and 7 on slot 8. The atom ring-size block
   (slots 33-36) does distinguish 3/4/5/6, so the two blocks disagree about
   what a ring size is.
4. ``ACCEPTOR_EXCLUDES_TERMINAL_OH`` -- slot 75 ("acceptor") excludes any N/O
   that bears a hydrogen and has degree 1. That removes hydroxyl and primary
   amine oxygen/nitrogen from the acceptor set, which is chemically wrong: a
   hydroxyl oxygen is a textbook acceptor as well as a donor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

#: Width of the atom feature vector. Read off :data:`ATOM_LAYOUT`, never typed twice.
ATOM_FEATURE_DIM = 78
#: Width of the bond feature vector.
BOND_FEATURE_DIM = 12

#: Bumped whenever any produced number changes. Participates in the cache key via
#: ``EncoderMetadata.feature_schema_version``, so a layout change cannot silently
#: reuse features computed under the old one.
FEATURE_SCHEMA_VERSION = "ligand-graph-legacy78-v1"

#: The symbol one-hot, in order. Anything outside this list gets no symbol bit
#: at all -- see ``FeaturizationQuirk.UNKNOWN_SYMBOL``.
ATOM_SYMBOLS: tuple[str, ...] = ("C", "N", "O", "S", "F", "P", "Cl", "Br", "I")

HYBRIDIZATIONS: tuple[Chem.HybridizationType, ...] = (
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D,
    Chem.HybridizationType.SP3D2,
)

ELECTRONEGATIVITY: Mapping[str, float] = {
    "H": 2.20, "C": 2.55, "N": 3.04, "O": 3.44, "S": 2.58, "F": 3.98,
    "P": 2.19, "Cl": 3.16, "Br": 2.96, "I": 2.66, "B": 2.04, "Si": 1.90,
    "Se": 2.55, "Na": 0.93, "K": 0.82,
}
#: Used when the element is absent from :data:`ELECTRONEGATIVITY`; lands in the
#: 2.5-3.0 bin, which is where carbon and sulfur also land.
ELECTRONEGATIVITY_DEFAULT = 2.5

COVALENT_RADII: Mapping[str, float] = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05,
    "F": 0.57, "P": 1.07, "Cl": 1.02, "Br": 1.20, "I": 1.39,
}
COVALENT_RADIUS_DEFAULT = 0.8

BOND_TYPES: Mapping[Chem.rdchem.BondType, int] = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}


class Block(StrEnum):
    """Named slot ranges in the atom vector.

    The legacy code wrote raw offsets (``feat[42 + total_val - 1] = 1.0``), which
    is how two blocks can silently overlap. Naming them lets
    :func:`verify_layout` prove the blocks tile ``[0, 78)`` exactly once.
    """

    SYMBOL = "symbol"
    DEGREE = "degree"
    IMPLICIT_VALENCE = "implicit_valence"
    FORMAL_CHARGE = "formal_charge"
    HYBRIDIZATION = "hybridization"
    AROMATIC = "aromatic"
    RING_SIZE = "ring_size"
    TOTAL_NUM_HS = "total_num_hs"
    TOTAL_VALENCE = "total_valence"
    RADICAL_ELECTRONS = "radical_electrons"
    IN_RING = "in_ring"
    HETERO_IN_RING = "hetero_in_ring"
    CHIRAL_TAG = "chiral_tag"
    CIP = "cip"
    CHIRALITY_POSSIBLE = "chirality_possible"
    ELECTRONEGATIVITY = "electronegativity"
    MASS = "mass"
    ATOMIC_NUMBER = "atomic_number"
    COVALENT_RADIUS = "covalent_radius"
    GASTEIGER = "gasteiger"
    DONOR = "donor"
    ACCEPTOR = "acceptor"
    HYDROPHOBIC = "hydrophobic"
    BASIC_NITROGEN = "basic_nitrogen"


#: ``block -> (start, width)``. Transcribed from the legacy offsets and then
#: checked by :func:`verify_layout`, which runs at import time.
ATOM_LAYOUT: Mapping[Block, tuple[int, int]] = {
    Block.SYMBOL: (0, 9),
    Block.DEGREE: (9, 7),
    Block.IMPLICIT_VALENCE: (16, 6),
    Block.FORMAL_CHARGE: (22, 5),
    Block.HYBRIDIZATION: (27, 5),
    Block.AROMATIC: (32, 1),
    Block.RING_SIZE: (33, 4),
    Block.TOTAL_NUM_HS: (37, 5),
    Block.TOTAL_VALENCE: (42, 6),
    Block.RADICAL_ELECTRONS: (48, 5),
    Block.IN_RING: (53, 1),
    Block.HETERO_IN_RING: (54, 1),
    Block.CHIRAL_TAG: (55, 3),
    Block.CIP: (58, 2),
    Block.CHIRALITY_POSSIBLE: (60, 1),
    Block.ELECTRONEGATIVITY: (61, 6),
    Block.MASS: (67, 1),
    Block.ATOMIC_NUMBER: (68, 1),
    Block.COVALENT_RADIUS: (69, 1),
    Block.GASTEIGER: (70, 4),
    Block.DONOR: (74, 1),
    Block.ACCEPTOR: (75, 1),
    Block.HYDROPHOBIC: (76, 1),
    Block.BASIC_NITROGEN: (77, 1),
}


def verify_layout() -> None:
    """Prove the declared blocks tile ``[0, ATOM_FEATURE_DIM)`` exactly once.

    An overlap would mean two chemical properties writing the same slot, which
    is invisible in a 78-long list of offsets and fatal to interpretation. This
    runs at import time, so a bad edit cannot reach a training run.
    """
    owner: list[Block | None] = [None] * ATOM_FEATURE_DIM
    for block, (start, width) in ATOM_LAYOUT.items():
        if start < 0 or start + width > ATOM_FEATURE_DIM:
            raise ValueError(f"{block} spans [{start}, {start + width}) outside the vector")
        for slot in range(start, start + width):
            if owner[slot] is not None:
                raise ValueError(f"slot {slot} claimed by both {owner[slot]} and {block}")
            owner[slot] = block
    unclaimed = [slot for slot, block in enumerate(owner) if block is None]
    if unclaimed:
        raise ValueError(f"slots {unclaimed} are claimed by no block")


verify_layout()


class FeaturizationQuirk(StrEnum):
    """Legacy behaviours that are reproduced but are defects.

    Counted per call so their prevalence on the corpus is measurable. See the
    module docstring for what each one does.
    """

    UNKNOWN_SYMBOL = "unknown_symbol"
    GASTEIGER_FALLBACK = "gasteiger_fallback"
    GASTEIGER_NAN = "gasteiger_nan"
    SMALL_RING_COLLAPSE = "small_ring_collapse"
    ACCEPTOR_EXCLUDES_TERMINAL_OH = "acceptor_excludes_terminal_oh"


class LigandGraphError(ValueError):
    """The molecule could not be turned into a graph.

    Raised rather than returning a zero graph. The legacy function returned an
    all-zero ``[1, 78]`` tensor with an identity adjacency for unparseable
    SMILES, which is a valid-looking molecule made of nothing: it trains, it
    contributes gradient, and it is indistinguishable downstream from a real
    single-atom species.
    """


@dataclass(frozen=True)
class MolecularGraph:
    """One molecule as sparse tensors, plus the bookkeeping iBAM needs.

    ``edge_index`` is ``[2, E]`` over **directed** edges, both directions of each
    RDKit bond present and adjacent (``2k`` and ``2k + 1`` for bond ``k``), so
    ``edge_to_bond`` is ``[0, 0, 1, 1, ...]``. Self-loops are *not* included
    here; the legacy dense adjacency adds them, and
    :meth:`dense_views` does so too.
    """

    atom_features: torch.Tensor  # [N, 78] float32
    edge_index: torch.Tensor  # [2, E] long, directed
    bond_features: torch.Tensor  # [E, 12] float32, one row per directed edge
    edge_to_bond: torch.Tensor  # [E] long, index of the RDKit bond
    num_atoms: int
    num_bonds: int
    smiles: str
    quirks: Mapping[FeaturizationQuirk, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.atom_features.shape != (self.num_atoms, ATOM_FEATURE_DIM):
            raise ValueError(
                f"atom_features must be [{self.num_atoms}, {ATOM_FEATURE_DIM}], "
                f"got {tuple(self.atom_features.shape)}"
            )
        num_edges = 2 * self.num_bonds
        if self.edge_index.shape != (2, num_edges):
            raise ValueError(
                f"edge_index must be [2, {num_edges}], got {tuple(self.edge_index.shape)}"
            )
        if self.bond_features.shape != (num_edges, BOND_FEATURE_DIM):
            raise ValueError(
                f"bond_features must be [{num_edges}, {BOND_FEATURE_DIM}], "
                f"got {tuple(self.bond_features.shape)}"
            )
        if self.edge_to_bond.shape != (num_edges,):
            raise ValueError(
                f"edge_to_bond must be [{num_edges}], got {tuple(self.edge_to_bond.shape)}"
            )
        if num_edges and int(self.edge_index.max()) >= self.num_atoms:
            raise ValueError(
                f"edge_index references atom {int(self.edge_index.max())} but the "
                f"molecule has {self.num_atoms} atoms"
            )

    @property
    def num_edges(self) -> int:
        """Directed edge count: twice the bond count."""
        return 2 * self.num_bonds

    def dense_views(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct the legacy ``(adjacency, edge_tensor)`` pair.

        Quadratic in atom count, so this exists for the equivalence test and for
        small-molecule inspection, not for the training path.
        """
        n = self.num_atoms
        adjacency = torch.eye(n, dtype=torch.float32)
        edges = torch.zeros((n, n, BOND_FEATURE_DIM), dtype=torch.float32)
        src, dst = self.edge_index
        adjacency[src, dst] = 1.0
        edges[src, dst] = self.bond_features
        return adjacency, edges

    def bond_atom_pairs(self) -> tuple[tuple[int, int], ...]:
        """``(begin, end)`` atom indices per RDKit bond, in bond order.

        This is what ``LigandFeatures.token_to_atom`` needs for bond tokens, so
        an attention weight on a bond can be drawn on the right edge of a
        depiction.
        """
        src, dst = self.edge_index
        return tuple(
            (int(src[2 * bond]), int(dst[2 * bond])) for bond in range(self.num_bonds)
        )


def _ring_size_quirk_fires(mol: Chem.Mol) -> bool:
    """Does this molecule actually contain a 3-, 4- or 5-membered ring bond?

    Only then does ``SMALL_RING_COLLAPSE`` affect its features.
    """
    # One pass over the ring systems rather than bonds x sizes. The old form
    # called IsBondInRingOfSize up to three times per ring bond, which on a
    # fused polycyclic molecule is hundreds of calls to answer a yes/no question
    # the ring list already contains.
    return any(len(ring) in (3, 4, 5) for ring in mol.GetRingInfo().BondRings())


def _atom_features(
    mol: Chem.Mol, quirks: dict[FeaturizationQuirk, int]
) -> torch.Tensor:
    """The 78-dimensional atom block, slot for slot as the legacy model built it."""
    ring_info = mol.GetRingInfo()
    rows: list[list[float]] = []
    for atom in mol.GetAtoms():
        feat = [0.0] * ATOM_FEATURE_DIM
        symbol = atom.GetSymbol()

        start, _ = ATOM_LAYOUT[Block.SYMBOL]
        if symbol in ATOM_SYMBOLS:
            feat[start + ATOM_SYMBOLS.index(symbol)] = 1.0
        else:
            # No "other" bin exists; the block stays all zero.
            quirks[FeaturizationQuirk.UNKNOWN_SYMBOL] = (
                quirks.get(FeaturizationQuirk.UNKNOWN_SYMBOL, 0) + 1
            )

        start, width = ATOM_LAYOUT[Block.DEGREE]
        feat[start + min(atom.GetDegree(), width - 1)] = 1.0

        start, width = ATOM_LAYOUT[Block.IMPLICIT_VALENCE]
        implicit = atom.GetValence(Chem.ValenceType.IMPLICIT)
        feat[start + min(implicit, width - 1)] = 1.0

        # Charge is clamped to [-2, +2]: a 3+ cation is indistinguishable from 2+.
        start, width = ATOM_LAYOUT[Block.FORMAL_CHARGE]
        feat[start + max(0, min(width - 1, atom.GetFormalCharge() + 2))] = 1.0

        start, _ = ATOM_LAYOUT[Block.HYBRIDIZATION]
        hybridization = atom.GetHybridization()
        if hybridization in HYBRIDIZATIONS:
            feat[start + HYBRIDIZATIONS.index(hybridization)] = 1.0

        feat[ATOM_LAYOUT[Block.AROMATIC][0]] = 1.0 if atom.GetIsAromatic() else 0.0

        # Not one-hot: a spiro atom in both a 5- and a 6-ring sets two slots.
        start, _ = ATOM_LAYOUT[Block.RING_SIZE]
        for offset, size in enumerate((3, 4, 5, 6)):
            if ring_info.IsAtomInRingOfSize(atom.GetIdx(), size):
                feat[start + offset] = 1.0

        start, width = ATOM_LAYOUT[Block.TOTAL_NUM_HS]
        feat[start + min(atom.GetTotalNumHs(), width - 1)] = 1.0

        start, width = ATOM_LAYOUT[Block.TOTAL_VALENCE]
        total_valence = min(max(atom.GetTotalValence(), 1), width)
        feat[start + total_valence - 1] = 1.0

        start, width = ATOM_LAYOUT[Block.RADICAL_ELECTRONS]
        feat[start + min(atom.GetNumRadicalElectrons(), width - 1)] = 1.0

        in_ring = atom.IsInRing()
        feat[ATOM_LAYOUT[Block.IN_RING][0]] = 1.0 if in_ring else 0.0
        feat[ATOM_LAYOUT[Block.HETERO_IN_RING][0]] = (
            1.0 if in_ring and symbol != "C" else 0.0
        )

        start, _ = ATOM_LAYOUT[Block.CHIRAL_TAG]
        chiral_tag = atom.GetChiralTag()
        if chiral_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            feat[start] = 1.0
        elif chiral_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            feat[start + 1] = 1.0
        elif chiral_tag == Chem.ChiralType.CHI_UNSPECIFIED:
            feat[start + 2] = 1.0

        start, _ = ATOM_LAYOUT[Block.CIP]
        if atom.HasProp("_CIPCode"):
            cip = atom.GetProp("_CIPCode")
            feat[start] = 1.0 if cip == "R" else 0.0
            feat[start + 1] = 1.0 if cip == "S" else 0.0

        feat[ATOM_LAYOUT[Block.CHIRALITY_POSSIBLE][0]] = (
            1.0 if atom.HasProp("_ChiralityPossible") else 0.0
        )

        # Six half-open electronegativity bins; the last catches >= 4.0.
        start, _ = ATOM_LAYOUT[Block.ELECTRONEGATIVITY]
        en = ELECTRONEGATIVITY.get(symbol, ELECTRONEGATIVITY_DEFAULT)
        for offset, bound in enumerate((2.0, 2.5, 3.0, 3.5, 4.0)):
            if en < bound:
                feat[start + offset] = 1.0
                break
        else:
            feat[start + 5] = 1.0

        feat[ATOM_LAYOUT[Block.MASS][0]] = atom.GetMass() / 200.0
        feat[ATOM_LAYOUT[Block.ATOMIC_NUMBER][0]] = atom.GetAtomicNum() / 53.0
        feat[ATOM_LAYOUT[Block.COVALENT_RADIUS][0]] = (
            COVALENT_RADII.get(symbol, COVALENT_RADIUS_DEFAULT) / 1.5
        )

        # Three distinct outcomes, and the legacy code treats them differently.
        # Absent or unparseable lands in the "slightly positive" bin; NaN leaves
        # the whole block zero; an infinity bins as if it were a real extreme.
        start, _ = ATOM_LAYOUT[Block.GASTEIGER]
        charge = _gasteiger_charge(atom)
        if charge is None:
            feat[start + 2] = 1.0
            quirks[FeaturizationQuirk.GASTEIGER_FALLBACK] = (
                quirks.get(FeaturizationQuirk.GASTEIGER_FALLBACK, 0) + 1
            )
        elif charge != charge:  # NaN
            quirks[FeaturizationQuirk.GASTEIGER_NAN] = (
                quirks.get(FeaturizationQuirk.GASTEIGER_NAN, 0) + 1
            )
        else:
            for offset, bound in enumerate((-0.25, 0.0, 0.25)):
                if charge < bound:
                    feat[start + offset] = 1.0
                    break
            else:
                feat[start + 3] = 1.0

        num_hs = atom.GetTotalNumHs()
        is_donor = symbol in ("N", "O") and num_hs > 0
        feat[ATOM_LAYOUT[Block.DONOR][0]] = 1.0 if is_donor else 0.0

        # Slot 75: excludes terminal N-H / O-H, which is chemically wrong.
        terminal_donor = symbol in ("N", "O") and num_hs > 0 and atom.GetDegree() == 1
        if symbol in ("N", "O", "F") and not terminal_donor:
            feat[ATOM_LAYOUT[Block.ACCEPTOR][0]] = 1.0
        if terminal_donor:
            quirks[FeaturizationQuirk.ACCEPTOR_EXCLUDES_TERMINAL_OH] = (
                quirks.get(FeaturizationQuirk.ACCEPTOR_EXCLUDES_TERMINAL_OH, 0) + 1
            )

        hydrophobic = False
        if symbol == "C":
            hydrophobic = not any(
                neighbor.GetSymbol() in ("N", "O", "S", "P")
                for neighbor in atom.GetNeighbors()
            )
        elif symbol in ("F", "Cl", "Br", "I"):
            hydrophobic = True
        feat[ATOM_LAYOUT[Block.HYDROPHOBIC][0]] = 1.0 if hydrophobic else 0.0

        if symbol == "N" and atom.GetFormalCharge() >= 0:
            if num_hs > 0 or atom.GetFormalCharge() > 0:
                feat[ATOM_LAYOUT[Block.BASIC_NITROGEN][0]] = 1.0

        rows.append(feat)
    # np.asarray on a list of lists is markedly faster than torch.tensor, which
    # walks the nested Python structure element by element.
    return torch.from_numpy(
        np.asarray(rows, dtype=np.float32).reshape(mol.GetNumAtoms(), ATOM_FEATURE_DIM)
    )


def _gasteiger_charge(atom: Chem.Atom) -> float | None:
    """The Gasteiger charge, or ``None`` if the property is absent/unparseable.

    NaN and the infinities Gasteiger emits on pathological valences are returned
    **as-is**, because the legacy code routes them differently from an absent
    property and this port is bit-exact. The caller distinguishes the three.
    """
    if not atom.HasProp("_GasteigerCharge"):
        return None
    try:
        return float(atom.GetProp("_GasteigerCharge"))
    except (ValueError, TypeError):
        return None


def _bond_features(mol: Chem.Mol) -> torch.Tensor:
    """``[num_bonds, 12]`` in RDKit bond order."""
    ring_info = mol.GetRingInfo()
    rows: list[list[float]] = []
    for bond in mol.GetBonds():
        feat = [0.0] * BOND_FEATURE_DIM
        bond_type = bond.GetBondType()
        if bond_type in BOND_TYPES:
            feat[BOND_TYPES[bond_type]] = 1.0
        feat[4] = 1.0 if bond.GetIsConjugated() else 0.0
        in_ring = bond.IsInRing()
        feat[5] = 1.0 if in_ring else 0.0
        if in_ring:
            # Sizes 3, 4 and 5 all land on slot 6; see SMALL_RING_COLLAPSE.
            for size, slot in ((3, 6), (4, 6), (5, 6), (6, 7), (7, 8)):
                if ring_info.IsBondInRingOfSize(bond.GetIdx(), size):
                    feat[slot] = 1.0
                    break
        feat[9] = (
            1.0 if bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE else 0.0
        )
        feat[10] = bond.GetBondTypeAsDouble() / 3.0
        feat[11] = (
            1.0
            if (
                bond_type == Chem.rdchem.BondType.SINGLE
                and not in_ring
                and bond.GetBeginAtom().GetDegree() > 1
                and bond.GetEndAtom().GetDegree() > 1
            )
            else 0.0
        )
        rows.append(feat)
    return torch.from_numpy(
        np.asarray(rows, dtype=np.float32).reshape(mol.GetNumBonds(), BOND_FEATURE_DIM)
    )


def featurize_molecule(molecule: str | Chem.Mol) -> MolecularGraph:
    """Featurize one molecule. Raises :class:`LigandGraphError` on failure.

    The input is expected to be already standardized by
    ``bindra.data.bindingdb.standardize``; this function does no fragment
    selection, no uncharging and no tautomer handling, so that the standardizer
    stays the single place where chemistry policy lives.
    """
    if isinstance(molecule, str):
        mol = Chem.MolFromSmiles(molecule)
        if mol is None:
            raise LigandGraphError(f"RDKit could not parse SMILES: {molecule!r}")
        smiles = molecule
    else:
        mol = molecule
        smiles = Chem.MolToSmiles(mol)
    if mol.GetNumAtoms() == 0:
        raise LigandGraphError(f"molecule has no atoms: {smiles!r}")

    # Charges are a per-atom property set in place, so this must precede
    # featurization. Failure is tolerated and shows up as GASTEIGER_FALLBACK.
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:  # noqa: BLE001 - RDKit raises assorted types here
        pass

    quirks: dict[FeaturizationQuirk, int] = {}
    if _ring_size_quirk_fires(mol):
        quirks[FeaturizationQuirk.SMALL_RING_COLLAPSE] = 1

    atom_features = _atom_features(mol, quirks)
    per_bond = _bond_features(mol)

    num_bonds = mol.GetNumBonds()
    # Built in numpy and converted once. Element-wise assignment into a torch
    # tensor costs a dispatch per write: four per bond, which on a 38-bond
    # molecule is 152 dispatches to move 152 integers.
    ends = np.empty((num_bonds, 2), dtype=np.int64)
    for index, bond in enumerate(mol.GetBonds()):
        ends[index, 0] = bond.GetBeginAtomIdx()
        ends[index, 1] = bond.GetEndAtomIdx()
    pairs = np.empty((2 * num_bonds, 2), dtype=np.int64)
    pairs[0::2] = ends
    pairs[1::2] = ends[:, ::-1]
    edge_index = torch.from_numpy(np.ascontiguousarray(pairs.T))
    edge_to_bond = torch.from_numpy(np.repeat(np.arange(num_bonds, dtype=np.int64), 2))
    # Each directed edge carries its bond's features; interleaving matches
    # edge_index so row 2k and 2k+1 are the two directions of bond k.
    bond_features = per_bond.repeat_interleave(2, dim=0) if num_bonds else per_bond.reshape(
        0, BOND_FEATURE_DIM
    )

    return MolecularGraph(
        atom_features=atom_features,
        edge_index=edge_index,
        bond_features=bond_features,
        edge_to_bond=edge_to_bond,
        num_atoms=mol.GetNumAtoms(),
        num_bonds=num_bonds,
        smiles=smiles,
        quirks=dict(quirks),
    )


@dataclass(frozen=True)
class BatchedGraphs:
    """Several molecules packed for one GPU forward pass.

    Atoms are **concatenated**, not padded: padding to the batch maximum wastes
    work proportional to the spread of molecule sizes, and the corpus spread is
    wide. ``batch_index`` says which molecule each atom belongs to, which is all
    a segment-sum readout needs. ``atom_offsets`` lets a caller slice one
    molecule's rows back out.
    """

    atom_features: torch.Tensor  # [sum_N, 78]
    edge_index: torch.Tensor  # [2, sum_E] into the concatenated atom axis
    bond_features: torch.Tensor  # [sum_E, 12]
    batch_index: torch.Tensor  # [sum_N] long, molecule id per atom
    atom_offsets: torch.Tensor  # [B + 1] long, prefix sums of atom counts
    num_graphs: int
    #: Largest atom count in the batch, as a Python int. Precomputed here
    #: because the alternative, ``int(counts.max())`` inside the forward pass,
    #: is a device-to-host synchronisation on every step. Collation already runs
    #: on the CPU over numpy, so it costs nothing to carry.
    max_atoms: int = 0
    #: Row each atom occupies in a padded ``[B, max_atoms, D]`` tensor, flattened:
    #: ``batch_index * max_atoms + position_within_molecule``. Also precomputed:
    #: deriving it on device needs ``repeat_interleave`` with a tensor of
    #: repeats, which must learn its own output size and therefore also syncs.
    token_destination: torch.Tensor | None = None

    @property
    def total_atoms(self) -> int:
        return int(self.atom_features.shape[0])

    def to(self, device: torch.device | str) -> "BatchedGraphs":
        return BatchedGraphs(
            atom_features=self.atom_features.to(device),
            edge_index=self.edge_index.to(device),
            bond_features=self.bond_features.to(device),
            batch_index=self.batch_index.to(device),
            atom_offsets=self.atom_offsets.to(device),
            num_graphs=self.num_graphs,
            max_atoms=self.max_atoms,
            token_destination=(
                None if self.token_destination is None
                else self.token_destination.to(device)
            ),
        )

    def graph_slice(self, index: int) -> slice:
        """Row range of molecule ``index`` on the concatenated atom axis."""
        return slice(int(self.atom_offsets[index]), int(self.atom_offsets[index + 1]))


def collate_graphs(graphs: Sequence[MolecularGraph]) -> BatchedGraphs:
    """Concatenate graphs, shifting each one's edge indices by its atom offset.

    Forgetting the shift is the classic error here: every molecule's edges would
    point into the first molecule's atoms, which trains without complaint and
    produces nonsense.
    """
    if not graphs:
        raise ValueError("cannot collate an empty batch")
    counts = [graph.num_atoms for graph in graphs]
    offsets = torch.zeros(len(graphs) + 1, dtype=torch.long)
    offsets[1:] = torch.tensor(counts, dtype=torch.long).cumsum(0)

    edge_parts = [
        graph.edge_index + int(offsets[index])
        for index, graph in enumerate(graphs)
        if graph.num_edges
    ]
    edge_index = (
        torch.cat(edge_parts, dim=1) if edge_parts else torch.empty((2, 0), dtype=torch.long)
    )
    bond_parts = [graph.bond_features for graph in graphs if graph.num_edges]
    bond_features = (
        torch.cat(bond_parts, dim=0)
        if bond_parts
        else torch.empty((0, BOND_FEATURE_DIM), dtype=torch.float32)
    )
    count_array = np.asarray(counts, dtype=np.int64)
    batch_index_np = np.repeat(np.arange(len(graphs), dtype=np.int64), count_array)
    max_atoms = int(count_array.max())
    # Position of each atom inside its own molecule, then its row in a padded
    # [B, max_atoms, D] view. Done here, in numpy, so the forward pass never has
    # to ask the GPU a question it must stop to answer.
    within = np.arange(len(batch_index_np), dtype=np.int64) - np.repeat(
        offsets[:-1].numpy(), count_array
    )
    token_destination = torch.from_numpy(batch_index_np * max_atoms + within)
    batch_index = torch.from_numpy(batch_index_np)
    return BatchedGraphs(
        atom_features=torch.cat([graph.atom_features for graph in graphs], dim=0),
        edge_index=edge_index,
        bond_features=bond_features,
        batch_index=batch_index,
        atom_offsets=offsets,
        num_graphs=len(graphs),
        max_atoms=max_atoms,
        token_destination=token_destination,
    )


def summarize_quirks(graphs: Iterable[MolecularGraph]) -> dict[str, int]:
    """Total each legacy quirk over a collection, for the curation report."""
    totals: dict[str, int] = {quirk.value: 0 for quirk in FeaturizationQuirk}
    for graph in graphs:
        for quirk, count in graph.quirks.items():
            totals[quirk.value] += count
    return totals


__all__ = [
    "ATOM_FEATURE_DIM",
    "ATOM_LAYOUT",
    "ATOM_SYMBOLS",
    "BOND_FEATURE_DIM",
    "BatchedGraphs",
    "Block",
    "FEATURE_SCHEMA_VERSION",
    "FeaturizationQuirk",
    "LigandGraphError",
    "MolecularGraph",
    "collate_graphs",
    "featurize_molecule",
    "summarize_quirks",
    "verify_layout",
]
