"""Molecules as dense category tensors, and back.

The diffusion works on ``[N, A]`` atom one-hots and ``[N, N, E]`` bond
one-hots, which is a different representation from the line graph the affinity
arm uses: there a *bond* is the token, here an *atom* is, because generation has
to decide the atoms as well as the bonds between them.

**The vocabularies are fixed here, not inferred.** An inferred vocabulary
changes width between the training set and whatever is generated later, and a
checkpoint stops loading. Measured on 40,000 standardized molecules from the
202610 snapshot: nine elements cover 99.996% of atoms, and four bond types
cover all bonds. Everything outside the vocabulary is refused rather than mapped
to a nearest neighbour, because there is no nearest neighbour among categories.

Sizes from the same sample: median 28 atoms, p95 46, p99 57, max 128. The
default cap of 64 keeps p99 and costs a 64x64 bond tensor per molecule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: The "there is no atom here" symbol. Declared rather than spelled inline,
#: because the decoder has to recognise it and a mismatch between the two
#: spellings reaches RDKit as `Element '<pad>' not found`.
PAD_SYMBOL = "<pad>"

#: Index 0 is the padding / absent atom, so a padded position is a real
#: category the model can predict rather than a masked hole.
ATOM_VOCABULARY: tuple[str, ...] = (
    PAD_SYMBOL, "C", "N", "O", "S", "F", "Cl", "Br", "P", "I", "B",
)

#: Index 0 is "no bond", which is the overwhelming majority of any pair matrix
#: and the category the diagonal always takes.
BOND_VOCABULARY: tuple[str, ...] = (
    "none", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC",
)

#: Formal charge is carried as a small separate category rather than folded
#: into the element, which would multiply the atom vocabulary by five for the
#: 0.3% of atoms that are charged.
CHARGE_VOCABULARY: tuple[int, ...] = (0, 1, -1, 2, -2)

N_ATOM_TYPES = len(ATOM_VOCABULARY)
N_BOND_TYPES = len(BOND_VOCABULARY)
N_CHARGE_TYPES = len(CHARGE_VOCABULARY)

#: p99 of the corpus is 57 atoms. Molecules above the cap are skipped and
#: counted, never silently truncated -- a truncated molecule is a different
#: molecule, and training on one teaches the model to end rings in mid-air.
DEFAULT_MAX_ATOMS = 64


class GraphEncodingError(ValueError):
    """A molecule cannot be represented in the declared vocabulary."""


@dataclass(frozen=True)
class DenseGraph:
    """One molecule as padded category indices."""

    atoms: np.ndarray  # [N] int64, index into ATOM_VOCABULARY
    bonds: np.ndarray  # [N, N] int64, index into BOND_VOCABULARY
    charges: np.ndarray  # [N] int64
    mask: np.ndarray  # [N] bool, True = real atom
    n_atoms: int

    def __post_init__(self) -> None:
        n = len(self.atoms)
        if self.bonds.shape != (n, n):
            raise GraphEncodingError(
                f"bonds must be [{n}, {n}], got {self.bonds.shape}"
            )
        if not np.array_equal(self.bonds, self.bonds.T):
            raise GraphEncodingError("bond matrix is not symmetric")


def encode_molecule(
    smiles: str, *, max_atoms: int = DEFAULT_MAX_ATOMS
) -> DenseGraph:
    """SMILES to padded category indices. Raises on anything out of vocabulary."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise GraphEncodingError(f"RDKit could not parse {smiles!r}")
    n_real = mol.GetNumAtoms()
    if n_real == 0:
        raise GraphEncodingError(f"{smiles!r} has no atoms")
    if n_real > max_atoms:
        raise GraphEncodingError(
            f"{smiles!r} has {n_real} atoms, above the {max_atoms} cap. Skipped "
            "rather than truncated: a truncated molecule is a different molecule."
        )

    atom_index = {s: i for i, s in enumerate(ATOM_VOCABULARY)}
    bond_index = {s: i for i, s in enumerate(BOND_VOCABULARY)}
    charge_index = {c: i for i, c in enumerate(CHARGE_VOCABULARY)}

    atoms = np.zeros(max_atoms, dtype=np.int64)
    charges = np.zeros(max_atoms, dtype=np.int64)
    bonds = np.zeros((max_atoms, max_atoms), dtype=np.int64)
    mask = np.zeros(max_atoms, dtype=bool)

    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in atom_index:
            raise GraphEncodingError(
                f"{smiles!r} contains {symbol!r}, outside the declared vocabulary "
                f"{ATOM_VOCABULARY[1:]}. Mapping it to a nearest element would be "
                "inventing chemistry; the molecule is skipped instead."
            )
        charge = atom.GetFormalCharge()
        if charge not in charge_index:
            raise GraphEncodingError(f"{smiles!r} has formal charge {charge}, out of range")
        i = atom.GetIdx()
        atoms[i] = atom_index[symbol]
        charges[i] = charge_index[charge]
        mask[i] = True

    for bond in mol.GetBonds():
        kind = str(bond.GetBondType())
        if kind not in bond_index:
            raise GraphEncodingError(f"{smiles!r} has bond type {kind!r}, out of vocabulary")
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonds[i, j] = bonds[j, i] = bond_index[kind]

    return DenseGraph(atoms=atoms, bonds=bonds, charges=charges, mask=mask, n_atoms=n_real)


def scaffold_mask(smiles: str, *, max_atoms: int = DEFAULT_MAX_ATOMS) -> np.ndarray:
    """``[N]`` True where the atom belongs to the Bemis-Murcko scaffold.

    The scaffold is what generation must preserve, so this is the ``keep`` mask
    the noise process and the sampler are handed. Derived by substructure match
    of the molecule's own scaffold against itself, which gives the *atom
    indices* rather than merely the fact that a scaffold exists -- the sampler
    freezes positions, not patterns.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise GraphEncodingError(f"RDKit could not parse {smiles!r}")
    keep = np.zeros(max_atoms, dtype=bool)
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception as exc:  # noqa: BLE001 - a scaffold-less molecule is ordinary
        raise GraphEncodingError(f"no scaffold for {smiles!r}: {exc}") from exc
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        # An acyclic molecule has no Murcko scaffold. Nothing is frozen, which
        # is the honest answer rather than freezing an arbitrary fragment.
        return keep
    match = mol.GetSubstructMatch(scaffold)
    for index in match:
        if index < max_atoms:
            keep[index] = True
    return keep


def collate(graphs: "list[DenseGraph]"):
    """Stack encoded molecules into the tensors the denoiser expects."""
    import torch

    atoms = torch.from_numpy(np.stack([g.atoms for g in graphs]))
    bonds = torch.from_numpy(np.stack([g.bonds for g in graphs]))
    charges = torch.from_numpy(np.stack([g.charges for g in graphs]))
    mask = torch.from_numpy(np.stack([g.mask for g in graphs]))
    return {
        "atoms": torch.nn.functional.one_hot(atoms, N_ATOM_TYPES).float(),
        "bonds": torch.nn.functional.one_hot(bonds, N_BOND_TYPES).float(),
        "charges": torch.nn.functional.one_hot(charges, N_CHARGE_TYPES).float(),
        "mask": mask,
    }


def corpus_marginals(smiles: "list[str]", *, max_atoms: int = DEFAULT_MAX_ATOMS):
    """Empirical atom and bond marginals, for the noise process to decay toward.

    Uniform noise over atom types makes the model spend its capacity
    rediscovering that molecules are mostly carbon. Starting from the corpus
    marginal means the fully-noised graph is already element-plausible.
    """
    import torch

    from bindra.generation.digress.noise import CategoricalMarginals

    atom_counts = np.zeros(N_ATOM_TYPES, dtype=np.int64)
    bond_counts = np.zeros(N_BOND_TYPES, dtype=np.int64)
    used = 0
    for one in smiles:
        try:
            graph = encode_molecule(one, max_atoms=max_atoms)
        except GraphEncodingError:
            continue
        used += 1
        atom_counts += np.bincount(graph.atoms[graph.mask], minlength=N_ATOM_TYPES)
        real = graph.mask
        sub = graph.bonds[np.ix_(real, real)]
        bond_counts += np.bincount(sub.reshape(-1), minlength=N_BOND_TYPES)
    if not used:
        raise GraphEncodingError("no molecule in the sample could be encoded")
    return CategoricalMarginals.from_counts(
        torch.from_numpy(atom_counts), torch.from_numpy(bond_counts)
    ), used


__all__ = [
    "ATOM_VOCABULARY",
    "PAD_SYMBOL",
    "BOND_VOCABULARY",
    "CHARGE_VOCABULARY",
    "DEFAULT_MAX_ATOMS",
    "N_ATOM_TYPES",
    "N_BOND_TYPES",
    "N_CHARGE_TYPES",
    "DenseGraph",
    "GraphEncodingError",
    "collate",
    "corpus_marginals",
    "encode_molecule",
    "scaffold_mask",
]
