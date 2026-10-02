"""KPGT's triplet-graph featurization, ported from DGL/dgllife to plain RDKit.

The reference (``src/data/featurizer.py::smiles_to_graph_tune``) uses DGL only as
a container for arrays it has already built, and ``dgllife`` only for a dozen
one-hot helper functions. Neither is reachable on this GPU, and neither is doing
anything that needs a graph library, so both are transcribed here.

``tests/test_kpgt_port.py`` checks the output against the reference running in the
CPU oracle environment -- graph structure, every feature column, and the atom
mapping -- on real curated molecules, not only on hand-picked ones.

## What a node is

A **triplet**: one per bond, holding that bond's two atom feature vectors and the
bond's own. Unbonded atoms get a triplet whose second atom is a learned
placeholder. Then ``n_virtual_nodes`` knowledge nodes are appended, carrying the
fingerprint and the descriptor vector.

This is why KPGT suits iBAM better than an atom-level encoder: a token *is* a
bond, so the specification's residue x bond heatmap is native and the residue x
atom view derives through a two-atom mapping that the data actually contains.
:attr:`KPGTGraph.triplet_to_atoms` carries that mapping, already mapped back
through the canonical renumbering to original atom indices.

## Reference quirks preserved

* ``RenumberAtoms(mol, CanonicalRankAtoms(mol))`` passes *ranks* where RDKit
  expects an *ordering*. Those are different things, but the result is still a
  permutation, so the reference is self-consistent and the pretrained weights
  were fitted under it. Reproduced, and inverted explicitly so atom indices can
  be mapped back.
* The knowledge nodes are wired to the first ``len(nodes) - n`` triplets, which
  means the second knowledge node is **not** connected to the first. Reproduced.
* ``one_hot_encoding`` appends ``None`` to the allowable set when
  ``encode_unknown`` is set, so an unseen value lands in a real trailing slot
  rather than producing an all-zero block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Any, Sequence

import numpy as np
import torch
from rdkit import Chem

from bindra.encoders.kpgt_light import KPGTBatch

#: Reference sentinels.
VIRTUAL_ATOM_INDICATOR = -1
VIRTUAL_ATOM_FEATURE_PLACEHOLDER = -1
VIRTUAL_BOND_FEATURE_PLACEHOLDER = -1
VIRTUAL_PATH_INDICATOR = -1_000_000

D_ATOM_FEATS = 137
D_BOND_FEATS = 14

FEATURE_SCHEMA_VERSION = "kpgt-triplet-v1"


class KPGTGraphError(ValueError):
    """The molecule could not be turned into a triplet graph."""


def one_hot_encoding(
    value: Any, allowable_set: list[Any], encode_unknown: bool = False
) -> list[bool]:
    """Transcribed from ``dgllife.utils.one_hot_encoding``, quirk included.

    With ``encode_unknown`` the allowable set gains a trailing ``None`` slot and
    an unseen value is mapped onto it, so the block is never all-zero. That
    differs from the more common "no bit set" convention and changes the feature
    layout, so it is reproduced rather than simplified.
    """
    if encode_unknown and allowable_set[-1] is not None:
        allowable_set = [*allowable_set, None]
    if encode_unknown and value not in allowable_set:
        value = None
    return [value == item for item in allowable_set]


def atom_features(atom: Chem.Atom) -> list[float]:
    """The 137-dimensional atom vector, block for block as the reference builds it."""
    return [
        *one_hot_encoding(atom.GetAtomicNum(), list(range(1, 101)), True),  # 101
        *one_hot_encoding(atom.GetDegree(), list(range(11)), True),  # 12
        atom.GetFormalCharge(),  # 1
        *one_hot_encoding(atom.GetNumRadicalElectrons(), list(range(5)), True),  # 6
        *one_hot_encoding(
            atom.GetHybridization(),
            [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
            ],
            True,
        ),  # 6
        atom.GetIsAromatic(),  # 1
        *one_hot_encoding(atom.GetTotalNumHs(), list(range(5)), True),  # 6
        atom.HasProp("_ChiralityPossible"),  # 1
        *_chirality_type(atom),  # 2
        atom.GetMass() * 0.01,  # 1
    ]


def _chirality_type(atom: Chem.Atom) -> list[bool]:
    if not atom.HasProp("_CIPCode"):
        return [False, False]
    return one_hot_encoding(atom.GetProp("_CIPCode"), ["R", "S"], False)


def bond_features(bond: Chem.Bond) -> list[float]:
    """The 14-dimensional bond vector."""
    return [
        *one_hot_encoding(
            bond.GetBondType(),
            [
                Chem.rdchem.BondType.SINGLE,
                Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE,
                Chem.rdchem.BondType.AROMATIC,
            ],
            True,
        ),  # 5
        bond.GetIsConjugated(),  # 1
        bond.IsInRing(),  # 1
        *one_hot_encoding(
            bond.GetStereo(),
            [
                Chem.rdchem.BondStereo.STEREONONE,
                Chem.rdchem.BondStereo.STEREOANY,
                Chem.rdchem.BondStereo.STEREOZ,
                Chem.rdchem.BondStereo.STEREOE,
                Chem.rdchem.BondStereo.STEREOCIS,
                Chem.rdchem.BondStereo.STEREOTRANS,
            ],
            True,
        ),  # 7
    ]


@dataclass(frozen=True)
class KPGTPolicy:
    """Featurization settings. Defaults are the published ones."""

    max_length: int = 5
    n_virtual_nodes: int = 2
    add_self_loop: bool = True
    fp_size: int = 512
    fp_min_path: int = 1
    fp_max_path: int = 7

    def to_dict(self) -> dict[str, object]:
        return {
            "max_length": self.max_length,
            "n_virtual_nodes": self.n_virtual_nodes,
            "add_self_loop": self.add_self_loop,
            "fp_size": self.fp_size,
            "fp_min_path": self.fp_min_path,
            "fp_max_path": self.fp_max_path,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
        }


DEFAULT_POLICY = KPGTPolicy()


@dataclass(frozen=True)
class KPGTGraph:
    """One molecule's triplet graph, plus the mapping iBAM needs."""

    begin_end: np.ndarray  # [N, 2, 137]
    edge_feats: np.ndarray  # [N, 14]
    vavn: np.ndarray  # [N]
    edge_index: np.ndarray  # [2, E]
    path: np.ndarray  # [E, max_length]
    line_graph_path: np.ndarray  # [E] bool
    mol_graph_path: np.ndarray  # [E] bool
    virtual_path: np.ndarray  # [E] bool
    self_loop: np.ndarray  # [E] bool
    #: Original-molecule atom indices covered by each triplet node: two for a
    #: bond, one for an unbonded atom, none for a knowledge node.
    triplet_to_atoms: tuple[tuple[int, ...], ...]
    #: ``canonical_order[i]`` is the original index of the atom at new position i.
    canonical_order: tuple[int, ...]
    num_atoms: int
    smiles: str

    @property
    def num_nodes(self) -> int:
        return int(self.begin_end.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def __post_init__(self) -> None:
        if len(self.triplet_to_atoms) != self.num_nodes:
            raise KPGTGraphError(
                f"triplet_to_atoms has {len(self.triplet_to_atoms)} entries for "
                f"{self.num_nodes} nodes"
            )


def featurize_kpgt(smiles: str, policy: KPGTPolicy = DEFAULT_POLICY) -> KPGTGraph:
    """Build the triplet graph for one molecule. No DGL, no dgllife, no networkx."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise KPGTGraphError(f"RDKit could not parse SMILES: {smiles!r}")
    if mol.GetNumAtoms() == 0:
        raise KPGTGraphError(f"molecule has no atoms: {smiles!r}")

    # reference quirk: ranks passed where an ordering is expected. Kept, because
    # the pretrained weights were fitted on graphs built this way.
    new_order = list(Chem.rdmolfiles.CanonicalRankAtoms(mol))
    mol = Chem.rdmolops.RenumberAtoms(mol, new_order)

    n_atoms = mol.GetNumAtoms()
    per_atom = [atom_features(mol.GetAtomWithIdx(i)) for i in range(n_atoms)]

    pair_to_triplet = np.full((n_atoms, n_atoms), np.nan)
    begin_end: list[list[list[float]]] = []
    edges_feats: list[list[float]] = []
    indicators: list[int] = []
    triplet_atoms: list[tuple[int, ...]] = []

    bonded: set[int] = set()
    for triplet_id, bond in enumerate(mol.GetBonds()):
        begin, end = sorted((bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()))
        begin_end.append([per_atom[begin], per_atom[end]])
        edges_feats.append(bond_features(bond))
        indicators.append(0)
        triplet_atoms.append((new_order[begin], new_order[end]))
        bonded.add(begin)
        bonded.add(end)
        pair_to_triplet[begin, end] = pair_to_triplet[end, begin] = triplet_id

    for atom_id in range(n_atoms):
        if atom_id not in bonded:
            begin_end.append(
                [per_atom[atom_id], [VIRTUAL_ATOM_FEATURE_PLACEHOLDER] * D_ATOM_FEATS]
            )
            edges_feats.append([VIRTUAL_BOND_FEATURE_PLACEHOLDER] * D_BOND_FEATS)
            indicators.append(VIRTUAL_ATOM_INDICATOR)
            triplet_atoms.append((new_order[atom_id],))

    edge_list: list[tuple[int, int]] = []
    paths: list[list[int]] = []
    line_graph: list[int] = []
    mol_graph: list[int] = []
    virtual: list[int] = []
    self_loops: list[int] = []
    pad = policy.max_length - 2

    # Triplets sharing an atom are adjacent: the line graph of the molecule.
    for atom_id in range(n_atoms):
        node_ids = pair_to_triplet[atom_id]
        node_ids = node_ids[~np.isnan(node_ids)].astype(np.int64)
        if len(node_ids) >= 2:
            for a, b in permutations(node_ids.tolist(), 2):
                edge_list.append((a, b))
                paths.append([a] + [VIRTUAL_PATH_INDICATOR] * pad + [b])
                line_graph.append(1)
                mol_graph.append(0)
                virtual.append(0)
                self_loops.append(0)

    # Longer through-bond paths, up to max_length.
    for atom_path in _shortest_paths(mol, policy.max_length + 1):
        if not 3 < len(atom_path) <= policy.max_length + 1:
            continue
        triplet_ids = [
            int(pair_to_triplet[atom_path[i], atom_path[i + 1]])
            for i in range(len(atom_path) - 1)
        ]
        interior = triplet_ids[1:-1]
        edge_list.append((triplet_ids[0], triplet_ids[-1]))
        paths.append(
            [triplet_ids[0]]
            + interior
            + [VIRTUAL_PATH_INDICATOR] * (policy.max_length - len(interior) - 2)
            + [triplet_ids[-1]]
        )
        line_graph.append(0)
        mol_graph.append(1)
        virtual.append(0)
        self_loops.append(0)

    # Knowledge nodes. reference quirk: node n attaches to the first
    # len(nodes) - n triplets, so the second never connects to the first.
    for n in range(policy.n_virtual_nodes):
        new_id = len(begin_end)
        for i in range(len(begin_end) - n):
            edge_list.append((new_id, i))
            edge_list.append((i, new_id))
            paths.append([new_id] + [VIRTUAL_PATH_INDICATOR] * pad + [i])
            paths.append([i] + [VIRTUAL_PATH_INDICATOR] * pad + [new_id])
            line_graph.extend((0, 0))
            mol_graph.extend((0, 0))
            virtual.extend((n + 1, n + 1))
            self_loops.extend((0, 0))
        begin_end.append(
            [
                [VIRTUAL_ATOM_FEATURE_PLACEHOLDER] * D_ATOM_FEATS,
                [VIRTUAL_ATOM_FEATURE_PLACEHOLDER] * D_ATOM_FEATS,
            ]
        )
        edges_feats.append([VIRTUAL_BOND_FEATURE_PLACEHOLDER] * D_BOND_FEATS)
        indicators.append(n + 1)
        triplet_atoms.append(())

    if policy.add_self_loop:
        for i in range(len(begin_end)):
            edge_list.append((i, i))
            paths.append([i] + [VIRTUAL_PATH_INDICATOR] * pad + [i])
            line_graph.append(0)
            mol_graph.append(0)
            virtual.append(0)
            self_loops.append(1)

    edge_array = np.array(edge_list, dtype=np.int64)
    return KPGTGraph(
        begin_end=np.array(begin_end, dtype=np.float32),
        edge_feats=np.array(edges_feats, dtype=np.float32),
        vavn=np.array(indicators, dtype=np.int64),
        edge_index=edge_array.T.copy(),
        path=np.array(paths, dtype=np.int64),
        line_graph_path=np.array(line_graph, dtype=bool),
        mol_graph_path=np.array(mol_graph, dtype=bool),
        virtual_path=np.array(virtual, dtype=bool),
        self_loop=np.array(self_loops, dtype=bool),
        triplet_to_atoms=tuple(triplet_atoms),
        canonical_order=tuple(new_order),
        num_atoms=n_atoms,
        smiles=smiles,
    )


def _shortest_paths(mol: Chem.Mol, cutoff: int):
    """All-pairs shortest paths up to ``cutoff`` atoms, in the reference's order.

    networkx, not RDKit's ``GetShortestPath``. That substitution was tried and
    **rejected on measurement**: where several shortest paths tie -- which is
    every ring system -- the two libraries pick different representatives, and on
    aspirin alone 8 of 185 edges came out different. Since a different
    representative connects a different pair of triplets, the graph the
    pretrained weights were fitted on would not be the graph being built.

    networkx costs nothing to depend on here: it is pure Python and already an
    install dependency of torch.
    """
    import networkx as nx

    adjacency = np.array(Chem.rdmolops.GetAdjacencyMatrix(mol))
    graph = nx.from_numpy_array(adjacency)
    # dict() then nested iteration, matching the reference's traversal order.
    paths = dict(nx.algorithms.all_pairs_shortest_path(graph, cutoff))
    for source in paths:
        for target in paths[source]:
            yield paths[source][target]


def rdk_fingerprint(smiles: str, policy: KPGTPolicy = DEFAULT_POLICY) -> np.ndarray:
    """The 512-bit RDKit path fingerprint KPGT's first knowledge node carries."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise KPGTGraphError(f"RDKit could not parse SMILES: {smiles!r}")
    bits = Chem.RDKFingerprint(
        mol, minPath=policy.fp_min_path, maxPath=policy.fp_max_path, fpSize=policy.fp_size
    )
    return np.array(list(bits), dtype=np.float32)


def collate_kpgt(graphs: Sequence[KPGTGraph]) -> KPGTBatch:
    """Concatenate triplet graphs, shifting edge **and path** indices.

    The reference does this in two places -- ``dgl.batch`` for the edges and
    ``preprocess_batch_light`` for the paths -- and forgetting the second is a
    silent corruption: every molecule's path features would be gathered from the
    first molecule's triplets. Doing both here keeps them impossible to separate.
    """
    if not graphs:
        raise KPGTGraphError("cannot collate an empty batch")
    offsets = np.concatenate(([0], np.cumsum([g.num_nodes for g in graphs])))

    edge_parts, path_parts = [], []
    for index, graph in enumerate(graphs):
        shift = int(offsets[index])
        edge_parts.append(graph.edge_index + shift)
        shifted = graph.path.copy()
        # Only real indices shift; the -1e6 sentinel must stay a sentinel.
        shifted[shifted >= 0] += shift
        path_parts.append(shifted)

    graph_index = np.repeat(
        np.arange(len(graphs), dtype=np.int64), [g.num_nodes for g in graphs]
    )
    return KPGTBatch(
        begin_end=torch.from_numpy(np.concatenate([g.begin_end for g in graphs])),
        edge_feats=torch.from_numpy(np.concatenate([g.edge_feats for g in graphs])),
        vavn=torch.from_numpy(np.concatenate([g.vavn for g in graphs])),
        edge_index=torch.from_numpy(np.concatenate(edge_parts, axis=1)),
        path=torch.from_numpy(np.concatenate(path_parts, axis=0)),
        virtual_path=torch.from_numpy(np.concatenate([g.virtual_path for g in graphs])),
        self_loop=torch.from_numpy(np.concatenate([g.self_loop for g in graphs])),
        graph_index=torch.from_numpy(graph_index),
        num_graphs=len(graphs),
    )


__all__ = [
    "DEFAULT_POLICY",
    "D_ATOM_FEATS",
    "D_BOND_FEATS",
    "FEATURE_SCHEMA_VERSION",
    "KPGTGraph",
    "KPGTGraphError",
    "KPGTPolicy",
    "VIRTUAL_ATOM_INDICATOR",
    "VIRTUAL_PATH_INDICATOR",
    "atom_features",
    "bond_features",
    "collate_kpgt",
    "featurize_kpgt",
    "one_hot_encoding",
    "rdk_fingerprint",
]
