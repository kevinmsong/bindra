"""Ligand graph featurization, checked bit-for-bit against the legacy model.

The test that carries the real weight is
``TestLegacyEquivalence::test_atom_features_are_bit_identical``. Ablation 1 in
the specification is "the existing ESMC + GAT model retrained on the same curated
data", and that is only a controlled comparison if the ligand features are the
*same numbers* the legacy model consumed. A port that is merely "equivalent in
spirit" turns the ablation into a comparison of two different inputs.

The oracle is the legacy repository's own function, loaded from disk and run in
this interpreter, so the comparison is against executed code rather than against
my reading of it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch
from rdkit import Chem

from bindra.encoders.ligand_graph import (
    ATOM_FEATURE_DIM,
    ATOM_LAYOUT,
    BOND_FEATURE_DIM,
    Block,
    FeaturizationQuirk,
    LigandGraphError,
    MolecularGraph,
    collate_graphs,
    featurize_molecule,
    summarize_quirks,
    verify_layout,
)

LEGACY_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "DeepDTA-iBAM" / "models" / "rmse_model.py"
)

#: Chosen to exercise the branches, not to look like a drug library: unusual
#: elements that fall outside the symbol one-hot, small rings that trip the
#: ring-size collapse, explicit charges, declared stereocentres, radicals and a
#: single-atom molecule with no bonds at all.
PROBE_SMILES: tuple[str, ...] = (
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",  # caffeine
    "C1CC1",  # cyclopropane: ring size 3
    "C1CCC1",  # cyclobutane: ring size 4
    "C1CCCC1",  # cyclopentane: ring size 5
    "C1CCCCCC1",  # cycloheptane: ring size 7
    "OB(O)c1ccccc1",  # boronic acid: boron is outside the symbol one-hot
    "[Pt](Cl)(Cl)([NH3])[NH3]",  # cisplatin-like: platinum, also outside it
    "[Se]1CCCC1",  # selenolane
    "C[N+](C)(C)C",  # quaternary ammonium: +1 formal charge
    "[O-]C(=O)c1ccccc1",  # benzoate: -1 formal charge
    "N[C@@H](C)C(=O)O",  # L-alanine: declared stereocentre
    "N[C@H](C)C(=O)O",  # D-alanine: the other one
    "CO",  # methanol: terminal OH, the acceptor quirk
    "CN",  # methylamine: terminal NH2
    "CC#N",  # acetonitrile: triple bond
    "C",  # methane: one atom, zero bonds
    "c1ccc2ccccc2c1",  # naphthalene: fused aromatic, spiro-free
    "C[CH2]",  # ethyl radical
    "CS(=O)(=O)N",  # sulfonamide: S with high valence
    "FC(F)(F)c1ccccc1",  # trifluoromethyl: halogens
    "C1CC2(CCC2)CC1",  # spiro: an atom in two rings at once
)


def _load_legacy_oracle():
    """Import the legacy ``smiles_to_graph`` from the sibling repository."""
    spec = importlib.util.spec_from_file_location(
        "bindra_tests_legacy_rmse", LEGACY_MODEL_PATH
    )
    if spec is None or spec.loader is None:
        pytest.skip(f"cannot load the legacy oracle at {LEGACY_MODEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.smiles_to_graph


@pytest.fixture(scope="module")
def legacy_smiles_to_graph():
    if not LEGACY_MODEL_PATH.exists():
        pytest.skip(f"legacy repository not present at {LEGACY_MODEL_PATH}")
    return _load_legacy_oracle()


class TestLayout:
    """The slot map is the thing that makes an attribution interpretable."""

    def test_blocks_tile_the_vector_exactly_once(self):
        verify_layout()  # raises on overlap or gap

    def test_widths_sum_to_the_declared_dimension(self):
        assert sum(width for _, width in ATOM_LAYOUT.values()) == ATOM_FEATURE_DIM

    def test_every_block_is_named(self):
        assert set(ATOM_LAYOUT) == set(Block)

    def test_an_overlapping_layout_is_rejected(self):
        """The failure this guard exists for: two properties on one slot."""
        import bindra.encoders.ligand_graph as module

        original = module.ATOM_LAYOUT
        broken = dict(original)
        start, width = broken[Block.MASS]
        broken[Block.MASS] = (start - 1, width)  # now collides with COVALENT_RADIUS
        module.ATOM_LAYOUT = broken
        try:
            with pytest.raises(ValueError, match="claimed by both"):
                module.verify_layout()
        finally:
            module.ATOM_LAYOUT = original


class TestFeaturization:
    def test_shapes_follow_the_molecule(self):
        graph = featurize_molecule("CC(=O)Oc1ccccc1C(=O)O")
        assert graph.num_atoms == 13
        assert graph.atom_features.shape == (13, ATOM_FEATURE_DIM)
        assert graph.edge_index.shape == (2, 2 * graph.num_bonds)
        assert graph.bond_features.shape == (2 * graph.num_bonds, BOND_FEATURE_DIM)

    def test_each_bond_contributes_two_adjacent_opposite_edges(self):
        graph = featurize_molecule("CCO")
        src, dst = graph.edge_index
        for bond in range(graph.num_bonds):
            forward, backward = 2 * bond, 2 * bond + 1
            assert (src[forward], dst[forward]) == (dst[backward], src[backward])
            assert int(graph.edge_to_bond[forward]) == bond
            assert int(graph.edge_to_bond[backward]) == bond
            # Both directions carry the same bond features.
            assert torch.equal(
                graph.bond_features[forward], graph.bond_features[backward]
            )

    def test_a_single_atom_molecule_has_no_edges(self):
        graph = featurize_molecule("C")
        assert graph.num_atoms == 1
        assert graph.num_bonds == 0
        assert graph.edge_index.shape == (2, 0)
        assert graph.bond_features.shape == (0, BOND_FEATURE_DIM)

    def test_unparseable_smiles_raises_instead_of_returning_a_null_molecule(self):
        """The legacy function returned an all-zero graph here.

        A zero graph is a valid-looking molecule made of nothing: it trains, it
        contributes gradient, and downstream it is indistinguishable from a real
        single-atom species. Refusing is the only safe behaviour.
        """
        with pytest.raises(LigandGraphError, match="could not parse"):
            featurize_molecule("this is not a molecule")

    def test_an_empty_molecule_raises(self):
        with pytest.raises(LigandGraphError, match="no atoms"):
            featurize_molecule(Chem.MolFromSmiles(""))

    def test_bond_atom_pairs_match_rdkit(self):
        smiles = "CC(=O)Oc1ccccc1C(=O)O"
        graph = featurize_molecule(smiles)
        mol = Chem.MolFromSmiles(smiles)
        expected = tuple(
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()
        )
        assert graph.bond_atom_pairs() == expected

    @pytest.mark.parametrize("smiles", PROBE_SMILES)
    def test_every_probe_molecule_featurizes(self, smiles):
        graph = featurize_molecule(smiles)
        assert graph.num_atoms > 0
        assert torch.isfinite(graph.atom_features).all()
        assert torch.isfinite(graph.bond_features).all()

    def test_a_mismatched_tensor_is_rejected_by_the_contract(self):
        graph = featurize_molecule("CCO")
        with pytest.raises(ValueError, match="atom_features must be"):
            MolecularGraph(
                atom_features=graph.atom_features[:-1],
                edge_index=graph.edge_index,
                bond_features=graph.bond_features,
                edge_to_bond=graph.edge_to_bond,
                num_atoms=graph.num_atoms,
                num_bonds=graph.num_bonds,
                smiles=graph.smiles,
            )


class TestQuirks:
    """The preserved legacy defects are counted, so their reach is measurable."""

    def test_an_element_outside_the_one_hot_is_counted(self):
        graph = featurize_molecule("OB(O)c1ccccc1")
        assert graph.quirks[FeaturizationQuirk.UNKNOWN_SYMBOL] == 1

    def test_an_unknown_element_leaves_the_symbol_block_all_zero(self):
        """Boron and platinum are indistinguishable in the symbol block."""
        start, width = ATOM_LAYOUT[Block.SYMBOL]
        boron = featurize_molecule("OB(O)c1ccccc1")
        index = next(
            atom.GetIdx()
            for atom in Chem.MolFromSmiles("OB(O)c1ccccc1").GetAtoms()
            if atom.GetSymbol() == "B"
        )
        assert boron.atom_features[index, start : start + width].sum() == 0.0

    @pytest.mark.parametrize("smiles", ["C1CC1", "C1CCC1", "C1CCCC1"])
    def test_small_rings_trip_the_collapse_quirk(self, smiles):
        graph = featurize_molecule(smiles)
        assert FeaturizationQuirk.SMALL_RING_COLLAPSE in graph.quirks

    def test_a_six_ring_alone_does_not_trip_it(self):
        graph = featurize_molecule("C1CCCCC1")
        assert FeaturizationQuirk.SMALL_RING_COLLAPSE not in graph.quirks

    def test_ring_sizes_three_four_and_five_are_indistinguishable_in_bonds(self):
        """The defect itself, pinned so a future 'fix' is a deliberate choice."""
        bonds = [
            featurize_molecule(smiles).bond_features[0, 6:9]
            for smiles in ("C1CC1", "C1CCC1", "C1CCCC1")
        ]
        assert torch.equal(bonds[0], bonds[1])
        assert torch.equal(bonds[1], bonds[2])

    def test_a_terminal_hydroxyl_is_not_marked_an_acceptor(self):
        """Chemically wrong, and reproduced on purpose. See the module docstring."""
        graph = featurize_molecule("CO")
        oxygen = next(
            atom.GetIdx()
            for atom in Chem.MolFromSmiles("CO").GetAtoms()
            if atom.GetSymbol() == "O"
        )
        acceptor = ATOM_LAYOUT[Block.ACCEPTOR][0]
        donor = ATOM_LAYOUT[Block.DONOR][0]
        assert graph.atom_features[oxygen, donor] == 1.0
        assert graph.atom_features[oxygen, acceptor] == 0.0
        assert graph.quirks[FeaturizationQuirk.ACCEPTOR_EXCLUDES_TERMINAL_OH] == 1

    def test_an_ether_oxygen_is_marked_an_acceptor(self):
        graph = featurize_molecule("COC")
        oxygen = next(
            atom.GetIdx()
            for atom in Chem.MolFromSmiles("COC").GetAtoms()
            if atom.GetSymbol() == "O"
        )
        assert graph.atom_features[oxygen, ATOM_LAYOUT[Block.ACCEPTOR][0]] == 1.0

    def test_quirks_summarize_over_a_collection(self):
        graphs = [featurize_molecule(smiles) for smiles in ("CO", "CO", "C1CC1")]
        totals = summarize_quirks(graphs)
        assert totals[FeaturizationQuirk.ACCEPTOR_EXCLUDES_TERMINAL_OH.value] == 2
        assert totals[FeaturizationQuirk.SMALL_RING_COLLAPSE.value] == 1
        assert set(totals) == {quirk.value for quirk in FeaturizationQuirk}


class TestCollation:
    def test_edge_indices_are_shifted_into_the_concatenated_axis(self):
        """The classic batching error: every molecule's edges aiming at the first.

        It trains without complaint and produces nonsense, so it is asserted
        rather than assumed.
        """
        graphs = [featurize_molecule(s) for s in ("CCO", "c1ccccc1", "C")]
        batch = collate_graphs(graphs)
        assert batch.total_atoms == sum(g.num_atoms for g in graphs)
        assert batch.atom_offsets.tolist() == [0, 3, 9, 10]
        # The second molecule's edges must live in [3, 9), not in [0, 3).
        second = batch.edge_index[:, 2 * graphs[0].num_bonds :]
        second = second[:, : 2 * graphs[1].num_bonds]
        assert int(second.min()) >= 3
        assert int(second.max()) < 9

    def test_batch_index_assigns_every_atom_to_its_molecule(self):
        graphs = [featurize_molecule(s) for s in ("CCO", "c1ccccc1", "C")]
        batch = collate_graphs(graphs)
        assert batch.batch_index.tolist() == [0] * 3 + [1] * 6 + [2]

    def test_graph_slice_recovers_each_molecule_unchanged(self):
        graphs = [featurize_molecule(s) for s in ("CCO", "c1ccccc1", "C")]
        batch = collate_graphs(graphs)
        for index, graph in enumerate(graphs):
            assert torch.equal(
                batch.atom_features[batch.graph_slice(index)], graph.atom_features
            )

    def test_a_bondless_molecule_in_the_batch_is_handled(self):
        batch = collate_graphs([featurize_molecule("C"), featurize_molecule("C")])
        assert batch.edge_index.shape == (2, 0)
        assert batch.total_atoms == 2

    def test_an_empty_batch_is_refused(self):
        with pytest.raises(ValueError, match="empty batch"):
            collate_graphs([])


class TestLegacyEquivalence:
    """Bit-exact against the legacy repository's own executed code."""

    @pytest.mark.parametrize("smiles", PROBE_SMILES)
    def test_atom_features_are_bit_identical(self, legacy_smiles_to_graph, smiles):
        expected, _, _, _ = legacy_smiles_to_graph(smiles)
        actual = featurize_molecule(smiles).atom_features
        assert actual.shape == expected.shape
        mismatched = (actual != expected).any(dim=0).nonzero().flatten().tolist()
        assert not mismatched, (
            f"{smiles}: slots {mismatched} differ from the legacy featurization; "
            f"owning blocks "
            f"{ {b.value for b, (s, w) in ATOM_LAYOUT.items() for m in mismatched if s <= m < s + w} }"
        )

    @pytest.mark.parametrize("smiles", PROBE_SMILES)
    def test_dense_views_reproduce_the_legacy_adjacency_and_bond_tensors(
        self, legacy_smiles_to_graph, smiles
    ):
        _, expected_adjacency, _, expected_edges = legacy_smiles_to_graph(smiles)
        adjacency, edges = featurize_molecule(smiles).dense_views()
        assert torch.equal(adjacency, expected_adjacency)
        assert torch.equal(edges, expected_edges)

    def test_the_modern_valence_accessor_agrees_with_the_deprecated_one(self):
        """The one substitution made in the port, verified rather than assumed.

        ``GetImplicitValence()`` warns on RDKit 2026.03; the port calls
        ``GetValence(Chem.ValenceType.IMPLICIT)``. If they ever diverged, the
        implicit-valence block would silently shift.
        """
        for smiles in PROBE_SMILES:
            mol = Chem.MolFromSmiles(smiles)
            for atom in mol.GetAtoms():
                assert atom.GetValence(
                    Chem.ValenceType.IMPLICIT
                ) == atom.GetImplicitValence(), smiles


@pytest.mark.snapshot
class TestAgainstTheCuratedCorpus:
    """Equivalence on real curated SMILES, not only on hand-picked probes."""

    @pytest.fixture(scope="class")
    @classmethod
    def corpus_smiles(cls) -> list[str]:
        import pandas as pd

        shards = sorted(Path("data/processed/202610/ligands").glob("*.parquet"))
        if not shards:
            pytest.skip("curated ligand table not present")
        frame = pd.read_parquet(shards[0], columns=["standardized_smiles", "status"])
        ok = frame[frame["status"] == "OK"]["standardized_smiles"]
        unique = ok.dropna().drop_duplicates()
        # Deterministic sample: the test must not change its verdict run to run.
        return list(unique.sample(n=min(400, len(unique)), random_state=20261001))

    def test_bit_identical_on_real_molecules(
        self, legacy_smiles_to_graph, corpus_smiles
    ):
        differing: list[str] = []
        for smiles in corpus_smiles:
            try:
                actual = featurize_molecule(smiles).atom_features
            except LigandGraphError:
                continue
            expected, _, _, _ = legacy_smiles_to_graph(smiles)
            if not torch.equal(actual, expected):
                differing.append(smiles)
        assert not differing, f"{len(differing)}/{len(corpus_smiles)} differ: {differing[:5]}"

    def test_quirk_prevalence_is_reported(self, corpus_smiles):
        """Not an assertion about chemistry: a measurement for the report."""
        graphs = []
        for smiles in corpus_smiles:
            try:
                graphs.append(featurize_molecule(smiles))
            except LigandGraphError:
                continue
        totals = summarize_quirks(graphs)
        print(f"\nquirk prevalence over {len(graphs)} curated molecules: {totals}")
        assert set(totals) == {quirk.value for quirk in FeaturizationQuirk}
