"""The bond map, and the control that has to pass before it means anything.

``test_distance_control_detects_a_map_that_is_only_topology`` is the one that
matters. Attention over a line graph will always *look* structured, because the
graph is structured; the question is whether it carries anything beyond path
length. The control answers that, and is meant to be run before any heatmap is
read.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bindra.encoders.ligand_arm import build_ligand_arm
from bindra.encoders.ligand_kpgt import KPGTConfig, KPGTLigandEncoder
from bindra.ibam.bond_map import BondMap, build_bond_map, distance_control

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
#: 13 heavy atoms, 13 bonds, and 2 knowledge nodes on top.
ASPIRIN_BONDS = 13


def small_config() -> KPGTConfig:
    return KPGTConfig.mini(
        d_g_feats=32, n_mol_layers=3, n_trainable_layers=3, n_heads=4, d_hpath_ratio=2
    )


@pytest.fixture
def pieces():
    torch.manual_seed(20261008)
    config = small_config()
    arm = build_ligand_arm(config)
    batch = arm.collate_smiles([ASPIRIN, "CCO"])
    states, attention, distance = arm.module.triplet_states(
        batch.graphs, batch.fingerprints, batch.descriptors, return_attention=True
    )
    return config, arm, batch, attention, distance


class TestTheMap:
    def test_it_is_bonds_by_bonds_with_the_knowledge_nodes_removed(self, pieces):
        _, _, batch, attention, distance = pieces
        bond_map = build_bond_map(attention, distance, batch.graphs, index=0)
        assert bond_map.matrix.shape == (ASPIRIN_BONDS, ASPIRIN_BONDS)
        assert bond_map.n_bonds == ASPIRIN_BONDS
        # The batch carries two knowledge nodes per molecule on top of the bonds.
        assert batch.graphs.num_nodes == (ASPIRIN_BONDS + 2) + (2 + 2)

    def test_dropped_knowledge_mass_is_reported_not_renormalised(self, pieces):
        """The missing row mass is a real quantity, not rounding.

        The softmax normalised over every incoming edge, knowledge nodes
        included. What a bond drew from the molecule's global descriptors is
        exactly what is missing from its row, and it is reported rather than
        scaled away -- the same rule `nonphysical_mass` follows on the protein
        side.
        """
        _, _, batch, attention, distance = pieces
        bond_map = build_bond_map(attention, distance, batch.graphs, index=0)
        total = bond_map.matrix.sum(axis=1) + bond_map.knowledge_mass
        assert np.allclose(total, 1.0, atol=1e-5), total
        assert bond_map.knowledge_mass.min() >= 0.0

    def test_rows_cannot_rank_but_columns_can(self, pieces):
        """Row sums are a softmax constant; the hub score is the usable direction."""
        _, _, batch, attention, distance = pieces
        bond_map = build_bond_map(attention, distance, batch.graphs, index=0)
        rows = bond_map.matrix.sum(axis=1) + bond_map.knowledge_mass
        assert rows.std() < 1e-5  # constant by construction
        assert bond_map.hub.std() > 1e-6  # varies, so it can order bonds
        assert np.allclose(bond_map.hub, bond_map.matrix.sum(axis=0))

    def test_it_is_not_symmetric(self, pieces):
        """`i -> j` and `j -> i` are different questions."""
        _, _, batch, attention, distance = pieces
        matrix = build_bond_map(attention, distance, batch.graphs, index=0).matrix
        assert not np.allclose(matrix, matrix.T)

    def test_per_layer_per_head_is_retained_unaveraged(self, pieces):
        _, _, batch, attention, distance = pieces
        bond_map = build_bond_map(attention, distance, batch.graphs, index=0)
        assert bond_map.by_layer_head.shape == (3, 4, ASPIRIN_BONDS, ASPIRIN_BONDS)
        assert np.allclose(bond_map.by_layer_head.mean(axis=(0, 1)), bond_map.matrix)

    def test_each_molecule_in_a_batch_gets_its_own_map(self, pieces):
        _, _, batch, attention, distance = pieces
        assert build_bond_map(attention, distance, batch.graphs, index=1).n_bonds == 2
        with pytest.raises(ValueError, match="no triplets"):
            build_bond_map(attention, distance, batch.graphs, index=9)

    def test_rows_are_drawable_only_when_the_mapping_is_supplied(self, pieces):
        """`KPGTBatch` drops `triplet_to_atoms` at collation, so it must be passed."""
        config, arm, batch, attention, distance = pieces
        bare = build_bond_map(attention, distance, batch.graphs, index=0)
        assert all(atoms == () for atoms in bare.bond_to_atoms)

        features = KPGTLigandEncoder(config, module=arm.module).encode(ASPIRIN)
        physical = tuple(
            atoms
            for atoms, kind in zip(features.token_to_atom, features.token_types)
            if str(kind) in ("atom", "bond")
        )
        drawn = build_bond_map(
            attention, distance, batch.graphs, index=0, bond_to_atoms=physical
        )
        assert len(drawn.bond_to_atoms) == ASPIRIN_BONDS
        # Every bond maps to two atoms; nothing maps to none.
        assert all(len(a) == 2 for a in drawn.bond_to_atoms)


class TestTheDistanceControl:
    def test_distance_control_detects_a_map_that_is_only_topology(self, pieces):
        """A map equal to its own distance prior scores 1.0 and means nothing.

        Fed the distance bias as if it were the attention, the control must
        report complete explanation. If this ever returns a low number for
        identical inputs the control is broken, and every interpretation it
        was meant to gate is unguarded.
        """
        _, _, _, _, distance = pieces
        assert distance_control((distance,), distance) == pytest.approx(1.0, abs=1e-6)

    def test_unrelated_attention_scores_near_zero(self, pieces):
        torch.manual_seed(0)
        _, _, _, _, distance = pieces
        noise = torch.rand_like(distance)
        assert distance_control((noise,), distance) < 0.2

    def test_the_real_map_carries_its_control_with_it(self, pieces):
        """So a reader cannot pick up the heatmap without the caveat."""
        _, _, batch, attention, distance = pieces
        bond_map = build_bond_map(attention, distance, batch.graphs, index=0)
        assert 0.0 <= bond_map.distance_r2 <= 1.0
        assert "NOT bond order" in bond_map.disclaimer
