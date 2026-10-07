"""The KPGT ligand encoder: vendored descriptors and the bond-level token contract.

Two things are being protected here.

The **descriptors** are vendored third-party code whose exact values the
pretrained weights were fitted against. A different descriptastorus version, or
a differently fitted normalization, would feed the encoder inputs it has never
seen -- silently, with no error. ``test_vendored_descriptors_match_the_reference``
is what makes "exact" a measurement.

The **token contract** is where KPGT differs from every other arm: a token is a
*bond* and maps to two atoms, not one. If that mapping were wrong, every iBAM
attribution would be painted on the wrong part of the molecule while looking
entirely plausible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from rdkit import Chem

from bindra.encoders.kpgt_descriptors import N_DESCRIPTORS, RDKit2DNormalized
from bindra.encoders.kpgt_graph import featurize_kpgt
from bindra.encoders.ligand import (
    LigandEncoderError,
    available_ligand_encoders,
    build_ligand_encoder,
)
from bindra.encoders.ligand_kpgt import ARCHITECTURE_VERSION, KPGTConfig, KPGTLigandEncoder
from bindra.features.types import LigandFeatures, TokenType

REFERENCE_MD = Path("C:/bindra/oracle/reference_md.npz")

MOLECULES = (
    "CCO",
    "CC(=O)Oc1ccccc1C(=O)O",
    "c1ccccc1",
    "C",
    "N[C@@H](C)C(=O)O",
)

#: Small for speed; the architecture is verified at base scale in
#: tests/test_kpgt_port.py, so nothing here depends on the width.
#: ``n_trainable_layers`` is pinned rather than defaulted because this stub has
#: only 2 layers and the project default is 3; freezing policy is covered in
#: tests/test_kpgt_finetune.py.
SMALL = KPGTConfig(
    d_g_feats=48, n_mol_layers=2, n_heads=4, d_hpath_ratio=4, n_trainable_layers=1
)


@pytest.fixture(scope="module")
def encoder() -> KPGTLigandEncoder:
    torch.manual_seed(20261001)
    built = KPGTLigandEncoder(SMALL)
    built.module.eval()
    return built


class TestVendoredDescriptors:
    def test_the_width_is_two_hundred(self):
        values = RDKit2DNormalized().process("CCO")
        assert len(values) == N_DESCRIPTORS + 1  # plus the leading validity flag

    def test_vendored_descriptors_match_the_reference(self):
        """Against the original running in the CPU-only DGL environment."""
        if not REFERENCE_MD.exists():
            pytest.skip(f"reference not generated: {REFERENCE_MD}")
        reference = np.load(REFERENCE_MD, allow_pickle=True)
        generator = RDKit2DNormalized()
        produced = np.array(
            [generator.process(s)[1:] for s in list(reference["smiles"])]
        )
        deviation = np.abs(produced - reference["md"]).max()
        assert deviation < 1e-12, f"descriptors deviate by {deviation:.3e}"

    def test_the_column_set_matches_the_reference(self):
        if not REFERENCE_MD.exists():
            pytest.skip(f"reference not generated: {REFERENCE_MD}")
        reference = np.load(REFERENCE_MD, allow_pickle=True)
        columns = [name for name, _ in RDKit2DNormalized().GetColumns()]
        assert columns == list(reference["columns"])

    def test_values_are_cdf_normalised_into_the_unit_interval(self):
        values = np.array([RDKit2DNormalized().process(s)[1:] for s in MOLECULES])
        assert np.isfinite(values).all()
        assert values.min() >= 0.0 and values.max() <= 1.0

    def test_the_generator_is_deterministic(self):
        generator = RDKit2DNormalized()
        assert generator.process("CCO") == generator.process("CCO")


class TestTokenContract:
    """A KPGT token is a bond, and the contract has to say so."""

    def test_a_bond_token_maps_to_exactly_two_atoms(self, encoder):
        features = encoder.encode("CC(=O)Oc1ccccc1C(=O)O")
        bonds = [
            atoms
            for kind, atoms in zip(features.token_types, features.token_to_atom)
            if kind is TokenType.BOND
        ]
        assert bonds and all(len(atoms) == 2 for atoms in bonds)

    def test_there_is_one_bond_token_per_bond_of_the_molecule(self, encoder):
        smiles = "CC(=O)Oc1ccccc1C(=O)O"
        features = encoder.encode(smiles)
        expected = Chem.MolFromSmiles(smiles).GetNumBonds()
        assert sum(k is TokenType.BOND for k in features.token_types) == expected

    def test_knowledge_nodes_are_present_and_map_to_no_atom(self, encoder):
        features = encoder.encode("CCO")
        knowledge = [
            atoms
            for kind, atoms in zip(features.token_types, features.token_to_atom)
            if kind is TokenType.KNOWLEDGE
        ]
        assert len(knowledge) == 2  # fingerprint node and descriptor node
        assert all(atoms == () for atoms in knowledge)

    def test_knowledge_nodes_are_not_physical_so_ibam_will_not_draw_them(self, encoder):
        features = encoder.encode("CCO")
        physical = set(features.physical_token_positions())
        for position, kind in enumerate(features.token_types):
            if kind is TokenType.KNOWLEDGE:
                assert position not in physical

    def test_an_unbonded_atom_becomes_an_atom_token(self, encoder):
        features = encoder.encode("C")
        atom_tokens = [
            atoms
            for kind, atoms in zip(features.token_types, features.token_to_atom)
            if kind is TokenType.ATOM
        ]
        assert atom_tokens == [(0,)]

    def test_bond_tokens_carry_their_bond_index(self, encoder):
        features = encoder.encode("CCO")
        pairs = [
            (kind, bond)
            for kind, bond in zip(features.token_types, features.token_to_bond)
        ]
        bond_indices = [b for k, b in pairs if k is TokenType.BOND]
        assert bond_indices == list(range(len(bond_indices)))
        assert all(b is None for k, b in pairs if k is not TokenType.BOND)

    def test_every_atom_of_the_molecule_is_reachable_through_some_token(self, encoder):
        smiles = "CC(=O)Oc1ccccc1C(=O)O"
        features = encoder.encode(smiles)
        covered = {a for atoms in features.token_to_atom for a in atoms}
        assert covered == set(range(features.num_atoms))

    def test_token_atom_indices_are_in_original_molecule_order(self, encoder):
        """KPGT renumbers atoms canonically; the mapping must undo that.

        Otherwise every attribution lands on the wrong atom of the user's own
        depiction, with nothing to signal it.
        """
        smiles = "CC(=O)Oc1ccccc1C(=O)O"
        graph = featurize_kpgt(smiles)
        mol = Chem.MolFromSmiles(smiles)
        bonds = {
            frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()
        }
        mapped = {
            frozenset(atoms)
            for atoms, role in zip(graph.triplet_to_atoms, graph.vavn)
            if role == 0
        }
        assert mapped == bonds


class TestEncoder:
    def test_the_readout_is_three_blocks_wide(self, encoder):
        features = encoder.encode("CCO")
        assert features.graph_features.shape == (encoder.graph_dim,)
        assert encoder.graph_dim == 3 * encoder.token_dim

    def test_token_features_have_one_row_per_token(self, encoder):
        features = encoder.encode("CCO")
        assert features.token_features.shape == (
            len(features.token_types),
            encoder.token_dim,
        )

    def test_encode_batch_matches_repeated_encode(self, encoder):
        """A missing offset shift in collation would show up only here."""
        batched = encoder.encode_batch(list(MOLECULES))
        for smiles, features in zip(MOLECULES, batched):
            alone = encoder.encode(smiles)
            assert torch.allclose(
                features.token_features, alone.token_features, atol=1e-5
            ), smiles
            assert torch.allclose(
                features.graph_features, alone.graph_features, atol=1e-5
            ), smiles

    def test_an_empty_batch_returns_nothing(self, encoder):
        assert encoder.encode_batch([]) == []

    def test_mismatched_atom_index_maps_are_refused(self, encoder):
        with pytest.raises(LigandEncoderError, match="atom index maps"):
            encoder.encode_batch(["CCO", "CC"], atom_index_maps=[{}])

    def test_the_knowledge_inputs_have_the_published_widths(self, encoder):
        fingerprints, descriptors = encoder.knowledge_inputs(list(MOLECULES))
        assert fingerprints.shape == (len(MOLECULES), 512)
        assert descriptors.shape == (len(MOLECULES), N_DESCRIPTORS)
        assert torch.isfinite(descriptors).all()

    def test_output_satisfies_the_ligand_features_contract(self, encoder):
        features = encoder.encode("CC(=O)Oc1ccccc1C(=O)O")
        assert isinstance(features, LigandFeatures)  # __post_init__ validates arities

    def test_encoding_leaves_the_module_in_its_original_mode(self):
        own = KPGTLigandEncoder(KPGTConfig(d_g_feats=24, n_mol_layers=1, n_heads=2, d_hpath_ratio=2, n_trainable_layers=1))
        own.module.train()
        own.encode("CCO")
        assert own.module.training


class TestMetadata:
    def test_untrained_weights_are_named_in_the_cache_key(self, encoder):
        metadata = encoder.metadata()
        assert metadata.checkpoint == "untrained"
        assert metadata.name == "ligand_kpgt"


    def test_the_graph_dimension_travels_with_the_metadata(self, encoder):
        assert encoder.metadata().extra["graph_dim"] == encoder.graph_dim

    def test_the_architecture_version_is_recorded(self, encoder):
        assert encoder.metadata().revision == ARCHITECTURE_VERSION

    def test_the_featurization_policy_is_part_of_the_key(self, encoder):
        from bindra.encoders.kpgt_graph import KPGTPolicy

        other = KPGTLigandEncoder(SMALL, policy=KPGTPolicy(n_virtual_nodes=3))
        assert encoder.metadata().cache_key("CCO") != other.metadata().cache_key("CCO")


class TestRegistry:
    def test_kpgt_is_now_available(self):
        """It was absent while DGL blocked it; the port changed that."""
        assert "kpgt" in available_ligand_encoders()

    def test_the_arm_is_registered(self):
        assert "kpgt" in set(available_ligand_encoders())
        assert "gat" not in set(available_ligand_encoders())

    def test_building_by_name_works(self):
        built = build_ligand_encoder(
            "kpgt", d_g_feats=24, n_mol_layers=1, n_heads=2,
            d_hpath_ratio=2, n_trainable_layers=1,
        )
        assert built.token_dim == 24
        assert built.encode("CCO").graph_features.shape == (72,)

    def test_a_missing_checkpoint_says_where_to_get_it(self, tmp_path):
        """The weights are not downloadable by script; the error must say so."""
        with pytest.raises(LigandEncoderError, match="not downloaded automatically"):
            KPGTLigandEncoder.from_pretrained(tmp_path / "absent.pth")

    def test_a_partial_checkpoint_is_refused_rather_than_half_loaded(self, tmp_path):
        """A silent partial load leaves random layers inside a 'pretrained' model."""
        path = tmp_path / "truncated.pth"
        full = KPGTLigandEncoder(SMALL).module.state_dict()
        torch.save({k: v for i, (k, v) in enumerate(full.items()) if i % 2 == 0}, path)
        with pytest.raises(RuntimeError, match="Missing key"):
            KPGTLigandEncoder.from_pretrained(path, config=SMALL)

    def test_a_loaded_checkpoint_is_identified_by_content_hash(self, tmp_path):
        path = tmp_path / "weights.pth"
        torch.save(KPGTLigandEncoder(SMALL).module.state_dict(), path)
        loaded = KPGTLigandEncoder.from_pretrained(path, config=SMALL)
        assert loaded.checkpoint.startswith("weights.pth@")
        assert loaded.metadata().cache_key("CCO") != KPGTLigandEncoder(
            SMALL
        ).metadata().cache_key("CCO")


@pytest.mark.gpu
class TestOnTheBigGpu:
    def test_the_encoder_runs_on_sm_120(self):
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")
        from bindra.runtime.device import list_cuda_devices

        device = next(
            (f"cuda:{g.index}" for g in list_cuda_devices() if g.capability[0] >= 12),
            None,
        )
        if device is None:
            pytest.skip("no sm_120 device present")
        torch.manual_seed(20261001)
        on_cpu = KPGTLigandEncoder(SMALL)
        on_cpu.module.eval()
        reference = on_cpu.encode_batch(list(MOLECULES))

        on_gpu = KPGTLigandEncoder(SMALL, device=device, module=on_cpu.module)
        produced = on_gpu.encode_batch(list(MOLECULES))
        for expected, actual in zip(reference, produced):
            assert torch.allclose(
                expected.graph_features, actual.graph_features, atol=1e-4
            )
        # Features come back on the CPU, which is what a cache writer wants.
        assert produced[0].token_features.device.type == "cpu"


class TestUnnormalizedDescriptorsArePinnedNotPrinted:
    """The vendored module logs 17 root-logger warnings at import time.

    They are expected, nothing downstream can act on them, and they fired every
    time anything imported the KPGT encoder -- noise that was read by nobody.
    Capturing them turns the fact into something a test can hold: the set is a
    property of the installed RDKit crossed with a pinned data file, so an RDKit
    upgrade that adds a descriptor changes it and fails here instead of scrolling
    past in a log.
    """

    #: Measured against RDKit 2024.3+ and the pinned ``dists.py``.
    EXPECTED = (
        "SPS", "BCUT2D_MWHI", "BCUT2D_MWLOW", "BCUT2D_CHGHI", "BCUT2D_CHGLO",
        "BCUT2D_LOGPHI", "BCUT2D_LOGPLOW", "BCUT2D_MRHI", "BCUT2D_MRLOW",
        "AvgIpc", "NumAmideBonds", "NumAtomStereoCenters", "NumBridgeheadAtoms",
        "NumHeterocycles", "NumSpiroAtoms", "NumUnspecifiedAtomStereoCenters",
        "Phi",
    )

    def test_the_set_is_exactly_what_was_measured(self):
        from bindra.encoders.kpgt_descriptors import UNNORMALIZED_DESCRIPTORS

        assert UNNORMALIZED_DESCRIPTORS == self.EXPECTED

    def test_importing_the_package_logs_nothing(self):
        """Scoped suppression: the import is quiet, other warnings still pass."""
        import importlib
        import logging

        import bindra.encoders.kpgt_descriptors as package

        records: list[logging.LogRecord] = []

        class _Collect(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = _Collect()
        logging.root.addHandler(handler)
        previous = logging.root.level
        logging.root.setLevel(logging.DEBUG)
        try:
            importlib.reload(package)
            logging.warning("a warning that must still get through")
        finally:
            logging.root.setLevel(previous)
            logging.root.removeHandler(handler)

        assert not [r for r in records if str(r.msg).startswith("No normalization")]
        assert any("must still get through" in str(r.msg) for r in records), (
            "the filter must be removed after the vendored import, not left installed"
        )

    def test_the_filter_is_not_left_on_the_root_logger(self):
        import logging

        import bindra.encoders.kpgt_descriptors  # noqa: F401

        assert not [
            f for f in logging.root.filters
            if type(f).__name__ == "_CaptureUnnormalized"
        ]

    def test_an_unnormalized_descriptor_returns_zero(self):
        """Upstream behaviour, and what the pretrained weights were fitted to."""
        from bindra.encoders.kpgt_descriptors import rdNormalizedDescriptors

        from rdkit import Chem

        molecule = Chem.MolFromSmiles("CCO")
        assert rdNormalizedDescriptors.applyNormalizedFunc("Phi", molecule) == 0.0
