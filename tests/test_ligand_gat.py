"""The GAT ligand encoder: ablation 1, and the first arm that can actually run.

The load-bearing tests, in order of how much damage their failure would do:

``test_batching_does_not_change_any_molecule``
    A segment-indexing bug -- a missing offset shift, a readout that pools over
    the whole batch instead of per molecule -- trains without complaint and
    silently mixes molecules together. Nothing else in the suite would catch it.
``test_attention_weights_sum_to_one_per_destination_atom``
    If the softmax is taken over the wrong axis the model still learns something,
    and every iBAM attribution built on it is meaningless.
``test_a_single_atom_molecule_is_not_the_zero_vector``
    Methane is in the corpus. Without self-loops it has no incoming edge and its
    representation is zero regardless of its features.
``test_permuting_atoms_permutes_the_output_identically``
    Graph encoders must not depend on RDKit's atom ordering.
"""

from __future__ import annotations

import pytest
import torch
from rdkit import Chem

from bindra.encoders.ligand import (
    LigandEncoderError,
    available_ligand_encoders,
    build_ligand_encoder,
    segment_max,
    segment_mean,
    segment_softmax,
    segment_sum,
)
from bindra.encoders.ligand_gat import (
    ARCHITECTURE_VERSION,
    GATConfig,
    GATLigandEncoder,
    GATLigandEncoderModule,
)
from bindra.encoders.ligand_graph import collate_graphs, featurize_molecule
from bindra.features.types import LigandFeatures, TokenType

MOLECULES = (
    "CC(=O)Oc1ccccc1C(=O)O",
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "C",
    "CCO",
    "c1ccc2ccccc2c1",
    "N[C@@H](C)C(=O)O",
)


@pytest.fixture(scope="module")
def encoder() -> GATLigandEncoder:
    """A small encoder in **eval mode**; the architecture is what is under test.

    Eval mode is not incidental. Several tests below compare two forward passes
    and must attribute any difference to the batching or the atom ordering, so
    dropout -- which is on by default, because ``nn.Module`` starts in train mode
    -- has to be off or the comparison measures the RNG instead.
    """
    torch.manual_seed(20261001)
    built = GATLigandEncoder(GATConfig(hidden_dim=32, num_layers=2, num_heads=4))
    built.module.eval()
    return built


class TestSegmentOps:
    """The primitives everything else rests on, including their edge cases."""

    def test_softmax_sums_to_one_within_each_segment(self):
        logits = torch.randn(10, 3)
        index = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 3])
        weights = segment_softmax(logits, index, 4)
        totals = segment_sum(weights, index, 4)
        assert torch.allclose(totals, torch.ones(4, 3), atol=1e-6)

    def test_softmax_survives_logits_that_would_overflow_a_plain_exp(self):
        """Attention logits here are unbounded; exp(800) is inf, inf/inf is NaN."""
        logits = torch.tensor([[800.0], [799.0], [-800.0]])
        index = torch.tensor([0, 0, 0])
        weights = segment_softmax(logits, index, 1)
        assert torch.isfinite(weights).all()
        assert weights.sum().item() == pytest.approx(1.0)
        assert weights[0, 0] > weights[1, 0] > weights[2, 0]

    def test_an_empty_segment_yields_zero_not_nan_or_inf(self):
        logits = torch.randn(2, 2)
        index = torch.tensor([0, 0])
        weights = segment_softmax(logits, index, 3)  # segments 1 and 2 are empty
        assert torch.isfinite(weights).all()

    def test_segment_max_of_an_empty_segment_is_zero_not_negative_infinity(self):
        """A -inf would poison the next linear layer."""
        values = torch.randn(2, 4)
        result = segment_max(values, torch.tensor([0, 0]), 3)
        assert torch.isfinite(result).all()
        assert torch.equal(result[1], torch.zeros(4))

    def test_segment_mean_of_an_empty_segment_is_zero_not_nan(self):
        values = torch.randn(2, 4)
        result = segment_mean(values, torch.tensor([0, 0]), 3)
        assert torch.isfinite(result).all()

    def test_segment_mean_matches_a_direct_computation(self):
        values = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        index = torch.tensor([0, 0, 0, 1, 1, 2])
        result = segment_mean(values, index, 3)
        assert torch.allclose(result[0], values[:3].mean(0))
        assert torch.allclose(result[1], values[3:5].mean(0))
        assert torch.allclose(result[2], values[5])

    def test_softmax_rejects_a_wrongly_shaped_input(self):
        with pytest.raises(ValueError, match=r"must be \[E, H\]"):
            segment_softmax(torch.randn(4), torch.tensor([0, 0, 1, 1]), 2)


class TestConfig:
    def test_heads_must_divide_the_hidden_dimension(self):
        """Truncating integer division would silently drop channels."""
        with pytest.raises(ValueError, match="not divisible"):
            GATConfig(hidden_dim=30, num_heads=4)

    def test_mean_max_readout_doubles_the_graph_dimension(self):
        assert GATConfig(hidden_dim=64, readout="mean_max").graph_dim == 128
        assert GATConfig(hidden_dim=64, readout="mean").graph_dim == 64

    def test_an_unknown_readout_is_refused(self):
        with pytest.raises(ValueError, match="unknown readout"):
            GATConfig(readout="attention_please")

    def test_zero_layers_is_refused(self):
        with pytest.raises(ValueError, match="num_layers"):
            GATConfig(num_layers=0)


class TestForward:
    def test_output_shapes_follow_the_config(self, encoder):
        graphs = [featurize_molecule(s) for s in MOLECULES]
        batch = collate_graphs(graphs)
        output = encoder.module(batch)
        assert output["token_states"].shape == (batch.total_atoms, 32)
        assert output["graph_states"].shape == (len(MOLECULES), 64)

    def test_batching_does_not_change_any_molecule(self, encoder):
        """The error that silently mixes molecules together.

        A missing offset shift in collation, or a readout that pools over the
        whole batch, produces plausible-looking numbers that depend on which
        molecules happened to share a batch.
        """
        batch = collate_graphs([featurize_molecule(s) for s in MOLECULES])
        batched = encoder.module(batch)
        for index, smiles in enumerate(MOLECULES):
            alone = encoder.module(collate_graphs([featurize_molecule(smiles)]))
            assert torch.allclose(
                batched["token_states"][batch.graph_slice(index)],
                alone["token_states"],
                atol=1e-5,
            ), smiles
            assert torch.allclose(
                batched["graph_states"][index], alone["graph_states"][0], atol=1e-5
            ), smiles

    def test_batch_order_does_not_matter(self, encoder):
        forward = collate_graphs([featurize_molecule(s) for s in MOLECULES])
        reverse = collate_graphs([featurize_molecule(s) for s in reversed(MOLECULES)])
        a = encoder.module(forward)["graph_states"]
        b = encoder.module(reverse)["graph_states"]
        assert torch.allclose(a, b.flip(0), atol=1e-5)

    def test_attention_weights_sum_to_one_per_destination_atom(self, encoder):
        """Softmax over the wrong axis still trains, and makes iBAM meaningless."""
        batch = collate_graphs([featurize_molecule(s) for s in MOLECULES])
        output = encoder.module(batch, return_attention=True)
        edge_index = output["edge_index"]
        for weights in output["attention"]:
            totals = segment_sum(weights, edge_index[1], batch.total_atoms)
            # Every atom has at least its self-loop, so every segment is non-empty.
            assert torch.allclose(totals, torch.ones_like(totals), atol=1e-5)

    def test_attention_is_returned_per_layer_and_unaveraged_over_heads(self, encoder):
        batch = collate_graphs([featurize_molecule("CC(=O)Oc1ccccc1C(=O)O")])
        output = encoder.module(batch, return_attention=True)
        assert len(output["attention"]) == encoder.config.num_layers
        for weights in output["attention"]:
            assert weights.shape[1] == encoder.config.num_heads

    def test_a_single_atom_molecule_is_not_the_zero_vector(self, encoder):
        """Methane is in the corpus. Without self-loops it would encode to zero."""
        output = encoder.module(collate_graphs([featurize_molecule("C")]))
        assert output["token_states"].shape[0] == 1
        assert output["token_states"].abs().sum() > 0
        assert torch.isfinite(output["graph_states"]).all()

    def test_permuting_atoms_permutes_the_output_identically(self, encoder):
        """The output must not depend on RDKit's atom ordering."""
        smiles = "CC(=O)Oc1ccccc1C(=O)O"
        mol = Chem.MolFromSmiles(smiles)
        order = [3, 0, 7, 1, 11, 2, 9, 4, 12, 5, 10, 6, 8]
        permuted = Chem.RenumberAtoms(mol, order)

        original = encoder.module(collate_graphs([featurize_molecule(mol)]))
        shuffled = encoder.module(collate_graphs([featurize_molecule(permuted)]))
        # Row i of the permuted graph is row order[i] of the original.
        assert torch.allclose(
            shuffled["token_states"], original["token_states"][order], atol=1e-5
        )
        # And the graph readout is order-invariant, not merely permuted.
        assert torch.allclose(
            shuffled["graph_states"], original["graph_states"], atol=1e-5
        )

    def test_gradients_reach_every_parameter(self, encoder):
        module = GATLigandEncoderModule(GATConfig(hidden_dim=16, num_layers=2, num_heads=2))
        batch = collate_graphs([featurize_molecule(s) for s in MOLECULES])
        module(batch)["graph_states"].sum().backward()
        missing = [
            name
            for name, parameter in module.named_parameters()
            if parameter.grad is None or parameter.grad.abs().sum() == 0
        ]
        assert not missing, f"no gradient reached: {missing}"

    def test_forward_is_finite_on_every_probe_molecule(self, encoder):
        for smiles in MOLECULES:
            output = encoder.module(collate_graphs([featurize_molecule(smiles)]))
            assert torch.isfinite(output["token_states"]).all(), smiles
            assert torch.isfinite(output["graph_states"]).all(), smiles


class TestEncoderContract:
    def test_encode_produces_one_atom_token_per_atom(self, encoder):
        features = encoder.encode("CC(=O)Oc1ccccc1C(=O)O")
        assert isinstance(features, LigandFeatures)
        assert features.num_atoms == 13
        assert features.token_features.shape == (13, encoder.token_dim)
        assert set(features.token_types) == {TokenType.ATOM}
        assert features.token_to_atom == tuple((i,) for i in range(13))
        assert features.token_to_bond == (None,) * 13

    def test_one_token_equals_one_atom_is_stated_not_assumed(self, encoder):
        """Every token is declared physical, so iBAM may draw all of them."""
        features = encoder.encode("CCO")
        assert features.physical_token_positions() == (0, 1, 2)

    def test_the_graph_readout_has_the_declared_width(self, encoder):
        features = encoder.encode("CCO")
        assert features.graph_features.shape == (encoder.graph_dim,)

    def test_encode_batch_matches_repeated_encode(self, encoder):
        batched = encoder.encode_batch(list(MOLECULES))
        for smiles, features in zip(MOLECULES, batched):
            alone = encoder.encode(smiles)
            assert torch.allclose(
                features.token_features, alone.token_features, atol=1e-5
            ), smiles
            assert torch.allclose(
                features.graph_features, alone.graph_features, atol=1e-5
            ), smiles

    def test_an_empty_batch_returns_no_features(self, encoder):
        assert encoder.encode_batch([]) == []

    def test_mismatched_atom_index_maps_are_refused(self, encoder):
        with pytest.raises(LigandEncoderError, match="atom index maps"):
            encoder.encode_batch(["CCO", "CC"], atom_index_maps=[{}])

    def test_atom_index_maps_are_carried_through(self, encoder):
        maps = {"std_to_output": (2, 0, 1)}
        features = encoder.encode("CCO", atom_index_maps=maps)
        assert features.atom_index_maps == maps

    def test_encoding_leaves_the_module_in_its_original_mode(self):
        """Owns its encoder: it mutates train/eval, which other tests rely on."""
        own = GATLigandEncoder(GATConfig(hidden_dim=16, num_layers=1, num_heads=2))
        own.module.train()
        own.encode("CCO")
        assert own.module.training
        own.module.eval()
        own.encode("CCO")
        assert not own.module.training


class TestMetadata:
    def test_untrained_weights_are_named_in_the_cache_key(self, encoder):
        """Untrained activations must never be mistaken for trained ones."""
        metadata = encoder.metadata()
        assert metadata.checkpoint == "untrained"
        trained = GATLigandEncoder(encoder.config, checkpoint="epoch42-abc123")
        assert metadata.cache_key("CCO") != trained.metadata().cache_key("CCO")

    def test_the_architecture_version_is_recorded(self, encoder):
        assert encoder.metadata().revision == ARCHITECTURE_VERSION

    def test_a_config_change_changes_the_cache_key(self):
        a = GATLigandEncoder(GATConfig(hidden_dim=32, num_heads=4)).metadata()
        b = GATLigandEncoder(GATConfig(hidden_dim=64, num_heads=4)).metadata()
        assert a.cache_key("CCO") != b.cache_key("CCO")

    def test_the_graph_dimension_travels_with_the_metadata(self, encoder):
        assert encoder.metadata().extra["graph_dim"] == encoder.graph_dim


class TestRegistry:
    def test_the_gat_arm_is_registered(self):
        assert "gat" in available_ligand_encoders()

    def test_building_by_name_works(self):
        built = build_ligand_encoder("gat", hidden_dim=16, num_heads=2, num_layers=1)
        assert built.config.hidden_dim == 16
        assert built.encode("CCO").token_features.shape == (3, 16)

    def test_an_unknown_name_names_what_is_available(self):
        with pytest.raises(LigandEncoderError, match="available: .*gat"):
            build_ligand_encoder("not_an_encoder")

    def test_kpgt_is_now_registered_alongside_the_gat_arm(self):
        """This assertion used to read "kpgt is absent", and that was correct.

        DGL could not load under any torch built for this GPU, so the KPGT arm
        did not exist. It now does: LiGhT is ported to pure PyTorch and verified
        against the reference (tests/test_kpgt_port.py). The registry is what
        lets the rest of the system not care which arm is installed.
        """
        assert {"gat", "kpgt"} <= set(available_ligand_encoders())


@pytest.mark.gpu
class TestOnCuda:
    @pytest.fixture(scope="class")
    @classmethod
    def cuda_encoder(cls):
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")
        torch.manual_seed(20261001)
        return GATLigandEncoder(
            GATConfig(hidden_dim=32, num_layers=2, num_heads=4), device="cuda"
        )

    def test_the_encoder_runs_on_the_gpu(self, cuda_encoder):
        features = cuda_encoder.encode_batch(list(MOLECULES))
        assert len(features) == len(MOLECULES)
        # Features come back on the CPU, which is what the cache writer wants.
        assert features[0].token_features.device.type == "cpu"

    def test_cuda_and_cpu_agree_to_float32_tolerance(self, cuda_encoder):
        cpu = GATLigandEncoder(cuda_encoder.config)
        cpu.module.load_state_dict(
            {k: v.cpu() for k, v in cuda_encoder.module.state_dict().items()}
        )
        for smiles in MOLECULES:
            a = cuda_encoder.encode(smiles).token_features
            b = cpu.encode(smiles).token_features
            assert torch.allclose(a, b, atol=1e-4), smiles
