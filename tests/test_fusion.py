"""Adapters and bidirectional cross-attention.

The tests that carry real weight:

``test_padding_does_not_change_a_real_sequence``
    Protein lengths in this corpus span two orders of magnitude, so every batch is
    mostly padding. If padding leaks into the attention or the pooling, every
    result depends on which proteins happened to share a batch -- a corruption
    that is invisible in a loss curve and irreproducible across runs.
``test_the_two_directions_are_not_transposes``
    The specification requires both directions kept because they are different
    questions. A refactor that computes one score matrix and reads it both ways
    would pass most other tests here.
``test_an_all_padded_side_does_not_produce_nan``
    One NaN propagates through the whole batch and surfaces far from its cause.
``test_every_loss_reaches_the_shared_adapters``
    "Shared" is easy to claim and easy to break. This is the specification's
    "log which parameters receive gradients from each loss", as an assertion.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from bindra.features.types import EncoderMetadata
from bindra.fusion.adapters import (
    AdapterConfig,
    AdapterError,
    FeatureAdapter,
    SharedAdapters,
    assert_shared,
    attribute_gradients,
    masked_mean,
)
from bindra.fusion.cross_attention import (
    CrossAttentionConfig,
    CrossAttentionError,
    CrossAttentionStack,
    DirectedCrossAttention,
)


@pytest.fixture
def config() -> AdapterConfig:
    return AdapterConfig(
        protein_dim=1152, ligand_dim=256, ligand_graph_dim=512, model_dim=64, dropout=0.0
    )


@pytest.fixture
def adapters(config) -> SharedAdapters:
    torch.manual_seed(20261001)
    module = SharedAdapters(config)
    module.eval()
    return module


@pytest.fixture
def stack() -> CrossAttentionStack:
    torch.manual_seed(20261001)
    module = CrossAttentionStack(
        CrossAttentionConfig(model_dim=64, num_heads=4, num_layers=2, dropout=0.0)
    )
    module.eval()
    return module


class TestAdapterConfig:
    def test_widths_are_read_off_the_encoder_metadata(self):
        """Dimensions are discovered, never assumed."""
        protein = EncoderMetadata(
            name="esmc", checkpoint="x", output_dim=1152, precision="float32",
            feature_schema_version="v1",
        )
        ligand = EncoderMetadata(
            name="ligand_light_mini", checkpoint="untrained", output_dim=256,
            precision="float32", feature_schema_version="v1",
            extra={"graph_dim": 512},
        )
        built = AdapterConfig.from_encoders(protein, ligand, model_dim=128)
        assert (built.protein_dim, built.ligand_dim, built.ligand_graph_dim) == (
            1152, 256, 512
        )
        assert built.model_dim == 128

    def test_a_ligand_encoder_without_a_graph_dim_is_an_error_not_a_default(self):
        """Assuming ligand_dim would mis-shape the conditioning path silently."""
        protein = EncoderMetadata(
            name="esmc", checkpoint="x", output_dim=1152, precision="float32",
            feature_schema_version="v1",
        )
        ligand = EncoderMetadata(
            name="mystery", checkpoint="y", output_dim=256, precision="float32",
            feature_schema_version="v1",
        )
        with pytest.raises(AdapterError, match="graph_dim"):
            AdapterConfig.from_encoders(protein, ligand)

    def test_the_real_encoders_configure_without_hardcoding(self):
        """Against the encoders actually in use, not invented metadata."""
        from bindra.encoders.ligand_kpgt import KPGTConfig, KPGTLigandEncoder

        encoder = KPGTLigandEncoder(
            KPGTConfig.mini(d_g_feats=32, n_mol_layers=1, n_trainable_layers=1,
                            n_heads=4, d_hpath_ratio=2)
        )
        protein = EncoderMetadata(
            name="esmc", checkpoint="x", output_dim=1152, precision="float32",
            feature_schema_version="v1",
        )
        built = AdapterConfig.from_encoders(protein, encoder.metadata())
        assert built.ligand_dim == 32
        assert built.ligand_graph_dim == 96  # fp ++ md ++ mean readout

    @pytest.mark.parametrize("field", ["protein_dim", "ligand_dim", "model_dim"])
    def test_a_nonpositive_width_is_refused(self, field):
        kwargs = {"protein_dim": 8, "ligand_dim": 8, "ligand_graph_dim": 8, field: 0}
        with pytest.raises(AdapterError, match=field):
            AdapterConfig(**kwargs)


class TestAdapters:
    def test_every_stream_lands_at_the_model_width(self, adapters, config):
        out = adapters(
            torch.randn(2, 7, config.protein_dim),
            torch.randn(2, 5, config.ligand_dim),
            torch.randn(2, config.ligand_graph_dim),
        )
        assert out["protein"].shape == (2, 7, config.model_dim)
        assert out["ligand"].shape == (2, 5, config.model_dim)
        assert out["ligand_graph"].shape == (2, config.model_dim)

    def test_the_graph_stream_is_optional(self, adapters, config):
        out = adapters(
            torch.randn(2, 7, config.protein_dim), torch.randn(2, 5, config.ligand_dim)
        )
        assert "ligand_graph" not in out

    def test_a_wrong_input_width_is_refused(self, config):
        adapter = FeatureAdapter(config.protein_dim, config)
        with pytest.raises(AdapterError, match="expected last dimension 1152"):
            adapter(torch.randn(2, 7, 768))

    def test_a_bottleneck_still_reaches_the_model_width(self, config):
        narrow = AdapterConfig(**{**config.to_dict(), "bottleneck_dim": 8})
        adapter = FeatureAdapter(narrow.protein_dim, narrow)
        assert adapter(torch.randn(2, 3, narrow.protein_dim)).shape[-1] == narrow.model_dim

    def test_streams_of_very_different_scale_are_normalised_together(self, adapters, config):
        """ESM-C states and a fresh GAT differ by orders of magnitude in scale.

        Without input normalisation the larger stream dominates the fused
        representation by magnitude alone for the first thousands of steps.
        """
        out = adapters(
            torch.randn(4, 9, config.protein_dim) * 100.0,
            torch.randn(4, 6, config.ligand_dim) * 0.01,
            torch.randn(4, config.ligand_graph_dim),
        )
        protein_scale = out["protein"].std().item()
        ligand_scale = out["ligand"].std().item()
        assert 0.2 < protein_scale / ligand_scale < 5.0, (protein_scale, ligand_scale)


class TestMaskedMean:
    def test_padding_is_excluded_from_the_mean(self):
        features = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]])
        mask = torch.tensor([[True, True, False]])
        assert torch.allclose(masked_mean(features, mask), torch.tensor([[2.0, 2.0]]))

    def test_a_plain_mean_would_have_been_wrong(self):
        """Names the bug: dividing by the padded length, not the real one."""
        features = torch.tensor([[[2.0], [2.0], [0.0], [0.0]]])
        mask = torch.tensor([[True, True, False, False]])
        assert masked_mean(features, mask).item() == pytest.approx(2.0)
        assert features.mean(dim=-2).item() == pytest.approx(1.0)

    def test_an_all_padded_row_gives_zero_not_nan(self):
        result = masked_mean(torch.randn(1, 3, 4), torch.zeros(1, 3, dtype=torch.bool))
        assert torch.isfinite(result).all()
        assert torch.equal(result, torch.zeros(1, 4))

    def test_a_non_boolean_mask_is_refused(self):
        with pytest.raises(AdapterError, match="must be bool"):
            masked_mean(torch.randn(1, 3, 4), torch.ones(1, 3))


class TestCrossAttentionShapes:
    def test_the_documented_axes_are_what_comes_out(self, stack):
        out = stack(
            torch.randn(3, 11, 64),
            torch.randn(3, 7, 64),
            torch.ones(3, 11, dtype=torch.bool),
            torch.ones(3, 7, dtype=torch.bool),
        )
        assert out.protein.shape == (3, 11, 64)
        assert out.ligand.shape == (3, 7, 64)
        assert out.num_layers == 2
        for p2l, l2p in zip(out.p2l_attention, out.l2p_attention):
            assert p2l.shape == (3, 4, 11, 7)  # [B, H, L, T]
            assert l2p.shape == (3, 4, 7, 11)  # [B, H, T, L]

    def test_protein_to_ligand_rows_are_distributions_over_atoms(self, stack):
        out = stack(
            torch.randn(2, 9, 64),
            torch.randn(2, 5, 64),
            torch.ones(2, 9, dtype=torch.bool),
            torch.ones(2, 5, dtype=torch.bool),
        )
        for p2l in out.p2l_attention:
            assert torch.allclose(p2l.sum(-1), torch.ones(2, 4, 9), atol=1e-5)

    def test_ligand_to_protein_rows_are_distributions_over_residues(self, stack):
        out = stack(
            torch.randn(2, 9, 64),
            torch.randn(2, 5, 64),
            torch.ones(2, 9, dtype=torch.bool),
            torch.ones(2, 5, dtype=torch.bool),
        )
        for l2p in out.l2p_attention:
            assert torch.allclose(l2p.sum(-1), torch.ones(2, 4, 5), atol=1e-5)

    def test_unadapted_features_are_refused_with_a_useful_message(self, stack):
        with pytest.raises(CrossAttentionError, match="SharedAdapters"):
            stack(
                torch.randn(2, 9, 1152),
                torch.randn(2, 5, 64),
                torch.ones(2, 9, dtype=torch.bool),
                torch.ones(2, 5, dtype=torch.bool),
            )

    def test_a_float_mask_is_refused(self, stack):
        with pytest.raises(CrossAttentionError, match="True = real"):
            stack(
                torch.randn(2, 9, 64),
                torch.randn(2, 5, 64),
                torch.ones(2, 9),
                torch.ones(2, 5, dtype=torch.bool),
            )

    def test_a_batch_mismatch_is_refused(self, stack):
        with pytest.raises(CrossAttentionError, match="batch mismatch"):
            stack(
                torch.randn(3, 9, 64),
                torch.randn(2, 5, 64),
                torch.ones(3, 9, dtype=torch.bool),
                torch.ones(2, 5, dtype=torch.bool),
            )


class TestMasking:
    def test_padding_does_not_change_a_real_sequence(self, stack):
        """The corruption that depends on batch composition.

        A 9-residue protein fused alone must give the same answer as the same
        protein padded to 20 alongside something longer.
        """
        torch.manual_seed(7)
        protein = torch.randn(1, 9, 64)
        ligand = torch.randn(1, 5, 64)

        tight = stack(
            protein,
            ligand,
            torch.ones(1, 9, dtype=torch.bool),
            torch.ones(1, 5, dtype=torch.bool),
        )

        padded_protein = torch.cat((protein, torch.randn(1, 11, 64)), dim=1)
        padded_ligand = torch.cat((ligand, torch.randn(1, 4, 64)), dim=1)
        protein_mask = torch.zeros(1, 20, dtype=torch.bool)
        protein_mask[:, :9] = True
        ligand_mask = torch.zeros(1, 9, dtype=torch.bool)
        ligand_mask[:, :5] = True
        loose = stack(padded_protein, padded_ligand, protein_mask, ligand_mask)

        assert torch.allclose(loose.protein[:, :9], tight.protein, atol=1e-5)
        assert torch.allclose(loose.ligand[:, :5], tight.ligand, atol=1e-5)

    def test_no_attention_mass_lands_on_padding(self, stack):
        protein_mask = torch.zeros(2, 10, dtype=torch.bool)
        protein_mask[:, :4] = True
        ligand_mask = torch.zeros(2, 8, dtype=torch.bool)
        ligand_mask[:, :3] = True
        out = stack(torch.randn(2, 10, 64), torch.randn(2, 8, 64), protein_mask, ligand_mask)
        for p2l in out.p2l_attention:
            assert p2l[..., 3:].abs().sum() == 0.0  # no mass on padded atoms
        for l2p in out.l2p_attention:
            assert l2p[..., 4:].abs().sum() == 0.0  # no mass on padded residues

    def test_padded_query_positions_are_zeroed(self, stack):
        protein_mask = torch.zeros(1, 6, dtype=torch.bool)
        protein_mask[:, :2] = True
        out = stack(
            torch.randn(1, 6, 64),
            torch.randn(1, 3, 64),
            protein_mask,
            torch.ones(1, 3, dtype=torch.bool),
        )
        assert out.protein[:, 2:].abs().sum() == 0.0

    def test_an_all_padded_side_does_not_produce_nan(self, stack):
        """Reachable whenever a batch holds a molecule that failed featurization.

        Softmax over an entirely masked row is softmax over all -inf, which is
        NaN, and one NaN propagates through the whole batch.
        """
        out = stack(
            torch.randn(2, 6, 64),
            torch.randn(2, 4, 64),
            torch.ones(2, 6, dtype=torch.bool),
            torch.zeros(2, 4, dtype=torch.bool),  # no real ligand tokens at all
        )
        assert torch.isfinite(out.protein).all()
        assert torch.isfinite(out.ligand).all()
        for p2l in out.p2l_attention:
            assert torch.isfinite(p2l).all()
            assert p2l.abs().sum() == 0.0

    def test_one_all_padded_row_does_not_poison_its_neighbours(self, stack):
        """A single bad row in a batch must not take the batch with it."""
        ligand_mask = torch.ones(2, 4, dtype=torch.bool)
        ligand_mask[1] = False
        out = stack(
            torch.randn(2, 6, 64),
            torch.randn(2, 4, 64),
            torch.ones(2, 6, dtype=torch.bool),
            ligand_mask,
        )
        assert torch.isfinite(out.protein).all()
        for p2l in out.p2l_attention:
            assert torch.allclose(p2l[0].sum(-1), torch.ones(4, 6), atol=1e-5)
            assert p2l[1].abs().sum() == 0.0


class TestDirectionality:
    def test_the_two_directions_are_not_transposes(self, stack):
        """The specification's requirement, as an assertion.

        Computing one score matrix and reading it along both axes would make
        ``l2p`` the transpose of ``p2l``. It must not be.
        """
        out = stack(
            torch.randn(2, 9, 64),
            torch.randn(2, 5, 64),
            torch.ones(2, 9, dtype=torch.bool),
            torch.ones(2, 5, dtype=torch.bool),
        )
        for p2l, l2p in zip(out.p2l_attention, out.l2p_attention):
            assert not torch.allclose(p2l, l2p.transpose(-2, -1), atol=1e-3)

    def test_the_directions_have_independent_parameters(self, stack):
        layer = stack.layers[0]
        p2l = dict(layer.protein_to_ligand.named_parameters())
        l2p = dict(layer.ligand_to_protein.named_parameters())
        assert set(p2l) == set(l2p)
        shared = [
            name for name in p2l if p2l[name].data_ptr() == l2p[name].data_ptr()
        ]
        assert not shared, f"these parameters are shared between directions: {shared}"

    def test_neither_direction_sees_the_other_s_output(self):
        """Applied in parallel, so the returned maps describe the inputs.

        Sequential application would make ``l2p_attention`` describe a protein
        that no longer exists by the time the layer returns.
        """
        torch.manual_seed(3)
        config = CrossAttentionConfig(model_dim=32, num_heads=4, num_layers=1, dropout=0.0)
        stack = CrossAttentionStack(config)
        stack.eval()
        protein, ligand = torch.randn(1, 5, 32), torch.randn(1, 4, 32)
        p_mask = torch.ones(1, 5, dtype=torch.bool)
        l_mask = torch.ones(1, 4, dtype=torch.bool)
        out = stack(protein, ligand, p_mask, l_mask)

        # Recompute l2p directly from the *inputs*; it must match what the layer
        # reported, which it only can if the layer did not feed it p2l's output.
        expected = stack.layers[0].ligand_to_protein(ligand, protein, p_mask)[1]
        assert torch.allclose(out.l2p_attention[0], expected, atol=1e-6)


class TestContactMap:
    def test_it_is_a_derived_aggregation_not_a_distribution(self, stack):
        """Rows do not sum to 1, and nothing may present them as if they did."""
        out = stack(
            torch.randn(2, 9, 64),
            torch.randn(2, 5, 64),
            torch.ones(2, 9, dtype=torch.bool),
            torch.ones(2, 5, dtype=torch.bool),
        )
        contact = out.contact_map()
        assert contact.shape == (2, 9, 5)
        assert not torch.allclose(contact.sum(-1), torch.ones(2, 9), atol=1e-2)

    def test_padded_pairs_are_zero(self, stack):
        protein_mask = torch.zeros(1, 8, dtype=torch.bool)
        protein_mask[:, :3] = True
        ligand_mask = torch.zeros(1, 6, dtype=torch.bool)
        ligand_mask[:, :2] = True
        out = stack(torch.randn(1, 8, 64), torch.randn(1, 6, 64), protein_mask, ligand_mask)
        contact = out.contact_map()
        assert contact[:, 3:, :].abs().sum() == 0.0
        assert contact[:, :, 2:].abs().sum() == 0.0

    def test_the_min_reduction_is_no_greater_than_the_mean(self, stack):
        """Agreement of both directions is the stricter claim."""
        out = stack(
            torch.randn(2, 9, 64),
            torch.randn(2, 5, 64),
            torch.ones(2, 9, dtype=torch.bool),
            torch.ones(2, 5, dtype=torch.bool),
        )
        assert (out.contact_map(reduction="min") <= out.contact_map() + 1e-6).all()

    def test_an_unknown_reduction_is_refused(self, stack):
        out = stack(
            torch.randn(1, 4, 64),
            torch.randn(1, 3, 64),
            torch.ones(1, 4, dtype=torch.bool),
            torch.ones(1, 3, dtype=torch.bool),
        )
        with pytest.raises(CrossAttentionError, match="unknown reduction"):
            out.contact_map(reduction="whatever")


class TestGradientAttribution:
    """The specification's "log which parameters receive gradients from each loss"."""

    def test_every_loss_reaches_the_shared_adapters(self, adapters, config):
        out = adapters(
            torch.randn(2, 7, config.protein_dim),
            torch.randn(2, 5, config.ligand_dim),
            torch.randn(2, config.ligand_graph_dim),
        )
        losses = {
            "affinity": out["protein"].sum() + out["ligand"].sum(),
            "retrieval": out["ligand"].pow(2).mean(),
            "diffusion": out["ligand_graph"].abs().mean(),
        }
        attribution = attribute_gradients(adapters, losses)
        assert_shared(attribution, prefix="ligand", losses=["affinity", "retrieval"])
        assert_shared(attribution, prefix="ligand_graph", losses=["diffusion"])

    def test_a_loss_that_touches_only_one_stream_is_reported_as_such(
        self, adapters, config
    ):
        out = adapters(
            torch.randn(2, 7, config.protein_dim), torch.randn(2, 5, config.ligand_dim)
        )
        attribution = attribute_gradients(adapters, {"protein_only": out["protein"].sum()})
        touched = attribution["protein_only"]
        assert any(name.startswith("protein.") for name in touched)
        assert not any(name.startswith("ligand.") for name in touched)

    def test_a_detached_branch_is_caught(self, adapters, config):
        """The failure mode: a stray detach that silently unshares the adapter.

        It trains, it converges, it reports plausible numbers, and the multi-task
        claim is false. Nothing in a loss curve shows it.
        """
        out = adapters(
            torch.randn(2, 7, config.protein_dim), torch.randn(2, 5, config.ligand_dim)
        )
        attribution = attribute_gradients(
            adapters, {"broken": out["ligand"].detach().sum() + out["protein"].sum()}
        )
        with pytest.raises(AdapterError, match="not shared"):
            assert_shared(attribution, prefix="ligand.", losses=["broken"])

    def test_losses_are_attributed_separately_not_to_their_sum(self, adapters, config):
        out = adapters(
            torch.randn(2, 7, config.protein_dim), torch.randn(2, 5, config.ligand_dim)
        )
        attribution = attribute_gradients(
            adapters,
            {"a": out["protein"].sum(), "b": out["ligand"].sum()},
        )
        assert not any(n.startswith("ligand.") for n in attribution["a"])
        assert not any(n.startswith("protein.") for n in attribution["b"])

    def test_gradients_are_left_clean_afterwards(self, adapters, config):
        out = adapters(
            torch.randn(2, 7, config.protein_dim), torch.randn(2, 5, config.ligand_dim)
        )
        attribute_gradients(adapters, {"a": out["protein"].sum()})
        assert all(p.grad is None for p in adapters.parameters())

    def test_a_module_without_trainable_parameters_reports_nothing(self):
        frozen = nn.Linear(4, 4)
        for parameter in frozen.parameters():
            parameter.requires_grad_(False)
        assert attribute_gradients(frozen, {"loss": torch.zeros(1, requires_grad=True).sum()}) == {
            "loss": ()
        }


class TestStackGradients:
    def test_gradient_reaches_every_parameter_of_both_directions(self):
        stack = CrossAttentionStack(
            CrossAttentionConfig(model_dim=32, num_heads=4, num_layers=2, dropout=0.0)
        )
        out = stack(
            torch.randn(2, 6, 32),
            torch.randn(2, 4, 32),
            torch.ones(2, 6, dtype=torch.bool),
            torch.ones(2, 4, dtype=torch.bool),
        )
        (out.protein.sum() + out.ligand.sum()).backward()
        missing = [
            name
            for name, parameter in stack.named_parameters()
            if parameter.grad is None or parameter.grad.abs().sum() == 0
        ]
        assert not missing, f"no gradient reached: {missing}"


@pytest.mark.gpu
class TestOnCuda:
    def test_the_stack_runs_on_the_gpu_and_agrees_with_the_cpu(self):
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")
        torch.manual_seed(20261001)
        config = CrossAttentionConfig(model_dim=64, num_heads=4, num_layers=2, dropout=0.0)
        cpu = CrossAttentionStack(config).eval()
        gpu = CrossAttentionStack(config).eval().cuda()
        gpu.load_state_dict(cpu.state_dict())

        protein, ligand = torch.randn(2, 11, 64), torch.randn(2, 7, 64)
        p_mask = torch.ones(2, 11, dtype=torch.bool)
        l_mask = torch.ones(2, 7, dtype=torch.bool)
        a = cpu(protein, ligand, p_mask, l_mask)
        b = gpu(protein.cuda(), ligand.cuda(), p_mask.cuda(), l_mask.cuda())
        assert torch.allclose(a.protein, b.protein.cpu(), atol=1e-4)
        assert torch.allclose(
            a.p2l_attention[0], b.p2l_attention[0].cpu(), atol=1e-4
        )


class TestFusedAttentionPath:
    """``need_attention=False`` skips materialising the maps.

    Not a micro-optimisation: the explicit path builds several [B, H, Lq, Lk]
    tensors per direction per layer and retains them for backward, which is why
    a 5.7M-parameter model peaked at 9.89 GiB. Training does not need the maps;
    iBAM does. The two paths must agree, or the maps iBAM exports would describe
    a model that was never trained.
    """

    def test_the_two_paths_give_the_same_output(self, stack):
        torch.manual_seed(11)
        protein, ligand = torch.randn(3, 9, 64), torch.randn(3, 5, 64)
        p_mask = torch.ones(3, 9, dtype=torch.bool)
        l_mask = torch.ones(3, 5, dtype=torch.bool)
        with_maps = stack(protein, ligand, p_mask, l_mask, need_attention=True)
        without = stack(protein, ligand, p_mask, l_mask, need_attention=False)
        assert torch.allclose(with_maps.protein, without.protein, atol=1e-5)
        assert torch.allclose(with_maps.ligand, without.ligand, atol=1e-5)

    def test_they_agree_under_padding_too(self, stack):
        """Where the fused kernel's mask handling could silently differ."""
        torch.manual_seed(13)
        protein, ligand = torch.randn(2, 12, 64), torch.randn(2, 7, 64)
        p_mask = torch.zeros(2, 12, dtype=torch.bool)
        p_mask[:, :5] = True
        l_mask = torch.zeros(2, 7, dtype=torch.bool)
        l_mask[0, :3] = True
        l_mask[1, :6] = True
        a = stack(protein, ligand, p_mask, l_mask, need_attention=True)
        b = stack(protein, ligand, p_mask, l_mask, need_attention=False)
        assert torch.allclose(a.protein, b.protein, atol=1e-5)
        assert torch.allclose(a.ligand, b.ligand, atol=1e-5)

    def test_an_all_padded_side_still_gives_zeros_not_nan(self, stack):
        """The fused kernel NaNs on a fully masked row unless it is widened."""
        protein = torch.randn(2, 6, 64)
        ligand = torch.randn(2, 4, 64)
        l_mask = torch.ones(2, 4, dtype=torch.bool)
        l_mask[1] = False
        a = stack(protein, ligand, torch.ones(2, 6, dtype=torch.bool), l_mask,
                  need_attention=True)
        b = stack(protein, ligand, torch.ones(2, 6, dtype=torch.bool), l_mask,
                  need_attention=False)
        assert torch.isfinite(b.protein).all()
        assert torch.allclose(a.protein, b.protein, atol=1e-5)

    def test_no_maps_are_returned_when_they_are_not_asked_for(self, stack):
        out = stack(
            torch.randn(2, 9, 64), torch.randn(2, 5, 64),
            torch.ones(2, 9, dtype=torch.bool), torch.ones(2, 5, dtype=torch.bool),
            need_attention=False,
        )
        assert out.p2l_attention == () and out.l2p_attention == ()

    def test_gradients_match_between_the_paths(self, stack):
        """The optimisation must not change what the model learns."""
        def grads(flag):
            torch.manual_seed(5)
            protein = torch.randn(2, 9, 64, requires_grad=True)
            ligand = torch.randn(2, 5, 64, requires_grad=True)
            out = stack(protein, ligand, torch.ones(2, 9, dtype=torch.bool),
                        torch.ones(2, 5, dtype=torch.bool), need_attention=flag)
            (out.protein.sum() + out.ligand.sum()).backward()
            return protein.grad.clone(), ligand.grad.clone()

        a_p, a_l = grads(True)
        b_p, b_l = grads(False)
        assert torch.allclose(a_p, b_p, atol=1e-5)
        assert torch.allclose(a_l, b_l, atol=1e-5)
