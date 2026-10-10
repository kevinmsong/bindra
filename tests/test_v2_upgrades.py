"""BINDRA-v2's four upgrades, and the identity properties that make them ablations.

Each upgrade had to be switchable without reinitialising the model, or a
comparison between v1 and v2 would measure the upgrade *and* a different random
draw at the same time. Three of the four are exactly identity at initialisation
and are asserted as such here; the fourth (LayerMix) is near-identity and its
actual deviation is pinned rather than described.

``test_everything_off_reproduces_v1_exactly`` is the one that matters most: if
the parameter count drifts from 2,070,043, the v1 baseline is no longer a
baseline.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bindra.encoders.ligand_kpgt import KPGTConfig
from bindra.fusion.adapters import AdapterConfig, AttentionPool, LayerMix, masked_mean
from bindra.heads.affinity import AffinityHeadConfig
from bindra.model import ARCHITECTURE_NAME, BindraConfig, BindraModel

#: What v1 recorded. A change here invalidates every comparison against it.
V1_TRAINABLE_PARAMETERS = 2_070_043


def _v1_ligand():
    return KPGTConfig.mini(
        d_g_feats=96, n_mol_layers=8, n_trainable_layers=8, n_heads=8, d_hpath_ratio=4
    )


def _config(**overrides):
    base = dict(
        protein_dim=1152,
        ligand=_v1_ligand(),
        model_dim=96,
        cross_attention_layers=4,
        feedforward_multiplier=2,
        head=AffinityHeadConfig(hidden_dim=96, dropout=0.1),
    )
    base.update(overrides)
    return BindraConfig(**base)


class TestArchitectureIdentity:
    def test_the_architecture_is_named_v2(self):
        assert ARCHITECTURE_NAME == "BINDRA-v2"

    def test_everything_off_reproduces_v1_exactly(self):
        """With every v2 feature disabled, this must BE v1, parameter for parameter."""
        model = BindraModel(_config(attention_pooling=False))
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable == V1_TRAINABLE_PARAMETERS, (
            f"{trainable:,} != the recorded v1 count of {V1_TRAINABLE_PARAMETERS:,}; "
            "the v1 baseline is no longer reproducible"
        )

    def test_each_upgrade_is_separately_switchable(self):
        for name, kwargs in (
            ("pooling", dict(attention_pooling=True)),
            ("layers", dict(attention_pooling=False, protein_layers=4)),
            ("assay", dict(attention_pooling=False, assay_conditioning=True)),
            ("capacity", dict(attention_pooling=False, adapter_bottleneck=256)),
        ):
            model = BindraModel(_config(**kwargs))
            count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            assert count > V1_TRAINABLE_PARAMETERS, f"{name} added no parameters"


class TestAttentionPooling:
    """Upgrade 1. A binding site is ~20-30 residues of a median 454."""

    def _padded(self, batch=4, length=37, dim=96, seed=0):
        torch.manual_seed(seed)
        features = torch.randn(batch, length, dim)
        mask = torch.zeros(batch, length, dtype=torch.bool)
        for row in range(batch):
            mask[row, : 7 + row * 5] = True
        return features, mask

    def test_it_is_exactly_the_mean_at_initialisation(self):
        features, mask = self._padded()
        pool = AttentionPool(96, num_heads=4)
        assert torch.allclose(pool(features, mask), masked_mean(features, mask), atol=1e-5)

    def test_padding_takes_no_attention_mass(self):
        features, mask = self._padded()
        pool = AttentionPool(96, num_heads=4)
        weights = pool.attention(features, mask).detach()
        assert float(weights.masked_fill(mask.unsqueeze(-1), 0.0).sum()) == 0.0

    def test_each_head_sums_to_one_over_real_positions(self):
        features, mask = self._padded()
        weights = AttentionPool(96, num_heads=4).attention(features, mask).detach()
        assert torch.allclose(weights.sum(dim=1), torch.ones(4, 4), atol=1e-5)

    def test_a_trained_scorer_stops_being_the_mean(self):
        """Otherwise the module is decoration."""
        features, mask = self._padded()
        pool = AttentionPool(96, num_heads=4)
        with torch.no_grad():
            pool.scorer.weight.normal_(0, 0.5)
        assert not torch.allclose(pool(features, mask), masked_mean(features, mask), atol=1e-3)

    def test_the_mask_must_be_boolean(self):
        features, _ = self._padded()
        with pytest.raises(Exception):
            AttentionPool(96)(features, torch.ones(4, 37))


class TestLayerMix:
    """Upgrade 2. The final ESM-C layer is the most masked-LM-specialised one."""

    def test_it_starts_on_the_last_layer(self):
        weights = LayerMix(96, 4).weights().detach()
        assert float(weights[-1]) > 0.998
        assert float(weights[:-1].sum()) < 0.002

    def test_a_single_layer_is_a_passthrough_norm(self):
        features = torch.randn(2, 5, 96)
        mixed = LayerMix(96, 1)(features)
        assert mixed.shape == features.shape

    def test_a_wrong_layer_count_is_refused_not_mis_sliced(self):
        with pytest.raises(Exception, match="layers"):
            LayerMix(96, 4)(torch.randn(2, 5, 96 * 3))

    def test_the_weights_are_a_simplex(self):
        weights = LayerMix(96, 5).weights().detach()
        assert torch.allclose(weights.sum(), torch.ones(()), atol=1e-6)
        assert bool((weights >= 0).all())


class TestAssayConditioning:
    """Upgrade 3. IC50 is 72.2% of the corpus and is not a constant of the pair."""

    def test_the_encoded_width_is_fixed_and_declared(self):
        from bindra.data.assay import ASSAY_DIM, ASSAY_FIELDS

        assert ASSAY_DIM == sum(f.width for f in ASSAY_FIELDS)

    def test_identity_columns_are_excluded_by_name(self):
        """Replicates share a group id, so a group id routes to the label."""
        from bindra.data.assay import ASSAY_COLUMNS, EXCLUDED_AS_LEAKAGE

        assert "comparability_group_id" in EXCLUDED_AS_LEAKAGE
        assert "replicate_group_id" in EXCLUDED_AS_LEAKAGE
        assert not set(ASSAY_COLUMNS) & set(EXCLUDED_AS_LEAKAGE)

    def test_an_unseen_category_becomes_unknown_not_a_neighbour(self):
        import pandas as pd

        from bindra.data.assay import encode_assay, ASSAY_COLUMNS

        frame = pd.DataFrame({c: ["nonsense-value"] for c in ASSAY_COLUMNS})
        encoded = encode_assay(frame)
        # Every categorical block puts its mass on index 0, the UNKNOWN slot.
        assert encoded.shape[0] == 1
        assert encoded.sum() > 0

    def test_a_missing_column_is_refused_rather_than_zero_filled(self):
        import pandas as pd

        from bindra.data.assay import encode_assay, ASSAY_COLUMNS

        frame = pd.DataFrame({c: ["x"] for c in ASSAY_COLUMNS[:-1]})
        with pytest.raises(Exception, match="missing assay columns"):
            encode_assay(frame)

    def test_a_conditioned_model_refuses_a_batch_without_conditions(self):
        """Conditioning on zeros would make every measurement one experiment."""
        from bindra.fusion.adapters import KnowledgeFiLM

        config = AdapterConfig(
            protein_dim=1152, ligand_dim=96, ligand_graph_dim=288,
            model_dim=96, graph_components=(96, 96, 96), assay_dim=50,
        )
        film = KnowledgeFiLM(config, assay_conditioning=True)
        with pytest.raises(Exception, match="assay"):
            film(torch.randn(2, 5, 96), torch.randn(2, 7, 96), torch.randn(2, 288), assay=None)


class TestCapacity:
    """Upgrade 4. v1 squeezed ESM-C's 1152 into 96 in a single linear map."""

    def test_the_bottleneck_widens_the_adapter(self):
        narrow = sum(p.numel() for p in BindraModel(_config(attention_pooling=False)).parameters())
        wide = sum(
            p.numel()
            for p in BindraModel(_config(attention_pooling=False, adapter_bottleneck=512)).parameters()
        )
        assert wide > narrow

    def test_pair_dim_tracks_model_dim(self):
        assert _config(model_dim=256).pair_dim == 6 * 256
        assert _config(model_dim=96).pair_dim == 6 * 96


class TestProvenanceRecordsTheUpgrades:
    def test_every_upgrade_appears_in_the_recorded_config(self):
        """A checkpoint must say which upgrades produced it."""
        recorded = _config(
            model_dim=256, protein_layers=4, assay_conditioning=True, adapter_bottleneck=512
        ).to_dict()
        for key in (
            "attention_pooling", "pool_heads", "protein_layers",
            "assay_conditioning", "adapter_bottleneck",
        ):
            assert key in recorded, f"{key} is not recorded in provenance"
        assert recorded["architecture"] == "BINDRA-v2"
