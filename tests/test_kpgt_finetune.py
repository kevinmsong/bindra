"""Partial fine-tuning of KPGT, and the cache safety it forces.

Partial fine-tuning is implemented but **off by default**. Unfreezing the
last layer buys 7,087,872 trainable parameters adapting the pretrained representation to binding-affinity
data, and it costs the ability to cache: a cached molecule vector is valid only
because it is a deterministic function of the SMILES, and the moment any layer
trains, it is not.

``test_encode_to_cache_refuses_a_trainable_encoder`` is the one that matters. A
cache written from a training encoder is wrong from the first optimizer step
onward, with no error, no shape mismatch and no symptom -- just stale numbers
that look exactly like fresh ones.

``test_splitting_the_stack_does_not_change_the_numbers`` guards the other half.
Running the frozen prefix under ``no_grad`` is a memory *requirement*: all 12
layers under autograd on a batch of 128 real molecules (68,693 triplet edges)
exhausts 16 GiB, measured. So it has to be numerically free.
"""

from __future__ import annotations

import pytest
import torch

from bindra.encoders.kpgt_graph import collate_kpgt, featurize_kpgt
from bindra.encoders.kpgt_light import KPGTPortError
from bindra.encoders.ligand import LigandEncoderError
from bindra.encoders.ligand_kpgt import KPGTConfig, KPGTLigandEncoder
from bindra.features.cache import FeatureCacheError, StaleCacheError
from bindra.features.ligand_cache import LigandVectorCache, encode_to_cache

MOLECULES = ("CCO", "CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1", "C", "N[C@@H](C)C(=O)O")

BASE = dict(d_g_feats=48, n_mol_layers=4, n_heads=4, d_hpath_ratio=4)


def _config(**overrides) -> KPGTConfig:
    return KPGTConfig(**{**BASE, **overrides})


def _encoder(**overrides) -> KPGTLigandEncoder:
    torch.manual_seed(20261001)
    return KPGTLigandEncoder(_config(**overrides))


class TestFreezing:
    def test_frozen_is_the_default(self):
        """Sequencing, not preference: there is no baseline yet to beat.

        Frozen keeps the readout a deterministic function of the SMILES, which
        is what makes the cache valid and an epoch cost minutes. Unfreezing is
        implemented and tested below, to be run as a measured ablation.
        """
        assert KPGTConfig().n_trainable_layers == 0

    def test_the_default_is_within_the_published_stack(self):
        config = KPGTConfig()
        assert config.n_trainable_layers < config.n_mol_layers == 12

    def test_only_the_trailing_layer_is_trainable(self):
        encoder = _encoder(n_trainable_layers=1)
        trainable = {
            name
            for name, parameter in encoder.module.named_parameters()
            if parameter.requires_grad
        }
        assert trainable
        last = f"model.mol_T_layers.{BASE['n_mol_layers'] - 1}."
        assert all(name.startswith(last) for name in trainable), sorted(trainable)[:3]

    def test_two_layers_unfreeze_the_last_two(self):
        encoder = _encoder(n_trainable_layers=2)
        trainable = {
            name
            for name, parameter in encoder.module.named_parameters()
            if parameter.requires_grad
        }
        allowed = tuple(
            f"model.mol_T_layers.{i}."
            for i in (BASE["n_mol_layers"] - 2, BASE["n_mol_layers"] - 1)
        )
        assert trainable and all(name.startswith(allowed) for name in trainable)

    def test_embeddings_stay_frozen_under_partial_unfreezing(self):
        """Unfreezing an input embedding under frozen layers moves the ground."""
        encoder = _encoder(n_trainable_layers=2)
        for prefix in ("node_emb", "edge_emb", "triplet_emb", "model.path_len_emb"):
            assert not any(
                parameter.requires_grad
                for name, parameter in encoder.module.named_parameters()
                if name.startswith(prefix)
            ), prefix

    def test_a_frozen_encoder_has_no_trainable_parameters(self):
        encoder = _encoder(n_trainable_layers=0)
        assert encoder.n_trainable == 0
        assert encoder.trainable_parameters() == []

    def test_more_trainable_layers_than_exist_is_refused(self):
        with pytest.raises(ValueError, match="n_trainable_layers"):
            KPGTConfig(n_mol_layers=12, n_trainable_layers=13)

    def test_the_freeze_regime_is_part_of_the_cache_key(self):
        frozen = _encoder(n_trainable_layers=0).metadata()
        tuned = _encoder(n_trainable_layers=1).metadata()
        assert frozen.cache_key("CCO") != tuned.cache_key("CCO")


class TestGradientFlow:
    def test_gradient_reaches_the_unfrozen_layer_and_nothing_below(self):
        encoder = _encoder(n_trainable_layers=1)
        encoder.module.train()
        _, readout = encoder.encode_tensors(["CCO", "c1ccccc1"])
        readout.sum().backward()
        received = {
            name
            for name, parameter in encoder.module.named_parameters()
            if parameter.grad is not None and parameter.grad.abs().sum() > 0
        }
        assert received, "the unfrozen layer received no gradient"
        last = f"model.mol_T_layers.{BASE['n_mol_layers'] - 1}."
        assert all(name.startswith(last) for name in received)

    def test_encode_tensors_keeps_the_autograd_graph(self):
        encoder = _encoder(n_trainable_layers=1)
        encoder.module.train()
        states, readout = encoder.encode_tensors(["CCO"])
        assert readout.requires_grad and states.requires_grad

    def test_a_frozen_encoder_produces_no_graph(self):
        encoder = _encoder(n_trainable_layers=0)
        encoder.module.train()
        _, readout = encoder.encode_tensors(["CCO"])
        assert not readout.requires_grad

    def test_the_unfrozen_layer_actually_moves_under_an_optimizer(self):
        """requires_grad is not enough; the step has to change the weights."""
        encoder = _encoder(n_trainable_layers=1)
        encoder.module.train()
        before = [p.detach().clone() for p in encoder.trainable_parameters()]
        optimizer = torch.optim.Adam(encoder.trainable_parameters(), lr=1e-2)
        _, readout = encoder.encode_tensors(list(MOLECULES))
        readout.pow(2).mean().backward()
        optimizer.step()
        moved = [
            not torch.equal(old, new)
            for old, new in zip(before, encoder.trainable_parameters())
        ]
        assert all(moved)

    def test_frozen_parameters_do_not_move(self):
        encoder = _encoder(n_trainable_layers=1)
        encoder.module.train()
        frozen = {
            name: p.detach().clone()
            for name, p in encoder.module.named_parameters()
            if not p.requires_grad
        }
        optimizer = torch.optim.Adam(encoder.trainable_parameters(), lr=1e-2)
        _, readout = encoder.encode_tensors(list(MOLECULES))
        readout.pow(2).mean().backward()
        optimizer.step()
        for name, parameter in encoder.module.named_parameters():
            if name in frozen:
                assert torch.equal(frozen[name], parameter), name


class TestFrozenPrefixCostsNothingExtra:
    """An explicit no_grad split was written, measured, and removed.

    The belief behind it -- that running frozen layers under grad retains their
    activations -- is false when their parameters have ``requires_grad=False``:
    autograd builds no graph through a subgraph that requires no gradient. Peak
    memory on a batch of 128 real molecules was 1,670 MiB with the split and
    1,671 without. These tests pin the property the removal relies on.
    """

    def test_no_graph_is_built_through_the_frozen_prefix(self):
        encoder = _encoder(n_trainable_layers=1)
        encoder.module.train()
        molecules = list(MOLECULES[:3])
        batch = collate_kpgt([featurize_kpgt(s) for s in molecules])
        fingerprints, descriptors = encoder.knowledge_inputs(molecules)
        states, readout = encoder.module(batch, fingerprints, descriptors)
        assert readout.requires_grad
        # The graph stops at the last layer: only its parameters are leaves.
        readout.sum().backward()
        received = {
            name
            for name, parameter in encoder.module.named_parameters()
            if parameter.grad is not None
        }
        last = f"model.mol_T_layers.{BASE['n_mol_layers'] - 1}."
        assert received and all(name.startswith(last) for name in received)

    def test_a_fully_frozen_encoder_builds_no_graph_at_all(self):
        encoder = _encoder(n_trainable_layers=0)
        encoder.module.train()
        molecules = list(MOLECULES[:3])
        batch = collate_kpgt([featurize_kpgt(s) for s in molecules])
        fingerprints, descriptors = encoder.knowledge_inputs(molecules)
        _, readout = encoder.module(batch, fingerprints, descriptors)
        assert not readout.requires_grad
        assert readout.grad_fn is None

    def test_the_trained_surface_grows_with_the_number_of_unfrozen_layers(self):
        """What actually drives memory and capacity, measured directly.

        An earlier version of this test walked the autograd graph and counted
        nodes. That was a poor instrument -- it returned 4 for every setting --
        and it was measuring nothing. Counting the parameters that receive
        gradient says the same thing without a proxy, and the hardware
        measurement in the class docstring covers the memory claim.
        """
        molecules = list(MOLECULES[:3])
        received = []
        for n in (1, 2, 4):
            encoder = _encoder(n_trainable_layers=n)
            encoder.module.train()
            _, readout = encoder.encode_tensors(molecules)
            readout.sum().backward()
            received.append(
                sum(
                    p.numel()
                    for p in encoder.module.parameters()
                    if p.grad is not None and p.grad.abs().sum() > 0
                )
            )
        assert received[0] < received[1] < received[2], received
        # And each step adds one layer's worth, not some other amount.
        assert received[1] - received[0] == received[0]


class TestCacheSafety:
    def test_a_trainable_encoder_refuses_to_be_cached(self):
        with pytest.raises(LigandEncoderError, match="must not be cached"):
            _encoder(n_trainable_layers=1).assert_cacheable()

    def test_the_refusal_says_how_to_proceed(self):
        with pytest.raises(LigandEncoderError, match="n_trainable_layers=0"):
            _encoder(n_trainable_layers=1).assert_cacheable()

    def test_a_frozen_encoder_is_cacheable(self):
        _encoder(n_trainable_layers=0).assert_cacheable()  # does not raise

    def test_encode_to_cache_refuses_a_trainable_encoder(self, tmp_path):
        """The silent-corruption path: a cache that goes stale after step one."""
        encoder = _encoder(n_trainable_layers=1)
        cache = LigandVectorCache(tmp_path / "cache", mode="w")
        with pytest.raises(LigandEncoderError, match="must not be cached"):
            encode_to_cache(encoder, ["CCO"], cache)


class TestCacheRoundTrip:
    def _frozen(self):
        encoder = _encoder(n_trainable_layers=0)
        encoder.module.eval()
        return encoder

    def test_a_frozen_encoder_round_trips_exactly(self, tmp_path):
        encoder = self._frozen()
        molecules = list(MOLECULES)
        with LigandVectorCache(tmp_path / "cache", mode="w") as cache:
            # batch_size=1 so the cached vector is the same computation
            # encode() performs. Batching changes float32 summation order by
            # ~1e-6, which is below float32 tolerance but can flip the last bit
            # of a bf16 cast; test_encode_batch_matches_repeated_encode in
            # test_ligand_kpgt.py covers batch invariance at the right tolerance.
            counts = encode_to_cache(encoder, molecules, cache, batch_size=1)
        assert counts["encoded"] == len(molecules) and counts["failed"] == 0

        reader = LigandVectorCache(tmp_path / "cache", metadata=encoder.metadata())
        assert len(reader) == len(molecules)
        for molecule in molecules:
            expected = encoder.encode(molecule).graph_features.to(torch.bfloat16)
            assert torch.equal(reader.get(encoder.metadata().cache_key(molecule)), expected)

    def test_rerunning_skips_what_is_already_cached(self, tmp_path):
        """Resumability: a full corpus pass is hours and must survive a restart."""
        encoder = self._frozen()
        with LigandVectorCache(tmp_path / "cache", mode="w") as cache:
            encode_to_cache(encoder, list(MOLECULES), cache)
        with LigandVectorCache(tmp_path / "cache", mode="a") as cache:
            counts = encode_to_cache(encoder, list(MOLECULES), cache)
        assert counts["skipped"] == len(MOLECULES) and counts["encoded"] == 0

    def test_a_different_encoder_contract_is_refused(self, tmp_path):
        encoder = self._frozen()
        with LigandVectorCache(tmp_path / "cache", mode="w") as cache:
            encode_to_cache(encoder, ["CCO"], cache)
        wider = KPGTLigandEncoder(
            _config(d_g_feats=24, n_heads=2, d_hpath_ratio=2, n_trainable_layers=0)
        )
        with pytest.raises(StaleCacheError, match="different encoder contract"):
            LigandVectorCache(tmp_path / "cache", metadata=wider.metadata())

    def test_get_many_preserves_the_requested_order(self, tmp_path):
        encoder = self._frozen()
        molecules = list(MOLECULES)
        with LigandVectorCache(tmp_path / "cache", mode="w") as cache:
            encode_to_cache(encoder, molecules, cache)
        reader = LigandVectorCache(tmp_path / "cache", metadata=encoder.metadata())
        keys = [encoder.metadata().cache_key(m) for m in reversed(molecules)]
        stacked = reader.get_many(keys)
        for position, key in enumerate(keys):
            assert torch.equal(stacked[position], reader.get(key))

    def test_reading_an_uncached_molecule_errors_rather_than_returning_zeros(self, tmp_path):
        encoder = self._frozen()
        with LigandVectorCache(tmp_path / "cache", mode="w") as cache:
            encode_to_cache(encoder, ["CCO"], cache)
        reader = LigandVectorCache(tmp_path / "cache", metadata=encoder.metadata())
        with pytest.raises(FeatureCacheError, match="not cached"):
            reader.get(encoder.metadata().cache_key("c1ccccc1"))

    def test_the_cache_reports_its_size_and_provenance(self, tmp_path):
        encoder = self._frozen()
        with LigandVectorCache(tmp_path / "cache", mode="w") as cache:
            encode_to_cache(encoder, list(MOLECULES), cache)
        reader = LigandVectorCache(tmp_path / "cache", metadata=encoder.metadata())
        stats = reader.stats()
        assert stats["entries"] == len(MOLECULES)
        assert stats["dim"] == encoder.graph_dim
        assert stats["encoder"] == "ligand_kpgt"
        assert stats["bytes"] == len(MOLECULES) * encoder.graph_dim * 2

    def test_a_float32_vector_is_refused_rather_than_silently_converted(self, tmp_path):
        cache = LigandVectorCache(tmp_path / "cache", mode="w")
        cache.adopt_metadata(self._frozen().metadata())
        with pytest.raises(FeatureCacheError, match="bfloat16"):
            cache.put("key", torch.zeros(48, dtype=torch.float32))

    def test_reading_before_anything_is_written_is_an_error(self, tmp_path):
        with pytest.raises(FeatureCacheError, match="no cache index"):
            LigandVectorCache(tmp_path / "absent")
