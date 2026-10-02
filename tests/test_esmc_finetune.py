"""Partial fine-tuning of ESM-C, and the 14 GiB cache it invalidates.

The last three transformer blocks plus the final LayerNorm are unfrozen on
request. Two properties have to hold, and both are failure modes that produce
plausible numbers rather than errors:

``test_the_protein_cache_refuses_a_trainable_encoder``
    The 14.023 GiB protein cache is valid only because features are a
    deterministic function of the sequence. One trainable block ends that, and a
    cache written afterwards is stale from the first optimizer step with nothing
    to signal it.
``test_nothing_below_the_unfrozen_blocks_moves``
    Unfreezing an embedding or a lower block under frozen layers above moves the
    representation out from under them.

Most tests here run against a small module double rather than the 573M-parameter
checkpoint, because what is under test is the freezing *policy*, not ESM-C.
Tests that need the real weights are marked ``encoders``.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from bindra.encoders.protein_esmc import ESMCProteinEncoder, ProteinEncoderError


class _Block(nn.Module):
    """Stands in for one ESM-C transformer block."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.attn = nn.Linear(dim, dim)
        self.ffn = nn.Linear(dim, dim)

    def forward(self, x):  # pragma: no cover - shape plumbing only
        return self.ffn(torch.relu(self.attn(x)))


class _Transformer(nn.Module):
    def __init__(self, n_layers: int = 6, dim: int = 8) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(_Block(dim) for _ in range(n_layers))
        self.norm = nn.LayerNorm(dim)


class _Model(nn.Module):
    """ESM-C's shape: an embedding, a block stack, a final norm."""

    def __init__(self, n_layers: int = 6, dim: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(32, dim)
        self.transformer = _Transformer(n_layers, dim)


def _encoder(n_trainable_layers: int = 0, n_layers: int = 6) -> ESMCProteinEncoder:
    encoder = ESMCProteinEncoder(
        model=_Model(n_layers),
        tokenizer=object(),
        precision="float32",
        n_trainable_layers=n_trainable_layers,
    )
    encoder.apply_freezing()
    return encoder


class TestFreezingPolicy:
    def test_frozen_is_the_default(self):
        """Frozen keeps the 14 GiB cache valid and random batching viable.

        A trainable protein encoder forces target-grouped batching, because
        otherwise an epoch costs 43+ hours. Unfreezing is implemented and
        tested here, to be run as a measured ablation against a baseline.
        """
        assert ESMCProteinEncoder(model=_Model(), tokenizer=object()).n_trainable_layers == 0

    def test_the_cache_cli_pins_frozen_rather_than_inheriting_the_default(self):
        """Caching requires a frozen encoder, so that path states it itself."""
        import inspect

        from bindra.cli import cache as cache_cli

        assert "n_trainable_layers=0" in inspect.getsource(cache_cli._encoder)

    def test_one_layer_unfreezes_only_the_last_block(self):
        encoder = _encoder(1, n_layers=6)
        unfrozen = {
            name
            for name, parameter in encoder.model.named_parameters()
            if parameter.requires_grad
        }
        assert any(".blocks.5." in name for name in unfrozen)
        assert not any(f".blocks.{i}." in name for i in range(5) for name in unfrozen)

    def test_freezing_is_applied_at_construction_not_only_after_a_load(self):
        """An injected model used to skip freezing entirely, because
        ``_ensure_loaded`` returns early when one is supplied."""
        encoder = ESMCProteinEncoder(
            model=_Model(6), tokenizer=object(), n_trainable_layers=1
        )
        assert encoder.n_trainable > 0
        assert not any(
            p.requires_grad for n, p in encoder.model.named_parameters()
            if n.startswith("embed")
        )

    def test_a_non_module_double_is_left_alone(self):
        """Its parameters are the caller's to manage."""
        ESMCProteinEncoder(model=lambda _t: None, tokenizer=object())

    def test_a_frozen_encoder_has_no_trainable_parameters(self):
        encoder = _encoder(0)
        assert encoder.n_trainable == 0
        assert encoder.is_frozen

    def test_three_layers_unfreezes_the_last_three_blocks(self):
        encoder = _encoder(3, n_layers=6)
        unfrozen = {
            name
            for name, parameter in encoder.model.named_parameters()
            if parameter.requires_grad
        }
        assert any(".blocks.3." in name for name in unfrozen)
        assert any(".blocks.4." in name for name in unfrozen)
        assert any(".blocks.5." in name for name in unfrozen)

    def test_nothing_below_the_unfrozen_blocks_moves(self):
        """The embedding and the lower blocks stay fixed."""
        encoder = _encoder(3, n_layers=6)
        for name, parameter in encoder.model.named_parameters():
            below = name.startswith("embed") or any(
                f".blocks.{i}." in name for i in (0, 1, 2)
            )
            if below:
                assert not parameter.requires_grad, name

    def test_the_final_norm_is_unfrozen_with_the_blocks(self):
        """It sits above them and features are read after it.

        Leaving it fixed would force the unfrozen blocks to preserve a
        normalisation statistic they are otherwise free to change.
        """
        encoder = _encoder(3)
        assert all(
            parameter.requires_grad
            for parameter in encoder.model.transformer.norm.parameters()
        )

    def test_the_final_norm_stays_frozen_when_nothing_else_is(self):
        encoder = _encoder(0)
        assert not any(
            parameter.requires_grad
            for parameter in encoder.model.transformer.norm.parameters()
        )

    def test_asking_for_more_layers_than_exist_is_refused(self):
        with pytest.raises(ProteinEncoderError, match="exceeds the 6 blocks"):
            _encoder(99, n_layers=6)

    def test_a_negative_count_is_refused(self):
        with pytest.raises(ValueError, match="n_trainable_layers"):
            ESMCProteinEncoder(model=_Model(), tokenizer=object(), n_trainable_layers=-1)

    def test_the_block_stack_is_located_not_assumed(self):
        """A checkpoint without transformer.blocks must fail loudly, not freeze nothing."""

        class Bare(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(2))

        with pytest.raises(ProteinEncoderError, match="could not locate"):
            ESMCProteinEncoder(model=Bare(), tokenizer=object(), n_trainable_layers=1)

    def test_trainable_parameters_is_what_an_optimizer_should_receive(self):
        encoder = _encoder(2, n_layers=6)
        expected = sum(
            p.numel() for p in encoder.model.parameters() if p.requires_grad
        )
        assert sum(p.numel() for p in encoder.trainable_parameters()) == expected
        assert encoder.n_trainable == expected


class TestGradientFlow:
    def test_only_the_unfrozen_blocks_receive_gradient(self):
        encoder = _encoder(2, n_layers=5)
        model = encoder.model
        hidden = model.embed(torch.randint(0, 32, (2, 7)))
        for block in model.transformer.blocks:
            hidden = block(hidden)
        model.transformer.norm(hidden).pow(2).mean().backward()

        received = {
            name
            for name, parameter in model.named_parameters()
            if parameter.grad is not None and parameter.grad.abs().sum() > 0
        }
        assert received
        assert all(
            (".blocks.3." in name or ".blocks.4." in name or "transformer.norm" in name)
            for name in received
        ), sorted(received)

    def test_frozen_weights_do_not_move_under_an_optimizer(self):
        encoder = _encoder(2, n_layers=5)
        model = encoder.model
        before = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if not p.requires_grad
        }
        optimizer = torch.optim.Adam(encoder.trainable_parameters(), lr=1e-1)
        hidden = model.embed(torch.randint(0, 32, (2, 7)))
        for block in model.transformer.blocks:
            hidden = block(hidden)
        model.transformer.norm(hidden).pow(2).mean().backward()
        optimizer.step()
        for name, parameter in model.named_parameters():
            if name in before:
                assert torch.equal(before[name], parameter), name

    def test_the_unfrozen_blocks_actually_move(self):
        encoder = _encoder(2, n_layers=5)
        model = encoder.model
        before = [p.detach().clone() for p in encoder.trainable_parameters()]
        optimizer = torch.optim.Adam(encoder.trainable_parameters(), lr=1e-1)
        hidden = model.embed(torch.randint(0, 32, (2, 7)))
        for block in model.transformer.blocks:
            hidden = block(hidden)
        model.transformer.norm(hidden).pow(2).mean().backward()
        optimizer.step()
        assert any(
            not torch.equal(old, new)
            for old, new in zip(before, encoder.trainable_parameters())
        )


class TestCacheSafety:
    def test_a_frozen_encoder_is_cacheable(self):
        _encoder(0).assert_cacheable()  # does not raise

    def test_the_protein_cache_refuses_a_trainable_encoder(self):
        """The 14 GiB cache would be stale from the first optimizer step."""
        with pytest.raises(ProteinEncoderError, match="must not be cached"):
            _encoder(3).assert_cacheable()

    def test_the_refusal_says_how_to_proceed(self):
        with pytest.raises(ProteinEncoderError, match="n_trainable_layers=0"):
            _encoder(3).assert_cacheable()

    def test_the_cli_encode_path_calls_the_guard(self):
        """`bindra cache proteins` must not silently write stale features."""
        import inspect

        from bindra.cli import cache as cache_cli

        assert "assert_cacheable" in inspect.getsource(cache_cli._encode_frame)

    def test_the_freeze_regime_is_part_of_the_cache_key(self):
        frozen = _encoder(0).metadata(8, weights_checksum="x")
        tuned = _encoder(3).metadata(8, weights_checksum="x")
        assert frozen.cache_key("MKV") != tuned.cache_key("MKV")
        # Frozen emits no marker at all, so the key matches caches written
        # before the field existed; see TestCacheKeyStability.
        assert "n_trainable_layers" not in frozen.extra
        assert tuned.extra["n_trainable_layers"] == 3


@pytest.mark.encoders
class TestAgainstTheRealCheckpoint:
    """Against ESM-C 600M itself, which has 36 blocks of 15,932,160 parameters."""

    @pytest.fixture(scope="class")
    @classmethod
    def tuned(cls):
        encoder = ESMCProteinEncoder(precision="float32", n_trainable_layers=3)
        try:
            encoder._ensure_loaded()
        except Exception as exc:  # noqa: BLE001 - weights may be absent
            pytest.skip(f"ESM-C weights unavailable: {exc}")
        return encoder

    def test_three_blocks_is_forty_eight_million_parameters(self, tuned):
        assert tuned.n_trainable == 47_797_632
        total = sum(p.numel() for p in tuned.model.parameters())
        assert total == 573_632_640

    def test_the_unfrozen_blocks_are_the_last_three_of_thirty_six(self, tuned):
        blocks = tuned.transformer_blocks()
        assert len(blocks) == 36
        for index, block in enumerate(blocks):
            should_train = index >= 33
            for parameter in block.parameters():
                assert parameter.requires_grad is should_train, index

    def test_a_frozen_real_encoder_trains_nothing(self):
        encoder = ESMCProteinEncoder(precision="float32", n_trainable_layers=0)
        try:
            encoder._ensure_loaded()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"ESM-C weights unavailable: {exc}")
        assert encoder.n_trainable == 0


class TestCacheKeyStability:
    """A frozen encoder must keep matching the already-populated cache.

    The 14.023 GiB protein cache was written before ``n_trainable_layers``
    existed. ``cache_key`` hashes the whole ``extra`` dict, so emitting the new
    field unconditionally would have changed the *frozen* encoder's key too and
    orphaned the cache along with the 40 minutes that produced it -- reported as
    a stale-cache error at read time, far from the edit that caused it.
    """

    def test_a_frozen_encoder_emits_no_freezing_field(self):
        extra = _encoder(0).metadata(8, weights_checksum="x").extra
        assert "n_trainable_layers" not in extra

    def test_a_trainable_encoder_does_emit_it(self):
        extra = _encoder(1).metadata(8, weights_checksum="x").extra
        assert extra["n_trainable_layers"] == 1

    def test_the_frozen_key_is_unchanged_by_the_feature_existing(self):
        """Pinned against a metadata dict built without the field at all."""
        from bindra.features.types import EncoderMetadata

        frozen = _encoder(0).metadata(8, weights_checksum="x")
        legacy = EncoderMetadata(
            **{**frozen.to_dict(), "extra": {
                k: v for k, v in frozen.extra.items() if k != "n_trainable_layers"
            }}
        )
        assert frozen.cache_key("MKV") == legacy.cache_key("MKV")

    def test_freezing_regimes_still_key_apart(self):
        a = _encoder(0).metadata(8, weights_checksum="x")
        b = _encoder(1).metadata(8, weights_checksum="x")
        c = _encoder(2).metadata(8, weights_checksum="x")
        assert len({a.cache_key("MKV"), b.cache_key("MKV"), c.cache_key("MKV")}) == 3


@pytest.mark.encoders
class TestAgainstThePopulatedCache:
    """The real check: the 14 GiB cache on disk still opens."""

    def test_the_populated_cache_accepts_a_fresh_frozen_encoder(self):
        from pathlib import Path

        from bindra.features.cache import ProteinFeatureCache
        from bindra.encoders.protein_esmc import (
            ESMC_600M_WEIGHTS_SHA256,
            EXPECTED_ESMC_600M_DIM,
        )

        directory = Path("data/cache/proteins")
        if not (directory / "index.json").exists():
            pytest.skip("protein cache not populated")
        frozen = ESMCProteinEncoder(n_trainable_layers=0).metadata(
            EXPECTED_ESMC_600M_DIM, weights_checksum=ESMC_600M_WEIGHTS_SHA256
        )
        cache = ProteinFeatureCache(directory, metadata=frozen)
        assert len(cache) == 10_820
