"""Feature cache: bit-exact bf16, ragged storage, and stale-cache refusal.

The legacy cache loaded itself entirely into RAM with ``torch.load``; at 14.03 GiB
measured for the protein half that is no longer an option, so these tests pin the
properties the replacement has to hold.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from bindra.features.cache import (
    CACHE_ROOT_ENV,
    CACHE_SCHEMA_VERSION,
    FeatureCacheError,
    ProteinFeatureCache,
    StaleCacheError,
    default_cache_root,
)
from bindra.features.types import EncoderMetadata, ProteinFeatures

DIM = 8


def _metadata(**overrides) -> EncoderMetadata:
    payload = dict(
        name="esmc",
        checkpoint="biohub/ESMC-600M",
        output_dim=DIM,
        precision="bfloat16",
        feature_schema_version="protein-esmc-v2-postln-native",
        revision="fcc9d36f",
        weights_checksum="7855e032",
        preprocessing_policy="last_hidden_state",
        extra={"kernel_backends": []},
    )
    payload.update(overrides)
    return EncoderMetadata(**payload)


def _features(length: int, metadata: EncoderMetadata, *, hash_: str = "h") -> ProteinFeatures:
    return ProteinFeatures(
        residue_features=torch.randn(length, DIM).to(torch.bfloat16),
        residue_mask=torch.ones(length, dtype=torch.bool),
        residue_indices=torch.arange(length, dtype=torch.long),
        sequence_hash=hash_,
        encoder_metadata=metadata,
        sequence_length=length,
    )


class TestRoundTrip:
    def test_bf16_is_bit_exact(self, tmp_path):
        """numpy has no bfloat16, so raw bits are stored as uint16.

        Storing float16 instead would silently change the numbers the encoder
        produced while the cache key still claimed bf16.
        """
        metadata = _metadata()
        original = _features(300, metadata)
        with ProteinFeatureCache(tmp_path / "c", metadata=metadata, mode="w") as cache:
            cache.put("k", original)
        restored = ProteinFeatureCache(tmp_path / "c", metadata=metadata).get("k")
        assert restored.residue_features.dtype is torch.bfloat16
        assert torch.equal(original.residue_features, restored.residue_features)

    def test_reconstructed_fields_match_the_invariant(self, tmp_path):
        metadata = _metadata()
        with ProteinFeatureCache(tmp_path / "c", metadata=metadata, mode="w") as cache:
            cache.put("k", _features(42, metadata, hash_="abc"))
        restored = ProteinFeatureCache(tmp_path / "c", metadata=metadata).get("k")
        assert torch.equal(restored.residue_indices, torch.arange(42))
        assert bool(restored.residue_mask.all())
        assert restored.sequence_length == 42
        assert restored.sequence_hash == "abc"

    def test_a_very_long_sequence_round_trips(self, tmp_path):
        """Titin is 34,350 residues in this release."""
        metadata = _metadata()
        original = _features(34_350, metadata)
        with ProteinFeatureCache(tmp_path / "c", metadata=metadata, mode="w") as cache:
            cache.put("titin", original)
        restored = ProteinFeatureCache(tmp_path / "c", metadata=metadata).get("titin")
        assert torch.equal(original.residue_features, restored.residue_features)


class TestRaggedStorage:
    def test_storage_is_not_padded(self, tmp_path):
        """Padding to the longest sequence would turn 14 GiB into ~850 GiB."""
        metadata = _metadata()
        lengths = [7, 50, 300, 5000]
        with ProteinFeatureCache(tmp_path / "c", metadata=metadata, mode="w") as cache:
            for index, length in enumerate(lengths):
                cache.put(f"k{index}", _features(length, metadata))
        cache = ProteinFeatureCache(tmp_path / "c", metadata=metadata)
        assert cache.total_residues == sum(lengths)
        assert cache.total_bytes == sum(lengths) * DIM * 2

    def test_entries_spill_into_multiple_shards(self, tmp_path):
        metadata = _metadata()
        with ProteinFeatureCache(
            tmp_path / "c", metadata=metadata, mode="w", shard_bytes=4_000
        ) as cache:
            for index in range(6):
                cache.put(f"k{index}", _features(400, metadata))
        reopened = ProteinFeatureCache(tmp_path / "c", metadata=metadata)
        assert reopened.stats()["n_shards"] > 1
        assert len(reopened) == 6
        for index in range(6):
            assert reopened.get(f"k{index}").residue_features.shape == (400, DIM)


class TestStaleness:
    @pytest.mark.parametrize(
        "field,value",
        [
            ("checkpoint", "other/ESMC-600M"),
            ("precision", "float16"),
            ("revision", "deadbeef"),
            ("weights_checksum", "0000"),
            ("feature_schema_version", "v-old"),
            ("preprocessing_policy", "hidden_states"),
            ("output_dim", DIM),
        ],
    )
    def test_any_encoder_change_refuses_the_cache(self, tmp_path, field, value):
        """A changed encoder contract must never serve its old features."""
        metadata = _metadata()
        with ProteinFeatureCache(tmp_path / "c", metadata=metadata, mode="w") as cache:
            cache.put("k", _features(10, metadata))

        if field == "output_dim":
            changed = _metadata(extra={"kernel_backends": ["flash_attn"]})
        else:
            changed = _metadata(**{field: value})
        with pytest.raises(StaleCacheError):
            ProteinFeatureCache(tmp_path / "c", metadata=changed)

    def test_installing_a_kernel_backend_invalidates_the_cache(self, tmp_path):
        """ESM-C numerics change with flash-attn/xformers present."""
        metadata = _metadata()
        with ProteinFeatureCache(tmp_path / "c", metadata=metadata, mode="w") as cache:
            cache.put("k", _features(10, metadata))
        with pytest.raises(StaleCacheError):
            ProteinFeatureCache(
                tmp_path / "c", metadata=_metadata(extra={"kernel_backends": ["xformers"]})
            )

    def test_mixing_two_contracts_in_one_cache_is_refused(self, tmp_path):
        metadata = _metadata()
        other = _metadata(precision="float16")
        with pytest.raises(StaleCacheError):
            with ProteinFeatureCache(tmp_path / "c", metadata=metadata, mode="w") as cache:
                cache.put("a", _features(10, metadata))
                cache.put("b", _features(10, other))

    def test_schema_version_mismatch_is_refused(self, tmp_path):
        metadata = _metadata()
        with ProteinFeatureCache(tmp_path / "c", metadata=metadata, mode="w") as cache:
            cache.put("k", _features(10, metadata))
        index = tmp_path / "c" / "index.json"
        payload = json.loads(index.read_text(encoding="utf-8"))
        payload["schema_version"] = "feature-cache-v0"
        index.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(StaleCacheError):
            ProteinFeatureCache(tmp_path / "c", metadata=metadata)


class TestWriteGuards:
    def test_non_bf16_features_are_refused(self, tmp_path):
        metadata = _metadata()
        features = _features(10, metadata)
        float32 = ProteinFeatures(
            residue_features=features.residue_features.float(),
            residue_mask=features.residue_mask,
            residue_indices=features.residue_indices,
            sequence_hash="h",
            encoder_metadata=metadata,
            sequence_length=10,
        )
        with ProteinFeatureCache(tmp_path / "c", metadata=metadata, mode="w") as cache:
            with pytest.raises(FeatureCacheError, match="bfloat16"):
                cache.put("k", float32)

    def test_non_trivial_residue_indices_are_refused(self, tmp_path):
        """The cache reconstructs arange(L); a real numbering would be lost."""
        metadata = _metadata()
        features = _features(10, metadata)
        shifted = ProteinFeatures(
            residue_features=features.residue_features,
            residue_mask=features.residue_mask,
            residue_indices=torch.arange(10, dtype=torch.long) + 5,
            sequence_hash="h",
            encoder_metadata=metadata,
            sequence_length=10,
        )
        with ProteinFeatureCache(tmp_path / "c", metadata=metadata, mode="w") as cache:
            with pytest.raises(FeatureCacheError, match="arange"):
                cache.put("k", shifted)

    def test_reading_a_missing_cache_raises(self, tmp_path):
        with pytest.raises(FeatureCacheError, match="no cache index"):
            ProteinFeatureCache(tmp_path / "absent", metadata=_metadata())

    def test_requesting_an_absent_key_raises(self, tmp_path):
        metadata = _metadata()
        with ProteinFeatureCache(tmp_path / "c", metadata=metadata, mode="w") as cache:
            cache.put("k", _features(10, metadata))
        cache = ProteinFeatureCache(tmp_path / "c", metadata=metadata)
        with pytest.raises(FeatureCacheError, match="not in this cache"):
            cache.get("missing")


class TestResumability:
    def test_reopening_preserves_entries_and_skips_them(self, tmp_path):
        """A 34-minute encode must be resumable, not restartable."""
        metadata = _metadata()
        with ProteinFeatureCache(tmp_path / "c", metadata=metadata, mode="w") as cache:
            cache.put("a", _features(100, metadata))
        with ProteinFeatureCache(tmp_path / "c", metadata=metadata, mode="w") as cache:
            assert "a" in cache
            cache.put("b", _features(100, metadata))
        reopened = ProteinFeatureCache(tmp_path / "c", metadata=metadata)
        assert set(reopened.keys) == {"a", "b"}
        assert reopened.get("a").residue_features.shape == (100, DIM)

    def test_putting_a_known_key_twice_is_a_no_op(self, tmp_path):
        metadata = _metadata()
        with ProteinFeatureCache(tmp_path / "c", metadata=metadata, mode="w") as cache:
            cache.put("a", _features(100, metadata))
            cache.put("a", _features(100, metadata))
        assert len(ProteinFeatureCache(tmp_path / "c", metadata=metadata)) == 1

    def test_index_write_is_atomic(self, tmp_path):
        metadata = _metadata()
        with ProteinFeatureCache(tmp_path / "c", metadata=metadata, mode="w") as cache:
            cache.put("a", _features(10, metadata))
        assert not list((tmp_path / "c").glob("*.tmp"))


class TestCacheRoot:
    def test_default_is_inside_the_project(self, tmp_path, monkeypatch):
        monkeypatch.delenv(CACHE_ROOT_ENV, raising=False)
        assert default_cache_root(tmp_path) == tmp_path / "data" / "cache"

    def test_environment_override_wins(self, tmp_path, monkeypatch):
        """So a 14 GiB cache can be moved off a synced folder without code changes."""
        monkeypatch.setenv(CACHE_ROOT_ENV, str(tmp_path / "elsewhere"))
        assert default_cache_root() == tmp_path / "elsewhere"


class TestStats:
    def test_stats_describe_the_length_distribution(self, tmp_path):
        metadata = _metadata()
        with ProteinFeatureCache(tmp_path / "c", metadata=metadata, mode="w") as cache:
            for index, length in enumerate([7, 300, 2000]):
                cache.put(f"k{index}", _features(length, metadata))
        stats = ProteinFeatureCache(tmp_path / "c", metadata=metadata).stats()
        assert stats["n_entries"] == 3
        assert stats["residues_min"] == 7
        assert stats["residues_max"] == 2000
        assert stats["total_residues"] == 2307
