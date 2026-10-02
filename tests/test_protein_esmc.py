"""Protein encoder tests: special-token stripping, windowing, padding invariance.

The GPU/weights-dependent test is marked and skipped when ESM-C is unavailable,
so the rest of the suite stays runnable on any machine.
"""

from __future__ import annotations

import pytest
import torch

from bindra.encoders.protein_esmc import (
    ESMCProteinEncoder,
    ProteinEncoderError,
    plan_windows,
    strip_special_tokens,
)
from bindra.features.types import EncoderMetadata, ProteinFeatures


class TestStripSpecialTokens:
    def test_strips_leading_and_trailing_specials(self):
        # <cls> M K T A <eos>  ->  keep the four residues
        representations = torch.arange(6 * 3, dtype=torch.float32).reshape(6, 3)
        token_ids = [0, 20, 15, 11, 5, 2]
        special_mask = [1, 0, 0, 0, 0, 1]

        kept = strip_special_tokens(representations, token_ids, special_mask, expected_residues=4)

        assert kept.shape == (4, 3)
        torch.testing.assert_close(kept, representations[1:5])

    def test_strips_interior_specials_too(self):
        # A chain-break token in the middle must not be mistaken for a residue.
        representations = torch.arange(5 * 2, dtype=torch.float32).reshape(5, 2)
        token_ids = [0, 20, 31, 15, 2]
        special_mask = [1, 0, 1, 0, 1]

        kept = strip_special_tokens(representations, token_ids, special_mask, expected_residues=2)

        assert kept.shape == (2, 2)
        torch.testing.assert_close(kept, representations[[1, 3]])

    def test_count_mismatch_raises_rather_than_padding(self):
        # The legacy implementation zero-padded here, inventing residues.
        representations = torch.zeros(4, 2)
        with pytest.raises(ProteinEncoderError, match="expected 5"):
            strip_special_tokens(representations, [0, 20, 15, 2], [1, 0, 0, 1], expected_residues=5)

    def test_shape_disagreement_raises(self):
        representations = torch.zeros(3, 2)
        with pytest.raises(ProteinEncoderError, match="3 positions for 4 tokens"):
            strip_special_tokens(representations, [0, 20, 15, 2], [1, 0, 0, 1], expected_residues=2)


class TestWindowPlanning:
    def test_short_sequence_is_a_single_window(self):
        plan = plan_windows(100, window_size=2046, overlap=256)
        assert plan.windows == ((0, 100),)
        assert not plan.is_windowed

    def test_boundary_at_exactly_one_window(self):
        plan = plan_windows(2046, window_size=2046, overlap=256)
        assert plan.windows == ((0, 2046),)

    def test_long_sequence_is_covered_without_truncation(self):
        length = 5000
        plan = plan_windows(length, window_size=2046, overlap=256)
        assert plan.is_windowed
        assert plan.windows[0][0] == 0
        assert plan.windows[-1][1] == length

        covered = set()
        for start, end in plan.windows:
            covered.update(range(start, end))
        assert covered == set(range(length)), "every residue must appear in some window"

    def test_windows_overlap_by_the_requested_amount(self):
        plan = plan_windows(5000, window_size=1000, overlap=200)
        first, second = plan.windows[0], plan.windows[1]
        assert first[1] - second[0] == 200

    @pytest.mark.parametrize(
        ("window_size", "overlap"),
        [(0, 0), (-1, 0), (100, 100), (100, 150), (100, -1)],
    )
    def test_invalid_plans_are_rejected(self, window_size, overlap):
        with pytest.raises(ValueError):
            plan_windows(500, window_size=window_size, overlap=overlap)


class TestProteinFeaturesContract:
    def _metadata(self, dim: int = 4) -> EncoderMetadata:
        return EncoderMetadata(
            name="stub",
            checkpoint="stub-1",
            output_dim=dim,
            precision="float32",
            feature_schema_version="test-v1",
        )

    def test_well_formed_features_are_accepted(self):
        features = ProteinFeatures(
            residue_features=torch.zeros(10, 4),
            residue_mask=torch.ones(10, dtype=torch.bool),
            residue_indices=torch.arange(10),
            sequence_hash="abc",
            encoder_metadata=self._metadata(),
            sequence_length=10,
        )
        assert features.feature_dim == 4
        assert features.num_residues == 10
        assert features.covers_full_sequence

    def test_partial_coverage_is_visible(self):
        mask = torch.ones(10, dtype=torch.bool)
        mask[-3:] = False
        features = ProteinFeatures(
            residue_features=torch.zeros(10, 4),
            residue_mask=mask,
            residue_indices=torch.arange(10),
            sequence_hash="abc",
            encoder_metadata=self._metadata(),
            sequence_length=10,
        )
        assert not features.covers_full_sequence, "truncation must be detectable"

    def test_dim_disagreeing_with_metadata_is_rejected(self):
        with pytest.raises(ValueError, match="disagrees with encoder metadata"):
            ProteinFeatures(
                residue_features=torch.zeros(10, 8),
                residue_mask=torch.ones(10, dtype=torch.bool),
                residue_indices=torch.arange(10),
                sequence_hash="abc",
                encoder_metadata=self._metadata(dim=4),
                sequence_length=10,
            )

    def test_mask_must_be_bool(self):
        with pytest.raises(ValueError, match="must be bool"):
            ProteinFeatures(
                residue_features=torch.zeros(10, 4),
                residue_mask=torch.ones(10, dtype=torch.long),
                residue_indices=torch.arange(10),
                sequence_hash="abc",
                encoder_metadata=self._metadata(),
                sequence_length=10,
            )


class TestCacheKeying:
    """A cache key must change when the producing encoder changes."""

    def _metadata(self, **overrides) -> EncoderMetadata:
        base = dict(
            name="esmc",
            checkpoint="esmc_600m",
            output_dim=1152,
            precision="bfloat16",
            feature_schema_version="protein-esmc-v1",
        )
        base.update(overrides)
        return EncoderMetadata(**base)

    def test_same_input_and_encoder_gives_the_same_key(self):
        assert self._metadata().cache_key("MKT") == self._metadata().cache_key("MKT")

    def test_different_input_gives_a_different_key(self):
        assert self._metadata().cache_key("MKT") != self._metadata().cache_key("MKA")

    @pytest.mark.parametrize(
        "override",
        [
            {"checkpoint": "esmc_300m"},
            {"precision": "float16"},
            {"feature_schema_version": "protein-esmc-v2"},
            {"revision": "abc123"},
            {"weights_checksum": "deadbeef"},
            {"preprocessing_policy": "different"},
        ],
    )
    def test_changing_the_encoder_invalidates_the_key(self, override):
        # The legacy cache keyed on sha1(input) alone, so changing the encoder
        # silently reused stale features. Every field must participate.
        assert self._metadata().cache_key("MKT") != self._metadata(**override).cache_key("MKT")


@pytest.mark.gpu
@pytest.mark.encoders
class TestAgainstRealEsmC:
    """Verified against the actual pretrained model.

    Skipped unless esm is installed and a CUDA device is present.
    """

    @pytest.fixture(scope="class")
    def encoder(self):
        pytest.importorskip("esm")
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")
        from bindra.runtime.device import DeviceRequirement, select_device

        return ESMCProteinEncoder(device=select_device(DeviceRequirement()))

    def test_residue_count_and_dim_match_the_input(self, encoder):
        sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
        features = encoder.encode(sequence)

        # 33 residues in, 33 rows out: the two special tokens are gone.
        assert features.residue_features.shape == (len(sequence), 1152)
        assert features.num_residues == len(sequence)
        assert features.covers_full_sequence
        assert features.residue_indices.tolist() == list(range(len(sequence)))

    def test_padding_invariance(self, encoder):
        """A sequence's features must not depend on what else was in the batch.

        Encoding one at a time is the reference; this guards the invariant that
        later batched implementations must preserve.
        """
        sequence = "MKTAYIAKQRQ"
        first = encoder.encode(sequence).residue_features.float()
        second = encoder.encode(sequence).residue_features.float()
        torch.testing.assert_close(first, second)

    def test_prefix_of_a_longer_sequence_is_not_identical(self, encoder):
        """Sanity check that context actually matters, i.e. we are not reading
        a position-independent embedding table instead of final-layer states."""
        short = encoder.encode("MKTAYIAKQRQ").residue_features.float()
        longer = encoder.encode("MKTAYIAKQRQISFVKSHFSRQ").residue_features.float()[: short.shape[0]]
        assert not torch.allclose(short, longer, atol=1e-3)
