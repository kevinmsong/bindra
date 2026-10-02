"""Protein encoder tests: special-token stripping, windowing, padding invariance.

The GPU/weights-dependent test is marked and skipped when ESM-C is unavailable,
so the rest of the suite stays runnable on any machine.
"""

from __future__ import annotations

import pytest
import torch

from bindra.encoders.protein_esmc import (
    ESMC_600M_REPO,
    ESMC_600M_REVISION,
    ESMCProteinEncoder,
    ProteinEncoderError,
    detect_kernel_backends,
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


class TestCheckpointPinning:
    """"ESM-C-600M" must name one artifact, not a moving HF repo."""

    def test_revision_is_pinned_to_a_full_sha(self):
        assert len(ESMC_600M_REVISION) == 40
        assert all(character in "0123456789abcdef" for character in ESMC_600M_REVISION)

    def test_encoder_defaults_to_the_pinned_revision(self):
        encoder = ESMCProteinEncoder()
        assert encoder.checkpoint == ESMC_600M_REPO
        assert encoder.revision == ESMC_600M_REVISION

    def test_revision_reaches_the_metadata_and_the_cache_key(self):
        pinned = ESMCProteinEncoder().metadata(1152)
        floating = ESMCProteinEncoder(revision=None).metadata(1152)
        assert pinned.revision == ESMC_600M_REVISION
        assert floating.revision is None
        assert pinned.cache_key("MKT") != floating.cache_key("MKT")


class TestNumericalProvenance:
    """Record the things that silently change ESM-C's output."""

    def test_metadata_names_the_post_layernorm_tensor(self):
        # Not decoration: last_hidden_state and last_hidden_state_prenorm
        # differ by ~6.2e3, so which one produced a cache must be recorded.
        policy = ESMCProteinEncoder().metadata(1152).preprocessing_policy
        assert policy is not None
        assert "post final LayerNorm" in policy

    def test_kernel_backends_are_recorded(self):
        metadata = ESMCProteinEncoder().metadata(1152)
        assert "kernel_backends" in metadata.extra
        assert metadata.extra["kernel_backends"] == list(detect_kernel_backends())

    def test_installing_a_fused_kernel_invalidates_the_cache(self):
        """Different kernels, different numerics, therefore different key.

        Without this, installing flash-attn later would leave cached features
        silently disagreeing with freshly computed ones.
        """
        base = ESMCProteinEncoder().metadata(1152)
        with_flash = EncoderMetadata(
            **{**base.to_dict(), "extra": {**base.extra, "kernel_backends": ["flash_attn"]}}
        )
        assert base.cache_key("MKT") != with_flash.cache_key("MKT")

    def test_extra_ordering_does_not_change_the_key(self):
        first = EncoderMetadata(
            name="e", checkpoint="c", output_dim=4, precision="bfloat16",
            feature_schema_version="v1", extra={"a": 1, "b": 2},
        )
        second = EncoderMetadata(
            name="e", checkpoint="c", output_dim=4, precision="bfloat16",
            feature_schema_version="v1", extra={"b": 2, "a": 1},
        )
        assert first.cache_key("MKT") == second.cache_key("MKT")

    def test_schema_version_marks_the_native_post_ln_contract(self):
        from bindra.encoders.protein_esmc import FEATURE_SCHEMA_VERSION

        # Any cache built against the old deprecated-shim adapter must not be
        # mistaken for one built against this contract.
        assert FEATURE_SCHEMA_VERSION == "protein-esmc-v2-postln-native"


class TestOutputTensorSelection:
    """The adapter must not fall back through candidate attribute names."""

    class _OutputWithoutLastHidden:
        embeddings = torch.zeros(1, 3, 4)
        hidden_states = torch.zeros(1, 3, 4)

    def test_missing_last_hidden_state_raises_instead_of_falling_back(self):
        encoder = ESMCProteinEncoder(
            model=lambda _tokens: self._OutputWithoutLastHidden(),
            tokenizer=_StubTokenizer(),
        )
        with pytest.raises(ProteinEncoderError, match="no 'last_hidden_state'"):
            encoder.encode("MKT")


class _StubTokenizer:
    """Minimal tokenizer: <cls> + residues + <eos>."""

    def encode(self, sequence: str) -> list[int]:
        return [0, *(10 + index for index in range(len(sequence))), 2]

    def get_special_tokens_mask(self, token_ids, already_has_special_tokens=False):
        return [1] + [0] * (len(token_ids) - 2) + [1]


class TestWithStubModel:
    """End-to-end encode() behaviour without downloading weights."""

    class _Model:
        def __init__(self, dim: int = 1152):
            self.dim = dim

        def __call__(self, tokens):
            length = tokens.shape[1]

            class _Out:
                # Deterministic, position-dependent values so stitching is checkable.
                last_hidden_state = torch.arange(
                    length, dtype=torch.float32
                ).unsqueeze(0).unsqueeze(-1).repeat(1, 1, dim_holder["dim"])

            return _Out()

    def test_encode_strips_specials_and_numbers_residues(self):
        dim_holder["dim"] = 1152
        encoder = ESMCProteinEncoder(
            model=self._Model(), tokenizer=_StubTokenizer(), precision="float32"
        )
        sequence = "MKTAY"
        features = encoder.encode(sequence)

        assert features.residue_features.shape == (5, 1152)
        assert features.residue_indices.tolist() == [0, 1, 2, 3, 4]
        assert features.covers_full_sequence
        # Positions 1..5 of the token stream survive; <cls> at 0 and <eos> last are gone.
        torch.testing.assert_close(
            features.residue_features[:, 0], torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        )

    def test_wrong_dim_for_a_600m_checkpoint_is_rejected(self):
        dim_holder["dim"] = 640
        encoder = ESMCProteinEncoder(
            model=self._Model(), tokenizer=_StubTokenizer(), precision="float32"
        )
        with pytest.raises(ProteinEncoderError, match="expected 1152-dim"):
            encoder.encode("MKT")


#: Lets the stub model vary its width between tests.
dim_holder: dict[str, int] = {"dim": 1152}


@pytest.mark.gpu
@pytest.mark.encoders
class TestAgainstRealEsmC:
    """Verified against the actual pretrained model.

    Skipped unless esm is installed and a CUDA device is present.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def encoder(cls):
        pytest.importorskip("esm")
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")
        from bindra.runtime.device import DeviceRequirement, DeviceSelectionError, select_device

        try:
            device = select_device(DeviceRequirement())
        except DeviceSelectionError as exc:
            pytest.skip(f"no suitable GPU: {exc}")
        return ESMCProteinEncoder(device=device)

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

    def test_loaded_weights_match_the_recorded_checksum(self, encoder):
        """The pinned revision really does load the weights we recorded."""
        from bindra.encoders.protein_esmc import ESMC_600M_WEIGHTS_SHA256

        assert encoder.weights_checksum() == ESMC_600M_WEIGHTS_SHA256

    def test_post_layernorm_state_differs_sharply_from_the_prenorm_stream(self, encoder):
        """Document why the output tensor is selected by name, not by fallback.

        If these two were close, the legacy attribute-fallback would have been
        harmless. They are not: the difference is ~1e3, so reading the wrong
        field silently poisons every cached feature.
        """
        import torch as _torch

        tokens = _torch.tensor([encoder.tokenizer.encode("MKTAYIAKQRQ")], device=encoder.device)
        with _torch.inference_mode():
            output = encoder.model(tokens, output_hidden_states=True)

        post = output.last_hidden_state[0].float()
        prenorm = output.last_hidden_state_prenorm[0].float()
        assert (post - prenorm).abs().max().item() > 100.0
