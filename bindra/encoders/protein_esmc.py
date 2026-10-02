"""ESM-C-600M protein encoder adapter.

Ported from ``models/rmse_model.py::ProteinESM`` in DeepDTA-iBAM, with the two
correctness problems in that implementation fixed:

1. **Special tokens were stripped by guessing.** The legacy ``_align_embeddings``
   compared the output length to the sequence length and sliced ``[1:1+L]`` when
   the output happened to be at least two longer, zero-padding when it was
   shorter. That silently fabricates residues when a tokenizer inserts an
   unexpected token. Here the tokenizer's own ``get_special_tokens_mask`` says
   which positions are special, and a length disagreement is an error.

2. **Residue numbering was lost when windowing.** The legacy ``_stitch_chunks``
   averaged overlapping windows into a dense array with no record of which
   input position each row came from. ``ProteinFeatures.residue_indices``
   now carries that mapping through, which is what lets an iBAM heatmap label
   a real residue position on a long protein.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Sequence

import torch

from bindra.features.types import EncoderMetadata, ProteinFeatures, sequence_digest

#: Bump when the meaning or layout of the emitted features changes. This takes
#: part in the cache key, so a bump invalidates every cached protein.
FEATURE_SCHEMA_VERSION = "protein-esmc-v1"

#: ESM-C-600M hidden width. Asserted against the real output, never assumed.
EXPECTED_ESMC_600M_DIM = 1152

#: ESM-C was trained with a 2048-position context including special tokens.
#: Longer sequences are windowed rather than truncated.
DEFAULT_CONTEXT_LIMIT = 2048


class ProteinEncoderError(RuntimeError):
    """The encoder could not produce trustworthy features for a sequence."""


@dataclass(frozen=True)
class WindowPlan:
    """How one sequence is split across encoder context windows."""

    windows: tuple[tuple[int, int], ...]  # (start, end) half-open, in residues
    window_size: int
    overlap: int

    @property
    def is_windowed(self) -> bool:
        return len(self.windows) > 1


def plan_windows(
    sequence_length: int,
    window_size: int,
    overlap: int,
) -> WindowPlan:
    """Cover ``[0, sequence_length)`` with overlapping windows.

    Deterministic and total: every residue lands in at least one window, so a
    long protein is never silently truncated.
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if not 0 <= overlap < window_size:
        raise ValueError(f"overlap must satisfy 0 <= overlap < window_size, got {overlap}")

    if sequence_length <= window_size:
        return WindowPlan(((0, sequence_length),), window_size, overlap)

    stride = window_size - overlap
    windows: list[tuple[int, int]] = []
    start = 0
    while start < sequence_length:
        end = min(start + window_size, sequence_length)
        windows.append((start, end))
        if end >= sequence_length:
            break
        start += stride
    return WindowPlan(tuple(windows), window_size, overlap)


def strip_special_tokens(
    representations: torch.Tensor,
    token_ids: Sequence[int],
    special_mask: Sequence[int],
    expected_residues: int,
) -> torch.Tensor:
    """Drop special-token rows using the tokenizer's own mask.

    ``representations`` is ``[T, dp]`` aligned with ``token_ids``. Returns the
    ``[expected_residues, dp]`` rows that correspond to actual amino acids.

    Raises rather than padding or trimming if the counts disagree: a mismatch
    means our understanding of the tokenizer is wrong, and the right response
    is to stop, not to invent residues.
    """
    if len(token_ids) != len(special_mask):
        raise ProteinEncoderError(
            f"token_ids ({len(token_ids)}) and special mask ({len(special_mask)}) disagree"
        )
    if representations.shape[0] != len(token_ids):
        raise ProteinEncoderError(
            f"model returned {representations.shape[0]} positions for {len(token_ids)} tokens"
        )

    keep = torch.tensor([flag == 0 for flag in special_mask], dtype=torch.bool)
    kept = representations[keep]
    if kept.shape[0] != expected_residues:
        raise ProteinEncoderError(
            f"after stripping special tokens got {kept.shape[0]} residue positions, "
            f"expected {expected_residues}. The tokenizer's layout is not what this "
            "adapter assumes; do not proceed with misaligned residues."
        )
    return kept


class ESMCProteinEncoder:
    """Frozen ESM-C-600M residue embedder.

    Produces final-layer residue representations, not vocabulary logits. The
    model is held in eval mode and every call runs under ``inference_mode``;
    this adapter is for cache population, not for fine-tuning.
    """

    def __init__(
        self,
        checkpoint: str = "esmc_600m",
        device: torch.device | str = "cpu",
        precision: str = "bfloat16",
        context_limit: int = DEFAULT_CONTEXT_LIMIT,
        overlap: int = 256,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.device = torch.device(device)
        self.precision = precision
        self.context_limit = context_limit
        self.overlap = overlap

        # Two special tokens (<cls>, <eos>) are added per window, so the usable
        # residue window is smaller than the raw context limit.
        self.window_size = context_limit - 2
        if self.overlap >= self.window_size:
            raise ValueError(
                f"overlap {overlap} must be smaller than the usable window {self.window_size}"
            )

        self._model = model
        self._tokenizer = tokenizer
        self._output_dim: int | None = None

    # -- lazy loading -----------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            from esm.models.esmc import ESMC
            from esm.tokenization import EsmSequenceTokenizer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ProteinEncoderError(
                "The 'esm' package is required for the ESM-C encoder. "
                "Install the 'encoders' extra."
            ) from exc

        if self._model is None:
            self._model = ESMC.from_pretrained(self.checkpoint).to(self.device).eval()
        if self._tokenizer is None:
            self._tokenizer = EsmSequenceTokenizer()

    @property
    def model(self) -> Any:
        self._ensure_loaded()
        return self._model

    @property
    def tokenizer(self) -> Any:
        self._ensure_loaded()
        return self._tokenizer

    # -- metadata ---------------------------------------------------------

    def weights_checksum(self) -> str:
        """Hash of the loaded parameters, pinning the actual weights in use.

        The specification requires distinguishing a newer checkpoint from an
        older one carrying the same name; a name alone cannot do that.
        """
        digest = hashlib.sha256()
        for name, parameter in sorted(self.model.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def metadata(self, output_dim: int, *, weights_checksum: str | None = None) -> EncoderMetadata:
        return EncoderMetadata(
            name="esmc",
            checkpoint=self.checkpoint,
            output_dim=output_dim,
            precision=self.precision,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            weights_checksum=weights_checksum,
            preprocessing_policy=(
                f"final-layer residue states; special tokens stripped via tokenizer mask; "
                f"window={self.window_size} overlap={self.overlap} merge=mean"
            ),
            extra={"context_limit": self.context_limit},
        )

    # -- encoding ---------------------------------------------------------

    def _encode_window(self, segment: str) -> torch.Tensor:
        """Final-layer residue states for one window, special tokens removed."""
        token_ids = self.tokenizer.encode(segment)
        special_mask = self.tokenizer.get_special_tokens_mask(
            token_ids, already_has_special_tokens=True
        )
        tokens = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        with torch.inference_mode():
            output = self.model(tokens)

        embeddings = getattr(output, "embeddings", None)
        if embeddings is None:
            embeddings = getattr(output, "hidden_states", None)
        if embeddings is None:
            raise ProteinEncoderError(
                "ESM-C output exposed neither 'embeddings' nor 'hidden_states'; "
                "refusing to guess which tensor holds the residue states."
            )

        return strip_special_tokens(
            embeddings[0].float(),
            token_ids,
            special_mask,
            expected_residues=len(segment),
        )

    def encode(self, sequence: str) -> ProteinFeatures:
        """Encode one sequence into ``ProteinFeatures``.

        Long sequences are covered by overlapping windows and merged by
        averaging each residue's representations across the windows containing
        it. The merge is deterministic and the residue numbering is preserved.
        """
        if not sequence:
            raise ProteinEncoderError("cannot encode an empty sequence")

        plan = plan_windows(len(sequence), self.window_size, self.overlap)

        accumulator: torch.Tensor | None = None
        counts = torch.zeros(len(sequence), 1, dtype=torch.float32)

        for start, end in plan.windows:
            window_features = self._encode_window(sequence[start:end]).cpu()
            if accumulator is None:
                dim = window_features.shape[1]
                accumulator = torch.zeros(len(sequence), dim, dtype=torch.float32)
            accumulator[start:end] += window_features
            counts[start:end] += 1.0

        assert accumulator is not None
        if not bool((counts > 0).all()):
            uncovered = int((counts == 0).sum().item())
            raise ProteinEncoderError(
                f"{uncovered} residues received no representation; the window plan is wrong"
            )
        residue_features = accumulator / counts

        output_dim = int(residue_features.shape[1])
        if self.checkpoint == "esmc_600m" and output_dim != EXPECTED_ESMC_600M_DIM:
            raise ProteinEncoderError(
                f"expected {EXPECTED_ESMC_600M_DIM}-dim features from esmc_600m, got {output_dim}"
            )

        if self.precision == "bfloat16":
            residue_features = residue_features.to(torch.bfloat16)
        elif self.precision == "float16":
            residue_features = residue_features.to(torch.float16)

        return ProteinFeatures(
            residue_features=residue_features,
            residue_mask=torch.ones(len(sequence), dtype=torch.bool),
            residue_indices=torch.arange(len(sequence), dtype=torch.long),
            sequence_hash=sequence_digest(sequence),
            encoder_metadata=self.metadata(output_dim),
            sequence_length=len(sequence),
        )


__all__ = [
    "DEFAULT_CONTEXT_LIMIT",
    "ESMCProteinEncoder",
    "EXPECTED_ESMC_600M_DIM",
    "FEATURE_SCHEMA_VERSION",
    "ProteinEncoderError",
    "WindowPlan",
    "plan_windows",
    "strip_special_tokens",
]
