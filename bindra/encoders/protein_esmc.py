"""ESM-C-600M protein encoder adapter.

Ported from ``models/rmse_model.py::ProteinESM`` in DeepDTA-iBAM, with four
correctness problems fixed. The first two were in the legacy code; the second
two are hazards in the current ``esm`` package.

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

3. **The deprecated ``ESMC`` class is no longer used.** In ``esm`` >= 3.3 it is
   a compatibility shim over ``EsmcForMaskedLM`` that also *realigns*
   ``hidden_states`` onto the old layout. We use the native ``EsmcModel`` and
   ``EsmcTokenizer`` directly, with the HF repo and revision pinned, so that
   "ESM-C-600M" names one specific artifact rather than a moving target.

4. **The post-LayerNorm state is selected explicitly.** ``EsmcModel`` exposes
   both ``last_hidden_state`` (after the final LayerNorm) and
   ``last_hidden_state_prenorm`` (the raw residual stream). Measured on this
   machine they differ by a max of ~6.2e3, so picking the wrong one would
   silently poison every downstream module. We take the post-LayerNorm state,
   which is also the numerically stable choice: without fused kernels the
   package warns that the *unnormalized* stream drifts by O(10)-O(100), while
   after the final LayerNorm the difference is a few ULP.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
from typing import Any, Sequence

import torch

from bindra.features.types import EncoderMetadata, ProteinFeatures, sequence_digest

#: Bump when the meaning or layout of the emitted features changes. This takes
#: part in the cache key, so a bump invalidates every cached protein.
FEATURE_SCHEMA_VERSION = "protein-esmc-v2-postln-native"

#: ESM-C-600M hidden width. Asserted against the real output, never assumed.
EXPECTED_ESMC_600M_DIM = 1152

#: ESM-C was trained with a 2048-position context including special tokens.
#: Longer sequences are windowed rather than truncated.
DEFAULT_CONTEXT_LIMIT = 2048

#: Pinned checkpoint. The revision is pinned because "ESMC-600M" is a mutable
#: HF repo: resolving it by name alone would let the weights change underneath
#: a cache that claims to hold ESM-C-600M features.
ESMC_600M_REPO = "biohub/ESMC-600M"
ESMC_600M_REVISION = "fcc9d36f5fff97d9bb69a36e3252de61cc78c968"

#: SHA-256 over the sorted state dict of the pinned revision, measured on
#: 2026-10-01. The revision pin already fixes the artifact; this detects the
#: case where the loaded weights are not the ones the revision claims -- a
#: corrupted HF cache, or a local override. Verify with
#: ``ESMCProteinEncoder().weights_checksum()``.
ESMC_600M_WEIGHTS_SHA256 = "7855e03269e5241e11bbb879108b888ffe10e625c730272eff80f99bf7cb8538"

#: Optional kernel libraries that change ESM-C's numerics when present. Their
#: presence is recorded in encoder metadata and therefore in the cache key.
ACCELERATION_LIBRARIES = ("transformer_engine", "xformers", "flash_attn")


def detect_kernel_backends() -> tuple[str, ...]:
    """Which acceleration libraries are importable right now.

    ESM-C falls back to pure-PyTorch LayerNorm, attention and RoPE when these
    are missing, and the package warns that results differ numerically. Two
    machines with different backends produce different features from the same
    weights, so this belongs in the cache key, not in a comment.
    """
    return tuple(
        name for name in ACCELERATION_LIBRARIES if importlib.util.find_spec(name) is not None
    )


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
        checkpoint: str = ESMC_600M_REPO,
        revision: str | None = ESMC_600M_REVISION,
        device: torch.device | str = "cpu",
        precision: str = "bfloat16",
        context_limit: int = DEFAULT_CONTEXT_LIMIT,
        overlap: int = 256,
        model: Any | None = None,
        tokenizer: Any | None = None,
        n_trainable_layers: int = 0,
    ) -> None:
        self.checkpoint = checkpoint
        self.revision = revision
        self.device = torch.device(device)
        self.precision = precision
        self.context_limit = context_limit
        self.overlap = overlap
        if n_trainable_layers < 0:
            raise ValueError(
                f"n_trainable_layers must be >= 0, got {n_trainable_layers}"
            )
        #: How many trailing transformer blocks to leave trainable, plus the
        #: final LayerNorm above them. **Default 0: frozen.** That is what
        #: makes the 14.023 GiB protein cache valid, and it removes the
        #: target-grouped batching requirement entirely -- with cached features
        #: a random batch is free, where a trainable encoder forces 43+ hours
        #: per epoch. Partial fine-tuning is implemented and tested; it is an
        #: ablation to run against a baseline, not a starting configuration.
        #:
        #: **Unfreezing has a statistical cost that the throughput numbers do not
        #: show.** Each ESM-C block is 15,932,160 parameters. The corpus has
        #: 10,474 observed targets but a Kish effective sample size of **572
        #: target units**, so one unfrozen block is roughly 27,900 trainable
        #: parameters per effective independent target (three would be 83,000). The protein axis is where this corpus is
        #: poorest, and fine-tuning there fits the encoder to the targets rather
        #: than to the interaction. The split audit cannot detect it: a model that
        #: has memorised a target family still scores well on held-out *ligands*
        #: for those targets, and only the cold-target regimes would show it.
        self.n_trainable_layers = n_trainable_layers

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
        self._weights_checksum_cache: str | None = None
        #: True when the caller supplied the model rather than letting the adapter
        #: load the pinned checkpoint. Injected models are test doubles or custom
        #: builds, so their identity is the caller's to assert, and metadata() does
        #: not try to hash them.
        self._model_injected = model is not None
        self.apply_freezing()

    # -- lazy loading -----------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            # The native API, not the deprecated ESMC shim. The shim wraps
            # EsmcForMaskedLM and realigns hidden_states onto the pre-3.3
            # layout; we want the encoder itself and its documented outputs.
            from esm.models.esmc.model import EsmcModel
            from esm.models.esmc.tokenizer import EsmcTokenizer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ProteinEncoderError(
                "The 'esm' package (>= 3.3) is required for the ESM-C encoder. "
                "Install the 'encoders' extra."
            ) from exc

        load_kwargs: dict[str, Any] = {}
        if self.revision is not None:
            load_kwargs["revision"] = self.revision

        if self._tokenizer is None:
            self._tokenizer = EsmcTokenizer.from_pretrained(self.checkpoint, **load_kwargs)
        if self._model is None:
            self._model = (
                EsmcModel.from_pretrained(self.checkpoint, **load_kwargs).to(self.device).eval()
            )
        self.apply_freezing()

    # -- partial fine-tuning ----------------------------------------------

    def transformer_blocks(self) -> Any:
        """The ``ModuleList`` of transformer blocks, located rather than assumed.

        ESM-C exposes them at ``transformer.blocks``, but a test double or a
        future release may not, so this looks for the stack instead of indexing a
        hardcoded path and silently freezing nothing.
        """
        model = self._model
        if model is None:
            raise ProteinEncoderError("model is not loaded")
        blocks = getattr(getattr(model, "transformer", None), "blocks", None)
        if blocks is None:
            for module in model.modules():
                if isinstance(module, torch.nn.ModuleList) and len(module) > 1:
                    blocks = module
                    break
        if blocks is None:
            raise ProteinEncoderError(
                "could not locate the transformer block stack; refusing to guess "
                "which parameters to unfreeze"
            )
        return blocks

    def apply_freezing(self) -> None:
        """Freeze everything but the trailing ``n_trainable_layers`` blocks.

        Called from ``__init__`` and again after a checkpoint load, so an
        encoder is never in a state its ``n_trainable_layers`` does not
        describe. A model the caller injected that is not an ``nn.Module`` --
        a test double or a plain callable -- is left alone, since its parameters
        are the caller's to manage.

        The final LayerNorm is unfrozen alongside them, because it sits above the
        trainable blocks and the features are read *after* it; leaving it fixed
        would force the unfrozen blocks to preserve a normalisation statistic
        they are otherwise free to change.

        Embeddings and lower blocks stay frozen in every partial regime.
        """
        if self._model is None or not hasattr(self._model, "parameters"):
            return
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)
        if not self.n_trainable_layers:
            return
        blocks = self.transformer_blocks()
        if self.n_trainable_layers > len(blocks):
            raise ProteinEncoderError(
                f"n_trainable_layers={self.n_trainable_layers} exceeds the "
                f"{len(blocks)} blocks this checkpoint has"
            )
        for block in blocks[-self.n_trainable_layers :]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        final_norm = getattr(getattr(self._model, "transformer", None), "norm", None)
        if final_norm is not None:
            for parameter in final_norm.parameters():
                parameter.requires_grad_(True)

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Parameters an optimizer should receive. Empty when fully frozen."""
        self._ensure_loaded()
        return [p for p in self._model.parameters() if p.requires_grad]

    @property
    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    @property
    def is_frozen(self) -> bool:
        return self.n_trainable_layers == 0

    def assert_cacheable(self) -> None:
        """Refuse to cache features from an encoder that is being trained.

        The 14.023 GiB protein cache is valid only because the features are a
        deterministic function of the sequence. One trainable block ends that,
        and a cache written afterwards would be stale from the first optimizer
        step with nothing to signal it -- the same failure the ligand cache
        guards against, and more expensive here because re-encoding the corpus
        takes 40 minutes.
        """
        if not self.is_frozen:
            raise ProteinEncoderError(
                f"this encoder has {self.n_trainable_layers} trainable block(s) "
                f"({self.n_trainable:,} parameters), so its features are not a "
                "function of the sequence alone and must not be cached. Set "
                "n_trainable_layers=0 to cache, or encode live each step."
            )

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

        Memoised, because :meth:`metadata` includes it in every cache key and
        hashing the full state dict reads every parameter.
        """
        if getattr(self, "_weights_checksum_cache", None) is not None:
            return self._weights_checksum_cache  # type: ignore[return-value]
        digest = hashlib.sha256()
        for name, parameter in sorted(self.model.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        self._weights_checksum_cache = digest.hexdigest()
        return self._weights_checksum_cache

    def metadata(self, output_dim: int, *, weights_checksum: str | None = None) -> EncoderMetadata:
        """Encoder identity, as it appears in every cache key.

        ``weights_checksum`` defaults to the **actual** loaded weights rather than
        to ``None``. Leaving it out of the key would mean a corrupted or locally
        overridden checkpoint produced cache entries indistinguishable from the
        pinned ones -- which is the whole failure the checksum exists to catch.
        It also kept callers that built metadata independently from agreeing with
        the metadata attached to encoded features.

        Skipped only when the model was injected by the caller, which means a test
        double or a custom build whose identity is the caller's to assert.
        """
        if weights_checksum is None and not self._model_injected:
            weights_checksum = self.weights_checksum()
        return EncoderMetadata(
            name="esmc",
            checkpoint=self.checkpoint,
            output_dim=output_dim,
            precision=self.precision,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            revision=self.revision,
            weights_checksum=weights_checksum,
            preprocessing_policy=(
                "EsmcModel.last_hidden_state (post final LayerNorm); "
                "special tokens stripped via tokenizer mask; "
                f"window={self.window_size} overlap={self.overlap} merge=mean"
            ),
            extra={
                "context_limit": self.context_limit,
                # Numerics depend on these, so they key the cache.
                "kernel_backends": list(detect_kernel_backends()),
                "api": "native-EsmcModel",
                # A partially unfrozen encoder is a different contract: its
                # features move as it trains, so they must never collide with
                # frozen ones in a cache.
                #
                # Emitted **only when non-zero**, and that is not cosmetic. The
                # key is a hash over this dict, so adding a field unconditionally
                # would change the frozen encoder's key too and orphan the
                # already-populated 14.023 GiB cache, which was written before
                # this field existed. Frozen is the baseline contract and needs
                # no marker; a deviation from it does.
                **(
                    {"n_trainable_layers": self.n_trainable_layers}
                    if self.n_trainable_layers
                    else {}
                ),
            },
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

        # Take the post-LayerNorm final state, by name and nothing else.
        #
        # The legacy adapter fell back through `embeddings` then `hidden_states`
        # until something was not None. That is how you end up silently encoding
        # the pre-norm residual stream, which on this machine differs from the
        # post-norm state by a max of ~6.2e3. There is no sensible fallback
        # here: if the field is missing, our model assumption is wrong.
        embeddings = getattr(output, "last_hidden_state", None)
        if embeddings is None:
            raise ProteinEncoderError(
                f"{type(output).__name__} has no 'last_hidden_state'. This adapter targets the "
                "native EsmcModel API; refusing to guess which tensor holds the residue states."
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

        The input must already be canonical (stripped, upper case). Upper-casing
        here instead would make two spellings of one protein hash to two cache
        keys holding identical features, so canonicalization belongs to the data
        layer -- but the check is explicit, because the failure it prevents is
        severe and its symptom is otherwise baffling: ESM-C tokenizes lower-case
        residues to ``<unk>``, ``<unk>`` is a *special* token, and special-token
        stripping therefore removes every position of the protein.
        """
        if not sequence:
            raise ProteinEncoderError("cannot encode an empty sequence")
        if sequence != sequence.strip().upper():
            raise ProteinEncoderError(
                "sequence is not canonical (expected stripped and upper-case). "
                "ESM-C maps lower-case residues to <unk>, which is a special "
                "token, so every residue would be stripped. Normalize with "
                "bindra.data.bindingdb.proteins.canonical_sequence first."
            )

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
        if "600M" in self.checkpoint.upper() and output_dim != EXPECTED_ESMC_600M_DIM:
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
    "ACCELERATION_LIBRARIES",
    "DEFAULT_CONTEXT_LIMIT",
    "ESMC_600M_REPO",
    "ESMC_600M_REVISION",
    "ESMC_600M_WEIGHTS_SHA256",
    "ESMCProteinEncoder",
    "EXPECTED_ESMC_600M_DIM",
    "FEATURE_SCHEMA_VERSION",
    "ProteinEncoderError",
    "WindowPlan",
    "detect_kernel_backends",
    "plan_windows",
    "strip_special_tokens",
]
