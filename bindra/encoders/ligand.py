"""The ligand encoder seam: one contract, several interchangeable encoders.

The specification asks for KPGT as the ligand encoder and for the legacy GAT as
ablation 1. Those are two encoders with different token semantics -- KPGT emits
bond and knowledge tokens, a GAT emits one token per atom -- so the fusion stack,
the iBAM exporter and the retrieval projections must talk to an interface rather
than to either one.

That seam is load-bearing here for a second reason. DGL, which KPGT is written
against, cannot run on this machine's GPU at all: its final release (2.2.1)
loads only under torch <= 2.3.0, built for CUDA 12.1, and this GPU is compute
capability 12.0. See ``envs/kpgt/README.md`` for the measurement. So the KPGT
arm needs a verified pure-PyTorch port before it can exist, while everything
downstream of the ligand encoder can be built and tested now against the GAT
arm. The registry is what lets that happen without the rest of the system
knowing which arm is installed.

**One token is not one atom.** It happens to be, for the GAT encoder, and that
is *stated* in the ``token_types`` it emits rather than assumed by any consumer.
Anything that aggregates tokens to atoms must go through
``LigandFeatures.token_to_atom``.
"""

from __future__ import annotations

from typing import Callable, Iterable, Mapping, Protocol, Sequence, runtime_checkable

import torch

from bindra.features.types import EncoderMetadata, LigandFeatures


class LigandEncoderError(RuntimeError):
    """The encoder could not produce features."""


@runtime_checkable
class LigandEncoder(Protocol):
    """What every ligand encoder must provide.

    Deliberately small. Anything an encoder needs that is specific to it --
    pretrained weights, a knowledge node, a window plan -- stays inside it and
    travels downstream only through :class:`EncoderMetadata`.
    """

    @property
    def token_dim(self) -> int:
        """Width of each token's representation."""

    @property
    def graph_dim(self) -> int:
        """Width of the whole-molecule readout.

        Separate from :attr:`token_dim` because a pooled readout is usually
        wider (mean concatenated with max, for instance), and the cross-attention
        adapters need to know which is which.
        """

    def metadata(self) -> EncoderMetadata:
        """Identity of this encoder, for the cache key and for provenance."""

    def encode(
        self,
        smiles: str,
        *,
        atom_index_maps: Mapping[str, tuple[int, ...]] | None = None,
    ) -> LigandFeatures:
        """Encode one molecule.

        ``atom_index_maps`` carries the standardizer's ``std_to_orig`` and
        ``std_to_output`` arrays when the caller has them. Without it, an
        attention weight can be attributed to a standardized atom index but not
        drawn on the depiction the user sees, so the maps are passed through
        rather than reconstructed here.
        """

    def encode_batch(
        self,
        smiles: Sequence[str],
        *,
        atom_index_maps: Sequence[Mapping[str, tuple[int, ...]]] | None = None,
    ) -> list[LigandFeatures]:
        """Encode several molecules in one pass.

        Required rather than optional: per-molecule encoding left the protein
        cache at ~8% GPU utilisation, and molecules are far smaller than
        proteins, so the same mistake would cost more here. Implementations must
        produce output identical to repeated :meth:`encode` calls, and
        ``tests/test_ligand_kpgt.py`` asserts exactly that.
        """


_REGISTRY: dict[str, Callable[..., LigandEncoder]] = {}


def register_ligand_encoder(name: str) -> Callable[[Callable[..., LigandEncoder]], Callable[..., LigandEncoder]]:
    """Register an encoder factory under ``name``.

    Refuses to replace an existing name: silently shadowing an encoder would
    make a results table ambiguous about which arm produced it.
    """

    def decorate(factory: Callable[..., LigandEncoder]) -> Callable[..., LigandEncoder]:
        if name in _REGISTRY and _REGISTRY[name] is not factory:
            raise ValueError(f"ligand encoder {name!r} is already registered")
        _REGISTRY[name] = factory
        return factory

    return decorate


def available_ligand_encoders() -> tuple[str, ...]:
    """Names that :func:`build_ligand_encoder` will accept, sorted."""
    _load_builtin_encoders()
    return tuple(sorted(_REGISTRY))


def build_ligand_encoder(name: str, **kwargs: object) -> LigandEncoder:
    """Construct a registered encoder.

    The error names what *is* available, because the common case for reaching it
    is asking for ``kpgt`` -- which is unavailable for a reason worth reading.
    """
    _load_builtin_encoders()
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise LigandEncoderError(
            f"unknown ligand encoder {name!r}; available: {', '.join(sorted(_REGISTRY))}"
        ) from None
    return factory(**kwargs)


def _load_builtin_encoders() -> None:
    """Import the modules that register encoders.

    Done lazily so that importing this module does not pull in torch modules or
    pretrained weights, and so a broken optional encoder cannot break the seam.
    """
    from bindra.encoders import ligand_kpgt  # noqa: F401  (registers "kpgt")


def segment_softmax(
    logits: torch.Tensor, index: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Softmax over the groups given by ``index``, stabilised by segment maxima.

    ``logits`` is ``[E, H]`` and ``index`` is ``[E]`` naming the destination of
    each edge; the result sums to 1 over the edges sharing a destination.

    The subtraction is not decoration. Attention logits here include a learned
    edge term, so they are unbounded during training; a plain ``exp`` overflows
    to ``inf`` and then ``inf/inf`` yields NaN, which propagates through the
    whole batch. Segments with no edges produce no output rows, so the
    denominator is clamped rather than left at zero.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be [E, H], got {tuple(logits.shape)}")
    num_heads = logits.shape[1]
    maxima = logits.new_full((num_segments, num_heads), float("-inf"))
    maxima.scatter_reduce_(
        0, index.unsqueeze(-1).expand_as(logits), logits, reduce="amax", include_self=True
    )
    # A segment with no edges keeps -inf, which would make 0 - (-inf) = inf.
    maxima = torch.where(torch.isinf(maxima), torch.zeros_like(maxima), maxima)
    shifted = (logits - maxima[index]).exp()
    totals = torch.zeros_like(maxima)
    totals.index_add_(0, index, shifted)
    return shifted / totals[index].clamp_min(torch.finfo(shifted.dtype).tiny)


def segment_sum(
    values: torch.Tensor, index: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Sum ``values`` ``[E, ...]`` into ``num_segments`` rows by ``index``."""
    shape = (num_segments,) + tuple(values.shape[1:])
    out = values.new_zeros(shape)
    out.index_add_(0, index, values)
    return out


def segment_mean(
    values: torch.Tensor, index: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Mean of ``values`` per segment; empty segments give zero, not NaN."""
    totals = segment_sum(values, index, num_segments)
    counts = torch.zeros(num_segments, dtype=values.dtype, device=values.device)
    counts.index_add_(0, index, torch.ones_like(index, dtype=values.dtype))
    return totals / counts.clamp_min(1.0).unsqueeze(-1)


def segment_max(
    values: torch.Tensor, index: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Max of ``values`` per segment; empty segments give zero, not ``-inf``.

    Returning ``-inf`` for an empty segment would poison any downstream linear
    layer, and an empty segment is reachable: a batch can legitimately contain
    no atoms for a graph only if collation is wrong, but the pooled readout runs
    before that is detected.
    """
    shape = (num_segments,) + tuple(values.shape[1:])
    out = values.new_full(shape, float("-inf"))
    out.scatter_reduce_(
        0,
        index.reshape(-1, *([1] * (values.ndim - 1))).expand_as(values),
        values,
        reduce="amax",
        include_self=True,
    )
    return torch.where(torch.isinf(out), torch.zeros_like(out), out)


__all__ = [
    "LigandEncoder",
    "LigandEncoderError",
    "available_ligand_encoders",
    "build_ligand_encoder",
    "register_ligand_encoder",
    "segment_max",
    "segment_mean",
    "segment_softmax",
    "segment_sum",
]
