"""Fixed-width cache for frozen ligand molecule vectors.

Written for the frozen KPGT arm. KPGT's forward pass costs 2.5 hours per epoch
over the curated corpus against 5 minutes for the trained GAT arm -- 300 mol/s
versus 9,189 on the RTX 5060 Ti, because a real molecule's triplet graph averages
36 nodes but **559 edges**. Frozen, its 2,304-dim readout is a deterministic
function of the SMILES, so it is computed once and read back forever: 6.2 GiB in
bf16 for all 1,436,561 curated molecules.

Deliberately separate from :class:`~bindra.features.cache.ProteinFeatureCache`
rather than reusing it. That cache is *ragged* -- proteins have different
residue counts, so it carries per-entry offsets and lengths and memory-maps one
shard at a time to avoid loading 14 GiB. A molecule vector is fixed width, which
makes the whole apparatus unnecessary: rows are addressable arithmetically, a
shard is a plain `[rows, dim]` matrix, and a lookup is one index. Forcing the
ragged structure onto fixed-width data would add machinery that only has to be
maintained and can only go wrong.

The discipline is the same, and it is the part that matters:

* **Encoder identity is in the key**, through ``EncoderMetadata.cache_key``, so
  changing the checkpoint, the config or the featurization policy cannot reuse
  stale vectors. An untrained KPGT reports ``checkpoint="untrained"`` and
  therefore keys differently from any pretrained one.
* **One cache holds one encoder contract.** Mixing two is refused, not merged.
* **The index is written after the shard.** An interrupted run leaves a smaller
  cache, never one pointing at bytes that do not exist.
* **bf16 is stored as raw bits** (``uint16``), never silently up- or
  down-converted, so the precision the cache reports is the precision it holds.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch

from bindra.features.cache import FeatureCacheError, StaleCacheError
from bindra.features.types import EncoderMetadata

#: Rows per shard. At 2,304 bf16 columns this is ~1.1 GiB per file, which keeps
#: a single shard comfortably memory-mappable while bounding the file count.
DEFAULT_SHARD_ROWS = 250_000

INDEX_VERSION = 2


#: Raw integer type each storage dtype is written as. The cache stores raw bits
#: and views them back, never converting, so a value read is bit-identical to
#: the value written.
#:
#: bf16 is right for encoder activations, where ~3 decimal digits is far below
#: the noise in the thing being stored. It is wrong for KPGT's knowledge inputs:
#: the 200 descriptors are CDF-normalised values the published weights were
#: fitted against, and rounding them to bf16 would move the input distribution
#: out from under any future pretrained checkpoint.
RAW_DTYPE = {torch.bfloat16: np.uint16, torch.float32: np.uint32}


@dataclass(frozen=True)
class VectorEntry:
    """Where one molecule's vector lives."""

    shard: int
    row: int
    molecule_id: str

    def to_dict(self) -> dict[str, Any]:
        return {"shard": self.shard, "row": self.row, "molecule_id": self.molecule_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "VectorEntry":
        return cls(
            shard=int(payload["shard"]),
            row=int(payload["row"]),
            molecule_id=str(payload["molecule_id"]),
        )


class LigandVectorCache:
    """Append-only fixed-width cache of molecule vectors, keyed by encoder identity."""

    def __init__(
        self,
        directory: Path | str,
        *,
        metadata: EncoderMetadata | None = None,
        dim: int | None = None,
        shard_rows: int = DEFAULT_SHARD_ROWS,
        mode: str = "r",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if dtype not in RAW_DTYPE:
            raise FeatureCacheError(
                f"unsupported storage dtype {dtype}; expected one of "
                f"{sorted(str(d) for d in RAW_DTYPE)}"
            )
        self.directory = Path(directory)
        self.dtype = dtype
        self.raw_dtype = RAW_DTYPE[dtype]
        self.shard_rows = shard_rows
        self.mode = mode
        self.dim = dim
        self._entries: dict[str, VectorEntry] = {}
        self._metadata_payload: dict[str, Any] | None = (
            metadata.to_dict() if metadata is not None else None
        )
        self._open_shards: dict[int, np.memmap] = {}
        self._pending: list[tuple[str, str, np.ndarray]] = []
        self._next_shard = 0

        if self.index_path.exists():
            self._load_index()
        elif mode == "r":
            raise FeatureCacheError(
                f"no cache index at {self.index_path}. Populate it with "
                "`bindra cache ligands` before reading."
            )
        else:
            self.directory.mkdir(parents=True, exist_ok=True)

    # -- paths -------------------------------------------------------------

    @property
    def raw_dtype_torch(self) -> torch.dtype:
        """The torch integer type matching :attr:`raw_dtype`, for bit views."""
        return torch.uint16 if self.dtype is torch.bfloat16 else torch.int32

    @property
    def index_path(self) -> Path:
        return self.directory / "index.json"

    def shard_path(self, shard: int) -> Path:
        return self.directory / f"vectors-{shard:05d}.bin"

    # -- index -------------------------------------------------------------

    def _load_index(self) -> None:
        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        if int(payload.get("index_version", 0)) != INDEX_VERSION:
            raise StaleCacheError(
                f"cache index at {self.index_path} is version "
                f"{payload.get('index_version')}, this build writes {INDEX_VERSION}"
            )
        stored = payload["encoder_metadata"]
        if self._metadata_payload is not None and stored != self._metadata_payload:
            raise StaleCacheError(
                "cached vectors were produced by a different encoder contract.\n"
                f"  cached:    {json.dumps(stored, sort_keys=True)}\n"
                f"  requested: {json.dumps(self._metadata_payload, sort_keys=True)}\n"
                "Re-encode rather than reusing these."
            )
        self._metadata_payload = stored
        cached_dim = int(payload["dim"])
        if self.dim is not None and cached_dim != self.dim:
            raise StaleCacheError(
                f"cache holds {cached_dim}-dim vectors, {self.dim} was requested"
            )
        self.dim = cached_dim
        self.shard_rows = int(payload["shard_rows"])
        self._entries = {
            key: VectorEntry.from_dict(value) for key, value in payload["entries"].items()
        }
        self._next_shard = int(payload["n_shards"])

    def _write_index(self) -> None:
        if self._metadata_payload is None or self.dim is None:
            raise FeatureCacheError("cannot write an index before anything is cached")
        payload = {
            "index_version": INDEX_VERSION,
            "encoder_metadata": self._metadata_payload,
            "dim": self.dim,
            "dtype": str(self.dtype).removeprefix("torch."),
            "shard_rows": self.shard_rows,
            "n_shards": self._next_shard,
            "n_entries": len(self._entries),
            "entries": {key: entry.to_dict() for key, entry in self._entries.items()},
        }
        temporary = self.index_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        temporary.replace(self.index_path)

    # -- reading -----------------------------------------------------------

    def __contains__(self, cache_key: str) -> bool:
        return cache_key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def keys(self) -> frozenset[str]:
        return frozenset(self._entries)

    @property
    def total_bytes(self) -> int:
        return len(self._entries) * (self.dim or 0) * 2

    def metadata(self) -> EncoderMetadata:
        if self._metadata_payload is None:
            raise FeatureCacheError("cache holds no encoder metadata yet")
        payload = dict(self._metadata_payload)
        payload["extra"] = dict(payload.get("extra") or {})
        return EncoderMetadata(**payload)

    def _shard(self, index: int) -> np.memmap:
        if index not in self._open_shards:
            path = self.shard_path(index)
            if not path.exists():
                raise FeatureCacheError(f"shard {index} missing at {path}")
            # From the storage dtype, not a hardcoded 2: a wrong itemsize
            # asks to map past the end of the file, which Windows reports
            # as "not enough memory resources" rather than as a bad size.
            itemsize = np.dtype(self.raw_dtype).itemsize
            rows = path.stat().st_size // (itemsize * (self.dim or 1))
            self._open_shards[index] = np.memmap(
                path, dtype=self.raw_dtype, mode="r", shape=(rows, self.dim)
            )
        return self._open_shards[index]

    def get(self, cache_key: str) -> torch.Tensor:
        """One molecule vector as bf16. Raises if absent."""
        try:
            entry = self._entries[cache_key]
        except KeyError:
            raise FeatureCacheError(
                f"{cache_key} is not cached; encode it before reading"
            ) from None
        raw = np.asarray(self._shard(entry.shard)[entry.row])
        return torch.from_numpy(raw.copy()).view(self.dtype)

    def get_many(self, cache_keys: Sequence[str]) -> torch.Tensor:
        """``[len(cache_keys), dim]`` bf16, in the order given.

        Grouped by shard internally so a batch drawn from one shard touches it
        once, which is the common case after a sequential encode.
        """
        if not cache_keys:
            return torch.empty(0, self.dim or 0, dtype=self.dtype)
        out = np.empty((len(cache_keys), self.dim), dtype=self.raw_dtype)
        by_shard: dict[int, list[tuple[int, int]]] = {}
        for position, key in enumerate(cache_keys):
            try:
                entry = self._entries[key]
            except KeyError:
                raise FeatureCacheError(f"{key} is not cached") from None
            by_shard.setdefault(entry.shard, []).append((position, entry.row))
        for shard, items in by_shard.items():
            memmap = self._shard(shard)
            positions = [p for p, _ in items]
            rows = [r for _, r in items]
            out[positions] = memmap[rows]
        return torch.from_numpy(out).view(self.dtype)

    # -- writing -----------------------------------------------------------

    def put(self, cache_key: str, vector: torch.Tensor, *, molecule_id: str = "") -> None:
        if self.mode == "r":
            raise FeatureCacheError("cache opened read-only")
        if cache_key in self._entries:
            return
        if vector.ndim != 1:
            raise FeatureCacheError(
                f"expected a 1-D molecule vector, got {tuple(vector.shape)}"
            )
        if vector.dtype is not self.dtype:
            raise FeatureCacheError(
                f"expected {self.dtype}, got {vector.dtype}. The cache stores raw "
                "bits; converting here would misreport precision."
            )
        width = int(vector.shape[0])
        if self.dim is None:
            self.dim = width
        elif width != self.dim:
            raise FeatureCacheError(
                f"vector width {width} disagrees with the cache's {self.dim}"
            )
        block = vector.contiguous().view(self.raw_dtype_torch).cpu().numpy()
        self._pending.append((cache_key, molecule_id, block))
        if len(self._pending) >= self.shard_rows:
            self.flush()

    def put_batch(
        self,
        cache_keys: Sequence[str],
        vectors: torch.Tensor,
        *,
        molecule_ids: Sequence[str] | None = None,
    ) -> None:
        if vectors.shape[0] != len(cache_keys):
            raise FeatureCacheError(
                f"{vectors.shape[0]} vectors for {len(cache_keys)} keys"
            )
        identifiers = molecule_ids or [""] * len(cache_keys)
        for key, vector, identifier in zip(cache_keys, vectors, identifiers):
            self.put(key, vector, molecule_id=identifier)

    def adopt_metadata(self, metadata: EncoderMetadata) -> None:
        """Declare the encoder contract, refusing to mix two in one cache."""
        payload = metadata.to_dict()
        if self._metadata_payload is None:
            self._metadata_payload = payload
        elif payload != self._metadata_payload:
            raise StaleCacheError(
                "refusing to mix vectors from two encoder contracts in one cache"
            )

    def flush(self) -> None:
        """Write the buffer as one complete new shard, then rewrite the index."""
        if self.mode == "r" or not self._pending:
            return
        if self._metadata_payload is None:
            raise FeatureCacheError(
                "no encoder metadata; call adopt_metadata() before flushing"
            )
        shard = self._next_shard
        path = self.shard_path(shard)
        block = np.stack([vector for _, _, vector in self._pending])
        temporary = path.with_suffix(".bin.tmp")
        with open(temporary, "wb") as handle:
            block.tofile(handle)
        temporary.replace(path)

        for row, (key, molecule_id, _) in enumerate(self._pending):
            self._entries[key] = VectorEntry(shard=shard, row=row, molecule_id=molecule_id)
        self._next_shard += 1
        self._pending.clear()
        self._write_index()

    def stats(self) -> dict[str, Any]:
        return {
            "entries": len(self._entries),
            "dim": self.dim,
            "shards": self._next_shard,
            "bytes": self.total_bytes,
            "gib": round(self.total_bytes / 2**30, 3),
            "encoder": (self._metadata_payload or {}).get("name"),
            "checkpoint": (self._metadata_payload or {}).get("checkpoint"),
        }

    def __enter__(self) -> "LigandVectorCache":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.flush()


def encode_to_cache(
    encoder: Any,
    smiles: Iterable[str],
    cache: LigandVectorCache,
    *,
    batch_size: int = 64,
    progress: bool = False,
) -> dict[str, int]:
    """Encode molecules into ``cache``, skipping any already present.

    Resumable by construction: the cache key is a content hash of the SMILES and
    the encoder identity, so a rerun recomputes only what is missing. That
    matters at this scale -- a full corpus pass is hours, and losing it to an
    interruption once was enough.
    """
    # A cached vector is valid only because it is a deterministic function of the
    # SMILES. If any encoder parameter is trainable that stops being true, and the
    # cache would go stale after the first optimizer step with nothing to signal it.
    if hasattr(encoder, "assert_cacheable"):
        encoder.assert_cacheable()
    metadata = encoder.metadata()
    cache.adopt_metadata(metadata)
    counts = {"encoded": 0, "skipped": 0, "failed": 0}
    pending: list[str] = []

    def drain() -> None:
        if not pending:
            return
        try:
            features = encoder.encode_batch(pending)
        except Exception:  # noqa: BLE001 - one bad molecule must not lose the run
            for one in pending:
                try:
                    produced = encoder.encode_batch([one])[0]
                except Exception:  # noqa: BLE001
                    counts["failed"] += 1
                    continue
                cache.put(
                    metadata.cache_key(one),
                    produced.graph_features.to(torch.bfloat16),
                    molecule_id=one,
                )
                counts["encoded"] += 1
            pending.clear()
            return
        for one, produced in zip(pending, features):
            cache.put(
                metadata.cache_key(one),
                produced.graph_features.to(torch.bfloat16),
                molecule_id=one,
            )
            counts["encoded"] += 1
        pending.clear()

    iterator: Iterator[str] = iter(smiles)
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="encoding ligands", unit="mol")
        except ImportError:
            pass

    for one in iterator:
        if metadata.cache_key(one) in cache:
            counts["skipped"] += 1
            continue
        pending.append(one)
        if len(pending) >= batch_size:
            drain()
    drain()
    cache.flush()
    return counts


__all__ = [
    "DEFAULT_SHARD_ROWS",
    "LigandVectorCache",
    "VectorEntry",
    "encode_to_cache",
]
