"""Sharded, memory-mapped feature cache for frozen encoder outputs.

Replaces the legacy design, which called ``torch.load`` on the whole cache at
construction. That was tolerable at 229 proteins and 367 MB; the protein cache
for this corpus is **14.03 GiB measured** (6,536,409 residues x 1152 dims x bf16),
and the ligand cache will be larger still.

Three properties the legacy cache lacked:

**Ragged, not padded.** Construct lengths run from 7 to 34,350 residues. Padding
every sequence to the maximum would turn 14.03 GiB into roughly 850 GiB, so
features are stored concatenated with an offset per entry.

**Bit-exact bf16.** numpy has no bfloat16, so the raw 16 bits are stored as
``uint16`` and viewed back as ``torch.bfloat16``. Verified lossless. Storing
float16 instead would silently change the numbers the encoder produced, and the
cache key claims bf16.

**Stale caches are refused, not served.** The index records the full
``EncoderMetadata``; any disagreement with the live encoder raises. The legacy
keys were ``sha1(sequence)[:16]``, so changing checkpoint, precision or
preprocessing silently reused incompatible features.

``residue_mask`` and ``residue_indices`` are not stored. The adapter always emits
``ones(L)`` and ``arange(L)`` -- windows are merged back onto the full sequence --
so they are reconstructed, and the invariant is asserted on write rather than
assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator, Mapping

import numpy as np
import torch

from bindra.features.types import EncoderMetadata, ProteinFeatures

CACHE_SCHEMA_VERSION = "feature-cache-v1"

#: Target bytes per shard. Large enough that per-file overhead is irrelevant,
#: small enough that one shard memory-maps comfortably and a partial run leaves
#: usable shards behind.
DEFAULT_SHARD_BYTES = 2 * 1024**3

#: Environment override for the cache root, so bulk artifacts can be moved off a
#: synced folder without touching any call site.
CACHE_ROOT_ENV = "BINDRA_CACHE_ROOT"


class FeatureCacheError(RuntimeError):
    """The cache could not be read, written, or trusted."""


class StaleCacheError(FeatureCacheError):
    """The cache was written by a different encoder contract."""


def default_cache_root(project_root: Path | str = ".") -> Path:
    """``$BINDRA_CACHE_ROOT`` if set, else ``<project>/data/cache``."""
    override = os.environ.get(CACHE_ROOT_ENV)
    if override:
        return Path(override)
    return Path(project_root) / "data" / "cache"


@dataclass(frozen=True)
class CacheEntry:
    shard: int
    offset: int
    n_residues: int
    dim: int
    sequence_length: int
    sequence_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard": self.shard,
            "offset": self.offset,
            "n_residues": self.n_residues,
            "dim": self.dim,
            "sequence_length": self.sequence_length,
            "sequence_hash": self.sequence_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CacheEntry":
        return cls(
            shard=int(payload["shard"]),
            offset=int(payload["offset"]),
            n_residues=int(payload["n_residues"]),
            dim=int(payload["dim"]),
            sequence_length=int(payload["sequence_length"]),
            sequence_hash=str(payload["sequence_hash"]),
        )


class ProteinFeatureCache:
    """Append-only ragged cache of residue features, keyed by encoder identity.

    Typical use is write-once then read-many, and both paths stream: a reader
    memory-maps one shard at a time rather than loading 14 GiB.
    """

    def __init__(
        self,
        directory: Path | str,
        *,
        metadata: EncoderMetadata | None = None,
        shard_bytes: int = DEFAULT_SHARD_BYTES,
        mode: str = "r",
    ) -> None:
        self.directory = Path(directory)
        self.shard_bytes = shard_bytes
        self.mode = mode
        self._entries: dict[str, CacheEntry] = {}
        self._metadata: EncoderMetadata | None = metadata
        self._metadata_payload: dict[str, Any] | None = (
            metadata.to_dict() if metadata is not None else None
        )
        self._open_shards: dict[int, np.memmap] = {}

        # Write buffer. Entries are assigned a shard and offset at *flush* time,
        # not at put time, so the index never references bytes that are not yet on
        # disk -- an interrupted run leaves a smaller cache, never a torn one.
        self._pending: list[tuple[str, CacheEntry, np.ndarray]] = []
        self._buffer_rows = 0
        self._buffer_bytes = 0
        self._next_shard = 0

        if self.index_path.exists():
            self._load_index()
        elif mode == "r":
            raise FeatureCacheError(
                f"no cache index at {self.index_path}. Populate it with "
                "`bindra cache proteins` before reading."
            )
        else:
            self.directory.mkdir(parents=True, exist_ok=True)

    # -- paths -------------------------------------------------------------

    @property
    def index_path(self) -> Path:
        return self.directory / "index.json"

    def shard_path(self, shard: int) -> Path:
        return self.directory / f"shard-{shard:05d}.npy"

    # -- index -------------------------------------------------------------

    def _load_index(self) -> None:
        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise StaleCacheError(
                f"cache schema is {payload.get('schema_version')!r}, this code "
                f"writes {CACHE_SCHEMA_VERSION!r}"
            )
        stored = payload["encoder_metadata"]
        if self._metadata_payload is not None and stored != self._metadata_payload:
            raise StaleCacheError(
                "cached features were produced by a different encoder contract and "
                "will not be served.\n"
                f"  cached: {json.dumps(stored, sort_keys=True)[:400]}\n"
                f"  live:   {json.dumps(self._metadata_payload, sort_keys=True)[:400]}\n"
                "Re-encode, or point at a different cache directory."
            )
        self._metadata_payload = stored
        self._entries = {
            key: CacheEntry.from_dict(value) for key, value in payload["entries"].items()
        }
        # Resume writes into a fresh shard rather than appending to the last one.
        # Appending would mean reopening a file that may be memory-mapped by a
        # reader (which Windows refuses) and rewriting it whole each flush, which
        # is quadratic in shard size.
        self._next_shard = max(
            (entry.shard for entry in self._entries.values()), default=-1
        ) + 1

    def _write_index(self) -> None:
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "encoder_metadata": self._metadata_payload,
            "storage": {
                "dtype": "uint16 holding raw bfloat16 bits",
                "layout": "ragged: concatenated [sum(L_i), dim], per-entry offset",
                "reconstructed_fields": [
                    "residue_mask = ones(L)",
                    "residue_indices = arange(L)",
                ],
            },
            "n_entries": len(self._entries),
            "n_shards": self._next_shard,
            "total_residues": sum(e.n_residues for e in self._entries.values()),
            "entries": {key: entry.to_dict() for key, entry in self._entries.items()},
        }
        # Atomic: an interrupted run must not leave a torn index pointing at
        # offsets that were never written.
        handle = tempfile.NamedTemporaryFile(
            "w", delete=False, dir=self.directory, suffix=".tmp", encoding="utf-8"
        )
        try:
            json.dump(payload, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            handle.close()
        os.replace(handle.name, self.index_path)

    # -- membership --------------------------------------------------------

    def __contains__(self, cache_key: str) -> bool:
        if cache_key in self._entries:
            return True
        return any(key == cache_key for key, _, _ in self._pending)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def keys(self) -> frozenset[str]:
        return frozenset(self._entries)

    @property
    def total_residues(self) -> int:
        return sum(entry.n_residues for entry in self._entries.values())

    @property
    def total_bytes(self) -> int:
        return sum(entry.n_residues * entry.dim * 2 for entry in self._entries.values())

    # -- writing -----------------------------------------------------------

    def put(self, cache_key: str, features: ProteinFeatures) -> None:
        if self.mode == "r":
            raise FeatureCacheError("cache opened read-only")
        if cache_key in self._entries:
            return

        if self._metadata_payload is None:
            self._metadata_payload = features.encoder_metadata.to_dict()
        elif features.encoder_metadata.to_dict() != self._metadata_payload:
            raise StaleCacheError(
                "refusing to mix features from two encoder contracts in one cache"
            )

        tensor = features.residue_features
        if tensor.dtype is not torch.bfloat16:
            raise FeatureCacheError(
                f"expected bfloat16 residue features, got {tensor.dtype}. The cache "
                "stores raw bf16 bits; converting here would misreport precision."
            )
        # Assert the invariant instead of storing the arrays it implies.
        length = int(tensor.shape[0])
        if not torch.equal(
            features.residue_indices, torch.arange(length, dtype=features.residue_indices.dtype)
        ):
            raise FeatureCacheError(
                "residue_indices is not arange(L); this cache reconstructs it, so a "
                "non-trivial numbering would be lost. Store it explicitly instead."
            )
        if not bool(features.residue_mask.all()):
            raise FeatureCacheError(
                "residue_mask is not all-True; this cache reconstructs it."
            )

        block = tensor.contiguous().view(torch.uint16).cpu().numpy()
        entry = CacheEntry(
            shard=-1,  # assigned at flush, when the bytes actually land
            offset=self._buffer_rows,
            n_residues=length,
            dim=int(tensor.shape[1]),
            sequence_length=features.sequence_length,
            sequence_hash=features.sequence_hash,
        )
        self._pending.append((cache_key, entry, block))
        self._buffer_rows += length
        self._buffer_bytes += block.nbytes

        if self._buffer_bytes >= self.shard_bytes:
            self.flush()

    def flush(self) -> None:
        """Write the buffer as one complete new shard, then rewrite the index.

        Each flush produces a whole file. Nothing is ever appended to an existing
        shard, which keeps writes linear and avoids reopening a file a reader may
        have memory-mapped. The index is written only after the shard is on disk,
        so it can never point at bytes that do not exist.
        """
        if self.mode == "r":
            return
        if self._pending:
            shard = self._next_shard
            path = self.shard_path(shard)
            block = np.concatenate([b for _, _, b in self._pending], axis=0)
            temporary = path.with_name(path.name + ".tmp")
            # Through an open handle: np.save appends ".npy" to any path that does
            # not already end in it, which would write beside the temp name.
            with open(temporary, "wb") as handle:
                np.save(handle, block)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            for key, entry, _ in self._pending:
                self._entries[key] = CacheEntry(
                    shard=shard,
                    offset=entry.offset,
                    n_residues=entry.n_residues,
                    dim=entry.dim,
                    sequence_length=entry.sequence_length,
                    sequence_hash=entry.sequence_hash,
                )
            self._next_shard += 1
            self._pending.clear()
            self._buffer_rows = 0
            self._buffer_bytes = 0
        self._write_index()

    # -- reading -----------------------------------------------------------

    def _shard(self, index: int) -> np.memmap:
        shard = self._open_shards.get(index)
        if shard is None:
            path = self.shard_path(index)
            if not path.exists():
                raise FeatureCacheError(f"shard {index} is missing: {path}")
            shard = np.load(path, mmap_mode="r")
            self._open_shards[index] = shard
        return shard

    def get(self, cache_key: str) -> ProteinFeatures:
        try:
            entry = self._entries[cache_key]
        except KeyError as exc:
            raise FeatureCacheError(f"{cache_key!r} is not in this cache") from exc

        shard = self._shard(entry.shard)
        block = np.asarray(shard[entry.offset : entry.offset + entry.n_residues])
        if block.shape[0] != entry.n_residues:
            raise FeatureCacheError(
                f"shard {entry.shard} is shorter than the index claims for {cache_key!r}"
            )
        tensor = torch.from_numpy(block.copy()).view(torch.bfloat16)
        return ProteinFeatures(
            residue_features=tensor,
            residue_mask=torch.ones(entry.n_residues, dtype=torch.bool),
            residue_indices=torch.arange(entry.n_residues, dtype=torch.long),
            sequence_hash=entry.sequence_hash,
            encoder_metadata=self.metadata,
            sequence_length=entry.sequence_length,
        )

    @property
    def metadata(self) -> EncoderMetadata:
        if self._metadata is not None:
            return self._metadata
        if self._metadata_payload is None:
            raise FeatureCacheError("cache has no encoder metadata yet")
        payload = dict(self._metadata_payload)
        return EncoderMetadata(
            name=payload["name"],
            checkpoint=payload["checkpoint"],
            output_dim=int(payload["output_dim"]),
            precision=payload["precision"],
            feature_schema_version=payload["feature_schema_version"],
            revision=payload.get("revision"),
            weights_checksum=payload.get("weights_checksum"),
            preprocessing_policy=payload.get("preprocessing_policy"),
            extra=payload.get("extra", {}),
        )

    def stats(self) -> dict[str, Any]:  # noqa: D102
        lengths = [entry.n_residues for entry in self._entries.values()]
        lengths.sort()
        return {
            "n_entries": len(self._entries),
            "n_shards": self._next_shard,
            "total_residues": sum(lengths),
            "total_gib": round(self.total_bytes / 1024**3, 3),
            "residues_min": lengths[0] if lengths else 0,
            "residues_median": lengths[len(lengths) // 2] if lengths else 0,
            "residues_max": lengths[-1] if lengths else 0,
        }

    def __enter__(self) -> "ProteinFeatureCache":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None and self.mode != "r":
            self.flush()


__all__ = [
    "CACHE_ROOT_ENV",
    "CACHE_SCHEMA_VERSION",
    "DEFAULT_SHARD_BYTES",
    "CacheEntry",
    "FeatureCacheError",
    "ProteinFeatureCache",
    "StaleCacheError",
    "default_cache_root",
]
