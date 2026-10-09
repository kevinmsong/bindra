"""Cache for KPGT's two knowledge inputs: the fingerprint and the descriptors.

Measured on 150 real curated molecules, single core:

====================  =========  ======
part                  ms/mol     share
====================  =========  ======
RDKit descriptors     57.74      81%
triplet graph         10.20      14%
fingerprint            3.73       5%
**total**             **71.67**
====================  =========  ======

The descriptors alone are 81% of ligand featurization, and featurization is what
the training loop waits on: 71.67 ms/mol is 14 mol/s per core, and eight worker
processes give ~112 mol/s against an observed ~98 pairs/s. The pool is
saturated and the GPU is idle behind it.

Both cached parts are **deterministic functions of the standardized SMILES**,
which is the only thing that makes caching them valid. The triplet graph is left
live: it is the remaining 14%, and caching it would cost far more disk for a
sixth of the saving.

Stored as **float32, not bf16**. The 200 descriptors are CDF-normalised values
that KPGT's published weights were fitted against; bf16 carries about three
decimal digits, and rounding them would move the input distribution out from
under any future pretrained checkpoint. The fingerprint is binary and would
survive either way. 1,436,561 molecules x 712 floats is ~4.1 GiB.

The key is a content hash of the SMILES **and** of the policy that produced the
vectors, so a changed fingerprint size or descriptor set cannot silently read a
stale entry.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bindra.encoders.kpgt_descriptors import N_DESCRIPTORS
from bindra.features.types import EncoderMetadata

#: 512-bit RDKit path fingerprint followed by the 200 normalized descriptors.
FP_WIDTH = 512
KNOWLEDGE_WIDTH = FP_WIDTH + N_DESCRIPTORS
SCHEMA_VERSION = "kpgt-knowledge-v1"


def knowledge_cache_key(smiles: str, policy_hash: str) -> str:
    """Content hash of the molecule and the policy that featurized it."""
    digest = hashlib.sha256(f"{policy_hash}\x00{smiles}".encode("utf-8"))
    return digest.hexdigest()[:32]


def policy_digest(policy: Any) -> str:
    """Stable short hash of the fingerprint/descriptor policy."""
    parts = sorted(f"{k}={v}" for k, v in policy.to_dict().items())
    payload = "|".join(parts) + f"|descriptors={N_DESCRIPTORS}|{SCHEMA_VERSION}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def knowledge_metadata(policy_hash: str) -> EncoderMetadata:
    """Identity recorded in the cache index, so a stale cache is detectable."""
    return EncoderMetadata(
        name="kpgt_knowledge",
        checkpoint="featurization-only",
        output_dim=KNOWLEDGE_WIDTH,
        precision="float32",
        feature_schema_version=SCHEMA_VERSION,
        revision=policy_hash,
        preprocessing_policy="rdk_fingerprint ++ RDKit2DNormalized",
        extra={"fp_width": FP_WIDTH, "descriptor_width": N_DESCRIPTORS},
    )


def pack(fingerprint: np.ndarray, descriptors: np.ndarray) -> torch.Tensor:
    """One row: fingerprint then descriptors, float32."""
    if fingerprint.shape[-1] != FP_WIDTH:
        raise ValueError(f"fingerprint must be {FP_WIDTH} wide, got {fingerprint.shape[-1]}")
    if descriptors.shape[-1] != N_DESCRIPTORS:
        raise ValueError(
            f"descriptors must be {N_DESCRIPTORS} wide, got {descriptors.shape[-1]}"
        )
    row = np.concatenate(
        (np.asarray(fingerprint, dtype=np.float32), np.asarray(descriptors, dtype=np.float32))
    )
    return torch.from_numpy(row).to(torch.float32)


def unpack(row: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a cached row back into ``(fingerprint, descriptors)``."""
    if row.shape[-1] != KNOWLEDGE_WIDTH:
        raise ValueError(f"expected {KNOWLEDGE_WIDTH} values, got {row.shape[-1]}")
    return row[..., :FP_WIDTH], row[..., FP_WIDTH:]


def open_cache(directory: Path | str, *, mode: str = "r", policy_hash: str = ""):
    """Open the knowledge cache. float32, fixed width, sharded."""
    from bindra.features.ligand_cache import LigandVectorCache

    return LigandVectorCache(
        directory,
        metadata=knowledge_metadata(policy_hash) if mode != "r" else None,
        dim=KNOWLEDGE_WIDTH,
        mode=mode,
        dtype=torch.float32,
    )


__all__ = [
    "CompactKnowledgeCache",
    "build_compact_index",
    "FP_WIDTH",
    "KNOWLEDGE_WIDTH",
    "SCHEMA_VERSION",
    "knowledge_cache_key",
    "knowledge_metadata",
    "open_cache",
    "pack",
    "policy_digest",
    "unpack",
]


# --- compact index ---------------------------------------------------------
#
# The JSON index that `LigandVectorCache` writes is 209 MB for this cache, and
# parsing it costs **1.31 GiB of Python heap and 7.9 s in every process that
# opens it**. That is tolerable for one reader and fatal for a featurize pool:
# at fourteen workers it is 18.3 GiB of duplicated dictionary on top of the
# trainer, which is what exhausted memory and killed the pool mid-validation.
#
# These sidecars hold the same mapping as two flat arrays, memmapped rather
# than parsed: the operating system keeps one copy in the page cache no matter
# how many workers open it, and opening costs nothing. 1,433,489 entries is
# 34 MB rather than 209, and lookup is a binary search instead of a dict hit.
#
# One column is then pulled into memory contiguously, 11.0 MB per process. That
# is deliberate and it is the difference between a 4.139 ms lookup and a
# 0.004 ms one -- see `CompactKnowledgeCache.__init__` for why a strided memmap
# column cannot be binary-searched without copying itself.
#
# The key is kept at its full 128 bits. Truncating to 64 would make the arrays
# smaller and a silent collision -- returning another molecule's descriptors --
# about 5e-8 likely across this corpus, which is small but not a risk worth
# taking for 11 MB.

KEYS_FILE = "compact-keys.npy"
LOCS_FILE = "compact-locs.npy"


def _key_to_pair(key: str) -> tuple[int, int]:
    """A 32-hex-character key as two uint64 halves, for sortable comparison."""
    return int(key[:16], 16), int(key[16:32], 16)


def build_compact_index(directory: Path | str) -> int:
    """Write the memmappable sidecars from the cache's JSON index. Returns N."""
    import json

    directory = Path(directory)
    payload = json.loads((directory / "index.json").read_text(encoding="utf-8"))
    entries = payload["entries"]
    items = sorted(
        (_key_to_pair(k), int(v["shard"]), int(v["row"])) for k, v in entries.items()
    )
    keys = np.empty((len(items), 2), dtype=np.uint64)
    locs = np.empty((len(items), 2), dtype=np.uint32)
    for i, (pair, shard, row) in enumerate(items):
        keys[i] = pair
        locs[i] = (shard, row)
    np.save(directory / KEYS_FILE, keys)
    np.save(directory / LOCS_FILE, locs)
    return len(items)


class CompactKnowledgeCache:
    """Read-only knowledge cache with a memmapped index and no torch in the path.

    Built for the featurize workers: opening is free, the index is shared
    through the page cache, and a lookup returns numpy arrays directly rather
    than round-tripping through torch.
    """

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)
        keys_path = self.directory / KEYS_FILE
        if not keys_path.exists():
            raise FileNotFoundError(
                f"no compact index at {keys_path}; build it with "
                "`bindra cache knowledge --compact-index`"
            )
        self.keys = np.load(keys_path, mmap_mode="r")
        self.locs = np.load(self.directory / LOCS_FILE, mmap_mode="r")
        # The high halves, contiguous and in memory. `keys` is [N, 2], so
        # `keys[:, 0]` is a STRIDED view, and `np.searchsorted` cannot work on
        # one -- it materialises a contiguous copy first. That copy is 11.0 MiB
        # and `_find` calls searchsorted twice, so every single molecule lookup
        # was allocating and filling 22 MiB: measured at **4.139 ms per lookup
        # against 0.004 ms here, a 468x difference**, and 61% of the 6.77 ms
        # that one molecule's whole featurization costs. It was also what
        # exhausted memory and killed a featurize pool under load.
        #
        # One 11.0 MiB array per process buys all of that back. It is read-only
        # and never written, so it costs exactly its size and nothing else.
        self._high = np.ascontiguousarray(self.keys[:, 0])
        self._shards: dict[int, np.memmap] = {}

    def __len__(self) -> int:
        return int(self.keys.shape[0])

    def _find(self, key: str) -> tuple[int, int] | None:
        hi, lo = _key_to_pair(key)
        # Binary search the contiguous high halves, then scan the (tiny) tie
        # block. The probe is an explicit uint64 rather than a Python int so
        # numpy compares in the array's own type instead of choosing one.
        probe = np.uint64(hi)
        left = int(np.searchsorted(self._high, probe, side="left"))
        right = int(np.searchsorted(self._high, probe, side="right"))
        for i in range(left, right):
            if int(self.keys[i, 1]) == lo:
                return int(self.locs[i, 0]), int(self.locs[i, 1])
        return None

    def _shard(self, index: int) -> np.memmap:
        if index not in self._shards:
            path = self.directory / f"vectors-{index:05d}.bin"
            rows = path.stat().st_size // (4 * KNOWLEDGE_WIDTH)
            self._shards[index] = np.memmap(
                path, dtype=np.float32, mode="r", shape=(rows, KNOWLEDGE_WIDTH)
            )
        return self._shards[index]

    def get(self, key: str) -> tuple[np.ndarray, np.ndarray] | None:
        """``(fingerprint, descriptors)`` as numpy, or None if absent."""
        found = self._find(key)
        if found is None:
            return None
        shard, row = found
        values = np.asarray(self._shard(shard)[row])
        return values[:FP_WIDTH].copy(), values[FP_WIDTH:].copy()
