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
