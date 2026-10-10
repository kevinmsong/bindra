"""Checkpoint metadata with code/config/data/split/encoder hashes.

A checkpoint whose provenance is incomplete cannot be reproduced, and the gap is
discovered months later when the number it produced needs defending. This
collects every hash that identifies what a run saw, and
:func:`assert_complete` refuses a record that is missing one rather than letting
it be written and found wanting later.

``Trainer.state_dict`` already assembles most of this. The value added here is
the **completeness check** and the **comparison** between two records, which is
what answers "why does this rerun not match".

Owed by the specification:
    Deliverables: checkpoint metadata containing all provenance hashes.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


class ProvenanceError(ValueError):
    """A provenance record is incomplete or inconsistent."""


#: Every field a checkpoint must carry. Named here so the requirement lives in
#: one place rather than in whichever writer happened to be updated last.
REQUIRED_FIELDS: tuple[str, ...] = (
    "snapshot_release",
    "curation_profile_hash",
    "observations_combined_sha256",
    "observations_rows",
    "split_regime",
    "split_id",
    "split_manifest_sha256",
    "split_seed",
    "protein_encoder",
    "ligand_encoder",
    "model",
)


def git_revision(root: "Path | str" = ".") -> dict:
    """The code that produced the checkpoint, dirty flag included.

    A clean hash on a dirty tree is a lie that is only detected when someone
    checks out the hash and gets different numbers.
    """
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(root),
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=str(root),
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001 - absent git must not stop training
        return {"revision": None, "dirty": None, "error": f"{type(exc).__name__}: {exc}"}
    return {"revision": revision, "dirty": bool(status), "n_dirty_files": len(status.splitlines())}


def config_hash(payload: dict) -> str:
    """Stable hash of a config, independent of key order."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


@dataclass
class CheckpointProvenance:
    """Everything needed to say what a checkpoint saw."""

    fields: dict = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict) -> "CheckpointProvenance":
        return cls(fields=dict(payload.get("provenance", payload)))

    def assert_complete(self) -> None:
        missing = [name for name in REQUIRED_FIELDS if not self.fields.get(name)]
        if missing:
            raise ProvenanceError(
                f"provenance is missing {missing}. A checkpoint without these "
                "cannot be reproduced, and the gap is only noticed when the "
                "number it produced has to be defended."
            )

    def digest(self) -> str:
        return config_hash({k: self.fields.get(k) for k in REQUIRED_FIELDS})

    def to_dict(self) -> dict:
        return dict(self.fields)


def compare(left: CheckpointProvenance, right: CheckpointProvenance) -> dict:
    """What differs between two runs. The answer to 'why do these disagree'."""
    keys = sorted(set(left.fields) | set(right.fields))
    differences = {
        key: {"left": left.fields.get(key), "right": right.fields.get(key)}
        for key in keys
        if left.fields.get(key) != right.fields.get(key)
    }
    return {
        "identical": not differences,
        "n_differences": len(differences),
        "differences": differences,
        "comparable": not any(
            key in differences
            for key in ("split_manifest_sha256", "observations_combined_sha256", "split_id")
        ),
        "note": (
            "comparable=False means these runs saw different data; their metrics "
            "are not a head-to-head comparison however similar the models are."
        ),
    }


def read(path: "Path | str") -> CheckpointProvenance:
    """Provenance from a checkpoint on disk, without loading the weights."""
    import torch

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return CheckpointProvenance.from_payload(payload)


__all__ = [
    "REQUIRED_FIELDS",
    "CheckpointProvenance",
    "ProvenanceError",
    "compare",
    "config_hash",
    "git_revision",
    "read",
]
