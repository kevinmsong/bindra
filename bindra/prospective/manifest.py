"""Blinded candidate manifests with a separate assignment key.

A prospective study is only prospective if the commitments were made before the
results existed. This freezes them and hashes them: **model version, candidate
universe, budget, ranking rule and selection date**, written once and sealed.
A manifest that can be edited after a result arrives is a retrospective study
with a confident name.

**The assignment key lives in a separate file.** The manifest names blinded
candidate codes; the key maps codes to molecules. Whoever runs the assay reads
the manifest and never the key, so the ranking cannot influence how carefully a
compound is handled. The key's own hash is recorded in the manifest, so the two
can be checked against each other afterwards without either having been
readable from the other beforehand.

Owed by the specification:
    Upgrade C: freeze model version, candidate universe, budget, ranking
    rule and selection date before inspecting new measurements.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path


class ManifestError(RuntimeError):
    """A prospective commitment was violated."""


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class ProspectiveManifest:
    """The commitments, frozen. Every field is part of the hash."""

    round_id: str
    #: Which checkpoint. A different checkpoint is a different study.
    model_checkpoint: str
    model_provenance_digest: str
    #: How many candidates may be carried forward. Fixed in advance so the
    #: cut cannot be widened once the first results look promising.
    budget: int
    #: The rule, in words, before it was run.
    ranking_rule: str
    #: ISO date. Selection happened before any new measurement was seen.
    selection_date: str
    #: Blinded codes only. Never structures.
    candidate_codes: tuple[str, ...]
    #: Hash of the universe the candidates were drawn from, so nobody can
    #: later claim a wider or narrower library.
    candidate_universe_digest: str
    #: Hash of the separate key file, never its contents.
    assignment_key_digest: str
    notes: str = ""
    schema_version: str = "prospective-manifest-v1"

    def __post_init__(self) -> None:
        if len(self.candidate_codes) > self.budget:
            raise ManifestError(
                f"{len(self.candidate_codes)} candidates exceeds the frozen "
                f"budget of {self.budget}"
            )
        if len(set(self.candidate_codes)) != len(self.candidate_codes):
            raise ManifestError("candidate codes must be distinct")

    def digest(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict:
        return {
            "round_id": self.round_id,
            "model_checkpoint": self.model_checkpoint,
            "model_provenance_digest": self.model_provenance_digest,
            "budget": self.budget,
            "ranking_rule": self.ranking_rule,
            "selection_date": self.selection_date,
            "candidate_codes": list(self.candidate_codes),
            "candidate_universe_digest": self.candidate_universe_digest,
            "assignment_key_digest": self.assignment_key_digest,
            "notes": self.notes,
            "schema_version": self.schema_version,
        }


def freeze(
    manifest: ProspectiveManifest,
    directory: "Path | str",
    assignment: "dict[str, str]",
) -> dict:
    """Write the manifest and the key as two files, and refuse to overwrite.

    Overwriting is the whole failure this guards: a manifest rewritten after a
    result arrives is indistinguishable from one written before it, unless
    writing twice is impossible.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / f"{manifest.round_id}.manifest.json"
    key_path = directory / f"{manifest.round_id}.key.json"
    for path in (manifest_path, key_path):
        if path.exists():
            raise ManifestError(
                f"{path.name} already exists. A prospective commitment is "
                "written once; rewriting it makes the study retrospective."
            )
    if set(assignment) != set(manifest.candidate_codes):
        raise ManifestError(
            "the assignment key does not cover exactly the manifest's codes"
        )
    if _digest(assignment) != manifest.assignment_key_digest:
        raise ManifestError(
            "assignment_key_digest does not match the key being written; the "
            "manifest would not be checkable against it afterwards"
        )
    payload = manifest.to_dict() | {"manifest_digest": manifest.digest()}
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    key_path.write_text(json.dumps(assignment, indent=2, sort_keys=True), encoding="utf-8")
    return {"manifest": str(manifest_path), "key": str(key_path), "digest": manifest.digest()}


def load(path: "Path | str", *, verify: bool = True) -> ProspectiveManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    recorded = payload.pop("manifest_digest", None)
    manifest = ProspectiveManifest(
        round_id=payload["round_id"],
        model_checkpoint=payload["model_checkpoint"],
        model_provenance_digest=payload["model_provenance_digest"],
        budget=int(payload["budget"]),
        ranking_rule=payload["ranking_rule"],
        selection_date=payload["selection_date"],
        candidate_codes=tuple(payload["candidate_codes"]),
        candidate_universe_digest=payload["candidate_universe_digest"],
        assignment_key_digest=payload["assignment_key_digest"],
        notes=payload.get("notes", ""),
        schema_version=payload.get("schema_version", "prospective-manifest-v1"),
    )
    if verify and recorded and recorded != manifest.digest():
        raise ManifestError(
            f"manifest digest mismatch: recorded {recorded[:16]}..., recomputed "
            f"{manifest.digest()[:16]}.... The file was edited after it was frozen."
        )
    return manifest


def universe_digest(identifiers: "list[str]") -> str:
    """Hash of the candidate universe, order-independent."""
    return _digest(sorted(set(identifiers)))


__all__ = [
    "ManifestError",
    "ProspectiveManifest",
    "freeze",
    "load",
    "universe_digest",
]
