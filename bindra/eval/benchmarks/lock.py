"""Locked benchmark manifests, with test labels sealed behind an audited gate.

The specification requires that test labels be used *"only to assign the
prespecified diagnostic category and calculate metrics after predictions are
frozen"*. That is easy to agree to and easy to violate by accident, so it is
enforced mechanically here rather than by convention:

* A panel's labels live under ``sealed/`` inside the manifest directory.
* :func:`load_sealed_labels` refuses to open them without an explicit
  ``allow_labels=True`` **and** a reason string, and appends every access to
  ``sealed/access_log.jsonl``.
* A test walks the training, scaler-fitting and calibration code paths with
  ``Path.open`` monkeypatched and asserts that no ``sealed/`` path is ever
  touched.

Panels are also **locked**: :func:`lock` refuses to overwrite an existing
manifest, and :func:`load_locked` re-verifies every recorded hash, so a panel
cannot be quietly widened after results are known.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from bindra.config.profiles import canonical_json
from bindra.data.tables import TableRef, write_json, write_table

BENCHMARK_SCHEMA_VERSION = "benchmark-v1"

SEALED_DIRNAME = "sealed"
ACCESS_LOG = "access_log.jsonl"


class BenchmarkError(RuntimeError):
    """A panel could not be locked, loaded, or trusted."""


class SealedLabelError(BenchmarkError):
    """Test labels were requested without an explicit, logged authorisation."""


@dataclass
class BenchmarkManifest:
    """One locked evaluation panel.

    ``queries`` holds only identifiers -- what a model is allowed to see.
    ``labels`` holds the values and is sealed. ``unknowns`` holds pairs whose
    outcome is genuinely unknown and which must never be scored as negatives.
    """

    panel: str
    definition: Mapping[str, Any]
    inputs: Mapping[str, Any]
    counts: Mapping[str, int] = field(default_factory=dict)
    attrition: Mapping[str, int] = field(default_factory=dict)
    queries: Mapping[str, Any] | None = None
    labels: Mapping[str, Any] | None = None
    unknowns: Mapping[str, Any] | None = None
    controls: Mapping[str, Any] | None = None
    notes: tuple[str, ...] = ()
    created_at: str = ""
    schema_version: str = BENCHMARK_SCHEMA_VERSION
    definition_sha256: str = ""
    manifest_sha256: str = ""

    @property
    def contains_test_labels(self) -> bool:
        return self.labels is not None

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "panel": self.panel,
            "created_at": self.created_at,
            "definition": dict(self.definition),
            "definition_sha256": self.definition_sha256,
            "inputs": dict(self.inputs),
            "counts": dict(self.counts),
            "attrition": dict(self.attrition),
            "queries": dict(self.queries) if self.queries else None,
            "labels": dict(self.labels) if self.labels else None,
            "unknowns": dict(self.unknowns) if self.unknowns else None,
            "controls": dict(self.controls) if self.controls else None,
            "notes": list(self.notes),
            "contains_test_labels": self.contains_test_labels,
        }

    def compute_hash(self) -> str:
        return hashlib.sha256(canonical_json(self.payload()).encode("utf-8")).hexdigest()


def definition_hash(definition: Any) -> str:
    payload = definition if isinstance(definition, Mapping) else _as_dict(definition)
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:24]


def _as_dict(obj: Any) -> dict[str, Any]:
    from dataclasses import asdict, is_dataclass

    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise TypeError(f"cannot serialize {type(obj).__name__} as a definition")


def lock(
    panel: str,
    definition: Any,
    inputs: Mapping[str, Any],
    directory: Path | str,
    *,
    queries: pd.DataFrame,
    labels: pd.DataFrame | None = None,
    unknowns: pd.DataFrame | None = None,
    controls: pd.DataFrame | None = None,
    attrition: Mapping[str, int] | None = None,
    notes: Iterable[str] = (),
    overwrite: bool = False,
) -> BenchmarkManifest:
    """Write a panel and freeze it.

    Refuses to overwrite: a panel that can be rewritten after results are known
    is not a locked panel.
    """
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise BenchmarkError(
            f"{manifest_path} already exists. Benchmark panels are locked; write to "
            "a new directory, or pass overwrite=True deliberately and say why in the "
            "notes."
        )
    directory.mkdir(parents=True, exist_ok=True)
    sealed = directory / SEALED_DIRNAME
    sealed.mkdir(parents=True, exist_ok=True)

    definition_payload = definition if isinstance(definition, Mapping) else _as_dict(definition)

    query_ref = write_table(queries, directory / "queries")
    label_ref = write_table(labels, sealed / "labels") if labels is not None else None
    unknown_ref = write_table(unknowns, directory / "unknowns") if unknowns is not None else None
    control_ref = write_table(controls, directory / "controls") if controls is not None else None

    manifest = BenchmarkManifest(
        panel=panel,
        definition=definition_payload,
        definition_sha256=definition_hash(definition_payload),
        inputs=dict(inputs),
        counts={
            "queries": int(len(queries)),
            "labels": int(len(labels)) if labels is not None else 0,
            "unknowns": int(len(unknowns)) if unknowns is not None else 0,
            "controls": int(len(controls)) if controls is not None else 0,
        },
        attrition=dict(attrition or {}),
        queries=query_ref.to_dict(),
        labels=label_ref.to_dict() if label_ref else None,
        unknowns=unknown_ref.to_dict() if unknown_ref else None,
        controls=control_ref.to_dict() if control_ref else None,
        notes=tuple(notes),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    manifest.manifest_sha256 = manifest.compute_hash()

    payload = manifest.payload()
    payload["manifest_sha256"] = manifest.manifest_sha256
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    if label_ref is not None:
        (sealed / "README.md").write_text(
            "# Sealed labels\n\n"
            "These are evaluation labels. They exist so metrics can be computed "
            "**after predictions are frozen**, and for nothing else.\n\n"
            "Do not read them from training, scaler-fitting, hyperparameter "
            "selection or calibration code. Access goes through "
            "`bindra.eval.benchmarks.lock.load_sealed_labels`, which requires an "
            "explicit authorisation and a reason, and appends every access to "
            "`access_log.jsonl`.\n",
            encoding="utf-8",
        )
    return manifest


def load_locked(directory: Path | str) -> BenchmarkManifest:
    """Load a panel and re-verify its hashes. Never returns labels."""
    directory = Path(directory)
    payload = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    recorded = payload.pop("manifest_sha256", None)
    payload.pop("contains_test_labels", None)

    manifest = BenchmarkManifest(
        panel=payload["panel"],
        definition=payload["definition"],
        inputs=payload["inputs"],
        counts=payload.get("counts", {}),
        attrition=payload.get("attrition", {}),
        queries=payload.get("queries"),
        labels=payload.get("labels"),
        unknowns=payload.get("unknowns"),
        controls=payload.get("controls"),
        notes=tuple(payload.get("notes", [])),
        created_at=payload.get("created_at", ""),
        schema_version=payload["schema_version"],
        definition_sha256=payload.get("definition_sha256", ""),
    )
    if recorded is not None and manifest.compute_hash() != recorded:
        raise BenchmarkError(
            f"{directory / 'manifest.json'} has been edited: its recorded hash does "
            "not match its content. A locked panel cannot be widened after the fact."
        )
    manifest.manifest_sha256 = recorded or manifest.compute_hash()

    for name in ("queries", "unknowns", "controls"):
        reference = getattr(manifest, name)
        if reference:
            TableRef.from_dict(reference).verify()
    return manifest


def load_queries(directory: Path | str) -> pd.DataFrame:
    """The identifiers a model may see. Safe to call from anywhere."""
    from bindra.data.tables import read_table

    manifest = load_locked(directory)
    if not manifest.queries:
        raise BenchmarkError("panel has no queries table")
    return read_table(Path(manifest.queries["path"]))


def load_unknowns(directory: Path | str) -> pd.DataFrame:
    """Pairs whose outcome is unknown.

    Deliberately a separate loader from the labels, and deliberately *not*
    sealed: the whole point is that these carry no label. The specification is
    explicit that a missing countertarget measurement is unknown, never a
    negative, and that computational decoys are presumed controls with unknown
    binding status.
    """
    from bindra.data.tables import read_table

    manifest = load_locked(directory)
    if not manifest.unknowns:
        return pd.DataFrame()
    return read_table(Path(manifest.unknowns["path"]))


def load_sealed_labels(
    directory: Path | str,
    *,
    allow_labels: bool = False,
    reason: str = "",
) -> pd.DataFrame:
    """Open a panel's sealed labels, loudly and on the record.

    Both arguments are required by design. ``allow_labels`` cannot be satisfied
    by a default, and ``reason`` makes the access self-documenting in the log, so
    an inappropriate read is visible in version control rather than invisible in
    a stack trace.
    """
    directory = Path(directory)
    if not allow_labels:
        raise SealedLabelError(
            f"{directory} holds sealed evaluation labels. Reading them requires "
            "allow_labels=True and a reason. They exist to compute metrics after "
            "predictions are frozen -- not for training, scaler fitting, "
            "hyperparameter selection or calibration."
        )
    if not reason.strip():
        raise SealedLabelError(
            "a reason is required when opening sealed labels, so the access log "
            "says why"
        )

    manifest = load_locked(directory)
    if not manifest.labels:
        raise BenchmarkError(f"{directory} has no sealed labels")

    reference = TableRef.from_dict(manifest.labels)
    reference.verify()

    log = directory / SEALED_DIRNAME / ACCESS_LOG
    entry = {
        "at": datetime.now(timezone.utc).isoformat(),
        "panel": manifest.panel,
        "reason": reason.strip(),
        "labels_sha256": reference.sha256,
    }
    with open(log, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")

    from bindra.data.tables import read_table

    return read_table(reference.path)


def sealed_paths(directory: Path | str) -> list[Path]:
    """Every sealed artifact under a panel, for leak tests to assert against."""
    sealed = Path(directory) / SEALED_DIRNAME
    return sorted(p for p in sealed.rglob("*") if p.is_file()) if sealed.exists() else []


__all__ = [
    "ACCESS_LOG",
    "BENCHMARK_SCHEMA_VERSION",
    "SEALED_DIRNAME",
    "BenchmarkError",
    "BenchmarkManifest",
    "SealedLabelError",
    "definition_hash",
    "load_locked",
    "load_queries",
    "load_sealed_labels",
    "load_unknowns",
    "lock",
    "sealed_paths",
]
