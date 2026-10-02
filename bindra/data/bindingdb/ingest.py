"""Chunked, versioned BindingDB TSV ingestion with provenance.

Status: scaffolded. See docs/STATUS.md.

Owed by the specification:
    Priority 1: save release/date, source URL, checksum, parser version and
    filtering configuration; process large files in chunks.
"""

from __future__ import annotations

_SPEC_CLAUSE = (
    "Priority 1: save release/date, source URL, checksum, parser version and filtering configuration; process large files in chunks."
)


def _not_implemented(what: str) -> "NotImplementedError":
    return NotImplementedError(
        f"{what} is scaffolded but not implemented.\n"
        f"Specification requirement: {_SPEC_CLAUSE}\n"
        "See docs/STATUS.md for the current stage."
    )


__all__ = ["_not_implemented"]
