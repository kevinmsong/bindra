"""Experimental result ingestion and QC.

Status: scaffolded. See docs/STATUS.md.

Owed by the specification:
    Upgrade C: keep identity, assay, binding, functional and failure records
    separate; a technical failure is not a negative result.
"""

from __future__ import annotations

_SPEC_CLAUSE = (
    "Upgrade C: keep identity, assay, binding, functional and failure records separate; a technical failure is not a negative result."
)


def _not_implemented(what: str) -> "NotImplementedError":
    return NotImplementedError(
        f"{what} is scaffolded but not implemented.\n"
        f"Specification requirement: {_SPEC_CLAUSE}\n"
        "See docs/STATUS.md for the current stage."
    )


__all__ = ["_not_implemented"]
