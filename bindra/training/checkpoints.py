"""Checkpoint metadata with code/config/data/split/encoder hashes.

Status: scaffolded. See docs/STATUS.md.

Owed by the specification:
    Deliverables: checkpoint metadata containing all provenance hashes.
"""

from __future__ import annotations

_SPEC_CLAUSE = (
    "Deliverables: checkpoint metadata containing all provenance hashes."
)


def _not_implemented(what: str) -> "NotImplementedError":
    return NotImplementedError(
        f"{what} is scaffolded but not implemented.\n"
        f"Specification requirement: {_SPEC_CLAUSE}\n"
        "See docs/STATUS.md for the current stage."
    )


__all__ = ["_not_implemented"]
