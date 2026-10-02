"""Applicability-domain distance against training entities only.

Status: scaffolded. See docs/STATUS.md.

Owed by the specification:
    Priority 4: return an OOD flag independently of ensemble spread.
"""

from __future__ import annotations

_SPEC_CLAUSE = (
    "Priority 4: return an OOD flag independently of ensemble spread."
)


def _not_implemented(what: str) -> "NotImplementedError":
    return NotImplementedError(
        f"{what} is scaffolded but not implemented.\n"
        f"Specification requirement: {_SPEC_CLAUSE}\n"
        "See docs/STATUS.md for the current stage."
    )


__all__ = ["_not_implemented"]
