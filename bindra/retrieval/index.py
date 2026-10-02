"""Versioned approximate-nearest-neighbour index over ligand vectors.

Status: scaffolded. See docs/STATUS.md.

Owed by the specification:
    Priority 2: measure ANN recall against exact search; report losses from
    both approximation and shortlisting.
"""

from __future__ import annotations

_SPEC_CLAUSE = (
    "Priority 2: measure ANN recall against exact search; report losses from both approximation and shortlisting."
)


def _not_implemented(what: str) -> "NotImplementedError":
    return NotImplementedError(
        f"{what} is scaffolded but not implemented.\n"
        f"Specification requirement: {_SPEC_CLAUSE}\n"
        "See docs/STATUS.md for the current stage."
    )


__all__ = ["_not_implemented"]
