"""Temporal holdout panel.

Status: scaffolded. See docs/STATUS.md.

Owed by the specification:
    Upgrade B: a recent database import of an old measurement is not new
    evidence; exclude unknown-date records from temporal claims.
"""

from __future__ import annotations

_SPEC_CLAUSE = (
    "Upgrade B: a recent database import of an old measurement is not new evidence; exclude unknown-date records from temporal claims."
)


def _not_implemented(what: str) -> "NotImplementedError":
    return NotImplementedError(
        f"{what} is scaffolded but not implemented.\n"
        f"Specification requirement: {_SPEC_CLAUSE}\n"
        "See docs/STATUS.md for the current stage."
    )


__all__ = ["_not_implemented"]
