"""Per-round prospective reports.

Status: scaffolded. See docs/STATUS.md.

Owed by the specification:
    Upgrade C: separate planned, selected, obtained, assayed and confirmed
    status. Never present a retrospective refit as the original prospective
    prediction.
"""

from __future__ import annotations

_SPEC_CLAUSE = (
    "Upgrade C: separate planned, selected, obtained, assayed and confirmed status. Never present a retrospective refit as the original prospective prediction."
)


def _not_implemented(what: str) -> "NotImplementedError":
    return NotImplementedError(
        f"{what} is scaffolded but not implemented.\n"
        f"Specification requirement: {_SPEC_CLAUSE}\n"
        "See docs/STATUS.md for the current stage."
    )


__all__ = ["_not_implemented"]
