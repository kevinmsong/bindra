"""Target / countertarget selectivity panel.

Status: scaffolded. See docs/STATUS.md.

Owed by the specification:
    Upgrade B: missing countertarget measurements are unknown, not negative.
"""

from __future__ import annotations

_SPEC_CLAUSE = (
    "Upgrade B: missing countertarget measurements are unknown, not negative."
)


def _not_implemented(what: str) -> "NotImplementedError":
    return NotImplementedError(
        f"{what} is scaffolded but not implemented.\n"
        f"Specification requirement: {_SPEC_CLAUSE}\n"
        "See docs/STATUS.md for the current stage."
    )


__all__ = ["_not_implemented"]
