"""Locked activity-cliff panel.

Status: scaffolded. See docs/STATUS.md.

Owed by the specification:
    Upgrade B: hold out both members and their chemical series; assign
    diagnostic categories only after predictions are frozen.
"""

from __future__ import annotations

_SPEC_CLAUSE = (
    "Upgrade B: hold out both members and their chemical series; assign diagnostic categories only after predictions are frozen."
)


def _not_implemented(what: str) -> "NotImplementedError":
    return NotImplementedError(
        f"{what} is scaffolded but not implemented.\n"
        f"Specification requirement: {_SPEC_CLAUSE}\n"
        "See docs/STATUS.md for the current stage."
    )


__all__ = ["_not_implemented"]
