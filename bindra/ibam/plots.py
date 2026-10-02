"""Colorblind-safe heatmaps, vector export, zoomed residue windows.

Status: scaffolded. See docs/STATUS.md.

Owed by the specification:
    Priority 2: do not draw knowledge/virtual nodes as physical atoms;
    disclose any visualization renormalization.
"""

from __future__ import annotations

_SPEC_CLAUSE = (
    "Priority 2: do not draw knowledge/virtual nodes as physical atoms; disclose any visualization renormalization."
)


def _not_implemented(what: str) -> "NotImplementedError":
    return NotImplementedError(
        f"{what} is scaffolded but not implemented.\n"
        f"Specification requirement: {_SPEC_CLAUSE}\n"
        "See docs/STATUS.md for the current stage."
    )


__all__ = ["_not_implemented"]
