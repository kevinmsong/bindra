"""KPGT molecular encoder adapter via the isolated feature environment.

Status: scaffolded. See docs/STATUS.md.

Owed by the specification:
    Priority 2: use the author's preprocessing and pretrained pathway;
    expose token states AND the graph readout; do not assume one token
    equals one atom.
"""

from __future__ import annotations

_SPEC_CLAUSE = (
    "Priority 2: use the author's preprocessing and pretrained pathway; expose token states AND the graph readout; do not assume one token equals one atom."
)


def _not_implemented(what: str) -> "NotImplementedError":
    return NotImplementedError(
        f"{what} is scaffolded but not implemented.\n"
        f"Specification requirement: {_SPEC_CLAUSE}\n"
        "See docs/STATUS.md for the current stage."
    )


__all__ = ["_not_implemented"]
