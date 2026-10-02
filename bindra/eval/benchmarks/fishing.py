"""Locked drug-fishing and analog-fishing challenge.

Status: scaffolded. See docs/STATUS.md.

Owed by the specification:
    Priority 5: computational decoys are presumed controls with unknown
    binding status and must never acquire fabricated affinity labels.
"""

from __future__ import annotations

_SPEC_CLAUSE = (
    "Priority 5: computational decoys are presumed controls with unknown binding status and must never acquire fabricated affinity labels."
)


def _not_implemented(what: str) -> "NotImplementedError":
    return NotImplementedError(
        f"{what} is scaffolded but not implemented.\n"
        f"Specification requirement: {_SPEC_CLAUSE}\n"
        "See docs/STATUS.md for the current stage."
    )


__all__ = ["_not_implemented"]
