"""Independently seeded ensemble inference.

Status: scaffolded. See docs/STATUS.md.

Owed by the specification:
    Priority 4: three independently trained models; dropout passes are not
    an ensemble.
"""

from __future__ import annotations

_SPEC_CLAUSE = (
    "Priority 4: three independently trained models; dropout passes are not an ensemble."
)


def _not_implemented(what: str) -> "NotImplementedError":
    return NotImplementedError(
        f"{what} is scaffolded but not implemented.\n"
        f"Specification requirement: {_SPEC_CLAUSE}\n"
        "See docs/STATUS.md for the current stage."
    )


__all__ = ["_not_implemented"]
