"""Categorical forward noising over atom and bond categories.

Status: scaffolded. See docs/STATUS.md.

Owed by the specification:
    Priority 3: a real discrete process, not a Gaussian scheduler applied to
    categorical graphs.
"""

from __future__ import annotations

_SPEC_CLAUSE = (
    "Priority 3: a real discrete process, not a Gaussian scheduler applied to categorical graphs."
)


def _not_implemented(what: str) -> "NotImplementedError":
    return NotImplementedError(
        f"{what} is scaffolded but not implemented.\n"
        f"Specification requirement: {_SPEC_CLAUSE}\n"
        "See docs/STATUS.md for the current stage."
    )


__all__ = ["_not_implemented"]
