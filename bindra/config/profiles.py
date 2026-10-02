"""Named experiment profiles and config hashing.

Status: scaffolded. See docs/STATUS.md.

Owed by the specification:
    Initial settings to expose in configuration; checkpoint metadata
    containing source code/config/data/split/encoder hashes.
"""

from __future__ import annotations

_SPEC_CLAUSE = (
    "Initial settings to expose in configuration; checkpoint metadata containing source code/config/data/split/encoder hashes."
)


def _not_implemented(what: str) -> "NotImplementedError":
    return NotImplementedError(
        f"{what} is scaffolded but not implemented.\n"
        f"Specification requirement: {_SPEC_CLAUSE}\n"
        "See docs/STATUS.md for the current stage."
    )


__all__ = ["_not_implemented"]
