"""Sharded, memory-mapped feature cache with staleness rejection.

Status: scaffolded. See docs/STATUS.md.

Owed by the specification:
    Priority 2: cache keys must include standardized input, encoder
    checkpoint hash, feature schema, precision and preprocessing policy;
    cache base features, not trainable projections.
"""

from __future__ import annotations

_SPEC_CLAUSE = (
    "Priority 2: cache keys must include standardized input, encoder checkpoint hash, feature schema, precision and preprocessing policy; cache base features, not trainable projections."
)


def _not_implemented(what: str) -> "NotImplementedError":
    return NotImplementedError(
        f"{what} is scaffolded but not implemented.\n"
        f"Specification requirement: {_SPEC_CLAUSE}\n"
        "See docs/STATUS.md for the current stage."
    )


__all__ = ["_not_implemented"]
