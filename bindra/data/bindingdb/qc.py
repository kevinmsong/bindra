"""Deduplication, replicate aggregation, assay-comparability grouping.

Status: scaffolded. See docs/STATUS.md.

Owed by the specification:
    Priority 1: remove duplicate imports before splitting; a missing assay
    identifier must not collapse a target into one ranking group.
"""

from __future__ import annotations

_SPEC_CLAUSE = (
    "Priority 1: remove duplicate imports before splitting; a missing assay identifier must not collapse a target into one ranking group."
)


def _not_implemented(what: str) -> "NotImplementedError":
    return NotImplementedError(
        f"{what} is scaffolded but not implemented.\n"
        f"Specification requirement: {_SPEC_CLAUSE}\n"
        "See docs/STATUS.md for the current stage."
    )


__all__ = ["_not_implemented"]
