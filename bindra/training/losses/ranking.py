"""Within-assay pairwise ranking.

Status: scaffolded. See docs/STATUS.md.

Owed by the specification:
    Priority 2: compare only within a compatible target/endpoint/assay
    group, with a configurable p-unit margin.
"""

from __future__ import annotations

_SPEC_CLAUSE = (
    "Priority 2: compare only within a compatible target/endpoint/assay group, with a configurable p-unit margin."
)


def _not_implemented(what: str) -> "NotImplementedError":
    return NotImplementedError(
        f"{what} is scaffolded but not implemented.\n"
        f"Specification requirement: {_SPEC_CLAUSE}\n"
        "See docs/STATUS.md for the current stage."
    )


__all__ = ["_not_implemented"]
