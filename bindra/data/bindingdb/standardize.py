"""Deterministic RDKit standardization with atom-index mapping.

Status: scaffolded. See docs/STATUS.md.

Owed by the specification:
    Priority 1: preserve stereochemistry and formal charges; keep a mapping
    between original and standardized atom indices; quarantine ambiguous
    mixtures rather than coercing them.
"""

from __future__ import annotations

_SPEC_CLAUSE = (
    "Priority 1: preserve stereochemistry and formal charges; keep a mapping between original and standardized atom indices; quarantine ambiguous mixtures rather than coercing them."
)


def _not_implemented(what: str) -> "NotImplementedError":
    return NotImplementedError(
        f"{what} is scaffolded but not implemented.\n"
        f"Specification requirement: {_SPEC_CLAUSE}\n"
        "See docs/STATUS.md for the current stage."
    )


__all__ = ["_not_implemented"]
