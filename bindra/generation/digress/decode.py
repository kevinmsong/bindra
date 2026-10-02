"""Graph tensors to a sanitized RDKit molecule and canonical SMILES.

Status: scaffolded. See docs/STATUS.md.

Owed by the specification:
    Priority 3: report validity before and after repair; identify newly
    created unspecified stereochemistry instead of inventing it.
"""

from __future__ import annotations

_SPEC_CLAUSE = (
    "Priority 3: report validity before and after repair; identify newly created unspecified stereochemistry instead of inventing it."
)


def _not_implemented(what: str) -> "NotImplementedError":
    return NotImplementedError(
        f"{what} is scaffolded but not implemented.\n"
        f"Specification requirement: {_SPEC_CLAUSE}\n"
        "See docs/STATUS.md for the current stage."
    )


__all__ = ["_not_implemented"]
