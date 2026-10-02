"""Independent protein and ligand retrieval projections.

Status: scaffolded. See docs/STATUS.md.

Owed by the specification:
    Priority 2: computable without knowing the other member of a pair; do
    not index pair-conditioned cross-attention embeddings.
"""

from __future__ import annotations

_SPEC_CLAUSE = (
    "Priority 2: computable without knowing the other member of a pair; do not index pair-conditioned cross-attention embeddings."
)


def _not_implemented(what: str) -> "NotImplementedError":
    return NotImplementedError(
        f"{what} is scaffolded but not implemented.\n"
        f"Specification requirement: {_SPEC_CLAUSE}\n"
        "See docs/STATUS.md for the current stage."
    )


__all__ = ["_not_implemented"]
