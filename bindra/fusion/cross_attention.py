"""Bidirectional protein-ligand cross-attention with retained weights.

Status: scaffolded. See docs/STATUS.md.

Owed by the specification:
    Priority 2: protein-to-ligand and ligand-to-protein are not
    interchangeable transposes; keep both, with documented axes/masks.
"""

from __future__ import annotations

_SPEC_CLAUSE = (
    "Priority 2: protein-to-ligand and ligand-to-protein are not interchangeable transposes; keep both, with documented axes/masks."
)


def _not_implemented(what: str) -> "NotImplementedError":
    return NotImplementedError(
        f"{what} is scaffolded but not implemented.\n"
        f"Specification requirement: {_SPEC_CLAUSE}\n"
        "See docs/STATUS.md for the current stage."
    )


__all__ = ["_not_implemented"]
