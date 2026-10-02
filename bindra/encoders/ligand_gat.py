"""Legacy 78-feature GAT ligand encoder, retained as an ablation arm.

Status: scaffolded. See docs/STATUS.md.

Owed by the specification:
    Ablation 1: existing ESMC + GAT model retrained on the same curated
    data.
"""

from __future__ import annotations

_SPEC_CLAUSE = (
    "Ablation 1: existing ESMC + GAT model retrained on the same curated data."
)


def _not_implemented(what: str) -> "NotImplementedError":
    return NotImplementedError(
        f"{what} is scaffolded but not implemented.\n"
        f"Specification requirement: {_SPEC_CLAUSE}\n"
        "See docs/STATUS.md for the current stage."
    )


__all__ = ["_not_implemented"]
