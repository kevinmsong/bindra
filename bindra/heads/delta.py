"""Analog delta prediction: difference baseline and antisymmetric head.

Status: scaffolded. See docs/STATUS.md.

Owed by the specification:
    Upgrade A: enforce zero change for A=A and antisymmetry under reversal;
    the dedicated head is an ablation, not the default.
"""

from __future__ import annotations

_SPEC_CLAUSE = (
    "Upgrade A: enforce zero change for A=A and antisymmetry under reversal; the dedicated head is an ablation, not the default."
)


def _not_implemented(what: str) -> "NotImplementedError":
    return NotImplementedError(
        f"{what} is scaffolded but not implemented.\n"
        f"Specification requirement: {_SPEC_CLAUSE}\n"
        "See docs/STATUS.md for the current stage."
    )


__all__ = ["_not_implemented"]
