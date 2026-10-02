"""Conditional denoiser over noisy graph tensors.

Status: scaffolded. See docs/STATUS.md.

Owed by the specification:
    Priority 3: target and seed conditioning enter via cross-attention; the
    clean product representation must never be a condition.
"""

from __future__ import annotations

_SPEC_CLAUSE = (
    "Priority 3: target and seed conditioning enter via cross-attention; the clean product representation must never be a condition."
)


def _not_implemented(what: str) -> "NotImplementedError":
    return NotImplementedError(
        f"{what} is scaffolded but not implemented.\n"
        f"Specification requirement: {_SPEC_CLAUSE}\n"
        "See docs/STATUS.md for the current stage."
    )


__all__ = ["_not_implemented"]
