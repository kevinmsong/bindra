"""Training loop: EMA, AMP, ensembling, resumable checkpoints.

Status: scaffolded. See docs/STATUS.md.

Owed by the specification:
    Port of the legacy engine, single-GPU only, with composed losses.
"""

from __future__ import annotations

_SPEC_CLAUSE = (
    "Port of the legacy engine, single-GPU only, with composed losses."
)


def _not_implemented(what: str) -> "NotImplementedError":
    return NotImplementedError(
        f"{what} is scaffolded but not implemented.\n"
        f"Specification requirement: {_SPEC_CLAUSE}\n"
        "See docs/STATUS.md for the current stage."
    )


__all__ = ["_not_implemented"]
