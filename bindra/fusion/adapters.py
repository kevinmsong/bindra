"""Shared trainable projections over frozen encoder features.

Status: scaffolded. See docs/STATUS.md.

Owed by the specification:
    Priority 2: the adapters must actually be shared by affinity, retrieval
    and diffusion conditioning; log which parameters receive gradients from
    each loss.
"""

from __future__ import annotations

_SPEC_CLAUSE = (
    "Priority 2: the adapters must actually be shared by affinity, retrieval and diffusion conditioning; log which parameters receive gradients from each loss."
)


def _not_implemented(what: str) -> "NotImplementedError":
    return NotImplementedError(
        f"{what} is scaffolded but not implemented.\n"
        f"Specification requirement: {_SPEC_CLAUSE}\n"
        "See docs/STATUS.md for the current stage."
    )


__all__ = ["_not_implemented"]
