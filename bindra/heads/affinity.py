"""Separate pKd / pKi / pIC50 heads with endpoint masks.

Status: scaffolded. See docs/STATUS.md.

Owed by the specification:
    Priority 2: an unmeasured endpoint must never receive a fabricated zero
    label.
"""

from __future__ import annotations

_SPEC_CLAUSE = (
    "Priority 2: an unmeasured endpoint must never receive a fabricated zero label."
)


def _not_implemented(what: str) -> "NotImplementedError":
    return NotImplementedError(
        f"{what} is scaffolded but not implemented.\n"
        f"Specification requirement: {_SPEC_CLAUSE}\n"
        "See docs/STATUS.md for the current stage."
    )


__all__ = ["_not_implemented"]
