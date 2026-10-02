"""Huber for exact labels plus one-sided penalties for bounds.

Status: scaffolded. See docs/STATUS.md.

Owed by the specification:
    Priority 1: do not train against a bound as if it were exact; report
    exact-label metrics separately from bound violations.
"""

from __future__ import annotations

_SPEC_CLAUSE = (
    "Priority 1: do not train against a bound as if it were exact; report exact-label metrics separately from bound violations."
)


def _not_implemented(what: str) -> "NotImplementedError":
    return NotImplementedError(
        f"{what} is scaffolded but not implemented.\n"
        f"Specification requirement: {_SPEC_CLAUSE}\n"
        "See docs/STATUS.md for the current stage."
    )


__all__ = ["_not_implemented"]
