"""Target fishing and seed-assisted analog fishing APIs.

Status: scaffolded. See docs/STATUS.md.

Owed by the specification:
    Priority 5: report target-only and seed-assisted fishing separately;
    seed-assisted evaluation is not zero-shot discovery.
"""

from __future__ import annotations

_SPEC_CLAUSE = (
    "Priority 5: report target-only and seed-assisted fishing separately; seed-assisted evaluation is not zero-shot discovery."
)


def _not_implemented(what: str) -> "NotImplementedError":
    return NotImplementedError(
        f"{what} is scaffolded but not implemented.\n"
        f"Specification requirement: {_SPEC_CLAUSE}\n"
        "See docs/STATUS.md for the current stage."
    )


__all__ = ["_not_implemented"]
