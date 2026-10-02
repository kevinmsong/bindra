"""Listwise objective over measured candidates.

Status: scaffolded. See docs/STATUS.md.

Owed by the specification:
    Priority 2: do not treat every unmeasured pair as a nonbinder; a weaker
    measured ligand is not automatically a nonbinder.
"""

from __future__ import annotations

_SPEC_CLAUSE = (
    "Priority 2: do not treat every unmeasured pair as a nonbinder; a weaker measured ligand is not automatically a nonbinder."
)


def _not_implemented(what: str) -> "NotImplementedError":
    return NotImplementedError(
        f"{what} is scaffolded but not implemented.\n"
        f"Specification requirement: {_SPEC_CLAUSE}\n"
        "See docs/STATUS.md for the current stage."
    )


__all__ = ["_not_implemented"]
