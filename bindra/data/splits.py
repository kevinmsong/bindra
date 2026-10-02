"""Immutable split manifests: random-pair, cold-drug, cold-target, double-cold, temporal.

Status: scaffolded. See docs/STATUS.md.

Owed by the specification:
    Priority 5: assign protein and drug groups independently for
    double-cold, then keep an interaction only when both sides agree; verify
    forbidden cross-partition relationships, do not trust cluster labels
    alone.
"""

from __future__ import annotations

_SPEC_CLAUSE = (
    "Priority 5: assign protein and drug groups independently for double-cold, then keep an interaction only when both sides agree; verify forbidden cross-partition relationships, do not trust cluster labels alone."
)


def _not_implemented(what: str) -> "NotImplementedError":
    return NotImplementedError(
        f"{what} is scaffolded but not implemented.\n"
        f"Specification requirement: {_SPEC_CLAUSE}\n"
        "See docs/STATUS.md for the current stage."
    )


__all__ = ["_not_implemented"]
