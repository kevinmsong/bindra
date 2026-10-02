"""Blinded candidate manifests with a separate assignment key.

Status: scaffolded. See docs/STATUS.md.

Owed by the specification:
    Upgrade C: freeze model version, candidate universe, budget, ranking
    rule and selection date before inspecting new measurements.
"""

from __future__ import annotations

_SPEC_CLAUSE = (
    "Upgrade C: freeze model version, candidate universe, budget, ranking rule and selection date before inspecting new measurements."
)


def _not_implemented(what: str) -> "NotImplementedError":
    return NotImplementedError(
        f"{what} is scaffolded but not implemented.\n"
        f"Specification requirement: {_SPEC_CLAUSE}\n"
        "See docs/STATUS.md for the current stage."
    )


__all__ = ["_not_implemented"]
