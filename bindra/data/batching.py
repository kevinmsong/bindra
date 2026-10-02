"""Datasets, token-budget batch samplers, endpoint-aware collation.

Status: scaffolded. See docs/STATUS.md.

Owed by the specification:
    Port of the legacy TokenBudgetBatchSampler, extended with endpoint masks
    and censoring relations.
"""

from __future__ import annotations

_SPEC_CLAUSE = (
    "Port of the legacy TokenBudgetBatchSampler, extended with endpoint masks and censoring relations."
)


def _not_implemented(what: str) -> "NotImplementedError":
    return NotImplementedError(
        f"{what} is scaffolded but not implemented.\n"
        f"Specification requirement: {_SPEC_CLAUSE}\n"
        "See docs/STATUS.md for the current stage."
    )


__all__ = ["_not_implemented"]
