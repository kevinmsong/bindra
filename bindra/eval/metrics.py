"""Affinity, ranking, retrieval and calibration metrics.

Status: scaffolded. See docs/STATUS.md.

Owed by the specification:
    Priority 5: report per-endpoint and per-split; say when a metric is
    undefined rather than emitting a placeholder.
"""

from __future__ import annotations

_SPEC_CLAUSE = (
    "Priority 5: report per-endpoint and per-split; say when a metric is undefined rather than emitting a placeholder."
)


def _not_implemented(what: str) -> "NotImplementedError":
    return NotImplementedError(
        f"{what} is scaffolded but not implemented.\n"
        f"Specification requirement: {_SPEC_CLAUSE}\n"
        "See docs/STATUS.md for the current stage."
    )


__all__ = ["_not_implemented"]
