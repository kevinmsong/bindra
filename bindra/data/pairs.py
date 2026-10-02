"""Matched analog pairs and measured delta_y, train partitions only.

Status: scaffolded. See docs/STATUS.md.

Owed by the specification:
    Upgrade A: compatible target construct, endpoint and assay context;
    exact labels only; do not subtract censoring thresholds as if they were
    measured affinities; cap large-series influence.
"""

from __future__ import annotations

_SPEC_CLAUSE = (
    "Upgrade A: compatible target construct, endpoint and assay context; exact labels only; do not subtract censoring thresholds as if they were measured affinities; cap large-series influence."
)


def _not_implemented(what: str) -> "NotImplementedError":
    return NotImplementedError(
        f"{what} is scaffolded but not implemented.\n"
        f"Specification requirement: {_SPEC_CLAUSE}\n"
        "See docs/STATUS.md for the current stage."
    )


__all__ = ["_not_implemented"]
