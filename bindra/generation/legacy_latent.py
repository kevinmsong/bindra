"""Quarantined legacy Gaussian DDPM over atom feature vectors.

Status: scaffolded. See docs/STATUS.md.

Owed by the specification:
    Priority 3: retained only as a labelled ablation. It has no molecular
    decoder and holds bond structure fixed, so it cannot be presented as
    analog generation.
"""

from __future__ import annotations

_SPEC_CLAUSE = (
    "Priority 3: retained only as a labelled ablation. It has no molecular decoder and holds bond structure fixed, so it cannot be presented as analog generation."
)


def _not_implemented(what: str) -> "NotImplementedError":
    return NotImplementedError(
        f"{what} is scaffolded but not implemented.\n"
        f"Specification requirement: {_SPEC_CLAUSE}\n"
        "See docs/STATUS.md for the current stage."
    )


__all__ = ["_not_implemented"]
