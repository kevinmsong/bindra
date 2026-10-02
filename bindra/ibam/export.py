"""Raw attention matrices plus plotting metadata.

Status: scaffolded. See docs/STATUS.md.

Owed by the specification:
    Priority 2: export numeric matrices, not just images; residue x
    token/bond and derived residue x atom views; bond tokens split equally
    between endpoints and labelled as a derived aggregation.
"""

from __future__ import annotations

_SPEC_CLAUSE = (
    "Priority 2: export numeric matrices, not just images; residue x token/bond and derived residue x atom views; bond tokens split equally between endpoints and labelled as a derived aggregation."
)


def _not_implemented(what: str) -> "NotImplementedError":
    return NotImplementedError(
        f"{what} is scaffolded but not implemented.\n"
        f"Specification requirement: {_SPEC_CLAUSE}\n"
        "See docs/STATUS.md for the current stage."
    )


__all__ = ["_not_implemented"]
