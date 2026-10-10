"""Target fishing and seed-assisted analog fishing APIs.

Two questions that are routinely reported as one, and must not be:

**Target-only fishing** is zero-shot. Given a target and a library, rank the
library. Nothing about a known binder is supplied.

**Seed-assisted fishing** is given a known active for that target and asked to
find more. That is a far easier question -- a 2-D similarity search from the
seed does well at it without any model -- and a number from it is not evidence
of discovery. :func:`fish` refuses to merge the two: the mode is recorded in the
result, and :func:`combined_report` keeps them in separate blocks.

Because seed-assisted results are so easily explained by similarity alone, the
seed-assisted path also reports a **similarity-only baseline** beside the model,
so a reader can see how much of the ranking the model actually contributed.

Owed by the specification:
    Priority 5: report target-only and seed-assisted fishing separately;
    seed-assisted evaluation is not zero-shot discovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from bindra.retrieval.index import VectorIndex


class FishingError(ValueError):
    """A fishing query was posed in a way its result could not be read."""


Mode = Literal["target_only", "seed_assisted"]


@dataclass(frozen=True)
class FishingResult:
    """A ranking, and the mode that produced it, inseparably."""

    mode: Mode
    ranked_ids: list[str]
    scores: list[float]
    n_library: int
    seed_id: str | None = None

    def to_dict(self) -> dict:
        payload = {
            "mode": self.mode,
            "n_library": self.n_library,
            "n_returned": len(self.ranked_ids),
            "ranked_ids": self.ranked_ids,
            "scores": self.scores,
        }
        if self.mode == "seed_assisted":
            payload["seed_id"] = self.seed_id
            payload["caveat"] = (
                "seed-assisted: a known active for this target was supplied. "
                "This is not zero-shot discovery and must not be reported beside "
                "a target-only number as if it were."
            )
        return payload


def fish(
    index: VectorIndex,
    query_vector: np.ndarray,
    *,
    k: int = 100,
    mode: Mode = "target_only",
    seed_id: str | None = None,
    seed_vector: np.ndarray | None = None,
    seed_weight: float = 0.5,
) -> FishingResult:
    """Rank a library for one target.

    ``mode="seed_assisted"`` requires a seed and blends its similarity into the
    score. The seed is required rather than optional in that mode so a
    seed-assisted call cannot silently degrade into a target-only one and be
    reported under the easier label.
    """
    if mode not in ("target_only", "seed_assisted"):
        raise FishingError(f"unknown mode {mode!r}")
    if mode == "seed_assisted":
        if seed_vector is None or seed_id is None:
            raise FishingError(
                "seed-assisted fishing needs both seed_id and seed_vector; "
                "without them this is a target-only query and must say so"
            )
        if not 0.0 <= seed_weight <= 1.0:
            raise FishingError(f"seed_weight must be in [0, 1], got {seed_weight}")

    if index.vectors is None:
        raise FishingError("the index is empty")
    scores = np.asarray(query_vector, dtype=np.float32).reshape(-1) @ index.vectors.T
    if mode == "seed_assisted":
        seed_scores = np.asarray(seed_vector, dtype=np.float32).reshape(-1) @ index.vectors.T
        scores = (1.0 - seed_weight) * scores + seed_weight * seed_scores

    k = min(k, scores.size)
    order = np.argsort(-scores)[:k]
    return FishingResult(
        mode=mode,
        ranked_ids=[index.ids[i] for i in order],
        scores=[float(scores[i]) for i in order],
        n_library=len(index),
        seed_id=seed_id if mode == "seed_assisted" else None,
    )


def enrichment(ranked_ids: "list[str]", actives: "set[str]", n_library: int, fraction: float) -> dict:
    """Enrichment factor at a fraction of the library."""
    if not actives or n_library <= 0:
        return {"status": "undefined", "reason": "no actives or empty library"}
    top = max(1, int(round(fraction * n_library)))
    hits = sum(1 for identifier in ranked_ids[:top] if identifier in actives)
    prevalence = len(actives) / n_library
    return {
        "fraction": fraction,
        "n_top": top,
        "n_hits": hits,
        "hit_rate": hits / top,
        "enrichment_factor": (hits / top) / prevalence if prevalence else float("nan"),
        "max_possible": (min(1.0, len(actives) / top)) / prevalence if prevalence else float("nan"),
    }


def similarity_baseline(
    index: VectorIndex, seed_vector: np.ndarray, actives: "set[str]", *, fractions=(0.01, 0.05)
) -> dict:
    """What ranking by seed similarity alone achieves, with no target at all.

    The number a seed-assisted result must beat to mean anything.
    """
    scores = np.asarray(seed_vector, dtype=np.float32).reshape(-1) @ index.vectors.T
    order = np.argsort(-scores)
    ranked = [index.ids[i] for i in order]
    return {
        "method": "seed similarity only, target ignored",
        "enrichment": {
            f"top_{f:.0%}": enrichment(ranked, actives, len(index), f) for f in fractions
        },
    }


def combined_report(
    target_only: "dict | None" = None, seed_assisted: "dict | None" = None
) -> dict:
    """Keep the two questions in separate blocks, and say why."""
    return {
        "target_only": target_only
        or {"status": "not run", "note": "the zero-shot question"},
        "seed_assisted": seed_assisted
        or {"status": "not run", "note": "the easier, seeded question"},
        "warning": (
            "These answer different questions and their numbers are not "
            "comparable. Seed-assisted fishing is supplied a known active and "
            "is not zero-shot discovery; compare it against the "
            "similarity-only baseline, never against the target-only result."
        ),
    }


__all__ = [
    "FishingError",
    "FishingResult",
    "Mode",
    "combined_report",
    "enrichment",
    "fish",
    "similarity_baseline",
]
