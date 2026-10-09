"""Applicability-domain distance against training entities only.

Owed by the specification:
    Priority 4: return an OOD flag independently of ensemble spread.

Two separate questions get confused constantly, and the clause above exists to
keep them apart:

* **Is this input like anything the model was trained on?** A property of the
  *data*, answerable without running the model at all. That is this module.
* **Do independently trained models disagree about it?** A property of the
  *models*, from :mod:`bindra.uncertainty.ensemble`.

They come apart in the case that matters. An ensemble can agree confidently and
be confidently wrong on a molecule unlike anything it has seen; deriving the
domain flag from the spread would hide exactly that. So nothing here reads the
ensemble, and :func:`domain_status` takes no spread argument -- the independence
is enforced by the signature rather than promised in prose.

**Against training entities only.** Similarity to a validation or test molecule
says nothing about whether the model has seen its like. :func:`ligand_domain`
is given the training set explicitly and holds no default.

The fingerprint and the exact top-k Tanimoto come from
:mod:`bindra.data.similarity`, which the benchmark panels already use. Sharing
it matters: a different radius or bit width would silently redefine what
"similar" means relative to every other similarity claim in the project.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

_SPEC_CLAUSE = "Priority 4: return an OOD flag independently of ensemble spread."

#: Maximum Tanimoto to any training molecule below which a ligand is called
#: out of domain. An operational threshold, not a discovered constant: 0.4 is
#: the conventional "not a close analogue" line for Morgan fingerprints, and
#: it is stored in the report so a number can be traced to the rule that made
#: it rather than re-derived later.
DEFAULT_LIGAND_THRESHOLD = 0.40


class DomainError(ValueError):
    """The domain query is malformed."""


@dataclass(frozen=True)
class DomainStatus:
    """How far one input sits from the training set."""

    #: Maximum Tanimoto to any TRAINING molecule. 1.0 means it is in there.
    nearest_similarity: float
    #: Index of that training molecule, for provenance.
    nearest_index: int
    out_of_domain: bool
    threshold: float
    #: Named so a report states the rule, not just the verdict.
    method: str = "morgan-tanimoto-max-vs-train"

    def to_dict(self) -> dict[str, object]:
        return {
            "nearest_similarity": self.nearest_similarity,
            "nearest_index": self.nearest_index,
            "out_of_domain": self.out_of_domain,
            "threshold": self.threshold,
            "method": self.method,
            "note": (
                "Computed from the data alone. Independent of ensemble spread: "
                "a confident ensemble on an out-of-domain input is the case "
                "this flag exists to surface."
            ),
        }


def ligand_domain(
    query_smiles: Sequence[str],
    training_smiles: Sequence[str],
    *,
    threshold: float = DEFAULT_LIGAND_THRESHOLD,
    device: str = "cpu",
) -> list[DomainStatus]:
    """Nearest training molecule for each query, and the resulting flag.

    ``training_smiles`` is required and has no default, because the one error
    this is here to prevent is measuring distance against the wrong set.
    """
    from bindra.data.similarity import morgan_fingerprints, nearest_neighbour

    if not training_smiles:
        raise DomainError(
            "no training molecules supplied; applicability domain is only "
            f"meaningful against the training set. {_SPEC_CLAUSE}"
        )
    if not 0.0 <= threshold <= 1.0:
        raise DomainError(f"threshold must be in [0, 1], got {threshold}")

    query_packed, query_valid = morgan_fingerprints(list(query_smiles))
    train_packed, train_valid = morgan_fingerprints(list(training_smiles))
    train_packed = train_packed[train_valid]
    if train_packed.shape[0] == 0:
        raise DomainError("no training molecule parsed; cannot define a domain")

    result = nearest_neighbour(query_packed, train_packed, top_k=1, device=device)
    statuses: list[DomainStatus] = []
    for row, valid in enumerate(query_valid):
        if not valid:
            # An unparseable query is maximally out of domain, not similarity 0
            # by accident. Stated rather than inferred from a zero row.
            statuses.append(
                DomainStatus(
                    nearest_similarity=float("nan"),
                    nearest_index=-1,
                    out_of_domain=True,
                    threshold=threshold,
                    method="unparseable-query",
                )
            )
            continue
        similarity = float(result.similarity[row, 0])
        statuses.append(
            DomainStatus(
                nearest_similarity=similarity,
                nearest_index=int(result.index[row, 0]),
                out_of_domain=similarity < threshold,
                threshold=threshold,
            )
        )
    return statuses


def domain_status(
    query_smiles: Sequence[str],
    training_smiles: Sequence[str],
    *,
    threshold: float = DEFAULT_LIGAND_THRESHOLD,
) -> list[DomainStatus]:
    """Public entry point. Takes no ensemble argument, by design.

    The specification requires the flag be independent of ensemble spread, so
    the spread is not available here to be used. Enforced by the signature
    rather than by a comment asking the next caller not to.
    """
    return ligand_domain(query_smiles, training_smiles, threshold=threshold)


def summarise(statuses: Sequence[DomainStatus]) -> dict[str, object]:
    """Counts for a report, with the threshold that produced them."""
    if not statuses:
        return {"n": 0}
    similarities = np.array(
        [s.nearest_similarity for s in statuses], dtype=np.float64
    )
    finite = similarities[np.isfinite(similarities)]
    return {
        "n": len(statuses),
        "n_out_of_domain": int(sum(s.out_of_domain for s in statuses)),
        "share_out_of_domain": float(
            sum(s.out_of_domain for s in statuses) / len(statuses)
        ),
        "median_nearest_similarity": float(np.median(finite)) if finite.size else None,
        "threshold": statuses[0].threshold,
        "n_unparseable": int((~np.isfinite(similarities)).sum()),
    }


__all__ = [
    "DEFAULT_LIGAND_THRESHOLD",
    "DomainError",
    "DomainStatus",
    "domain_status",
    "ligand_domain",
    "summarise",
]
