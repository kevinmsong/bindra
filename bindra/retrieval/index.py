"""Versioned approximate-nearest-neighbour index over ligand vectors.

**Approximation loss is measured, not assumed.** An ANN index trades recall for
speed, and the trade is invisible unless someone checks: a fishing result that
misses a true hit because the index never retrieved it looks exactly like a
model that ranked it poorly. :meth:`VectorIndex.recall_against_exact` measures
the first; only then does a ranking metric measure the second.

**Shortlisting loss is reported separately from approximation loss.** They are
different failures. Approximation loss is the index not returning a vector it
should have; shortlisting loss is the pipeline cutting the list at K before the
true hit's rank. A single "recall" number conflates them and points at the wrong
fix.

The index is **versioned by the embedding contract** that produced it. A vector
from one checkpoint is not comparable to a vector from another, and an index
silently queried with the wrong model's embeddings returns confident nonsense.

Owed by the specification:
    Priority 2: measure ANN recall against exact search; report losses from
    both approximation and shortlisting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class IndexError_(RuntimeError):
    """The index was built or queried inconsistently with its contract."""


@dataclass(frozen=True)
class IndexContract:
    """What produced the vectors. Participates in every compatibility check."""

    checkpoint: str
    embedding_dim: int
    normalized: bool
    metric: str = "inner_product"
    schema_version: str = "retrieval-index-v1"
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "checkpoint": self.checkpoint,
            "embedding_dim": self.embedding_dim,
            "normalized": self.normalized,
            "metric": self.metric,
            "schema_version": self.schema_version,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "IndexContract":
        return cls(
            checkpoint=payload["checkpoint"],
            embedding_dim=int(payload["embedding_dim"]),
            normalized=bool(payload["normalized"]),
            metric=payload.get("metric", "inner_product"),
            schema_version=payload.get("schema_version", "retrieval-index-v1"),
            extra=payload.get("extra", {}),
        )

    def assert_compatible(self, other: "IndexContract") -> None:
        if self.to_dict() != other.to_dict():
            raise IndexError_(
                "index contract mismatch; these vectors were not produced by the "
                f"model this index was built from.\n  index: {self.to_dict()}\n"
                f"  query: {other.to_dict()}"
            )


class VectorIndex:
    """Exact search by default, with an optional IVF-style coarse quantiser.

    Exact is the default deliberately. At BindingDB's scale a brute-force
    inner product over a few million 256-d vectors is seconds, and an
    approximation that is not needed is an unmeasured error source.
    """

    def __init__(self, contract: IndexContract) -> None:
        self.contract = contract
        self.vectors: np.ndarray | None = None
        self.ids: list[str] = []
        self._centroids: np.ndarray | None = None
        self._assignments: np.ndarray | None = None

    # -- build -----------------------------------------------------------

    def add(self, ids: "list[str]", vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.contract.embedding_dim:
            raise IndexError_(
                f"expected [n, {self.contract.embedding_dim}], got {vectors.shape}"
            )
        if len(ids) != vectors.shape[0]:
            raise IndexError_(f"{len(ids)} ids for {vectors.shape[0]} vectors")
        if self.contract.normalized:
            norms = np.linalg.norm(vectors, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-3):
                raise IndexError_(
                    "contract says normalized but vectors are not unit length "
                    f"(min {norms.min():.4f}, max {norms.max():.4f})"
                )
        self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])
        self.ids.extend(ids)

    def train_coarse(self, n_clusters: int, *, seed: int = 20261001, iters: int = 12) -> None:
        """A simple k-means coarse quantiser. Optional; exact search ignores it."""
        if self.vectors is None:
            raise IndexError_("nothing to train on; add vectors first")
        rng = np.random.default_rng(seed)
        n = self.vectors.shape[0]
        n_clusters = min(n_clusters, n)
        centroids = self.vectors[rng.choice(n, n_clusters, replace=False)].copy()
        for _ in range(iters):
            assignments = np.argmax(self.vectors @ centroids.T, axis=1)
            for k in range(n_clusters):
                members = self.vectors[assignments == k]
                if len(members):
                    centroids[k] = members.mean(axis=0)
            if self.contract.normalized:
                centroids /= np.linalg.norm(centroids, axis=1, keepdims=True).clip(1e-8)
        self._centroids = centroids
        self._assignments = np.argmax(self.vectors @ centroids.T, axis=1)

    # -- query -----------------------------------------------------------

    def search_exact(self, query: np.ndarray, k: int) -> "tuple[np.ndarray, np.ndarray]":
        if self.vectors is None:
            raise IndexError_("index is empty")
        scores = np.asarray(query, dtype=np.float32) @ self.vectors.T
        k = min(k, scores.shape[-1])
        top = np.argpartition(-scores, k - 1, axis=-1)[..., :k]
        ordered = np.take_along_axis(top, np.argsort(-np.take_along_axis(scores, top, -1), -1), -1)
        return ordered, np.take_along_axis(scores, ordered, -1)

    def search(self, query: np.ndarray, k: int, *, n_probe: int = 8):
        """Approximate when a coarse quantiser was trained, exact otherwise."""
        if self._centroids is None:
            return self.search_exact(query, k)
        query = np.atleast_2d(np.asarray(query, dtype=np.float32))
        out_idx, out_scores = [], []
        for row in query:
            probes = np.argsort(-(row @ self._centroids.T))[:n_probe]
            candidates = np.flatnonzero(np.isin(self._assignments, probes))
            if candidates.size == 0:
                candidates = np.arange(self.vectors.shape[0])
            scores = row @ self.vectors[candidates].T
            kk = min(k, scores.size)
            local = np.argsort(-scores)[:kk]
            out_idx.append(candidates[local])
            out_scores.append(scores[local])
        return np.array(out_idx), np.array(out_scores)

    # -- the measurement that makes the approximation reportable ---------

    def recall_against_exact(self, queries: np.ndarray, k: int, *, n_probe: int = 8) -> dict:
        """Share of the exact top-k that the approximate search also returned."""
        exact, _ = self.search_exact(queries, k)
        approximate, _ = self.search(queries, k, n_probe=n_probe)
        hits = [
            len(set(e.tolist()) & set(a.tolist())) / max(len(e), 1)
            for e, a in zip(np.atleast_2d(exact), np.atleast_2d(approximate))
        ]
        return {
            "k": k,
            "n_probe": n_probe,
            "n_queries": len(hits),
            "approximation_recall": float(np.mean(hits)),
            "approximation_loss": float(1.0 - np.mean(hits)),
            "exact": self._centroids is None,
            "note": "approximation loss only; shortlisting loss is measured by shortlist_loss",
        }

    @staticmethod
    def shortlist_loss(true_ranks: "list[int]", shortlist_k: int) -> dict:
        """Share of true hits whose exact rank falls outside the shortlist.

        Reported apart from approximation loss on purpose: this one is the
        pipeline's cut, not the index's, and the fixes are different.
        """
        ranks = [r for r in true_ranks if r is not None]
        if not ranks:
            return {"status": "undefined", "reason": "no true hits with a known rank"}
        lost = sum(1 for r in ranks if r >= shortlist_k)
        return {
            "shortlist_k": shortlist_k,
            "n_true_hits": len(ranks),
            "n_lost_to_shortlist": lost,
            "shortlisting_loss": lost / len(ranks),
            "note": "independent of approximation loss; both must be reported",
        }

    # -- persistence -----------------------------------------------------

    def save(self, directory: "Path | str") -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        payload = {"contract": self.contract.to_dict(), "ids": self.ids}
        (directory / "index.json").write_text(json.dumps(payload), encoding="utf-8")
        if self._centroids is not None:
            np.save(directory / "centroids.npy", self._centroids)
            np.save(directory / "assignments.npy", self._assignments)

    @classmethod
    def load(cls, directory: "Path | str", *, expect: IndexContract | None = None):
        directory = Path(directory)
        payload = json.loads((directory / "index.json").read_text(encoding="utf-8"))
        contract = IndexContract.from_dict(payload["contract"])
        if expect is not None:
            contract.assert_compatible(expect)
        index = cls(contract)
        index.vectors = np.load(directory / "vectors.npy")
        index.ids = payload["ids"]
        if (directory / "centroids.npy").exists():
            index._centroids = np.load(directory / "centroids.npy")
            index._assignments = np.load(directory / "assignments.npy")
        return index

    def __len__(self) -> int:
        return 0 if self.vectors is None else int(self.vectors.shape[0])


__all__ = ["IndexContract", "IndexError_", "VectorIndex"]
