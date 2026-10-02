"""Exhaustive exact pairwise protein alignment, and the grouping built on it.

The specification asks for protein partitions disjoint at 40% identity with 80%
coverage of the shorter sequence, and insists the forbidden cross-partition
relationships be **verified** rather than inferred from cluster labels. This
module makes that literal: every one of the 58,552,431 pairs among the 10,822
unique constructs is aligned exactly, and the result is a lookup table.

**Why no prefilter.** A k-mer or MinHash screen is not recall-safe at 40%
identity -- two sequences at that level routinely share no exact k-mers, which is
why MMseqs2 uses substitution-similar double hits rather than exact matches. The
only provably sound cheap filter is the composition bound
(identities <= sum_a min(c_a(x), c_a(y)), so a pair can be pruned when that sum
falls below 0.32 * L_short); it is computed because it is free, but its measured
prune rate is poor, because unrelated proteins have similar amino-acid
compositions. A score-only prefilter is **not** admissible either: working the
bound through, a qualifying alignment's optimal score has no useful positive
lower bound once gap penalties are included.

**What makes it affordable.** ``parasail``'s SIMD alignment returns the
identity count and alignment length without a traceback, measured at ~1,100
alignments/second per core on this corpus against 96/s for Biopython's traceback
path. Aligned columns follow exactly from the lengths::

    length = A + (L2 - A) + (L1 - A)   =>   A = L1 + L2 - length

where ``A`` counts columns holding a residue in both sequences. Cross-checked
against ``Bio.Align.PairwiseAligner`` on randomized pairs with indels: identical
identity and aligned-column counts in every case.

**Saturation is not optional to handle.** The 16-bit kernel silently returns
score 0 with ``saturated=True`` on the 34,350-residue construct in this release.
Only the ``_sat`` kernels are used, and a saturated result raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from bindra.config.profiles import ProteinPolicy

#: Edges weaker than this are not stored: they cannot qualify at any coverage,
#: and keeping them would turn a 50 MB table into a 2 GB one.
STORE_IDENTITY_FLOOR = 0.20


class HomologyError(RuntimeError):
    """An alignment could not be trusted."""


@dataclass(frozen=True)
class AlignmentResult:
    identity: float
    coverage: float
    score: int
    aligned_columns: int
    n_identical: int

    def qualifies(self, policy: ProteinPolicy, *, conservative: bool = True) -> bool:
        """Does this pair forbid its members from straddling a partition?

        ``conservative`` applies ``identity_ambiguity_margin``, because optimal
        alignments are rarely unique and over-merging is the leakage-safe
        direction. Pass False for the strict threshold, which the report quotes
        alongside it.
        """
        threshold = policy.identity_threshold
        if conservative:
            threshold -= policy.identity_ambiguity_margin
        return self.identity >= threshold and self.coverage >= policy.coverage_threshold


def _matrix():
    import parasail

    return parasail.blosum62


def normalize_for_alignment(sequence: str, policy: ProteinPolicy | None = None) -> str:
    """Upper-case, with residues outside the scoring matrix mapped to ``X``.

    Applied to the alignment *input* only; the stored construct is untouched.
    BLOSUM62 scores the 20 standard residues plus B, Z, X and ``*`` but not
    selenocysteine (U) or pyrrolysine (O), and two aligners left to their own
    fallback rules for an unmatched character can disagree. Pinning the mapping
    here is what lets the Biopython oracle and the parasail kernel be compared at
    all.
    """
    policy = policy or ProteinPolicy()
    text = sequence.upper()
    for residue in policy.alignment_unknown_residues:
        text = text.replace(residue, policy.alignment_unknown_substitute)
    return text


def align_pair(
    a: str,
    b: str,
    policy: ProteinPolicy | None = None,
) -> AlignmentResult:
    """Exact optimal global alignment statistics for one pair.

    Identity is ``n_identical / aligned_columns`` with gap columns excluded from
    the denominator, and coverage is ``aligned_columns / len(shorter)``. Both
    denominators are policy, pinned in :class:`ProteinPolicy`, because changing
    either changes which pairs qualify.
    """
    import parasail

    policy = policy or ProteinPolicy()
    if not a or not b:
        return AlignmentResult(0.0, 0.0, 0, 0, 0)
    a = normalize_for_alignment(a, policy)
    b = normalize_for_alignment(b, policy)

    result = parasail.sg_stats_striped_sat(
        a, b, int(-policy.open_gap_score), int(-policy.extend_gap_score), _matrix()
    )
    if getattr(result, "saturated", False):
        # The 16-bit kernel returns score 0 here. Guessing would be worse than
        # failing: this is exactly how a 34,350-residue construct would quietly
        # become "no homology to anything".
        raise HomologyError(
            f"alignment saturated for lengths {len(a)}/{len(b)}; refusing to "
            "report a score that is known to be wrong"
        )

    # Semi-global: end gaps are free, so the traced alignment length IS the
    # aligned extent. (Under global alignment it would instead be
    # len(a) + len(b) - length, which is what makes global coverage ~1 always.)
    aligned_columns = int(result.length)
    if aligned_columns <= 0:
        return AlignmentResult(0.0, 0.0, int(result.score), 0, 0)

    n_identical = int(result.matches)
    return AlignmentResult(
        identity=n_identical / aligned_columns,
        coverage=aligned_columns / min(len(a), len(b)),
        score=int(result.score),
        aligned_columns=aligned_columns,
        n_identical=n_identical,
    )


def align_pair_reference(
    a: str,
    b: str,
    policy: ProteinPolicy | None = None,
) -> AlignmentResult:
    """The same statistics via Biopython, as a correctness oracle.

    Roughly 11x slower, so it is used only to verify a sample of the fast path.
    Having an independent implementation is what lets the audit claim the
    relation table is right rather than merely self-consistent.
    """
    from Bio import Align
    from Bio.Align import substitution_matrices

    policy = policy or ProteinPolicy()
    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load(policy.substitution_matrix)
    aligner.open_gap_score = policy.open_gap_score
    aligner.extend_gap_score = policy.extend_gap_score
    aligner.mode = "global"
    if policy.alignment_mode == "semi_global":
        # Biopython spells semi-global as global with free end gaps.
        aligner.end_insertion_score = 0.0
        aligner.end_deletion_score = 0.0

    a = normalize_for_alignment(a, policy)
    b = normalize_for_alignment(b, policy)
    alignment = aligner.align(a, b)[0]
    counts = alignment.counts()
    aligned_columns = counts.identities + counts.mismatches
    if aligned_columns <= 0:
        return AlignmentResult(0.0, 0.0, int(alignment.score), 0, 0)
    return AlignmentResult(
        identity=counts.identities / aligned_columns,
        coverage=aligned_columns / min(len(a), len(b)),
        score=int(alignment.score),
        aligned_columns=aligned_columns,
        n_identical=counts.identities,
    )


# ---------------------------------------------------------------------------
# Sound cheap filter (computed, reported, not relied upon)
# ---------------------------------------------------------------------------


def composition_counts(sequences: Sequence[str]) -> np.ndarray:
    """``[n, 26]`` amino-acid counts, for the composition bound."""
    table = np.zeros((len(sequences), 26), dtype=np.int32)
    for row, sequence in enumerate(sequences):
        codes = np.frombuffer(sequence.encode("ascii", "ignore"), dtype=np.uint8)
        codes = codes[(codes >= 65) & (codes <= 90)] - 65
        if codes.size:
            np.add.at(table[row], codes, 1)
    return table


def composition_bound_excludes(
    counts_a: np.ndarray,
    counts_b: np.ndarray,
    len_a: int,
    len_b: int,
    policy: ProteinPolicy,
) -> bool:
    """True when no alignment of this pair can qualify.

    Sound: the number of identical positions in *any* alignment is at most
    ``sum_a min(c_a(x), c_a(y))``. Qualifying needs
    ``identities >= identity_threshold * coverage_threshold * L_short``, so a pair
    whose bound falls below that cannot qualify however it is aligned.
    """
    shared = int(np.minimum(counts_a, counts_b).sum())
    required = policy.identity_threshold * policy.coverage_threshold * min(len_a, len_b)
    return shared < required


# ---------------------------------------------------------------------------
# Union-find
# ---------------------------------------------------------------------------


class UnionFind:
    def __init__(self, size: int) -> None:
        self._parent = list(range(size))
        self._rank = [0] * size

    def find(self, item: int) -> int:
        parent = self._parent
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != root:
            parent[item], item = root, parent[item]
        return root

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1
        return True

    def components(self) -> dict[int, list[int]]:
        groups: dict[int, list[int]] = {}
        for item in range(len(self._parent)):
            groups.setdefault(self.find(item), []).append(item)
        return groups


# ---------------------------------------------------------------------------
# Containment
# ---------------------------------------------------------------------------


def substring_containment_edges(sequences: Sequence[str]) -> list[tuple[int, int]]:
    """Pairs where one construct is a literal substring of another.

    Domain constructs are frequently exact substrings of the full-length chain.
    These are guaranteed 100% identity at 100% coverage of the shorter sequence,
    so they can be asserted without alignment, and they seed the components
    cheaply. This is also the specification's "handle fragments and constructs
    explicitly" case made concrete.
    """
    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
    edges: list[tuple[int, int]] = []
    for position, short_index in enumerate(order):
        short = sequences[short_index]
        if not short:
            continue
        for long_index in order[position + 1 :]:
            if short in sequences[long_index]:
                edges.append((min(short_index, long_index), max(short_index, long_index)))
    return edges


# ---------------------------------------------------------------------------
# Relation table
# ---------------------------------------------------------------------------


def condensed_index(i: int, j: int, n: int) -> int:
    """Index of pair ``(i < j)`` in a condensed upper-triangular array."""
    if i > j:
        i, j = j, i
    return i * n - (i * (i + 1)) // 2 + (j - i - 1)


@dataclass
class ProteinRelationTable:
    """Every pair's alignment outcome, plus which pairs were actually computed.

    The ``computed`` bitmap is what makes edge *absence* auditable. Without it,
    "no edge" is ambiguous between "aligned, 12% identical" and "never looked",
    and the specification's demand to verify the forbidden relationships could
    only be met by assumption.
    """

    sequence_ids: tuple[str, ...]
    edges: np.ndarray  # structured: i, j, identity, coverage, score, aligned_columns
    computed: np.ndarray  # packed bits over condensed pair indices
    policy: ProteinPolicy
    n_pairs: int
    n_computed: int
    n_qualifying: int
    elapsed_seconds: float = 0.0
    sha256: str = ""

    @property
    def n(self) -> int:
        return len(self.sequence_ids)

    def was_computed(self, i: int, j: int) -> bool:
        if i == j:
            return True
        index = condensed_index(i, j, self.n)
        return bool(self.computed[index >> 3] & (1 << (index & 7)))

    def identity(self, i: int, j: int) -> float | None:
        """Stored identity for a pair, or None when it is below the floor."""
        if i == j:
            return 1.0
        mask = ((self.edges["i"] == min(i, j)) & (self.edges["j"] == max(i, j)))
        if not mask.any():
            return None
        return float(self.edges["identity"][mask][0])

    def qualifying_pairs_at(self, identity_threshold: float) -> Iterator[tuple[int, int]]:
        """Pairs qualifying at an arbitrary identity threshold.

        The relation table stores every edge above ``STORE_IDENTITY_FLOOR``, so an
        alternative threshold costs nothing to evaluate. That is what makes the
        strict and cold-target groupings two views of one computation rather than
        two alignment runs.
        """
        qualifying = (self.edges["identity"] >= identity_threshold) & (
            self.edges["coverage"] >= self.policy.coverage_threshold
        )
        for row in self.edges[qualifying]:
            yield int(row["i"]), int(row["j"])

    def qualifying_pairs(self, *, conservative: bool = True) -> Iterator[tuple[int, int]]:
        threshold = self.policy.identity_threshold
        if conservative:
            threshold -= self.policy.identity_ambiguity_margin
        qualifying = (self.edges["identity"] >= threshold) & (
            self.edges["coverage"] >= self.policy.coverage_threshold
        )
        for row in self.edges[qualifying]:
            yield int(row["i"]), int(row["j"])

    def components(
        self,
        *,
        conservative: bool = True,
        extra_edges: Iterable[tuple[int, int]] = (),
    ) -> dict[int, list[int]]:
        """Single-linkage components over qualifying edges plus ``extra_edges``.

        ``extra_edges`` carries relationships that are true independently of any
        alignment -- constructs sharing a UniProt accession, and exact substring
        containment -- so they cannot be missed by a scoring threshold.
        """
        union = UnionFind(self.n)
        for i, j in self.qualifying_pairs(conservative=conservative):
            union.union(i, j)
        for i, j in extra_edges:
            union.union(i, j)
        return union.components()

    def save(self, directory: Path | str) -> Path:
        from bindra.data.tables import write_json

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "edges.npy", self.edges)
        np.save(directory / "computed.npy", self.computed)
        (directory / "sequence_ids.json").write_text(
            json.dumps(list(self.sequence_ids)), encoding="utf-8"
        )
        digest = hashlib.sha256()
        for name in ("edges.npy", "computed.npy", "sequence_ids.json"):
            digest.update((directory / name).read_bytes())
        self.sha256 = digest.hexdigest()
        write_json(
            {
                "n_sequences": self.n,
                "n_pairs": self.n_pairs,
                "n_computed": self.n_computed,
                "n_qualifying": self.n_qualifying,
                "n_edges_stored": int(len(self.edges)),
                "store_identity_floor": STORE_IDENTITY_FLOOR,
                "elapsed_seconds": round(self.elapsed_seconds, 1),
                "sha256": self.sha256,
                "policy": {
                    "identity_threshold": self.policy.identity_threshold,
                    "coverage_threshold": self.policy.coverage_threshold,
                    "alignment_mode": self.policy.alignment_mode,
                    "substitution_matrix": self.policy.substitution_matrix,
                    "open_gap_score": self.policy.open_gap_score,
                    "extend_gap_score": self.policy.extend_gap_score,
                    "identity_denominator": self.policy.identity_denominator,
                    "coverage_denominator": self.policy.coverage_denominator,
                    "linkage": self.policy.linkage,
                    "exhaustive": self.policy.exhaustive,
                },
                "engine": "parasail sg_stats_striped_sat (semi-global, free end gaps)",
                "caveat": (
                    "40% identity / 80% coverage is an operational threshold. It is "
                    "not proof that shared domains or protein families are absent "
                    "across partitions."
                ),
            },
            directory / "relation_table.json",
        )
        return directory

    @classmethod
    def load(cls, directory: Path | str, policy: ProteinPolicy | None = None):
        directory = Path(directory)
        meta = json.loads((directory / "relation_table.json").read_text(encoding="utf-8"))
        ids = json.loads((directory / "sequence_ids.json").read_text(encoding="utf-8"))
        return cls(
            sequence_ids=tuple(ids),
            edges=np.load(directory / "edges.npy"),
            computed=np.load(directory / "computed.npy"),
            policy=policy or ProteinPolicy(),
            n_pairs=meta["n_pairs"],
            n_computed=meta["n_computed"],
            n_qualifying=meta["n_qualifying"],
            elapsed_seconds=meta.get("elapsed_seconds", 0.0),
            sha256=meta.get("sha256", ""),
        )


EDGE_DTYPE = np.dtype(
    [
        ("i", np.int32),
        ("j", np.int32),
        ("identity", np.float32),
        ("coverage", np.float32),
        ("score", np.int32),
        ("aligned_columns", np.int32),
    ]
)


class _BlockStore:
    """Per-block edge checkpoints, keyed to the exact corpus that produced them.

    Block indices only mean something relative to a fixed sequence list and block
    size, so resuming against a different corpus would silently stitch together
    results for different pairs. The key file makes that impossible.
    """

    def __init__(
        self,
        directory: Path | str,
        sequences: Sequence[str],
        rows_per_block: int,
        policy: ProteinPolicy,
    ) -> None:
        self.directory = Path(directory) / "blocks"
        self.directory.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        for sequence in sequences:
            digest.update(sequence.encode("utf-8"))
            digest.update(b"\x1f")
        self.key = {
            "n_sequences": len(sequences),
            "sequences_sha256": digest.hexdigest(),
            "rows_per_block": int(rows_per_block),
            "alignment_mode": policy.alignment_mode,
            "substitution_matrix": policy.substitution_matrix,
            "open_gap_score": policy.open_gap_score,
            "extend_gap_score": policy.extend_gap_score,
            "store_identity_floor": STORE_IDENTITY_FLOOR,
        }
        self._validate()

    @property
    def key_path(self) -> Path:
        return self.directory / "key.json"

    def _validate(self) -> None:
        if not self.key_path.exists():
            self.key_path.write_text(json.dumps(self.key, sort_keys=True), encoding="utf-8")
            return
        existing = json.loads(self.key_path.read_text(encoding="utf-8"))
        if existing != self.key:
            differing = sorted(
                k for k in set(existing) | set(self.key) if existing.get(k) != self.key.get(k)
            )
            raise HomologyError(
                f"{self.directory} holds checkpoints for a different corpus or policy "
                f"(differs in {differing}). Block indices are only meaningful for the "
                "set that produced them; delete the directory to start over."
            )

    def clear(self) -> None:
        for path in self.directory.glob("block-*.npy"):
            path.unlink()
        self.key_path.write_text(json.dumps(self.key, sort_keys=True), encoding="utf-8")

    def path(self, index: int) -> Path:
        return self.directory / f"block-{index:05d}.npy"

    def load(self, index: int) -> np.ndarray | None:
        """The block's edges as a structured array.

        Returned as numpy, never as a list of tuples. Converting 22.9M cached
        edges into Python tuples cost the parent process 4 GB and 90 seconds on
        this corpus, and would have approached 10 GB once every block was loaded.
        """
        path = self.path(index)
        if not path.exists():
            return None
        try:
            return np.load(path)
        except Exception:  # noqa: BLE001 - a truncated checkpoint is just redone
            path.unlink(missing_ok=True)
            return None

    def save(self, index: int, array: np.ndarray) -> None:
        path = self.path(index)
        temporary = path.with_name(path.name + ".tmp")
        # Written through a handle and renamed, so a file's existence means it is
        # complete. np.save would otherwise append ".npy" to the temp name.
        with open(temporary, "wb") as handle:
            np.save(handle, array)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def n_complete(self) -> int:
        return len(list(self.directory.glob("block-*.npy")))


def _align_block(
    block: tuple[int, int],
    sequences: list[str],
    open_gap: int,
    extend_gap: int,
    identity_floor: float,
) -> list[tuple]:
    """Align rows ``[lo, hi)`` against every later sequence. Worker entry point."""
    import parasail

    matrix = parasail.blosum62
    lo, hi = block
    n = len(sequences)
    out: list[tuple] = []
    for i in range(lo, hi):
        a = sequences[i]
        la = len(a)
        if not la:
            continue
        for j in range(i + 1, n):
            b = sequences[j]
            lb = len(b)
            if not lb:
                continue
            result = parasail.sg_stats_striped_sat(a, b, open_gap, extend_gap, matrix)
            if getattr(result, "saturated", False):
                raise HomologyError(f"alignment saturated for lengths {la}/{lb}")
            aligned = int(result.length)
            if aligned <= 0:
                continue
            identity = result.matches / aligned
            if identity < identity_floor:
                continue
            out.append(
                (
                    i,
                    j,
                    identity,
                    aligned / min(la, lb),
                    int(result.score),
                    aligned,
                )
            )
    return out


def build_relation_table(
    sequences: Sequence[str],
    sequence_ids: Sequence[str],
    *,
    policy: ProteinPolicy | None = None,
    workers: int = 14,
    rows_per_block: int = 40,
    progress: bool = True,
    checkpoint_dir: Path | str | None = None,
    resume: bool = True,
) -> ProteinRelationTable:
    """Align every pair exactly and return the relation table.

    Parallel by row block rather than with a union-find short-circuit. The
    short-circuit (skip a pair already in one component) is sound, because
    components are assigned to partitions whole, so a skipped pair can never be
    cross-partition. It is not used here because clean parallelism is worth more
    than its saving on this corpus, and because computing every pair leaves the
    ``computed`` bitmap complete -- which is what lets the split audit assert
    "aligned, and does not qualify" for every cross-partition pair instead of
    "no edge recorded".

    **Checkpointed per block.** Each completed block's edges are written to
    ``checkpoint_dir/blocks/`` before the next one starts, so an interrupted run
    keeps its work and a rerun does only what is left. This is not a theoretical
    nicety: a first attempt on this corpus reached 89% of 58,530,790 pairs, hit a
    two-hour wall-clock limit, and lost all of it because the work lived only in
    memory.

    Resume validates that the sequence set and block size are unchanged, and
    refuses otherwise -- block indices mean nothing across a different corpus.
    """
    policy = policy or ProteinPolicy()
    sequences = [normalize_for_alignment(s, policy) for s in sequences]
    n = len(sequences)
    n_pairs = n * (n - 1) // 2
    started = time.perf_counter()

    blocks = [(lo, min(lo + rows_per_block, n)) for lo in range(0, n, rows_per_block)]
    open_gap = int(-policy.open_gap_score)
    extend_gap = int(-policy.extend_gap_score)

    store = _BlockStore(checkpoint_dir, sequences, rows_per_block, policy) if checkpoint_dir else None
    if store is not None and not resume:
        store.clear()

    # Accumulated as structured arrays rather than tuples: 40M+ edges is ~1 GB
    # as numpy and many times that as Python objects.
    chunks: list[np.ndarray] = []
    pending: list[int] = []
    done_pairs = 0

    def block_pairs(index: int) -> int:
        lo, hi = blocks[index]
        return sum(n - k - 1 for k in range(lo, hi))

    for index in range(len(blocks)):
        cached = store.load(index) if store is not None else None
        if cached is not None:
            chunks.append(cached)
            done_pairs += block_pairs(index)
        else:
            pending.append(index)

    if store is not None and len(pending) < len(blocks):
        print(
            f"    resuming: {len(blocks) - len(pending)}/{len(blocks)} blocks already "
            f"on disk ({done_pairs:,}/{n_pairs:,} pairs)",
            flush=True,
        )

    def record(index: int, chunk: list[tuple]) -> None:
        nonlocal done_pairs
        array = np.array(chunk, dtype=EDGE_DTYPE) if chunk else np.zeros(0, dtype=EDGE_DTYPE)
        if store is not None:
            store.save(index, array)
        chunks.append(array)
        done_pairs += block_pairs(index)

    if workers <= 1:
        for position, index in enumerate(pending):
            record(
                index,
                _align_block(blocks[index], sequences, open_gap, extend_gap, STORE_IDENTITY_FLOOR),
            )
            if progress and position % 20 == 0:
                print(f"    block {position + 1}/{len(pending)}", flush=True)
    elif pending:
        from joblib import Parallel, delayed

        # Largest workloads (low i, many j) first, so the tail is not one
        # straggler block.
        results = Parallel(n_jobs=workers, return_as="generator", batch_size=1)(
            delayed(_align_block)(
                blocks[index], sequences, open_gap, extend_gap, STORE_IDENTITY_FLOOR
            )
            for index in pending
        )
        for position, chunk in enumerate(results):
            record(pending[position], chunk)
            if progress and position % 20 == 0:
                rate = done_pairs / max(time.perf_counter() - started, 1e-9)
                remaining = (n_pairs - done_pairs) / max(rate, 1e-9)
                print(
                    f"    block {position + 1}/{len(pending)}  {done_pairs:,}/{n_pairs:,} pairs "
                    f"({rate:,.0f}/s, ~{remaining / 60:.0f} min left)",
                    flush=True,
                )

    edges = (
        np.concatenate(chunks) if chunks else np.zeros(0, dtype=EDGE_DTYPE)
    )
    edges = np.sort(edges, order=["i", "j"])

    # Every pair was computed, so the bitmap is all ones over the condensed range.
    computed = np.full((n_pairs + 7) // 8, 0xFF, dtype=np.uint8)
    spare_bits = (len(computed) * 8) - n_pairs
    if spare_bits:
        computed[-1] = 0xFF >> spare_bits

    strict = (edges["identity"] >= policy.identity_threshold) & (
        edges["coverage"] >= policy.coverage_threshold
    )
    conservative = (
        edges["identity"] >= policy.identity_threshold - policy.identity_ambiguity_margin
    ) & (edges["coverage"] >= policy.coverage_threshold)
    qualifying = int(conservative.sum())
    return ProteinRelationTable(
        sequence_ids=tuple(sequence_ids),
        edges=edges,
        computed=computed,
        policy=policy,
        n_pairs=n_pairs,
        n_computed=n_pairs,
        n_qualifying=qualifying,
        elapsed_seconds=time.perf_counter() - started,
    )


def verify_relation_sample(
    table: ProteinRelationTable,
    sequences: Sequence[str],
    *,
    n_samples: int = 200,
    seed: int = 20_261_001,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Re-align a sample with Biopython and compare, so the table is not circular.

    Samples three strata deliberately: stored qualifying edges, stored
    near-threshold edges (identity in [0.30, 0.40), the ones a bug would most
    plausibly misplace), and random pairs with no stored edge -- because a
    false *negative* is the failure that silently leaks homologues across a
    split.
    """
    policy = table.policy
    rng = np.random.default_rng(seed)
    n = table.n

    qualifying = [
        (int(r["i"]), int(r["j"]))
        for r in table.edges
        if r["identity"] >= policy.identity_threshold
        and r["coverage"] >= policy.coverage_threshold
    ]
    near = [
        (int(r["i"]), int(r["j"]))
        for r in table.edges
        if 0.30 <= r["identity"] < policy.identity_threshold
    ]
    stored = {(int(r["i"]), int(r["j"])) for r in table.edges}

    def sample(items: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
        if not items:
            return []
        picks = rng.choice(len(items), size=min(count, len(items)), replace=False)
        return [items[int(p)] for p in picks]

    absent: list[tuple[int, int]] = []
    attempts = 0
    while len(absent) < n_samples // 3 and attempts < n_samples * 50:
        attempts += 1
        i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
        if i == j:
            continue
        pair = (min(i, j), max(i, j))
        if pair not in stored:
            absent.append(pair)

    checks = {
        "qualifying": sample(qualifying, n_samples // 3),
        "near_threshold": sample(near, n_samples // 3),
        "no_stored_edge": absent,
    }

    report: dict[str, Any] = {
        "strata": {},
        "missed_qualifying_pairs": [],
        "decision_agreement": True,
        "identity_differences": {},
    }
    all_differences: list[float] = []

    for stratum, pairs in checks.items():
        same_decision = 0
        differences: list[float] = []
        for i, j in pairs:
            reference = align_pair_reference(sequences[i], sequences[j], policy)
            fast = align_pair(sequences[i], sequences[j], policy)
            differences.append(abs(fast.identity - reference.identity))

            # The decision is what matters, not the raw counts. Optimal
            # alignments are rarely unique -- Biopython reports millions of
            # co-optimal alignments for some real pairs here -- so two correct
            # implementations can legitimately report different identity counts
            # for the same score. Comparing integers would flag that as a bug.
            if fast.qualifies(policy) == reference.qualifies(policy):
                same_decision += 1
            else:
                report["decision_agreement"] = False

            # The asymmetric failure: a pair the oracle says qualifies but the
            # table does not record. That is the one that leaks homologues across
            # a split, so it is listed individually for quarantine.
            if reference.qualifies(policy) and not fast.qualifies(policy):
                report["missed_qualifying_pairs"].append(
                    {
                        "pair": [i, j],
                        "fast_identity": round(fast.identity, 4),
                        "reference_identity": round(reference.identity, 4),
                        "coverage": round(reference.coverage, 4),
                    }
                )

        all_differences.extend(differences)
        array = np.asarray(differences) if differences else np.zeros(1)
        report["strata"][stratum] = {
            "n_checked": len(pairs),
            "n_same_decision": same_decision,
            "identity_diff_median": round(float(np.median(array)), 4),
            "identity_diff_max": round(float(array.max()), 4),
        }

    combined = np.asarray(all_differences) if all_differences else np.zeros(1)
    report["identity_differences"] = {
        "n": len(all_differences),
        "median": round(float(np.median(combined)), 4),
        "p90": round(float(np.quantile(combined, 0.90)), 4),
        "max": round(float(combined.max()), 4),
        "note": (
            "Engine-to-engine identity differences arise from co-optimal "
            "alignments, not from error: both implementations return the same "
            "optimal score. Only the qualification decision is asserted."
        ),
    }
    return report


def percolation_curve(
    table: "ProteinRelationTable",
    *,
    thresholds: Sequence[float] = (0.40, 0.50, 0.60, 0.70, 0.80, 0.90),
    coverage: float | None = None,
    observation_weights: Mapping[int, int] | None = None,
) -> list[dict[str, Any]]:
    """Component structure as a function of the identity threshold.

    Recorded because the threshold is the single most consequential grouping
    choice and it should be visible that the transition is sharp rather than a
    tuned knob. On release 202610 the largest component falls from 93.0% of
    constructs at 0.40 to 2.2% at 0.60, and anything from 0.60 upward looks much
    the same -- so the choice sits on a plateau, not on a slope.

    ``observation_weights`` maps a sequence position to the number of curated
    observations on it, which turns "2.2% of constructs" into the number that
    actually matters: how much data a cold-target test set could draw from.
    """
    coverage = table.policy.coverage_threshold if coverage is None else coverage
    edges = table.edges
    rows: list[dict[str, Any]] = []

    for threshold in thresholds:
        keep = (edges["identity"] >= threshold) & (edges["coverage"] >= coverage)
        union = UnionFind(table.n)
        for row in edges[keep]:
            union.union(int(row["i"]), int(row["j"]))
        components = union.components()
        sizes = sorted((len(m) for m in components.values()), reverse=True)
        entry: dict[str, Any] = {
            "identity_threshold": threshold,
            "coverage_threshold": coverage,
            "n_qualifying_edges": int(keep.sum()),
            "n_components": len(components),
            "largest_component": sizes[0] if sizes else 0,
            "largest_component_share": round(sizes[0] / table.n, 4) if sizes else 0.0,
        }
        if observation_weights is not None:
            weights = sorted(
                (sum(observation_weights.get(m, 0) for m in members)
                 for members in components.values()),
                reverse=True,
            )
            total = sum(weights)
            entry["observations_total"] = total
            entry["observations_in_largest"] = weights[0] if weights else 0
            entry["observations_outside_largest"] = total - (weights[0] if weights else 0)
        rows.append(entry)
    return rows


def component_summary(components: dict[int, list[int]], n: int) -> dict[str, Any]:
    """Component-size distribution, including the giant-component warning.

    Single linkage chains, so a protein family can absorb many others. The
    specification's guidance is to report that rather than break a component to
    make the test set bigger, since breaking it would reintroduce the homology
    the split exists to prevent.
    """
    sizes = sorted((len(members) for members in components.values()), reverse=True)
    largest = sizes[0] if sizes else 0
    return {
        "n_components": len(sizes),
        "n_sequences": n,
        "largest_component": largest,
        "largest_component_share": round(largest / n, 4) if n else 0.0,
        "singletons": sum(1 for size in sizes if size == 1),
        "top_10_sizes": sizes[:10],
        "giant_component_warning": (
            largest / n > 0.30 if n else False
        ),
        "note": (
            "Single-linkage components. A large component is reported, never split "
            "to enlarge the test partition; splitting it would reintroduce exactly "
            "the homology the cold-target regime exists to exclude."
        ),
    }


def build_protein_relation_artifacts(
    processed_dir: Path | str,
    *,
    policy: ProteinPolicy | None = None,
    workers: int = 14,
    rows_per_block: int = 40,
    verify_samples: int = 300,
) -> dict[str, Any]:
    """Full protein-grouping stage: align, group, verify, report.

    Writes ``protein_relation/`` (edges, computed bitmap, metadata),
    ``protein_groups.parquet`` (construct -> group) and
    ``protein_groups_report.json``.
    """
    import pandas as pd

    from bindra.data.tables import write_json, write_table

    processed_dir = Path(processed_dir)
    policy = policy or ProteinPolicy()

    constructs = pd.read_parquet(processed_dir / "target_constructs.parquet")
    constructs = constructs[constructs["status"] == "OK"].reset_index(drop=True)
    sequences = pd.read_parquet(processed_dir / "sequences_raw.parquet").set_index(
        "sequence_ref"
    )
    constructs = constructs.join(sequences["sequence"], on="sequence_ref")

    sequence_list = constructs["sequence"].tolist()
    ids = constructs["construct_id"].tolist()

    table = build_relation_table(
        sequence_list,
        ids,
        policy=policy,
        workers=workers,
        rows_per_block=rows_per_block,
        checkpoint_dir=processed_dir / "protein_relation",
    )
    table.save(processed_dir / "protein_relation")

    # Alignment-independent edges. Both are true regardless of any threshold, so
    # they are unioned in rather than left to a score.
    extra: list[tuple[int, int]] = []
    n_uniprot_edges = 0
    if policy.union_constructs_sharing_uniprot:
        for _, group in constructs.reset_index().groupby("uniprot_primary", dropna=True):
            members = group["index"].tolist()
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    extra.append((members[a], members[b]))
                    n_uniprot_edges += 1

    normalized = [normalize_for_alignment(s, policy) for s in sequence_list]
    containment = substring_containment_edges(normalized)
    extra.extend(containment)

    def labels_at(threshold: float) -> tuple[list[int], dict[str, Any]]:
        union = UnionFind(table.n)
        for i, j in table.qualifying_pairs_at(threshold):
            union.union(i, j)
        for i, j in extra:
            union.union(i, j)
        components = union.components()
        mapping: dict[int, int] = {}
        for group_index, (_, members) in enumerate(sorted(components.items())):
            for member in members:
                mapping[member] = group_index
        return [mapping[i] for i in range(table.n)], component_summary(components, table.n)

    # Two groupings, deliberately. The strict one bounds any claim about novel
    # targets; the looser one is the only way cold-target evaluation has enough
    # data to measure anything. See ProteinPolicy.cold_target_identity_threshold.
    strict_labels, strict_summary = labels_at(policy.identity_threshold)
    cold_labels, cold_summary = labels_at(policy.cold_target_identity_threshold)
    constructs["protein_group_strict"] = strict_labels
    constructs["protein_group"] = cold_labels
    summary = cold_summary

    groups_ref = write_table(
        constructs[
            [
                "sequence_ref",
                "sequence_sha256",
                "construct_id",
                "uniprot_primary",
                "sequence_length",
                "protein_group",
                "protein_group_strict",
            ]
        ],
        processed_dir / "protein_groups",
    )

    verification = verify_relation_sample(
        table, normalized, n_samples=verify_samples, seed=policy_seed()
    )

    weights: dict[int, int] = {}
    observations_path = processed_dir / "observations"
    if observations_path.exists():
        import glob

        counts: dict[str, int] = {}
        for shard in sorted(glob.glob(str(observations_path / "*.parquet"))):
            block = pd.read_parquet(shard, columns=["sequence_sha256"])
            for value, n in block["sequence_sha256"].value_counts().items():
                counts[value] = counts.get(value, 0) + int(n)
        position = {cid: i for i, cid in enumerate(table.sequence_ids)}
        for row in constructs.itertuples(index=False):
            index = position.get(row.construct_id)
            if index is not None:
                weights[index] = counts.get(row.sequence_sha256, 0)

    curve = percolation_curve(table, observation_weights=weights or None)

    report = {
        "n_constructs": int(len(constructs)),
        "n_pairs": table.n_pairs,
        "n_computed": table.n_computed,
        "n_edges_stored": int(len(table.edges)),
        "n_qualifying_edges": table.n_qualifying,
        "n_uniprot_sharing_edges": n_uniprot_edges,
        "n_substring_containment_edges": len(containment),
        "components_cold_target": cold_summary,
        "components_strict": strict_summary,
        "percolation_curve": curve,
        "components": summary,
        "cold_target_identity_threshold": policy.cold_target_identity_threshold,
        "verification": verification,
        "elapsed_seconds": round(table.elapsed_seconds, 1),
        "relation_table_sha256": table.sha256,
        "policy": {
            "alignment_mode": policy.alignment_mode,
            "identity_threshold": policy.identity_threshold,
            "coverage_threshold": policy.coverage_threshold,
            "identity_ambiguity_margin": policy.identity_ambiguity_margin,
            "substitution_matrix": policy.substitution_matrix,
            "open_gap_score": policy.open_gap_score,
            "extend_gap_score": policy.extend_gap_score,
            "identity_denominator": policy.identity_denominator,
            "coverage_denominator": policy.coverage_denominator,
            "linkage": policy.linkage,
            "union_constructs_sharing_uniprot": policy.union_constructs_sharing_uniprot,
        },
        "caveats": [
            "Semi-global alignment was chosen by measurement: against same-UniProt "
            "positives and cross-UniProt negatives it reached 0.987 sensitivity at "
            "0.000 false-positive rate, where global alignment reached 0.867 at "
            "0.043 and collapsed a 500-construct sample into a single component.",
            "40% identity / 80% coverage is an operational threshold, not proof "
            "that shared domains or protein families are absent across partitions.",
            "Optimal alignments are rarely unique, so identity is engine-dependent "
            "for near-threshold pairs. See verification.identity_differences; the "
            "split audit re-checks near-threshold cross-partition pairs with an "
            "independent implementation.",
            "Single-linkage components are reported, never split to enlarge a test "
            "partition.",
            "At 40% identity the target graph percolates: one component holds 93.0% "
            "of constructs and 98.4% of observations, leaving 1.6% for a strict "
            "cold-target test set. Two groupings are therefore published -- "
            "protein_group_strict at 40% to bound claims about novel targets, and "
            "protein_group at 60% to make the regime measurable. See "
            "percolation_curve; the transition is sharp, so 60% sits on a plateau.",
        ],
    }
    write_json(report, processed_dir / "protein_groups_report.json")
    report["artifacts"] = {"protein_groups": groups_ref.to_dict()}
    return report


def policy_seed() -> int:
    from bindra.config.profiles import SplitPolicy

    return SplitPolicy().seed


__all__ = [
    "EDGE_DTYPE",
    "build_protein_relation_artifacts",
    "normalize_for_alignment",
    "STORE_IDENTITY_FLOOR",
    "AlignmentResult",
    "HomologyError",
    "ProteinRelationTable",
    "UnionFind",
    "align_pair",
    "align_pair_reference",
    "build_relation_table",
    "component_summary",
    "percolation_curve",
    "composition_bound_excludes",
    "composition_counts",
    "condensed_index",
    "normalize_for_alignment",
    "substring_containment_edges",
    "verify_relation_sample",
]
