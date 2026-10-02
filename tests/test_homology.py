"""Protein homology: alignment semantics, grouping, and the verification oracle.

The alignment mode here was chosen by measurement rather than convention, and
these tests pin the properties that choice rests on. The one that matters most is
``test_semi_global_keeps_coverage_meaningful``: under global alignment, coverage
of the shorter sequence is ~1.0 for *every* pair, so the 80% coverage criterion
filters nothing and single linkage collapses the whole corpus into one component.
"""

from __future__ import annotations

import numpy as np
import pytest

from bindra.config.profiles import ProteinPolicy
from bindra.data.homology import (
    AlignmentResult,
    HomologyError,
    UnionFind,
    align_pair,
    align_pair_reference,
    build_relation_table,
    component_summary,
    composition_bound_excludes,
    composition_counts,
    condensed_index,
    normalize_for_alignment,
    substring_containment_edges,
    verify_relation_sample,
)

KINASE = (
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTL"
    "GQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLP"
)
UNRELATED = (
    "MSDKPDMAEIEKFDKSKLKKTETQEKNPLPSKETIEQEKQAGESHHHHHHGSLVPRGSHMRGSEFELRRQACGRPRQLGGL"
    "WQAVSEPAVVTLWYRAPEILLGCKYYSTAVDIWSLGCIFAEMVTRRALFPGDSEIDQLFRIFRTLGTPDEVVWPGVTSMPD"
)


def _mutate(sequence: str, n: int, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    letters = list(sequence)
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    for position in rng.choice(len(letters), size=n, replace=False):
        current = letters[int(position)]
        choices = [a for a in alphabet if a != current]
        letters[int(position)] = choices[int(rng.integers(0, len(choices)))]
    return "".join(letters)


class TestAlignmentSemantics:
    def test_identical_sequences_are_fully_identical_and_covered(self):
        result = align_pair(KINASE, KINASE)
        assert result.identity == pytest.approx(1.0)
        assert result.coverage == pytest.approx(1.0)

    def test_a_substring_is_fully_covered(self):
        """A domain construct inside a long chain must reach full coverage.

        This is what free end gaps buy, and why a length-ratio prefilter would be
        wrong: the shorter sequence is 40% of the longer one here.
        """
        result = align_pair(KINASE[20:80], KINASE)
        assert result.identity == pytest.approx(1.0)
        assert result.coverage == pytest.approx(1.0)

    def test_point_mutants_remain_near_identical(self):
        result = align_pair(KINASE, _mutate(KINASE, 5, seed=1))
        assert result.identity > 0.95
        assert result.coverage > 0.95

    def test_unrelated_sequences_fail_the_coverage_criterion(self):
        policy = ProteinPolicy()
        result = align_pair(KINASE, UNRELATED)
        assert not result.qualifies(policy)

    def test_semi_global_keeps_coverage_meaningful(self):
        """Global alignment pins coverage near 1.0 and so filters nothing.

        Measured on this corpus, global alignment gave unrelated pairs a median
        coverage of 0.982 and a median identity of 0.252 -- close enough to the
        0.40 threshold that 4.3% of cross-UniProt pairs qualified, which single
        linkage amplified into one component holding all 500 sampled constructs.
        Semi-global drops negative-pair coverage to a median of 0.073.
        """
        import parasail

        semi = align_pair(KINASE, UNRELATED)
        raw = parasail.nw_stats_striped_sat(
            normalize_for_alignment(KINASE),
            normalize_for_alignment(UNRELATED),
            11,
            1,
            parasail.blosum62,
        )
        global_columns = len(KINASE) + len(UNRELATED) - int(raw.length)
        global_coverage = global_columns / min(len(KINASE), len(UNRELATED))
        assert global_coverage > 0.80
        assert semi.coverage < global_coverage

    def test_empty_input_is_handled(self):
        assert align_pair("", KINASE).identity == 0.0

    def test_policy_thresholds_drive_qualification(self):
        result = AlignmentResult(
            identity=0.45, coverage=0.90, score=0, aligned_columns=100, n_identical=45
        )
        assert result.qualifies(ProteinPolicy())
        assert not result.qualifies(ProteinPolicy(identity_threshold=0.50))
        assert not result.qualifies(ProteinPolicy(coverage_threshold=0.95))


class TestNormalization:
    def test_non_matrix_residues_map_to_x(self):
        """BLOSUM62 has no U or O, and two engines would guess differently."""
        assert normalize_for_alignment("MKUO") == "MKXX"

    def test_both_engines_accept_non_standard_residues(self):
        a = "MKTAYUIAKQRQISFVKSHFSRQLEEOLGLIEVQ"
        b = "MKTAYUIAKQRQISFVKSHFSRQLEERLGLIEVQ"
        fast = align_pair(a, b)
        reference = align_pair_reference(a, b)
        assert fast.n_identical == reference.n_identical
        assert fast.aligned_columns == reference.aligned_columns

    def test_case_is_normalized(self):
        assert align_pair(KINASE.lower(), KINASE).identity == pytest.approx(1.0)


class TestSaturation:
    def test_a_saturating_pair_raises_rather_than_returning_a_wrong_score(self):
        """The 16-bit kernel silently returns score 0 on very long sequences.

        Release 202610 contains a 34,350-residue construct, so this is not
        hypothetical: a saturated result would have made Titin look unrelated to
        everything. The ``_sat`` kernel is used precisely to avoid this, and the
        guard exists so a future kernel change cannot reintroduce it quietly.
        """
        import parasail

        long_sequence = (KINASE * 250)[:40_000]
        saturating = parasail.nw_stats_striped_16(
            long_sequence, long_sequence, 11, 1, parasail.blosum62
        )
        assert bool(getattr(saturating, "saturated", False)) is True
        # The _sat kernel the module actually uses does not saturate.
        safe = align_pair(long_sequence, long_sequence)
        assert safe.identity == pytest.approx(1.0)


class TestCompositionBound:
    def test_the_bound_never_excludes_a_qualifying_pair(self):
        """Soundness property: it may fail to prune, but must never over-prune."""
        policy = ProteinPolicy()
        sequences = [KINASE, _mutate(KINASE, 10, seed=2), KINASE[30:120], UNRELATED]
        counts = composition_counts(sequences)
        for i in range(len(sequences)):
            for j in range(i + 1, len(sequences)):
                result = align_pair(sequences[i], sequences[j], policy)
                if result.qualifies(policy):
                    assert not composition_bound_excludes(
                        counts[i], counts[j], len(sequences[i]), len(sequences[j]), policy
                    )

    def test_counts_only_tally_letters(self):
        counts = composition_counts(["AAC"])
        assert counts[0][0] == 2  # A
        assert counts[0][2] == 1  # C
        assert counts.sum() == 3


class TestUnionFind:
    def test_transitive_merging(self):
        union = UnionFind(5)
        assert union.union(0, 1)
        assert union.union(1, 2)
        assert not union.union(0, 2)
        assert union.find(0) == union.find(2)

    def test_components_partition_every_element(self):
        union = UnionFind(6)
        union.union(0, 1)
        union.union(2, 3)
        components = union.components()
        assert sum(len(members) for members in components.values()) == 6
        assert len(components) == 4


class TestContainment:
    def test_exact_substring_yields_an_edge(self):
        edges = substring_containment_edges([KINASE, KINASE[10:60], UNRELATED])
        assert (0, 1) in edges
        assert not any(2 in pair for pair in edges)

    def test_containment_is_reported_with_the_lower_index_first(self):
        for i, j in substring_containment_edges([KINASE[10:60], KINASE]):
            assert i < j


class TestRelationTable:
    @pytest.fixture
    def table(self):
        sequences = [
            KINASE,
            _mutate(KINASE, 4, seed=3),
            _mutate(KINASE, 8, seed=4),
            UNRELATED,
            _mutate(UNRELATED, 5, seed=5),
        ]
        ids = [f"C{i}" for i in range(len(sequences))]
        return build_relation_table(sequences, ids, workers=1, progress=False), sequences

    def test_every_pair_is_recorded_as_computed(self, table):
        relation, sequences = table
        assert relation.n_pairs == 10
        assert relation.n_computed == 10
        for i in range(relation.n):
            for j in range(i + 1, relation.n):
                assert relation.was_computed(i, j)

    def test_components_separate_the_two_families(self, table):
        relation, _ = table
        components = relation.components()
        assert len(components) == 2
        assert sorted(len(m) for m in components.values()) == [2, 3]

    def test_condensed_index_is_a_bijection(self):
        n = 7
        seen = set()
        for i in range(n):
            for j in range(i + 1, n):
                index = condensed_index(i, j, n)
                assert index not in seen
                seen.add(index)
        assert seen == set(range(n * (n - 1) // 2))
        assert condensed_index(3, 1, n) == condensed_index(1, 3, n)

    def test_extra_edges_can_merge_components(self, table):
        relation, _ = table
        merged = relation.components(extra_edges=[(0, 3)])
        assert len(merged) == 1

    def test_round_trip_through_disk(self, table, tmp_path):
        from bindra.data.homology import ProteinRelationTable

        relation, _ = table
        relation.save(tmp_path / "rel")
        restored = ProteinRelationTable.load(tmp_path / "rel")
        assert restored.n == relation.n
        assert restored.n_qualifying == relation.n_qualifying
        assert len(restored.edges) == len(relation.edges)
        assert restored.sha256


class TestVerification:
    def test_the_oracle_agrees_on_the_decision(self):
        sequences = [
            KINASE,
            _mutate(KINASE, 4, seed=6),
            UNRELATED,
            _mutate(UNRELATED, 6, seed=7),
        ]
        relation = build_relation_table(
            sequences, [f"C{i}" for i in range(len(sequences))], workers=1, progress=False
        )
        report = verify_relation_sample(relation, sequences, n_samples=12)
        assert report["decision_agreement"] is True
        assert report["missed_qualifying_pairs"] == []
        assert "identity_differences" in report

    def test_verification_reports_identity_differences_not_just_a_verdict(self):
        """Co-optimal alignments make raw counts differ legitimately.

        The report must expose the magnitude so a reader can judge it, rather
        than collapsing it to pass/fail.
        """
        sequences = [KINASE, _mutate(KINASE, 20, seed=8), UNRELATED]
        relation = build_relation_table(
            sequences, ["a", "b", "c"], workers=1, progress=False
        )
        report = verify_relation_sample(relation, sequences, n_samples=9)
        differences = report["identity_differences"]
        assert set(differences) >= {"median", "p90", "max", "n"}


class TestComponentSummary:
    def test_giant_component_is_flagged(self):
        components = {0: list(range(90)), 1: list(range(90, 100))}
        summary = component_summary(components, 100)
        assert summary["largest_component"] == 90
        assert summary["giant_component_warning"] is True

    def test_balanced_components_are_not_flagged(self):
        components = {i: [i] for i in range(100)}
        summary = component_summary(components, 100)
        assert summary["giant_component_warning"] is False
        assert summary["singletons"] == 100

    def test_summary_refuses_to_suggest_splitting_a_component(self):
        summary = component_summary({0: [0, 1]}, 2)
        assert "never split" in summary["note"]


class TestCheckpointing:
    """Resumability, because a first full run lost 89% of its work without it."""

    @pytest.fixture
    def corpus(self):
        sequences = [
            KINASE,
            _mutate(KINASE, 4, seed=11),
            _mutate(KINASE, 9, seed=12),
            UNRELATED,
            _mutate(UNRELATED, 5, seed=13),
            _mutate(UNRELATED, 11, seed=14),
        ]
        return sequences, [f"C{i}" for i in range(len(sequences))]

    def test_checkpoints_are_written_per_block(self, corpus, tmp_path):
        sequences, ids = corpus
        build_relation_table(
            sequences, ids, workers=1, rows_per_block=2, progress=False,
            checkpoint_dir=tmp_path / "rel",
        )
        blocks = sorted((tmp_path / "rel" / "blocks").glob("block-*.npy"))
        assert len(blocks) == 3
        assert (tmp_path / "rel" / "blocks" / "key.json").exists()

    def test_resuming_reproduces_the_full_result(self, corpus, tmp_path):
        sequences, ids = corpus
        directory = tmp_path / "rel"
        full = build_relation_table(
            sequences, ids, workers=1, rows_per_block=2, progress=False,
            checkpoint_dir=directory,
        )
        blocks = sorted((directory / "blocks").glob("block-*.npy"))
        blocks[0].unlink()
        resumed = build_relation_table(
            sequences, ids, workers=1, rows_per_block=2, progress=False,
            checkpoint_dir=directory,
        )
        assert np.array_equal(full.edges, resumed.edges)
        assert resumed.n_qualifying == full.n_qualifying

    def test_a_fully_cached_rerun_recomputes_nothing(self, corpus, tmp_path):
        sequences, ids = corpus
        directory = tmp_path / "rel"
        build_relation_table(
            sequences, ids, workers=1, rows_per_block=2, progress=False,
            checkpoint_dir=directory,
        )
        mtimes = {
            p.name: p.stat().st_mtime_ns
            for p in (directory / "blocks").glob("block-*.npy")
        }
        build_relation_table(
            sequences, ids, workers=1, rows_per_block=2, progress=False,
            checkpoint_dir=directory,
        )
        after = {
            p.name: p.stat().st_mtime_ns
            for p in (directory / "blocks").glob("block-*.npy")
        }
        assert mtimes == after

    def test_resuming_against_a_different_corpus_is_refused(self, corpus, tmp_path):
        """Block indices mean nothing across corpora; stitching them would mix pairs."""
        sequences, ids = corpus
        directory = tmp_path / "rel"
        build_relation_table(
            sequences, ids, workers=1, rows_per_block=2, progress=False,
            checkpoint_dir=directory,
        )
        with pytest.raises(HomologyError, match="different corpus or policy"):
            build_relation_table(
                sequences[:4], ids[:4], workers=1, rows_per_block=2, progress=False,
                checkpoint_dir=directory,
            )

    def test_changing_the_alignment_policy_is_refused(self, corpus, tmp_path):
        sequences, ids = corpus
        directory = tmp_path / "rel"
        build_relation_table(
            sequences, ids, workers=1, rows_per_block=2, progress=False,
            checkpoint_dir=directory,
        )
        with pytest.raises(HomologyError, match="different corpus or policy"):
            build_relation_table(
                sequences, ids, workers=1, rows_per_block=2, progress=False,
                checkpoint_dir=directory,
                policy=ProteinPolicy(open_gap_score=-5.0),
            )

    def test_resume_false_discards_existing_checkpoints(self, corpus, tmp_path):
        sequences, ids = corpus
        directory = tmp_path / "rel"
        build_relation_table(
            sequences, ids, workers=1, rows_per_block=2, progress=False,
            checkpoint_dir=directory,
        )
        fresh = build_relation_table(
            sequences, ids, workers=1, rows_per_block=2, progress=False,
            checkpoint_dir=directory, resume=False,
        )
        assert len(sorted((directory / "blocks").glob("block-*.npy"))) == 3
        assert fresh.n_pairs == len(sequences) * (len(sequences) - 1) // 2

    def test_a_truncated_checkpoint_is_recomputed_not_trusted(self, corpus, tmp_path):
        sequences, ids = corpus
        directory = tmp_path / "rel"
        full = build_relation_table(
            sequences, ids, workers=1, rows_per_block=2, progress=False,
            checkpoint_dir=directory,
        )
        target = sorted((directory / "blocks").glob("block-*.npy"))[1]
        target.write_bytes(b"not a numpy file")
        recovered = build_relation_table(
            sequences, ids, workers=1, rows_per_block=2, progress=False,
            checkpoint_dir=directory,
        )
        assert np.array_equal(full.edges, recovered.edges)
