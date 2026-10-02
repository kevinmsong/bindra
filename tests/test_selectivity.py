"""Selectivity panel: unknowns are not negatives, censored sides stay intervals.

The two properties that carry real risk of a wrong scientific claim are
``test_a_missing_countertarget_is_unknown_not_a_negative`` and
``test_a_censored_side_yields_an_interval_not_a_point``. Both are ways to
manufacture selectivity that is not in the data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from bindra.eval.benchmarks.selectivity import (
    SelectivityDefinition,
    build_selectivity_triples,
    load_seed_pairs,
    related_target_pairs,
)

TARGET_A = "a" * 64
TARGET_B = "b" * 64
TARGET_C = "c" * 64


def _constructs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_sha256": [TARGET_A, TARGET_B, TARGET_C],
            "construct_id": ["P1:aaaa", "P2:bbbb", "P3:cccc"],
            "uniprot_primary": ["P23219", "P35354", "P99999"],
        }
    )


def _observation(
    observation_id: str,
    identity_key: str,
    target: str,
    *,
    p_value: float = 7.0,
    endpoint: str = "Ki",
    group: str = "G1",
    is_exact: bool = True,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    citation: str = "PMID:1",
    condition: str = "pH?|T?",
    source_class: str = "literature_curated",
) -> dict:
    return {
        "observation_id": observation_id,
        "identity_key": identity_key,
        "sequence_sha256": target,
        "endpoint": endpoint,
        "p_value": p_value,
        "is_exact": is_exact,
        "is_censored": not is_exact,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "comparability_group_id": group,
        "citation_key": citation,
        "assay_condition_bucket": condition,
        "curation_source_class": source_class,
        "ic50_tier": "na",
    }


class TestSeedPairs:
    def test_seed_file_loads_with_a_hash(self):
        pairs, sha = load_seed_pairs()
        assert len(pairs) >= 10
        assert len(sha) == 64
        for entry in pairs:
            assert {"a", "b", "label"} <= set(entry)

    def test_a_wrong_accession_matches_nothing_rather_than_inventing_a_pair(self):
        """The safe failure mode: a bad entry misses a pair, never fabricates one."""
        constructs = _constructs()
        frame, report = related_target_pairs(
            constructs,
            relation=None,
            seed_pairs=[{"a": "NOT_A_REAL_ACCESSION", "b": "P23219", "label": "bogus"}],
        )
        assert report.n_seed_pairs_listed == 1
        assert report.n_seed_pairs_matched == 0
        assert len(frame) == 0

    def test_a_matching_seed_pair_produces_a_relationship(self):
        frame, report = related_target_pairs(
            _constructs(),
            relation=None,
            seed_pairs=[{"a": "P23219", "b": "P35354", "label": "COX-1 / COX-2"}],
        )
        assert report.n_seed_pairs_matched == 1
        assert len(frame) == 1
        assert frame["evidence"].iloc[0] == "curated_seed"

    def test_pairs_are_canonically_ordered_so_a_b_and_b_a_are_one(self):
        frame, _ = related_target_pairs(
            _constructs(),
            relation=None,
            seed_pairs=[
                {"a": "P23219", "b": "P35354", "label": "forward"},
                {"a": "P35354", "b": "P23219", "label": "forward"},
            ],
        )
        assert len(frame) == 1


class TestTriples:
    def _related(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "target_a": TARGET_A,
                    "target_b": TARGET_B,
                    "evidence": "curated_seed",
                    "identity": None,
                    "coverage": None,
                    "label": "COX-1 / COX-2",
                }
            ]
        )

    def test_a_ligand_measured_against_both_gives_a_selectivity_value(self):
        observations = pd.DataFrame(
            [
                _observation("o1", "L1", TARGET_A, p_value=8.0),
                _observation("o2", "L1", TARGET_B, p_value=6.0),
            ]
        )
        exact, censored, unknowns, report = build_selectivity_triples(
            observations, self._related()
        )
        forward = exact[~exact["swapped"]]
        assert len(forward) == 1
        assert forward["selectivity"].iloc[0] == pytest.approx(2.0)
        assert len(censored) == 0
        assert len(unknowns) == 0

    def test_every_triple_has_a_swapped_control_with_the_negated_label(self):
        observations = pd.DataFrame(
            [
                _observation("o1", "L1", TARGET_A, p_value=8.0),
                _observation("o2", "L1", TARGET_B, p_value=6.0),
            ]
        )
        exact, _, _, _ = build_selectivity_triples(observations, self._related())
        assert len(exact) == 2
        swapped = exact[exact["swapped"]].iloc[0]
        forward = exact[~exact["swapped"]].iloc[0]
        assert swapped["selectivity"] == pytest.approx(-forward["selectivity"])
        assert swapped["target"] == forward["countertarget"]

    def test_a_missing_countertarget_is_unknown_not_a_negative(self):
        """The error that manufactures selectivity out of absence.

        A compound measured against one target and simply not tested against the
        other says nothing about activity there, and must never be scored as
        inactive on the countertarget.
        """
        observations = pd.DataFrame([_observation("o1", "L1", TARGET_A, p_value=8.0)])
        exact, censored, unknowns, report = build_selectivity_triples(
            observations, self._related()
        )
        assert len(exact) == 0
        assert len(censored) == 0
        assert len(unknowns) == 1
        row = unknowns.iloc[0]
        assert row["measured_target"] == TARGET_A
        assert row["unmeasured_target"] == TARGET_B
        # No column anywhere carries a numeric selectivity for this row.
        assert "selectivity" not in unknowns.columns
        assert report.n_unknowns == 1

    def test_a_censored_side_yields_an_interval_not_a_point(self):
        """'More than 100-fold selective' is real, and is not collapsed."""
        observations = pd.DataFrame(
            [
                _observation("o1", "L1", TARGET_A, p_value=9.0),
                _observation(
                    "o2",
                    "L1",
                    TARGET_B,
                    p_value=5.0,
                    is_exact=False,
                    upper_bound=5.0,
                ),
            ]
        )
        exact, censored, _, report = build_selectivity_triples(
            observations, self._related()
        )
        assert len(exact) == 0
        assert len(censored) == 1
        row = censored.iloc[0]
        # p(target) is 9.0 exactly; p(counter) is at most 5.0, so the difference is
        # at least 4.0 and unbounded above.
        assert row["selectivity_lower_bound"] == pytest.approx(4.0)
        assert row["selectivity_upper_bound"] is None or pd.isna(
            row["selectivity_upper_bound"]
        )
        assert report.n_triples_censored == 1

    def test_measurements_from_different_papers_are_rejected_at_tier_one(self):
        """Tier 1 means one selectivity table in one paper."""
        observations = pd.DataFrame(
            [
                _observation("o1", "L1", TARGET_A, p_value=8.0, citation="PMID:111"),
                _observation("o2", "L1", TARGET_B, p_value=6.0, citation="PMID:222"),
            ]
        )
        exact, _, _, report = build_selectivity_triples(observations, self._related())
        assert len(exact) == 0
        assert report.rejected.get("different_assay_panel") == 1

    def test_different_assay_conditions_are_rejected_at_tier_one(self):
        observations = pd.DataFrame(
            [
                _observation("o1", "L1", TARGET_A, p_value=8.0, condition="pH7.5|T2"),
                _observation("o2", "L1", TARGET_B, p_value=6.0, condition="pH5|T2"),
            ]
        )
        exact, _, _, report = build_selectivity_triples(observations, self._related())
        assert len(exact) == 0
        assert report.rejected.get("different_assay_panel") == 1

    def test_the_target_itself_is_not_part_of_the_panel_key(self):
        """The trap this module hit: including the target makes tier 1 impossible.

        ``comparability_group_id`` is keyed on the target hash, so a selectivity
        pair -- which by definition spans two targets -- can never share one.
        Comparing those ids rejected all 2,391,516 candidate pairs on the real
        corpus and produced an empty panel.
        """
        observations = pd.DataFrame(
            [
                _observation("o1", "L1", TARGET_A, p_value=8.0, group="group-for-A"),
                _observation("o2", "L1", TARGET_B, p_value=6.0, group="group-for-B"),
            ]
        )
        exact, _, _, report = build_selectivity_triples(observations, self._related())
        # Same paper, same conditions, different targets -> a real triple.
        assert len(exact) == 2
        assert not report.rejected.get("different_assay_panel")

    def test_different_endpoints_never_form_a_triple(self):
        observations = pd.DataFrame(
            [
                _observation("o1", "L1", TARGET_A, p_value=8.0, endpoint="Ki"),
                _observation("o2", "L1", TARGET_B, p_value=6.0, endpoint="Kd"),
            ]
        )
        exact, _, unknowns, _ = build_selectivity_triples(observations, self._related())
        assert len(exact) == 0
        # Each becomes an unknown for its own endpoint, not a cross-endpoint delta.
        assert set(unknowns["endpoint"]) <= {"Ki", "Kd"}

    def test_an_unrelated_target_never_enters_a_selectivity_quantity(self):
        """Measuring against an unrelated protein is not a selectivity datum.

        The related pair here is (A, B). A ligand measured against A and against
        the unrelated C is correctly an *unknown* for the A/B question -- it was
        never tested against B -- but C must never appear as a target or
        countertarget, and no selectivity value may be formed from it.
        """
        observations = pd.DataFrame(
            [
                _observation("o1", "L1", TARGET_A, p_value=8.0),
                _observation("o2", "L1", TARGET_C, p_value=6.0),
            ]
        )
        exact, censored, unknowns, _ = build_selectivity_triples(
            observations, self._related()
        )
        assert len(exact) == 0
        assert len(censored) == 0
        # The A/B question is open for this ligand, which is what unknowns record.
        assert len(unknowns) == 1
        assert unknowns["unmeasured_target"].iloc[0] == TARGET_B
        # C appears nowhere: it is not part of any related pair.
        for frame in (exact, censored, unknowns):
            if len(frame):
                for column in frame.columns:
                    assert TARGET_C not in set(frame[column].astype(str))

    def test_a_ligand_measured_against_neither_target_produces_nothing(self):
        observations = pd.DataFrame([_observation("o1", "L1", TARGET_C, p_value=6.0)])
        exact, censored, unknowns, _ = build_selectivity_triples(
            observations, self._related()
        )
        assert len(exact) == 0 and len(censored) == 0 and len(unknowns) == 0


class TestDefinition:
    def test_relatedness_is_looser_than_the_split_threshold(self):
        """Selectivity arises between proteins similar enough to share a ligand.

        That is a weaker relationship than the 40%/80% used to keep partitions
        disjoint, so using the split threshold here would miss most real pairs.
        """
        from bindra.config.profiles import get_profile

        definition = SelectivityDefinition()
        assert definition.related_identity_min < get_profile().proteins.identity_threshold
        assert definition.related_coverage_min < get_profile().proteins.coverage_threshold

    def test_baselines_are_declared_in_the_definition(self):
        assert set(SelectivityDefinition().declared_baselines) == {
            "ligand_only_mean",
            "constant_zero",
        }

    def test_swapped_controls_are_on_by_default(self):
        assert SelectivityDefinition().include_swapped_controls is True
