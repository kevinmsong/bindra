"""Activity-cliff panel definition and its diagnostic categories.

Two properties carry the weight. The magnitude threshold is derived from measured
replicate noise per endpoint, so a pKd "cliff" is not allowed to sit inside
measurement error. And ``cliff_kind`` separates mechanisms that an
ECFP4-equivalent representation cannot distinguish from ones it can -- on this
corpus that is 49% of cliffs, so pooling them would blame a model for a question
its features cannot answer.
"""

from __future__ import annotations

import pandas as pd
import pytest

from bindra.config.profiles import get_profile
from bindra.data.endpoints import Endpoint
from bindra.eval.benchmarks.activity_cliff import (
    CliffDefinition,
    _cliff_kind,
    required_test_drug_groups,
)


class TestDefinition:
    def test_every_choice_is_pinned_and_hashable(self):
        definition = CliffDefinition()
        payload = definition.to_dict()
        for key in (
            "fingerprint",
            "radius",
            "n_bits",
            "use_chirality",
            "tanimoto_threshold",
            "generic_scaffold_threshold",
            "delta_sigma_multiple",
            "delta_floor",
            "comparability_tier",
            "series_unit",
        ):
            assert key in payload

    def test_threshold_is_endpoint_specific_and_noise_aware(self):
        """A flat 1 p-unit cliff in pKd is ~1.06 sigma -- inside the noise."""
        profile = get_profile()
        assert profile.cliff_delta_threshold(Endpoint.KD) == pytest.approx(1.89, abs=0.01)
        assert profile.cliff_delta_threshold(Endpoint.KI) == pytest.approx(1.282, abs=0.01)
        assert profile.cliff_delta_threshold(Endpoint.KD) > profile.cliff_delta_threshold(
            Endpoint.KI
        )

    def test_threshold_never_drops_below_the_literature_convention(self):
        from bindra.config.profiles import ErrorModel

        quiet = ErrorModel(sigma_rep={"Kd": 0.05, "Ki": 0.05, "IC50": 0.05})
        profile = get_profile().with_overrides(error_model=quiet)
        assert profile.cliff_delta_threshold(Endpoint.KD) == pytest.approx(1.0)

    def test_censored_members_are_excluded_by_definition(self):
        assert CliffDefinition().require_exact_both is True

    def test_series_are_preassigned_to_test(self):
        """Selecting pairs that happen to land in test would be post-hoc."""
        definition = CliffDefinition()
        assert definition.preassign_series_to_test is True
        assert definition.series_unit == "drug_group"


class TestCliffKind:
    def test_identical_fingerprints_with_one_parent_are_stereochemical(self):
        assert (
            _cliff_kind(1.0, "LEAK1", "LEAK1", CliffDefinition()) == "stereochemical"
        )

    def test_identical_fingerprints_with_two_parents_are_fingerprint_blind(self):
        """Homologues differing in chain length, ring size or linker length.

        ECFP4 at radius 2 cannot see those differences, so the pair scores 1.0
        while being genuinely different molecules.
        """
        assert (
            _cliff_kind(1.0, "LEAK1", "LEAK2", CliffDefinition()) == "fingerprint_blind"
        )

    def test_ordinary_analogue_pairs_are_substitutions(self):
        assert _cliff_kind(0.92, "LEAK1", "LEAK2", CliffDefinition()) == "substitution"

    def test_the_identity_threshold_is_configurable(self):
        loose = CliffDefinition(fingerprint_identical_threshold=0.90)
        assert _cliff_kind(0.95, "A", "A", loose) == "stereochemical"
        assert _cliff_kind(0.95, "A", "A", CliffDefinition()) == "substitution"

    def test_a_missing_leakage_key_does_not_claim_stereochemistry(self):
        assert _cliff_kind(1.0, None, None, CliffDefinition()) == "fingerprint_blind"


class TestSeriesHoldout:
    def test_both_members_series_are_required_in_test(self):
        """Holding out only the two molecules would leave their series in training."""
        candidates = pd.DataFrame(
            {
                "label": ["cliff", "cliff", "noncliff_control"],
                "identity_key_a": ["a1", "b1", "c1"],
                "identity_key_b": ["a2", "b2", "c2"],
            }
        )
        drug_group_of = {"a1": 10, "a2": 11, "b1": 12, "b2": 12, "c1": 99, "c2": 98}
        groups = required_test_drug_groups(candidates, drug_group_of)
        assert groups == frozenset({10, 11, 12})
        # Controls do not force a holdout.
        assert 99 not in groups

    def test_unknown_molecules_are_skipped_rather_than_guessed(self):
        candidates = pd.DataFrame(
            {"label": ["cliff"], "identity_key_a": ["x"], "identity_key_b": ["y"]}
        )
        assert required_test_drug_groups(candidates, {"x": 5}) == frozenset({5})


@pytest.fixture(scope="module")
def candidates():
    """Module-scoped so the corpus scan happens once, not per test."""
    from pathlib import Path

    processed = Path("data/processed/202610")
    if not (processed / "observations").exists():
        pytest.skip("curated corpus not present")
    from bindra.eval.benchmarks.activity_cliff import build_cliff_candidates

    return build_cliff_candidates(processed, max_groups=600)


@pytest.mark.snapshot
class TestAgainstTheCuratedCorpus:
    """Runs only where the curated corpus exists."""

    def test_every_cliff_clears_its_endpoint_threshold(self, candidates):
        frame, _ = candidates
        cliffs = frame[frame["label"] == "cliff"]
        if not len(cliffs):
            pytest.skip("no cliffs in this sample")
        assert bool((cliffs["abs_delta_p"] >= cliffs["cliff_threshold"]).all())

    def test_controls_stay_inside_the_non_cliff_band(self, candidates):
        frame, _ = candidates
        controls = frame[frame["label"] == "noncliff_control"]
        if not len(controls):
            pytest.skip("no controls in this sample")
        assert bool((controls["abs_delta_p"] <= 0.5).all())

    def test_pairs_share_a_comparability_group_and_endpoint(self, candidates):
        frame, _ = candidates
        if not len(frame):
            pytest.skip("no candidates")
        assert frame["comparability_group_id"].notna().all()
        assert frame["endpoint"].notna().all()

    def test_cliff_kinds_are_assigned(self, candidates):
        frame, report = candidates
        if not report.n_cliffs:
            pytest.skip("no cliffs in this sample")
        assert set(report.cliff_kinds) <= {
            "stereochemical",
            "fingerprint_blind",
            "substitution",
        }
