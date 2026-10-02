"""Policy objects, config hashing, and thresholds derived from measured noise.

The point of these tests is that the curation thresholds are *computed from the
measured error model*, not written down as constants. The specification is
explicit that its starting 0.5 p-unit ranking margin is "a starting
hyperparameter, not a universal assay-error threshold", and on BindingDB 202610
that value turns out to sit below the noise floor of every endpoint.
"""

from __future__ import annotations

import json

import pytest

from bindra.config.profiles import (
    CurationProfile,
    ErrorModel,
    StandardizationPolicy,
    canonical_json,
    config_hash,
    get_profile,
)
from bindra.data.endpoints import Endpoint


@pytest.fixture
def profile() -> CurationProfile:
    return get_profile()


class TestCanonicalJson:
    def test_key_order_does_not_affect_the_serialization(self):
        assert canonical_json({"a": 1, "b": 2}) == canonical_json({"b": 2, "a": 1})

    def test_hash_is_stable_across_calls(self, profile):
        assert profile.profile_hash == get_profile().profile_hash

    def test_hash_changes_when_any_policy_field_changes(self, profile):
        changed = profile.with_overrides(plausible_range=(-2.0, 20.0))
        assert changed.profile_hash != profile.profile_hash

    def test_hash_changes_on_a_nested_policy_change(self, profile):
        changed = profile.with_overrides(
            standardization=StandardizationPolicy(canonicalize_tautomers=True)
        )
        assert changed.profile_hash != profile.profile_hash

    def test_profile_serializes_to_json(self, profile):
        payload = json.loads(profile.to_json())
        assert payload["snapshot_release"] == "202610"
        for section in ("standardization", "proteins", "comparability", "dedup", "ic50"):
            assert section in payload


class TestErrorModel:
    def test_measured_sigma_is_available_per_endpoint(self):
        model = ErrorModel()
        assert model.sigma(Endpoint.KD) == pytest.approx(0.945)
        assert model.sigma(Endpoint.KI) == pytest.approx(0.641)
        assert model.sigma(Endpoint.IC50) == pytest.approx(0.671)

    def test_kd_is_the_noisiest_endpoint(self):
        """Counterintuitive but measured: pKd is noisier than pKi, and scarcer."""
        model = ErrorModel()
        assert model.sigma(Endpoint.KD) > model.sigma(Endpoint.KI)
        assert model.sigma(Endpoint.KD) > model.sigma(Endpoint.IC50)

    def test_patent_sigma_is_lower_than_literature_sigma(self):
        """Not better data: re-statement of one value, so less independent."""
        model = ErrorModel()
        for endpoint in ("Kd", "Ki", "IC50"):
            tiers = model.sigma_rep_by_tier[endpoint]
            assert tiers["patent"] < tiers["literature"]

    def test_repeatability_limit_uses_the_iso_factor(self):
        model = ErrorModel()
        assert model.repeatability_limit(Endpoint.KI) == pytest.approx(2.77 * 0.641)

    def test_unknown_endpoint_raises_rather_than_defaulting(self):
        with pytest.raises(KeyError):
            ErrorModel().sigma("EC50")

    def test_the_caveat_travels_with_the_numbers(self):
        payload = ErrorModel().to_dict()
        assert "LOWER BOUND" in payload["caveat"]
        assert "within-publication" in payload["caveat"]


class TestDerivedThresholds:
    def test_ranking_margin_is_raised_to_the_noise_floor(self, profile):
        """0.5 p-units is below sigma_rep for every endpoint on this corpus.

        Ordering a pair that differs by less than measurement error is training
        on noise, so the margin becomes one sigma instead of the spec's starting
        value.
        """
        for endpoint in (Endpoint.KD, Endpoint.KI, Endpoint.IC50):
            assert profile.ranking_margin(endpoint) > 0.5

    def test_activity_cliff_threshold_is_endpoint_specific(self, profile):
        """A 1.0 p-unit pKd difference is only ~1.06 sigma -- inside the noise."""
        assert profile.cliff_delta_threshold(Endpoint.KD) == pytest.approx(1.89, abs=0.01)
        assert profile.cliff_delta_threshold(Endpoint.KI) == pytest.approx(1.282, abs=0.01)
        assert profile.cliff_delta_threshold(Endpoint.IC50) == pytest.approx(1.342, abs=0.01)

    def test_cliff_threshold_never_falls_below_the_literature_convention(self):
        """Even a hypothetically quiet endpoint keeps the 1.0 p-unit floor."""
        quiet = ErrorModel(sigma_rep={"Kd": 0.1, "Ki": 0.1, "IC50": 0.1})
        profile = get_profile().with_overrides(error_model=quiet)
        assert profile.cliff_delta_threshold(Endpoint.KD) == pytest.approx(1.0)

    def test_discordance_bands_bracket_the_repeatability_limit(self, profile):
        moderate, high = profile.discordance_thresholds(Endpoint.KI)
        assert moderate < high
        assert high == pytest.approx(profile.error_model.repeatability_limit(Endpoint.KI))

    def test_discordance_is_endpoint_specific(self, profile):
        _, kd_high = profile.discordance_thresholds(Endpoint.KD)
        _, ki_high = profile.discordance_thresholds(Endpoint.KI)
        assert kd_high > ki_high


class TestDependenceStructure:
    def test_effective_sample_size_is_far_below_the_observed_count(self, profile):
        """10,474 targets correspond to 572 effective independent units.

        This is why intervals come from a cluster bootstrap over protein
        components and why per-target macro averages are reported beside pooled
        metrics.
        """
        dependence = profile.dependence
        assert dependence.n_targets_effective < dependence.n_targets_observed / 10
        assert dependence.n_ligands_effective < dependence.n_ligands_observed


class TestPolicyIntent:
    def test_charges_are_preserved_by_default(self, profile):
        assert profile.standardization.uncharge_identity is False

    def test_largest_fragment_fallback_is_disabled_by_default(self, profile):
        assert profile.standardization.largest_fragment_fallback is False

    def test_expensive_tautomer_canonicalization_is_off_by_default(self, profile):
        assert profile.standardization.canonicalize_tautomers is False

    def test_missing_citation_is_a_singleton_group(self, profile):
        """The specification's hard requirement against giant ranking groups."""
        assert profile.comparability.unknown_citation_is_singleton is True

    def test_tier_b_is_not_keyed_on_pubchem_aid_presence(self, profile):
        """68% of rows carry an AID, including ~545k patent rows."""
        assert profile.ic50.tier_b_by_aid_presence is False

    def test_tier_b_is_barred_from_regression_and_pairs_but_allowed_auxiliaries(
        self, profile
    ):
        assert profile.ic50.tier_b_eligible_regression is False
        assert profile.ic50.tier_b_eligible_pairs is False
        assert profile.ic50.tier_b_eligible_ranking is True
        assert profile.ic50.tier_b_eligible_retrieval is True

    def test_tiering_applies_only_to_ic50(self, profile):
        assert profile.ic50.tier_applies_to == ("IC50",)

    def test_calibration_is_a_separate_partition(self, profile):
        assert set(profile.splits.fractions) == {"train", "val", "cal", "test"}
        assert sum(profile.splits.fractions.values()) == pytest.approx(1.0)

    def test_temporal_split_uses_publication_not_import_date(self, profile):
        assert profile.splits.temporal_evidence_field == "date_published"

    def test_double_cold_assigns_axes_independently(self, profile):
        assert profile.splits.double_cold_independent is True

    def test_homology_grouping_is_exhaustive(self, profile):
        """No prefilter in the qualifying path: k-mers are not recall-safe at 40%."""
        assert profile.proteins.exhaustive is True
        assert profile.proteins.identity_threshold == pytest.approx(0.40)
        assert profile.proteins.coverage_threshold == pytest.approx(0.80)

    def test_similarity_screen_is_narrower_than_the_rescore(self, profile):
        """Folded Tanimoto >= true Tanimoto, so the screen cannot miss a pair."""
        assert profile.splits.fingerprint_screen_bits < profile.splits.fingerprint_bits

    def test_high_cross_partition_similarity_is_reported_not_quarantined(self, profile):
        assert profile.splits.quarantine_nn_above is None


class TestConfigHashHelper:
    def test_hash_accepts_a_policy_or_a_plain_payload(self):
        assert config_hash(StandardizationPolicy()) == config_hash(StandardizationPolicy())
        assert config_hash({"a": 1}) != config_hash({"a": 2})

    def test_unknown_profile_name_raises(self):
        with pytest.raises(KeyError):
            get_profile("does-not-exist")
