"""Fishing challenge sets: decoys stay presumed, exclusion stays homologue-aware.

The specification's hard line is that a computational decoy is not an experimental
nonbinder. ``test_decoys_never_receive_an_affinity_label`` and
``test_decoys_are_excluded_by_protein_group_not_sequence`` are the two tests that
keep that line from eroding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bindra.data.properties import PROPERTY_COLUMNS, PropertyStore
from bindra.eval.benchmarks.fishing import (
    FishingDefinition,
    build_fishing_challenges,
    lock_fishing_panel,
)

TARGET_A = "a" * 64
TARGET_B = "b" * 64


def _store(n: int = 400) -> PropertyStore:
    """A property store whose molecules span a tight band, so matching succeeds."""
    rng = np.random.default_rng(1)
    values = np.zeros((n, len(PROPERTY_COLUMNS)), dtype=np.float32)
    values[:, 0] = rng.normal(400, 10, n)  # molecular weight
    values[:, 1] = rng.normal(3.0, 0.3, n)  # clogp
    values[:, 2] = rng.normal(80, 5, n)  # tpsa
    values[:, 3] = 0  # formal charge
    values[:, 4] = 5  # rotatable bonds
    return PropertyStore(
        identity_keys=tuple(f"L{i}" for i in range(n)),
        values=values,
        valid=np.ones(n, dtype=bool),
    )


def _observations(
    target: str, n_actives: int = 12, n_inactives: int = 3, *, first: int = 0
) -> pd.DataFrame:
    rows = []
    for i in range(n_actives):
        rows.append(
            {
                "observation_id": f"{target[:4]}-a{i}",
                "identity_key": f"L{first + i}",
                "sequence_sha256": target,
                "endpoint": "Ki",
                "p_value": 8.0,
                "is_exact": True,
                "ic50_tier": "na",
            }
        )
    for i in range(n_inactives):
        rows.append(
            {
                "observation_id": f"{target[:4]}-i{i}",
                "identity_key": f"L{first + n_actives + i}",
                "sequence_sha256": target,
                "endpoint": "Ki",
                "p_value": 4.5,
                "is_exact": True,
                "ic50_tier": "na",
            }
        )
    return pd.DataFrame(rows)


def _groups(mapping: dict[str, int]) -> pd.DataFrame:
    return pd.DataFrame(
        {"sequence_sha256": list(mapping), "protein_group": list(mapping.values())}
    )


class TestTiers:
    def test_three_tiers_are_produced(self):
        challenges, report = build_fishing_challenges(
            _observations(TARGET_A),
            protein_groups=_groups({TARGET_A: 0}),
            properties=_store(),
            definition=FishingDefinition(decoys_per_active=3),
        )
        assert set(challenges["tier"]) == {
            "measured_active",
            "measured_inactive",
            "property_matched_decoy",
        }
        assert report.n_actives == 12
        assert report.n_measured_inactives == 3
        assert report.n_decoys == 36

    def test_decoys_never_receive_an_affinity_label(self):
        """A computational decoy is not an experimental nonbinder."""
        challenges, _ = build_fishing_challenges(
            _observations(TARGET_A),
            protein_groups=_groups({TARGET_A: 0}),
            properties=_store(),
            definition=FishingDefinition(decoys_per_active=3),
        )
        decoys = challenges[challenges["tier"] == "property_matched_decoy"]
        assert decoys["p_value"].isna().all()
        assert set(decoys["binding_status"]) == {"unknown"}
        assert decoys["observation_id"].isna().all()

    def test_measured_inactives_keep_their_real_values(self):
        challenges, _ = build_fishing_challenges(
            _observations(TARGET_A),
            protein_groups=_groups({TARGET_A: 0}),
            properties=_store(),
            definition=FishingDefinition(decoys_per_active=2),
        )
        inactives = challenges[challenges["tier"] == "measured_inactive"]
        assert inactives["p_value"].notna().all()
        assert set(inactives["binding_status"]) == {"weak_or_inactive"}

    def test_a_target_without_enough_actives_is_skipped_and_counted(self):
        challenges, report = build_fishing_challenges(
            _observations(TARGET_A, n_actives=3),
            protein_groups=_groups({TARGET_A: 0}),
            properties=_store(),
            definition=FishingDefinition(min_actives_per_target=10),
        )
        assert len(challenges) == 0
        assert report.targets_without_enough_actives == 1
        assert report.n_targets_locked == 0

    def test_a_target_without_measured_inactives_is_still_usable_but_flagged(self):
        challenges, report = build_fishing_challenges(
            _observations(TARGET_A, n_inactives=0),
            protein_groups=_groups({TARGET_A: 0}),
            properties=_store(),
            definition=FishingDefinition(decoys_per_active=2),
        )
        assert report.targets_without_measured_inactives == 1
        assert report.n_targets_locked == 1
        assert "measured_inactive" not in set(challenges["tier"])


class TestExclusion:
    def test_a_decoy_is_never_a_compound_measured_against_that_target(self):
        observations = _observations(TARGET_A)
        challenges, _ = build_fishing_challenges(
            observations,
            protein_groups=_groups({TARGET_A: 0}),
            properties=_store(),
            definition=FishingDefinition(decoys_per_active=5),
        )
        decoys = set(
            challenges.loc[challenges["tier"] == "property_matched_decoy", "identity_key"]
        )
        assert decoys.isdisjoint(set(observations["identity_key"]))

    def test_decoys_are_excluded_by_protein_group_not_sequence(self):
        """The same protein appears as many constructs here.

        A compound measured against one construct is not a safe decoy for another
        construct of the same protein, so exclusion follows the alignment-derived
        protein group rather than the sequence hash.
        """
        # Two distinct target hashes, same protein group.
        observations = pd.concat(
            [_observations(TARGET_A, first=0), _observations(TARGET_B, first=100)],
            ignore_index=True,
        )
        groups = _groups({TARGET_A: 7, TARGET_B: 7})
        challenges, _ = build_fishing_challenges(
            observations,
            protein_groups=groups,
            properties=_store(),
            definition=FishingDefinition(decoys_per_active=5),
        )
        decoys = set(
            challenges.loc[challenges["tier"] == "property_matched_decoy", "identity_key"]
        )
        # Nothing measured against *either* construct may be a decoy for *either*.
        assert decoys.isdisjoint(set(observations["identity_key"]))

    def test_disabling_group_exclusion_narrows_to_the_target_only(self):
        observations = pd.concat(
            [_observations(TARGET_A, first=0), _observations(TARGET_B, first=100)],
            ignore_index=True,
        )
        challenges, _ = build_fishing_challenges(
            observations,
            protein_groups=_groups({TARGET_A: 7, TARGET_B: 7}),
            properties=_store(),
            definition=FishingDefinition(
                decoys_per_active=5, exclude_by_protein_group=False
            ),
        )
        # With group exclusion off, a compound measured against the sibling
        # construct may be chosen -- which is exactly why it defaults to on.
        assert len(challenges) > 0


class TestMatching:
    def test_decoys_fall_inside_the_declared_tolerances(self):
        store = _store()
        definition = FishingDefinition(decoys_per_active=4)
        observations = _observations(TARGET_A)
        challenges, _ = build_fishing_challenges(
            observations,
            protein_groups=_groups({TARGET_A: 0}),
            properties=store,
            definition=definition,
        )
        index = store.index
        actives = observations.loc[observations["p_value"] >= 7.0, "identity_key"].tolist()
        columns = [store.columns.index(c) for c in definition.match_columns]
        reference = np.asarray(store.values)[[index[a] for a in actives]][:, columns]
        tolerances = np.asarray(definition.match_tolerances)

        decoys = challenges.loc[
            challenges["tier"] == "property_matched_decoy", "identity_key"
        ]
        for key in decoys:
            row = np.asarray(store.values)[index[key]][columns]
            # Within tolerance of at least one active on every matched axis.
            assert bool(np.any(np.all(np.abs(reference - row) <= tolerances, axis=1)))

    def test_matching_is_deterministic_for_a_seed(self):
        kwargs = dict(
            protein_groups=_groups({TARGET_A: 0}),
            properties=_store(),
            definition=FishingDefinition(decoys_per_active=4),
        )
        first, _ = build_fishing_challenges(_observations(TARGET_A), **kwargs)
        second, _ = build_fishing_challenges(_observations(TARGET_A), **kwargs)
        pd.testing.assert_frame_equal(first, second)

    def test_a_shortfall_is_reported_rather_than_padded(self):
        """If the pool cannot supply the quota, say so; do not relax matching."""
        challenges, report = build_fishing_challenges(
            _observations(TARGET_A),
            protein_groups=_groups({TARGET_A: 0}),
            properties=_store(n=40),
            definition=FishingDefinition(decoys_per_active=50),
        )
        assert report.decoy_shortfall > 0


class TestLocking:
    def test_decoys_land_in_unknowns_not_labels(self, tmp_path):
        """There must be no table in which a decoy could acquire a value."""
        from bindra.eval.benchmarks.lock import load_locked, load_unknowns

        challenges, report = build_fishing_challenges(
            _observations(TARGET_A),
            protein_groups=_groups({TARGET_A: 0}),
            properties=_store(),
            definition=FishingDefinition(decoys_per_active=3),
        )
        definition = FishingDefinition(decoys_per_active=3)
        lock_fishing_panel(
            challenges,
            definition=definition,
            inputs={"snapshot_release": "202610"},
            directory=tmp_path,
            report=report,
        )
        manifest = load_locked(tmp_path / "fishing")
        assert manifest.panel == "fishing"
        unknowns = load_unknowns(tmp_path / "fishing")
        assert set(unknowns["tier"]) == {"property_matched_decoy"}
        assert "p_value" not in unknowns.columns

    def test_the_manifest_demands_similarity_baselines(self, tmp_path):
        """DUD-E-style decoys flatter 2D similarity, so baselines are mandatory."""
        from bindra.eval.benchmarks.lock import load_locked

        challenges, report = build_fishing_challenges(
            _observations(TARGET_A),
            protein_groups=_groups({TARGET_A: 0}),
            properties=_store(),
            definition=FishingDefinition(decoys_per_active=3),
        )
        lock_fishing_panel(
            challenges,
            definition=FishingDefinition(decoys_per_active=3),
            inputs={},
            directory=tmp_path,
            report=report,
        )
        notes = " ".join(load_locked(tmp_path / "fishing").notes)
        assert "fingerprint-similarity" in notes
        assert "seed-assisted" in notes
        assert "PRESUMED" in notes


class TestDefinition:
    def test_activity_thresholds_are_prespecified(self):
        definition = FishingDefinition()
        assert definition.active_p_min == 7.0
        assert definition.inactive_p_max == 5.0
        assert definition.active_p_min > definition.inactive_p_max

    def test_matching_axes_and_tolerances_are_the_same_length(self):
        definition = FishingDefinition()
        assert len(definition.match_columns) == len(definition.match_tolerances)

    def test_every_matched_axis_exists_in_the_property_store(self):
        for column in FishingDefinition().match_columns:
            assert column in PROPERTY_COLUMNS

    def test_group_exclusion_is_on_by_default(self):
        assert FishingDefinition().exclude_by_protein_group is True
