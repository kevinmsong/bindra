"""Split grouping, weighted assignment, and manifest immutability."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from bindra.config.profiles import get_profile
from bindra.data.splits import (
    Partition,
    Regime,
    SplitError,
    SplitManifest,
    assign_groups,
    drug_groups,
    read_split,
    partition_counts,
    verify_manifest,
    write_split,
)


def _ligands(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["identity_key", "leakage_key", "scaffold_key"])


class TestDrugGroups:
    def test_salt_and_parent_share_a_group_via_the_leakage_key(self):
        groups = drug_groups(
            _ligands([("acid", "L1", "S1"), ("salt", "L1", "S1"), ("other", "L2", "S2")])
        )
        assert groups["acid"] == groups["salt"]
        assert groups["other"] != groups["acid"]

    def test_a_chemical_series_shares_a_group_via_the_scaffold(self):
        """A cold-drug split must separate series, not just compounds.

        Otherwise a "novel" test molecule can be one methyl group from a training
        molecule, and the regime measures nothing.
        """
        groups = drug_groups(
            _ligands([("a", "L1", "SHARED"), ("b", "L2", "SHARED"), ("c", "L3", "OTHER")])
        )
        assert groups["a"] == groups["b"]
        assert groups["c"] != groups["a"]

    def test_grouping_is_transitive_across_both_edge_types(self):
        """a-b share a scaffold, b-c share a leakage key, so all three merge."""
        groups = drug_groups(
            _ligands([("a", "L1", "S1"), ("b", "L2", "S1"), ("c", "L2", "S2")])
        )
        assert groups["a"] == groups["b"] == groups["c"]

    def test_acyclic_molecules_do_not_all_collapse_together(self):
        """Their scaffold key is their own leakage key, so they stay separate."""
        groups = drug_groups(
            _ligands([("a", "LA", "LA"), ("b", "LB", "LB"), ("c", "LC", "LC")])
        )
        assert len({groups["a"], groups["b"], groups["c"]}) == 3


class TestAssignGroups:
    def _realized(self, weights, fractions, **kwargs):
        assignment = assign_groups(weights, fractions, **kwargs)
        total = sum(weights.values())
        realized = {name: 0 for name in fractions}
        for key, partition in assignment.items():
            realized[partition] += weights[key]
        return assignment, {name: value / total for name, value in realized.items()}

    def test_fractions_are_respected_with_moderately_skewed_weights(self):
        """Group sizes here span orders of magnitude; a shuffle misses badly."""
        weights = {f"g{i}": 1 for i in range(400)}
        weights.update({"big1": 40, "big2": 30, "big3": 25})
        fractions = {"train": 0.70, "val": 0.10, "cal": 0.10, "test": 0.10}
        _, realized = self._realized(weights, fractions, seed=1)
        for name, share in fractions.items():
            assert realized[name] == pytest.approx(share, abs=0.02)

    def test_a_group_larger_than_a_small_target_goes_to_train(self):
        """Groups are indivisible, so an oversized one must land where it hurts least.

        A component holding 55% of all observations cannot fit a 10% partition.
        Largest-first packing sends it to the partition with the biggest absolute
        deficit rather than letting it overshoot test by 45 points.
        """
        weights = {f"g{i}": 1 for i in range(200)}
        weights["giant"] = 500
        fractions = {"train": 0.70, "val": 0.10, "cal": 0.10, "test": 0.10}
        assignment, realized = self._realized(weights, fractions, seed=1)
        assert assignment["giant"] == "train"
        # The small partitions stay close to target despite the giant group.
        for name in ("val", "cal", "test"):
            assert realized[name] == pytest.approx(0.10, abs=0.04)

    def test_assignment_is_deterministic_for_a_seed(self):
        weights = {f"g{i}": i + 1 for i in range(50)}
        fractions = {"train": 0.8, "test": 0.2}
        first = assign_groups(weights, fractions, seed=7)
        second = assign_groups(weights, fractions, seed=7)
        assert first == second

    def test_a_different_seed_gives_a_different_assignment(self):
        weights = {f"g{i}": 1 for i in range(50)}
        fractions = {"train": 0.8, "test": 0.2}
        assert assign_groups(weights, fractions, seed=1) != assign_groups(
            weights, fractions, seed=2
        )

    def test_assignment_is_independent_of_input_order(self):
        """A manifest must be reproducible from its seed alone."""
        weights = {f"g{i}": i + 1 for i in range(40)}
        reversed_weights = dict(reversed(list(weights.items())))
        fractions = {"train": 0.7, "test": 0.3}
        assert assign_groups(weights, fractions, seed=3) == assign_groups(
            reversed_weights, fractions, seed=3
        )

    def test_every_group_is_assigned_exactly_once(self):
        weights = {f"g{i}": 1 for i in range(30)}
        assignment = assign_groups(weights, {"train": 0.5, "test": 0.5}, seed=1)
        assert set(assignment) == set(weights)

    def test_preassigned_groups_are_honoured(self):
        """The activity-cliff panel needs its series forced into test."""
        weights = {f"g{i}": 10 for i in range(20)}
        assignment = assign_groups(
            weights,
            {"train": 0.8, "test": 0.2},
            seed=1,
            preassigned={"g0": "test", "g1": "test"},
        )
        assert assignment["g0"] == "test"
        assert assignment["g1"] == "test"

    def test_empty_input_is_handled(self):
        assert assign_groups({}, {"train": 1.0}, seed=1) == {}


class TestManifest:
    @pytest.fixture
    def built(self, tmp_path) -> tuple[Path, SplitManifest, dict[str, str]]:
        assignment = {f"obs{i}": ("train" if i % 4 else "test") for i in range(40)}
        manifest = SplitManifest(
            regime=Regime.COLD_DRUG.value,
            seed=get_profile().splits.seed,
            fractions={"train": 0.75, "test": 0.25},
            counts={"train": {"n_observations": 30}, "test": {"n_observations": 10}},
            quarantine={"n_observations": 0, "reasons": {}},
            inputs={
                "snapshot_release": "202610",
                "snapshot_sha256": "abc",
                "observations_combined_sha256": "table-hash",
            },
            policy={"identity_threshold": 0.40},
            created_at="2026-10-02T00:00:00Z",
        )
        directory = write_split(manifest, assignment, tmp_path / "cold_drug")
        return directory, manifest, assignment

    def test_round_trip_preserves_the_assignment(self, built):
        directory, _, assignment = built
        _, restored = read_split(directory)
        assert restored == assignment

    def test_manifest_records_a_self_hash_and_split_id(self, built):
        directory, manifest, _ = built
        payload = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        assert payload["manifest_sha256"] == manifest.manifest_sha256
        assert payload["split_id"].startswith("cold_drug-")

    def test_editing_the_manifest_is_detected(self, built):
        directory, _, _ = built
        path = directory / "manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["seed"] = 999
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(SplitError, match="has been edited"):
            read_split(directory)

    def test_editing_an_id_file_is_detected(self, built):
        directory, _, _ = built
        path = directory / "ids" / "test.txt.gz"
        with gzip.open(path, "wb") as handle:
            handle.write(b"obs999\n")
        with pytest.raises(SplitError, match="has changed"):
            read_split(directory)

    def test_a_changed_observation_table_invalidates_the_split(self, built):
        """Curated data moving under a manifest is a hard error, not a reinterpretation."""
        directory, _, _ = built
        verify_manifest(directory, observations_sha256="table-hash")
        with pytest.raises(SplitError, match="curated observation table has changed"):
            verify_manifest(directory, observations_sha256="different-hash")

    def test_writing_twice_is_refused(self, built):
        directory, manifest, assignment = built
        with pytest.raises(SplitError, match="immutable"):
            write_split(manifest, assignment, directory)

    def test_overwrite_requires_an_explicit_flag(self, built):
        directory, manifest, assignment = built
        write_split(manifest, assignment, directory, overwrite=True)

    def test_id_files_are_byte_identical_for_the_same_assignment(self, tmp_path):
        """A committed manifest alone must be a complete reproducibility record.

        The id files are not committed, so regenerating them has to produce the
        same bytes or the recorded hashes are meaningless.
        """
        assignment = {f"obs{i}": ("train" if i % 3 else "test") for i in range(30)}

        def build(where: Path) -> dict[str, str]:
            manifest = SplitManifest(
                regime=Regime.RANDOM_PAIR.value,
                seed=1,
                fractions={"train": 0.67, "test": 0.33},
                counts={},
                quarantine={},
                inputs={},
                policy={},
                created_at="fixed",
            )
            write_split(manifest, assignment, where)
            return {k: v["sha256"] for k, v in manifest.id_files.items()}

        assert build(tmp_path / "a") == build(tmp_path / "b")

    def test_hash_changes_when_the_assignment_changes(self, tmp_path):
        def build(where: Path, assignment: dict[str, str]) -> str:
            manifest = SplitManifest(
                regime=Regime.RANDOM_PAIR.value,
                seed=1,
                fractions={"train": 0.5, "test": 0.5},
                counts={},
                quarantine={},
                inputs={},
                policy={},
                created_at="fixed",
            )
            write_split(manifest, assignment, where)
            return manifest.manifest_sha256

        first = build(tmp_path / "a", {"o1": "train", "o2": "test"})
        second = build(tmp_path / "b", {"o1": "test", "o2": "train"})
        assert first != second


def _corpus(years: list[int] | None = None) -> pd.DataFrame:
    """A small synthetic corpus with independent drug and protein axes."""
    years = years or [2019, 2021, 2022, 2023, 2023, 2024, 2025]
    rows = []
    for drug in range(len(years)):
        for protein in range(5):
            for replicate in range(4):
                month = "02" if drug != 4 else "09"
                rows.append(
                    {
                        "observation_id": f"o{drug}-{protein}-{replicate}",
                        "pair_key": f"pair{drug}-{protein}",
                        "identity_key": f"lig{drug}-{replicate}",
                        "leakage_key": f"leak{drug}",
                        "scaffold_key": f"scaf{drug}",
                        "sequence_sha256": f"seq{protein}",
                        "drug_group": drug,
                        "protein_group": protein,
                        "is_exact": True,
                        "is_censored": False,
                        "first_evidence_date": f"{years[drug]}-{month}-15",
                        "date_in_bindingdb": "2025-06-01",
                    }
                )
    return pd.DataFrame(rows)


class TestRegimeBuilders:
    @pytest.fixture
    def policy(self):
        return get_profile().splits

    def test_cold_drug_keeps_whole_drug_groups_together(self, policy):
        from bindra.data.splits import audit_split, make_grouped_split

        frame = _corpus()
        assignment = make_grouped_split(frame, "drug_group", policy=policy)
        audit = audit_split(frame, assignment, Regime.COLD_DRUG.value, policy=policy)
        assert audit["all_gating_checks_passed"]
        names = {check["check"] for check in audit["checks"]}
        assert {"leakage_key_disjoint", "scaffold_disjoint"} <= names

    def test_cold_target_keeps_whole_protein_groups_together(self, policy):
        from bindra.data.splits import audit_split, make_grouped_split

        frame = _corpus()
        assignment = make_grouped_split(frame, "protein_group", policy=policy)
        audit = audit_split(frame, assignment, Regime.COLD_TARGET.value, policy=policy)
        assert audit["all_gating_checks_passed"]

    def test_every_observation_is_assigned(self, policy):
        from bindra.data.splits import make_grouped_split

        frame = _corpus()
        assignment = make_grouped_split(frame, "pair_key", policy=policy)
        assert set(assignment) == set(frame["observation_id"])

    def test_double_cold_quarantines_disagreeing_assignments(self, policy):
        """The axes are assigned independently, so most cells are discarded.

        A combined protein-x-drug cluster id would retain everything and
        guarantee only that the *combination* is unseen -- a weaker property.
        """
        from bindra.data.splits import make_double_cold_split

        frame = _corpus()
        assignment, attrition = make_double_cold_split(frame, policy=policy)
        assert attrition["level1_quarantined"] > 0
        assert attrition["retained"] < attrition["input"]
        assert set(assignment) == set(frame["observation_id"])

    def test_double_cold_separates_both_axes(self, policy):
        from bindra.data.splits import audit_split, make_double_cold_split

        frame = _corpus()
        assignment, _ = make_double_cold_split(frame, policy=policy)
        audit = audit_split(frame, assignment, Regime.DOUBLE_COLD.value, policy=policy)
        assert audit["all_gating_checks_passed"]

    def test_double_cold_would_fail_under_combined_cluster_grouping(self, policy):
        """A fixture that passes for the wrong algorithm and fails for it here.

        Grouping by the combined (drug, protein) cell keeps every observation and
        splits cells across partitions, which leaves the same drug and the same
        protein on both sides. The audit must reject that.
        """
        from bindra.data.splits import audit_split, make_grouped_split

        frame = _corpus()
        frame = frame.assign(
            combined=frame["drug_group"].astype(str) + "x" + frame["protein_group"].astype(str)
        )
        naive = make_grouped_split(frame, "combined", policy=policy)
        audit = audit_split(frame, naive, Regime.DOUBLE_COLD.value, policy=policy)
        assert not audit["all_gating_checks_passed"]

    def test_temporal_fills_every_partition_from_dates(self, policy):
        from bindra.data.splits import make_temporal_split

        frame = _corpus()
        _, attrition = make_temporal_split(frame, policy=policy)
        for partition in ("train", "val", "cal", "test"):
            assert attrition[partition] > 0

    def test_temporal_excludes_unknown_dates(self, policy):
        from bindra.data.splits import make_temporal_split

        frame = _corpus()
        frame.loc[frame.index[:4], "first_evidence_date"] = None
        assignment, attrition = make_temporal_split(frame, policy=policy)
        assert attrition["pairs_with_unknown_date"] > 0
        assert assignment["o0-0-0"] == Partition.QUARANTINE.value

    def test_reimported_old_measurements_never_land_in_test(self, policy):
        """A 2025 import of a 2019 measurement is not future evidence."""
        from bindra.data.splits import audit_split, make_temporal_split

        frame = _corpus()
        assignment, _ = make_temporal_split(frame, policy=policy)
        audit = audit_split(frame, assignment, Regime.TEMPORAL.value, policy=policy)
        check = next(c for c in audit["checks"] if c["check"] == "temporal_integrity")
        assert check["n_reimports"] > 0
        assert check["n_reimports_misplaced"] == 0
        assert check["passed"]

    def test_a_future_label_in_train_is_caught(self, policy):
        """The audit has to be able to fail, or it proves nothing."""
        from bindra.data.splits import audit_split, make_temporal_split

        frame = _corpus()
        assignment, _ = make_temporal_split(frame, policy=policy)
        leaked = dict(assignment)
        first_test = next(o for o, p in assignment.items() if p == Partition.TEST.value)
        leaked[first_test] = Partition.TRAIN.value
        audit = audit_split(frame, leaked, Regime.TEMPORAL.value, policy=policy)
        check = next(c for c in audit["checks"] if c["check"] == "temporal_integrity")
        assert check["n_future_labels_in_train"] == 1
        assert not audit["all_gating_checks_passed"]

    def test_pair_straddling_is_caught(self, policy):
        from bindra.data.splits import audit_split, make_grouped_split

        frame = _corpus()
        assignment = make_grouped_split(frame, "pair_key", policy=policy)
        leaked = dict(assignment)
        leaked["o0-0-0"] = (
            Partition.TEST.value
            if assignment["o0-0-0"] != Partition.TEST.value
            else Partition.TRAIN.value
        )
        audit = audit_split(frame, leaked, Regime.RANDOM_PAIR.value, policy=policy)
        check = next(c for c in audit["checks"] if c["check"] == "pair_integrity")
        assert check["n_violations"] >= 1

    def test_audit_detects_an_unassigned_observation(self, policy):
        from bindra.data.splits import audit_split, make_grouped_split

        frame = _corpus()
        assignment = make_grouped_split(frame, "pair_key", policy=policy)
        partial = {k: v for k, v in assignment.items() if k != "o1-1-1"}
        audit = audit_split(frame, partial, Regime.RANDOM_PAIR.value, policy=policy)
        check = next(c for c in audit["checks"] if c["check"] == "set_algebra")
        assert check["n_violations"] >= 1

    def test_every_check_reports_the_same_fields(self, policy):
        """So a reader or a report can iterate checks uniformly."""
        from bindra.data.splits import audit_split, make_temporal_split

        frame = _corpus()
        assignment, _ = make_temporal_split(frame, policy=policy)
        audit = audit_split(frame, assignment, Regime.TEMPORAL.value, policy=policy)
        for check in audit["checks"]:
            assert {"check", "gating", "n_checked", "n_violations", "passed"} <= set(check)


class TestPolicyIntent:
    def test_four_partitions_including_a_separate_calibration_set(self):
        assert {p.value for p in Partition} >= {"train", "val", "cal", "test", "quarantine"}

    def test_five_regimes_are_defined(self):
        assert {r.value for r in Regime} == {
            "random_pair",
            "cold_drug",
            "cold_target",
            "double_cold",
            "temporal",
        }

    def test_the_declared_seed_is_a_constant(self):
        """Seed shopping must be visible, so the seed is pinned and asserted."""
        assert get_profile().splits.seed == 20_261_001
