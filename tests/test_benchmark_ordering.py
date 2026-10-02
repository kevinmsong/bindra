"""The ordering constraint between cliff candidates and the splits.

Cliff series must be frozen *before* the splits are built and then preassigned to
test. Keeping whichever cliff pairs happen to land in test instead would let the
split decide the panel, which is post-hoc selection, and would risk an empty panel.

These tests cover the glue that enforces it, because the failure is silent: if the
frozen set is never read, splits still build cleanly and the panel simply comes out
small or empty for reasons nobody notices.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bindra.cli.lock_benchmarks import load_required_test_groups
from bindra.data.splits import Partition, assign_groups, make_grouped_split
from bindra.config.profiles import get_profile


def _write_frozen(directory: Path, groups: list[int]) -> Path:
    panel = directory / "activity_cliff"
    panel.mkdir(parents=True, exist_ok=True)
    payload = {
        "panel": "activity_cliff",
        "definition_sha256": "abc",
        "n_required_groups": len(groups),
        "required_test_drug_groups": groups,
        "set_sha256": "deadbeef",
    }
    (panel / "required_test_groups.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return panel


class TestLoadingTheFrozenSet:
    def test_a_frozen_set_round_trips(self, tmp_path):
        _write_frozen(tmp_path, [3, 7, 11])
        assert load_required_test_groups(tmp_path) == frozenset({3, 7, 11})

    def test_an_absent_file_is_empty_not_an_error(self, tmp_path):
        """A missing set means `candidates` has not run, which is reported."""
        assert load_required_test_groups(tmp_path) == frozenset()

    def test_group_ids_are_integers_however_json_stored_them(self, tmp_path):
        panel = _write_frozen(tmp_path, [1, 2])
        payload = json.loads((panel / "required_test_groups.json").read_text())
        payload["required_test_drug_groups"] = ["1", "2"]
        (panel / "required_test_groups.json").write_text(json.dumps(payload))
        assert load_required_test_groups(tmp_path) == frozenset({1, 2})


class TestPreassignment:
    def _corpus(self) -> pd.DataFrame:
        rows = []
        for drug in range(20):
            for replicate in range(5):
                rows.append(
                    {
                        "observation_id": f"o{drug}-{replicate}",
                        "pair_key": f"pair{drug}",
                        "identity_key": f"lig{drug}",
                        "leakage_key": f"leak{drug}",
                        "scaffold_key": f"scaf{drug}",
                        "sequence_sha256": "t" * 64,
                        "drug_group": drug,
                        "is_exact": True,
                        "is_censored": False,
                    }
                )
        return pd.DataFrame(rows)

    def test_preassigned_groups_land_in_test(self):
        frame = self._corpus()
        policy = get_profile().splits
        preassign = {2: Partition.TEST.value, 5: Partition.TEST.value}
        assignment = make_grouped_split(
            frame, "drug_group", policy=policy, preassigned=preassign
        )
        for group in (2, 5):
            observations = frame.loc[frame["drug_group"] == group, "observation_id"]
            assert {assignment[o] for o in observations} == {Partition.TEST.value}

    def test_preassignment_does_not_split_a_group(self):
        """A preassigned group still moves as one unit."""
        frame = self._corpus()
        assignment = make_grouped_split(
            frame,
            "drug_group",
            policy=get_profile().splits,
            preassigned={7: Partition.TEST.value},
        )
        by_group: dict[int, set[str]] = {}
        for row in frame.itertuples(index=False):
            by_group.setdefault(row.drug_group, set()).add(assignment[row.observation_id])
        assert all(len(parts) == 1 for parts in by_group.values())

    def test_without_preassignment_a_group_may_land_anywhere(self):
        """Shows the constraint is doing work, not coinciding with the default."""
        frame = self._corpus()
        policy = get_profile().splits
        plain = make_grouped_split(frame, "drug_group", policy=policy)
        forced = make_grouped_split(
            frame,
            "drug_group",
            policy=policy,
            preassigned={g: Partition.TEST.value for g in range(20)},
        )
        assert set(forced.values()) == {Partition.TEST.value}
        assert set(plain.values()) != {Partition.TEST.value}

    def test_assign_groups_honours_preassignment_even_against_the_fractions(self):
        """The constraint wins over the target shares, and the manifest records the
        realized counts so the discrepancy is visible."""
        weights = {g: 10 for g in range(10)}
        assignment = assign_groups(
            weights,
            {"train": 0.9, "test": 0.1},
            seed=1,
            preassigned={g: "test" for g in range(5)},
        )
        assert sum(1 for v in assignment.values() if v == "test") >= 5
