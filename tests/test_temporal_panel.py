"""Temporal panels: three populations, kept separate, with the exposure non-claim."""

from __future__ import annotations

import pandas as pd
import pytest

from bindra.data.splits import Partition
from bindra.eval.benchmarks.temporal import (
    TemporalDefinition,
    build_temporal_panels,
    lock_temporal_panels,
    pretraining_exposure,
)


def _observation(
    observation_id: str,
    pair: str,
    published: str,
    *,
    imported: str = "2025-06-01",
    target: str = "t" * 64,
    p_value: float = 7.0,
) -> dict:
    return {
        "observation_id": observation_id,
        "pair_key": pair,
        "identity_key": f"lig-{pair}",
        "sequence_sha256": target,
        "endpoint": "Ki",
        "p_value": p_value,
        "is_exact": True,
        "is_censored": False,
        "first_evidence_date": published,
        "date_in_bindingdb": imported,
    }


def _corpus() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # Old pair, imported recently -> train, and a re-import control.
            _observation("o1", "old", "2019-03-01"),
            # Pair entirely in the future -> strict future panel.
            _observation("o2", "future", "2025-03-01", imported="2025-04-01"),
            # Pair straddling the cutoff -> repeated-measurement panel.
            _observation("o3", "straddle", "2021-01-01"),
            _observation("o4", "straddle", "2025-01-01", imported="2025-02-01"),
            # 2023 H1 -> validation.
            _observation("o5", "val", "2023-02-01"),
            # Unknown date -> excluded entirely.
            _observation("o6", "unknown", None),
        ]
    )


class TestPanels:
    @pytest.fixture
    def built(self):
        return build_temporal_panels(_corpus())

    def test_three_panels_are_produced(self, built):
        panels, _ = built
        assert set(panels) == {
            "panel_strict_future",
            "panel_repeated_measurement",
            "panel_reimport_control",
        }

    def test_the_strict_future_panel_holds_only_wholly_future_pairs(self, built):
        panels, _ = built
        assert set(panels["panel_strict_future"]["pair_key"]) == {"future"}

    def test_straddling_pairs_are_separated_not_pooled(self, built):
        """A pair already measured before the cutoff is not a novel observation."""
        panels, _ = built
        assert set(panels["panel_repeated_measurement"]["pair_key"]) == {"straddle"}
        assert "straddle" not in set(panels["panel_strict_future"]["pair_key"])

    def test_the_reimport_control_catches_old_measurements_imported_late(self, built):
        panels, _ = built
        reimports = panels["panel_reimport_control"]
        assert "old" in set(reimports["pair_key"])
        # The wholly-future pair is not a re-import.
        assert "future" not in set(reimports["pair_key"])

    def test_unknown_dates_are_excluded_from_every_panel(self, built):
        panels, report = built
        for panel in panels.values():
            assert "unknown" not in set(panel["pair_key"])
        assert report["attrition"]["pairs_with_unknown_date"] >= 1

    def test_counts_are_reported_per_panel(self, built):
        _, report = built
        for name, counts in report["counts"].items():
            assert {"n_observations", "n_pairs", "n_targets"} <= set(counts)


class TestPretrainingExposure:
    def test_the_non_claim_is_explicit(self):
        """Nothing here can establish that the pretrained system is time-clean."""
        payload = pretraining_exposure()
        assert "cannot be established" in payload["claim"]
        assert payload["protein_encoder"]["training_corpus_enumerable"] is False

    def test_the_pinned_revision_is_recorded(self):
        from bindra.encoders.protein_esmc import ESMC_600M_REVISION

        assert pretraining_exposure()["protein_encoder"]["revision"] == ESMC_600M_REVISION

    def test_the_exposure_block_travels_into_the_report(self):
        _, report = build_temporal_panels(_corpus())
        assert "cannot be established" in report["pretraining_exposure"]["claim"]


class TestDefinition:
    def test_evidence_is_the_publication_date(self):
        """A 2025 import of a 1998 measurement is not 2025 evidence."""
        assert TemporalDefinition().evidence_field == "date_published"

    def test_the_train_cutoff_precedes_the_test_cutoff(self):
        definition = TemporalDefinition()
        assert definition.train_cutoff < definition.cutoff

    def test_tuning_windows_sit_between_the_two_cutoffs(self):
        definition = TemporalDefinition()
        assert definition.train_cutoff < definition.val_window[0]
        assert definition.val_window[1] < definition.cal_window[0]
        assert definition.cal_window[1] <= definition.cutoff


class TestLocking:
    def test_each_panel_locks_separately_with_sealed_labels(self, tmp_path):
        from bindra.eval.benchmarks.lock import load_locked, load_queries

        panels, report = build_temporal_panels(_corpus())
        lock_temporal_panels(
            panels, report, inputs={"snapshot_release": "202610"}, directory=tmp_path
        )
        for name in panels:
            manifest = load_locked(tmp_path / name)
            assert manifest.panel == name
            assert manifest.contains_test_labels
            # Queries carry identifiers, never the value.
            assert "p_value" not in load_queries(tmp_path / name).columns

    def test_labels_require_authorisation(self, tmp_path):
        from bindra.eval.benchmarks.lock import SealedLabelError, load_sealed_labels

        panels, report = build_temporal_panels(_corpus())
        lock_temporal_panels(panels, report, inputs={}, directory=tmp_path)
        with pytest.raises(SealedLabelError):
            load_sealed_labels(tmp_path / "panel_strict_future")
        frame = load_sealed_labels(
            tmp_path / "panel_strict_future",
            allow_labels=True,
            reason="metrics, predictions frozen",
        )
        assert "p_value" in frame.columns

    def test_relocking_is_refused(self, tmp_path):
        from bindra.eval.benchmarks.lock import BenchmarkError

        panels, report = build_temporal_panels(_corpus())
        lock_temporal_panels(panels, report, inputs={}, directory=tmp_path)
        with pytest.raises(BenchmarkError):
            lock_temporal_panels(panels, report, inputs={}, directory=tmp_path)
