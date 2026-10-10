"""`bindra evaluate`: ordering discipline and honest coverage.

The two properties worth protecting:

``test_scoring_the_test_partition_requires_an_explicit_flag``
    The held-out set cannot be un-seen. Scoring several models against it and
    keeping the best overfits it, and nothing downstream reveals that.
``test_panels_that_are_not_scored_are_named``
    A report covering three of seven panels without saying so invites the
    reader to assume seven.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from bindra.cli.evaluate import (
    AFFINITY_PANELS,
    UNSCORED_PANELS,
    _metrics,
    build_parser,
)


class TestTestPartitionGuard:
    def test_scoring_the_test_partition_requires_an_explicit_flag(self, tmp_path, capsys):
        from bindra.cli.evaluate import cmd_evaluate

        checkpoint = tmp_path / "fake.pt"
        checkpoint.write_bytes(b"x")
        args = build_parser().parse_args(
            [str(checkpoint), "--partition", "test"]
        )
        assert cmd_evaluate(args) == 2
        assert "overfits it" in capsys.readouterr().out

    def test_val_needs_no_flag(self):
        args = build_parser().parse_args(["ckpt.pt", "--partition", "val"])
        assert args.partition == "val" and args.i_mean_it is False

    def test_the_flag_exists_and_is_off_by_default(self):
        assert build_parser().parse_args(["ckpt.pt"]).i_mean_it is False

    def test_a_missing_checkpoint_is_refused_before_anything_is_loaded(self, tmp_path):
        from bindra.cli.evaluate import cmd_evaluate

        args = build_parser().parse_args([str(tmp_path / "absent.pt")])
        assert cmd_evaluate(args) == 1


class TestCoverageHonesty:
    def test_panels_that_are_not_scored_are_named(self):
        assert set(UNSCORED_PANELS) == {
            "selectivity",
            "censored_selectivity",
            "fishing",
            "activity_cliff",
        }
        for panel, why in UNSCORED_PANELS.items():
            assert why and not why.endswith("."), f"{panel} needs a reason"

    def test_the_scored_and_unscored_sets_do_not_overlap(self):
        assert not set(AFFINITY_PANELS) & set(UNSCORED_PANELS)

    def test_every_built_panel_is_either_scored_or_explained(self):
        """No panel may fall through the gap between the two lists."""
        from pathlib import Path

        directory = Path("data/benchmarks/202610")
        if not directory.exists():
            pytest.skip("benchmarks not built")
        built = {p.name for p in directory.iterdir() if (p / "manifest.json").exists()}
        known = set(AFFINITY_PANELS) | set(UNSCORED_PANELS)
        assert built <= known, f"unaccounted panels: {sorted(built - known)}"


class TestMetrics:
    def _synthetic(self, n=4000, noise=0.5, seed=0):
        rng = np.random.default_rng(seed)
        truth = rng.uniform(3.0, 11.0, n)
        predicted = truth + rng.normal(0, noise, n)
        endpoint = rng.integers(0, 3, n)
        return truth, predicted, endpoint, np.ones(n, dtype=bool)

    def test_a_perfect_predictor_scores_perfectly(self):
        truth, _, endpoint, exact = self._synthetic()
        report = _metrics(truth, truth, endpoint, exact)
        assert report["overall"]["rmse"] == pytest.approx(0.0, abs=1e-9)
        assert report["overall"]["concordance_index"] == pytest.approx(1.0, abs=1e-6)

    def test_rmse_is_reported_against_the_measured_noise_floor(self):
        truth, predicted, endpoint, exact = self._synthetic()
        report = _metrics(truth, predicted, endpoint, exact)
        assert report["Kd"]["noise_floor"] == 0.945
        assert report["Ki"]["noise_floor"] == 0.641
        assert report["IC50"]["noise_floor"] == 0.671
        for name in ("Kd", "Ki", "IC50"):
            entry = report[name]
            assert entry["rmse_over_floor"] == pytest.approx(
                entry["rmse"] / entry["noise_floor"]
            )

    def test_the_fold_error_is_the_chemist_facing_translation(self):
        """RMSE is in log units; a bench scientist thinks in fold."""
        truth, predicted, endpoint, exact = self._synthetic(noise=1.0)
        report = _metrics(truth, predicted, endpoint, exact)
        entry = report["Kd"]
        assert entry["typical_fold_error"] == pytest.approx(10 ** entry["rmse"])

    def test_a_coin_flip_predictor_scores_near_half_on_ranking(self):
        rng = np.random.default_rng(1)
        n = 4000
        truth = rng.uniform(3, 11, n)
        report = _metrics(truth, rng.uniform(3, 11, n), rng.integers(0, 3, n),
                          np.ones(n, dtype=bool))
        assert 0.45 < report["overall"]["concordance_index"] < 0.55

    def test_censored_rows_are_excluded_from_the_regression_metrics(self):
        truth, predicted, endpoint, exact = self._synthetic(n=2000)
        exact[:1000] = False
        report = _metrics(truth, predicted, endpoint, exact)
        assert report["overall"]["n"] == 1000
        assert report["censored"]["n"] == 1000

    def test_a_tiny_sample_reports_nothing_rather_than_noise(self):
        truth, predicted, endpoint, exact = self._synthetic(n=20)
        assert _metrics(truth, predicted, endpoint, exact)["overall"] is None


class TestQueryDataset:
    """Panel queries carry no labels, and the dataset must not need any."""

    def test_the_constructor_exists_and_is_label_free(self):
        import inspect

        from bindra.data.batching import PairDataset

        source = inspect.getsource(PairDataset.from_queries)
        assert "p_value=0.0" in source, "placeholders must be explicit"
        assert "sealed" in source.lower(), "the docstring must say labels are untouched"

    @pytest.mark.snapshot
    def test_it_builds_from_a_real_panel_query_table(self):
        from pathlib import Path

        import pandas as pd

        from bindra.data.batching import PairDataset
        from bindra.encoders.protein_esmc import (
            ESMC_600M_WEIGHTS_SHA256,
            EXPECTED_ESMC_600M_DIM,
            ESMCProteinEncoder,
        )
        from bindra.features.cache import ProteinFeatureCache

        panel = Path("data/benchmarks/202610/panel_strict_future/queries.parquet")
        if not panel.exists() or not Path("data/cache/proteins/index.json").exists():
            pytest.skip("panel or protein cache not present")
        metadata = ESMCProteinEncoder(n_trainable_layers=0).metadata(
            EXPECTED_ESMC_600M_DIM, weights_checksum=ESMC_600M_WEIGHTS_SHA256
        )
        queries = pd.read_parquet(panel).head(500)
        dataset = PairDataset.from_queries(
            "data/processed/202610",
            queries,
            protein_cache=ProteinFeatureCache("data/cache/proteins", metadata=metadata),
        )
        assert len(dataset) > 0
        assert "standardized_smiles" in dataset.frame.columns
        assert dataset.lengths.min() > 0


@pytest.mark.snapshot
class TestAgainstTheRealRun:
    def test_the_smoke_evaluation_wrote_a_usable_report(self):
        from pathlib import Path

        path = Path("runs/eval-v1-smoke.json")
        if not path.exists():
            pytest.skip("no evaluation report on disk")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["used_ema_weights"] is True
        assert payload["partition"]["metrics"]["overall"]["n"] > 1000
        # Coverage honesty survives serialisation.
        assert set(payload["panels_not_scored"]) == set(UNSCORED_PANELS)


class TestRowGrouping:
    """The grouping keys behind the clustered intervals and macro averages.

    These exist because the first implementation rebuilt a 235,473-entry dict
    once per observation. That is 5.5e10 operations on a real partition and it
    does not finish -- but on a toy fixture it looks fine, so only a test with
    a realistic row count and a shuffled frame catches it.
    """

    def _dataset(self, n, seed=0):
        import types

        import pandas as pd

        rng = np.random.default_rng(seed)
        digests = np.array([f"h{i:04d}" for i in range(200)])
        pick = rng.integers(0, len(digests), n)
        frame = pd.DataFrame(
            {
                "observation_id": [f"o{i}" for i in range(n)],
                "sequence_sha256": digests[pick],
            }
        )
        # Shuffled on purpose: the frame's row order is not the prediction
        # order, and a mapping that assumes it is would pass on sorted data.
        return types.SimpleNamespace(frame=frame.sample(frac=1.0, random_state=1)), digests[pick]

    def _groups_parquet(self, tmp_path):
        import pandas as pd

        digests = [f"h{i:04d}" for i in range(200)]
        pd.DataFrame(
            {"sequence_sha256": digests, "protein_group": [i % 37 for i in range(200)]}
        ).to_parquet(tmp_path / "protein_groups.parquet", index=False)
        return tmp_path

    def test_it_resolves_in_prediction_order_not_frame_order(self, tmp_path):
        from bindra.cli.evaluate import _row_grouping

        processed = self._groups_parquet(tmp_path)
        dataset, expected_digests = self._dataset(500)
        order = [f"o{i}" for i in range(500)]
        components, constructs, why = _row_grouping(processed, dataset, order)
        assert why is None
        assert list(constructs) == list(expected_digests)
        assert list(components) == [int(d[1:]) % 37 for d in expected_digests]

    def test_a_missing_component_refuses_rather_than_guessing(self, tmp_path):
        """A wrong clustering looks exactly like a right one in the output."""
        from bindra.cli.evaluate import _row_grouping

        processed = self._groups_parquet(tmp_path)
        dataset, _ = self._dataset(100)
        dataset.frame = dataset.frame.assign(sequence_sha256="unknown-digest")
        components, constructs, why = _row_grouping(
            processed, dataset, [f"o{i}" for i in range(100)]
        )
        assert components is None and constructs is None
        assert "no protein component" in why

    def test_it_is_not_quadratic_in_the_row_count(self, tmp_path):
        """Ten times the rows must not cost a hundred times the work."""
        import time

        from bindra.cli.evaluate import _row_grouping

        processed = self._groups_parquet(tmp_path)

        def timed(n):
            dataset, _ = self._dataset(n)
            order = [f"o{i}" for i in range(n)]
            start = time.perf_counter()
            assert _row_grouping(processed, dataset, order)[0] is not None
            return time.perf_counter() - start

        small, large = timed(2_000), timed(20_000)
        # Linear would be ~10x. Quadratic would be ~100x. The parquet read is a
        # fixed cost that dominates at these sizes, so the bound is loose on
        # purpose -- it is here to catch an order-of-growth regression, not to
        # police constant factors.
        assert large < max(small * 25, 2.0), (
            f"10x the rows cost {large / small:.0f}x the time; this looks quadratic"
        )
