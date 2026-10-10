"""`bindra.eval.metrics`: the intervals, and what they are allowed to claim.

The property worth protecting is the one the module exists for:

``test_a_dominant_cluster_widens_the_interval_over_the_iid_one``
    A cluster bootstrap over protein components must return a *wider* interval
    than resampling rows, whenever the clusters carry their own offsets. If it
    does not, the clustering is not doing anything and the interval is the
    i.i.d. one wearing a different label -- which is worse than no interval,
    because it looks like it accounted for the dependence.

The rest guard the honesty rules: a metric that is undefined says so rather
than emitting a number, and a macro average reports the coverage it rests on.
"""

from __future__ import annotations

import numpy as np
import pytest

from bindra.eval import metrics as M


def _clustered(n_clusters=150, big=2500, small=25, offset=0.9, noise=0.6, seed=0):
    """Clusters with their own offsets, one of them carrying most of the rows.

    This is the shape of the real corpus in miniature: the top 1% of targets
    carry a third of the observations, and different targets sit at different
    baselines.
    """
    rng = np.random.default_rng(seed)
    truth, predicted, clusters = [], [], []
    for cluster in range(n_clusters):
        count = big if cluster == 0 else small
        shift = rng.normal(0, offset)
        y = rng.uniform(3.0, 11.0, count)
        truth.append(y)
        predicted.append(y + shift + rng.normal(0, noise, count))
        clusters.append(np.full(count, cluster))
    return (
        np.concatenate(truth),
        np.concatenate(predicted),
        np.concatenate(clusters),
    )


class TestClusterBootstrap:
    def test_a_dominant_cluster_widens_the_interval_over_the_iid_one(self):
        """The whole reason this module exists, asserted rather than assumed."""
        truth, predicted, clusters = _clustered()
        report = M.bootstrap_metrics(
            truth, predicted, clusters, n_resamples=150, seed=7
        )
        for name in ("rmse", "mae", "spearman", "concordance_index"):
            entry = report["metrics"][name]
            assert entry["cluster"]["width"] > entry["iid"]["width"], (
                f"{name}: the clustered interval is not wider than the i.i.d. "
                "one, so the clustering is doing nothing"
            )
            assert entry["cluster_over_iid_width"] > 1.0

    def test_the_point_estimate_is_the_statistic_on_the_real_data(self):
        """A bootstrap reports the interval; it must not move the estimate."""
        truth, predicted, clusters = _clustered()
        report = M.bootstrap_metrics(
            truth, predicted, clusters, n_resamples=40, seed=1
        )
        assert report["metrics"]["rmse"]["point"] == pytest.approx(
            float(np.sqrt(np.mean((truth - predicted) ** 2)))
        )

    def test_the_interval_brackets_the_point_estimate(self):
        truth, predicted, clusters = _clustered()
        report = M.bootstrap_metrics(
            truth, predicted, clusters, n_resamples=150, seed=3
        )
        for name, entry in report["metrics"].items():
            low, high = entry["cluster"]["low"], entry["cluster"]["high"]
            assert low <= entry["point"] <= high, f"{name} falls outside its interval"

    def test_it_says_what_it_clustered_by_and_how_many(self):
        truth, predicted, clusters = _clustered(n_clusters=40)
        report = M.bootstrap_metrics(truth, predicted, clusters, n_resamples=20)
        assert report["n_clusters"] == 40
        assert report["clustered_by"] == "protein_group"

    def test_without_clusters_it_refuses_to_imply_it_clustered(self):
        truth, predicted, _ = _clustered(n_clusters=30)
        report = M.bootstrap_metrics(truth, predicted, None, n_resamples=20)
        assert report["clustered_by"] is None
        assert report["n_clusters"] is None
        for entry in report["metrics"].values():
            assert "cluster" not in entry

    def test_a_tiny_sample_is_undefined_rather_than_noisy(self):
        rng = np.random.default_rng(0)
        y = rng.uniform(3, 11, 12)
        report = M.bootstrap_metrics(y, y, np.arange(12), n_resamples=10)
        assert report["status"] == "undefined"

    def test_a_one_sided_cut_drops_the_binary_metrics_and_says_so(self):
        """Every row strong means AUROC has nothing to separate."""
        rng = np.random.default_rng(0)
        truth = rng.uniform(8.0, 11.0, 400)
        report = M.bootstrap_metrics(
            truth, truth + rng.normal(0, 0.5, 400), np.arange(400) % 40,
            n_resamples=10,
        )
        assert "auc_100nM_vs_rest" not in report["metrics"]
        assert "auc_100nM_vs_rest" in report["skipped"]


class TestGroupIndex:
    def test_a_draw_returns_whole_groups(self):
        labels = np.repeat(np.arange(10), 5)
        index = M._GroupIndex(labels)
        rows = index.draw(np.random.default_rng(0))
        # Every group appears a multiple of its size, never partially.
        counts = np.bincount(labels[rows], minlength=10)
        assert set(counts % 5) == {0}

    def test_rows_recovers_each_group_exactly(self):
        labels = np.array([2, 0, 1, 0, 2, 2])
        index = M._GroupIndex(labels)
        recovered = {
            int(labels[index.rows(g)][0]): sorted(index.rows(g).tolist())
            for g in range(index.n_groups)
        }
        assert recovered == {0: [1, 3], 1: [2], 2: [0, 4, 5]}


class TestMacroByGroup:
    def test_it_reports_the_coverage_the_average_rests_on(self):
        truth, predicted, clusters = _clustered()
        report = M.macro_by_group(truth, predicted, clusters)
        entry = report["metrics"]["rmse"]
        assert 0.0 < entry["observation_coverage"] <= 1.0
        assert entry["n_targets"] <= report["n_groups_total"]

    def test_the_macro_average_differs_from_the_pooled_one(self):
        """If they agreed there would be no reason to report both."""
        truth, predicted, clusters = _clustered()
        pooled = float(np.sqrt(np.mean((truth - predicted) ** 2)))
        macro = M.macro_by_group(truth, predicted, clusters)["metrics"]["rmse"]
        assert macro["macro"] != pytest.approx(pooled, abs=1e-6)

    def test_groups_too_small_for_a_rank_metric_are_excluded_not_invented(self):
        rng = np.random.default_rng(0)
        # 60 groups of 5 rows: enough rows overall, no group big enough to rank.
        y = rng.uniform(3, 11, 300)
        report = M.macro_by_group(y, y + rng.normal(0, 0.4, 300), np.repeat(np.arange(60), 5))
        assert report["metrics"]["spearman"]["status"] == "undefined"


class TestCensored:
    def test_a_prediction_inside_the_bound_is_not_penalised(self):
        """'At most 5.0' is satisfied by 2.0; only crossing counts."""
        predicted = np.array([2.0, 4.9, 5.0, 7.0])
        upper = np.full(4, 5.0)
        lower = np.full(4, np.nan)
        report = M.censored_metrics(predicted, lower, upper)
        assert report["overall"]["violation_rate"] == pytest.approx(0.25)
        assert report["overall"]["max_violation_log_units"] == pytest.approx(2.0)

    def test_both_bound_directions_are_reported_separately(self):
        predicted = np.array([9.0, 9.0, 1.0, 1.0])
        upper = np.array([5.0, 5.0, np.nan, np.nan])
        lower = np.array([np.nan, np.nan, 6.0, 6.0])
        report = M.censored_metrics(predicted, lower, upper)
        assert report["upper_bounded"]["n"] == 2
        assert report["lower_bounded"]["n"] == 2
        assert report["upper_bounded"]["violation_rate"] == 1.0
        assert report["lower_bounded"]["violation_rate"] == 1.0

    def test_no_censored_rows_is_undefined_not_zero(self):
        nan = np.full(5, np.nan)
        assert M.censored_metrics(np.zeros(5), nan, nan)["status"] == "undefined"


class TestClassification:
    def test_auprc_is_reported_with_the_baseline_it_needs(self):
        rng = np.random.default_rng(0)
        truth = rng.uniform(3, 11, 2000)
        report = M.classification_metrics(truth, truth + rng.normal(0, 0.5, 2000))
        assert report["auprc_baseline"] == pytest.approx(report["prevalence"])
        assert report["auprc"] > report["auprc_baseline"]
        assert report["auroc"] > 0.5

    def test_a_random_ranker_enriches_by_about_one(self):
        rng = np.random.default_rng(2)
        truth = rng.uniform(3, 11, 20000)
        report = M.classification_metrics(truth, rng.uniform(0, 1, 20000))
        assert report["enrichment"]["top_5%"]["enrichment_factor"] == pytest.approx(
            1.0, abs=0.35
        )

    def test_a_perfect_ranker_hits_the_ceiling_it_reports(self):
        rng = np.random.default_rng(3)
        truth = rng.uniform(3, 11, 5000)
        report = M.classification_metrics(truth, truth)
        top = report["enrichment"]["top_1%"]
        assert top["enrichment_factor"] == pytest.approx(
            top["max_possible_enrichment"], rel=1e-6
        )

    def test_one_sided_data_is_undefined(self):
        truth = np.full(200, 9.0)
        assert M.classification_metrics(truth, truth)["status"] == "undefined"


class TestCalibration:
    def test_a_perfect_predictor_has_unit_slope_and_no_bias(self):
        rng = np.random.default_rng(0)
        truth = rng.uniform(3, 11, 5000)
        report = M.calibration_curve(truth, truth)
        assert report["slope"] == pytest.approx(1.0, abs=1e-6)
        assert report["mean_absolute_bias"] == pytest.approx(0.0, abs=1e-6)

    def test_a_compressed_predictor_shows_a_slope_above_one(self):
        """Predictions shrunk toward the mean need a slope > 1 to undo."""
        rng = np.random.default_rng(0)
        truth = rng.uniform(3, 11, 5000)
        mean = truth.mean()
        report = M.calibration_curve(truth, mean + 0.5 * (truth - mean))
        assert report["slope"] == pytest.approx(2.0, rel=1e-6)

    def test_bins_are_equal_count(self):
        rng = np.random.default_rng(0)
        truth = rng.uniform(3, 11, 1000)
        report = M.calibration_curve(truth, truth, n_bins=10)
        sizes = {b["n"] for b in report["bins"]}
        assert max(sizes) - min(sizes) <= 1

    def test_too_few_rows_is_undefined(self):
        rng = np.random.default_rng(0)
        truth = rng.uniform(3, 11, 50)
        assert M.calibration_curve(truth, truth)["status"] == "undefined"


class TestHonesty:
    def test_undefined_carries_a_reason(self):
        entry = M.undefined("because", n=3)
        assert entry["status"] == "undefined" and entry["reason"] and entry["n"] == 3

    def test_no_function_returns_a_placeholder_number_when_undefined(self):
        """A caller must not be able to read a sentinel as a measurement."""
        nan = np.full(3, np.nan)
        for report in (
            M.bootstrap_metrics(np.zeros(3), np.zeros(3), None),
            M.censored_metrics(np.zeros(3), nan, nan),
            M.classification_metrics(np.zeros(3), np.zeros(3)),
            M.calibration_curve(np.zeros(3), np.zeros(3)),
        ):
            assert report["status"] == "undefined"
            assert not any(
                isinstance(v, float) and v in (0.0, -1.0) for v in report.values()
            )
