"""Affinity and ranking metrics, with intervals that respect the dependence.

Every number here is a *pooled* statistic over observations that are not
independent. 10,474 observed targets correspond to **572 effective independent
target units** (Kish), and the top 1% of targets carry 34.2% of all
observations -- see :class:`bindra.config.profiles.DependenceStructure`. Two
consequences, and this module exists to serve both:

1.  **Intervals come from a cluster bootstrap over protein components**, not
    from resampling rows. Resampling rows treats 181,178 observations as
    181,178 independent facts when they are closer to a few thousand, and
    returns an interval far too narrow to be honest. :func:`bootstrap_metrics`
    computes the clustered interval and the naive i.i.d. one side by side, so
    the gap is visible rather than asserted.

2.  **Per-target macro averages are reported beside the pooled ones.** A pooled
    metric here is dominated by roughly a hundred targets; a macro average over
    targets answers a different and usually more interesting question.
    :func:`macro_by_group` does that, and reports what it had to drop.

A metric that is not defined on the data it was handed returns a dict carrying
``status`` and a reason, never a placeholder number. That is the specification
clause this module owes:

    Priority 5: report per-endpoint and per-split; say when a metric is
    undefined rather than emitting a placeholder.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np

#: Enough rows for a rank statistic to mean anything. Matches the guard the
#: evaluate CLI already applied before this module existed.
MIN_ROWS = 30

#: Enough rows for RMSE/MAE on a single target. Lower than MIN_ROWS because a
#: mean squared error over ten points is far better behaved than a rank
#: correlation over ten points.
MIN_ROWS_REGRESSION = 10

#: Resamples for the bootstrap. 1,000 is enough for a 95% percentile interval;
#: the Monte-Carlo error on the endpoints is then well under the width.
BOOTSTRAP_RESAMPLES = 1_000

#: Two-sided, so a 95% interval.
BOOTSTRAP_ALPHA = 0.05

#: The potency cut everything binary is defined at: pX >= 7.0 is 100 nM.
STRONG_THRESHOLD = 7.0


def undefined(reason: str, **context) -> dict:
    """The return value for a metric that does not exist on this data."""
    return {"status": "undefined", "reason": reason, **context}


def _is_defined(entry: object) -> bool:
    """True only for a mapping that is present and not flagged undefined.

    ``None`` means the section was never computed, which is not the same as a
    computed-and-undefined one and must not read as usable either.
    """
    return isinstance(entry, Mapping) and entry.get("status") != "undefined"


# -- the statistics themselves ---------------------------------------------
#
# Each takes (truth, predicted) and returns a float, so the bootstrap can treat
# them uniformly. They are deliberately plain: anything clever here would have
# to be re-verified inside the resample loop.


def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p)))


def _pearson(y: np.ndarray, p: np.ndarray) -> float:
    from scipy.stats import pearsonr

    return float(pearsonr(y, p).statistic)


def _spearman(y: np.ndarray, p: np.ndarray) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(y, p).statistic)


def _concordance_index(y: np.ndarray, p: np.ndarray) -> float:
    """The chance a randomly chosen pair is ordered correctly.

    ``(1 + tau_b) / 2``. The tie-corrected tau is the right one: a bootstrap
    resample draws the same row more than once, so ties are guaranteed even
    when the original data had none.
    """
    from scipy.stats import kendalltau

    return float((1.0 + kendalltau(y, p).statistic) / 2.0)


def _auroc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y >= STRONG_THRESHOLD, p))


def _auprc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y >= STRONG_THRESHOLD, p))


#: Metric name -> callable. Order is the reporting order.
STATISTICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "rmse": _rmse,
    "mae": _mae,
    "concordance_index": _concordance_index,
    "spearman": _spearman,
    "pearson": _pearson,
    "auc_100nM_vs_rest": _auroc,
    "auprc_100nM_vs_rest": _auprc,
}

#: Metrics that need both classes present, so they are skipped when the cut
#: puts everything on one side.
NEEDS_BOTH_CLASSES = ("auc_100nM_vs_rest", "auprc_100nM_vs_rest")


# -- resampling -------------------------------------------------------------


class _GroupIndex:
    """Row indices grouped by label, laid out for ragged concatenation.

    Rows are sorted once by label so that each group is a contiguous run. A
    resample then draws group *slots* and the rows follow by arithmetic, with
    no Python-level loop over the groups -- which matters, because that loop
    would run n_resamples x n_groups times and the groups number in the
    thousands.
    """

    def __init__(self, labels: np.ndarray) -> None:
        labels = np.asarray(labels)
        order = np.argsort(labels, kind="stable")
        in_order = labels[order]
        boundaries = np.flatnonzero(
            np.concatenate(([True], in_order[1:] != in_order[:-1]))
        )
        self.order = order
        self.starts = boundaries
        self.counts = np.diff(np.concatenate((boundaries, [len(labels)])))
        self.n_groups = len(boundaries)

    def rows(self, group: int) -> np.ndarray:
        start = self.starts[group]
        return self.order[start : start + self.counts[group]]

    def draw(self, rng: np.random.Generator) -> np.ndarray:
        """One cluster-bootstrap resample, as row indices into the original."""
        drawn = rng.integers(0, self.n_groups, self.n_groups)
        lengths = self.counts[drawn]
        total = int(lengths.sum())
        if total == 0:
            return np.empty(0, dtype=np.int64)
        # Offset of each drawn run in the output, subtracted from its start so
        # that a single arange lands every row in the right place.
        offsets = np.concatenate(([0], np.cumsum(lengths)[:-1]))
        base = np.repeat(self.starts[drawn] - offsets, lengths)
        return self.order[base + np.arange(total)]


def _percentile_interval(values: Sequence[float], alpha: float) -> dict:
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if finite.size < 2:
        return undefined(
            "too few finite bootstrap replicates to form an interval",
            n_finite=int(finite.size),
        )
    low, high = np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "low": float(low),
        "high": float(high),
        "width": float(high - low),
        "n_finite": int(finite.size),
    }


def bootstrap_metrics(
    truth: np.ndarray,
    predicted: np.ndarray,
    clusters: np.ndarray | None,
    *,
    statistics: Mapping[str, Callable] | None = None,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    alpha: float = BOOTSTRAP_ALPHA,
    seed: int = 20261010,
    include_iid: bool = True,
) -> dict:
    """Point estimates with clustered and i.i.d. percentile intervals.

    Both resampling schemes run the *same* metric set on their own replicates,
    so the only difference between the two intervals is what was resampled --
    which is the comparison worth making.

    ``clusters`` is one label per row, normally a protein component id. Passing
    ``None`` reports point estimates and the i.i.d. interval only, and says so
    in the result rather than presenting the i.i.d. interval as the clustered
    one.
    """
    truth = np.asarray(truth, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    n = len(truth)
    statistics = dict(statistics or STATISTICS)

    if n < MIN_ROWS:
        return undefined(f"fewer than {MIN_ROWS} rows", n=int(n))

    # Drop the binary metrics up front when the cut has nothing on one side,
    # rather than letting every resample raise.
    strong = int((truth >= STRONG_THRESHOLD).sum())
    skipped: dict[str, str] = {}
    if not (MIN_ROWS < strong < n - MIN_ROWS):
        for name in NEEDS_BOTH_CLASSES:
            if name in statistics:
                statistics.pop(name)
                skipped[name] = (
                    f"only {strong:,} of {n:,} rows are at or above "
                    f"pX {STRONG_THRESHOLD}; the cut does not split this sample"
                )

    point = {name: float(fn(truth, predicted)) for name, fn in statistics.items()}

    index = _GroupIndex(clusters) if clusters is not None else None
    wanted = (["cluster"] if index is not None else []) + (
        ["iid"] if include_iid else []
    )

    collected: dict[str, dict[str, list[float]]] = {}
    for offset, scheme in enumerate(wanted):
        # A separate stream per scheme, so the clustered and i.i.d. intervals
        # are not correlated through a shared sequence of draws.
        stream = np.random.default_rng([seed, offset])
        replicates: dict[str, list[float]] = {name: [] for name in statistics}
        for _ in range(n_resamples):
            rows = (
                index.draw(stream) if scheme == "cluster" else stream.integers(0, n, n)
            )
            if len(rows) < MIN_ROWS:
                continue
            y, p = truth[rows], predicted[rows]
            for name, fn in statistics.items():
                try:
                    replicates[name].append(float(fn(y, p)))
                except ValueError:
                    # A replicate that lost one class entirely, for instance.
                    # Dropped and counted via n_finite, never substituted.
                    continue
        collected[scheme] = replicates

    metrics: dict[str, dict] = {}
    for name in statistics:
        entry: dict = {"point": point[name]}
        for scheme, replicates in collected.items():
            entry[scheme] = _percentile_interval(replicates[name], alpha)
        cluster_ci, iid_ci = entry.get("cluster"), entry.get("iid")
        if _is_defined(cluster_ci) and _is_defined(iid_ci) and iid_ci["width"] > 0:
            # How much wider the honest interval is. Above 1 is the evidence
            # that the i.i.d. interval understates the uncertainty here.
            entry["cluster_over_iid_width"] = float(
                cluster_ci["width"] / iid_ci["width"]
            )
        metrics[name] = entry

    result = {
        "n": int(n),
        "n_resamples": int(n_resamples),
        "alpha": float(alpha),
        "interval": f"{100 * (1 - alpha):.0f}% percentile",
        "clustered_by": "protein_group" if clusters is not None else None,
        "n_clusters": int(index.n_groups) if index is not None else None,
        "metrics": metrics,
    }
    if skipped:
        result["skipped"] = skipped
    return result


# -- macro averages ---------------------------------------------------------


def macro_by_group(
    truth: np.ndarray,
    predicted: np.ndarray,
    groups: np.ndarray,
    *,
    label: str = "target",
    min_rows_regression: int = MIN_ROWS_REGRESSION,
    min_rows_ranking: int = MIN_ROWS,
) -> dict:
    """Unweighted mean of each statistic computed *within* a group.

    The pooled metric answers "how well does the model do on an average
    observation", which here means an average dominated by the hundred or so
    targets that carry most of the data. This answers "how well does it do on
    an average target", which is usually the question being asked.

    Groups too small for a statistic are excluded from *that* statistic rather
    than from the whole table, and the coverage each average rests on is
    reported beside it -- a macro average over 300 of 6,400 targets is a
    different claim from one over all of them.
    """
    truth = np.asarray(truth, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    groups = np.asarray(groups)
    n = len(truth)
    if n < MIN_ROWS:
        return undefined(f"fewer than {MIN_ROWS} rows", n=int(n))

    index = _GroupIndex(groups)
    minimum = {
        name: (min_rows_regression if name in ("rmse", "mae") else min_rows_ranking)
        for name in STATISTICS
    }
    collected: dict[str, list[float]] = {name: [] for name in STATISTICS}
    covered: dict[str, int] = {name: 0 for name in STATISTICS}

    for group in range(index.n_groups):
        rows = index.rows(group)
        y, p = truth[rows], predicted[rows]
        strong = int((y >= STRONG_THRESHOLD).sum())
        both_classes = 0 < strong < len(y)
        for name, fn in STATISTICS.items():
            if len(rows) < minimum[name]:
                continue
            if name in NEEDS_BOTH_CLASSES and not both_classes:
                continue
            try:
                value = float(fn(y, p))
            except ValueError:
                continue
            if not np.isfinite(value):
                # A group whose truth is constant has no rank correlation.
                # Skipped, and the skip shows up as reduced coverage.
                continue
            collected[name].append(value)
            covered[name] += len(rows)

    metrics: dict[str, dict] = {}
    for name, values in collected.items():
        if not values:
            metrics[name] = undefined(
                f"no {label} had enough rows for this statistic",
                minimum_rows=minimum[name],
            )
            continue
        array = np.asarray(values)
        metrics[name] = {
            "macro": float(array.mean()),
            "median": float(np.median(array)),
            f"n_{label}s": int(array.size),
            "observation_coverage": float(covered[name] / n),
            "minimum_rows": int(minimum[name]),
        }
    return {
        "grouped_by": label,
        "n": int(n),
        "n_groups_total": int(index.n_groups),
        "metrics": metrics,
    }


# -- censored labels --------------------------------------------------------


def censored_metrics(
    predicted: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
) -> dict:
    """What the model does with the labels every regression metric drops.

    Roughly a fifth of each partition is censored -- "weaker than 10 uM" and
    the like. Those rows have no point label, so RMSE cannot see them, and a
    report built only from exact rows silently omits a fifth of the evidence.
    What *is* answerable is whether the prediction respects the bound, and by
    how much it misses when it does not.

    The asymmetry is the whole point and is not reimplemented here: a
    prediction consistent with "at most 5.0" is consistent, however far below
    5.0 it sits. :func:`bindra.training.losses.affinity.censoring_violation` is
    the same function the training loss uses.
    """
    import torch

    from bindra.training.losses.affinity import censoring_violation

    predicted = np.asarray(predicted, dtype=float)
    lower_bound = np.asarray(lower_bound, dtype=float)
    upper_bound = np.asarray(upper_bound, dtype=float)

    has_lower = ~np.isnan(lower_bound)
    has_upper = ~np.isnan(upper_bound)
    censored = has_lower | has_upper
    n = int(censored.sum())
    if n == 0:
        return undefined("no censored rows in this sample", n=0)

    violation = (
        censoring_violation(
            torch.from_numpy(predicted),
            torch.from_numpy(lower_bound),
            torch.from_numpy(upper_bound),
        )
        .numpy()
        .astype(float)
    )

    def block(mask: np.ndarray, direction: str) -> dict | None:
        count = int(mask.sum())
        if count == 0:
            return None
        values = violation[mask]
        crossed = values > 0
        return {
            "n": count,
            "direction": direction,
            "violation_rate": float(crossed.mean()),
            "mean_violation_log_units": float(values.mean()),
            "mean_violation_when_crossed": (
                float(values[crossed].mean()) if crossed.any() else 0.0
            ),
            "p90_violation_log_units": float(np.percentile(values, 90)),
            "max_violation_log_units": float(values.max()),
        }

    report: dict = {
        "n": n,
        "note": (
            "violation is the distance outside the half-line the bound allows, "
            "in log-10 units; zero means the prediction is consistent with the "
            "bound, however far inside it sits"
        ),
        "overall": block(censored, "either"),
    }
    upper_only = block(has_upper, "at most (weaker than the bound)")
    lower_only = block(has_lower, "at least (stronger than the bound)")
    if upper_only:
        report["upper_bounded"] = upper_only
    if lower_only:
        report["lower_bounded"] = lower_only
    return report


# -- the binary call --------------------------------------------------------


def classification_metrics(
    truth: np.ndarray,
    predicted: np.ndarray,
    *,
    threshold: float = STRONG_THRESHOLD,
    fractions: Sequence[float] = (0.01, 0.05),
) -> dict:
    """Strong-vs-rest at the 100 nM cut: AUROC, AUPRC and early enrichment.

    AUPRC is reported with its own baseline. Average precision is bounded below
    by the positive rate, not by 0.5, so an AUPRC of 0.6 is excellent at 10%
    prevalence and poor at 55%; quoting it without the prevalence beside it
    invites exactly that mistake.

    Enrichment answers the screening question instead of the ranking one: among
    the top 1% the model would pick, how many times the background rate are
    actually strong binders.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    truth = np.asarray(truth, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    n = len(truth)
    if n < MIN_ROWS:
        return undefined(f"fewer than {MIN_ROWS} rows", n=int(n))

    positive = truth >= threshold
    n_positive = int(positive.sum())
    if not (0 < n_positive < n):
        return undefined(
            f"every row falls on one side of pX {threshold}",
            n=int(n),
            n_positive=n_positive,
        )

    prevalence = n_positive / n
    nanomolar = 10 ** (9 - threshold)
    report: dict = {
        "n": int(n),
        "threshold_pX": float(threshold),
        "threshold_note": f"pX >= {threshold} is {nanomolar:.0f} nM or stronger",
        "n_positive": n_positive,
        "prevalence": float(prevalence),
        "auroc": float(roc_auc_score(positive, predicted)),
        "auroc_baseline": 0.5,
        "auprc": float(average_precision_score(positive, predicted)),
        # The floor AUPRC cannot go below; the number means nothing without it.
        "auprc_baseline": float(prevalence),
    }

    order = np.argsort(-predicted, kind="stable")
    ranked = positive[order]
    enrichment: dict[str, dict] = {}
    for fraction in fractions:
        top = int(round(fraction * n))
        if top < MIN_ROWS:
            enrichment[f"top_{fraction:.0%}"] = undefined(
                f"the top {fraction:.0%} is only {top} rows", n_top=top
            )
            continue
        hits = int(ranked[:top].sum())
        enrichment[f"top_{fraction:.0%}"] = {
            "n_top": top,
            "n_hits": hits,
            "hit_rate": float(hits / top),
            "enrichment_factor": float((hits / top) / prevalence),
            # What a perfect ranker would score, which is not always large:
            # once the top slice is bigger than the positive set, even a
            # perfect model cannot fill it with hits.
            "max_possible_enrichment": float(min(1.0, n_positive / top) / prevalence),
        }
    report["enrichment"] = enrichment
    return report


# -- calibration ------------------------------------------------------------


def calibration_curve(
    truth: np.ndarray,
    predicted: np.ndarray,
    *,
    n_bins: int = 10,
) -> dict:
    """Is a predicted pX of 8 actually an 8 on average?

    This is calibration of a *magnitude*, not of a probability. The head emits
    a real-valued pX and nothing resembling a confidence, so there is no
    reliability diagram in the usual sense and no expected-calibration-error in
    the usual sense. What can be asked is whether the predictions are
    systematically compressed or shifted: bin by predicted value, and compare
    each bin's mean prediction against its mean observation.

    Regression to the mean makes a slope below 1 the expected finding for a
    model trained on squared error. Reporting it is the point -- it tells a
    reader that a predicted 9 is usually a weaker binder than 9.
    """
    truth = np.asarray(truth, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    n = len(truth)
    if n < n_bins * MIN_ROWS:
        return undefined(
            f"fewer than {n_bins * MIN_ROWS} rows for {n_bins} bins",
            n=int(n),
            n_bins=int(n_bins),
        )

    # Equal-count bins, so every point on the curve carries the same weight.
    # Equal-width bins would put a handful of rows in the extreme bins and
    # invite reading noise as miscalibration.
    order = np.argsort(predicted, kind="stable")
    bins = []
    for rows in np.array_split(order, n_bins):
        if len(rows) == 0:
            continue
        y, p = truth[rows], predicted[rows]
        observed = float(y.mean())
        bins.append(
            {
                "n": int(len(rows)),
                "predicted_mean": float(p.mean()),
                "predicted_low": float(p.min()),
                "predicted_high": float(p.max()),
                "observed_mean": observed,
                "observed_standard_error": (
                    float(y.std(ddof=1) / np.sqrt(len(rows)))
                    if len(rows) > 1
                    else float("nan")
                ),
                "bias": float(p.mean() - observed),
            }
        )

    # Slope and intercept of observed on predicted, fitted over the raw rows
    # rather than the bin means: binning first would throw away the within-bin
    # spread the fit should see.
    slope, intercept = np.polyfit(predicted, truth, 1)
    weights = np.array([b["n"] for b in bins], dtype=float)
    absolute_bias = np.array([abs(b["bias"]) for b in bins], dtype=float)
    return {
        "n": int(n),
        "n_bins": len(bins),
        "binning": "equal-count on the predicted value",
        "kind": "magnitude calibration of a real-valued pX, not a probability",
        "slope": float(slope),
        "intercept": float(intercept),
        "slope_ideal": 1.0,
        # One number for "how far off is the curve", weighted by bin size.
        "mean_absolute_bias": float((absolute_bias * weights).sum() / weights.sum()),
        "max_absolute_bias": float(absolute_bias.max()),
        "bins": bins,
    }


__all__ = [
    "BOOTSTRAP_ALPHA",
    "BOOTSTRAP_RESAMPLES",
    "MIN_ROWS",
    "MIN_ROWS_REGRESSION",
    "NEEDS_BOTH_CLASSES",
    "STATISTICS",
    "STRONG_THRESHOLD",
    "bootstrap_metrics",
    "calibration_curve",
    "censored_metrics",
    "classification_metrics",
    "macro_by_group",
    "undefined",
]
