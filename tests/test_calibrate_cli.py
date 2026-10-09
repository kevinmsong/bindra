"""``bindra calibrate``: the guards, the censoring rule, and the coverage check.

The three uncertainty modules have their own tests in ``test_uncertainty.py``;
this file is about the command that drives them, and specifically about what it
**refuses**. Four tests carry the weight:

``test_three_passes_over_one_checkpoint_are_refused``
    The specification's own clause, arriving through the CLI rather than the
    library. Copying one checkpoint to three filenames is the easy way to
    produce a convincing spread from nothing.
``test_members_from_different_splits_are_refused``
    Not in the specification, and the quieter failure. Each member is a valid
    model and the arithmetic succeeds; what is gone is the exchangeability the
    guarantee rests on, and the interval looks identical either way.
``test_censored_rows_are_excluded_and_counted``
    ``|y - mean|`` needs a ``y``. Imputing a censored record to its bound would
    tighten every interval on the one partition that exists to measure honestly.
``test_a_short_endpoint_reports_a_status_rather_than_an_interval``
    The fallback that fabricates a finite interval is one line away and its
    output is indistinguishable from a valid one.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from bindra.cli.calibrate import (
    CalibrationRefused,
    _assert_one_split,
    _calibrate_endpoints,
    _members,
    build_parser,
)
from bindra.uncertainty import conformal, ensemble


def write_checkpoint(
    path: Path, *, seed: int, split_id: str = "random_pair-5ef33512", scale: float = 1.0
) -> Path:
    """A checkpoint with just enough of the real shape to be identified.

    ``describe_member`` reads only the model tensors and ``config['seed']``, and
    ``_assert_one_split`` only ``provenance['split_id']`` -- so a full model is
    not needed to exercise either, and building one would make the test about
    something else.
    """
    generator = torch.Generator().manual_seed(seed)
    torch.save(
        {
            "model": {"w": torch.randn(4, 4, generator=generator) * scale},
            "config": {"seed": seed},
            "provenance": {"split_id": split_id, "split_regime": "random_pair"},
            "epoch": 3,
        },
        path,
    )
    return path


@pytest.fixture
def three(tmp_path) -> list[Path]:
    return [
        write_checkpoint(tmp_path / f"m{i}.pt", seed=100 + i) for i in range(3)
    ]


class TestTheEnsembleGuards:
    def test_three_distinct_runs_are_accepted(self, three):
        members = _members(three)
        assert len(members) == 3
        assert len({m.weight_digest for m in members}) == 3
        assert {m.seed for m in members} == {100, 101, 102}

    def test_three_passes_over_one_checkpoint_are_refused(self, tmp_path):
        """Copying one file to three names is the cheapest way to fake a spread."""
        import shutil

        original = write_checkpoint(tmp_path / "a.pt", seed=1)
        copies = [original]
        for name in ("b.pt", "c.pt"):
            shutil.copy(original, tmp_path / name)
            copies.append(tmp_path / name)
        with pytest.raises(ensemble.EnsembleError, match="identical weights"):
            _members(copies)

    def test_members_sharing_a_seed_are_refused(self, tmp_path):
        paths = [
            write_checkpoint(tmp_path / "a.pt", seed=7, scale=1.0),
            write_checkpoint(tmp_path / "b.pt", seed=7, scale=2.0),
        ]
        # Different weights (scale differs) but the same seed: same
        # initialisation and same data order, so not independent runs.
        with pytest.raises(ensemble.EnsembleError, match="share seed"):
            _members(paths)

    def test_one_member_is_refused(self, tmp_path):
        with pytest.raises(ensemble.EnsembleError, match="at least 2"):
            _members([write_checkpoint(tmp_path / "a.pt", seed=1)])

    def test_a_missing_checkpoint_is_named(self, tmp_path):
        with pytest.raises(CalibrationRefused, match="no checkpoint at"):
            _members([tmp_path / "absent.pt"])


class TestTheSplitGuard:
    def test_members_from_one_split_are_accepted(self, three):
        provenance = _assert_one_split(three)
        assert provenance["split_id"] == "random_pair-5ef33512"

    def test_members_from_different_splits_are_refused(self, tmp_path):
        """Each model is valid; what is gone is exchangeability.

        A quantile fitted across models that held out different observations has
        nothing to rest on, and the interval it produces looks exactly like a
        sound one.
        """
        paths = [
            write_checkpoint(tmp_path / "a.pt", seed=1, split_id="random_pair-aaaa"),
            write_checkpoint(tmp_path / "b.pt", seed=2, split_id="random_pair-bbbb"),
        ]
        with pytest.raises(CalibrationRefused, match="different splits"):
            _assert_one_split(paths)

    def test_the_refusal_names_which_file_came_from_which_split(self, tmp_path):
        paths = [
            write_checkpoint(tmp_path / "a.pt", seed=1, split_id="split-aaaa"),
            write_checkpoint(tmp_path / "b.pt", seed=2, split_id="split-bbbb"),
        ]
        with pytest.raises(CalibrationRefused) as caught:
            _assert_one_split(paths)
        message = str(caught.value)
        assert "a.pt" in message and "b.pt" in message
        assert "split-aaaa" in message and "split-bbbb" in message


class TestPerEndpointCalibration:
    @pytest.fixture
    def exchangeable(self):
        """4,500 exact observations, 1,500 per endpoint, residuals ~ N(0, 0.5)."""
        rng = np.random.default_rng(20261008)
        n = 4500
        truth = rng.normal(size=n)
        mean = truth + rng.normal(scale=0.5, size=n)
        endpoint = np.repeat([0, 1, 2], n // 3)
        return truth, mean, endpoint, np.ones(n, dtype=bool)

    def test_each_endpoint_gets_its_own_quantile(self, exchangeable):
        """Noise floors differ by 1.5x across endpoints; one quantile cannot fit."""
        report = _calibrate_endpoints(*exchangeable, 0.10, 1)
        assert set(report) == {"Kd", "Ki", "IC50"}
        for entry in report.values():
            assert entry["quantile"]["status"] == "calibrated"
            assert entry["quantile"]["q"] > 0

    def test_measured_coverage_lands_near_nominal_on_exchangeable_data(
        self, exchangeable
    ):
        """The point of measuring it rather than asserting the guarantee."""
        report = _calibrate_endpoints(*exchangeable, 0.10, 1)
        for name, entry in report.items():
            empirical = entry["measured_coverage"]["empirical"]
            assert 0.84 < empirical < 0.96, (name, empirical)

    def test_the_coverage_check_fits_and_scores_on_disjoint_halves(
        self, exchangeable
    ):
        report = _calibrate_endpoints(*exchangeable, 0.10, 1)
        for entry in report.values():
            measured = entry["measured_coverage"]
            assert measured["n_fit"] + measured["n_scored"] == entry["n_exact"]

    def test_censored_rows_are_excluded_and_counted(self):
        """|y - mean| needs a y, and a bound is not one."""
        rng = np.random.default_rng(5)
        n = 600
        truth = rng.normal(size=n)
        mean = truth + rng.normal(scale=0.4, size=n)
        endpoint = np.zeros(n, dtype=np.int64)
        is_exact = np.ones(n, dtype=bool)
        is_exact[::3] = False  # a third censored

        report = _calibrate_endpoints(truth, mean, endpoint, is_exact, 0.10, 1)
        entry = report["Kd"]
        assert entry["n_observations"] == n
        assert entry["n_exact"] == int(is_exact.sum())
        assert entry["n_censored_excluded"] == n - int(is_exact.sum())
        assert entry["quantile"]["n_calibration"] == int(is_exact.sum())

    def test_a_censored_row_cannot_tighten_the_interval(self):
        """The failure imputing-to-the-bound would cause, stated as a test.

        The censored rows here are given predictions far from their bounds. If
        they were imputed and scored, the quantile would move; excluded, it is
        exactly the quantile of the exact rows alone.
        """
        rng = np.random.default_rng(11)
        n = 900
        truth = rng.normal(size=n)
        mean = truth + rng.normal(scale=0.3, size=n)
        endpoint = np.zeros(n, dtype=np.int64)
        is_exact = np.ones(n, dtype=bool)
        is_exact[::2] = False
        mean[~is_exact] += 50.0  # wildly wrong, and must not count

        report = _calibrate_endpoints(truth, mean, endpoint, is_exact, 0.10, 1)
        alone = conformal.calibrate(
            truth[is_exact], mean[is_exact], endpoint="Kd", alpha=0.10
        )
        assert report["Kd"]["quantile"]["q"] == pytest.approx(alone.q)
        assert report["Kd"]["quantile"]["q"] < 1.0

    def test_a_short_endpoint_reports_a_status_rather_than_an_interval(self):
        """n=5 at alpha=0.10 asks for the 6th smallest of 5. It does not exist."""
        truth = np.zeros(5)
        report = _calibrate_endpoints(
            truth, truth, np.zeros(5, dtype=np.int64), np.ones(5, bool), 0.10, 1
        )
        quantile = report["Kd"]["quantile"]
        assert quantile["status"] == "insufficient_calibration"
        assert "q" not in quantile
        assert quantile["required"] == 6 and quantile["n_calibration"] == 5

    def test_an_endpoint_with_no_exact_labels_says_so(self):
        n = 30
        report = _calibrate_endpoints(
            np.zeros(n), np.zeros(n), np.zeros(n, dtype=np.int64),
            np.zeros(n, dtype=bool), 0.10, 1,
        )
        assert report["Kd"]["status"] == "no_exact_labels"
        assert report["Kd"]["n_censored_excluded"] == n

    def test_the_nanomolar_interval_is_the_right_way_round(self, exchangeable):
        """The inverse logarithm reverses the ends; the spec calls this out."""
        report = _calibrate_endpoints(*exchangeable, 0.10, 1)
        band = report["Kd"]["interval_at_10nm"]
        assert band["low_nm"] < band["high_nm"]

    def test_a_tighter_alpha_widens_the_interval(self, exchangeable):
        loose = _calibrate_endpoints(*exchangeable, 0.20, 1)
        tight = _calibrate_endpoints(*exchangeable, 0.05, 1)
        assert tight["Kd"]["quantile"]["q"] > loose["Kd"]["quantile"]["q"]

    def test_the_coverage_split_is_reproducible_from_the_seed(self, exchangeable):
        first = _calibrate_endpoints(*exchangeable, 0.10, 42)
        again = _calibrate_endpoints(*exchangeable, 0.10, 42)
        other = _calibrate_endpoints(*exchangeable, 0.10, 43)
        assert first == again
        assert (
            first["Kd"]["measured_coverage"]["empirical"]
            != other["Kd"]["measured_coverage"]["empirical"]
        )


class TestTheCommandSurface:
    def test_calibrate_is_registered(self):
        from bindra.cli.__main__ import COMMANDS

        assert "calibrate" in COMMANDS

    def test_checkpoints_are_required(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_the_defaults_are_the_specification_starting_point(self):
        args = build_parser().parse_args(["--checkpoints", "a.pt", "b.pt"])
        assert args.alpha == 0.10
        assert args.regime == "random_pair"
        assert args.checkpoints == ["a.pt", "b.pt"]
        # The similarity pass defaults to cpu so it cannot quietly contend with
        # a training run for the GPU, and it measured faster there anyway.
        assert args.similarity_device == "cpu"

    def test_the_domain_pass_is_capped_by_default(self):
        """Uncapped it is 158G Tanimoto comparisons and about 6.3 hours.

        A command must not start a six-hour job because its default was None.
        The cap is a sample, so it is printed and recorded rather than applied
        quietly -- see ``sampled`` in the report payload.
        """
        args = build_parser().parse_args(["--checkpoints", "a.pt", "b.pt"])
        assert args.domain_sample == 5_000
        assert build_parser().parse_args(
            ["--checkpoints", "a.pt", "b.pt", "--domain-sample", "0"]
        ).domain_sample == 0

    def test_the_report_shape_is_json_serialisable(self, exchangeable=None):
        rng = np.random.default_rng(3)
        n = 900
        truth = rng.normal(size=n)
        mean = truth + rng.normal(scale=0.5, size=n)
        report = _calibrate_endpoints(
            truth, mean, np.repeat([0, 1, 2], n // 3), np.ones(n, bool), 0.10, 1
        )
        assert json.loads(json.dumps(report, default=str))

    def test_the_module_refuses_to_rank_members(self):
        """A per-member table on a held-apart partition is a model selection.

        Checked on the source rather than the output, because the absence of a
        ranking is the property and an output cannot show an absence.
        """
        source = Path("bindra/cli/calibrate.py").read_text(encoding="utf-8")
        assert "does not select a model" in source
        for forbidden in ("argmin", "argmax", "sort(", "best_member"):
            assert forbidden not in source, forbidden
