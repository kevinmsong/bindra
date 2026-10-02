"""Endpoint heads and the censored affinity loss.

The tests that would catch a wrong scientific claim rather than a crash:

``test_a_prediction_inside_the_bound_costs_nothing``
    The definition of not treating a bound as exact. 16.6% of the corpus is
    censored; if these rows are scored as point labels the model is pulled toward
    assay detection limits instead of affinities, and the weak end is exactly
    where the fishing and selectivity panels live.
``test_an_unmeasured_endpoint_never_contributes``
    The specification's "an unmeasured endpoint must never receive a fabricated
    zero label", as an assertion on gradients.
``test_tier_b_cannot_reach_the_primary_term``
    The curation decision was "auxiliary objectives only". Structural, not a weight.
``test_a_bound_in_the_wrong_field_is_rejected``
    Censoring reverses under the negative log. Getting the direction backwards
    trains the model toward the opposite of the measurement, with a normal-looking
    loss curve.
"""

from __future__ import annotations

import math

import pytest
import torch

from bindra.config.profiles import ErrorModel
from bindra.data.endpoints import Endpoint
from bindra.heads.affinity import (
    ENDPOINT_INDEX,
    ENDPOINT_ORDER,
    AffinityHeadConfig,
    AffinityHeadError,
    AffinityHeads,
    AffinityPrediction,
    endpoint_indices,
)
from bindra.training.losses.affinity import (
    AffinityLossConfig,
    AffinityLossError,
    AffinityTargets,
    affinity_loss,
    censoring_violation,
    endpoint_sigma,
    noise_floor_table,
)

NAN = float("nan")


def _targets(rows: list[dict]) -> AffinityTargets:
    """Build targets from readable row dicts."""
    return AffinityTargets(
        endpoint_index=torch.tensor([ENDPOINT_INDEX[r["endpoint"]] for r in rows]),
        p_value=torch.tensor([r["p_value"] for r in rows], dtype=torch.float32),
        is_exact=torch.tensor([r.get("is_exact", True) for r in rows]),
        lower_bound=torch.tensor(
            [r.get("lower_bound", NAN) for r in rows], dtype=torch.float32
        ),
        upper_bound=torch.tensor(
            [r.get("upper_bound", NAN) for r in rows], dtype=torch.float32
        ),
        is_primary=torch.tensor([r.get("is_primary", True) for r in rows]),
    )


def _prediction(values: list[list[float]]) -> AffinityPrediction:
    return AffinityPrediction(values=torch.tensor(values, dtype=torch.float32))


class TestEndpointIndexing:
    def test_the_column_order_is_fixed_and_declared(self):
        assert ENDPOINT_ORDER == (Endpoint.KD, Endpoint.KI, Endpoint.IC50)
        assert ENDPOINT_INDEX == {"Kd": 0, "Ki": 1, "IC50": 2}

    def test_curated_strings_map_to_columns(self):
        assert endpoint_indices(["IC50", "Kd", "Ki"]).tolist() == [2, 0, 1]

    def test_an_unknown_endpoint_raises_rather_than_mapping_to_zero(self):
        """Mapping to 0 would train the pKd head against an EC50."""
        with pytest.raises(AffinityHeadError, match="EC50"):
            endpoint_indices(["EC50"])


class TestHeads:
    def test_one_output_per_endpoint(self):
        heads = AffinityHeads(32, AffinityHeadConfig(hidden_dim=16, dropout=0.0))
        prediction = heads(torch.randn(4, 32))
        assert prediction.values.shape == (4, 3)

    def test_the_endpoints_have_independent_parameters_by_default(self):
        heads = AffinityHeads(16, AffinityHeadConfig(hidden_dim=8, share_trunk=False))
        pointers = [
            {p.data_ptr() for p in trunk.parameters()} for trunk in heads.trunk
        ]
        assert not (pointers[0] & pointers[1])
        assert not (pointers[1] & pointers[2])

    def test_a_shared_trunk_is_available_but_not_the_default(self):
        assert AffinityHeadConfig().share_trunk is False
        heads = AffinityHeads(16, AffinityHeadConfig(hidden_dim=8, share_trunk=True))
        assert heads(torch.randn(2, 16)).values.shape == (2, 3)

    def test_the_output_is_unbounded(self):
        """The corpus spans roughly pX 2 to 12; squashing would clip the extremes.

        Potency and selectivity claims live at the extremes, so a sigmoid would
        make them unreachable by construction.
        """
        heads = AffinityHeads(4, AffinityHeadConfig(hidden_dim=4, dropout=0.0))
        with torch.no_grad():
            for output in heads.outputs:
                output.bias.fill_(50.0)
        assert heads(torch.zeros(1, 4)).values.min() > 20.0

    def test_observed_selects_one_value_per_row(self):
        prediction = _prediction([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert prediction.observed(torch.tensor([2, 0])).tolist() == [3.0, 4.0]

    def test_for_endpoint_returns_a_whole_column_for_prospective_use(self):
        prediction = _prediction([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert prediction.for_endpoint(Endpoint.KI).tolist() == [2.0, 5.0]
        assert prediction.for_endpoint("IC50").tolist() == [3.0, 6.0]

    def test_an_out_of_range_endpoint_index_is_refused(self):
        with pytest.raises(AffinityHeadError, match="out of range"):
            _prediction([[1.0, 2.0, 3.0]]).observed(torch.tensor([7]))

    def test_a_float_endpoint_index_is_refused(self):
        with pytest.raises(AffinityHeadError, match="integer tensor"):
            _prediction([[1.0, 2.0, 3.0]]).observed(torch.tensor([0.0]))

    def test_a_wrong_input_width_is_refused(self):
        heads = AffinityHeads(32, AffinityHeadConfig(hidden_dim=8))
        with pytest.raises(AffinityHeadError, match="expected width 32"):
            heads(torch.randn(2, 16))


class TestCensoringViolation:
    def test_a_prediction_inside_an_upper_bound_costs_nothing(self):
        """'At most 5.0' is satisfied by 3.0 just as well as by 5.0."""
        violation = censoring_violation(
            torch.tensor([3.0, 5.0, 4.999]),
            torch.full((3,), NAN),
            torch.full((3,), 5.0),
        )
        assert violation.tolist() == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)

    def test_a_prediction_above_an_upper_bound_is_penalised_by_the_excess(self):
        violation = censoring_violation(
            torch.tensor([7.5]), torch.tensor([NAN]), torch.tensor([5.0])
        )
        assert violation.item() == pytest.approx(2.5)

    def test_a_prediction_inside_a_lower_bound_costs_nothing(self):
        violation = censoring_violation(
            torch.tensor([9.0, 10.0]), torch.full((2,), 9.0), torch.full((2,), NAN)
        )
        assert violation.tolist() == pytest.approx([0.0, 0.0], abs=1e-6)

    def test_a_prediction_below_a_lower_bound_is_penalised_by_the_shortfall(self):
        violation = censoring_violation(
            torch.tensor([6.0]), torch.tensor([9.0]), torch.tensor([NAN])
        )
        assert violation.item() == pytest.approx(3.0)

    def test_rows_with_no_bounds_contribute_nothing(self):
        violation = censoring_violation(
            torch.tensor([3.0, 99.0]), torch.full((2,), NAN), torch.full((2,), NAN)
        )
        assert violation.abs().sum().item() == 0.0

    def test_the_penalty_is_one_sided_not_symmetric(self):
        """A squared error would punish 3.0 and 7.0 equally against 'at most 5'."""
        low = censoring_violation(
            torch.tensor([3.0]), torch.tensor([NAN]), torch.tensor([5.0])
        )
        high = censoring_violation(
            torch.tensor([7.0]), torch.tensor([NAN]), torch.tensor([5.0])
        )
        assert low.item() == 0.0
        assert high.item() > 0.0


class TestLoss:
    def test_a_perfect_exact_prediction_gives_zero_loss(self):
        targets = _targets([{"endpoint": "Ki", "p_value": 8.0}])
        out = affinity_loss(_prediction([[0.0, 8.0, 0.0]]), targets)
        assert out.total.item() == pytest.approx(0.0, abs=1e-6)
        assert out.exact_rmse.item() == pytest.approx(0.0, abs=1e-6)
        assert out.n_exact == 1

    def test_a_prediction_inside_the_bound_costs_nothing(self):
        """Not treating a bound as exact, stated as a loss value.

        A model predicting pIC50 3.0 for a compound measured as 'IC50 greater than
        50 micromolar' (pIC50 at most 4.3) is correct, and must pay nothing.
        """
        targets = _targets(
            [{"endpoint": "IC50", "p_value": 4.3, "is_exact": False, "upper_bound": 4.3}]
        )
        out = affinity_loss(_prediction([[0.0, 0.0, 3.0]]), targets)
        assert out.total.item() == pytest.approx(0.0, abs=1e-6)
        assert out.n_censored == 1
        assert out.bound_violation_rate.item() == 0.0

    def test_the_same_row_scored_as_exact_would_have_cost_something(self):
        """Contrast with the test above: this is the error being avoided."""
        as_exact = _targets([{"endpoint": "IC50", "p_value": 4.3}])
        out = affinity_loss(_prediction([[0.0, 0.0, 3.0]]), as_exact)
        assert out.total.item() > 0.5

    def test_a_bound_violation_is_penalised_and_counted(self):
        targets = _targets(
            [{"endpoint": "IC50", "p_value": 4.3, "is_exact": False, "upper_bound": 4.3}]
        )
        out = affinity_loss(_prediction([[0.0, 0.0, 9.0]]), targets)
        assert out.total.item() > 0.0
        assert out.bound_violation_rate.item() == 1.0
        assert out.mean_bound_violation.item() == pytest.approx(4.7, abs=1e-5)

    def test_exact_error_and_bound_violations_are_reported_separately(self):
        """The specification requires these not be folded into one number."""
        targets = _targets(
            [
                {"endpoint": "Ki", "p_value": 8.0},
                {"endpoint": "IC50", "p_value": 5.0, "is_exact": False, "upper_bound": 5.0},
            ]
        )
        out = affinity_loss(_prediction([[0.0, 7.0, 0.0], [0.0, 0.0, 9.0]]), targets)
        assert out.n_exact == 1 and out.n_censored == 1
        assert out.exact_rmse.item() == pytest.approx(1.0, abs=1e-5)
        assert out.bound_violation_rate.item() == 1.0
        # Distinct fields, not a sum.
        assert out.exact_loss.item() != out.censored_loss.item()

    def test_an_unmeasured_endpoint_never_contributes(self):
        """The specification's no-fabricated-zero requirement, on gradients.

        Only the Ki column was measured. Moving the pKd and pIC50 predictions must
        leave the loss untouched, and no gradient may reach them.
        """
        targets = _targets([{"endpoint": "Ki", "p_value": 8.0}])
        base = affinity_loss(_prediction([[0.0, 7.0, 0.0]]), targets).total
        moved = affinity_loss(_prediction([[500.0, 7.0, -500.0]]), targets).total
        assert base.item() == pytest.approx(moved.item())

        values = torch.tensor([[0.0, 7.0, 0.0]], requires_grad=True)
        affinity_loss(AffinityPrediction(values=values), targets).total.backward()
        assert values.grad[0, 0].item() == 0.0  # pKd
        assert values.grad[0, 2].item() == 0.0  # pIC50
        assert values.grad[0, 1].item() != 0.0  # pKi, the measured one

    def test_the_huber_delta_is_the_measured_noise_floor_per_endpoint(self):
        """A residual inside the replicate noise is treated as noise.

        pKd's sigma_rep is 0.945 and pKi's is 0.641, so the same residual must cost
        less on pKd than on pKi -- fitting tighter than the noise is not progress.
        """
        error_model = ErrorModel()
        assert error_model.sigma(Endpoint.KD) > error_model.sigma(Endpoint.KI)
        residual = 0.6  # inside pKd's noise floor, outside pKi's quadratic region edge
        kd = affinity_loss(
            _prediction([[8.0 + residual, 0.0, 0.0]]),
            _targets([{"endpoint": "Kd", "p_value": 8.0}]),
        )
        ki = affinity_loss(
            _prediction([[0.0, 8.0 + residual, 0.0]]),
            _targets([{"endpoint": "Ki", "p_value": 8.0}]),
        )
        assert kd.exact_loss.item() < ki.exact_loss.item()

    def test_endpoint_sigma_follows_the_declared_column_order(self):
        sigma = endpoint_sigma(ErrorModel())
        assert sigma.tolist() == pytest.approx([0.945, 0.641, 0.671])

    def test_the_noise_floor_table_is_published_alongside_rmse(self):
        assert noise_floor_table() == pytest.approx(
            {"Kd": 0.945, "Ki": 0.641, "IC50": 0.671}
        )

    def test_each_term_averages_over_its_own_rows_not_the_batch(self):
        """Otherwise the effective learning rate depends on batch composition."""
        one_exact = _targets([{"endpoint": "Ki", "p_value": 8.0}])
        padded = _targets(
            [
                {"endpoint": "Ki", "p_value": 8.0},
                {"endpoint": "IC50", "p_value": 5.0, "is_exact": False, "upper_bound": 5.0},
            ]
        )
        a = affinity_loss(_prediction([[0.0, 7.0, 0.0]]), one_exact)
        # The added row is censored and satisfied, so it contributes zero.
        b = affinity_loss(_prediction([[0.0, 7.0, 0.0], [0.0, 0.0, 3.0]]), padded)
        assert a.exact_loss.item() == pytest.approx(b.exact_loss.item())

    def test_an_all_censored_batch_does_not_produce_nan(self):
        targets = _targets(
            [
                {"endpoint": "IC50", "p_value": 5.0, "is_exact": False, "upper_bound": 5.0},
                {"endpoint": "Kd", "p_value": 10.0, "is_exact": False, "lower_bound": 10.0},
            ]
        )
        out = affinity_loss(_prediction([[0.0, 0.0, 4.0], [11.0, 0.0, 0.0]]), targets)
        assert torch.isfinite(out.total)
        assert out.n_exact == 0
        assert out.exact_rmse.item() == 0.0

    def test_an_all_exact_batch_does_not_produce_nan(self):
        targets = _targets([{"endpoint": "Ki", "p_value": 8.0}])
        out = affinity_loss(_prediction([[0.0, 8.5, 0.0]]), targets)
        assert torch.isfinite(out.total)
        assert out.n_censored == 0
        assert out.bound_violation_rate.item() == 0.0


class TestTierB:
    def test_tier_b_cannot_reach_the_primary_term(self):
        """The curation decision was 'auxiliary objectives only'.

        Structural: there is no weighting that moves these rows into the primary
        loss, only relabelling them upstream.
        """
        targets = _targets(
            [{"endpoint": "IC50", "p_value": 7.0, "is_primary": False}]
        )
        out = affinity_loss(_prediction([[0.0, 0.0, 4.0]]), targets)
        assert out.n_exact == 0
        assert out.n_auxiliary == 1
        assert out.exact_loss.item() == 0.0
        assert out.auxiliary_loss.item() > 0.0

    def test_auxiliary_rows_still_carry_gradient(self):
        """They are weaker evidence, not no evidence."""
        values = torch.tensor([[0.0, 0.0, 4.0]], requires_grad=True)
        targets = _targets([{"endpoint": "IC50", "p_value": 7.0, "is_primary": False}])
        affinity_loss(AffinityPrediction(values=values), targets).total.backward()
        assert values.grad[0, 2].abs().item() > 0.0

    def test_the_auxiliary_weight_is_below_one_by_default(self):
        assert 0.0 < AffinityLossConfig().auxiliary_weight < 1.0

    def test_a_zero_auxiliary_weight_removes_them_from_the_total_but_not_the_report(self):
        targets = _targets([{"endpoint": "IC50", "p_value": 7.0, "is_primary": False}])
        out = affinity_loss(
            _prediction([[0.0, 0.0, 4.0]]),
            targets,
            config=AffinityLossConfig(auxiliary_weight=0.0),
        )
        assert out.total.item() == pytest.approx(0.0)
        assert out.auxiliary_loss.item() > 0.0  # still measured and reported
        assert out.n_auxiliary == 1


class TestTargetValidation:
    def test_a_censored_row_with_no_bound_is_rejected(self):
        """It states nothing; training on it invents information."""
        with pytest.raises(AffinityLossError, match="neither a lower nor an upper"):
            _targets([{"endpoint": "Ki", "p_value": 8.0, "is_exact": False}])

    def test_a_censored_row_with_both_bounds_is_rejected(self):
        with pytest.raises(AffinityLossError, match="both bounds"):
            _targets(
                [
                    {
                        "endpoint": "Ki",
                        "p_value": 8.0,
                        "is_exact": False,
                        "lower_bound": 7.0,
                        "upper_bound": 9.0,
                    }
                ]
            )

    def test_a_bound_in_the_wrong_field_is_rejected(self):
        """Censoring reverses under the negative log.

        An exact row carrying a bound means the two were conflated somewhere
        upstream, which would train the model toward the opposite of the
        measurement with a perfectly normal-looking loss curve.
        """
        with pytest.raises(AffinityLossError, match="exact row carries a censoring bound"):
            _targets([{"endpoint": "Ki", "p_value": 8.0, "upper_bound": 8.0}])

    def test_a_shape_mismatch_is_rejected(self):
        with pytest.raises(AffinityLossError, match="p_value must be"):
            AffinityTargets(
                endpoint_index=torch.tensor([0, 1]),
                p_value=torch.tensor([8.0]),
                is_exact=torch.tensor([True, True]),
                lower_bound=torch.tensor([NAN, NAN]),
                upper_bound=torch.tensor([NAN, NAN]),
                is_primary=torch.tensor([True, True]),
            )

    def test_a_non_boolean_flag_is_rejected(self):
        with pytest.raises(AffinityLossError, match="is_exact must be bool"):
            AffinityTargets(
                endpoint_index=torch.tensor([0]),
                p_value=torch.tensor([8.0]),
                is_exact=torch.tensor([1]),
                lower_bound=torch.tensor([NAN]),
                upper_bound=torch.tensor([NAN]),
                is_primary=torch.tensor([True]),
            )


class TestTobit:
    def test_tobit_is_not_the_default(self):
        """sigma_rep is a lower bound on error, so Tobit on it is over-confident."""
        assert AffinityLossConfig().censored_mode == "hinge"

    def test_tobit_still_prefers_predictions_inside_the_bound(self):
        targets = _targets(
            [{"endpoint": "IC50", "p_value": 5.0, "is_exact": False, "upper_bound": 5.0}]
        )
        config = AffinityLossConfig(censored_mode="tobit")
        inside = affinity_loss(_prediction([[0.0, 0.0, 3.0]]), targets, config=config)
        outside = affinity_loss(_prediction([[0.0, 0.0, 8.0]]), targets, config=config)
        assert inside.censored_loss.item() < outside.censored_loss.item()

    def test_tobit_unlike_the_hinge_is_not_exactly_zero_inside(self):
        """A likelihood is a different claim from a constraint; it is never flat."""
        targets = _targets(
            [{"endpoint": "IC50", "p_value": 5.0, "is_exact": False, "upper_bound": 5.0}]
        )
        config = AffinityLossConfig(censored_mode="tobit")
        near = affinity_loss(_prediction([[0.0, 0.0, 4.9]]), targets, config=config)
        far = affinity_loss(_prediction([[0.0, 0.0, 1.0]]), targets, config=config)
        assert far.censored_loss.item() < near.censored_loss.item()
        assert near.censored_loss.item() > 0.0

    def test_tobit_does_not_overflow_far_outside_the_bound(self):
        """-log Phi of a large negative argument is -log 0 = inf without clamping."""
        targets = _targets(
            [{"endpoint": "IC50", "p_value": 5.0, "is_exact": False, "upper_bound": 5.0}]
        )
        out = affinity_loss(
            _prediction([[0.0, 0.0, 500.0]]),
            targets,
            config=AffinityLossConfig(censored_mode="tobit"),
        )
        assert torch.isfinite(out.total)

    def test_an_unknown_censored_mode_is_refused(self):
        with pytest.raises(AffinityLossError, match="censored_mode"):
            AffinityLossConfig(censored_mode="whatever")


class TestPerEndpointReport:
    def test_metrics_are_reported_per_endpoint_because_noise_floors_differ(self):
        """A pooled RMSE cannot be compared to any single noise floor."""
        targets = _targets(
            [
                {"endpoint": "Kd", "p_value": 8.0},
                {"endpoint": "Ki", "p_value": 8.0},
                {"endpoint": "IC50", "p_value": 5.0, "is_exact": False, "upper_bound": 5.0},
            ]
        )
        out = affinity_loss(
            _prediction([[9.0, 0.0, 0.0], [0.0, 8.5, 0.0], [0.0, 0.0, 6.0]]), targets
        )
        assert out.per_endpoint["Kd"]["rmse"] == pytest.approx(1.0, abs=1e-5)
        assert out.per_endpoint["Ki"]["rmse"] == pytest.approx(0.5, abs=1e-5)
        assert out.per_endpoint["IC50"]["bound_violation_rate"] == 1.0
        assert out.per_endpoint["IC50"]["n_exact"] == 0

    def test_every_endpoint_appears_even_when_absent_from_the_batch(self):
        out = affinity_loss(
            _prediction([[0.0, 8.0, 0.0]]), _targets([{"endpoint": "Ki", "p_value": 8.0}])
        )
        assert set(out.per_endpoint) == {"Kd", "Ki", "IC50"}
        assert out.per_endpoint["Kd"]["n_exact"] == 0

    def test_the_output_serialises_for_a_run_log(self):
        out = affinity_loss(
            _prediction([[0.0, 8.0, 0.0]]), _targets([{"endpoint": "Ki", "p_value": 8.0}])
        )
        payload = out.to_dict()
        assert payload["n_exact"] == 1
        assert isinstance(payload["exact_rmse"], float)
        assert set(payload["per_endpoint"]) == {"Kd", "Ki", "IC50"}


@pytest.mark.snapshot
class TestAgainstTheCuratedCorpus:
    """The loss must accept the real table, with its real column names."""

    def test_targets_build_from_a_curated_frame(self):
        import json
        from pathlib import Path

        from bindra.data.tables import ShardedTableRef

        report = Path("data/processed/202610/qc_report.json")
        if not report.exists():
            pytest.skip("curated corpus not present")
        payload = json.loads(report.read_text(encoding="utf-8"))
        reference = ShardedTableRef.from_dict(payload["artifacts"]["observations"])
        frame = next(
            iter(
                reference.iter_frames(
                    columns=[
                        "endpoint", "p_value", "is_exact", "lower_bound", "upper_bound",
                        "ic50_tier", "primary_regression_eligible",
                    ]
                )
            )
        ).head(5000)

        targets = AffinityTargets.from_frame(frame)
        assert targets.batch_size == len(frame)
        # Tier B is excluded from primary by construction, not by the caller.
        import numpy as np

        tier_b = np.array((frame["ic50_tier"].astype(str) == "B").to_numpy(), copy=True)
        assert not bool(targets.is_primary[torch.from_numpy(tier_b)].any())

        torch.manual_seed(0)
        prediction = AffinityPrediction(values=torch.randn(len(frame), 3) + 7.0)
        out = affinity_loss(prediction, targets)
        assert torch.isfinite(out.total)
        assert out.n_exact + out.n_censored + out.n_auxiliary == len(frame)
        print(
            f"\nreal batch: {out.n_exact:,} exact, {out.n_censored:,} censored, "
            f"{out.n_auxiliary:,} auxiliary; "
            f"bound violation rate {out.bound_violation_rate.item():.3f}; "
            f"per-endpoint {out.per_endpoint}"
        )

    def test_a_random_predictor_sits_well_above_the_noise_floor(self):
        """A sanity check with teeth: an untrained model must not look skilful.

        If an untrained predictor's RMSE were near sigma_rep, the metric would be
        measuring the label distribution rather than the model.
        """
        import json
        from pathlib import Path

        from bindra.data.tables import ShardedTableRef

        report = Path("data/processed/202610/qc_report.json")
        if not report.exists():
            pytest.skip("curated corpus not present")
        payload = json.loads(report.read_text(encoding="utf-8"))
        reference = ShardedTableRef.from_dict(payload["artifacts"]["observations"])
        frame = next(
            iter(
                reference.iter_frames(
                    columns=[
                        "endpoint", "p_value", "is_exact", "lower_bound", "upper_bound",
                        "ic50_tier", "primary_regression_eligible",
                    ]
                )
            )
        ).head(5000)
        targets = AffinityTargets.from_frame(frame)
        torch.manual_seed(0)
        out = affinity_loss(
            AffinityPrediction(values=torch.randn(len(frame), 3) * 3.0 + 7.0), targets
        )
        floors = noise_floor_table()
        for endpoint, entry in out.per_endpoint.items():
            if entry.get("n_exact", 0) > 100:
                assert entry["rmse"] > floors[endpoint], (
                    f"an untrained predictor reached {entry['rmse']:.3f} on {endpoint}, "
                    f"below the measured noise floor {floors[endpoint]}"
                )
