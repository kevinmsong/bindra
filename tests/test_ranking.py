"""Within-assay pairwise ranking: the grouping, the margin, and censored bounds.

Four tests carry the weight of this file:

``test_a_batch_without_group_ids_is_refused``
    BindingDB has no assay identifier, so the group is a curated proxy. Falling
    back to "the whole batch is one group" when it is missing would rank across
    experiments, which is exactly what the specification's clause forbids, and
    the loss would fall just the same.
``test_pairs_from_different_groups_are_never_compared``
    The clause itself, checked on behaviour rather than promised in prose.
``test_the_default_margin_is_the_measured_noise_floor``
    The specification starts the margin at 0.5, which is below the replicate
    noise floor of every endpoint. At 0.5 the term trains on pairs the
    laboratory could not reproduce.
``test_a_censored_row_is_ranked_only_where_its_bound_allows``
    16.6% of the corpus is censored. "Weaker than 10 uM" orders against a 10 nM
    measurement and says nothing against another weak one; treating the bound
    as a value would invent the difference.
"""

from __future__ import annotations

import pytest
import torch

from bindra.training.losses.affinity import AffinityTargets, noise_floor_table
from bindra.training.losses.ranking import (
    RankingLossConfig,
    RankingLossError,
    certain_bounds,
    margin_per_endpoint,
    ranking_loss,
)

KD, KI, IC50 = 0, 1, 2


def targets(
    p_values,
    *,
    groups,
    endpoints=None,
    exact=None,
    lower=None,
    upper=None,
    eligible=None,
) -> AffinityTargets:
    n = len(p_values)
    exact = [True] * n if exact is None else exact
    return AffinityTargets(
        endpoint_index=torch.tensor(
            [KD] * n if endpoints is None else endpoints, dtype=torch.long
        ),
        p_value=torch.tensor(p_values, dtype=torch.float32),
        is_exact=torch.tensor(exact, dtype=torch.bool),
        lower_bound=torch.tensor(
            [float("nan")] * n if lower is None else lower, dtype=torch.float32
        ),
        upper_bound=torch.tensor(
            [float("nan")] * n if upper is None else upper, dtype=torch.float32
        ),
        is_primary=torch.ones(n, dtype=torch.bool),
        comparability_group=torch.tensor(groups, dtype=torch.long),
        ranking_eligible=torch.ones(n, dtype=torch.bool)
        if eligible is None
        else torch.tensor(eligible, dtype=torch.bool),
    )


class TestTheGrouping:
    def test_a_batch_without_group_ids_is_refused(self):
        """No silent fallback to "the batch is one group".

        BindingDB contains no assay identifier anywhere in its 640 columns, so
        the group is the curation layer's proxy. Ranking without it compares
        across experiments and the loss falls exactly the same way, which is
        what makes the silent fallback dangerous rather than merely wrong.
        """
        plain = AffinityTargets(
            endpoint_index=torch.zeros(3, dtype=torch.long),
            p_value=torch.tensor([9.0, 7.0, 8.0]),
            is_exact=torch.ones(3, dtype=torch.bool),
            lower_bound=torch.full((3,), float("nan")),
            upper_bound=torch.full((3,), float("nan")),
            is_primary=torch.ones(3, dtype=torch.bool),
        )
        with pytest.raises(RankingLossError, match="no comparability_group"):
            ranking_loss(torch.zeros(3), plain)

    def test_pairs_from_different_groups_are_never_compared(self):
        """Two rows, two groups, a huge potency gap: still no pair."""
        batch = targets([10.0, 5.0], groups=[0, 1])
        result = ranking_loss(torch.tensor([0.0, 0.0]), batch, RankingLossConfig())
        assert result.n_pairs_available == 0
        assert result.n_pairs == 0
        assert float(result.loss.detach()) == 0.0

    def test_pairs_inside_one_group_are_compared(self):
        batch = targets([10.0, 5.0], groups=[3, 3])
        result = ranking_loss(torch.tensor([0.0, 0.0]), batch, RankingLossConfig())
        assert result.n_pairs == 1
        assert result.n_groups == 1

    def test_endpoints_are_never_mixed_inside_a_group(self):
        """Kd and IC50 are different experiments even under one group id."""
        batch = targets([10.0, 5.0], groups=[1, 1], endpoints=[KD, IC50])
        assert ranking_loss(torch.zeros(2), batch).n_pairs == 0

    def test_rows_the_curation_layer_excluded_are_dropped(self):
        batch = targets([10.0, 5.0], groups=[1, 1], eligible=[True, False])
        assert ranking_loss(torch.zeros(2), batch).n_pairs == 0

    def test_the_eligibility_filter_can_be_turned_off(self):
        batch = targets([10.0, 5.0], groups=[1, 1], eligible=[True, False])
        config = RankingLossConfig(ranking_eligible_only=False)
        assert ranking_loss(torch.zeros(2), batch, config).n_pairs == 1

    def test_requiring_eligibility_without_the_flag_is_refused(self):
        batch = targets([10.0, 5.0], groups=[1, 1])
        batch = AffinityTargets(
            endpoint_index=batch.endpoint_index,
            p_value=batch.p_value,
            is_exact=batch.is_exact,
            lower_bound=batch.lower_bound,
            upper_bound=batch.upper_bound,
            is_primary=batch.is_primary,
            comparability_group=batch.comparability_group,
        )
        with pytest.raises(RankingLossError, match="no\\s+ranking_eligible"):
            ranking_loss(torch.zeros(2), batch)


class TestTheMargin:
    def test_the_default_margin_is_the_measured_noise_floor(self):
        """Not the specification's 0.5, which is below every endpoint's floor.

        Measured over the corpus: at 0.5 about 59.5% of pKd pairs qualify
        against 38.6% at the floor, and the extra 21 points are pairs whose
        difference the laboratory's own repeats could not resolve.
        """
        assert margin_per_endpoint(RankingLossConfig()) == noise_floor_table()
        assert noise_floor_table()["Kd"] == 0.945

    def test_the_specification_value_is_still_reachable(self):
        margins = margin_per_endpoint(RankingLossConfig(margin_override=0.5))
        assert set(margins.values()) == {0.5}

    def test_a_pair_inside_the_noise_floor_is_not_ranked(self):
        """0.8 p-units apart is a real-looking gap and below the pKd floor."""
        batch = targets([9.0, 8.2], groups=[1, 1])
        assert ranking_loss(torch.zeros(2), batch).n_pairs == 0

    def test_the_same_pair_is_ranked_at_the_specification_margin(self):
        """The two settings genuinely differ, so the choice is a real one."""
        batch = targets([9.0, 8.2], groups=[1, 1])
        config = RankingLossConfig(margin_override=0.5)
        assert ranking_loss(torch.zeros(2), batch, config).n_pairs == 1

    def test_the_margin_is_per_endpoint(self):
        """pKi's floor is 0.641; the same gap qualifies there and not for pKd."""
        gap = [9.0, 8.3]  # 0.7 apart: over Ki's floor, under Kd's
        assert ranking_loss(torch.zeros(2), targets(gap, groups=[1, 1])).n_pairs == 0
        ki = targets(gap, groups=[1, 1], endpoints=[KI, KI])
        assert ranking_loss(torch.zeros(2), ki).n_pairs == 1

    def test_scaling_the_margin_moves_the_threshold(self):
        batch = targets([9.0, 8.2], groups=[1, 1])
        assert ranking_loss(
            torch.zeros(2), batch, RankingLossConfig(margin_scale=0.5)
        ).n_pairs == 1

    def test_a_zero_margin_is_refused(self):
        with pytest.raises(RankingLossError, match="margin_scale"):
            RankingLossConfig(margin_scale=0.0)
        with pytest.raises(RankingLossError, match="margin_override"):
            RankingLossConfig(margin_override=0.0)


class TestTheLoss:
    def test_a_correct_ordering_past_the_margin_costs_nothing(self):
        batch = targets([10.0, 5.0], groups=[1, 1])
        # Predicted 2.0 apart, margin 0.945: the hinge is satisfied.
        result = ranking_loss(torch.tensor([3.0, 1.0]), batch)
        assert float(result.loss.detach()) == 0.0
        assert result.concordance == 1.0

    def test_a_reversed_ordering_is_penalised(self):
        batch = targets([10.0, 5.0], groups=[1, 1])
        result = ranking_loss(torch.tensor([1.0, 3.0]), batch)
        # hinge = margin - (1 - 3) = 0.945 + 2
        assert float(result.loss) == pytest.approx(2.945, abs=1e-5)
        assert result.concordance == 0.0

    def test_a_correct_but_too_close_ordering_is_still_penalised(self):
        """The hinge asks for the same gap that qualified the pair."""
        batch = targets([10.0, 5.0], groups=[1, 1])
        result = ranking_loss(torch.tensor([1.1, 1.0]), batch)
        assert float(result.loss) == pytest.approx(0.945 - 0.1, abs=1e-5)
        assert result.concordance == 1.0

    def test_the_gradient_pushes_the_pair_apart(self):
        batch = targets([10.0, 5.0], groups=[1, 1])
        prediction = torch.tensor([1.0, 3.0], requires_grad=True)
        ranking_loss(prediction, batch).loss.backward()
        # More potent row gets a negative gradient (raise it), weaker positive.
        assert prediction.grad[0] < 0 < prediction.grad[1]

    def test_an_empty_term_is_differentiable_and_zero(self):
        """A batch with no comparable pair must not break the backward pass."""
        batch = targets([10.0, 5.0], groups=[0, 1])
        prediction = torch.tensor([1.0, 2.0], requires_grad=True)
        result = ranking_loss(prediction, batch)
        result.loss.backward()
        assert float(result.loss.detach()) == 0.0
        assert prediction.grad is not None
        assert torch.equal(prediction.grad, torch.zeros(2))

    def test_concordance_is_reported_beside_the_loss(self):
        """A falling hinge and a correct ordering are not the same thing."""
        batch = targets([10.0, 5.0, 9.0, 4.0], groups=[1, 1, 2, 2])
        result = ranking_loss(torch.tensor([3.0, 1.0, 1.0, 3.0]), batch)
        assert result.n_pairs == 2
        assert result.concordance == 0.5
        assert result.n_groups == 2

    def test_pairs_are_capped_and_the_available_count_still_reported(self):
        """A 2,295-row group would contribute 2.6M pairs and dominate the term."""
        n = 40
        batch = targets(
            [10.0 - 0.0 * i for i in range(n // 2)] + [1.0] * (n // 2),
            groups=[7] * n,
        )
        result = ranking_loss(
            torch.zeros(n), batch, RankingLossConfig(max_pairs=10)
        )
        assert result.n_pairs == 10
        assert result.n_pairs_available == (n // 2) * (n // 2)

    def test_the_subsample_is_reproducible_from_a_generator(self):
        n = 30
        batch = targets([10.0] * (n // 2) + [1.0] * (n // 2), groups=[1] * n)
        config = RankingLossConfig(max_pairs=8)
        prediction = torch.arange(n, dtype=torch.float32)
        first = ranking_loss(
            prediction, batch, config, generator=torch.Generator().manual_seed(5)
        )
        again = ranking_loss(
            prediction, batch, config, generator=torch.Generator().manual_seed(5)
        )
        assert float(first.loss) == float(again.loss)

    def test_a_mismatched_prediction_shape_is_refused(self):
        batch = targets([10.0, 5.0], groups=[1, 1])
        with pytest.raises(RankingLossError, match="against"):
            ranking_loss(torch.zeros(3), batch)


class TestCensoredRows:
    def test_certain_bounds_collapse_an_exact_row_to_a_point(self):
        low, high = certain_bounds(
            torch.tensor([7.0]),
            torch.tensor([True]),
            torch.tensor([float("nan")]),
            torch.tensor([float("nan")]),
        )
        assert float(low[0]) == 7.0 and float(high[0]) == 7.0

    def test_certain_bounds_open_a_censored_row_to_a_half_line(self):
        """"At most 5.0" says nothing about how far below 5.0 the value sits."""
        low, high = certain_bounds(
            torch.tensor([5.0, 5.0]),
            torch.tensor([False, False]),
            torch.tensor([float("nan"), 5.0]),  # second has a LOWER bound
            torch.tensor([5.0, float("nan")]),  # first has an UPPER bound
        )
        assert float(low[0]) == float("-inf") and float(high[0]) == 5.0
        assert float(low[1]) == 5.0 and float(high[1]) == float("inf")

    def test_a_censored_row_is_ranked_only_where_its_bound_allows(self):
        """Weaker-than-10uM orders against 10 nM, and not against another weak one.

        Row 0 is exact at 8.0. Row 1 is "at most 5.0". The gap from 8.0 down to
        5.0 is 3.0, past the pKd floor, so the pair is usable. Row 2 is "at most
        7.5": the guaranteed gap to row 0 is 0.5, inside the floor, so it is not.
        """
        batch = targets(
            [8.0, 5.0, 7.5],
            groups=[1, 1, 1],
            exact=[True, False, False],
            upper=[float("nan"), 5.0, 7.5],
        )
        result = ranking_loss(torch.zeros(3), batch)
        assert result.n_pairs == 1

    def test_two_censored_rows_on_the_same_side_cannot_be_ordered(self):
        """"At most 5" against "at most 9" guarantees nothing about the order."""
        batch = targets(
            [5.0, 9.0],
            groups=[1, 1],
            exact=[False, False],
            upper=[5.0, 9.0],
        )
        assert ranking_loss(torch.zeros(2), batch).n_pairs == 0

    def test_a_lower_bound_above_an_upper_bound_is_orderable(self):
        """"At least 9" is certainly more potent than "at most 7"."""
        batch = targets(
            [9.0, 7.0],
            groups=[1, 1],
            exact=[False, False],
            lower=[9.0, float("nan")],
            upper=[float("nan"), 7.0],
        )
        assert ranking_loss(torch.zeros(2), batch).n_pairs == 1

    def test_censored_rows_can_be_excluded_entirely(self):
        batch = targets(
            [8.0, 5.0],
            groups=[1, 1],
            exact=[True, False],
            upper=[float("nan"), 5.0],
        )
        assert ranking_loss(torch.zeros(2), batch).n_pairs == 1
        config = RankingLossConfig(include_censored=False)
        assert ranking_loss(torch.zeros(2), batch, config).n_pairs == 0

    def test_a_bound_is_never_treated_as_a_value(self):
        """The failure this whole mechanism exists to avoid.

        Row 1 is "at most 5.0". If the bound were read as the value, the gap to
        row 0's 5.4 would be 0.4 -- still inside the floor -- but the gap to
        row 2's 5.5 would read as 0.1 and order them. Using the half-line, row 1
        is unbounded below and can never be shown *more* potent than anything.
        """
        batch = targets(
            [5.4, 5.0, 5.5],
            groups=[1, 1, 1],
            exact=[True, False, True],
            upper=[float("nan"), 5.0, float("nan")],
        )
        result = ranking_loss(torch.zeros(3), batch)
        # Only exact-vs-exact 5.5/5.4 could pair, and 0.1 is inside the floor.
        assert result.n_pairs == 0


class TestTheConfiguration:
    def test_the_term_is_off_by_default(self):
        """Adding a term changes the objective; it turns on deliberately."""
        assert RankingLossConfig().weight == 0.0

    def test_the_engine_default_is_also_off(self):
        from bindra.training.engine import TrainingConfig

        assert TrainingConfig().ranking.weight == 0.0

    def test_the_configuration_reaches_provenance(self):
        """A checkpoint must state the objective that produced it."""
        from bindra.training.engine import TrainingConfig

        payload = TrainingConfig().to_dict()["ranking"]
        assert payload["weight"] == 0.0
        assert payload["margin_source"].startswith("per-endpoint")
        assert "Priority 2" in payload["spec_clause"]

    def test_a_negative_weight_is_refused(self):
        with pytest.raises(RankingLossError, match="weight"):
            RankingLossConfig(weight=-1.0)

    def test_the_cli_exposes_the_flags_and_defaults_them_off(self):
        from bindra.cli.train import build_parser

        args = build_parser().parse_args([])
        assert args.ranking_weight == 0.0
        assert args.ranking_margin is None
        assert args.ranking_margin_scale == 1.0
        assert args.ranking_max_pairs == 4096
