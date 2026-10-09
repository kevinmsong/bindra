"""Ensemble, conformal calibration and applicability domain.

Each module names one specification clause, and those clauses are the point of
this file. Three tests carry the weight:

``test_repeated_passes_over_one_model_are_refused``
    "dropout passes are not an ensemble" is the cheapest thing to get wrong,
    because doing it wrong still produces a spread that plots convincingly.
``test_an_unsupported_order_statistic_returns_a_status_not_an_interval``
    The fallback that would make a fabricated interval is one line away, and
    the interval it returns is indistinguishable from a valid one.
``test_the_domain_flag_cannot_see_the_ensemble``
    Enforced by the signature, so it is checked as an interface property
    rather than by reading the implementation.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from bindra.uncertainty import conformal, domain, ensemble


def member(name: str, seed: int, digest: str) -> ensemble.EnsembleMember:
    from pathlib import Path

    return ensemble.EnsembleMember(path=Path(name), seed=seed, weight_digest=digest)


class TestEnsemble:
    def test_repeated_passes_over_one_model_are_refused(self) -> None:
        """Identical weights are not an ensemble however they are averaged."""
        members = [member("a.pt", 1, "same"), member("b.pt", 2, "same")]
        with pytest.raises(ensemble.EnsembleError, match="identical weights"):
            ensemble.assert_independent(members)

    def test_members_sharing_a_seed_are_refused(self) -> None:
        """Same seed means same init and same data order; not independent."""
        members = [member("a.pt", 7, "x"), member("b.pt", 7, "y")]
        with pytest.raises(ensemble.EnsembleError, match="share seed"):
            ensemble.assert_independent(members)

    def test_a_single_member_is_refused(self) -> None:
        with pytest.raises(ensemble.EnsembleError, match="at least 2"):
            ensemble.assert_independent([member("a.pt", 1, "x")])

    def test_three_distinct_runs_are_accepted_and_combined(self) -> None:
        members = [member("a.pt", 1, "x"), member("b.pt", 2, "y"), member("c.pt", 3, "z")]
        per_member = np.array([[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]])
        result = ensemble.combine(per_member, members)
        assert result.n_members == 3
        assert np.allclose(result.mean, [2.0, 2.0])
        # ddof=1: a sample of training runs, not the population of them.
        assert np.allclose(result.std, [1.0, 0.0])
        assert "heuristic" in result.to_dict()["note"]

    def test_the_spread_is_not_claimed_to_be_an_interval(self) -> None:
        """The report must not let a reader mistake spread for coverage."""
        members = [member("a.pt", 1, "x"), member("b.pt", 2, "y")]
        payload = ensemble.combine(np.array([[1.0], [2.0]]), members).to_dict()
        assert "not a calibrated interval" in payload["note"]


class TestConformal:
    def test_the_order_statistic_follows_the_specification(self) -> None:
        # ceil((n + 1)(1 - alpha))
        assert conformal.required_k(99, 0.10) == 90
        assert conformal.required_k(19, 0.10) == 18

    def test_an_unsupported_order_statistic_returns_a_status_not_an_interval(self) -> None:
        """Few calibration points must not yield a confident-looking width.

        With n=5 and alpha=0.10 the rule asks for the 6th smallest of 5
        scores. Returning the largest instead would produce a finite interval
        that looks exactly like a valid one and carries none of the guarantee.
        """
        result = conformal.calibrate(
            np.zeros(5), np.zeros(5), endpoint="Kd", alpha=0.10
        )
        assert isinstance(result, conformal.InsufficientCalibration)
        assert result.required == 6 and result.n_calibration == 5
        assert not isinstance(result, conformal.ConformalQuantile)
        assert result.to_dict()["status"] == "insufficient_calibration"

    def test_an_empty_calibration_set_is_also_a_status(self) -> None:
        result = conformal.calibrate(
            np.array([]), np.array([]), endpoint="Ki", alpha=0.10
        )
        assert isinstance(result, conformal.InsufficientCalibration)

    def test_coverage_is_near_nominal_on_exchangeable_data(self) -> None:
        """The guarantee, checked rather than asserted."""
        rng = np.random.default_rng(20261008)
        truth = rng.normal(size=4000)
        predicted = truth + rng.normal(scale=0.5, size=4000)
        quantile = conformal.calibrate(
            truth[:2000], predicted[:2000], endpoint="IC50", alpha=0.10
        )
        assert isinstance(quantile, conformal.ConformalQuantile)
        empirical = conformal.coverage(truth[2000:], predicted[2000:], quantile)
        assert 0.86 < empirical < 0.94, empirical

    def test_nanomolar_conversion_reverses_the_endpoints(self) -> None:
        """The inverse logarithm is order-reversing; the specification says so.

        The upper p-value is the LOWER concentration. Converting each end and
        keeping the order would return an interval the wrong way round.
        """
        quantile = conformal.ConformalQuantile(
            endpoint="Kd", q=1.0, alpha=0.1, n_calibration=100, k=91
        )
        low_nm, high_nm = quantile.to_nanomolar(8.0)  # 10 nM centre
        assert low_nm < high_nm
        assert np.isclose(low_nm, 1.0) and np.isclose(high_nm, 100.0)

    def test_endpoints_are_calibrated_separately(self) -> None:
        """Noise floors differ 1.5x across endpoints; one quantile cannot fit."""
        rng = np.random.default_rng(0)
        tight = rng.normal(scale=0.2, size=500)
        loose = rng.normal(scale=1.0, size=500)
        q_tight = conformal.calibrate(tight, np.zeros(500), endpoint="IC50")
        q_loose = conformal.calibrate(loose, np.zeros(500), endpoint="Kd")
        assert q_tight.q < q_loose.q


class TestApplicabilityDomain:
    def test_the_domain_flag_cannot_see_the_ensemble(self) -> None:
        """Independence enforced by the signature, not promised in prose.

        An ensemble can agree confidently on a molecule unlike anything it has
        seen. Deriving the flag from the spread would hide exactly that case.
        """
        signature = inspect.signature(domain.domain_status)
        taken = set(signature.parameters)
        assert not {"std", "spread", "ensemble", "uncertainty"} & taken, taken

    def test_a_training_molecule_is_in_domain(self) -> None:
        train = ["CC(=O)Oc1ccccc1C(=O)O", "CCO", "c1ccccc1"]
        statuses = domain.domain_status(["CC(=O)Oc1ccccc1C(=O)O"], train)
        assert statuses[0].nearest_similarity == pytest.approx(1.0)
        assert not statuses[0].out_of_domain

    def test_something_unlike_the_training_set_is_flagged(self) -> None:
        statuses = domain.domain_status(
            ["CC(=O)Oc1ccccc1C(=O)O"], ["CCO", "CC", "CCC"]
        )
        assert statuses[0].out_of_domain
        assert statuses[0].nearest_similarity < domain.DEFAULT_LIGAND_THRESHOLD

    def test_an_empty_training_set_is_refused(self) -> None:
        """Distance against nothing is not a domain."""
        with pytest.raises(domain.DomainError, match="no training molecules"):
            domain.domain_status(["CCO"], [])

    def test_an_unparseable_query_is_out_of_domain_not_similarity_zero(self) -> None:
        statuses = domain.domain_status(["not a molecule"], ["CCO", "CC"])
        assert statuses[0].out_of_domain
        assert statuses[0].method == "unparseable-query"
        assert np.isnan(statuses[0].nearest_similarity)

    def test_the_summary_carries_the_rule_that_produced_it(self) -> None:
        statuses = domain.domain_status(["CCO", "CC(=O)Oc1ccccc1C(=O)O"], ["CCO"])
        payload = domain.summarise(statuses)
        assert payload["n"] == 2
        assert payload["threshold"] == domain.DEFAULT_LIGAND_THRESHOLD
