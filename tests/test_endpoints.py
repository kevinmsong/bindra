"""Endpoint conversion tests.

The first four cases are the exact table the specification requires be turned
into unit tests. The rest cover the ways this conversion goes wrong quietly:
inequality direction, strictness, and endpoint identity.
"""

from __future__ import annotations

import math

import pytest

from bindra.data.endpoints import (
    Endpoint,
    PValue,
    Rejected,
    RejectionReason,
    Relation,
    nm_to_p_value,
    p_interval_to_nm,
    p_value_to_nm,
    parse_relation,
)


class TestSpecifiedConversions:
    """The table named in the specification."""

    def test_kd_100nm_becomes_pkd_7(self):
        result = nm_to_p_value(100, Endpoint.KD)
        assert isinstance(result, PValue)
        assert result.value == pytest.approx(7.0)
        assert result.endpoint is Endpoint.KD
        assert result.is_exact
        assert str(result) == "pKd = 7"

    def test_kd_above_10000nm_becomes_pkd_below_5(self):
        result = nm_to_p_value(10_000, Endpoint.KD, ">")
        assert isinstance(result, PValue)
        assert result.value == pytest.approx(5.0)
        # The inequality must reverse: a weak-binding bound in concentration
        # is an upper bound on affinity.
        assert result.relation is Relation.LT
        assert str(result) == "pKd < 5"

    def test_ki_below_10nm_becomes_pki_above_8(self):
        result = nm_to_p_value(10, Endpoint.KI, "<")
        assert isinstance(result, PValue)
        assert result.value == pytest.approx(8.0)
        assert result.relation is Relation.GT
        assert str(result) == "pKi > 8"

    @pytest.mark.parametrize(
        ("value", "reason"),
        [
            (None, RejectionReason.MISSING),
            ("", RejectionReason.MISSING),
            (0, RejectionReason.NON_POSITIVE),
            (-5.0, RejectionReason.NON_POSITIVE),
            (float("nan"), RejectionReason.NOT_A_NUMBER),
            (float("inf"), RejectionReason.NOT_FINITE),
            ("not a number", RejectionReason.NOT_A_NUMBER),
        ],
    )
    def test_invalid_values_are_rejected_with_a_reason(self, value, reason):
        result = nm_to_p_value(value, Endpoint.KD)
        assert isinstance(result, Rejected)
        assert result.reason is reason
        assert result.detail  # never an empty explanation


class TestInequalityDirection:
    @pytest.mark.parametrize(
        ("concentration_relation", "expected_p_relation"),
        [
            (Relation.EQ, Relation.EQ),
            (Relation.LT, Relation.GT),
            (Relation.GT, Relation.LT),
            (Relation.LE, Relation.GE),
            (Relation.GE, Relation.LE),
        ],
    )
    def test_every_relation_reverses(self, concentration_relation, expected_p_relation):
        result = nm_to_p_value(100, Endpoint.KD, concentration_relation)
        assert isinstance(result, PValue)
        assert result.relation is expected_p_relation

    def test_flipping_twice_is_the_identity(self):
        for relation in Relation:
            assert relation.flipped().flipped() is relation

    def test_strictness_is_preserved_through_flipping(self):
        assert Relation.LT.flipped().is_strict
        assert Relation.GT.flipped().is_strict
        assert not Relation.LE.flipped().is_strict
        assert not Relation.GE.flipped().is_strict

    def test_bounds_point_the_right_way(self):
        # Kd > 10 uM  ->  pKd < 5  ->  an UPPER bound on affinity.
        weak = nm_to_p_value(10_000, Endpoint.KD, ">")
        assert isinstance(weak, PValue)
        assert weak.upper_bound == pytest.approx(5.0)
        assert weak.lower_bound is None

        # Ki < 10 nM  ->  pKi > 8  ->  a LOWER bound on affinity.
        tight = nm_to_p_value(10, Endpoint.KI, "<")
        assert isinstance(tight, PValue)
        assert tight.lower_bound == pytest.approx(8.0)
        assert tight.upper_bound is None


class TestEndpointIdentity:
    def test_endpoints_keep_their_own_labels(self):
        assert Endpoint.KD.p_label == "pKd"
        assert Endpoint.KI.p_label == "pKi"
        assert Endpoint.IC50.p_label == "pIC50"

    def test_ic50_is_not_a_binding_constant(self):
        # IC50 is assay-dependent and must never be relabelled as Kd.
        assert not Endpoint.IC50.is_binding_constant
        assert Endpoint.KD.is_binding_constant
        assert Endpoint.KI.is_binding_constant

    def test_endpoint_is_carried_through_conversion(self):
        for endpoint in Endpoint:
            result = nm_to_p_value(1.0, endpoint)
            assert isinstance(result, PValue)
            assert result.endpoint is endpoint


class TestRoundTrip:
    @pytest.mark.parametrize("nanomolar", [0.001, 1.0, 100.0, 10_000.0, 1e6])
    def test_nm_to_p_and_back(self, nanomolar):
        result = nm_to_p_value(nanomolar, Endpoint.KD)
        assert isinstance(result, PValue)
        assert p_value_to_nm(result.value) == pytest.approx(nanomolar, rel=1e-9)

    def test_interval_endpoints_reverse_under_the_inverse_transform(self):
        # A pKd interval [6, 8] is a concentration interval [10 nM, 1000 nM]:
        # the HIGHER p-value maps to the LOWER concentration.
        low_nm, high_nm = p_interval_to_nm(6.0, 8.0)
        assert low_nm == pytest.approx(10.0)
        assert high_nm == pytest.approx(1000.0)
        assert low_nm < high_nm

    def test_reversed_interval_is_an_error_not_a_silent_swap(self):
        with pytest.raises(ValueError):
            p_interval_to_nm(8.0, 6.0)


class TestRelationParsing:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("=", Relation.EQ),
            ("", Relation.EQ),
            (None, Relation.EQ),
            ("<", Relation.LT),
            (">", Relation.GT),
            ("<=", Relation.LE),
            (">=", Relation.GE),
            (" > ", Relation.GT),
            ("≤", Relation.LE),
            ("≥", Relation.GE),
        ],
    )
    def test_known_spellings(self, raw, expected):
        assert parse_relation(raw) is expected

    def test_unknown_relation_is_rejected_not_guessed(self):
        assert parse_relation("??") is None
        result = nm_to_p_value(100, Endpoint.KD, "??")
        assert isinstance(result, Rejected)
        assert result.reason is RejectionReason.UNKNOWN_RELATION


class TestPlausibilityGuard:
    def test_disabled_by_default(self):
        # 1e-12 nM is absurd but passes when no range is requested, because
        # filtering is a curation policy decision, not a conversion concern.
        result = nm_to_p_value(1e-12, Endpoint.KD)
        assert isinstance(result, PValue)
        assert result.value == pytest.approx(21.0)

    def test_catches_unit_errors_when_enabled(self):
        result = nm_to_p_value(1e-12, Endpoint.KD, plausible_range=(-2.0, 20.0))
        assert isinstance(result, Rejected)
        assert result.reason is RejectionReason.IMPLAUSIBLE_MAGNITUDE
