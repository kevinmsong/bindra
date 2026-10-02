"""Affinity endpoints, p-value conversion, and censored measurements.

Three rules drive everything here, all from the specification:

* ``pX = 9 - log10(x)`` for a concentration ``x`` in nM.
* Inequalities **reverse** under the negative logarithm. ``Kd > 10,000 nM``
  becomes ``pKd < 5``. Getting this backwards silently inverts the meaning of
  every censored record in the dataset.
* pKd, pKi and pIC50 stay separate for ever. IC50 is an assay-dependent
  potency endpoint and is never relabelled as Kd.

A measurement that cannot be converted is *rejected with a reason*, never
coerced to a default or dropped silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math

#: pX = NEG_LOG_OFFSET - log10(value in nM). 9 because 1 nM = 1e-9 M.
NEG_LOG_OFFSET = 9.0


class Endpoint(StrEnum):
    """A measured affinity or potency endpoint.

    Deliberately small. Phenotypic EC50 and qualitative labels are excluded
    from the affinity regression entirely rather than represented here.
    """

    KD = "Kd"
    KI = "Ki"
    IC50 = "IC50"

    @property
    def p_label(self) -> str:
        """The log-scale name, e.g. ``pKd``."""
        return f"p{self.value}"

    @property
    def is_binding_constant(self) -> bool:
        """True for direct binding constants; False for IC50.

        IC50 depends on assay conditions (substrate concentration, enzyme
        concentration, mechanism), so it is not interchangeable with Kd/Ki.
        """
        return self is not Endpoint.IC50


class Relation(StrEnum):
    """The relation between the reported number and the true value."""

    EQ = "="
    LT = "<"
    GT = ">"
    LE = "<="
    GE = ">="

    @property
    def is_exact(self) -> bool:
        return self is Relation.EQ

    @property
    def is_censored(self) -> bool:
        return not self.is_exact

    @property
    def is_strict(self) -> bool:
        """True for ``<`` and ``>``; False for ``<=``, ``>=`` and ``=``."""
        return self in (Relation.LT, Relation.GT)

    def flipped(self) -> "Relation":
        """The relation after applying a strictly decreasing transform.

        ``pX`` is a *negative* logarithm, so it reverses order: a concentration
        upper bound is an affinity lower bound. Strictness is preserved.
        """
        return _FLIPPED[self]

    @property
    def bounds_from_above(self) -> bool:
        """Does this relation state the true value is at most the number?"""
        return self in (Relation.LT, Relation.LE)

    @property
    def bounds_from_below(self) -> bool:
        """Does this relation state the true value is at least the number?"""
        return self in (Relation.GT, Relation.GE)


_FLIPPED: dict[Relation, Relation] = {
    Relation.EQ: Relation.EQ,
    Relation.LT: Relation.GT,
    Relation.GT: Relation.LT,
    Relation.LE: Relation.GE,
    Relation.GE: Relation.LE,
}

#: Spellings seen in BindingDB and related sources, mapped to a Relation.
_RELATION_ALIASES: dict[str, Relation] = {
    "": Relation.EQ,
    "=": Relation.EQ,
    "==": Relation.EQ,
    "<": Relation.LT,
    ">": Relation.GT,
    "<=": Relation.LE,
    ">=": Relation.GE,
    "=<": Relation.LE,
    "=>": Relation.GE,
    "≤": Relation.LE,
    "≥": Relation.GE,
    "~": Relation.EQ,
}


def parse_relation(raw: str | None) -> Relation | None:
    """Parse a relation token, or None if it is not recognised.

    ``~`` (approximately) maps to ``=``; that is a policy choice, recorded here
    rather than buried in the parser, and it is lossy.
    """
    if raw is None:
        return Relation.EQ
    return _RELATION_ALIASES.get(raw.strip())


def parse_value_with_relation(raw: str | None) -> tuple[Relation, str] | None:
    """Split a BindingDB-style cell such as ``">10000"`` into relation and value.

    BindingDB embeds the relation in the value column rather than giving it a
    column of its own. Measured in release 202610, roughly 17-29% of values per
    endpoint carry a ``<`` or ``>`` prefix, so treating the cell as a plain
    float would either discard or silently corrupt about a fifth of the data.

    Returns None when the cell is empty. The numeric part is returned as text
    and validated downstream by :func:`nm_to_p_value`.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None

    for token in ("<=", ">=", "=<", "=>", "≤", "≥", "<", ">", "="):
        if text.startswith(token):
            relation = _RELATION_ALIASES[token]
            return relation, text[len(token) :].strip()
    return Relation.EQ, text


class RejectionReason(StrEnum):
    """Why a measurement could not become a p-value."""

    MISSING = "missing"
    NOT_A_NUMBER = "not_a_number"
    NOT_FINITE = "not_finite"
    NON_POSITIVE = "non_positive"
    UNKNOWN_RELATION = "unknown_relation"
    IMPLAUSIBLE_MAGNITUDE = "implausible_magnitude"


@dataclass(frozen=True)
class Rejected:
    """A measurement that was not converted, and why."""

    reason: RejectionReason
    detail: str

    @property
    def ok(self) -> bool:
        return False


@dataclass(frozen=True)
class PValue:
    """A converted measurement on the log scale.

    ``relation`` has already been flipped, so it describes the *p-value*:
    ``PValue(KD, 5.0, Relation.LT)`` means pKd < 5.
    """

    endpoint: Endpoint
    value: float
    relation: Relation

    @property
    def ok(self) -> bool:
        return True

    @property
    def is_exact(self) -> bool:
        return self.relation.is_exact

    @property
    def is_censored(self) -> bool:
        return self.relation.is_censored

    @property
    def label(self) -> str:
        return self.endpoint.p_label

    @property
    def lower_bound(self) -> float | None:
        """The value if this is a lower bound on the p-value, else None."""
        return self.value if self.relation.bounds_from_below else None

    @property
    def upper_bound(self) -> float | None:
        """The value if this is an upper bound on the p-value, else None."""
        return self.value if self.relation.bounds_from_above else None

    def __str__(self) -> str:
        if self.is_exact:
            return f"{self.label} = {self.value:g}"
        return f"{self.label} {self.relation.value} {self.value:g}"


#: Sanity window for a converted p-value. pKd of 25 would mean a femtomolar-
#: squared affinity and almost certainly means the source units were wrong.
#: Off by default: enabling it is a curation policy decision, and it must be
#: recorded in the curation report when used.
DEFAULT_PLAUSIBLE_RANGE: tuple[float, float] = (-2.0, 20.0)


def nm_to_p_value(
    value_nm: float | int | str | None,
    endpoint: Endpoint,
    relation: Relation | str | None = Relation.EQ,
    *,
    plausible_range: tuple[float, float] | None = None,
) -> PValue | Rejected:
    """Convert a concentration in nM to a p-value, preserving censoring.

    >>> nm_to_p_value(100, Endpoint.KD)
    PValue(endpoint=<Endpoint.KD: 'Kd'>, value=7.0, relation=<Relation.EQ: '='>)
    >>> str(nm_to_p_value(10_000, Endpoint.KD, ">"))
    'pKd < 5'
    >>> str(nm_to_p_value(10, Endpoint.KI, "<"))
    'pKi > 8'
    """
    if isinstance(relation, str) or relation is None:
        parsed = parse_relation(relation)
        if parsed is None:
            return Rejected(RejectionReason.UNKNOWN_RELATION, f"unrecognised relation {relation!r}")
        relation = parsed

    if value_nm is None:
        return Rejected(RejectionReason.MISSING, "no value supplied")

    if isinstance(value_nm, str):
        text = value_nm.strip()
        if not text:
            return Rejected(RejectionReason.MISSING, "empty value")
        try:
            numeric = float(text)
        except ValueError:
            return Rejected(RejectionReason.NOT_A_NUMBER, f"cannot parse {value_nm!r} as a number")
    else:
        numeric = float(value_nm)

    if math.isnan(numeric):
        return Rejected(RejectionReason.NOT_A_NUMBER, "value is NaN")
    if math.isinf(numeric):
        return Rejected(RejectionReason.NOT_FINITE, "value is infinite")
    if numeric <= 0.0:
        # log10 is undefined at and below zero. A zero or negative
        # concentration is a data error, not an infinitely tight binder.
        return Rejected(RejectionReason.NON_POSITIVE, f"concentration must be > 0 nM, got {numeric:g}")

    p_value = NEG_LOG_OFFSET - math.log10(numeric)

    if plausible_range is not None:
        low, high = plausible_range
        if not (low <= p_value <= high):
            return Rejected(
                RejectionReason.IMPLAUSIBLE_MAGNITUDE,
                f"{endpoint.p_label} {p_value:g} outside the plausible range [{low:g}, {high:g}]; "
                "check the source units",
            )

    return PValue(endpoint=endpoint, value=p_value, relation=relation.flipped())


def p_value_to_nm(p_value: float) -> float:
    """Inverse of the p-value transform, back to nM."""
    return 10.0 ** (NEG_LOG_OFFSET - p_value)


def p_interval_to_nm(low: float, high: float) -> tuple[float, float]:
    """Convert a p-value interval to nM, reversing the endpoints.

    The transform is decreasing, so the *lower* p-value bound maps to the
    *higher* concentration. Returning them in the wrong order would silently
    invert every reported confidence interval.
    """
    if low > high:
        raise ValueError(f"expected low <= high on the p scale, got [{low}, {high}]")
    return (p_value_to_nm(high), p_value_to_nm(low))


__all__ = [
    "DEFAULT_PLAUSIBLE_RANGE",
    "NEG_LOG_OFFSET",
    "Endpoint",
    "PValue",
    "Rejected",
    "RejectionReason",
    "Relation",
    "nm_to_p_value",
    "p_interval_to_nm",
    "p_value_to_nm",
    "parse_relation",
    "parse_value_with_relation",
]
