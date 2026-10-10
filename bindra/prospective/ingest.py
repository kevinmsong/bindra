"""Experimental result ingestion and QC.

Five record kinds, kept apart because merging them destroys information that
cannot be recovered:

``identity``    what was actually made, and whether it is what was ordered
``assay``       the conditions the measurement ran under
``binding``     a binding measurement
``functional``  a functional measurement, which is not a binding measurement
``failure``     the experiment did not produce a result

**A technical failure is not a negative result.** A compound that would not
dissolve, an assay that failed QC, a plate that was dropped -- none of these is
evidence that the compound does not bind, and scoring them as inactive turns
synthesis and handling problems into model errors. They are recorded as
``failure`` and excluded from every hit-rate denominator, with the exclusion
counted so the attrition is visible.

**A functional result is not a binding result.** An EC50 in a cell assay and a
Kd are different quantities; a model predicting affinity cannot be scored
against the former without saying so.

Owed by the specification:
    Upgrade C: keep identity, assay, binding, functional and failure records
    separate; a technical failure is not a negative result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class IngestError(ValueError):
    """A result could not be filed under any kind, or under the wrong one."""


class RecordKind(str, Enum):
    IDENTITY = "identity"
    ASSAY = "assay"
    BINDING = "binding"
    FUNCTIONAL = "functional"
    FAILURE = "failure"


#: Why an experiment produced no measurement. None of these is a negative
#: result, and the list is explicit so "other" cannot quietly absorb a real
#: inactive.
FAILURE_REASONS: tuple[str, ...] = (
    "synthesis_failed",
    "insufficient_material",
    "insoluble",
    "assay_qc_failed",
    "plate_error",
    "compound_unstable",
    "interference",
    "not_attempted",
    "other_technical",
)


@dataclass
class Record:
    """One filed result."""

    code: str
    kind: RecordKind
    payload: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind is RecordKind.FAILURE:
            reason = self.payload.get("reason")
            if reason not in FAILURE_REASONS:
                raise IngestError(
                    f"failure record for {self.code} has reason {reason!r}; must "
                    f"be one of {FAILURE_REASONS}. An unnamed failure tends to "
                    "get counted as an inactive."
                )
        if self.kind in (RecordKind.BINDING, RecordKind.FUNCTIONAL):
            if "value" not in self.payload:
                raise IngestError(f"{self.kind.value} record for {self.code} has no value")


@dataclass
class IngestedRound:
    """Every record for one round, filed by kind and never merged."""

    round_id: str
    records: list[Record] = field(default_factory=list)

    def of(self, kind: RecordKind) -> "list[Record]":
        return [r for r in self.records if r.kind is kind]

    def add(self, record: Record) -> None:
        self.records.append(record)

    # -- the denominators, which is where the rule bites -----------------

    def codes(self) -> set[str]:
        return {r.code for r in self.records}

    def measured_codes(self) -> set[str]:
        """Codes with a real binding measurement. The only valid denominator."""
        return {r.code for r in self.of(RecordKind.BINDING)}

    def failed_codes(self) -> set[str]:
        return {r.code for r in self.of(RecordKind.FAILURE)}

    def summary(self) -> dict:
        planned = self.codes()
        measured = self.measured_codes()
        failed = self.failed_codes()
        return {
            "round_id": self.round_id,
            "n_records": len(self.records),
            "n_codes": len(planned),
            "n_binding": len(measured),
            "n_functional": len({r.code for r in self.of(RecordKind.FUNCTIONAL)}),
            "n_failures": len(failed),
            "failures_by_reason": {
                reason: sum(
                    1 for r in self.of(RecordKind.FAILURE) if r.payload.get("reason") == reason
                )
                for reason in FAILURE_REASONS
                if any(r.payload.get("reason") == reason for r in self.of(RecordKind.FAILURE))
            },
            "hit_rate_denominator": len(measured),
            "note": (
                f"{len(failed)} technical failures are excluded from the "
                "denominator. They are not negative results and scoring them "
                "as inactive would charge handling problems to the model."
            ),
        }


def hit_rate(round_: IngestedRound, *, threshold: float, higher_is_better: bool = True) -> dict:
    """Confirmed hits over **measured** candidates, never over planned ones."""
    binding = round_.of(RecordKind.BINDING)
    if not binding:
        return {
            "status": "undefined",
            "reason": "no binding measurements in this round",
            "n_failures": len(round_.failed_codes()),
        }
    hits = [
        r for r in binding
        if (r.payload["value"] >= threshold if higher_is_better else r.payload["value"] <= threshold)
    ]
    return {
        "n_measured": len(binding),
        "n_hits": len(hits),
        "hit_rate": len(hits) / len(binding),
        "threshold": threshold,
        "excluded_technical_failures": len(round_.failed_codes()),
        "denominator": "measured binding records only",
    }


def assert_failures_not_counted(round_: IngestedRound, denominator_codes: "set[str]") -> None:
    """A denominator must contain no technical failure."""
    overlap = round_.failed_codes() & set(denominator_codes)
    if overlap:
        raise IngestError(
            f"{len(overlap)} technical failures are in the denominator "
            f"({sorted(overlap)[:5]}...); a failed experiment is not an inactive"
        )


__all__ = [
    "FAILURE_REASONS",
    "IngestError",
    "IngestedRound",
    "Record",
    "RecordKind",
    "assert_failures_not_counted",
    "hit_rate",
]
