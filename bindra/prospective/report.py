"""Per-round prospective reports.

Five statuses, reported as a funnel and never collapsed:

``planned``    in the frozen manifest
``selected``   carried forward under the frozen budget
``obtained``   physically in hand
``assayed``    a measurement was attempted and produced a result
``confirmed``  the measurement met the prespecified threshold

Collapsing them produces the two most common ways a prospective study
overstates itself: dividing confirmed by *selected* rather than by *assayed*
flatters nothing, but dividing by *obtained* while quietly dropping compounds
that failed synthesis inflates the hit rate by exactly the attrition.

**A retrospective refit is never presented as the original prediction.** The
manifest's digest is carried into every report, and
:func:`assert_prediction_is_original` refuses a report whose predictions were
produced by a checkpoint other than the frozen one. Refits are welcome -- as a
clearly labelled separate block, never in the headline.

Owed by the specification:
    Upgrade C: separate planned, selected, obtained, assayed and confirmed
    status. Never present a retrospective refit as the original prospective
    prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bindra.prospective.ingest import IngestedRound, RecordKind
from bindra.prospective.manifest import ProspectiveManifest


class ReportError(ValueError):
    """A report would misrepresent what the study actually committed to."""


#: The funnel, in order. Each stage is a subset of the one before it.
STAGES: tuple[str, ...] = ("planned", "selected", "obtained", "assayed", "confirmed")


@dataclass
class RoundReport:
    """One round's funnel, with the manifest it is accountable to."""

    manifest: ProspectiveManifest
    planned: set = field(default_factory=set)
    selected: set = field(default_factory=set)
    obtained: set = field(default_factory=set)
    assayed: set = field(default_factory=set)
    confirmed: set = field(default_factory=set)
    #: Anything computed after the results were seen. Reported apart.
    retrospective: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Each stage must nest inside the previous one, or the funnel is not a
        # funnel and the ratios between stages mean nothing.
        for earlier, later in zip(STAGES, STAGES[1:]):
            a, b = getattr(self, earlier), getattr(self, later)
            if not b <= a:
                raise ReportError(
                    f"{later} is not a subset of {earlier}: "
                    f"{sorted(b - a)[:5]} appear in {later} but not {earlier}"
                )

    def funnel(self) -> dict:
        counts = {stage: len(getattr(self, stage)) for stage in STAGES}
        return {
            "counts": counts,
            "attrition": {
                f"{a}_to_{b}": counts[a] - counts[b] for a, b in zip(STAGES, STAGES[1:])
            },
            # The only defensible hit rate: confirmed over what was actually
            # measured. Named so it cannot be mistaken for the others.
            "confirmed_over_assayed": (
                counts["confirmed"] / counts["assayed"] if counts["assayed"] else None
            ),
            "confirmed_over_selected": (
                counts["confirmed"] / counts["selected"] if counts["selected"] else None
            ),
            "note": (
                "confirmed_over_assayed is the hit rate. confirmed_over_selected "
                "is lower and includes compounds never made or never measured; "
                "it answers a different question and is shown so neither can be "
                "quoted as the other."
            ),
        }

    def to_dict(self) -> dict:
        payload = {
            "round_id": self.manifest.round_id,
            "manifest_digest": self.manifest.digest(),
            "model_checkpoint": self.manifest.model_checkpoint,
            "selection_date": self.manifest.selection_date,
            "ranking_rule": self.manifest.ranking_rule,
            "prospective": self.funnel(),
        }
        if self.retrospective:
            payload["retrospective"] = dict(self.retrospective) | {
                "warning": (
                    "computed after the results were known. Not a prospective "
                    "prediction and not comparable to the block above."
                )
            }
        return payload


def from_round(manifest: ProspectiveManifest, round_: IngestedRound, *, threshold: float,
               higher_is_better: bool = True) -> RoundReport:
    """Assemble the funnel from a frozen manifest and the ingested results."""
    planned = set(manifest.candidate_codes)
    obtained = {r.code for r in round_.of(RecordKind.IDENTITY)} & planned
    assayed = round_.measured_codes() & planned
    confirmed = {
        r.code
        for r in round_.of(RecordKind.BINDING)
        if r.code in planned
        and (r.payload["value"] >= threshold if higher_is_better else r.payload["value"] <= threshold)
    }
    # Obtained must cover assayed: something measured was necessarily in hand,
    # even if no identity record was filed for it.
    return RoundReport(
        manifest=manifest,
        planned=planned,
        selected=planned,
        obtained=obtained | assayed,
        assayed=assayed,
        confirmed=confirmed,
    )


def assert_prediction_is_original(report: RoundReport, checkpoint: str) -> None:
    """The predictions must come from the checkpoint the manifest froze."""
    if checkpoint != report.manifest.model_checkpoint:
        raise ReportError(
            f"predictions came from {checkpoint!r} but the manifest froze "
            f"{report.manifest.model_checkpoint!r}. A different checkpoint is a "
            "retrospective refit and must be reported as one."
        )


__all__ = [
    "STAGES",
    "ReportError",
    "RoundReport",
    "assert_prediction_is_original",
    "from_round",
]
