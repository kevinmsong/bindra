"""Locked temporal panels.

Three panels, locked together and reported separately, because pooling them would
answer a different question than the one asked:

``panel_strict_future``
    Pairs whose first known evidence postdates the cutoff, with no observation at
    or before it across any endpoint or source. This is the forward-looking test.
``panel_repeated_measurement``
    Pairs with evidence on both sides of the cutoff. Explicitly labelled and kept
    out of the strict panel: a pair already measured before the cutoff is not a
    novel future observation, however many new measurements it later accrued.
``panel_reimport_control``
    Observations published before the cutoff but *imported* after it. A positive
    control on real data rather than a synthetic fixture -- this class is populous
    in release 202610 -- asserting that a recent database import of an old
    measurement is never treated as new evidence.

``bindra.data.splits.make_temporal_split`` already produces exactly these three
populations, so this module packages them rather than recomputing them. That
matters: a panel derived by a second, parallel implementation could disagree with
the split a model was trained against.

The manifest also carries a ``pretraining_exposure`` block. Neither encoder can be
shown to predate the cutoff in the sense that matters, because their pretraining
corpora are not enumerable here, so the block records what is known and states
plainly that historical cleanliness of the pretrained system **cannot** be
established.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from bindra.config.profiles import CurationProfile, get_profile
from bindra.data.splits import Partition, make_temporal_split


@dataclass(frozen=True)
class TemporalDefinition:
    """Pinned temporal policy. Hashed into the manifest."""

    cutoff: str = "2023-12-31"
    #: Publication date, never the import date. A 2025 import of a 1998
    #: measurement is not 2025 evidence.
    evidence_field: str = "date_published"
    train_cutoff: str = "2022-12-31"
    val_window: tuple[str, str] = ("2023-01-01", "2023-06-30")
    cal_window: tuple[str, str] = ("2023-07-01", "2023-12-31")
    #: First evidence is the minimum over ALL of a pair's observations, across
    #: endpoints and sources, so a pair cannot look new because one measurement is.
    pair_level: bool = True
    exclude_unknown_date: bool = True
    #: Pairs straddling the cutoff get their own panel: they cannot satisfy pair
    #: integrity and temporal purity at once.
    straddling_pairs_separate_panel: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def pretraining_exposure() -> dict[str, Any]:
    """What is, and is not, known about the pretrained encoders' time coverage."""
    from bindra.encoders.protein_esmc import ESMC_600M_REPO, ESMC_600M_REVISION

    return {
        "protein_encoder": {
            "checkpoint": ESMC_600M_REPO,
            "revision": ESMC_600M_REVISION,
            "training_corpus_enumerable": False,
        },
        "ligand_encoder": {"checkpoint": "KPGT", "status": "not yet integrated"},
        "claim": (
            "The pretrained encoders may have seen test-set sequences or molecules "
            "as unlabelled pretraining data. This panel establishes that no "
            "labelled observation after the cutoff informed training, tuning or "
            "calibration. It does NOT establish that the pretrained system is "
            "historically clean, and that cannot be established from here."
        ),
    }


def build_temporal_panels(
    frame: pd.DataFrame,
    *,
    definition: TemporalDefinition | None = None,
    profile: CurationProfile | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Return the three panels plus a report, from one temporal split."""
    profile = profile or get_profile()
    definition = definition or TemporalDefinition(
        cutoff=profile.splits.temporal_cutoff,
        train_cutoff=profile.splits.temporal_train_cutoff,
        val_window=tuple(profile.splits.temporal_val_window),
        cal_window=tuple(profile.splits.temporal_cal_window),
    )

    assignment, attrition = make_temporal_split(frame, policy=profile.splits)
    labelled = frame.assign(partition=frame["observation_id"].map(assignment))

    cutoff = pd.Timestamp(definition.cutoff)
    published = pd.to_datetime(labelled["first_evidence_date"], errors="coerce")
    if "date_in_bindingdb" in labelled.columns:
        imported = pd.to_datetime(labelled["date_in_bindingdb"], errors="coerce")
        reimports = labelled[(published <= cutoff) & (imported > cutoff)]
    else:
        reimports = labelled.iloc[0:0]

    panels = {
        "panel_strict_future": labelled[labelled["partition"] == Partition.TEST.value],
        "panel_repeated_measurement": labelled[
            labelled["partition"] == Partition.REPEATED_MEASUREMENT.value
        ],
        "panel_reimport_control": reimports,
    }

    report = {
        "definition": definition.to_dict(),
        "attrition": dict(attrition),
        "counts": {
            name: {
                "n_observations": int(len(panel)),
                "n_pairs": int(panel["pair_key"].nunique()) if len(panel) else 0,
                "n_targets": int(panel["sequence_sha256"].nunique()) if len(panel) else 0,
            }
            for name, panel in panels.items()
        },
        "pretraining_exposure": pretraining_exposure(),
        "notes": [
            "panel_strict_future is the forward-looking test. "
            "panel_repeated_measurement is kept separate because a pair already "
            "measured before the cutoff is not a novel future observation.",
            "panel_reimport_control is a positive control on real data: these "
            "observations predate the cutoff and were imported after it, and must "
            "never be scored as future evidence.",
            "Unknown-date pairs are excluded from every temporal claim rather than "
            "assumed old or new.",
        ],
    }
    return panels, report


def lock_temporal_panels(
    panels: Mapping[str, pd.DataFrame],
    report: Mapping[str, Any],
    *,
    inputs: Mapping[str, Any],
    directory: Path | str,
    overwrite: bool = False,
) -> list[Any]:
    """Lock each panel separately, with labels sealed."""
    from bindra.eval.benchmarks.lock import lock

    directory = Path(directory)
    query_columns = [
        "observation_id",
        "pair_key",
        "identity_key",
        "sequence_sha256",
        "endpoint",
        "first_evidence_date",
    ]
    label_columns = ["observation_id", "endpoint", "p_value", "is_exact", "is_censored"]

    locked = []
    for name, panel in panels.items():
        queries = [c for c in query_columns if c in panel.columns]
        labels = [c for c in label_columns if c in panel.columns]
        locked.append(
            lock(
                name,
                {**report["definition"], "panel": name},
                dict(inputs),
                directory / name,
                queries=panel[queries],
                labels=panel[labels] if labels else None,
                attrition=report["attrition"],
                notes=[*report["notes"], report["pretraining_exposure"]["claim"]],
                overwrite=overwrite,
            )
        )
    return locked


__all__ = [
    "TemporalDefinition",
    "build_temporal_panels",
    "lock_temporal_panels",
    "pretraining_exposure",
]
