"""Curation-source classes, loaded from the authored map.

``Curation/DataSource`` is the only field in BindingDB that discriminates how a
measurement was obtained -- UniProt presence does not (it is near-universal), and
pH/temperature are populated too rarely to carry anything. So this small mapping
drives both IC50 tiering and metadata confidence, and it lives in a hashed JSON
file rather than in code.

An unmapped source is a hard error. Defaulting it to ``other`` would silently
admit or exclude a whole new data source from the primary regression the first
time BindingDB adds one.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

SOURCE_CLASSES_PATH = Path(__file__).with_name("source_classes.json")


class UnknownCurationSource(KeyError):
    """A ``Curation/DataSource`` value the authored map does not classify."""


@dataclass(frozen=True)
class SourceClassMap:
    """Source-value -> class, plus the policies keyed on those classes."""

    by_value: Mapping[str, str]
    ic50_tier_a_classes: frozenset[str]
    confidence_high_classes: frozenset[str]
    confidence_capped_medium_classes: frozenset[str]
    sha256: str
    schema_version: str

    def class_of(self, curation_source: str | None) -> str:
        """Class for a source value. Raises on an unmapped value."""
        key = (curation_source or "").strip()
        if not key:
            # A blank source is a real, representable state: it is classified as
            # unknown rather than guessed, and it fails the Tier A and
            # high-confidence tests below.
            return "unknown"
        try:
            return self.by_value[key]
        except KeyError as exc:
            raise UnknownCurationSource(
                f"Curation/DataSource {key!r} is not in {SOURCE_CLASSES_PATH.name}. "
                "Classify it deliberately: the class drives IC50 tiering and "
                "metadata confidence, so defaulting it would silently change which "
                "records enter the primary regression."
            ) from exc

    def is_ic50_tier_a_class(self, curation_source: str | None) -> bool:
        return self.class_of(curation_source) in self.ic50_tier_a_classes

    def confidence_ceiling(self, curation_source: str | None) -> str:
        """The best metadata confidence this source value can reach."""
        return self.confidence_ceiling_for_class(self.class_of(curation_source))

    def confidence_ceiling_for_class(self, source_class: str) -> str:
        """The best metadata confidence a class can reach.

        Separate from :meth:`confidence_ceiling` so a caller annotating millions
        of rows can resolve the class once per distinct value and then look the
        ceiling up by class, instead of re-classifying every row.
        """
        if source_class in self.confidence_capped_medium_classes:
            return "medium"
        if source_class in self.confidence_high_classes:
            return "high"
        return "low"


def load_source_classes(path: Path | str = SOURCE_CLASSES_PATH) -> SourceClassMap:
    path = Path(path)
    raw = path.read_bytes()
    payload: dict[str, Any] = json.loads(raw.decode("utf-8"))

    by_value: dict[str, str] = {}
    for klass, body in payload["classes"].items():
        for value in body["values"]:
            if value in by_value:
                raise ValueError(
                    f"source {value!r} is mapped to both {by_value[value]!r} and {klass!r}"
                )
            by_value[value] = klass

    policy = payload["policy"]
    return SourceClassMap(
        by_value=by_value,
        ic50_tier_a_classes=frozenset(policy["ic50_tier_a_classes"]),
        confidence_high_classes=frozenset(policy["metadata_confidence_high_classes"]),
        confidence_capped_medium_classes=frozenset(
            policy["metadata_confidence_capped_at_medium_classes"]
        ),
        sha256=hashlib.sha256(raw).hexdigest(),
        schema_version=payload["schema_version"],
    )


__all__ = [
    "SOURCE_CLASSES_PATH",
    "SourceClassMap",
    "UnknownCurationSource",
    "load_source_classes",
]
