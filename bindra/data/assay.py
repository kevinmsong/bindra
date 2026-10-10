"""The recorded conditions of a measurement, as a fixed-width vector.

An IC50 is not a property of a protein-ligand pair. It moves with substrate and
enzyme concentration, with pH and temperature, and with how carefully whoever
recorded it did so. IC50 is **72.2%** of the val partition, and part of what the
measured replicate-noise floor (0.671 for IC50) attributes to irreducible
scatter is condition variation that the corpus *already records* and the model
was never shown.

``bindra.model.BindraModel`` consumed only ``(protein_features, protein_mask,
ligand_batch)``. These columns were read into every batch by
``bindra.data.batching.OBSERVATION_COLUMNS`` and then discarded.

**What is deliberately excluded, and why it matters more than what is included.**
``comparability_group_id`` and ``replicate_group_id`` are *identity*, not
condition. Replicates of one measurement share a group id, so a model given the
group could learn the group's mean label rather than any chemistry -- a leak
that would look like a large improvement on a random split and vanish on a cold
one. Only columns describing *how the experiment was run* appear below.

Every field is one-hot over a vocabulary fixed here rather than inferred from
whatever rows a run happens to see. An inferred vocabulary would silently change
width between the training set and a panel, and a checkpoint would stop loading.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Unknown or unseen category. Always index 0, so an unseen value is explicitly
#: "unknown" rather than silently colliding with a real category.
UNKNOWN = "<unk>"


@dataclass(frozen=True)
class CategoricalField:
    """One column and the values it is allowed to take."""

    column: str
    values: tuple[str, ...]

    @property
    def width(self) -> int:
        return len(self.values) + 1  # +1 for UNKNOWN

    def encode(self, raw) -> np.ndarray:
        out = np.zeros((len(raw), self.width), dtype=np.float32)
        lookup = {value: index + 1 for index, value in enumerate(self.values)}
        indices = np.fromiter(
            (lookup.get(str(value), 0) for value in raw), dtype=np.int64, count=len(raw)
        )
        out[np.arange(len(raw)), indices] = 1.0
        return out


@dataclass(frozen=True)
class BooleanField:
    """One column that is already a yes/no fact."""

    column: str

    @property
    def width(self) -> int:
        return 1

    def encode(self, raw) -> np.ndarray:
        return np.asarray(
            [1.0 if bool(value) else 0.0 for value in raw], dtype=np.float32
        ).reshape(-1, 1)


#: Measured on the 202610 snapshot. ``assay_condition_bucket`` is the pH and
#: temperature bucket the curation layer assigned ("pH7.5|T2"); the long tail
#: beyond these falls to UNKNOWN rather than widening the vector.
CONDITION_BUCKETS = (
    "pH?|T?", "pH?|T0", "pH?|T1", "pH?|T2", "pH?|T3",
    "pH7.0|T?", "pH7.0|T0", "pH7.0|T1", "pH7.0|T2", "pH7.0|T3",
    "pH7.4|T?", "pH7.4|T0", "pH7.4|T1", "pH7.4|T2", "pH7.4|T3",
    "pH7.5|T?", "pH7.5|T0", "pH7.5|T1", "pH7.5|T2", "pH7.5|T3",
    "pH8.0|T?", "pH8.0|T0", "pH8.0|T1", "pH8.0|T2", "pH8.0|T3",
)

#: Order is the vector's layout and must not be reordered: a checkpoint holds
#: weights against these positions.
ASSAY_FIELDS: tuple[object, ...] = (
    CategoricalField("endpoint", ("Kd", "Ki", "IC50")),
    CategoricalField("assay_condition_bucket", CONDITION_BUCKETS),
    # A is a directly-reported IC50; B is one the curation layer had to infer.
    CategoricalField("ic50_tier", ("A", "B", "na")),
    CategoricalField("metadata_confidence", ("low", "medium", "high")),
    # Patent tables and screening panels are measurably noisier than
    # literature-curated values; the model can learn to trust them less.
    CategoricalField(
        "curation_source_class",
        ("literature_curated", "patent", "pubchem_bioassay", "screening_panel"),
    ),
    CategoricalField(
        "comparability_scope",
        ("literature_assay_proxy", "patent_table_proxy", "bioassay_aid", "unknown_singleton"),
    ),
    BooleanField("has_ph"),
    BooleanField("has_temperature"),
)

#: Columns the loader must read for :func:`encode_assay`.
ASSAY_COLUMNS: tuple[str, ...] = tuple(field.column for field in ASSAY_FIELDS)

#: Width of the encoded vector. Fixed, so a checkpoint's assay encoder is valid
#: against any partition or panel.
ASSAY_DIM: int = sum(field.width for field in ASSAY_FIELDS)

#: Identity columns that must never be encoded. Named rather than merely absent,
#: so a future addition has to argue with this list.
EXCLUDED_AS_LEAKAGE: tuple[str, ...] = (
    "comparability_group_id",
    "replicate_group_id",
    "observation_id",
    "citation_key",
    "pair_key",
)


class AssayEncodingError(ValueError):
    """The frame cannot produce the declared assay vector."""


def encode_assay(frame) -> np.ndarray:
    """``[n_rows, ASSAY_DIM]`` float32, one row per observation.

    Raises rather than filling a missing column with zeros. A zero block reads
    as a real category to the model, and a silently absent condition is exactly
    the failure this module exists to remove.
    """
    missing = [name for name in ASSAY_COLUMNS if name not in frame.columns]
    if missing:
        raise AssayEncodingError(
            f"frame is missing assay columns {missing}; add them to "
            "OBSERVATION_COLUMNS rather than encoding zeros for them"
        )
    for name in EXCLUDED_AS_LEAKAGE:
        if name in ASSAY_COLUMNS:  # pragma: no cover - guards the table above
            raise AssayEncodingError(
                f"{name!r} is identity, not condition, and must not be encoded"
            )
    blocks = [field.encode(frame[field.column].to_numpy()) for field in ASSAY_FIELDS]
    encoded = np.concatenate(blocks, axis=1)
    if encoded.shape[1] != ASSAY_DIM:
        raise AssayEncodingError(
            f"encoded width {encoded.shape[1]} does not match declared "
            f"ASSAY_DIM {ASSAY_DIM}"
        )
    return encoded


def describe() -> dict[str, object]:
    """What went into the vector, for the run's provenance record."""
    return {
        "assay_dim": ASSAY_DIM,
        "fields": [
            {"column": f.column, "width": f.width, "kind": type(f).__name__}
            for f in ASSAY_FIELDS
        ],
        "excluded_as_leakage": list(EXCLUDED_AS_LEAKAGE),
    }


__all__ = [
    "ASSAY_COLUMNS",
    "ASSAY_DIM",
    "ASSAY_FIELDS",
    "EXCLUDED_AS_LEAKAGE",
    "AssayEncodingError",
    "BooleanField",
    "CategoricalField",
    "describe",
    "encode_assay",
]
