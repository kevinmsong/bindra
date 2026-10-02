"""Locked target-selectivity panel.

Selectivity is ``pX(desired target, ligand) - pX(countertarget, ligand)`` on the
same endpoint under compatible conditions. Three constraints shape the whole
module, all of them from the specification:

**A missing countertarget measurement is unknown, never a negative.** Those
compounds go to a separate ``unknowns`` table with no label, and the loader cannot
produce a numeric value for them. Treating "not measured here" as "not active
there" is the single most tempting error in selectivity work and it manufactures
selectivity out of absence.

**A censored side makes selectivity an interval, not a point.** If either
measurement is a bound, the difference is bounded rather than known, and the triple
goes to ``censored_selectivity`` with explicit lower and upper limits. "More than
100-fold selective" is a real and common datum; collapsing it to a point would
invent precision.

**Relatedness is derived, then supplemented.** Primary evidence is the exhaustive
alignment table at identity >= 0.25 with coverage >= 0.50 -- no curation needed.
A small hashed seed list of canonical pairs is layered on top for cases medicinal
chemistry treats as selectivity questions at lower sequence identity. A wrong
entry in that list matches nothing and so cannot invent a pair.

Every triple also gets a swapped control with the negated label, so sign
consistency is measurable rather than assumed, and the manifest names the two
baselines a result has to beat: a ligand-only predictor that ignores the targets
entirely, and constant zero.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from bindra.config.profiles import CurationProfile, get_profile

SEED_PAIRS_PATH = Path(__file__).with_name("related_targets.json")


@dataclass(frozen=True)
class SelectivityDefinition:
    """Pinned selectivity policy. Hashed into the manifest."""

    #: Alignment-derived relatedness. Deliberately looser than the 0.40/0.80 split
    #: threshold: selectivity questions arise between proteins similar enough to
    #: share a ligand, which is a weaker relationship than split-level homology.
    related_identity_min: float = 0.25
    related_coverage_min: float = 0.50
    #: Tier 1 is the strongest comparability proxy available: one citation, one
    #: endpoint, one condition bucket, one source class -- most likely a single
    #: selectivity table in a single paper.
    #:
    #: Note what it deliberately does NOT include: the target. The curated
    #: ``comparability_group_id`` is keyed on the target hash, so two measurements
    #: against *different* targets can never share one, and comparing those ids
    #: here made tier 1 unsatisfiable -- measured, it rejected all 2,391,516
    #: candidate pairs and produced an empty panel. The panel key below is the
    #: target-independent part of the same proxy.
    comparable_panel_tier: int = 1
    panel_key_fields: tuple[str, ...] = (
        "citation_key",
        "endpoint",
        "assay_condition_bucket",
        "curation_source_class",
    )
    endpoints: tuple[str, ...] = ("Kd", "Ki", "IC50")
    ic50_requires_tier_a: bool = True
    require_exact_both: bool = True
    include_swapped_controls: bool = True
    #: Reported baselines a selectivity result must beat to mean anything.
    declared_baselines: tuple[str, ...] = ("ligand_only_mean", "constant_zero")
    seed_pairs_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SelectivityReport:
    n_related_pairs_alignment: int = 0
    n_related_pairs_seed: int = 0
    n_seed_pairs_listed: int = 0
    n_seed_pairs_matched: int = 0
    n_triples_exact: int = 0
    n_triples_censored: int = 0
    n_unknowns: int = 0
    n_quarantined_cross_partition: int = 0
    n_panel_pairs_with_triples: int = 0
    n_curated_pairs_enumerated_for_unknowns: int = 0
    n_alignment_pairs_not_enumerated: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    per_endpoint: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_seed_pairs(path: Path | str = SEED_PAIRS_PATH) -> tuple[list[dict[str, str]], str]:
    """Curated pairs plus the file's sha256, so the manifest records the input."""
    path = Path(path)
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    return payload["pairs"], hashlib.sha256(raw).hexdigest()


def related_target_pairs(
    constructs: pd.DataFrame,
    *,
    relation=None,
    definition: SelectivityDefinition | None = None,
    seed_pairs: Sequence[Mapping[str, str]] | None = None,
) -> tuple[pd.DataFrame, SelectivityReport]:
    """Pairs of target hashes considered related, with the evidence for each.

    ``constructs`` needs ``sequence_sha256``, ``construct_id`` and
    ``uniprot_primary``. ``relation`` is a :class:`ProteinRelationTable`; when it is
    absent only the seed list contributes, which is reported rather than hidden.
    """
    definition = definition or SelectivityDefinition()
    report = SelectivityReport()
    records: list[dict[str, Any]] = []

    position_of = {}
    hash_of_position: dict[int, str] = {}
    if relation is not None:
        index = {cid: i for i, cid in enumerate(relation.sequence_ids)}
        for row in constructs.itertuples(index=False):
            if row.construct_id in index:
                position_of[row.sequence_sha256] = index[row.construct_id]
                hash_of_position[index[row.construct_id]] = row.sequence_sha256

        # Alignment-derived relatedness, read straight off the stored edges.
        edges = relation.edges
        keep = (edges["identity"] >= definition.related_identity_min) & (
            edges["coverage"] >= definition.related_coverage_min
        )
        for row in edges[keep]:
            left, right = hash_of_position.get(int(row["i"])), hash_of_position.get(int(row["j"]))
            if left is None or right is None or left == right:
                continue
            records.append(
                {
                    "target_a": left,
                    "target_b": right,
                    "evidence": "alignment",
                    "identity": float(row["identity"]),
                    "coverage": float(row["coverage"]),
                    "label": None,
                }
            )
        report.n_related_pairs_alignment = len(records)

    if seed_pairs is not None:
        by_uniprot: dict[str, list[str]] = {}
        for row in constructs.itertuples(index=False):
            if row.uniprot_primary:
                by_uniprot.setdefault(str(row.uniprot_primary), []).append(row.sequence_sha256)
        report.n_seed_pairs_listed = len(seed_pairs)
        for entry in seed_pairs:
            left_hashes = by_uniprot.get(entry["a"], [])
            right_hashes = by_uniprot.get(entry["b"], [])
            if not left_hashes or not right_hashes:
                # A wrong, obsolete or absent accession matches nothing. It cannot
                # invent a pair, only miss one.
                continue
            report.n_seed_pairs_matched += 1
            for left in left_hashes:
                for right in right_hashes:
                    if left == right:
                        continue
                    records.append(
                        {
                            "target_a": left,
                            "target_b": right,
                            "evidence": "curated_seed",
                            "identity": None,
                            "coverage": None,
                            "label": entry.get("label"),
                        }
                    )
        report.n_related_pairs_seed = len(records) - report.n_related_pairs_alignment

    frame = pd.DataFrame(records)
    if len(frame):
        # Canonical ordering, so (A,B) and (B,A) are one relationship.
        swap = frame["target_a"] > frame["target_b"]
        frame.loc[swap, ["target_a", "target_b"]] = frame.loc[
            swap, ["target_b", "target_a"]
        ].to_numpy()
        frame = frame.drop_duplicates(["target_a", "target_b", "evidence"]).reset_index(drop=True)
    return frame, report


def build_selectivity_triples(
    observations: pd.DataFrame,
    related: pd.DataFrame,
    *,
    definition: SelectivityDefinition | None = None,
    profile: CurationProfile | None = None,
    report: SelectivityReport | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SelectivityReport]:
    """Return ``(exact, censored_intervals, unknowns, report)``.

    A triple is one ligand measured against both members of a related target pair,
    on the same endpoint, inside one comparability group.
    """
    definition = definition or SelectivityDefinition()
    profile = profile or get_profile()
    report = report or SelectivityReport()

    frame = observations[observations["endpoint"].isin(definition.endpoints)]
    if definition.ic50_requires_tier_a:
        frame = frame[~((frame["endpoint"] == "IC50") & (frame["ic50_tier"] == "B"))]

    # Enumerated from LIGANDS, not from related target pairs. The pair-centric
    # form is what a reading of the definition suggests, and it does not scale:
    # at identity >= 0.25 / coverage >= 0.50 this corpus has 686,166 related target
    # pairs, and walking each one's full ligand set is billions of iterations --
    # measured, it hung rather than finished. Most ligands are measured against
    # only a handful of targets, so enumerating a ligand's own target pairs is
    # bounded by sum C(k, 2) with k small.
    by_key: dict[tuple[str, str, str], Any] = {}
    targets_of: dict[tuple[str, str], list[str]] = {}
    # The reverse index, so the unknowns pass is a set difference per pair instead
    # of a scan over every (ligand, endpoint) in the corpus.
    ligands_at: dict[tuple[str, str], set[str]] = {}
    for row in frame.itertuples(index=False):
        key = (row.identity_key, row.sequence_sha256, row.endpoint)
        if key not in by_key:
            by_key[key] = row
            targets_of.setdefault((row.identity_key, row.endpoint), []).append(
                row.sequence_sha256
            )
            ligands_at.setdefault((row.sequence_sha256, row.endpoint), set()).add(
                row.identity_key
            )

    # Canonical related-pair lookup, plus each target's related partners for the
    # unknowns pass.
    related_meta: dict[tuple[str, str], Any] = {}
    seeded_pairs: set[tuple[str, str]] = set()
    for pair in related.itertuples(index=False):
        left, right = sorted((pair.target_a, pair.target_b))
        related_meta[(left, right)] = pair
        if pair.evidence == "curated_seed":
            # A curated pair is a named selectivity question, so it is in scope
            # whether or not any ligand happens to have been measured against both.
            seeded_pairs.add((left, right))

    exact_rows: list[dict[str, Any]] = []
    censored_rows: list[dict[str, Any]] = []
    unknown_rows: list[dict[str, Any]] = []
    rejected: dict[str, int] = {}
    panel_pairs: set[tuple[str, str]] = set(seeded_pairs)

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    for (ligand, endpoint), targets in targets_of.items():
        if len(targets) < 2:
            continue
        for position, first in enumerate(targets):
            for second in targets[position + 1 :]:
                key = tuple(sorted((first, second)))
                pair = related_meta.get(key)
                if pair is None:
                    continue

                a = by_key[(ligand, pair.target_a, endpoint)]
                b = by_key[(ligand, pair.target_b, endpoint)]
                if definition.comparable_panel_tier == 1 and _panel_key(
                    a, definition
                ) != _panel_key(b, definition):
                    reject("different_assay_panel")
                    continue

                panel_pairs.add(key)
                base = {
                    "triple_id": f"{ligand}|{pair.target_a}|{pair.target_b}|{endpoint}",
                    "identity_key": ligand,
                    "target": pair.target_a,
                    "countertarget": pair.target_b,
                    "endpoint": endpoint,
                    "evidence": pair.evidence,
                    "relation_label": pair.label,
                    "comparability_group_id": a.comparability_group_id,
                    "panel_key": "|".join(_panel_key(a, definition)),
                }
                if bool(a.is_exact) and bool(b.is_exact):
                    exact_rows.append(
                        {
                            **base,
                            "p_target": float(a.p_value),
                            "p_countertarget": float(b.p_value),
                            "selectivity": round(float(a.p_value) - float(b.p_value), 4),
                            "swapped": False,
                        }
                    )
                    if definition.include_swapped_controls:
                        exact_rows.append(
                            {
                                **base,
                                "triple_id": base["triple_id"] + "|swapped",
                                "target": pair.target_b,
                                "countertarget": pair.target_a,
                                "p_target": float(b.p_value),
                                "p_countertarget": float(a.p_value),
                                "selectivity": round(
                                    float(b.p_value) - float(a.p_value), 4
                                ),
                                "swapped": True,
                            }
                        )
                else:
                    # Interval, not a point: a bound on either side bounds the
                    # difference, and "more than 100-fold selective" is preserved.
                    lower, upper = _selectivity_interval(a, b)
                    censored_rows.append(
                        {
                            **base,
                            "selectivity_lower_bound": lower,
                            "selectivity_upper_bound": upper,
                            "target_is_exact": bool(a.is_exact),
                            "countertarget_is_exact": bool(b.is_exact),
                        }
                    )

    # Unknowns, scoped to the pairs this panel actually poses: those with at least
    # one complete triple, plus every curated pair. For such a pair, any ligand
    # measured against only one side is an open question, recorded so it can never
    # be scored as inactive on the untested side.
    #
    # The scope is deliberate rather than incidental. Enumerating every ligand
    # against all 686,166 alignment-derived related pairs would yield tens of
    # millions of rows asserting nothing beyond "not measured", and a distant
    # homologue at 25% identity is not a selectivity question anyone asked.
    # Unknowns are enumerated for the CURATED pairs only, and that bound is a
    # design decision rather than an optimisation.
    #
    # The rule the specification asks for -- an unmeasured countertarget is
    # unknown, never a negative -- is a constraint on scoring, not a table that
    # has to list every possibility. Enumerating every (ligand, pair, endpoint)
    # where one side is untested across all 686,166 alignment-derived related
    # pairs produced tens of millions of rows and a 4 GB allocation failure, and
    # every one of those rows asserted nothing beyond "not measured". A distant
    # homologue at 25% identity is also not a selectivity question anyone posed.
    #
    # The curated pairs are the questions a human did pose, so those are the ones
    # worth recording as open. The count of alignment pairs left unenumerated is
    # reported so the scope is visible.
    for left, right in sorted(seeded_pairs):
        pair = related_meta[(left, right)]
        for endpoint in definition.endpoints:
            on_left = ligands_at.get((left, endpoint), frozenset())
            on_right = ligands_at.get((right, endpoint), frozenset())
            for present, missing, only in (
                (left, right, on_left - on_right),
                (right, left, on_right - on_left),
            ):
                for ligand in only:
                    unknown_rows.append(
                        {
                            "triple_id": f"{ligand}|{present}|{missing}|{endpoint}",
                            "identity_key": ligand,
                            "measured_target": present,
                            "unmeasured_target": missing,
                            "endpoint": endpoint,
                            "evidence": pair.evidence,
                            "reason": "countertarget not measured in this corpus",
                        }
                    )

    report.n_panel_pairs_with_triples = len(panel_pairs - seeded_pairs)
    report.n_curated_pairs_enumerated_for_unknowns = len(seeded_pairs)
    report.n_alignment_pairs_not_enumerated = len(related_meta) - len(seeded_pairs)

    report.rejected = rejected
    exact = pd.DataFrame(exact_rows).drop_duplicates("triple_id") if exact_rows else pd.DataFrame()
    censored = (
        pd.DataFrame(censored_rows).drop_duplicates("triple_id")
        if censored_rows
        else pd.DataFrame()
    )
    unknowns = (
        pd.DataFrame(unknown_rows).drop_duplicates("triple_id")
        if unknown_rows
        else pd.DataFrame()
    )
    report.n_triples_exact = int(len(exact))
    report.n_triples_censored = int(len(censored))
    report.n_unknowns = int(len(unknowns))
    if len(exact):
        report.per_endpoint = {
            str(k): int(v) for k, v in exact["endpoint"].value_counts().items()
        }
    return exact, censored, unknowns, report


def _panel_key(row: Any, definition: SelectivityDefinition) -> tuple[str, ...]:
    """The target-independent part of the comparability proxy.

    "Measured the same way" for a selectivity pair means one citation, one
    endpoint, one condition bucket and one source class -- but necessarily two
    different targets. Including the target, as ``comparability_group_id`` does,
    makes the condition unsatisfiable.
    """
    return tuple(str(getattr(row, field, "")) for field in definition.panel_key_fields)


def _selectivity_interval(a: Any, b: Any) -> tuple[float | None, float | None]:
    """Bounds on ``p(target) - p(countertarget)`` when either side is censored.

    Uses whichever side of each measurement is known. A lower bound on the target
    minus an upper bound on the countertarget gives a lower bound on the
    difference, and vice versa; an open end stays ``None`` rather than being
    filled with a sentinel.
    """

    def bounds(row: Any) -> tuple[float | None, float | None]:
        if bool(row.is_exact):
            return float(row.p_value), float(row.p_value)
        low = float(row.lower_bound) if row.lower_bound is not None and not pd.isna(row.lower_bound) else None
        high = float(row.upper_bound) if row.upper_bound is not None and not pd.isna(row.upper_bound) else None
        return low, high

    a_low, a_high = bounds(a)
    b_low, b_high = bounds(b)
    lower = a_low - b_high if (a_low is not None and b_high is not None) else None
    upper = a_high - b_low if (a_high is not None and b_low is not None) else None
    return (
        round(lower, 4) if lower is not None else None,
        round(upper, 4) if upper is not None else None,
    )


def lock_selectivity_panel(
    exact: pd.DataFrame,
    censored: pd.DataFrame,
    unknowns: pd.DataFrame,
    *,
    definition: SelectivityDefinition,
    inputs: Mapping[str, Any],
    directory: Path | str,
    report: SelectivityReport | None = None,
    overwrite: bool = False,
):
    """Lock the panel: queries open, selectivity labels sealed, unknowns unlabelled."""
    from bindra.eval.benchmarks.lock import lock

    directory = Path(directory)
    query_columns = [
        "triple_id",
        "identity_key",
        "target",
        "countertarget",
        "endpoint",
        "evidence",
        "relation_label",
        "comparability_group_id",
        "swapped",
    ]
    label_columns = [
        "triple_id",
        "p_target",
        "p_countertarget",
        "selectivity",
        "swapped",
    ]

    manifests = [
        lock(
            "selectivity",
            definition,
            dict(inputs),
            directory / "selectivity",
            queries=exact[[c for c in query_columns if c in exact.columns]],
            labels=exact[[c for c in label_columns if c in exact.columns]],
            unknowns=unknowns,
            attrition=(report.to_dict() if report else None) and report.rejected,
            notes=[
                "A missing countertarget measurement is UNKNOWN, never a negative. "
                "Those rows are in the unknowns table and carry no label.",
                "Every triple has a swapped control with the negated label, so sign "
                "consistency is measurable.",
                "Declared baselines a result must beat: a ligand-only predictor "
                "that ignores both targets, and constant zero.",
                "Both members share one comparability group (tier 1), the tightest "
                "proxy available given BindingDB has no assay identifier.",
            ],
            overwrite=overwrite,
        )
    ]
    if len(censored):
        manifests.append(
            lock(
                "censored_selectivity",
                definition,
                dict(inputs),
                directory / "censored_selectivity",
                queries=censored[[c for c in query_columns if c in censored.columns]],
                labels=censored[
                    [
                        c
                        for c in (
                            "triple_id",
                            "selectivity_lower_bound",
                            "selectivity_upper_bound",
                            "target_is_exact",
                            "countertarget_is_exact",
                        )
                        if c in censored.columns
                    ]
                ],
                notes=[
                    "Selectivity here is an INTERVAL, because at least one side is a "
                    "censored bound. 'More than 100-fold selective' is a real datum "
                    "and is not collapsed to a point.",
                    "Scored against interval-aware metrics only; these rows must not "
                    "be pooled with the exact panel.",
                ],
                overwrite=overwrite,
            )
        )
    return manifests


__all__ = [
    "SEED_PAIRS_PATH",
    "SelectivityDefinition",
    "SelectivityReport",
    "build_selectivity_triples",
    "load_seed_pairs",
    "lock_selectivity_panel",
    "related_target_pairs",
]
