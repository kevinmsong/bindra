"""Deduplication, replicate aggregation, assay-comparability grouping, tiering.

This is where the corpus becomes a dataset, and where the two hardest honesty
problems in the whole data layer are handled.

**There is no assay identifier in BindingDB.** Not a missing value -- the column
does not exist in any of the 640. So an "assay group" here is an explicitly named
*proxy* built from the target, the endpoint, the citation, a condition bucket and
the curation source class, and ``comparability_scope`` records which proxy was
used so downstream code can state in prose what it assumed. The single most
important rule in this module is that a record with **no citation of any kind
becomes its own singleton group**: the specification forbids a missing identifier
collapsing a whole target into one giant ranking group, and a target with 10,000
provenance-free observations must yield 10,000 groups of one.

**Replicate disagreement is large and must not be averaged away.** Measured on
this release, 18.9% of (ligand, target, endpoint) groups hold more than one exact
measurement, and a third of those disagree by more than 0.5 p-units, with a
maximum spread of 11.0. Discordance is therefore judged against the *measured*
repeatability limit (2.77 sigma_rep, endpoint-specific) rather than a fixed
threshold, and a group that disagrees beyond it keeps its median but is barred
from analog-pair and activity-cliff construction -- a delta computed from
mutually inconsistent measurements means nothing.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from bindra.config.profiles import CurationProfile, get_profile
from bindra.data.bindingdb.sources import SourceClassMap, load_source_classes
from bindra.data.endpoints import Endpoint

#: Citation kinds, in the priority order the comparability key uses.
CITATION_PRIORITY = ("pmid", "doi", "patent", "pubchem_aid")

_DOI_PREFIX = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)\s*", re.IGNORECASE)
_WS = re.compile(r"\s+")


class QcStatus:
    OK = "OK"
    AGGREGATED = "AGGREGATED"
    AGGREGATED_DISCORDANT = "AGGREGATED_DISCORDANT"
    DUPLICATE_IMPORT = "DUPLICATE_IMPORT"
    QUARANTINED = "QUARANTINED"


class QcFlag:
    CROSS_SOURCE_IDENTICAL_VALUE = "CROSS_SOURCE_IDENTICAL_VALUE"
    CENSORING_INCONSISTENT = "CENSORING_INCONSISTENT"
    UNKNOWN_CITATION = "UNKNOWN_CITATION"


class Confidence:
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    ORDER = {LOW: 0, MEDIUM: 1, HIGH: 2}

    @classmethod
    def at_least(cls, value: str, floor: str) -> bool:
        return cls.ORDER.get(value, 0) >= cls.ORDER.get(floor, 0)


class ComparabilityScope:
    LITERATURE = "literature_assay_proxy"
    PATENT = "patent_table_proxy"
    BIOASSAY = "bioassay_aid"
    UNKNOWN_SINGLETON = "unknown_singleton"


def normalize_pmid(value: Any) -> str | None:
    """Digits only. PMIDs arrive with stray prefixes and whitespace."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = _WS.sub("", str(value)).strip()
    if not text:
        return None
    digits = re.sub(r"\D", "", text)
    return digits or None


def normalize_doi(value: Any) -> str | None:
    """Lowercase, prefix-stripped DOI, so one paper is one citation key."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    text = _DOI_PREFIX.sub("", text)
    text = _WS.sub("", text).lower().rstrip(".")
    return text or None


def _clean(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    return text or None


def citation_key(
    *,
    observation_id: str,
    pmid: Any = None,
    doi: Any = None,
    patent: Any = None,
    pubchem_aid: Any = None,
) -> tuple[str, str]:
    """``(key, kind)`` for an observation's most specific citation.

    Priority PMID > DOI > patent > AID: the earlier entries identify a single
    published experiment more tightly than the later ones.

    **When no citation exists the key is the observation's own id.** That single
    line is what satisfies the specification's hard requirement. Without it,
    every provenance-free record for a target would share one key and form a
    single enormous ranking group; with it, each becomes a group of one and is
    therefore structurally unusable for ranking rather than silently misused.
    """
    cleaned_pmid = normalize_pmid(pmid)
    if cleaned_pmid:
        return f"PMID:{cleaned_pmid}", "pmid"
    cleaned_doi = normalize_doi(doi)
    if cleaned_doi:
        return f"DOI:{cleaned_doi}", "doi"
    cleaned_patent = _clean(patent)
    if cleaned_patent:
        return f"PATENT:{cleaned_patent}", "patent"
    cleaned_aid = _clean(pubchem_aid)
    if cleaned_aid:
        return f"AID:{cleaned_aid}", "pubchem_aid"
    return f"UNKNOWN:{observation_id}", "none"


def assay_condition_bucket(
    ph: Any,
    temp_c: Any,
    *,
    ph_rounding: float = 0.5,
    temperature_bins: tuple[float, ...] = (15.0, 25.0, 30.0, 37.0),
) -> str:
    """A coarse condition label, where **missing is its own bucket**.

    ``"pH?"`` deliberately does not merge with ``"pH7.5"``. Treating an absent
    condition as a wildcard would pool records that might have been measured
    under different conditions, which is the opposite of conservative.

    Measured on this release, pH is populated for 6.5% of rows and temperature
    for 6.1%, so in practice this field mostly separates the small minority of
    records that carry conditions from the majority that do not -- which is
    itself information worth keying on.
    """
    ph_value = _clean(ph)
    if ph_value is None:
        ph_label = "pH?"
    else:
        try:
            rounded = round(float(ph_value) / ph_rounding) * ph_rounding
            ph_label = f"pH{rounded:g}"
        except ValueError:
            ph_label = "pH?"

    temp_value = _clean(temp_c)
    if temp_value is None:
        temp_label = "T?"
    else:
        try:
            # Cells carry forms like "25", "25 C", "20-25".
            match = re.search(r"-?\d+(?:\.\d+)?", temp_value)
            if match is None:
                raise ValueError(temp_value)
            number = float(match.group(0))
            index = int(np.searchsorted(np.asarray(temperature_bins), number, side="right"))
            temp_label = f"T{index}"
        except ValueError:
            temp_label = "T?"

    return f"{ph_label}|{temp_label}"


def _hash16(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


def comparability_group_id(
    *,
    sequence_sha256: str,
    endpoint: str,
    citation: str,
    condition_bucket: str,
    source_class: str,
) -> str:
    return _hash16(sequence_sha256, endpoint, citation, condition_bucket, source_class)


def comparability_scope(source_class: str, citation_kind: str) -> str:
    if citation_kind == "none":
        return ComparabilityScope.UNKNOWN_SINGLETON
    if source_class == "patent":
        return ComparabilityScope.PATENT
    if source_class == "pubchem_bioassay" or citation_kind == "pubchem_aid":
        return ComparabilityScope.BIOASSAY
    return ComparabilityScope.LITERATURE


def metadata_confidence(
    *,
    source_class: str,
    citation_kind: str,
    has_uniprot: bool,
    has_ph: bool,
    has_temperature: bool,
    sources: SourceClassMap,
    ceiling: str | None = None,
) -> str:
    """How much the comparability grouping can be trusted for this record.

    ``low`` whenever provenance is missing outright, the target is unidentified,
    or the source class cannot support a claim about the assay. ``high`` requires
    everything: a literature-grade source, a paper-level citation, a UniProt
    identification, and both reported conditions.

    High is therefore rare on this corpus, because conditions are rare. That is
    the honest outcome, and ranking supervision deliberately requires only
    ``medium`` so the objective is not starved by it.
    """
    if ceiling is None:
        ceiling = sources.confidence_ceiling_for_class(source_class)

    if citation_kind == "none" or not has_uniprot or ceiling == Confidence.LOW:
        return Confidence.LOW
    if citation_kind == "pubchem_aid":
        return Confidence.LOW
    if ceiling == Confidence.HIGH:
        if citation_kind in ("pmid", "doi") and has_ph and has_temperature:
            return Confidence.HIGH
        return Confidence.MEDIUM
    return Confidence.MEDIUM


def ic50_tier(
    *,
    endpoint: str,
    source_class: str,
    citation_kind: str,
    has_uniprot: bool,
    single_chain_consistent: bool,
    profile: CurationProfile,
) -> str:
    """``A``, ``B`` or ``na``.

    Only IC50 is tiered: the specification treats Ki and Kd as high-confidence
    binding constants and asks for IC50 to be admitted to the primary regression
    only where the record supports an appropriate biochemical assay.

    Tier B is keyed on the **curation source class**, never on PubChem AID
    presence -- 68% of rows carry an AID, including ~545k whose provenance is a
    US patent, so an AID-based rule would misclassify most of the patent corpus.
    """
    if endpoint not in profile.ic50.tier_applies_to:
        return "na"
    policy = profile.ic50
    qualifies = (
        (single_chain_consistent or not policy.tier_a_requires_chain_consistency)
        and (has_uniprot or not policy.tier_a_requires_uniprot)
        and (citation_kind in ("pmid", "doi", "patent") or not policy.tier_a_requires_citation)
        and source_class in policy.tier_a_source_classes
    )
    return "A" if qualifies else "B"


# ---------------------------------------------------------------------------
# Vectorised pipeline
# ---------------------------------------------------------------------------


@dataclass
class QcReport:
    n_input: int = 0
    n_unusable_after_join: int = 0
    n_duplicate_imports: int = 0
    n_after_dedup: int = 0
    n_cross_source_identical: int = 0
    n_censoring_inconsistent: int = 0
    duplicate_multiplicity: dict = field(default_factory=dict)
    citation_kinds: dict = field(default_factory=dict)
    comparability_scopes: dict = field(default_factory=dict)
    confidence_counts: dict = field(default_factory=dict)
    group_size_summary: dict = field(default_factory=dict)
    singleton_groups: int = 0
    unknown_citation_observations: int = 0
    ic50_tiers: dict = field(default_factory=dict)
    replicate_groups: int = 0
    replicate_groups_multi: int = 0
    discordance_counts: dict = field(default_factory=dict)
    spread_quantiles: dict = field(default_factory=dict)
    eligibility: dict = field(default_factory=dict)
    endpoint_exact_censored: dict = field(default_factory=dict)
    thresholds_used: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


#: Columns the QC pass needs from the raw observation shards.
QC_COLUMNS = (
    "observation_id",
    "source_line",
    "ligand_ref",
    "sequence_ref",
    "endpoint",
    "raw_value_cell",
    "relation_original",
    "relation_p",
    "p_value",
    "is_exact",
    "is_censored",
    "lower_bound",
    "upper_bound",
    "pmid",
    "article_doi",
    "patent_number",
    "pubchem_aid",
    "curation_source",
    "institution",
    "ph",
    "temp_c",
    "uniprot_primary",
    "chain_fields_consistent",
    "declared_chain_count",
    "date_published",
    "first_evidence_date",
    "first_evidence_year",
    # Carried through solely so the temporal audit can run its re-import positive
    # control: an observation published before the cutoff but imported after it is
    # not future evidence, and must never land in the test partition. Without this
    # column that check silently degrades to a no-op.
    "date_in_bindingdb",
)


def annotate_observations(
    frame: pd.DataFrame,
    *,
    ligands: pd.DataFrame,
    constructs: pd.DataFrame,
    profile: CurationProfile | None = None,
    sources: SourceClassMap | None = None,
) -> pd.DataFrame:
    """Join keys, derive citation / comparability / tier / confidence columns."""
    profile = profile or get_profile()
    sources = sources or load_source_classes()

    out = frame.merge(
        ligands[["ligand_ref", "identity_key", "leakage_key", "scaffold_key", "inchikey"]],
        on="ligand_ref",
        how="left",
    ).merge(
        constructs[["sequence_ref", "sequence_sha256", "construct_id", "sequence_length"]],
        on="sequence_ref",
        how="left",
    )

    source_values = out["curation_source"].fillna("")
    # Validate every distinct value once rather than per row, and fail loudly on
    # an unmapped source instead of defaulting it.
    class_of = {value: sources.class_of(value or None) for value in source_values.unique()}
    out["curation_source_class"] = source_values.map(class_of)

    citations = [
        citation_key(
            observation_id=oid,
            pmid=pmid,
            doi=doi,
            patent=patent,
            pubchem_aid=aid,
        )
        for oid, pmid, doi, patent, aid in zip(
            out["observation_id"],
            out["pmid"],
            out["article_doi"],
            out["patent_number"],
            out["pubchem_aid"],
        )
    ]
    out["citation_key"] = [c[0] for c in citations]
    out["citation_kind"] = [c[1] for c in citations]

    policy = profile.comparability
    out["assay_condition_bucket"] = [
        assay_condition_bucket(
            ph,
            temp,
            ph_rounding=policy.ph_rounding,
            temperature_bins=tuple(policy.temperature_bins),
        )
        for ph, temp in zip(out["ph"], out["temp_c"])
    ]

    out["has_uniprot"] = out["uniprot_primary"].notna()
    out["has_ph"] = out["ph"].notna()
    out["has_temperature"] = out["temp_c"].notna()

    ceilings = {
        klass: sources.confidence_ceiling_for_class(klass)
        for klass in out["curation_source_class"].unique()
    }
    out["metadata_confidence"] = [
        metadata_confidence(
            source_class=klass,
            citation_kind=kind,
            has_uniprot=bool(uni),
            has_ph=bool(ph),
            has_temperature=bool(temp),
            sources=sources,
            ceiling=ceilings[klass],
        )
        for klass, kind, uni, ph, temp in zip(
            out["curation_source_class"],
            out["citation_kind"],
            out["has_uniprot"],
            out["has_ph"],
            out["has_temperature"],
        )
    ]

    out["comparability_scope"] = [
        comparability_scope(klass, kind)
        for klass, kind in zip(out["curation_source_class"], out["citation_kind"])
    ]

    out["comparability_group_id"] = [
        comparability_group_id(
            sequence_sha256=str(seq),
            endpoint=str(endpoint),
            citation=str(citation),
            condition_bucket=str(bucket),
            source_class=str(klass),
        )
        for seq, endpoint, citation, bucket, klass in zip(
            out["sequence_sha256"],
            out["endpoint"],
            out["citation_key"],
            out["assay_condition_bucket"],
            out["curation_source_class"],
        )
    ]

    out["ic50_tier"] = [
        ic50_tier(
            endpoint=endpoint,
            source_class=klass,
            citation_kind=kind,
            has_uniprot=bool(uni),
            single_chain_consistent=bool(consistent),
            profile=profile,
        )
        for endpoint, klass, kind, uni, consistent in zip(
            out["endpoint"],
            out["curation_source_class"],
            out["citation_kind"],
            out["has_uniprot"],
            out["chain_fields_consistent"],
        )
    ]

    out["pair_key"] = [
        _hash16(str(identity), str(seq))
        for identity, seq in zip(out["identity_key"], out["sequence_sha256"])
    ]
    return out


def deduplicate(frame: pd.DataFrame, report: QcReport) -> pd.DataFrame:
    """Collapse re-imports of one measurement; keep independent experiments.

    The duplicate key includes the **byte-exact value cell**, not a numeric
    tolerance. Two records that are the same measurement re-imported carry the
    identical cell; two genuinely independent measurements that happen to agree
    to three significant figures are rare, and collapsing those would understate
    ``n`` rather than overstate certainty -- the conservative direction.

    Identical values under *different* citations are kept as separate rows and
    flagged: collapsing them would merge a primary paper with a review quoting
    it, while keeping them unflagged would fabricate a replicate.
    """
    # Keyed on the comparability group rather than on (target, endpoint,
    # citation) separately. The group already carries all of those *plus* the
    # condition bucket and the source class, which matters: two records from one
    # paper reporting the same number at different pH are not one measurement
    # re-imported, and a key without the conditions would silently merge them.
    key_columns = [
        "identity_key",
        "comparability_group_id",
        "raw_value_cell",
        "relation_original",
    ]
    frame = frame.sort_values("source_line", kind="stable").reset_index(drop=True)
    grouped = frame.groupby(key_columns, dropna=False, sort=False)

    frame["n_source_records"] = grouped["observation_id"].transform("size")
    rank = grouped.cumcount()
    survivors = rank == 0

    report.n_input = len(frame)
    report.n_duplicate_imports = int((~survivors).sum())
    multiplicity = Counter(frame.loc[survivors, "n_source_records"].tolist())
    report.duplicate_multiplicity = {str(k): int(v) for k, v in sorted(multiplicity.items())}

    duplicates = frame.loc[~survivors].copy()
    duplicates["qc_status"] = QcStatus.DUPLICATE_IMPORT
    first_ids = grouped["observation_id"].transform("first")
    duplicates["duplicate_of"] = first_ids.loc[~survivors].to_numpy()

    kept = frame.loc[survivors].copy()
    kept["duplicate_of"] = None
    report.n_after_dedup = len(kept)

    # Cross-source identical value under a different citation.
    cross_key = ["identity_key", "sequence_sha256", "endpoint", "raw_value_cell"]
    distinct_citations = kept.groupby(cross_key, dropna=False)["citation_key"].transform("nunique")
    kept["cross_source_identical_value"] = distinct_citations > 1
    report.n_cross_source_identical = int(kept["cross_source_identical_value"].sum())

    return kept, duplicates


def summarize_replicates(
    frame: pd.DataFrame,
    report: QcReport,
    *,
    profile: CurationProfile | None = None,
) -> pd.DataFrame:
    """Aggregate exact replicates within a comparability group.

    Exact values only. A censored bound is never pooled into a median -- it is
    not a measurement of the same kind -- so bounds are summarized separately as
    the tightest lower and loosest upper bound in the group.
    """
    profile = profile or get_profile()

    group_columns = ["identity_key", "sequence_sha256", "endpoint", "comparability_group_id"]
    frame["replicate_group_id"] = [
        _hash16(str(a), str(b), str(c), str(d))
        for a, b, c, d in zip(*(frame[column] for column in group_columns))
    ]

    exact = frame[frame["is_exact"].fillna(False)]
    if len(exact):
        stats = exact.groupby("replicate_group_id")["p_value"].agg(
            replicate_n_exact="size",
            replicate_p_median="median",
            replicate_p_min="min",
            replicate_p_max="max",
        )
        stats["replicate_p_range"] = stats["replicate_p_max"] - stats["replicate_p_min"]
        mad = exact.groupby("replicate_group_id")["p_value"].apply(
            lambda series: float(np.median(np.abs(series - np.median(series))))
        )
        stats["replicate_p_mad"] = mad
    else:
        stats = pd.DataFrame(
            columns=[
                "replicate_n_exact",
                "replicate_p_median",
                "replicate_p_min",
                "replicate_p_max",
                "replicate_p_range",
                "replicate_p_mad",
            ]
        )

    censored = frame[frame["is_censored"].fillna(False)]
    if len(censored):
        bound_stats = censored.groupby("replicate_group_id").agg(
            replicate_n_censored=("observation_id", "size"),
            replicate_tightest_lower=("lower_bound", "max"),
            replicate_loosest_upper=("upper_bound", "min"),
        )
    else:
        bound_stats = pd.DataFrame(
            columns=[
                "replicate_n_censored",
                "replicate_tightest_lower",
                "replicate_loosest_upper",
            ]
        )

    frame = frame.merge(stats, left_on="replicate_group_id", right_index=True, how="left")
    frame = frame.merge(bound_stats, left_on="replicate_group_id", right_index=True, how="left")
    frame["replicate_n_exact"] = frame["replicate_n_exact"].fillna(0).astype("int32")
    frame["replicate_n_censored"] = frame["replicate_n_censored"].fillna(0).astype("int32")

    # -- discordance, against the MEASURED repeatability limit --------------
    thresholds: dict[str, tuple[float, float]] = {}
    for endpoint in frame["endpoint"].unique():
        try:
            thresholds[endpoint] = profile.discordance_thresholds(endpoint)
        except KeyError:
            thresholds[endpoint] = (float("inf"), float("inf"))
    report.thresholds_used = {
        endpoint: {"moderate": round(lo, 3), "high": round(hi, 3)}
        for endpoint, (lo, hi) in thresholds.items()
    }

    moderate = frame["endpoint"].map({k: v[0] for k, v in thresholds.items()})
    high = frame["endpoint"].map({k: v[1] for k, v in thresholds.items()})
    spread = frame["replicate_p_range"]
    frame["replicate_discordance"] = np.select(
        [spread.isna() | (frame["replicate_n_exact"] <= 1), spread >= high, spread >= moderate],
        ["none", "high", "moderate"],
        default="none",
    )

    # -- censoring consistency ---------------------------------------------
    # An exact value that violates a bound reported for the same group means the
    # group is internally contradictory; flagged, never silently reconciled.
    lower = frame["replicate_tightest_lower"]
    upper = frame["replicate_loosest_upper"]
    exact_value = frame["p_value"].where(frame["is_exact"].fillna(False))
    # Compare only where both sides are present. Letting NaN into the comparison
    # works but emits a RuntimeWarning per column, which buries real warnings.
    comparable_lower = lower.notna() & exact_value.notna()
    comparable_upper = upper.notna() & exact_value.notna()
    violates = (comparable_lower & (exact_value.fillna(0) < lower.fillna(-np.inf))) | (
        comparable_upper & (exact_value.fillna(0) > upper.fillna(np.inf))
    )
    frame["censoring_consistent"] = ~violates.fillna(False)
    report.n_censoring_inconsistent = int((~frame["censoring_consistent"]).sum())

    group_sizes = frame.groupby("comparability_group_id")["observation_id"].transform("size")
    frame["comparability_group_size"] = group_sizes.astype("int32")

    report.replicate_groups = int(frame["replicate_group_id"].nunique())
    report.replicate_groups_multi = int(
        frame.loc[frame["replicate_n_exact"] > 1, "replicate_group_id"].nunique()
    )
    report.discordance_counts = {
        str(k): int(v) for k, v in frame["replicate_discordance"].value_counts().items()
    }

    multi = frame.loc[frame["replicate_n_exact"] > 1, "replicate_p_range"].dropna()
    if len(multi):
        report.spread_quantiles = {
            "median": round(float(multi.median()), 3),
            "mean": round(float(multi.mean()), 3),
            "p90": round(float(multi.quantile(0.90)), 3),
            "max": round(float(multi.max()), 3),
            "over_0.5": int((multi > 0.5).sum()),
            "over_1.0": int((multi > 1.0).sum()),
            "over_2.0": int((multi > 2.0).sum()),
        }
    return frame


def assign_eligibility(
    frame: pd.DataFrame,
    report: QcReport,
    *,
    profile: CurationProfile | None = None,
) -> pd.DataFrame:
    """Per-task eligibility columns, so the audit is a GROUP BY not a rerun."""
    profile = profile or get_profile()
    policy = profile.ic50
    comparability = profile.comparability

    is_ic50 = frame["endpoint"] == Endpoint.IC50.value
    tier_b = is_ic50 & (frame["ic50_tier"] == "B")

    usable = frame["qc_status"].isin([QcStatus.OK, QcStatus.AGGREGATED, QcStatus.AGGREGATED_DISCORDANT])

    frame["primary_regression_eligible"] = usable & ~tier_b
    if policy.tier_b_eligible_regression:
        frame["primary_regression_eligible"] = usable

    confidence_ok = frame["metadata_confidence"].map(
        lambda value: Confidence.at_least(value, comparability.ranking_min_confidence)
    )
    frame["ranking_eligible"] = (
        usable
        & confidence_ok
        & (frame["comparability_group_size"] >= comparability.ranking_min_group_size)
        & (tier_b.map({True: policy.tier_b_eligible_ranking, False: True}))
    )
    frame["retrieval_eligible"] = usable & (
        tier_b.map({True: policy.tier_b_eligible_retrieval, False: True})
    )
    frame["analog_pair_eligible"] = (
        usable
        & frame["is_exact"].fillna(False)
        & (frame["replicate_discordance"] != "high")
        & (tier_b.map({True: policy.tier_b_eligible_pairs, False: True}))
    )

    report.eligibility = {
        "primary_regression": int(frame["primary_regression_eligible"].sum()),
        "within_assay_ranking": int(frame["ranking_eligible"].sum()),
        "retrieval_supervision": int(frame["retrieval_eligible"].sum()),
        "analog_pair_training": int(frame["analog_pair_eligible"].sum()),
    }
    report.ic50_tiers = {str(k): int(v) for k, v in frame["ic50_tier"].value_counts().items()}
    report.citation_kinds = {
        str(k): int(v) for k, v in frame["citation_kind"].value_counts().items()
    }
    report.comparability_scopes = {
        str(k): int(v) for k, v in frame["comparability_scope"].value_counts().items()
    }
    report.confidence_counts = {
        str(k): int(v) for k, v in frame["metadata_confidence"].value_counts().items()
    }
    report.unknown_citation_observations = int((frame["citation_kind"] == "none").sum())

    sizes = frame.groupby("comparability_group_id")["observation_id"].size()
    report.singleton_groups = int((sizes == 1).sum())
    report.group_size_summary = {
        "n_groups": int(len(sizes)),
        "median": float(sizes.median()),
        "p90": float(sizes.quantile(0.90)),
        "max": int(sizes.max()),
        "singletons": report.singleton_groups,
    }
    report.endpoint_exact_censored = {
        endpoint: {
            "exact": int(((frame["endpoint"] == endpoint) & frame["is_exact"].fillna(False)).sum()),
            "censored": int(
                ((frame["endpoint"] == endpoint) & frame["is_censored"].fillna(False)).sum()
            ),
        }
        for endpoint in sorted(frame["endpoint"].unique())
    }
    return frame


#: Shards of the published observation table, keyed on the target hash so that
#: cold-target work touches whole shards and target-conditioned retrieval reads
#: contiguously.
PUBLISHED_SHARDS = 64


def _shard_of(sequence_sha256: str, shards: int = PUBLISHED_SHARDS) -> int:
    return int(sequence_sha256[:8], 16) % shards


def curate_observations(
    processed_dir: Path | str,
    *,
    profile: CurationProfile | None = None,
    limit_shards: int | None = None,
) -> tuple[Any, Any, QcReport]:
    """Run the whole QC pass and publish the curated observation table.

    Chains annotate -> deduplicate -> summarize replicates -> assign eligibility,
    then writes ``observations/`` sharded on the target hash plus
    ``duplicates.parquet`` and ``qc_report.json``.
    """
    import time

    from bindra.data.tables import ShardWriter, write_json, write_table

    processed_dir = Path(processed_dir)
    profile = profile or get_profile()
    sources = load_source_classes()
    started = time.perf_counter()
    report = QcReport()

    ligands = pd.read_parquet(
        processed_dir / "ligands",
        columns=["ligand_ref", "identity_key", "leakage_key", "scaffold_key", "inchikey"],
    )
    constructs = pd.read_parquet(
        processed_dir / "target_constructs.parquet",
        columns=["sequence_ref", "sequence_sha256", "construct_id", "sequence_length", "status"],
    )
    # A construct that failed validation cannot anchor a supervised observation.
    constructs = constructs[constructs["status"] == "OK"].drop(columns=["status"])

    shard_paths = sorted((processed_dir / "raw_observations").glob("*.parquet"))
    if limit_shards is not None:
        shard_paths = shard_paths[:limit_shards]
    frame = pd.concat(
        [pd.read_parquet(path, columns=list(QC_COLUMNS)) for path in shard_paths],
        ignore_index=True,
    )

    frame = annotate_observations(
        frame, ligands=ligands, constructs=constructs, profile=profile, sources=sources
    )
    # An observation whose ligand failed standardization or whose construct failed
    # validation has no usable key, so it cannot be curated further.
    unusable = frame["identity_key"].isna() | frame["sequence_sha256"].isna()
    report.n_unusable_after_join = int(unusable.sum())
    frame = frame[~unusable].copy()

    frame, duplicates = deduplicate(frame, report)
    frame = summarize_replicates(frame, report, profile=profile)

    frame["qc_status"] = np.select(
        [
            frame["replicate_discordance"] == "high",
            frame["replicate_n_exact"] > 1,
        ],
        [QcStatus.AGGREGATED_DISCORDANT, QcStatus.AGGREGATED],
        default=QcStatus.OK,
    )
    frame = assign_eligibility(frame, report, profile=profile)

    frame["shard"] = [_shard_of(str(value)) for value in frame["sequence_sha256"]]
    frame = frame.sort_values(
        ["shard", "sequence_sha256", "identity_key", "endpoint"], kind="stable"
    ).reset_index(drop=True)

    out_dir = processed_dir / "observations"
    with ShardWriter(out_dir, prefix="obs", rows_per_shard=10**12) as writer:
        # One file per target-hash bucket: deterministic, and a cold-target split
        # then reads whole files instead of filtering every shard.
        for shard_id, group in frame.groupby("shard", sort=True):
            writer.extend(group.drop(columns=["shard"]).to_dict("records"))
            writer.flush()
    observations = writer.close()

    duplicates_ref = write_table(duplicates, processed_dir / "duplicates")
    report.elapsed_seconds = time.perf_counter() - started

    write_json(
        {
            "report": report.to_dict(),
            "profile": {"name": profile.name, "hash": profile.profile_hash},
            "source_classes_sha256": sources.sha256,
            "artifacts": {
                "observations": observations.to_dict(),
                "duplicates": duplicates_ref.to_dict(),
            },
            "caveats": [
                "There is no assay identifier in BindingDB. comparability_group_id "
                "is an explicitly named PROXY; comparability_scope records which.",
                "A record with no citation of any kind forms its own singleton "
                "group, so it cannot enter within-assay ranking.",
                "Discordance is judged against the measured repeatability limit "
                "(2.77 * sigma_rep, endpoint-specific), not a fixed threshold.",
                "Censored bounds are never pooled into a replicate median.",
            ],
        },
        processed_dir / "qc_report.json",
    )
    return observations, duplicates_ref, report


__all__ = [
    "CITATION_PRIORITY",
    "PUBLISHED_SHARDS",
    "QC_COLUMNS",
    "curate_observations",
    "ComparabilityScope",
    "Confidence",
    "QcFlag",
    "QcReport",
    "QcStatus",
    "annotate_observations",
    "assay_condition_bucket",
    "assign_eligibility",
    "citation_key",
    "comparability_group_id",
    "comparability_scope",
    "deduplicate",
    "ic50_tier",
    "metadata_confidence",
    "normalize_doi",
    "normalize_pmid",
    "summarize_replicates",
]
