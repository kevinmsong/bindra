"""Curation and split policy, as data, with content hashing.

Every policy choice in the data layer lives here as a frozen dataclass field
rather than as a literal buried in the code that applies it. Two reasons, both
from the specification:

* *"Deliver a curation report with counts after every filter"* and record the
  normalization policy. A report can only state the policy it ran under if the
  policy is a value it can serialize.
* Checkpoint and manifest metadata must carry config hashes. ``config_hash``
  gives every artifact a short digest naming the exact policy that produced it,
  so an output can never be ambiguous about its own provenance.

Serialization is stdlib JSON with sorted keys, matching ``SnapshotProvenance``
and the rest of the project. There is no YAML anywhere on purpose.

**Thresholds here are measured, not conventional.** See :class:`ErrorModel`:
the replicate-noise estimate for this corpus drives the discordance threshold,
the activity-cliff magnitude and the ranking margin. The specification is
explicit that its starting 0.5 p-unit margin is *"a starting hyperparameter, not
a universal assay-error threshold"*, and on BindingDB 202610 that value turns
out to sit below the noise floor of every endpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from typing import Any, Mapping

from bindra.data.endpoints import Endpoint

#: Bumped when the meaning of a policy field changes in a way that should
#: invalidate artifacts produced under the old meaning.
POLICY_SCHEMA_VERSION = "curation-policy-v1"


def canonical_json(payload: Any) -> str:
    """Deterministic JSON: sorted keys, no incidental whitespace.

    Hashes are only comparable across runs if the serialization is canonical,
    so every hash in this module goes through here.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(payload: Any) -> str:
    """Short content digest of a policy object or plain payload."""
    if hasattr(payload, "to_dict"):
        payload = payload.to_dict()
    elif _is_policy(payload):
        payload = asdict(payload)
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:24]


def _is_policy(obj: Any) -> bool:
    return hasattr(obj, "__dataclass_fields__")


# ---------------------------------------------------------------------------
# Measured properties of the corpus
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ErrorModel:
    """Replicate-noise estimates, measured on this snapshot.

    ``sigma_rep`` is the pooled within-group standard deviation of exact
    p-values over **within-publication** groups
    ``(ligand, target, endpoint, citation)`` with at least two members --
    repeated reports of nominally the same quantity. Measured on BindingDB
    202610 over 207,317 such groups.

    Three honesty constraints travel with these numbers and are repeated in the
    curation report:

    1. This is a **within-publication** estimate. Between-laboratory
       reproducibility is strictly larger and is *not* estimated here, so
       ``sigma_rep`` is a lower bound on true measurement error.
    2. It is therefore also a lower bound on the **irreducible RMSE** for
       predicting a single measurement. A model reporting pKd RMSE near 0.95 is
       at the noise floor, not skilful.
    3. Patent-derived ``sigma_rep`` is *lower* than literature-derived
       (Kd 0.707 vs 1.019). That is not better data -- it indicates
       re-statement of one table value rather than independent measurement, so
       patent "replicates" are less independent, not more precise.
    """

    #: endpoint -> pooled within-publication SD, p-units
    sigma_rep: Mapping[str, float] = field(
        default_factory=lambda: {
            Endpoint.KD.value: 0.945,
            Endpoint.KI.value: 0.641,
            Endpoint.IC50.value: 0.671,
        }
    )
    #: endpoint -> {tier: sigma}, tiers are "literature" and "patent"
    sigma_rep_by_tier: Mapping[str, Mapping[str, float]] = field(
        default_factory=lambda: {
            Endpoint.KD.value: {"literature": 1.019, "patent": 0.707},
            Endpoint.KI.value: {"literature": 0.655, "patent": 0.605},
            Endpoint.IC50.value: {"literature": 0.713, "patent": 0.616},
        }
    )
    #: endpoint -> within-group degrees of freedom behind the estimate
    degrees_of_freedom: Mapping[str, int] = field(
        default_factory=lambda: {
            Endpoint.KD.value: 13_427,
            Endpoint.KI.value: 57_841,
            Endpoint.IC50.value: 203_756,
        }
    )
    measured_on_release: str = "202610"
    measured_at: str = "2026-10-02"
    estimator: str = (
        "pooled within-group SD of exact p-values over "
        "(ligand, target, endpoint, citation) groups with n>=2"
    )

    #: ISO 5725 repeatability limit for n=2: the difference exceeded by only 5%
    #: of true replicate pairs. 2*sqrt(2)*1.96/2 ~= 2.77.
    REPEATABILITY_FACTOR: float = 2.77

    def sigma(self, endpoint: Endpoint | str) -> float:
        key = endpoint.value if isinstance(endpoint, Endpoint) else str(endpoint)
        try:
            return float(self.sigma_rep[key])
        except KeyError as exc:
            raise KeyError(
                f"no sigma_rep for endpoint {key!r}; known: {sorted(self.sigma_rep)}"
            ) from exc

    def repeatability_limit(self, endpoint: Endpoint | str) -> float:
        """Spread beyond which a replicate group is inconsistent, not noisy."""
        return self.REPEATABILITY_FACTOR * self.sigma(endpoint)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sigma_rep": dict(self.sigma_rep),
            "sigma_rep_by_tier": {k: dict(v) for k, v in self.sigma_rep_by_tier.items()},
            "degrees_of_freedom": dict(self.degrees_of_freedom),
            "measured_on_release": self.measured_on_release,
            "measured_at": self.measured_at,
            "estimator": self.estimator,
            "repeatability_factor": self.REPEATABILITY_FACTOR,
            "caveat": (
                "within-publication estimate; a LOWER BOUND on true measurement "
                "error and on irreducible RMSE. Between-laboratory reproducibility "
                "is larger and is not estimated. Lower patent sigma reflects "
                "re-statement, not precision."
            ),
        }


@dataclass(frozen=True)
class DependenceStructure:
    """How far the corpus is from having independent observations.

    Kish effective sample size ``(sum w)^2 / sum w^2`` over per-entity
    observation counts. Measured on BindingDB 202610 single-chain observations.

    The headline "11k targets" corresponds to **572 effective independent
    target units**, and the top 1% of targets carry 34.2% of all observations.
    Confidence intervals must therefore come from a cluster bootstrap over
    protein components, and per-target macro averages must be reported beside
    pooled metrics -- pooled metrics here are dominated by ~104 targets.
    """

    n_targets_observed: int = 10_474
    n_targets_effective: int = 572
    target_top1pct_share: float = 0.342
    target_max_observations: int = 22_999
    n_ligands_observed: int = 1_182_431
    n_ligands_effective: int = 73_945
    ligand_top1pct_share: float = 0.137
    ligand_max_observations: int = 3_094
    measured_on_release: str = "202610"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StandardizationPolicy:
    """Deterministic RDKit pipeline, pinned step by step.

    The order is a policy, not an implementation detail: ``Normalizer`` before
    ``Reionizer`` before stereo assignment produces different keys than any
    other order, so it is recorded rather than left to whatever
    ``rdMolStandardize.Cleanup`` happens to do in the installed version.
    """

    steps: tuple[str, ...] = (
        "MolFromSmiles(sanitize=False)",
        "UpdatePropertyCache(strict=False)",
        "SanitizeMol",
        "tag_atoms(bindra_orig_idx)",
        "MetalDisconnector",
        "Normalizer",
        "Reionizer",
        "AssignStereochemistry(cleanIt=True,force=True)",
        "FragmentRemover(salt_definitions)",
        "quarantine_if_multiple_organic_fragments",
        "MolToSmiles(isomericSmiles=True,canonical=True)",
    )
    parent_steps: tuple[str, ...] = (
        "Uncharger",
        "IsotopeParent",
        "TautomerEnumerator.Canonicalize",
        "RemoveStereochemistry",
    )

    #: Formal charges are PRESERVED on the identity path. The specification says
    #: "preserve specified stereochemistry, formal charges"; neutralizing would
    #: merge a measured anion with a measured free acid.
    uncharge_identity: bool = False

    #: More than one organic fragment left after salt removal is QUARANTINED,
    #: never resolved by picking the largest. Silently choosing the largest
    #: fragment is the coercion the specification forbids.
    largest_fragment_fallback: bool = False

    #: Off by default, and the reason is measured rather than assumed.
    #:
    #: ``TautomerEnumerator.Canonicalize`` costs **96.5 ms/mol** on this corpus,
    #: against ~4 ms/mol for every other step combined -- 99.5% of the runtime,
    #: or 2.8 hours across 14 cores for 1.44M molecules versus 7 minutes.
    #:
    #: What it buys is small, because InChI's own mobile-hydrogen layer already
    #: collapses the common tautomer classes. Measured on the skeleton block:
    #: 2-hydroxypyridine/2-pyridone collapse, adenine tautomers collapse,
    #: keto-enol (enol vs diketone) does **not**.
    #:
    #: The residual gap is further covered downstream: a cold-drug group is the
    #: union-find over shared leakage key *or* shared Murcko scaffold, and
    #: keto-enol tautomers of a ring-containing molecule share their scaffold, so
    #: they land in one drug group regardless. Acyclic molecules (~0.5% of the
    #: corpus) are the uncovered case, and the curation report says so.
    #:
    #: Turning it on is supported and costs 2.8 hours. Lowering the transform cap
    #: to make it affordable is NOT supported: a truncated canonicalization is not
    #: a canonical form, so the same molecule written two ways could receive two
    #: different keys -- strictly worse than not canonicalizing at all.
    canonicalize_tautomers: bool = False
    tautomer_max_heavy_atoms: int = 100
    tautomer_max_transforms: int = 1_000

    salt_definitions_sha256: str | None = None
    rdkit_version: str | None = None
    scaffold_include_chirality: bool = False
    #: Ring-free molecules get their own leakage key as their scaffold group,
    #: never a shared "ACYCLIC" bucket that would fuse unrelated chemistry into
    #: one enormous drug group.
    acyclic_scaffold_fallback: str = "leakage_key"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProteinPolicy:
    """Construct handling and the homology-grouping definition."""

    #: The 20 standard residues plus ambiguity codes that appear in BindingDB.
    allowed_alphabet: str = "ACDEFGHIKLMNPQRSTVWYXBZUO"
    #: Outside this range is a FLAG, not a rejection: the measured maximum
    #: construct length in release 202610 is 34,350 residues.
    flag_length_below: int = 20
    flag_length_above: int = 5_000

    #: The modelling identity of a target is the construct hash, never the
    #: UniProt accession. The specification forbids replacing every construct
    #: with the canonical UniProt sequence.
    identity_from: str = "construct_sequence_sha256"

    # -- homology grouping ------------------------------------------------
    #: The conservative threshold. Used for the leakage AUDIT and for the
    #: ``cold_target_strict`` regime, and it is what any claim about novel targets
    #: should be bounded by.
    identity_threshold: float = 0.40
    coverage_threshold: float = 0.80

    #: The threshold the usable ``cold_target`` regime groups on, and why it
    #: differs. Measured on release 202610, single linkage at 40% identity puts
    #: 10,066 of 10,820 constructs (93.0%) into one component covering 98.4% of
    #: observations -- average degree is ~11.6, far above the percolation point,
    #: so the target graph is one connected mass and cold-target evaluation has
    #: 1.6% of the data to draw from.
    #:
    #: The graph shatters sharply between 0.50 and 0.60:
    #:
    #:   identity  components  largest  largest %  observations outside
    #:     0.40        666      10,066    93.0%          44,683
    #:     0.50      3,968       4,422    40.9%       1,138,822
    #:     0.60      5,764         241     2.2%       2,653,974
    #:     0.70      6,319         155     1.4%       2,680,629
    #:
    #: 0.60 is chosen because the transition is sharp rather than because it
    #: flatters a number: anything from 0.60 upward gives essentially the same
    #: structure. The cost is explicit and must be reported -- a test target may
    #: have a 40-60% identical homologue in training -- so both regimes are built,
    #: the strict one bounds the claim and this one powers the measurement.
    cold_target_identity_threshold: float = 0.60

    #: **Semi-global, chosen by measurement, not convention.** Using same-UniProt
    #: construct pairs as positives and cross-UniProt pairs as presumed
    #: negatives, measured on this corpus:
    #:
    #:   mode          sensitivity   false-positive
    #:   global (NW)      0.867          0.043
    #:   semi-global      0.987          0.000
    #:   local (SW)       0.980          0.000
    #:
    #: Global alignment is worse on *both* axes. It forces end-to-end alignment,
    #: which drives background identity for unrelated proteins up to a median of
    #: 0.252 -- only just below the 0.40 threshold -- and pins coverage near 1.0
    #: so the coverage criterion filters nothing. The consequence is not subtle:
    #: on a 500-construct sample, global alignment produced 4,825 qualifying
    #: edges and **one single component containing all 500 sequences**, which
    #: makes cold-target evaluation impossible. Semi-global produced 134 edges,
    #: 394 components, largest 54 (10.8%).
    #:
    #: Free end gaps are what make coverage meaningful: a domain construct inside
    #: a long chain aligns over the short sequence without being charged for the
    #: overhang, while an unrelated pair scores a short traced region (median
    #: coverage 0.073) and fails the 80% criterion as intended.
    alignment_mode: str = "semi_global"
    substitution_matrix: str = "BLOSUM62"
    open_gap_score: float = -11.0
    extend_gap_score: float = -1.0

    #: identity = n_identical / traced alignment length. Internal gap columns are
    #: INCLUDED in the denominator; end gaps are not, because they are free.
    #: Pinned because changing the denominator changes which pairs qualify.
    identity_denominator: str = "traced_alignment_length_internal_gaps_included"
    coverage_denominator: str = "length_of_shorter_sequence"

    #: Optimal alignments are rarely unique. Biopython reports 2,695,680
    #: co-optimal alignments for one real pair here, and different co-optimal
    #: alignments give different identity counts, so "identity on the optimal
    #: alignment" is not a well-defined quantity. Measured engine-to-engine
    #: disagreement under semi-global on near-threshold pairs: median 0.0000,
    #: p90 0.0000, max 0.0803.
    #:
    #: Set to **zero**, which took a measurement to justify. A blanket margin
    #: looks like the conservative choice, but single linkage amplifies it: a
    #: margin of 0.08 drops the effective threshold to 0.32, which sits inside
    #: the negative-pair identity distribution (p90 = 0.385), and on a
    #: 500-construct sample it turned 134 qualifying edges into 473 and the
    #: largest component from 54 (10.8%) into 269 (53.8%). Trading a rare
    #: borderline miss for the loss of cold-target evaluation is a bad trade.
    #:
    #: Ambiguity is handled where it can be handled precisely instead: the
    #: engine and its version are pinned, the disagreement distribution is
    #: reported, and the split audit re-aligns every near-threshold
    #: cross-partition pair with the independent Biopython implementation and
    #: quarantines any that the oracle says qualifies. That fixes actual
    #: violations rather than over-merging the whole corpus against hypothetical
    #: ones.
    identity_ambiguity_margin: float = 0.0

    #: Identity band around the threshold that the split audit re-checks with the
    #: second engine. Covers the measured disagreement (max 0.0803) with room.
    audit_near_threshold_band: float = 0.10

    #: Constructs sharing a UniProt accession are the same protein and must never
    #: straddle a split, whatever an alignment says. This closes the measured
    #: 1.3% of same-UniProt construct pairs that semi-global alignment alone does
    #: not join (distinct isoforms, very short fragments).
    union_constructs_sharing_uniprot: bool = True

    #: Residues mapped to ``X`` before alignment only; the stored construct is
    #: never modified. BLOSUM62 covers the 20 standard residues plus B, Z, X and
    #: ``*``, but **not** selenocysteine (U) or pyrrolysine (O). Two aligners
    #: would then handle them by their own private fallback rules and could
    #: disagree, so the mapping is pinned here instead of left to the library.
    #: 4 constructs in release 202610 contain U or O.
    alignment_unknown_residues: tuple[str, ...] = ("U", "O")
    alignment_unknown_substitute: str = "X"
    linkage: str = "single"
    #: Edges weaker than this are not stored; they can never qualify.
    store_edges_above_identity: float = 0.20
    #: Exhaustive all-pairs alignment, no prefilter in the qualifying path. A
    #: k-mer or MinHash prefilter is NOT recall-safe at 40% identity, and the
    #: exhaustive computation is affordable here (~10.8k sequences).
    exhaustive: bool = True

    #: Within-UniProt mutation detection: alignment must be gapless and at
    #: least this identical for a point-mutant call.
    mutation_min_identity: float = 0.98

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComparabilityPolicy:
    """Assay-comparability proxy, named as a proxy.

    BindingDB 202610 has **no assay identifier and no assay description column
    anywhere in its 640 columns**. This is not a parsing gap; the field does not
    exist. Every "within-assay" quantity in BINDRA therefore rests on the
    components below, and the curation report must say so prominently.
    """

    key_components: tuple[str, ...] = (
        "sequence_sha256",
        "endpoint",
        "citation_key",
        "assay_condition_bucket",
        "curation_source_class",
    )
    citation_priority: tuple[str, ...] = ("pmid", "doi", "patent", "pubchem_aid")

    #: THE critical rule. With no citation of any kind, the group key falls back
    #: to the observation's own id, making it a singleton. The specification is
    #: explicit: "A missing assay identifier must not cause all records for that
    #: target to become one giant ranking group." A target with 10,000
    #: provenance-free observations must yield 10,000 groups of one.
    unknown_citation_is_singleton: bool = True

    #: pH and temperature are populated in only 1.8-7.4% of records (measured),
    #: so they cannot carry the key. They are bucketed, and MISSING IS ITS OWN
    #: BUCKET -- "pH?" never merges with "pH 7.4".
    include_conditions: bool = True
    ph_rounding: float = 0.5
    temperature_bins: tuple[float, ...] = (15.0, 25.0, 30.0, 37.0)
    missing_condition_is_own_bucket: bool = True

    #: Patents cap at "medium" regardless of other fields: one patent's examples
    #: table can hold thousands of IC50s under more than one protocol.
    patent_confidence_cap: str = "medium"

    #: Ranking supervision needs a group with something to rank and metadata
    #: good enough to believe the grouping.
    ranking_min_group_size: int = 2
    ranking_min_confidence: str = "medium"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DedupPolicy:
    """Duplicate imports, replicates, and discordance."""

    #: A re-import carries the byte-identical value cell. Matching on an exact
    #: string rather than a float tolerance under-states ``n`` in the rare case
    #: of two genuinely independent agreeing measurements, which is the
    #: conservative direction: it never over-states certainty.
    #: Keyed on the comparability group, which already carries target, endpoint,
    #: citation, condition bucket and source class. Including the conditions
    #: matters: two records from one paper reporting the same number at different
    #: pH are not one measurement re-imported.
    duplicate_key: tuple[str, ...] = (
        "identity_key",
        "comparability_group_id",
        "raw_value_cell",
        "relation_original",
    )
    duplicate_value_match: str = "byte_exact"
    #: Identical values under *different* citations are kept as separate
    #: observations and flagged. Collapsing would merge a primary paper with a
    #: review quoting it; keeping them silently would fabricate a replicate.
    keep_cross_source_identical: bool = True
    #: Replicate counts count distinct citations, not rows.
    replicate_n_counts: str = "distinct_citation_keys"

    robust_summary: str = "median"
    spread_measures: tuple[str, ...] = ("range", "mad", "iqr")
    #: Censored members are summarized separately and NEVER pooled into a
    #: median with exact values.
    aggregate_censored_with_exact: bool = False

    #: Discordance is judged against the measured repeatability limit rather
    #: than a fixed 0.5 p-units. ``moderate`` at 2*sigma, ``high`` at
    #: 2.77*sigma. See :class:`ErrorModel`.
    discordance_moderate_sigma_multiple: float = 2.0
    discordance_high_sigma_multiple: float = 2.77
    #: A group disagreeing beyond repeatability keeps its median but is barred
    #: from analog pairs and the cliff panel: a delta computed from it is
    #: meaningless. It is never dropped and never silently averaged.
    discordant_excluded_from_pairs: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Ic50TierPolicy:
    """Which IC50 records may enter the primary affinity regression.

    The specification retains Ki and Kd as high-confidence and asks for IC50 to
    be admitted only when the record supports assignment to the intended
    protein and an appropriate biochemical assay, with less-certain IC50 in a
    separately flagged tier.
    """

    tier_a_requires_single_chain: bool = True
    tier_a_requires_chain_consistency: bool = True
    tier_a_requires_uniprot: bool = True
    tier_a_requires_citation: bool = True
    tier_a_source_classes: tuple[str, ...] = ("literature_curated", "patent")

    #: Tier B is keyed on the curation SOURCE CLASS, never on PubChem AID
    #: presence. Measured: 68% of rows carry an AID, including ~545k whose
    #: provenance is a US patent, so an AID-based rule would misclassify most of
    #: the patent corpus.
    tier_b_by_aid_presence: bool = False

    #: Ki and Kd are high-confidence regardless of source, including patents.
    tier_applies_to: tuple[str, ...] = (Endpoint.IC50.value,)

    #: Tier B is curated, counted and stored, and may supervise the auxiliary
    #: objectives whose signal is relative order within a group -- but it never
    #: enters the primary affinity metric or matched-pair/delta construction.
    tier_b_eligible_regression: bool = False
    tier_b_eligible_ranking: bool = True
    tier_b_eligible_retrieval: bool = True
    tier_b_eligible_pairs: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SplitPolicy:
    """Partition construction and the leakage audit."""

    #: Declared constant, asserted by a test. The specification forbids choosing
    #: a split seed on model performance, so every seed ever used is also
    #: appended to data/splits/seed_log.jsonl -- making seed shopping visible in
    #: version control rather than invisible.
    seed: int = 20_261_001

    #: Four partitions, because calibration must be separate from tuning
    #: validation: "Reserve a calibration partition separate from both training
    #: and hyperparameter validation."
    fractions: Mapping[str, float] = field(
        default_factory=lambda: {"train": 0.70, "val": 0.10, "cal": 0.10, "test": 0.10}
    )

    #: Group sizes are extremely skewed (one protein component can carry a large
    #: share of observations), so groups are bin-packed by observation weight
    #: rather than shuffled, or the realized fractions miss badly.
    assignment: str = "largest_remainder_weighted_binpack"

    # -- double cold ------------------------------------------------------
    #: Independent protein and drug assignments, intersected. Grouping by a
    #: combined protein-x-drug cluster id is a different and wrong algorithm.
    double_cold_independent: bool = True
    #: A single 70/10/10/10 on both axes retains sum(p_i*q_i) ~= 0.52 and leaves
    #: the test partition ~1% of interactions (~3k observations), too small to
    #: measure anything. The cascade splits test off first, then re-splits the
    #: train block by a further independent assignment.
    double_cold_cascade: bool = True
    double_cold_level1: Mapping[str, float] = field(
        default_factory=lambda: {"train_block": 0.80, "test": 0.20}
    )
    double_cold_level2: Mapping[str, float] = field(
        default_factory=lambda: {"train": 0.80, "val": 0.10, "cal": 0.10}
    )

    # -- temporal ---------------------------------------------------------
    #: Evidence date, never the import date: "a recent database import of an old
    #: measurement is not new evidence."
    temporal_evidence_field: str = "date_published"
    temporal_cutoff: str = "2023-12-31"
    #: Nested so tuning and calibration also respect a cutoff. train <= 2022,
    #: val = 2023-H1, cal = 2023-H2, test >= 2024.
    temporal_nested: bool = True
    temporal_train_cutoff: str = "2022-12-31"
    temporal_val_window: tuple[str, str] = ("2023-01-01", "2023-06-30")
    temporal_cal_window: tuple[str, str] = ("2023-07-01", "2023-12-31")
    #: 2.7% of rows carry no parseable date; they are excluded from every
    #: temporal claim rather than assumed old or new.
    temporal_exclude_unknown_date: bool = True

    # -- audit ------------------------------------------------------------
    fingerprint: str = "morgan"
    fingerprint_radius: int = 2
    fingerprint_bits: int = 2048
    #: Folded screen width for the cross-partition nearest-neighbour audit.
    #: Folding ORs bits together, so |A and B| can only increase and |A or B|
    #: can only decrease: folded Tanimoto >= true Tanimoto. The screen therefore
    #: over-reports similarity, which is the safe direction for leakage.
    fingerprint_screen_bits: int = 512
    fingerprint_rescore_top_k: int = 200
    similarity_report_thresholds: tuple[float, ...] = (0.70, 0.80, 0.90)
    #: High chemical similarity across a cold-drug boundary is REPORTED, not a
    #: failure: different scaffolds genuinely can be similar, which is the
    #: specification's point in asking for the audit.
    quarantine_nn_above: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fractions"] = dict(self.fractions)
        payload["double_cold_level1"] = dict(self.double_cold_level1)
        payload["double_cold_level2"] = dict(self.double_cold_level2)
        return payload


@dataclass(frozen=True)
class BenchmarkPolicy:
    """Locked panel definitions (Upgrade B).

    Locked at the data stage, before any modelling, so a later result cannot be
    reframed by moving a threshold.
    """

    # -- activity cliff ---------------------------------------------------
    cliff_similarity_threshold: float = 0.90
    cliff_generic_scaffold_threshold: float = 0.80
    #: |delta p| for a cliff is max(1.0, 2*sigma_rep) per endpoint, so it is
    #: never looser than the 1.0 p-unit convention but also never inside the
    #: measured noise. A 1.0 p-unit pKd difference is only ~1.06 sigma.
    cliff_delta_sigma_multiple: float = 2.0
    cliff_delta_floor: float = 1.0
    cliff_noncliff_delta_max: float = 0.50
    #: Both members must be exact: a delta computed from a censored bound is not
    #: a measured difference.
    cliff_requires_exact_both: bool = True
    #: Guards against two noisy replicate sets masquerading as a cliff.
    cliff_min_signal_to_noise: float = 2.0
    cliff_comparability_tier: int = 1
    #: Both cliff members AND their chemical series are held out. The series
    #: unit is the cold-drug group, and the required groups are preassigned to
    #: test before the split is generated -- selecting pairs that happen to land
    #: in test afterwards would be post-hoc.
    cliff_series_unit: str = "cold_drug_group"
    cliff_preassign_to_test: bool = True

    # -- selectivity ------------------------------------------------------
    selectivity_related_identity_min: float = 0.25
    selectivity_related_coverage_min: float = 0.50
    selectivity_panel_tier: int = 1
    #: A missing countertarget measurement is UNKNOWN. Never a negative, never
    #: a decoy, never an inferred inactive.
    selectivity_missing_is_unknown: bool = True
    #: If either side is censored the selectivity is an interval, kept in a
    #: separate panel. "More than 100-fold selective" is a real datum and is
    #: not collapsed to a point.
    selectivity_censored_as_interval: bool = True
    selectivity_include_swapped_controls: bool = True
    curated_related_pairs_sha256: str | None = None

    # -- fishing ----------------------------------------------------------
    #: Measured weak/inactive compounds are the preferred control tier.
    #: Property-matched computational decoys are a separate, explicitly
    #: labelled tier of presumed controls with unknown binding status, and
    #: never receive a fabricated affinity label.
    fishing_decoys_are_presumed_controls: bool = True
    fishing_report_full_pool: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CurationProfile:
    """The complete data-layer policy for one named configuration."""

    name: str
    snapshot_release: str
    standardization: StandardizationPolicy = field(default_factory=StandardizationPolicy)
    proteins: ProteinPolicy = field(default_factory=ProteinPolicy)
    comparability: ComparabilityPolicy = field(default_factory=ComparabilityPolicy)
    dedup: DedupPolicy = field(default_factory=DedupPolicy)
    ic50: Ic50TierPolicy = field(default_factory=Ic50TierPolicy)
    splits: SplitPolicy = field(default_factory=SplitPolicy)
    benchmarks: BenchmarkPolicy = field(default_factory=BenchmarkPolicy)
    error_model: ErrorModel = field(default_factory=ErrorModel)
    dependence: DependenceStructure = field(default_factory=DependenceStructure)

    #: Off by default. The specification treats the plausibility window as a
    #: curation policy decision that must be recorded in the report when used.
    plausible_range: tuple[float, float] | None = None

    schema_version: str = POLICY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "snapshot_release": self.snapshot_release,
            "schema_version": self.schema_version,
            "plausible_range": list(self.plausible_range) if self.plausible_range else None,
            "standardization": self.standardization.to_dict(),
            "proteins": self.proteins.to_dict(),
            "comparability": self.comparability.to_dict(),
            "dedup": self.dedup.to_dict(),
            "ic50": self.ic50.to_dict(),
            "splits": self.splits.to_dict(),
            "benchmarks": self.benchmarks.to_dict(),
            "error_model": self.error_model.to_dict(),
            "dependence": self.dependence.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @property
    def profile_hash(self) -> str:
        return config_hash(self)

    # -- derived thresholds ------------------------------------------------
    #
    # These are the point of the module: a threshold is computed from the
    # measured error model rather than written down as a constant.

    def discordance_thresholds(self, endpoint: Endpoint | str) -> tuple[float, float]:
        """(moderate, high) replicate spread in p-units for this endpoint."""
        sigma = self.error_model.sigma(endpoint)
        return (
            self.dedup.discordance_moderate_sigma_multiple * sigma,
            self.dedup.discordance_high_sigma_multiple * sigma,
        )

    def cliff_delta_threshold(self, endpoint: Endpoint | str) -> float:
        """Minimum |delta p| for an activity cliff on this endpoint."""
        sigma = self.error_model.sigma(endpoint)
        return max(
            self.benchmarks.cliff_delta_floor,
            self.benchmarks.cliff_delta_sigma_multiple * sigma,
        )

    def ranking_margin(self, endpoint: Endpoint | str) -> float:
        """Smallest p-unit difference worth ranking on.

        The specification's starting value is 0.5 and it explicitly calls that
        "a starting hyperparameter, not a universal assay-error threshold". On
        this corpus 0.5 is below the noise floor of every endpoint, so the
        margin is one sigma_rep instead; ordering a pair that differs by less
        than measurement error is training on noise.
        """
        return self.error_model.sigma(endpoint)

    def with_overrides(self, **overrides: Any) -> "CurationProfile":
        return replace(self, **overrides)


#: Named profiles. ``default`` is the one this stage runs.
PROFILES: dict[str, CurationProfile] = {
    "default": CurationProfile(name="default", snapshot_release="202610"),
}


def get_profile(name: str = "default") -> CurationProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise KeyError(f"unknown profile {name!r}; known: {sorted(PROFILES)}") from exc


__all__ = [
    "POLICY_SCHEMA_VERSION",
    "PROFILES",
    "BenchmarkPolicy",
    "ComparabilityPolicy",
    "CurationProfile",
    "DedupPolicy",
    "DependenceStructure",
    "ErrorModel",
    "Ic50TierPolicy",
    "ProteinPolicy",
    "SplitPolicy",
    "StandardizationPolicy",
    "canonical_json",
    "config_hash",
    "get_profile",
]
