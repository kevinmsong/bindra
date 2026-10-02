"""QC: comparability proxies, dedup, replicate aggregation, tiering.

The single most important test in this file is
``test_provenance_free_observations_become_singleton_groups``. The specification
forbids a missing assay identifier collapsing a target into one giant ranking
group, and BindingDB has no assay identifier at all, so that property has to be
asserted rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bindra.config.profiles import ErrorModel, get_profile
from bindra.data.bindingdb.qc import (
    Confidence,
    QcStatus,
    annotate_observations,
    assay_condition_bucket,
    assign_eligibility,
    citation_key,
    comparability_group_id,
    deduplicate,
    ic50_tier,
    metadata_confidence,
    normalize_doi,
    normalize_pmid,
    summarize_replicates,
    QcReport,
)
from bindra.data.bindingdb.sources import (
    UnknownCurationSource,
    load_source_classes,
)

SEQ_HASH_A = "a" * 64
SEQ_HASH_B = "b" * 64


def _observation(
    observation_id: str,
    *,
    ligand_ref: int = 0,
    sequence_ref: int = 0,
    endpoint: str = "Ki",
    raw_value_cell: str = "100",
    p_value: float = 7.0,
    relation_original: str = "=",
    relation_p: str = "=",
    is_exact: bool = True,
    is_censored: bool = False,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    pmid: str | None = "12345678",
    article_doi: str | None = None,
    patent_number: str | None = None,
    pubchem_aid: str | None = None,
    curation_source: str = "ChEMBL",
    ph: str | None = None,
    temp_c: str | None = None,
    uniprot_primary: str | None = "P00001",
    source_line: int = 1,
) -> dict:
    return {
        "observation_id": observation_id,
        "source_line": source_line,
        "ligand_ref": ligand_ref,
        "sequence_ref": sequence_ref,
        "endpoint": endpoint,
        "raw_value_cell": raw_value_cell,
        "relation_original": relation_original,
        "relation_p": relation_p,
        "p_value": p_value,
        "is_exact": is_exact,
        "is_censored": is_censored,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "pmid": pmid,
        "article_doi": article_doi,
        "patent_number": patent_number,
        "pubchem_aid": pubchem_aid,
        "curation_source": curation_source,
        "institution": None,
        "ph": ph,
        "temp_c": temp_c,
        "uniprot_primary": uniprot_primary,
        "chain_fields_consistent": True,
        "declared_chain_count": 1,
        "date_published": "2020-01-01",
        "first_evidence_year": 2020,
    }


def _ligands(n: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ligand_ref": list(range(n)),
            "identity_key": [f"ID{i}" for i in range(n)],
            "leakage_key": [f"LEAK{i}" for i in range(n)],
            "scaffold_key": [f"SCAF{i}" for i in range(n)],
            "inchikey": [f"KEY{i}-X-N" for i in range(n)],
        }
    )


def _constructs(n: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_ref": list(range(n)),
            "sequence_sha256": [SEQ_HASH_A, SEQ_HASH_B][:n],
            "construct_id": [f"P0000{i}:abcd" for i in range(n)],
            "sequence_length": [300, 400][:n],
        }
    )


def _pipeline(records: list[dict], profile=None):
    profile = profile or get_profile()
    report = QcReport()
    frame = annotate_observations(
        pd.DataFrame(records),
        ligands=_ligands(),
        constructs=_constructs(),
        profile=profile,
    )
    frame, duplicates = deduplicate(frame, report)
    frame = summarize_replicates(frame, report, profile=profile)
    frame["qc_status"] = np.select(
        [frame["replicate_discordance"] == "high", frame["replicate_n_exact"] > 1],
        [QcStatus.AGGREGATED_DISCORDANT, QcStatus.AGGREGATED],
        default=QcStatus.OK,
    )
    frame = assign_eligibility(frame, report, profile=profile)
    return frame, duplicates, report


class TestCitationKeys:
    def test_provenance_free_observations_become_singleton_groups(self):
        """The specification's hard requirement, asserted.

        A target with many provenance-free observations must yield many groups of
        one, not one group of many -- otherwise within-assay ranking would
        silently compare measurements from unrelated experiments.
        """
        records = [
            _observation(
                f"BDB-{i}-KI",
                pmid=None,
                article_doi=None,
                patent_number=None,
                pubchem_aid=None,
                p_value=5.0 + i * 0.01,
                raw_value_cell=str(100 + i),
                source_line=i,
            )
            for i in range(200)
        ]
        frame, _, report = _pipeline(records)
        # Every observation is its own comparability group.
        assert frame["comparability_group_id"].nunique() == 200
        assert report.singleton_groups == 200
        assert (frame["comparability_group_size"] == 1).all()
        # And so none of them can enter ranking supervision.
        assert report.eligibility["within_assay_ranking"] == 0
        assert (frame["comparability_scope"] == "unknown_singleton").all()

    def test_priority_order(self):
        assert citation_key(observation_id="X", pmid="1", doi="d", patent="p")[1] == "pmid"
        assert citation_key(observation_id="X", doi="d", patent="p")[1] == "doi"
        assert citation_key(observation_id="X", patent="p", pubchem_aid="7")[1] == "patent"
        assert citation_key(observation_id="X", pubchem_aid="7")[1] == "pubchem_aid"
        assert citation_key(observation_id="X")[1] == "none"

    def test_doi_normalization_makes_one_paper_one_key(self):
        variants = [
            "10.1021/jm123",
            "https://doi.org/10.1021/JM123",
            " doi:10.1021/jm123 ",
            "https://dx.doi.org/10.1021/Jm123",
        ]
        keys = {citation_key(observation_id="X", doi=v)[0] for v in variants}
        assert len(keys) == 1

    def test_pmid_normalization_strips_noise(self):
        assert normalize_pmid(" 12345678 ") == "12345678"
        assert normalize_pmid("PMID: 12345678") == "12345678"
        assert normalize_pmid("") is None
        assert normalize_doi(None) is None


class TestComparabilityGrouping:
    def test_same_pair_in_two_papers_forms_two_groups_and_is_never_averaged(self):
        """Two papers are two experiments; averaging them invents certainty."""
        records = [
            _observation("BDB-1-KI", pmid="111", p_value=6.0, raw_value_cell="1000"),
            _observation("BDB-2-KI", pmid="222", p_value=8.0, raw_value_cell="10", source_line=2),
        ]
        frame, _, _ = _pipeline(records)
        assert frame["comparability_group_id"].nunique() == 2
        assert frame["replicate_group_id"].nunique() == 2
        # Each group keeps its own value; no 7.0 average appears anywhere.
        assert sorted(frame["replicate_p_median"]) == [6.0, 8.0]

    def test_two_values_in_one_paper_aggregate_with_the_spread_recorded(self):
        records = [
            _observation("BDB-1-KI", p_value=7.0, raw_value_cell="100"),
            _observation("BDB-2-KI", p_value=7.4, raw_value_cell="40", source_line=2),
        ]
        frame, _, _ = _pipeline(records)
        assert frame["replicate_group_id"].nunique() == 1
        assert set(frame["replicate_n_exact"]) == {2}
        assert frame["replicate_p_median"].iloc[0] == pytest.approx(7.2)
        assert frame["replicate_p_range"].iloc[0] == pytest.approx(0.4)
        assert set(frame["qc_status"]) == {QcStatus.AGGREGATED}

    def test_different_conditions_split_a_group(self):
        records = [
            _observation("BDB-1-KI", ph="7.4", temp_c="25"),
            _observation("BDB-2-KI", ph="5.0", temp_c="25", source_line=2),
        ]
        frame, _, _ = _pipeline(records)
        assert frame["comparability_group_id"].nunique() == 2

    def test_missing_condition_does_not_merge_with_a_present_one(self):
        records = [
            _observation("BDB-1-KI", ph="7.4", temp_c="25"),
            _observation("BDB-2-KI", ph=None, temp_c="25", source_line=2),
        ]
        frame, _, _ = _pipeline(records)
        assert frame["comparability_group_id"].nunique() == 2

    def test_different_endpoints_never_share_a_group(self):
        records = [
            _observation("BDB-1-KI", endpoint="Ki"),
            _observation("BDB-2-KD", endpoint="Kd", source_line=2),
        ]
        frame, _, _ = _pipeline(records)
        assert frame["comparability_group_id"].nunique() == 2

    def test_group_id_is_deterministic(self):
        args = dict(
            sequence_sha256=SEQ_HASH_A,
            endpoint="Ki",
            citation="PMID:1",
            condition_bucket="pH?|T?",
            source_class="literature_curated",
        )
        assert comparability_group_id(**args) == comparability_group_id(**args)


class TestDeduplication:
    def test_byte_identical_reimport_collapses(self):
        record = _observation("BDB-1-KI")
        other = dict(record, observation_id="BDB-1b-KI", source_line=2)
        frame, duplicates, report = _pipeline([record, other])
        assert len(frame) == 1
        assert len(duplicates) == 1
        assert frame["n_source_records"].iloc[0] == 2
        assert report.n_duplicate_imports == 1
        assert duplicates["duplicate_of"].iloc[0] == "BDB-1-KI"
        assert duplicates["qc_status"].iloc[0] == QcStatus.DUPLICATE_IMPORT

    def test_different_values_from_one_paper_are_not_duplicates(self):
        records = [
            _observation("BDB-1-KI", p_value=7.0, raw_value_cell="100"),
            _observation("BDB-2-KI", p_value=7.4, raw_value_cell="40", source_line=2),
        ]
        frame, duplicates, report = _pipeline(records)
        assert len(frame) == 2
        assert len(duplicates) == 0
        assert report.n_duplicate_imports == 0

    def test_identical_value_under_different_citations_is_kept_and_flagged(self):
        """A review quoting a paper must not be collapsed into it, nor counted
        as an independent replicate."""
        records = [
            _observation("BDB-1-KI", pmid="111"),
            _observation("BDB-2-KI", pmid="222", source_line=2),
        ]
        frame, duplicates, report = _pipeline(records)
        assert len(frame) == 2
        assert len(duplicates) == 0
        assert frame["cross_source_identical_value"].all()
        assert report.n_cross_source_identical == 2

    def test_the_survivor_is_the_earliest_source_line(self):
        record = _observation("BDB-late-KI", source_line=9)
        other = dict(record, observation_id="BDB-early-KI", source_line=2)
        frame, _, _ = _pipeline([record, other])
        assert frame["observation_id"].iloc[0] == "BDB-early-KI"


class TestReplicatesAndCensoring:
    def test_a_censored_bound_never_enters_a_median(self):
        records = [
            _observation("BDB-1-KI", p_value=7.0, raw_value_cell="100"),
            _observation(
                "BDB-2-KI",
                p_value=4.0,
                raw_value_cell=">100000",
                relation_original=">",
                relation_p="<",
                is_exact=False,
                is_censored=True,
                upper_bound=4.0,
                source_line=2,
            ),
        ]
        frame, _, _ = _pipeline(records)
        # The median reflects only the exact member.
        assert set(frame["replicate_p_median"].dropna()) == {7.0}
        assert set(frame["replicate_n_exact"]) == {1}
        assert set(frame["replicate_n_censored"]) == {1}

    def test_an_exact_value_violating_a_group_bound_is_flagged(self):
        """pKi = 7 cannot coexist with pKi < 4 for the same group."""
        records = [
            _observation("BDB-1-KI", p_value=7.0, raw_value_cell="100"),
            _observation(
                "BDB-2-KI",
                p_value=4.0,
                raw_value_cell=">100000",
                relation_original=">",
                relation_p="<",
                is_exact=False,
                is_censored=True,
                upper_bound=4.0,
                source_line=2,
            ),
        ]
        frame, _, report = _pipeline(records)
        assert report.n_censoring_inconsistent >= 1
        exact_row = frame[frame["is_exact"]].iloc[0]
        assert bool(exact_row["censoring_consistent"]) is False

    def test_discordance_uses_the_endpoint_specific_repeatability_limit(self):
        """A 2.0 p-unit spread is discordant for Ki but not for Kd.

        sigma_rep is 0.641 for Ki and 0.945 for Kd, so the 2.77-sigma limits are
        1.78 and 2.62 respectively. A fixed threshold would get one of them wrong.
        """
        profile = get_profile()
        ki = _pipeline(
            [
                _observation("A", endpoint="Ki", p_value=6.0, raw_value_cell="1000"),
                _observation("B", endpoint="Ki", p_value=8.0, raw_value_cell="10", source_line=2),
            ],
            profile,
        )[0]
        kd = _pipeline(
            [
                _observation("A", endpoint="Kd", p_value=6.0, raw_value_cell="1000"),
                _observation("B", endpoint="Kd", p_value=8.0, raw_value_cell="10", source_line=2),
            ],
            profile,
        )[0]
        assert set(ki["replicate_discordance"]) == {"high"}
        assert set(kd["replicate_discordance"]) == {"moderate"}

    def test_a_discordant_group_keeps_its_median_but_is_barred_from_pairs(self):
        frame, _, report = _pipeline(
            [
                _observation("A", endpoint="Ki", p_value=5.0, raw_value_cell="10000"),
                _observation("B", endpoint="Ki", p_value=9.0, raw_value_cell="1", source_line=2),
            ]
        )
        assert set(frame["qc_status"]) == {QcStatus.AGGREGATED_DISCORDANT}
        assert frame["replicate_p_median"].iloc[0] == pytest.approx(7.0)
        # Never dropped...
        assert len(frame) == 2
        # ...but a delta computed from mutually inconsistent values is meaningless.
        assert report.eligibility["analog_pair_training"] == 0
        # Still usable for the affinity regression, with the flag carried.
        assert report.eligibility["primary_regression"] == 2

    def test_a_single_measurement_is_not_discordant(self):
        frame, _, _ = _pipeline([_observation("A")])
        assert frame["replicate_discordance"].iloc[0] == "none"
        assert frame["qc_status"].iloc[0] == QcStatus.OK


class TestTiering:
    def test_patent_ic50_with_a_pubchem_aid_is_tier_a(self):
        """The regression test for the AID mistake.

        68% of rows carry an AID, including ~545k whose provenance is a US
        patent. Keying Tier B on AID presence would have pushed most of the
        patent corpus out of the primary regression for the wrong reason.
        """
        frame, _, _ = _pipeline(
            [
                _observation(
                    "A",
                    endpoint="IC50",
                    curation_source="US Patent",
                    pmid=None,
                    patent_number="US1234567",
                    pubchem_aid="98765",
                )
            ]
        )
        assert frame["ic50_tier"].iloc[0] == "A"
        assert bool(frame["primary_regression_eligible"].iloc[0]) is True

    def test_pubchem_sourced_ic50_is_tier_b_and_barred_from_regression(self):
        frame, _, _ = _pipeline(
            [
                _observation(
                    "A",
                    endpoint="IC50",
                    curation_source="PubChem",
                    pmid=None,
                    pubchem_aid="98765",
                )
            ]
        )
        assert frame["ic50_tier"].iloc[0] == "B"
        assert bool(frame["primary_regression_eligible"].iloc[0]) is False

    def test_tier_b_may_still_supervise_retrieval(self):
        """Relative order within a group survives weaker assay provenance."""
        frame, _, _ = _pipeline(
            [
                _observation(
                    "A",
                    endpoint="IC50",
                    curation_source="PubChem",
                    pmid=None,
                    pubchem_aid="98765",
                )
            ]
        )
        assert bool(frame["retrieval_eligible"].iloc[0]) is True
        assert bool(frame["analog_pair_eligible"].iloc[0]) is False

    def test_ki_and_kd_are_never_tiered(self):
        frame, _, _ = _pipeline(
            [
                _observation("A", endpoint="Ki", curation_source="US Patent", pmid=None,
                             patent_number="US1"),
                _observation("B", endpoint="Kd", curation_source="PubChem", pmid=None,
                             pubchem_aid="7", source_line=2),
            ]
        )
        assert set(frame["ic50_tier"]) == {"na"}
        assert frame["primary_regression_eligible"].all()

    def test_ic50_without_uniprot_is_tier_b(self):
        frame, _, _ = _pipeline(
            [_observation("A", endpoint="IC50", uniprot_primary=None)]
        )
        assert frame["ic50_tier"].iloc[0] == "B"


class TestConfidence:
    def test_patent_is_capped_at_medium_even_when_complete(self):
        """One patent table can hold thousands of IC50s under several protocols."""
        sources = load_source_classes()
        assert (
            metadata_confidence(
                source_class="patent",
                citation_kind="patent",
                has_uniprot=True,
                has_ph=True,
                has_temperature=True,
                sources=sources,
            )
            == Confidence.MEDIUM
        )

    def test_high_requires_conditions(self):
        sources = load_source_classes()
        base = dict(
            source_class="literature_curated",
            citation_kind="pmid",
            has_uniprot=True,
            sources=sources,
        )
        assert metadata_confidence(has_ph=True, has_temperature=True, **base) == Confidence.HIGH
        assert metadata_confidence(has_ph=False, has_temperature=True, **base) == Confidence.MEDIUM

    def test_no_citation_or_no_target_is_low(self):
        sources = load_source_classes()
        assert (
            metadata_confidence(
                source_class="literature_curated",
                citation_kind="none",
                has_uniprot=True,
                has_ph=True,
                has_temperature=True,
                sources=sources,
            )
            == Confidence.LOW
        )
        assert (
            metadata_confidence(
                source_class="literature_curated",
                citation_kind="pmid",
                has_uniprot=False,
                has_ph=True,
                has_temperature=True,
                sources=sources,
            )
            == Confidence.LOW
        )

    def test_ranking_requires_at_least_medium_and_a_real_group(self):
        frame, _, report = _pipeline(
            [
                _observation("A", pmid="111", p_value=7.0, raw_value_cell="100"),
                _observation("B", pmid="111", p_value=8.0, raw_value_cell="10", source_line=2),
            ]
        )
        assert report.eligibility["within_assay_ranking"] == 2
        assert set(frame["metadata_confidence"]) == {Confidence.MEDIUM}


class TestSourceClassMap:
    def test_every_real_source_is_classified(self):
        sources = load_source_classes()
        for value in [
            "ChEMBL",
            "US Patent",
            "PubChem",
            "Curated from the literature by BindingDB",
            "PDSP Ki",
            "WIPO",
            "D3R",
            "CSAR",
            "Taylor Research Group, UCSD",
        ]:
            assert sources.class_of(value)

    def test_an_unmapped_source_raises_rather_than_defaulting(self):
        """A new BindingDB source must be classified deliberately."""
        sources = load_source_classes()
        with pytest.raises(UnknownCurationSource):
            sources.class_of("Some New Database 2027")

    def test_blank_source_is_unknown_not_an_error(self):
        assert load_source_classes().class_of("") == "unknown"


class TestConditionBuckets:
    def test_missing_is_its_own_bucket(self):
        assert assay_condition_bucket(None, None) == "pH?|T?"
        assert assay_condition_bucket(None, None) != assay_condition_bucket("7.4", "25")

    def test_temperature_ranges_are_parsed(self):
        assert assay_condition_bucket(None, "20-25") == assay_condition_bucket(None, "20")

    def test_unparseable_condition_falls_back_to_missing(self):
        assert assay_condition_bucket("n/a", "ambient") == "pH?|T?"
