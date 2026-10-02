"""Ingestion: row-to-observation expansion, ids, censoring, chain screening."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bindra.data.bindingdb.ingest import (
    Quarantine,
    ingest_snapshot,
    iter_snapshot_rows,
    observation_id,
)
from bindra.data.endpoints import Endpoint, Relation

from conftest import make_row, synthetic_header, write_synthetic_snapshot

SEQ_A = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
SEQ_B = "MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRGSEF"


def _ingest(tmp_path: Path, rows, **kwargs):
    snap = write_synthetic_snapshot(tmp_path / "snap", rows)
    return ingest_snapshot(snap, tmp_path / "out", **kwargs)


def _observations(result) -> pd.DataFrame:
    return result.observations.read_all()


def _quarantined(result) -> pd.DataFrame:
    return result.quarantine.read_all()


class TestRowExpansion:
    def test_one_row_with_three_endpoints_yields_three_observations(self, tmp_path):
        rows = [
            make_row(
                **{
                    "BindingDB Reactant_set_id": "1",
                    "Ligand SMILES": "c1ccccc1",
                    "Ligand InChI Key": "UHOVQNZJYSORNB-UHFFFAOYSA-N",
                    "Ki (nM)": "100",
                    "IC50 (nM)": "200",
                    "Kd (nM)": "300",
                    "Number of Protein Chains in Target (>1 implies a multichain complex)": "1",
                    "chain1_sequence": SEQ_A,
                }
            )
        ]
        frame = _observations(_ingest(tmp_path, rows))
        assert len(frame) == 3
        assert set(frame["endpoint"]) == {"Ki", "IC50", "Kd"}
        # Distinct ids, so three measurements never collapse into one.
        assert frame["observation_id"].nunique() == 3

    def test_ec50_is_counted_but_never_expanded(self, tmp_path):
        """EC50 is functional potency and is excluded from the regression."""
        rows = [
            make_row(
                **{
                    "BindingDB Reactant_set_id": "1",
                    "Ligand SMILES": "c1ccccc1",
                    "Ligand InChI Key": "UHOVQNZJYSORNB-UHFFFAOYSA-N",
                    "EC50 (nM)": "50",
                    "Ki (nM)": "100",
                    "Number of Protein Chains in Target (>1 implies a multichain complex)": "1",
                    "chain1_sequence": SEQ_A,
                }
            )
        ]
        result = _ingest(tmp_path, rows)
        frame = _observations(result)
        assert set(frame["endpoint"]) == {"Ki"}
        assert "EC50" not in set(frame["endpoint"])
        assert result.stats.rows_with_ec50 == 1

    def test_row_without_any_endpoint_produces_nothing_but_is_counted(self, tmp_path):
        rows = [
            make_row(
                **{
                    "BindingDB Reactant_set_id": "1",
                    "Ligand SMILES": "c1ccccc1",
                    "Number of Protein Chains in Target (>1 implies a multichain complex)": "1",
                    "chain1_sequence": SEQ_A,
                }
            )
        ]
        result = _ingest(tmp_path, rows)
        assert result.observations.rows == 0
        assert result.stats.rows_without_any_endpoint == 1


class TestCensoring:
    def test_specified_censored_conversion_survives_ingestion(self, tmp_path):
        """The specification's required case, end to end: Kd > 10,000 nM -> pKd < 5."""
        rows = [
            make_row(
                **{
                    "BindingDB Reactant_set_id": "1",
                    "Ligand SMILES": "c1ccccc1",
                    "Ligand InChI Key": "UHOVQNZJYSORNB-UHFFFAOYSA-N",
                    "Kd (nM)": ">10000",
                    "Number of Protein Chains in Target (>1 implies a multichain complex)": "1",
                    "chain1_sequence": SEQ_A,
                }
            )
        ]
        row = _observations(_ingest(tmp_path, rows)).iloc[0]
        assert row["relation_original"] == Relation.GT.value
        # Flipped exactly once, by nm_to_p_value, and never again by the parser.
        assert row["relation_p"] == Relation.LT.value
        assert row["p_value"] == pytest.approx(5.0)
        assert row["is_censored"] and not row["is_exact"]
        assert row["upper_bound"] == pytest.approx(5.0)
        assert pd.isna(row["lower_bound"])

    def test_potent_left_censored_becomes_a_lower_bound(self, tmp_path):
        rows = [
            make_row(
                **{
                    "BindingDB Reactant_set_id": "1",
                    "Ligand SMILES": "c1ccccc1",
                    "Ligand InChI Key": "UHOVQNZJYSORNB-UHFFFAOYSA-N",
                    "Ki (nM)": "<10",
                    "Number of Protein Chains in Target (>1 implies a multichain complex)": "1",
                    "chain1_sequence": SEQ_A,
                }
            )
        ]
        row = _observations(_ingest(tmp_path, rows)).iloc[0]
        assert row["relation_p"] == Relation.GT.value
        assert row["p_value"] == pytest.approx(8.0)
        assert row["lower_bound"] == pytest.approx(8.0)

    def test_non_positive_value_is_quarantined_with_a_reason(self, tmp_path):
        rows = [
            make_row(
                **{
                    "BindingDB Reactant_set_id": "1",
                    "Ligand SMILES": "c1ccccc1",
                    "Ligand InChI Key": "UHOVQNZJYSORNB-UHFFFAOYSA-N",
                    "Kd (nM)": "0",
                    "Number of Protein Chains in Target (>1 implies a multichain complex)": "1",
                    "chain1_sequence": SEQ_A,
                }
            )
        ]
        result = _ingest(tmp_path, rows)
        assert result.observations.rows == 0
        row = _quarantined(result).iloc[0]
        assert row["exclusion_reason"] == Quarantine.VALUE_REJECTED
        assert "non_positive" in row["exclusion_detail"]


class TestChainScreening:
    def _row(self, chains: str, **extra):
        values = {
            "BindingDB Reactant_set_id": "1",
            "Ligand SMILES": "c1ccccc1",
            "Ligand InChI Key": "UHOVQNZJYSORNB-UHFFFAOYSA-N",
            "Ki (nM)": "100",
            "Number of Protein Chains in Target (>1 implies a multichain complex)": chains,
            "chain1_sequence": SEQ_A,
        }
        values.update(extra)
        return make_row(**values)

    def test_single_chain_is_ingested(self, tmp_path):
        result = _ingest(tmp_path, [self._row("1")])
        assert result.observations.rows == 1

    def test_multichain_is_quarantined_not_truncated_to_chain_one(self, tmp_path):
        result = _ingest(tmp_path, [self._row("2", chain2_sequence=SEQ_B)])
        assert result.observations.rows == 0
        assert _quarantined(result).iloc[0]["exclusion_reason"] == Quarantine.MULTICHAIN

    def test_declared_single_chain_with_a_populated_second_chain_is_inconsistent(
        self, tmp_path
    ):
        """The trap: trusting the declared count would silently drop chain 2.

        A row claiming one chain while carrying a second sequence is corrupt, and
        reading chain 1 alone would turn a complex into a single-protein
        observation without any record.
        """
        result = _ingest(tmp_path, [self._row("1", chain2_sequence=SEQ_B)])
        assert result.observations.rows == 0
        row = _quarantined(result).iloc[0]
        assert row["exclusion_reason"] == Quarantine.CHAIN_COUNT_INCONSISTENT
        # bool(...) because a Parquet round-trip returns numpy bools.
        assert bool(row["chain_fields_consistent"]) is False

    def test_unparseable_chain_count_is_quarantined(self, tmp_path):
        result = _ingest(tmp_path, [self._row("")])
        assert _quarantined(result).iloc[0]["exclusion_reason"] == (
            Quarantine.CHAIN_COUNT_UNKNOWN
        )

    def test_missing_smiles_is_quarantined(self, tmp_path):
        rows = [
            make_row(
                **{
                    "BindingDB Reactant_set_id": "1",
                    "Ki (nM)": "100",
                    "Number of Protein Chains in Target (>1 implies a multichain complex)": "1",
                    "chain1_sequence": SEQ_A,
                }
            )
        ]
        result = _ingest(tmp_path, rows)
        assert _quarantined(result).iloc[0]["exclusion_reason"] == Quarantine.NO_SMILES


class TestObservationIds:
    def test_id_is_readable_and_composed_of_source_facts(self):
        assert observation_id("202610", "50012345", Endpoint.KI) == "BDB202610-50012345-KI"

    def test_collision_gets_a_deterministic_suffix(self):
        first = observation_id("202610", "7", Endpoint.KD, 0)
        second = observation_id("202610", "7", Endpoint.KD, 1)
        assert first == "BDB202610-7-KD"
        assert second == "BDB202610-7-KD#2"
        assert first != second

    def test_duplicate_reactant_set_id_is_counted_and_disambiguated(self, tmp_path):
        row = make_row(
            **{
                "BindingDB Reactant_set_id": "7",
                "Ligand SMILES": "c1ccccc1",
                "Ligand InChI Key": "UHOVQNZJYSORNB-UHFFFAOYSA-N",
                "Kd (nM)": "100",
                "Number of Protein Chains in Target (>1 implies a multichain complex)": "1",
                "chain1_sequence": SEQ_A,
            }
        )
        result = _ingest(tmp_path, [row, list(row)])
        assert result.stats.reactant_set_id_collisions == 1
        assert _observations(result)["observation_id"].nunique() == 2

    def test_ids_are_stable_across_runs_and_shard_sizes(self, tmp_path):
        """An id must not depend on sharding, ordering, or the parser version.

        Otherwise a parser fix would invalidate every split manifest ever written.
        """
        rows = [
            make_row(
                **{
                    "BindingDB Reactant_set_id": str(i),
                    "Ligand SMILES": "c1ccccc1" if i % 2 else "CCO",
                    "Ligand InChI Key": f"KEY{i}-UHFFFAOYSA-N",
                    "Ki (nM)": "100",
                    "Number of Protein Chains in Target (>1 implies a multichain complex)": "1",
                    "chain1_sequence": SEQ_A,
                }
            )
            for i in range(25)
        ]
        a = _observations(_ingest(tmp_path / "a", rows, rows_per_shard=5))
        b = _observations(_ingest(tmp_path / "b", rows, rows_per_shard=1000))
        assert sorted(a["observation_id"]) == sorted(b["observation_id"])


class TestDatesAndProvenance:
    def test_us_date_is_parsed_to_the_right_year(self, tmp_path):
        """``8/30/1996`` is M/D/YYYY; a four-character slice would yield '8/30'."""
        rows = [
            make_row(
                **{
                    "BindingDB Reactant_set_id": "1",
                    "Ligand SMILES": "c1ccccc1",
                    "Ligand InChI Key": "UHOVQNZJYSORNB-UHFFFAOYSA-N",
                    "Kd (nM)": "100",
                    "Date of publication": "8/30/1996",
                    "Date in BindingDB": "3/1/2025",
                    "Number of Protein Chains in Target (>1 implies a multichain complex)": "1",
                    "chain1_sequence": SEQ_A,
                }
            )
        ]
        row = _observations(_ingest(tmp_path, rows)).iloc[0]
        assert row["first_evidence_year"] == 1996
        assert row["date_published"] == "1996-08-30"
        # Evidence date is the publication date, never the import date.
        assert row["first_evidence_date"] == "1996-08-30"
        assert row["date_in_bindingdb"] == "2025-03-01"

    def test_curation_source_census_is_collected(self, tmp_path):
        rows = [
            make_row(
                **{
                    "BindingDB Reactant_set_id": str(i),
                    "Ligand SMILES": "c1ccccc1",
                    "Ligand InChI Key": "UHOVQNZJYSORNB-UHFFFAOYSA-N",
                    "Kd (nM)": "100",
                    "Curation/DataSource": source,
                    "Number of Protein Chains in Target (>1 implies a multichain complex)": "1",
                    "chain1_sequence": SEQ_A,
                }
            )
            for i, source in enumerate(["ChEMBL", "ChEMBL", "US Patent"])
        ]
        stats = _ingest(tmp_path, rows).stats
        assert stats.curation_sources["ChEMBL"] == 2
        assert stats.curation_sources["US Patent"] == 1


class TestReaders:
    def test_fast_and_csv_readers_agree(self, tmp_path):
        """The fast split reader is only safe if it matches the reference exactly."""
        rows = [
            make_row(
                **{
                    "BindingDB Reactant_set_id": str(i),
                    "Ligand SMILES": "c1ccccc1",
                    "Ligand InChI Key": f"KEY{i}-UHFFFAOYSA-N",
                    "Ki (nM)": ">100" if i % 3 == 0 else "100",
                    "Number of Protein Chains in Target (>1 implies a multichain complex)": "1",
                    "chain1_sequence": SEQ_A,
                }
            )
            for i in range(40)
        ]
        snap = write_synthetic_snapshot(tmp_path / "snap", rows)
        archive = next(snap.glob("*.zip"))
        fast = list(iter_snapshot_rows(archive, "BindingDB_All.tsv", fast=True))
        slow = list(iter_snapshot_rows(archive, "BindingDB_All.tsv", fast=False))
        assert fast == slow

    def test_both_readers_validate_the_header(self, tmp_path):
        rows = [make_row(**{"BindingDB Reactant_set_id": "1"})]
        snap = write_synthetic_snapshot(tmp_path / "snap", rows)
        archive = next(snap.glob("*.zip"))
        header = next(iter_snapshot_rows(archive, "BindingDB_All.tsv"))
        assert header == synthetic_header()


class TestArtifacts:
    def test_unique_dictionaries_hold_each_entity_once(self, tmp_path):
        rows = [
            make_row(
                **{
                    "BindingDB Reactant_set_id": str(i),
                    "Ligand SMILES": "c1ccccc1",  # same ligand every row
                    "Ligand InChI Key": "UHOVQNZJYSORNB-UHFFFAOYSA-N",
                    "Ki (nM)": "100",
                    "Number of Protein Chains in Target (>1 implies a multichain complex)": "1",
                    "chain1_sequence": SEQ_A,  # same target every row
                }
            )
            for i in range(10)
        ]
        result = _ingest(tmp_path, rows)
        assert result.ligands.rows == 1
        assert result.sequences.rows == 1
        # Observations reference them by integer rather than inlining the strings.
        frame = _observations(result)
        assert set(frame["ligand_ref"]) == {0}
        assert set(frame["sequence_ref"]) == {0}

    def test_emitted_plus_quarantined_equals_every_endpoint_cell(self, tmp_path):
        rows = [
            make_row(
                **{
                    "BindingDB Reactant_set_id": "1",
                    "Ligand SMILES": "c1ccccc1",
                    "Ligand InChI Key": "UHOVQNZJYSORNB-UHFFFAOYSA-N",
                    "Ki (nM)": "100",
                    "IC50 (nM)": "200",
                    "Number of Protein Chains in Target (>1 implies a multichain complex)": "1",
                    "chain1_sequence": SEQ_A,
                }
            ),
            make_row(
                **{
                    "BindingDB Reactant_set_id": "2",
                    "Ligand SMILES": "CCO",
                    "Ligand InChI Key": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
                    "Kd (nM)": "300",
                    "Number of Protein Chains in Target (>1 implies a multichain complex)": "3",
                    "chain1_sequence": SEQ_A,
                }
            ),
        ]
        result = _ingest(tmp_path, rows)
        stats = result.stats
        assert stats.observations_emitted + stats.observations_quarantined == sum(
            stats.endpoint_counts.values()
        )
