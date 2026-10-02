"""BindingDB layout and value-format tests, verified against release 202610.

The real file is 567 MB and gitignored, so the tests that need it are marked
`snapshot` and skip when it is absent.
"""

from __future__ import annotations

import csv
from datetime import date
import io
from pathlib import Path
import zipfile

import pytest

from bindra.data.bindingdb.schema import (
    CHAIN_BLOCK_SIZE,
    COL_FIRST_EVIDENCE_DATE,
    COL_IMPORT_DATE,
    ENDPOINT_COLUMNS,
    EXPECTED_COLUMN_COUNT,
    MAX_CHAINS,
    PREFIX_COLUMN_COUNT,
    SchemaError,
    chain_columns,
    declared_chain_count,
    parse_bindingdb_date,
    publication_year,
    validate_header,
)
from bindra.data.endpoints import Endpoint, PValue, Relation, nm_to_p_value, parse_value_with_relation

SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "data" / "snapshots" / "bindingdb_202610"
ARCHIVE = SNAPSHOT_DIR / "BindingDB_All_202610_tsv.zip"

csv.field_size_limit(2**31 - 1)


def _synthetic_header() -> list[str]:
    header = [f"prefix_{index}" for index in range(PREFIX_COLUMN_COUNT)]
    header[0] = "BindingDB Reactant_set_id"
    header[1] = "Ligand SMILES"
    header[39] = "Number of Protein Chains in Target (>1 implies a multichain complex)"
    header[8], header[9], header[10] = "Ki (nM)", "IC50 (nM)", "Kd (nM)"
    for chain in range(1, MAX_CHAINS + 1):
        columns = chain_columns(chain)
        block = [f"chain{chain}_{offset}" for offset in range(CHAIN_BLOCK_SIZE)]
        block[0] = columns.sequence
        header.extend(block)
    return header


class TestLayoutConstants:
    def test_column_arithmetic_is_consistent(self):
        assert EXPECTED_COLUMN_COUNT == 640
        assert PREFIX_COLUMN_COUNT + CHAIN_BLOCK_SIZE * MAX_CHAINS == EXPECTED_COLUMN_COUNT

    def test_chain_blocks_start_where_expected(self):
        assert chain_columns(1).start_index() == 40
        assert chain_columns(2).start_index() == 52
        assert chain_columns(MAX_CHAINS).start_index() == 628

    def test_chain_index_is_bounds_checked(self):
        for bad in (0, -1, MAX_CHAINS + 1):
            with pytest.raises(ValueError):
                chain_columns(bad)

    def test_ec50_is_not_an_affinity_endpoint(self):
        # Present in the file, deliberately excluded from the regression.
        assert set(ENDPOINT_COLUMNS) == {Endpoint.KI, Endpoint.IC50, Endpoint.KD}

    def test_temporal_default_is_publication_not_import(self):
        # A recent import of an old measurement is not new evidence.
        assert COL_FIRST_EVIDENCE_DATE != COL_IMPORT_DATE
        assert COL_FIRST_EVIDENCE_DATE == "Date of publication"


class TestHeaderValidation:
    def test_accepts_the_expected_layout(self):
        validate_header(_synthetic_header())

    def test_rejects_a_changed_column_count(self):
        with pytest.raises(SchemaError, match="expected 640 columns"):
            validate_header(_synthetic_header() + ["new column"])

    def test_rejects_a_missing_required_column(self):
        header = _synthetic_header()
        header[header.index("Ligand SMILES")] = "Renamed SMILES"
        with pytest.raises(SchemaError, match="missing required columns"):
            validate_header(header)

    def test_rejects_a_shifted_chain_block(self):
        header = _synthetic_header()
        header.insert(40, "inserted")
        header.pop()
        with pytest.raises(SchemaError, match="sequence column is at"):
            validate_header(header)


class TestChainCount:
    @pytest.mark.parametrize(("raw", "expected"), [("1", 1), ("2", 2), (" 3 ", 3), (4, 4)])
    def test_valid_counts(self, raw, expected):
        assert declared_chain_count(raw) == expected

    @pytest.mark.parametrize("raw", ["", "  ", "n/a", None, "0", "-1"])
    def test_unusable_counts_return_none(self, raw):
        assert declared_chain_count(raw) is None


class TestDateParsing:
    """BindingDB writes US M/D/YYYY, and getting this wrong fails silently."""

    def test_parses_us_format(self):
        assert parse_bindingdb_date("8/30/1996") == date(1996, 8, 30)
        assert parse_bindingdb_date("12/1/2024") == date(2024, 12, 1)

    def test_month_and_day_are_not_transposed(self):
        # 3/11 is 11 March in US format, not 3 November.
        parsed = parse_bindingdb_date("3/11/2020")
        assert (parsed.month, parsed.day) == (3, 11)

    def test_also_accepts_iso(self):
        assert parse_bindingdb_date("2020-03-11") == date(2020, 3, 11)

    @pytest.mark.parametrize("raw", ["", "   ", None, "not a date", "13/45/2020"])
    def test_unusable_dates_return_none(self, raw):
        assert parse_bindingdb_date(raw) is None

    def test_publication_year_extracts_the_year(self):
        assert publication_year("8/30/1996") == 1996

    def test_naive_four_character_slice_would_have_been_wrong(self):
        """Regression guard for the bug this function was written to fix.

        Taking date[:4] of "8/30/1996" yields "8/30", which is not a year, so
        every row classifies as unknown and the temporal panel silently empties
        instead of raising.
        """
        raw = "8/30/1996"
        assert not raw[:4].isdigit()
        assert publication_year(raw) == 1996


class TestValueWithRelation:
    """BindingDB embeds the relation in the value cell."""

    @pytest.mark.parametrize(
        ("cell", "relation", "value"),
        [
            ("100", Relation.EQ, "100"),
            (">10000", Relation.GT, "10000"),
            ("<10", Relation.LT, "10"),
            (">= 5", Relation.GE, "5"),
            ("<=5", Relation.LE, "5"),
            (" > 2.5 ", Relation.GT, "2.5"),
            ("=42", Relation.EQ, "42"),
        ],
    )
    def test_splits_relation_from_value(self, cell, relation, value):
        parsed = parse_value_with_relation(cell)
        assert parsed == (relation, value)

    @pytest.mark.parametrize("cell", ["", "   ", None])
    def test_empty_cells_return_none(self, cell):
        assert parse_value_with_relation(cell) is None

    def test_end_to_end_censored_cell_becomes_a_bound(self):
        relation, value = parse_value_with_relation(">10000")
        result = nm_to_p_value(value, Endpoint.KD, relation)
        assert isinstance(result, PValue)
        assert str(result) == "pKd < 5"
        assert result.is_censored


@pytest.mark.snapshot
@pytest.mark.skipif(not ARCHIVE.exists(), reason="BindingDB snapshot not downloaded")
class TestAgainstTheRealSnapshot:
    @pytest.fixture(scope="class")
    @classmethod
    def header(cls):
        with zipfile.ZipFile(ARCHIVE) as archive, archive.open("BindingDB_All.tsv") as raw:
            stream = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
            return next(csv.reader(stream, delimiter="\t", quoting=csv.QUOTE_NONE))

    def test_real_header_validates(self, header):
        validate_header(header)

    def test_endpoint_columns_exist_where_expected(self, header):
        assert header.index("Ki (nM)") == 8
        assert header.index("IC50 (nM)") == 9
        assert header.index("Kd (nM)") == 10
