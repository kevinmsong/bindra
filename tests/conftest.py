"""Shared fixtures: a synthetic BindingDB snapshot the ingest tests can own.

The real 8.4 GiB archive is present on this machine but is not a test fixture:
tests that depend on it are marked ``snapshot`` and skipped elsewhere. These
helpers build a tiny archive with the *real* 640-column layout, so the parser is
exercised against the same shape without the 567 MB download.
"""

from __future__ import annotations

from pathlib import Path
import zipfile

import pytest

from bindra.data.bindingdb.schema import (
    CHAIN_BLOCK_SIZE,
    EXPECTED_COLUMN_COUNT,
    MAX_CHAINS,
    PREFIX_COLUMN_COUNT,
    chain_columns,
)
from bindra.data.bindingdb.snapshot import record_snapshot

#: Prefix header, in the verified order of release 202610.
PREFIX_HEADER = [
    "BindingDB Reactant_set_id",
    "Ligand SMILES",
    "Ligand InChI",
    "Ligand InChI Key",
    "BindingDB MonomerID",
    "BindingDB Ligand Name",
    "Target Name",
    "Target Source Organism According to Curator or DataSource",
    "Ki (nM)",
    "IC50 (nM)",
    "Kd (nM)",
    "EC50 (nM)",
    "kon (M-1-s-1)",
    "koff (s-1)",
    "pH",
    "Temp (C)",
    "Curation/DataSource",
    "Article DOI",
    "BindingDB Entry DOI",
    "PMID",
    "PubChem AID",
    "Patent Number",
    "Authors",
    "Date of publication",
    "Date in BindingDB",
    "Institution",
    "Link to Ligand in BindingDB",
    "Link to Target in BindingDB",
    "Link to Ligand-Target Pair in BindingDB",
    "Ligand HET ID in PDB",
    "PDB ID(s) for Ligand-Target Complex",
    "PubChem CID",
    "PubChem SID",
    "ChEBI ID of Ligand",
    "ChEMBL ID of Ligand",
    "DrugBank ID of Ligand",
    "IUPHAR_GRAC ID of Ligand",
    "KEGG ID of Ligand",
    "ZINC ID of Ligand",
    "Number of Protein Chains in Target (>1 implies a multichain complex)",
]

_CHAIN_SUFFIXES = (
    "BindingDB Target Chain Sequence {n}",
    "PDB ID(s) of Target Chain {n}",
    "UniProt (SwissProt) Recommended Name of Target Chain {n}",
    "UniProt (SwissProt) Entry Name of Target Chain {n}",
    "UniProt (SwissProt) Primary ID of Target Chain {n}",
    "UniProt (SwissProt) Secondary ID(s) of Target Chain {n}",
    "UniProt (SwissProt) Alternative ID(s) of Target Chain {n}",
    "UniProt (TrEMBL) Submitted Name of Target Chain {n}",
    "UniProt (TrEMBL) Entry Name of Target Chain {n}",
    "UniProt (TrEMBL) Primary ID of Target Chain {n}",
    "UniProt (TrEMBL) Secondary ID(s) of Target Chain {n}",
    "UniProt (TrEMBL) Alternative ID(s) of Target Chain {n}",
)


def synthetic_header() -> list[str]:
    """The real 640-column header, so ``validate_header`` accepts it."""
    header = list(PREFIX_HEADER)
    assert len(header) == PREFIX_COLUMN_COUNT
    for chain in range(1, MAX_CHAINS + 1):
        for template in _CHAIN_SUFFIXES:
            header.append(template.format(n=chain))
    assert len(header) == EXPECTED_COLUMN_COUNT
    return header


#: Column name -> index, for building rows by name instead of by position.
HEADER_INDEX = {name: i for i, name in enumerate(synthetic_header())}


def make_row(**values: str) -> list[str]:
    """A 640-field row with the named columns set and the rest blank.

    Accepts real column names, plus the shorthands ``chain1_sequence`` /
    ``chain2_sequence`` / ``chain1_swissprot``.
    """
    row = [""] * EXPECTED_COLUMN_COUNT
    shorthand = {
        "chain1_sequence": chain_columns(1).sequence,
        "chain2_sequence": chain_columns(2).sequence,
        "chain1_swissprot": chain_columns(1).swissprot_primary_id,
        "chain1_trembl": chain_columns(1).trembl_primary_id,
    }
    for key, value in values.items():
        name = shorthand.get(key, key)
        if name not in HEADER_INDEX:
            raise KeyError(f"unknown column {name!r}")
        row[HEADER_INDEX[name]] = value
    return row


def write_synthetic_snapshot(
    directory: Path,
    rows: list[list[str]],
    *,
    release: str = "TEST01",
) -> Path:
    """Build a zipped TSV plus its provenance record, as the real loader expects."""
    directory.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(synthetic_header())]
    lines.extend("\t".join(row) for row in rows)
    payload = ("\n".join(lines) + "\n").encode("utf-8")

    archive = directory / f"BindingDB_All_{release}_tsv.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("BindingDB_All.tsv", payload)
    record_snapshot(archive, release, source_url="synthetic://test")
    return directory


@pytest.fixture
def synthetic_snapshot(tmp_path: Path):
    """Factory: pass rows, get a snapshot directory ready for ``ingest_snapshot``."""

    def build(rows: list[list[str]], release: str = "TEST01") -> Path:
        return write_synthetic_snapshot(tmp_path / "snap", rows, release=release)

    return build
