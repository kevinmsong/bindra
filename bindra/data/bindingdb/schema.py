"""The BindingDB TSV layout, verified against release 202610.

Every constant here was measured from the actual file rather than taken from
documentation, and :func:`validate_header` re-checks them at ingestion time so
a future release that changes shape fails loudly instead of being misparsed.

Layout of ``BindingDB_All.tsv`` (8.4 GiB uncompressed, 640 columns):

* columns 0-39   -- ligand, measurement, provenance
* columns 40-639 -- **50 repeating target-chain blocks of 12 columns each**

The chain blocks are the trap the specification warns about. A row for a
single-protein target still carries all 50 blocks, 49 of them empty, and the
true count is in ``Number of Protein Chains in Target``. Reading only "chain 1"
without consulting that count silently turns a multi-protein complex into a
single-protein observation.
"""

from __future__ import annotations

from dataclasses import dataclass

from bindra.data.endpoints import Endpoint

#: Columns before the first chain block.
PREFIX_COLUMN_COUNT = 40

#: Columns per target-chain block.
CHAIN_BLOCK_SIZE = 12

#: Chain blocks present in every row, regardless of the real chain count.
MAX_CHAINS = 50

EXPECTED_COLUMN_COUNT = PREFIX_COLUMN_COUNT + CHAIN_BLOCK_SIZE * MAX_CHAINS  # 640

COL_REACTANT_SET_ID = "BindingDB Reactant_set_id"
COL_LIGAND_SMILES = "Ligand SMILES"
COL_LIGAND_INCHI_KEY = "Ligand InChI Key"
COL_MONOMER_ID = "BindingDB MonomerID"
COL_TARGET_NAME = "Target Name"
COL_ORGANISM = "Target Source Organism According to Curator or DataSource"
COL_NUM_CHAINS = "Number of Protein Chains in Target (>1 implies a multichain complex)"
COL_CURATION_SOURCE = "Curation/DataSource"
COL_ARTICLE_DOI = "Article DOI"
COL_PMID = "PMID"
COL_PUBCHEM_AID = "PubChem AID"
COL_PATENT = "Patent Number"
COL_DATE_PUBLISHED = "Date of publication"
COL_DATE_IN_BINDINGDB = "Date in BindingDB"
COL_PH = "pH"
COL_TEMPERATURE = "Temp (C)"

#: Endpoint -> the column holding its value in nM. Values may carry a leading
#: relation, e.g. ">10000"; see ``parse_value_with_relation``.
ENDPOINT_COLUMNS: dict[Endpoint, str] = {
    Endpoint.KI: "Ki (nM)",
    Endpoint.IC50: "IC50 (nM)",
    Endpoint.KD: "Kd (nM)",
}

#: Present in the file but excluded from the affinity regression: EC50 is a
#: functional/phenotypic potency endpoint, not a binding constant.
COL_EC50_EXCLUDED = "EC50 (nM)"

#: Date columns. ``Date in BindingDB`` is an *import* date: a 2024 import of a
#: 1998 measurement is not 2024 evidence, so temporal splits must prefer
#: ``Date of publication``.
COL_FIRST_EVIDENCE_DATE = COL_DATE_PUBLISHED
COL_IMPORT_DATE = COL_DATE_IN_BINDINGDB


@dataclass(frozen=True)
class ChainColumns:
    """Column names for one target chain, 1-indexed as in the file."""

    chain: int

    @property
    def sequence(self) -> str:
        return f"BindingDB Target Chain Sequence {self.chain}"

    @property
    def pdb_ids(self) -> str:
        return f"PDB ID(s) of Target Chain {self.chain}"

    @property
    def swissprot_name(self) -> str:
        return f"UniProt (SwissProt) Recommended Name of Target Chain {self.chain}"

    @property
    def swissprot_entry(self) -> str:
        return f"UniProt (SwissProt) Entry Name of Target Chain {self.chain}"

    @property
    def swissprot_primary_id(self) -> str:
        return f"UniProt (SwissProt) Primary ID of Target Chain {self.chain}"

    @property
    def trembl_primary_id(self) -> str:
        return f"UniProt (TrEMBL) Primary ID of Target Chain {self.chain}"

    def start_index(self) -> int:
        """Index of this block's first column."""
        return PREFIX_COLUMN_COUNT + (self.chain - 1) * CHAIN_BLOCK_SIZE


def chain_columns(chain: int) -> ChainColumns:
    if not 1 <= chain <= MAX_CHAINS:
        raise ValueError(f"chain must be in 1..{MAX_CHAINS}, got {chain}")
    return ChainColumns(chain)


class SchemaError(RuntimeError):
    """The file's header is not the layout this parser was written against."""


def validate_header(header: list[str]) -> None:
    """Fail loudly if a new release changed the layout.

    Cheap insurance: BindingDB adds columns between releases, and a silently
    shifted column index would mis-assign every value in the file.
    """
    if len(header) != EXPECTED_COLUMN_COUNT:
        raise SchemaError(
            f"expected {EXPECTED_COLUMN_COUNT} columns "
            f"({PREFIX_COLUMN_COUNT} + {MAX_CHAINS}x{CHAIN_BLOCK_SIZE}), got {len(header)}"
        )

    required = [
        COL_REACTANT_SET_ID,
        COL_LIGAND_SMILES,
        COL_NUM_CHAINS,
        *ENDPOINT_COLUMNS.values(),
    ]
    missing = [name for name in required if name not in header]
    if missing:
        raise SchemaError(f"header is missing required columns: {missing}")

    for chain in (1, MAX_CHAINS):
        columns = chain_columns(chain)
        index = header.index(columns.sequence)
        if index != columns.start_index():
            raise SchemaError(
                f"chain {chain} sequence column is at {index}, "
                f"expected {columns.start_index()}"
            )


def declared_chain_count(raw: str) -> int | None:
    """Parse the declared protein-chain count, or None if unusable."""
    try:
        count = int(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return count if count >= 1 else None


__all__ = [
    "CHAIN_BLOCK_SIZE",
    "COL_ARTICLE_DOI",
    "COL_CURATION_SOURCE",
    "COL_DATE_IN_BINDINGDB",
    "COL_DATE_PUBLISHED",
    "COL_EC50_EXCLUDED",
    "COL_FIRST_EVIDENCE_DATE",
    "COL_IMPORT_DATE",
    "COL_LIGAND_INCHI_KEY",
    "COL_LIGAND_SMILES",
    "COL_MONOMER_ID",
    "COL_NUM_CHAINS",
    "COL_ORGANISM",
    "COL_PATENT",
    "COL_PH",
    "COL_PMID",
    "COL_PUBCHEM_AID",
    "COL_REACTANT_SET_ID",
    "COL_TARGET_NAME",
    "COL_TEMPERATURE",
    "ENDPOINT_COLUMNS",
    "EXPECTED_COLUMN_COUNT",
    "MAX_CHAINS",
    "PREFIX_COLUMN_COUNT",
    "ChainColumns",
    "SchemaError",
    "chain_columns",
    "declared_chain_count",
    "validate_header",
]
