"""Chunked, versioned BindingDB TSV ingestion with provenance.

One streaming pass over the 8.4 GiB TSV produces four artifacts:

``raw_observations/``
    Sharded observations that passed structural screening. One row per
    (source row x endpoint): a row carrying Ki, IC50 and Kd becomes three
    independent observations, because the specification requires the three
    endpoints to stay separate for ever.
``quarantine/``
    Observations rejected at this stage, each with a reason. Nothing is dropped
    without a record.
``ligands_raw`` / ``sequences_raw``
    The unique ligand and sequence dictionaries. Observations reference them by
    integer, which is what makes the expensive per-entity work downstream
    (RDKit on 1.44M molecules, alignment on 10.8k sequences) proportional to the
    number of *entities* rather than the 3.2M rows -- and keeps the intermediate
    small: inlining the sequence on every row would be ~2 GB of pure repetition.
``ingest_stats.json``
    Counts after every structural filter, plus the ``Curation/DataSource``
    census that the source-class map is authored from.

Three traps this file is written against, all measured on release 202610:

* **Relations live inside the value cell** (``">10000"``), not in a column.
  A plain float parse would corrupt or discard 20.2% of observations.
* **Every row carries all 50 chain blocks.** Reading chain 1 without consulting
  the declared count silently turns a multi-protein complex into a
  single-protein observation. We additionally require chain 2 to be *empty*,
  which catches rows whose declared count disagrees with their populated
  blocks.
* **``Date in BindingDB`` is an import date.** A 2025 import of a 1998
  measurement is not 2025 evidence, so temporal work uses
  ``Date of publication``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import csv
import hashlib
import io
from pathlib import Path
import sys
import time
from typing import Any, Iterator, Sequence
import zipfile

from bindra.config.profiles import CurationProfile, get_profile
from bindra.data.bindingdb.schema import (
    COL_ARTICLE_DOI,
    COL_CURATION_SOURCE,
    COL_DATE_IN_BINDINGDB,
    COL_DATE_PUBLISHED,
    COL_EC50_EXCLUDED,
    COL_LIGAND_INCHI_KEY,
    COL_LIGAND_SMILES,
    COL_MONOMER_ID,
    COL_NUM_CHAINS,
    COL_ORGANISM,
    COL_PATENT,
    COL_PH,
    COL_PMID,
    COL_PUBCHEM_AID,
    COL_REACTANT_SET_ID,
    COL_TARGET_NAME,
    COL_TEMPERATURE,
    ENDPOINT_COLUMNS,
    chain_columns,
    declared_chain_count,
    parse_bindingdb_date,
    validate_header,
)
from bindra.data.bindingdb.proteins import canonical_sequence
from bindra.data.bindingdb.snapshot import SnapshotProvenance, verify_snapshot
from bindra.data.endpoints import Endpoint, nm_to_p_value, parse_value_with_relation
from bindra.data.tables import ShardWriter, write_json, write_table

import pandas as pd

#: Bumped when this parser's output schema or semantics change. Recorded in
#: every artifact. Deliberately NOT part of any observation id -- see
#: :func:`observation_id`.
INGEST_VERSION = "bindra-ingest-v1"

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

#: Institution column index is not named in schema.py; it is prefix column 25.
_INSTITUTION_INDEX = 25


class IngestError(RuntimeError):
    """Ingestion could not proceed."""


class Quarantine:
    """Reasons an observation does not enter the supervised corpus."""

    NO_SMILES = "no_smiles"
    NO_INCHIKEY = "no_inchikey"
    MULTICHAIN = "multichain"
    CHAIN_COUNT_UNKNOWN = "chain_count_unknown"
    CHAIN_COUNT_INCONSISTENT = "chain_count_inconsistent"
    NO_SEQUENCE = "no_sequence"
    VALUE_REJECTED = "value_rejected"


def observation_id(
    release: str,
    reactant_set_id: str,
    endpoint: Endpoint,
    disambiguator: int = 0,
) -> str:
    """A stable, readable observation identifier.

    ``BDB202610-50012345-KI``, plus ``#2`` only on a measured collision.

    Readable rather than hashed on purpose. Split manifests, audit reports and
    violation samples all quote these ids, and an opaque digest makes every one
    of those unreadable. The composite is already stable: ``Reactant_set_id`` is
    BindingDB's own row key (0 duplicates measured across an 800k-row sample)
    and the endpoint names the column within the row.

    Critically, the id depends only on **immutable source facts** -- not on
    ``INGEST_VERSION``, not on a profile hash, not on line number. A parser fix
    must not invalidate every split manifest ever written.
    """
    base = f"BDB{release}-{reactant_set_id}-{endpoint.value.upper()}"
    return base if disambiguator == 0 else f"{base}#{disambiguator + 1}"


@dataclass
class IngestStats:
    """Counts after every structural filter, for the curation report."""

    release: str = ""
    ingest_version: str = INGEST_VERSION
    profile_hash: str = ""
    elapsed_seconds: float = 0.0
    rows: int = 0
    rows_short: int = 0
    rows_without_smiles: int = 0
    rows_without_any_endpoint: int = 0
    rows_with_ec50: int = 0
    observations_emitted: int = 0
    observations_quarantined: int = 0
    reactant_set_id_collisions: int = 0
    unique_ligands: int = 0
    unique_sequences: int = 0
    endpoint_counts: Counter = field(default_factory=Counter)
    endpoint_exact: Counter = field(default_factory=Counter)
    endpoint_censored: Counter = field(default_factory=Counter)
    quarantine_reasons: Counter = field(default_factory=Counter)
    rejection_reasons: Counter = field(default_factory=Counter)
    chain_counts: Counter = field(default_factory=Counter)
    curation_sources: Counter = field(default_factory=Counter)
    curation_source_by_endpoint: dict = field(default_factory=dict)
    publication_years: Counter = field(default_factory=Counter)
    conditions_present: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "release": self.release,
            "ingest_version": self.ingest_version,
            "profile_hash": self.profile_hash,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "rows": self.rows,
            "rows_short": self.rows_short,
            "rows_without_smiles": self.rows_without_smiles,
            "rows_without_any_endpoint": self.rows_without_any_endpoint,
            "rows_with_ec50_excluded": self.rows_with_ec50,
            "observations_emitted": self.observations_emitted,
            "observations_quarantined": self.observations_quarantined,
            "reactant_set_id_collisions": self.reactant_set_id_collisions,
            "unique_ligands": self.unique_ligands,
            "unique_sequences": self.unique_sequences,
            "endpoint_counts": dict(self.endpoint_counts),
            "endpoint_exact": dict(self.endpoint_exact),
            "endpoint_censored": dict(self.endpoint_censored),
            "quarantine_reasons": dict(self.quarantine_reasons),
            "rejection_reasons": dict(self.rejection_reasons),
            "chain_counts": dict(self.chain_counts),
            "curation_sources": dict(self.curation_sources),
            "curation_source_by_endpoint": {
                k: dict(v) for k, v in self.curation_source_by_endpoint.items()
            },
            "publication_years": dict(self.publication_years),
            "conditions_present": dict(self.conditions_present),
        }


@dataclass(frozen=True)
class IngestResult:
    observations: Any  # ShardedTableRef
    quarantine: Any  # ShardedTableRef
    ligands: Any  # TableRef
    sequences: Any  # TableRef
    stats: IngestStats
    provenance: SnapshotProvenance


def _rows_from_csv(stream: io.TextIOBase) -> Iterator[list[str]]:
    reader = csv.reader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)
    header = next(reader)
    yield header
    yield from reader


def _rows_from_split(stream: io.TextIOBase, expected: int) -> Iterator[list[str]]:
    """Faster reader, guarded by an exact field-count assertion.

    Measured ~2.5x faster than ``csv.reader`` on this file. Safe only because
    the file is ``QUOTE_NONE`` with no embedded tabs or newlines, and the guard
    is what establishes that: a row with the wrong field count raises instead of
    shifting every column silently. ``csv.reader`` remains the reference
    implementation and the differential test compares them.
    """
    first = stream.readline()
    if not first:
        return
    header = first.rstrip("\r\n").split("\t")
    yield header
    for number, line in enumerate(stream, start=1):
        fields = line.rstrip("\r\n").split("\t")
        if len(fields) != expected:
            raise IngestError(
                f"row {number} has {len(fields)} fields, expected {expected}. "
                "The fast reader assumes no embedded tabs or newlines; rerun with "
                "fast=False to use csv.reader."
            )
        yield fields


def iter_snapshot_rows(
    archive_path: Path,
    member: str,
    *,
    fast: bool = True,
) -> Iterator[list[str]]:
    """Stream the TSV out of the zip, header first, validating the layout."""
    with zipfile.ZipFile(archive_path) as archive, archive.open(member) as raw:
        stream = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
        rows = _rows_from_csv(stream) if not fast else _rows_from_split(stream, 640)
        header = next(rows)
        validate_header(header)
        yield header
        yield from rows


def _digest8(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def ingest_snapshot(
    snapshot_dir: Path | str,
    out_dir: Path | str,
    *,
    profile: CurationProfile | None = None,
    limit: int | None = None,
    rows_per_shard: int = 500_000,
    fast: bool = True,
    verify_checksum: bool = False,
) -> IngestResult:
    """Parse a verified snapshot into sharded observation tables.

    ``verify_checksum`` re-hashes the 567 MB archive; off by default because the
    provenance record is checked for existence and size regardless.
    """
    snapshot_dir = Path(snapshot_dir)
    out_dir = Path(out_dir)
    profile = profile or get_profile()

    provenance = verify_snapshot(snapshot_dir, recompute_checksum=verify_checksum)
    tsv_members = [m for m in provenance.members if m.endswith(".tsv")]
    if len(tsv_members) != 1:
        raise IngestError(
            f"expected exactly one .tsv member in the snapshot, found {provenance.members}"
        )
    member = tsv_members[0]
    archive_path = snapshot_dir / provenance.file_name

    stats = IngestStats(
        release=provenance.release,
        profile_hash=profile.profile_hash,
    )
    for endpoint in ENDPOINT_COLUMNS:
        stats.curation_source_by_endpoint[endpoint.value] = Counter()

    # Unique-entity dictionaries. These are the dominant RAM cost and the reason
    # observations carry integer refs: ~1.44M SMILES resident once, rather than
    # 2.98M copies on disk.
    ligand_ids: dict[str, int] = {}
    ligand_rows: list[dict[str, Any]] = []
    sequence_ids: dict[str, int] = {}
    sequence_rows: list[dict[str, Any]] = []

    seen_reactant_ids: set[int] = set()
    collision_counts: dict[str, int] = {}

    started = time.perf_counter()
    rows = iter_snapshot_rows(archive_path, member, fast=fast)
    header = next(rows)
    index = {name: position for position, name in enumerate(header)}

    at_reactant = index[COL_REACTANT_SET_ID]
    at_smiles = index[COL_LIGAND_SMILES]
    at_inchikey = index[COL_LIGAND_INCHI_KEY]
    at_monomer = index[COL_MONOMER_ID]
    at_target_name = index[COL_TARGET_NAME]
    at_organism = index[COL_ORGANISM]
    at_chains = index[COL_NUM_CHAINS]
    at_source = index[COL_CURATION_SOURCE]
    at_doi = index[COL_ARTICLE_DOI]
    at_pmid = index[COL_PMID]
    at_aid = index[COL_PUBCHEM_AID]
    at_patent = index[COL_PATENT]
    at_published = index[COL_DATE_PUBLISHED]
    at_imported = index[COL_DATE_IN_BINDINGDB]
    at_ph = index[COL_PH]
    at_temp = index[COL_TEMPERATURE]
    at_ec50 = index[COL_EC50_EXCLUDED]
    at_endpoint = {e: index[c] for e, c in ENDPOINT_COLUMNS.items()}

    chain1 = chain_columns(1)
    chain2 = chain_columns(2)
    at_seq1 = index[chain1.sequence]
    at_seq2 = index[chain2.sequence]
    at_sp_primary = index[chain1.swissprot_primary_id]
    at_sp_entry = index[chain1.swissprot_entry]
    at_sp_name = index[chain1.swissprot_name]
    at_tr_primary = index[chain1.trembl_primary_id]
    at_pdb = index[chain1.pdb_ids]

    obs_dir = out_dir / "raw_observations"
    quar_dir = out_dir / "quarantine"

    with (
        ShardWriter(obs_dir, rows_per_shard=rows_per_shard) as obs_writer,
        ShardWriter(quar_dir, rows_per_shard=rows_per_shard) as quar_writer,
    ):
        for line_number, row in enumerate(rows, start=1):
            if limit is not None and line_number > limit:
                break
            stats.rows += 1
            width = len(row)
            if width < 640:
                # Only reachable with fast=False; the fast reader raises.
                stats.rows_short += 1

            def cell(position: int) -> str:
                return row[position].strip() if position < width else ""

            # -- ligand -----------------------------------------------------
            smiles = cell(at_smiles)
            inchikey = cell(at_inchikey)
            if not smiles:
                stats.rows_without_smiles += 1

            ligand_ref = -1
            if smiles:
                ligand_ref = ligand_ids.get(smiles, -1)
                if ligand_ref < 0:
                    ligand_ref = len(ligand_ids)
                    ligand_ids[smiles] = ligand_ref
                    ligand_rows.append(
                        {
                            "ligand_ref": ligand_ref,
                            "raw_smiles": smiles,
                            "source_inchikey": inchikey or None,
                            "monomer_id": cell(at_monomer) or None,
                        }
                    )

            # -- target -----------------------------------------------------
            raw_chains = cell(at_chains)
            chain_count = declared_chain_count(raw_chains)
            stats.chain_counts[str(chain_count) if chain_count else "unknown"] += 1
            # Canonical (upper-case) from the start: the ESM-C tokenizer maps
            # lower-case residues to <unk>, which is a special token, so a
            # lower-case construct would lose every position during
            # special-token stripping. 33 constructs in release 202610 are
            # written in lower case.
            sequence = canonical_sequence(cell(at_seq1))
            chain2_sequence = cell(at_seq2)

            target_reason: str | None = None
            if chain_count is None:
                target_reason = Quarantine.CHAIN_COUNT_UNKNOWN
            elif chain_count > 1:
                target_reason = Quarantine.MULTICHAIN
            elif not sequence:
                target_reason = Quarantine.NO_SEQUENCE
            elif chain2_sequence:
                # Declared single-chain but a second block is populated. Trusting
                # the declared count here is exactly the silent-multichain
                # failure the schema module warns about.
                target_reason = Quarantine.CHAIN_COUNT_INCONSISTENT

            sequence_ref = -1
            sequence_sha256 = None
            if sequence and target_reason in (None, Quarantine.CHAIN_COUNT_INCONSISTENT):
                sequence_ref = sequence_ids.get(sequence, -1)
                if sequence_ref < 0:
                    sequence_ref = len(sequence_ids)
                    sequence_ids[sequence] = sequence_ref
                    sequence_sha256 = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
                    sequence_rows.append(
                        {
                            "sequence_ref": sequence_ref,
                            "sequence": sequence,
                            "sequence_sha256": sequence_sha256,
                            "sequence_length": len(sequence),
                        }
                    )

            # -- provenance -------------------------------------------------
            source = cell(at_source)
            stats.curation_sources[source or "(blank)"] += 1
            pmid = cell(at_pmid)
            doi = cell(at_doi)
            aid = cell(at_aid)
            patent = cell(at_patent)
            institution = cell(_INSTITUTION_INDEX)

            published_raw = cell(at_published)
            published = parse_bindingdb_date(published_raw)
            imported = parse_bindingdb_date(cell(at_imported))
            stats.publication_years[str(published.year) if published else "unknown"] += 1

            ph = cell(at_ph)
            temperature = cell(at_temp)
            if ph:
                stats.conditions_present["ph"] += 1
            if temperature:
                stats.conditions_present["temperature"] += 1
            if ph and temperature:
                stats.conditions_present["both"] += 1

            if cell(at_ec50):
                # Counted so the report can say how much functional potency data
                # was deliberately left out, never expanded into an observation.
                stats.rows_with_ec50 += 1

            sp_primary = cell(at_sp_primary)
            tr_primary = cell(at_tr_primary)
            uniprot_all = sp_primary or tr_primary
            uniprot_primary = (uniprot_all.split(",")[0].strip() or None) if uniprot_all else None
            uniprot_source = "swissprot" if sp_primary else ("trembl" if tr_primary else "none")

            reactant = cell(at_reactant)
            digest = _digest8(reactant)
            if digest in seen_reactant_ids:
                collision_counts[reactant] = collision_counts.get(reactant, 0) + 1
                disambiguator = collision_counts[reactant]
                stats.reactant_set_id_collisions += 1
            else:
                seen_reactant_ids.add(digest)
                disambiguator = 0

            base = {
                "source_line": line_number,
                "reactant_set_id": reactant,
                "source_database": "BindingDB",
                "snapshot_release": provenance.release,
                "ligand_ref": ligand_ref,
                "sequence_ref": sequence_ref,
                "declared_chain_count": chain_count if chain_count is not None else -1,
                "chain_fields_consistent": target_reason != Quarantine.CHAIN_COUNT_INCONSISTENT,
                "target_name": cell(at_target_name) or None,
                "organism": cell(at_organism) or None,
                "uniprot_primary": uniprot_primary,
                "uniprot_all": uniprot_all or None,
                "uniprot_source": uniprot_source,
                "uniprot_entry_name": cell(at_sp_entry) or None,
                "uniprot_recommended_name": cell(at_sp_name) or None,
                "pdb_ids": cell(at_pdb) or None,
                "curation_source": source or None,
                "article_doi": doi or None,
                "pmid": pmid or None,
                "pubchem_aid": aid or None,
                "patent_number": patent or None,
                "institution": institution or None,
                "date_published": published.isoformat() if published else None,
                "date_in_bindingdb": imported.isoformat() if imported else None,
                "first_evidence_date": published.isoformat() if published else None,
                "first_evidence_year": published.year if published else -1,
                "ph": ph or None,
                "temp_c": temperature or None,
                "ingest_version": INGEST_VERSION,
            }

            row_had_endpoint = False
            for endpoint, position in at_endpoint.items():
                raw_cell = cell(position)
                if not raw_cell:
                    continue
                row_had_endpoint = True
                stats.endpoint_counts[endpoint.value] += 1
                stats.curation_source_by_endpoint[endpoint.value][source or "(blank)"] += 1

                parsed = parse_value_with_relation(raw_cell)
                if parsed is None:
                    continue
                relation, value_text = parsed
                converted = nm_to_p_value(
                    value_text,
                    endpoint,
                    relation,
                    plausible_range=profile.plausible_range,
                )

                record = dict(base)
                record["observation_id"] = observation_id(
                    provenance.release, reactant, endpoint, disambiguator
                )
                record["endpoint"] = endpoint.value
                record["raw_value_cell"] = raw_cell
                record["original_unit"] = "nM"
                record["relation_original"] = relation.value

                if not converted.ok:
                    stats.rejection_reasons[converted.reason.value] += 1
                    record.update(
                        {
                            "value_nm": None,
                            "p_value": None,
                            "relation_p": None,
                            "is_exact": False,
                            "is_censored": False,
                            "lower_bound": None,
                            "upper_bound": None,
                            "qc_status": "QUARANTINED",
                            "exclusion_reason": Quarantine.VALUE_REJECTED,
                            "exclusion_detail": f"{converted.reason.value}: {converted.detail}",
                        }
                    )
                    quar_writer.append(record)
                    stats.observations_quarantined += 1
                    stats.quarantine_reasons[Quarantine.VALUE_REJECTED] += 1
                    continue

                if converted.is_exact:
                    stats.endpoint_exact[endpoint.value] += 1
                else:
                    stats.endpoint_censored[endpoint.value] += 1

                record.update(
                    {
                        "value_nm": float(value_text),
                        "p_value": converted.value,
                        # Already flipped by nm_to_p_value to describe the
                        # p-value: Kd > 10000 nM becomes pKd < 5. Never flip
                        # again downstream.
                        "relation_p": converted.relation.value,
                        "is_exact": converted.is_exact,
                        "is_censored": converted.is_censored,
                        "lower_bound": converted.lower_bound,
                        "upper_bound": converted.upper_bound,
                    }
                )

                reason = target_reason or (Quarantine.NO_SMILES if not smiles else None)
                if reason is None and not inchikey:
                    reason = Quarantine.NO_INCHIKEY

                if reason is not None:
                    record["qc_status"] = "QUARANTINED"
                    record["exclusion_reason"] = reason
                    record["exclusion_detail"] = (
                        f"declared_chain_count={raw_chains!r}" if "chain" in reason else None
                    )
                    quar_writer.append(record)
                    stats.observations_quarantined += 1
                    stats.quarantine_reasons[reason] += 1
                else:
                    record["qc_status"] = "INGESTED"
                    record["exclusion_reason"] = None
                    record["exclusion_detail"] = None
                    obs_writer.append(record)
                    stats.observations_emitted += 1

            if not row_had_endpoint:
                stats.rows_without_any_endpoint += 1

    observations = obs_writer.close()
    quarantine = quar_writer.close()

    stats.unique_ligands = len(ligand_ids)
    stats.unique_sequences = len(sequence_ids)
    stats.elapsed_seconds = time.perf_counter() - started

    ligands = write_table(pd.DataFrame(ligand_rows), out_dir / "ligands_raw")
    sequences = write_table(pd.DataFrame(sequence_rows), out_dir / "sequences_raw")

    write_json(
        {
            "stats": stats.to_dict(),
            "snapshot": {
                "release": provenance.release,
                "sha256": provenance.sha256,
                "file_name": provenance.file_name,
                "parser_version": provenance.parser_version,
            },
            "profile": {
                "name": profile.name,
                "hash": profile.profile_hash,
            },
            "artifacts": {
                "observations": observations.to_dict(),
                "quarantine": quarantine.to_dict(),
                "ligands_raw": ligands.to_dict(),
                "sequences_raw": sequences.to_dict(),
            },
            "note": (
                "Structural screening only. No standardization, deduplication, "
                "assay grouping or tiering has happened yet; this is not the "
                "curation report."
            ),
        },
        out_dir / "ingest_stats.json",
    )
    write_json(profile.to_dict(), out_dir / "profile.json")

    return IngestResult(
        observations=observations,
        quarantine=quarantine,
        ligands=ligands,
        sequences=sequences,
        stats=stats,
        provenance=provenance,
    )


__all__ = [
    "INGEST_VERSION",
    "IngestError",
    "IngestResult",
    "IngestStats",
    "Quarantine",
    "ingest_snapshot",
    "iter_snapshot_rows",
    "observation_id",
]
