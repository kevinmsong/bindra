"""First-pass census of a BindingDB snapshot.

Counts, per endpoint, how many observations are exact, censored, unusable, and
how many sit on single-protein targets. This is not the curation report the
specification requires -- no standardization, deduplication or assay grouping
happens here. It is the input to planning that work: it says how much data
each downstream filter actually has to work with.

Streams the archive; never loads the 8.4 GiB TSV into memory.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path
import sys
import time
import zipfile

from bindra.data.bindingdb.schema import (
    COL_FIRST_EVIDENCE_DATE,
    COL_NUM_CHAINS,
    COL_LIGAND_SMILES,
    ENDPOINT_COLUMNS,
    chain_columns,
    declared_chain_count,
    validate_header,
)
from bindra.data.bindingdb.snapshot import SnapshotProvenance
from bindra.data.endpoints import Endpoint, PValue, nm_to_p_value, parse_value_with_relation

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def census(archive_path: Path, member: str, limit: int | None = None) -> dict:
    counts: Counter[str] = Counter()
    endpoint_stats: dict[str, Counter[str]] = {
        endpoint.value: Counter() for endpoint in ENDPOINT_COLUMNS
    }
    chain_histogram: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    years: Counter[str] = Counter()

    unique_smiles: set[str] = set()
    unique_sequences: set[int] = set()  # hashes, to bound memory

    started = time.perf_counter()
    with zipfile.ZipFile(archive_path) as archive, archive.open(member) as raw:
        stream = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
        reader = csv.reader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        validate_header(header)

        index = {name: position for position, name in enumerate(header)}
        smiles_at = index[COL_LIGAND_SMILES]
        chains_at = index[COL_NUM_CHAINS]
        date_at = index[COL_FIRST_EVIDENCE_DATE]
        chain1_seq_at = index[chain_columns(1).sequence]
        endpoint_at = {
            endpoint: index[column] for endpoint, column in ENDPOINT_COLUMNS.items()
        }

        def field(row: list[str], position: int) -> str:
            return row[position].strip() if position < len(row) else ""

        for row_number, row in enumerate(reader, start=1):
            if limit is not None and row_number > limit:
                break
            counts["rows"] += 1

            smiles = field(row, smiles_at)
            if smiles:
                unique_smiles.add(smiles)
            else:
                counts["rows_without_smiles"] += 1

            chain_count = declared_chain_count(field(row, chains_at))
            chain_histogram[str(chain_count) if chain_count else "unknown"] += 1
            single_chain = chain_count == 1

            sequence = field(row, chain1_seq_at)
            if sequence:
                unique_sequences.add(hash(sequence))
            elif single_chain:
                counts["single_chain_without_sequence"] += 1

            date = field(row, date_at)
            years[date[:4] if len(date) >= 4 and date[:4].isdigit() else "unknown"] += 1

            row_has_endpoint = False
            for endpoint, position in endpoint_at.items():
                parsed = parse_value_with_relation(field(row, position))
                stats = endpoint_stats[endpoint.value]
                if parsed is None:
                    continue
                row_has_endpoint = True
                stats["present"] += 1
                relation, value = parsed
                result = nm_to_p_value(value, endpoint, relation)
                if isinstance(result, PValue):
                    stats["exact" if result.is_exact else "censored"] += 1
                    if single_chain:
                        stats["single_chain"] += 1
                        stats["single_chain_exact" if result.is_exact else
                              "single_chain_censored"] += 1
                else:
                    stats["rejected"] += 1
                    rejection_reasons[f"{endpoint.value}:{result.reason.value}"] += 1
            if not row_has_endpoint:
                counts["rows_without_any_affinity_endpoint"] += 1

            if row_number % 250_000 == 0:
                rate = row_number / (time.perf_counter() - started)
                print(f"  {row_number:,} rows  ({rate:,.0f}/s)", file=sys.stderr, flush=True)

    return {
        "rows": counts["rows"],
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "unique_ligand_smiles_raw": len(unique_smiles),
        "unique_chain1_sequences_raw": len(unique_sequences),
        "row_notes": dict(counts),
        "endpoints": {name: dict(stats) for name, stats in endpoint_stats.items()},
        "protein_chain_counts": dict(sorted(chain_histogram.items(), key=lambda kv: kv[0])),
        "rejection_reasons": dict(rejection_reasons),
        "publication_years": dict(sorted(years.items())),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="snapshot directory containing provenance.json")
    parser.add_argument("--limit", type=int, default=None, help="stop after N rows")
    parser.add_argument("--output", default=None, help="write JSON here")
    args = parser.parse_args(argv)

    directory = Path(args.directory)
    provenance = SnapshotProvenance.load(directory)
    member = next(name for name in provenance.members if name.lower().endswith(".tsv"))

    report = census(directory / provenance.file_name, member, args.limit)
    report["snapshot"] = {
        "release": provenance.release,
        "sha256": provenance.sha256,
        "parser_version": provenance.parser_version,
    }

    text = json.dumps(report, indent=2)
    output = Path(args.output) if args.output else directory / "census.json"
    output.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nwritten to {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
