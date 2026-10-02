"""``bindra prepare-data`` — record, verify and inspect a BindingDB snapshot.

Currently covers the provenance half of Priority 1: establishing *which* file
we have and what is in it. Parsing, standardization and QC are not implemented;
see docs/STATUS.md.
"""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path
import sys
import zipfile

from bindra.data.bindingdb.snapshot import (
    SnapshotError,
    SnapshotProvenance,
    record_snapshot,
    verify_snapshot,
)

# BindingDB rows are wide and contain long sequences; the default csv field
# limit is too small for them.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def _human(num_bytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(num_bytes) < 1024 or unit == "GiB":
            return f"{num_bytes:,.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:,.1f} GiB"


def cmd_record(args: argparse.Namespace) -> int:
    try:
        provenance = record_snapshot(
            Path(args.archive),
            release=args.release,
            source_url=args.source_url,
            server_last_modified=args.last_modified,
            notes=args.notes,
            expected_size_bytes=args.expected_size,
        )
    except SnapshotError as exc:
        print(f"prepare-data: {exc}", file=sys.stderr)
        return 1

    print("recorded snapshot provenance")
    print(f"    release        {provenance.release}")
    print(f"    file           {provenance.file_name}  ({_human(provenance.size_bytes)})")
    print(f"    sha256         {provenance.sha256}")
    print(f"    source         {provenance.source_url}")
    print(f"    server-dated   {provenance.server_last_modified}")
    print(f"    downloaded     {provenance.downloaded_at}")
    print(f"    parser         {provenance.parser_version}")
    print(f"    members        {', '.join(provenance.members)}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    try:
        provenance = verify_snapshot(Path(args.directory), recompute_checksum=not args.fast)
    except SnapshotError as exc:
        print(f"prepare-data: {exc}", file=sys.stderr)
        return 1
    scope = "size only" if args.fast else "size and sha256"
    print(f"snapshot {provenance.release} verified ({scope}): {provenance.file_name}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Report the archive's columns and a sample of rows, without parsing."""
    directory = Path(args.directory)
    try:
        provenance = SnapshotProvenance.load(directory)
    except SnapshotError as exc:
        print(f"prepare-data: {exc}", file=sys.stderr)
        return 1

    archive_path = directory / provenance.file_name
    with zipfile.ZipFile(archive_path) as archive:
        members = [info for info in archive.infolist() if info.filename.lower().endswith(".tsv")]
        if not members:
            print(f"prepare-data: no .tsv member in {provenance.file_name}", file=sys.stderr)
            return 1
        member = max(members, key=lambda info: info.file_size)
        print(f"snapshot {provenance.release}: {member.filename}")
        print(f"    uncompressed   {_human(member.file_size)}")
        print()

        with archive.open(member) as raw:
            stream = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
            reader = csv.reader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)
            header = next(reader)
            print(f"{len(header)} columns:")
            for index, name in enumerate(header):
                if args.all_columns or index < args.max_columns:
                    print(f"    [{index:3d}] {name}")
            if not args.all_columns and len(header) > args.max_columns:
                print(f"    ... {len(header) - args.max_columns} more (use --all-columns)")

            if args.sample:
                print(f"\nfirst {args.sample} data row(s), non-empty fields only:")
                for row_number, row in enumerate(reader, start=1):
                    if row_number > args.sample:
                        break
                    print(f"  --- row {row_number} ---")
                    for name, value in zip(header, row):
                        if value and value.strip():
                            shown = value if len(value) <= 70 else value[:67] + "..."
                            print(f"    {name[:45]:45s} {shown}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bindra prepare-data",
        description="Record, verify and inspect a BindingDB snapshot.",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    record = sub.add_parser("record", help="checksum an archive and write provenance.json")
    record.add_argument("archive", help="path to the downloaded .zip")
    record.add_argument("--release", required=True, help="release tag, e.g. 202610")
    record.add_argument("--source-url", default=None)
    record.add_argument("--last-modified", default=None, help="server Last-Modified header")
    record.add_argument("--expected-size", type=int, default=None, help="expected byte count")
    record.add_argument("--notes", default=None)
    record.set_defaults(func=cmd_record)

    verify = sub.add_parser("verify", help="re-check an archive against its provenance")
    verify.add_argument("directory", help="snapshot directory containing provenance.json")
    verify.add_argument("--fast", action="store_true", help="check size only, skip the checksum")
    verify.set_defaults(func=cmd_verify)

    inspect = sub.add_parser("inspect", help="list columns and sample rows")
    inspect.add_argument("directory", help="snapshot directory containing provenance.json")
    inspect.add_argument("--sample", type=int, default=0, help="show N data rows")
    inspect.add_argument("--max-columns", type=int, default=60)
    inspect.add_argument("--all-columns", action="store_true")
    inspect.set_defaults(func=cmd_inspect)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
