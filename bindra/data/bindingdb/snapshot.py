"""Versioned BindingDB snapshot provenance.

The specification requires that a snapshot record its release, date, source
URL, checksum and parser version, and that the original raw data be retained.
A dataset without this is not reproducible: BindingDB is republished monthly
under predictable filenames, so "we used BindingDB" names no particular data.

This module does not parse the TSV. It establishes *what file we have* so the
parser has something fixed to stand on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

#: Bump when the ingestion parser's output schema or semantics change.
#: Recorded alongside the snapshot so processed tables name the code that
#: produced them.
PARSER_VERSION = "bindingdb-parser-v0"

PROVENANCE_FILENAME = "provenance.json"

#: BindingDB serves the real files from this path. The Download.jsp wrapper
#: returns an HTML page, not the archive, and curl reports 200 for it -- so a
#: "successful" download can quietly be a web page. verify() catches that.
DOWNLOAD_URL_TEMPLATE = "https://www.bindingdb.org/rwd/bind/downloads/{filename}"

ZIP_MAGIC = b"PK\x03\x04"


class SnapshotError(RuntimeError):
    """The snapshot on disk is missing, corrupt, or not what was recorded."""


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Stream a file through SHA-256 without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SnapshotProvenance:
    """Identity of one downloaded BindingDB release."""

    release: str
    source_url: str
    file_name: str
    size_bytes: int
    sha256: str
    downloaded_at: str
    server_last_modified: str | None = None
    parser_version: str = PARSER_VERSION
    members: tuple[str, ...] = ()
    notes: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "SnapshotProvenance":
        payload = json.loads(text)
        payload["members"] = tuple(payload.get("members", ()))
        return cls(**payload)

    @classmethod
    def load(cls, directory: Path) -> "SnapshotProvenance":
        path = Path(directory) / PROVENANCE_FILENAME
        if not path.exists():
            raise SnapshotError(
                f"no {PROVENANCE_FILENAME} in {directory}. An unprovenanced snapshot "
                "cannot be used: record it with `record_snapshot` first."
            )
        return cls.from_json(path.read_text(encoding="utf-8"))

    def save(self, directory: Path) -> Path:
        path = Path(directory) / PROVENANCE_FILENAME
        path.write_text(self.to_json(), encoding="utf-8")
        return path


def looks_like_zip(path: Path) -> bool:
    """Is this actually a zip archive and not an HTML error page?"""
    with path.open("rb") as handle:
        return handle.read(4) == ZIP_MAGIC


def record_snapshot(
    archive_path: Path,
    release: str,
    *,
    source_url: str | None = None,
    server_last_modified: str | None = None,
    notes: str | None = None,
    expected_size_bytes: int | None = None,
) -> SnapshotProvenance:
    """Checksum a downloaded archive and write its provenance record.

    Validates that the file is a real archive before trusting it, because the
    BindingDB download wrapper returns HTTP 200 with an HTML page when given a
    URL it does not like.
    """
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise SnapshotError(f"no such file: {archive_path}")

    size = archive_path.stat().st_size
    if not looks_like_zip(archive_path):
        head = archive_path.read_bytes()[:200]
        raise SnapshotError(
            f"{archive_path.name} is {size} bytes but does not begin with a zip signature. "
            f"It is probably an HTML error page. First bytes: {head[:80]!r}"
        )
    if expected_size_bytes is not None and size != expected_size_bytes:
        raise SnapshotError(
            f"{archive_path.name} is {size} bytes, expected {expected_size_bytes}. "
            "The download is truncated or the release changed."
        )

    with zipfile.ZipFile(archive_path) as archive:
        broken = archive.testzip()
        if broken is not None:
            raise SnapshotError(f"corrupt entry in {archive_path.name}: {broken}")
        members = tuple(sorted(info.filename for info in archive.infolist()))

    provenance = SnapshotProvenance(
        release=release,
        source_url=source_url or DOWNLOAD_URL_TEMPLATE.format(filename=archive_path.name),
        file_name=archive_path.name,
        size_bytes=size,
        sha256=sha256_file(archive_path),
        downloaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        server_last_modified=server_last_modified,
        members=members,
        notes=notes,
    )
    provenance.save(archive_path.parent)
    return provenance


def verify_snapshot(directory: Path, *, recompute_checksum: bool = True) -> SnapshotProvenance:
    """Confirm the archive on disk still matches its recorded provenance."""
    directory = Path(directory)
    provenance = SnapshotProvenance.load(directory)
    archive_path = directory / provenance.file_name

    if not archive_path.exists():
        raise SnapshotError(f"provenance names {provenance.file_name}, which is missing")

    size = archive_path.stat().st_size
    if size != provenance.size_bytes:
        raise SnapshotError(
            f"{provenance.file_name} is {size} bytes; provenance records {provenance.size_bytes}"
        )
    if recompute_checksum:
        actual = sha256_file(archive_path)
        if actual != provenance.sha256:
            raise SnapshotError(
                f"checksum mismatch for {provenance.file_name}:\n"
                f"  recorded {provenance.sha256}\n  actual   {actual}"
            )
    return provenance


__all__ = [
    "DOWNLOAD_URL_TEMPLATE",
    "PARSER_VERSION",
    "PROVENANCE_FILENAME",
    "SnapshotError",
    "SnapshotProvenance",
    "looks_like_zip",
    "record_snapshot",
    "sha256_file",
    "verify_snapshot",
]
