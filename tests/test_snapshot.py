"""Snapshot provenance tests.

The failure these guard against is real and already happened once: BindingDB's
Download.jsp wrapper answered HTTP 200 with a 12 KB HTML page, and curl exited
successfully. A pipeline that trusts the exit code ingests a web page.
"""

from __future__ import annotations

import json
import zipfile

import pytest

from bindra.data.bindingdb.snapshot import (
    PARSER_VERSION,
    SnapshotError,
    SnapshotProvenance,
    looks_like_zip,
    record_snapshot,
    sha256_file,
    verify_snapshot,
)


@pytest.fixture
def zip_archive(tmp_path):
    path = tmp_path / "BindingDB_All_209901_tsv.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("BindingDB_All.tsv", "a\tb\n1\t2\n")
    return path


@pytest.fixture
def html_masquerading_as_zip(tmp_path):
    path = tmp_path / "BindingDB_All_209901_tsv.zip"
    path.write_bytes(b"\r\n\r\n<!DOCTYPE html><html><body>SDFile Download</body></html>")
    return path


class TestArchiveValidation:
    def test_real_zip_is_recognised(self, zip_archive):
        assert looks_like_zip(zip_archive)

    def test_html_page_is_not_a_zip(self, html_masquerading_as_zip):
        assert not looks_like_zip(html_masquerading_as_zip)

    def test_recording_an_html_page_fails_loudly(self, html_masquerading_as_zip):
        with pytest.raises(SnapshotError, match="does not begin with a zip signature"):
            record_snapshot(html_masquerading_as_zip, release="209901")

    def test_truncated_download_is_rejected(self, zip_archive):
        actual = zip_archive.stat().st_size
        with pytest.raises(SnapshotError, match="truncated or the release changed"):
            record_snapshot(zip_archive, release="209901", expected_size_bytes=actual + 1000)

    def test_missing_file_is_rejected(self, tmp_path):
        with pytest.raises(SnapshotError, match="no such file"):
            record_snapshot(tmp_path / "absent.zip", release="209901")


class TestProvenanceRecord:
    def test_records_the_required_fields(self, zip_archive):
        provenance = record_snapshot(
            zip_archive, release="209901", server_last_modified="Wed, 30 Sep 2026 03:16:08 GMT"
        )
        assert provenance.release == "209901"
        assert provenance.sha256 == sha256_file(zip_archive)
        assert provenance.size_bytes == zip_archive.stat().st_size
        assert provenance.parser_version == PARSER_VERSION
        assert provenance.source_url.endswith(zip_archive.name)
        assert provenance.downloaded_at.endswith("+00:00")
        assert provenance.server_last_modified is not None
        assert provenance.members == ("BindingDB_All.tsv",)

    def test_provenance_file_is_written_next_to_the_archive(self, zip_archive):
        record_snapshot(zip_archive, release="209901")
        written = zip_archive.parent / "provenance.json"
        assert written.exists()
        assert json.loads(written.read_text())["release"] == "209901"

    def test_round_trips_through_json(self, zip_archive):
        original = record_snapshot(zip_archive, release="209901")
        assert SnapshotProvenance.from_json(original.to_json()) == original

    def test_loading_without_a_record_refuses(self, tmp_path):
        with pytest.raises(SnapshotError, match="unprovenanced snapshot"):
            SnapshotProvenance.load(tmp_path)


class TestVerification:
    def test_unmodified_snapshot_verifies(self, zip_archive):
        record_snapshot(zip_archive, release="209901")
        assert verify_snapshot(zip_archive.parent).release == "209901"

    def test_modified_archive_fails_checksum(self, zip_archive):
        record_snapshot(zip_archive, release="209901")
        with zipfile.ZipFile(zip_archive, "w") as archive:
            archive.writestr("BindingDB_All.tsv", "a\tb\n9\t9\n")  # same name, new content
        with pytest.raises(SnapshotError, match="checksum mismatch|bytes; provenance records"):
            verify_snapshot(zip_archive.parent)

    def test_deleted_archive_is_reported(self, zip_archive):
        record_snapshot(zip_archive, release="209901")
        zip_archive.unlink()
        with pytest.raises(SnapshotError, match="which is missing"):
            verify_snapshot(zip_archive.parent)
