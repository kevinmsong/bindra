"""Table I/O: content hashing, atomicity, sharding, and tamper detection.

Split and benchmark manifests pin the sha256 of the tables they describe, so
these guarantees are what let ``verify_manifest`` detect curated data changing
underneath a manifest rather than silently re-interpreting it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bindra.data.tables import (
    ShardWriter,
    ShardedTableRef,
    TableError,
    TableRef,
    combine_hashes,
    hash_table,
    read_json,
    read_table,
    write_json,
    write_table,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "observation_id": [f"BDB-{i}-KD" for i in range(10)],
            "p_value": [5.0 + i * 0.1 for i in range(10)],
            "endpoint": ["Kd"] * 10,
        }
    )


class TestRoundTrip:
    def test_write_then_read_is_lossless(self, tmp_path, frame):
        ref = write_table(frame, tmp_path / "t")
        back = read_table(ref.path)
        pd.testing.assert_frame_equal(frame, back)

    def test_ref_describes_what_was_written(self, tmp_path, frame):
        ref = write_table(frame, tmp_path / "t")
        assert ref.rows == 10
        assert ref.columns == ("observation_id", "p_value", "endpoint")
        assert len(ref.sha256) == 64
        assert ref.backend == "parquet"

    def test_suffix_is_applied(self, tmp_path, frame):
        ref = write_table(frame, tmp_path / "noext")
        assert ref.path.suffix == ".parquet"

    def test_ref_serializes_and_restores(self, tmp_path, frame):
        ref = write_table(frame, tmp_path / "t")
        restored = TableRef.from_dict(ref.to_dict())
        assert restored == ref
        restored.verify()


class TestProjectionAndSelection:
    def test_column_projection(self, tmp_path, frame):
        ref = write_table(frame, tmp_path / "t")
        assert list(read_table(ref.path, columns=["p_value"]).columns) == ["p_value"]

    def test_id_selection(self, tmp_path, frame):
        ref = write_table(frame, tmp_path / "t")
        picked = read_table(ref.path, ids={"BDB-3-KD", "BDB-7-KD"})
        assert sorted(picked["observation_id"]) == ["BDB-3-KD", "BDB-7-KD"]

    def test_id_selection_with_projection_does_not_leak_the_id_column(
        self, tmp_path, frame
    ):
        ref = write_table(frame, tmp_path / "t")
        picked = read_table(ref.path, columns=["p_value"], ids={"BDB-3-KD"})
        assert list(picked.columns) == ["p_value"]
        assert len(picked) == 1


class TestVerification:
    def test_unmodified_table_verifies(self, tmp_path, frame):
        write_table(frame, tmp_path / "t").verify()

    def test_modified_table_is_detected(self, tmp_path, frame):
        ref = write_table(frame, tmp_path / "t")
        ref.path.write_bytes(ref.path.read_bytes() + b"junk")
        with pytest.raises(TableError, match="has changed"):
            ref.verify()

    def test_missing_table_is_detected(self, tmp_path, frame):
        ref = write_table(frame, tmp_path / "t")
        ref.path.unlink()
        with pytest.raises(TableError, match="missing"):
            ref.verify()

    def test_rewriting_identical_content_gives_the_same_hash(self, tmp_path, frame):
        """Determinism: a manifest must not break on a harmless re-run."""
        first = write_table(frame, tmp_path / "a")
        second = write_table(frame, tmp_path / "b")
        assert first.sha256 == second.sha256

    def test_changed_content_changes_the_hash(self, tmp_path, frame):
        first = write_table(frame, tmp_path / "a")
        altered = frame.copy()
        altered.loc[0, "p_value"] = 9.9
        second = write_table(altered, tmp_path / "b")
        assert first.sha256 != second.sha256


class TestAtomicity:
    def test_no_temp_file_survives_a_successful_write(self, tmp_path, frame):
        write_table(frame, tmp_path / "t")
        assert not list(tmp_path.glob("*.tmp"))

    def test_a_failed_write_leaves_no_partial_file(self, tmp_path):
        """A column type Parquet cannot store must not leave a torn table."""
        bad = pd.DataFrame({"x": [object()]})
        with pytest.raises(Exception):
            write_table(bad, tmp_path / "t")
        assert not list(tmp_path.glob("*.tmp"))
        assert not (tmp_path / "t.parquet").exists()


class TestSharding:
    def test_shards_split_at_the_requested_size(self, tmp_path):
        with ShardWriter(tmp_path / "sh", rows_per_shard=4) as writer:
            for i in range(10):
                writer.append({"observation_id": str(i), "v": i})
        ref = writer.close()
        assert len(ref.shards) == 3
        assert ref.rows == 10

    def test_streaming_reads_shard_by_shard(self, tmp_path):
        with ShardWriter(tmp_path / "sh", rows_per_shard=4) as writer:
            writer.extend({"observation_id": str(i), "v": i} for i in range(10))
        ref = writer.close()
        assert [len(f) for f in ref.iter_frames()] == [4, 4, 2]
        assert len(ref.read_all()) == 10

    def test_combined_hash_covers_every_shard(self, tmp_path):
        with ShardWriter(tmp_path / "sh", rows_per_shard=4) as writer:
            writer.extend({"observation_id": str(i), "v": i} for i in range(10))
        ref = writer.close()
        ref.verify()
        # Corrupt one shard; the whole set must fail.
        target = ref.shards[1].path
        target.write_bytes(target.read_bytes() + b"x")
        with pytest.raises(TableError):
            ref.verify()

    def test_combined_hash_is_order_sensitive(self):
        assert combine_hashes(["a", "b"]) != combine_hashes(["b", "a"])

    def test_column_mismatch_between_shards_is_rejected(self, tmp_path):
        """A shard set with inconsistent columns is unreadable as one table."""
        writer = ShardWriter(tmp_path / "sh", rows_per_shard=2)
        writer.reset()
        writer.append({"a": 1})
        writer.append({"a": 2})  # flushes shard 0 with column {a}
        writer.append({"b": 3})
        with pytest.raises(TableError, match="column mismatch"):
            writer.append({"b": 4})  # would flush shard 1 with column {b}

    def test_reset_clears_previous_output(self, tmp_path):
        with ShardWriter(tmp_path / "sh", rows_per_shard=2) as writer:
            writer.extend({"a": i} for i in range(6))
        first = writer.close()
        assert len(first.shards) == 3
        with ShardWriter(tmp_path / "sh", rows_per_shard=2) as writer:
            writer.append({"a": 0})
        second = writer.close()
        assert len(second.shards) == 1
        assert len(list((tmp_path / "sh").glob("*.parquet"))) == 1

    def test_sharded_ref_serializes_and_restores(self, tmp_path):
        with ShardWriter(tmp_path / "sh", rows_per_shard=4) as writer:
            writer.extend({"observation_id": str(i), "v": i} for i in range(10))
        ref = writer.close()
        restored = ShardedTableRef.from_dict(ref.to_dict())
        assert restored.combined_sha256 == ref.combined_sha256
        restored.verify()


class TestJsonSidecars:
    def test_json_round_trip_is_sorted_and_atomic(self, tmp_path):
        path = write_json({"b": 2, "a": 1}, tmp_path / "x.json")
        assert read_json(path) == {"a": 1, "b": 2}
        assert path.read_text(encoding="utf-8").index('"a"') < path.read_text(
            encoding="utf-8"
        ).index('"b"')
        assert not list(tmp_path.glob("*.tmp"))

    def test_hash_table_matches_the_ref(self, tmp_path, frame):
        ref = write_table(frame, tmp_path / "t")
        assert hash_table(ref.path) == ref.sha256
