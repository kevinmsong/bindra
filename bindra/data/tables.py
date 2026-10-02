"""Columnar table I/O with content hashing.

Every bulk artifact the data layer writes goes through here, for three reasons:

* **Atomic writes.** A table is written to a sibling temporary file and then
  moved into place, so an interrupted run leaves either the previous table or
  nothing -- never a half-written file that a later stage reads as if it were
  complete.
* **Content hashes.** Split and benchmark manifests pin the sha256 of the
  tables they describe. ``verify_manifest`` can then detect curated data
  changing underneath a manifest, instead of silently re-interpreting it.
* **One place to change the format.** Callers hold :class:`TableRef`, not
  Parquet specifics.

Only the Parquet backend exists. The plan allowed for a stdlib ``.npy``
fallback in case ``pyarrow`` could not be installed; it installed cleanly, so
the fallback would be untested dead code and is deliberately absent. The
``backend`` field is still recorded in every ref so an artifact written today
remains self-describing if that ever changes.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Collection, Iterable, Iterator, Sequence

import pandas as pd

#: Rows per Parquet row group. Large enough that per-group overhead is
#: negligible, small enough that a reader can skip most of a file and that peak
#: memory while writing stays near this many rows rather than the whole table.
DEFAULT_ROW_GROUP_SIZE = 100_000

BACKEND = "parquet"
SUFFIX = ".parquet"


class TableError(RuntimeError):
    """A table could not be written, read, or verified."""


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class TableRef:
    """A written table, identified by content rather than by path alone."""

    path: Path
    rows: int
    columns: tuple[str, ...]
    sha256: str
    backend: str = BACKEND

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path.as_posix(),
            "rows": self.rows,
            "columns": list(self.columns),
            "sha256": self.sha256,
            "backend": self.backend,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TableRef":
        return cls(
            path=Path(payload["path"]),
            rows=int(payload["rows"]),
            columns=tuple(payload["columns"]),
            sha256=str(payload["sha256"]),
            backend=str(payload.get("backend", BACKEND)),
        )

    def verify(self, *, root: Path | None = None) -> None:
        """Raise if the file on disk is not the one this ref describes."""
        path = (root / self.path) if root is not None else self.path
        if not path.exists():
            raise TableError(f"table is missing: {path}")
        actual = sha256_file(path)
        if actual != self.sha256:
            raise TableError(
                f"table {path} has changed: expected sha256 {self.sha256[:16]}..., "
                f"found {actual[:16]}.... An artifact that pins this table is no "
                "longer valid for it."
            )


def atomic_replace(tmp: Path, final: Path) -> None:
    """Move ``tmp`` onto ``final``, replacing it, as atomically as the OS allows."""
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, final)


def write_table(
    frame: pd.DataFrame,
    path: Path | str,
    *,
    row_group_size: int = DEFAULT_ROW_GROUP_SIZE,
    compression: str = "zstd",
) -> TableRef:
    """Write a DataFrame atomically and return a content-addressed ref."""
    path = Path(path)
    if path.suffix != SUFFIX:
        path = path.with_suffix(SUFFIX)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_name(path.name + ".tmp")
    try:
        frame.to_parquet(
            tmp,
            engine="pyarrow",
            compression=compression,
            row_group_size=row_group_size,
            index=False,
        )
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise TableError(
            "pyarrow is required to write tables. It is a declared dependency; "
            "install it with `uv pip install pyarrow`."
        ) from exc
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    atomic_replace(tmp, path)
    return TableRef(
        path=path,
        rows=int(len(frame)),
        columns=tuple(str(c) for c in frame.columns),
        sha256=sha256_file(path),
        backend=BACKEND,
    )


def read_table(
    path: Path | str,
    *,
    columns: Sequence[str] | None = None,
    ids: Collection[str] | None = None,
    id_column: str = "observation_id",
) -> pd.DataFrame:
    """Read a table, optionally projecting columns and selecting ids.

    ``ids`` filters after reading rather than pushing a predicate down, because
    an ``IN`` over millions of string keys is not a filter Parquet evaluates
    cheaply. Column projection *is* pushed down and is where the real saving is.
    """
    path = Path(path)
    if not path.exists():
        raise TableError(f"table is missing: {path}")

    requested = list(columns) if columns is not None else None
    if requested is not None and ids is not None and id_column not in requested:
        # The caller asked to filter on a column they did not ask to read.
        requested = [*requested, id_column]

    frame = pd.read_parquet(path, engine="pyarrow", columns=requested)
    if ids is not None:
        wanted = set(ids)
        frame = frame[frame[id_column].isin(wanted)]
        if columns is not None and id_column not in columns:
            frame = frame.drop(columns=[id_column])
    return frame.reset_index(drop=True)


def hash_table(path: Path | str) -> str:
    return sha256_file(Path(path))


# ---------------------------------------------------------------------------
# Sharded tables
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardedTableRef:
    """A table split across many files, hashed as a whole and per shard.

    The combined hash is over the shard hashes in shard order, so it changes if
    any shard changes *or* if shards are added, removed or reordered.
    """

    directory: Path
    shards: tuple[TableRef, ...]
    rows: int
    columns: tuple[str, ...]
    combined_sha256: str
    shard_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "directory": self.directory.as_posix(),
            "rows": self.rows,
            "columns": list(self.columns),
            "combined_sha256": self.combined_sha256,
            "shard_key": self.shard_key,
            "n_shards": len(self.shards),
            "shards": [ref.to_dict() for ref in self.shards],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ShardedTableRef":
        return cls(
            directory=Path(payload["directory"]),
            shards=tuple(TableRef.from_dict(s) for s in payload["shards"]),
            rows=int(payload["rows"]),
            columns=tuple(payload["columns"]),
            combined_sha256=str(payload["combined_sha256"]),
            shard_key=payload.get("shard_key"),
        )

    def verify(self) -> None:
        for ref in self.shards:
            ref.verify()
        if combine_hashes(ref.sha256 for ref in self.shards) != self.combined_sha256:
            raise TableError(
                f"shard set under {self.directory} no longer matches its combined hash"
            )

    def iter_frames(self, *, columns: Sequence[str] | None = None) -> Iterator[pd.DataFrame]:
        """Stream shard by shard, so the whole table is never resident."""
        for ref in self.shards:
            yield read_table(ref.path, columns=columns)

    def read_all(self, *, columns: Sequence[str] | None = None) -> pd.DataFrame:
        frames = list(self.iter_frames(columns=columns))
        if not frames:
            return pd.DataFrame(columns=list(columns or self.columns))
        return pd.concat(frames, ignore_index=True)


def combine_hashes(hashes: Iterable[str]) -> str:
    """Order-sensitive digest over a sequence of hex hashes."""
    digest = hashlib.sha256()
    for value in hashes:
        digest.update(value.encode("ascii"))
        digest.update(b"\x1f")
    return digest.hexdigest()


class ShardWriter:
    """Append rows and flush fixed-size shards, holding only one shard in RAM.

    Used by ingestion, which streams 3.2M rows and must never accumulate them.
    """

    def __init__(
        self,
        directory: Path | str,
        *,
        prefix: str = "part",
        rows_per_shard: int = 500_000,
        row_group_size: int = DEFAULT_ROW_GROUP_SIZE,
        shard_key: str | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.prefix = prefix
        self.rows_per_shard = rows_per_shard
        self.row_group_size = row_group_size
        self.shard_key = shard_key
        self._buffer: list[dict[str, Any]] = []
        self._refs: list[TableRef] = []
        self._rows = 0
        self._columns: tuple[str, ...] | None = None

    def reset(self) -> None:
        """Clear any previous output, so a rerun does not blend with it."""
        if self.directory.exists():
            shutil.rmtree(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> None:
        self._buffer.append(record)
        if len(self._buffer) >= self.rows_per_shard:
            self.flush()

    def extend(self, records: Iterable[dict[str, Any]]) -> None:
        for record in records:
            self.append(record)

    def flush(self) -> None:
        if not self._buffer:
            return
        frame = pd.DataFrame(self._buffer)
        if self._columns is None:
            self._columns = tuple(str(c) for c in frame.columns)
        else:
            # A shard with different columns would make the set unreadable as
            # one table, and the mismatch is far easier to debug here than at
            # read time.
            missing = set(self._columns) - set(frame.columns)
            extra = set(frame.columns) - set(self._columns)
            if missing or extra:
                raise TableError(
                    f"shard {len(self._refs)} column mismatch; missing={sorted(missing)} "
                    f"extra={sorted(extra)}"
                )
            frame = frame[list(self._columns)]
        path = self.directory / f"{self.prefix}-{len(self._refs):05d}{SUFFIX}"
        self._refs.append(write_table(frame, path, row_group_size=self.row_group_size))
        self._rows += len(frame)
        self._buffer.clear()

    def close(self) -> ShardedTableRef:
        self.flush()
        return ShardedTableRef(
            directory=self.directory,
            shards=tuple(self._refs),
            rows=self._rows,
            columns=self._columns or (),
            combined_sha256=combine_hashes(ref.sha256 for ref in self._refs),
            shard_key=self.shard_key,
        )

    def __enter__(self) -> "ShardWriter":
        self.reset()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.flush()


# ---------------------------------------------------------------------------
# JSON side-cars
# ---------------------------------------------------------------------------


def write_json(payload: Any, path: Path | str, *, indent: int = 2) -> Path:
    """Write JSON atomically, sorted, so side-cars are diffable and hashable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=indent, sort_keys=True, default=str), encoding="utf-8")
    atomic_replace(tmp, path)
    return path


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "BACKEND",
    "DEFAULT_ROW_GROUP_SIZE",
    "ShardWriter",
    "ShardedTableRef",
    "TableError",
    "TableRef",
    "atomic_replace",
    "combine_hashes",
    "hash_table",
    "read_json",
    "read_table",
    "sha256_file",
    "write_json",
    "write_table",
]
