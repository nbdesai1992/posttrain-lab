"""Append-only JSONL storage, keyed by a hash of the inputs.

Two things are built on it. ``JsonlStore`` holds one step's results, one row per unit
of work, so a rerun skips what is already done. ``ResponseCache`` holds raw model
responses so the same request is never sent twice.

Files are only ever appended to. If a key appears more than once the last row wins,
and a half-written final line (from a crash mid-write) is ignored on load.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator


def stable_hash(obj: Any) -> str:
    """Hash any JSON-serializable object, independent of dict key order."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=_default)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def _default(value: Any) -> Any:
    if isinstance(value, (tuple, set, frozenset)):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot hash a value of type {type(value).__name__}")


def read_jsonl(path: Path) -> Iterator[dict]:
    """Yield the rows of a JSONL file, skipping blank and unparseable lines."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n"
    with path.open("ab+") as handle:
        # If a crash left the last line unfinished, start this row on a new line
        # so it is not glued to the broken one.
        if handle.seek(0, 2) > 0:
            handle.seek(-1, 2)
            if handle.read(1) != b"\n":
                line = "\n" + line
        handle.write(line.encode("utf-8"))


class JsonlStore:
    """A dictionary of rows backed by one append-only JSONL file."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._rows: dict[str, dict] = {}
        for row in read_jsonl(self.path):
            key = row.get("key")
            if isinstance(key, str):
                self._rows[key] = row

    def __contains__(self, key: str) -> bool:
        return key in self._rows

    def __len__(self) -> int:
        return len(self._rows)

    def get(self, key: str) -> dict | None:
        return self._rows.get(key)

    def put(self, key: str, row: dict) -> dict:
        stored = {"key": key, **row}
        append_jsonl(self.path, stored)
        self._rows[key] = stored
        return stored

    def rows(self) -> list[dict]:
        return list(self._rows.values())


class ResponseCache:
    """Model responses on disk, sharded by key prefix so no single file grows too large.

    A shard is read the first time one of its keys is looked up.
    """

    def __init__(self, directory: Path | str):
        self.directory = Path(directory)
        self._shards: dict[str, JsonlStore] = {}

    def _shard(self, key: str) -> JsonlStore:
        prefix = key[:2]
        if prefix not in self._shards:
            self._shards[prefix] = JsonlStore(self.directory / f"{prefix}.jsonl")
        return self._shards[prefix]

    def get(self, key: str) -> dict | None:
        return self._shard(key).get(key)

    def put(self, key: str, row: dict) -> dict:
        return self._shard(key).put(key, row)
