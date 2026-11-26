"""Append-only storage and hashing."""

from __future__ import annotations

from posttrain_lab.store import JsonlStore, ResponseCache, stable_hash


def test_hash_ignores_key_order_and_sees_values():
    assert stable_hash({"a": 1, "b": [1, 2]}) == stable_hash({"b": [1, 2], "a": 1})
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})
    assert stable_hash({"a": (1, 2)}) == stable_hash({"a": [1, 2]})


def test_store_survives_reopening(tmp_path):
    path = tmp_path / "rows.jsonl"
    store = JsonlStore(path)
    store.put("k1", {"value": 1})
    store.put("k2", {"value": 2})
    reopened = JsonlStore(path)
    assert "k1" in reopened and "k2" in reopened
    assert reopened.get("k2")["value"] == 2
    assert len(reopened) == 2


def test_store_is_append_only_and_last_row_wins(tmp_path):
    path = tmp_path / "rows.jsonl"
    store = JsonlStore(path)
    store.put("k", {"value": 1})
    store.put("k", {"value": 2})
    assert len(path.read_text().splitlines()) == 2
    assert JsonlStore(path).get("k")["value"] == 2


def test_store_ignores_a_half_written_last_line(tmp_path):
    path = tmp_path / "rows.jsonl"
    JsonlStore(path).put("k", {"value": 1})
    with path.open("a") as handle:
        handle.write('{"key": "broken", "value"')
    store = JsonlStore(path)
    assert "k" in store and "broken" not in store
    store.put("k2", {"value": 2})
    assert "k2" in JsonlStore(path)


def test_cache_shards_by_key_prefix(tmp_path):
    cache = ResponseCache(tmp_path)
    cache.put("ab12", {"tokens": [1]})
    cache.put("cd34", {"tokens": [2]})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ab.jsonl", "cd.jsonl"]
    assert ResponseCache(tmp_path).get("cd34")["tokens"] == [2]
    assert ResponseCache(tmp_path).get("ab99") is None
