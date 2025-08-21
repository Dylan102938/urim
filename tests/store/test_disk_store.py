from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import orjson
import pandas as pd
import pytest

from urim.store.disk_store import DiskStore


def _write(store: DiskStore, items: Iterable[tuple[str, Any]]) -> None:
    for k, v in items:
        store.put(k, v)
    store._wp.flush()


def reload_disk_store(p: Path, page_size: int, lru_capacity: int) -> DiskStore:
    return DiskStore(
        p, page_capacity=100, lru_capacity=lru_capacity, page_size=page_size
    )


@pytest.fixture()
def disk_store(request: pytest.FixtureRequest, tmp_path: Path) -> DiskStore:
    params = getattr(request, "param", {}) or {}
    page_size: int = params.get("page_size", 2)
    lru_capacity: int = params.get("lru_capacity", 2)
    preload: list[tuple[str, Any]] = params.get("preload", [])

    store = reload_disk_store(tmp_path, page_size, lru_capacity)
    _write(store, preload)
    store = reload_disk_store(tmp_path, page_size, lru_capacity)

    return store


def with_disk_store(**kwargs: Any) -> Any:
    return pytest.mark.parametrize("disk_store", [kwargs], indirect=True)


@with_disk_store(preload=[("a", 1)])
def test_get_hit(disk_store: DiskStore) -> None:
    assert disk_store.get("a") == 1
    # internal: working page holds latest key until rotation
    assert "a" in set(disk_store._wp._page.get("key", []))


@with_disk_store()
def test_get_miss(disk_store: DiskStore) -> None:
    assert disk_store.get("x") is None


@with_disk_store()
def test_put(
    disk_store: DiskStore,
) -> None:
    disk_store.put("a", 1)
    assert disk_store.get("a") == 1


@with_disk_store(page_size=2, preload=[("a", 1), ("b", 2), ("c", 3)])
def test_page_rotation(disk_store: DiskStore, tmp_path: Path) -> None:
    disk_store._wp.flush()

    pages = sorted(tmp_path.glob("page_*.jsonl"))
    assert len(pages) == 2

    p0 = [orjson.loads(x) for x in pages[0].read_text().splitlines()]
    assert p0 == [{"key": "a", "value": 1}, {"key": "b", "value": 2}]

    p1 = [orjson.loads(x) for x in pages[1].read_text().splitlines()]
    assert p1 == [{"key": "c", "value": 3}]


@with_disk_store(page_size=2, preload=[("a", 1), ("b", 2), ("c", 3)])
def test_disk_scans_basic(
    monkeypatch: pytest.MonkeyPatch, disk_store: DiskStore
) -> None:
    calls = {"count": 0}
    original_scan_disk = DiskStore._scan_disk

    def _wrapped_scan_disk(self: DiskStore, key: str) -> Any:  # noqa: ANN401
        calls["count"] += 1
        return original_scan_disk(self, key)

    monkeypatch.setattr(DiskStore, "_scan_disk", _wrapped_scan_disk, raising=True)

    # scan disk since LRU cache is empty
    assert disk_store.get("a") == 1
    assert calls["count"] == 1

    # hit LRU cache (same page as a)
    assert disk_store.get("b") == 2
    assert calls["count"] == 1

    # hit the wp since it currently hasn't been flushed
    assert disk_store.get("c") == 3
    assert calls["count"] == 1

    disk_store.put("d", 4)
    disk_store.put("e", 5)
    disk_store.put("f", 6)
    disk_store.put("g", 7)

    # hit the LRU cache
    assert disk_store.get("d") == 4
    assert calls["count"] == 1

    # hit the wp
    assert disk_store.get("e") == 5
    assert calls["count"] == 1

    # hit the disk
    assert disk_store.get("a") == 1
    assert calls["count"] == 2


@with_disk_store(page_size=2, preload=[("a", 1), ("b", 2), ("c", 3)])
def test_disk_scans_lru_miss(
    monkeypatch: pytest.MonkeyPatch, disk_store: DiskStore
) -> None:
    calls = {"read_json": 0}
    original_read_json = pd.read_json

    def _wrapped_read_json(*args: Any, **kwargs: Any) -> Any:
        calls["read_json"] += 1
        return original_read_json(*args, **kwargs)

    monkeypatch.setattr(pd, "read_json", _wrapped_read_json, raising=True)

    # hit the disk
    assert disk_store.get("a") == 1
    assert calls["read_json"] == 1

    # hit the LRU cache
    assert disk_store.get("b") == 2
    assert calls["read_json"] == 1

    disk_store.put("d", 4)
    disk_store.put("e", 5)
    disk_store.put("f", 6)
    disk_store.put("g", 7)

    assert disk_store.get("a") == 1
    assert calls["read_json"] == 2


@with_disk_store(
    page_size=2, preload=[("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", 5)]
)
def test_disk_scans_no_hits(
    monkeypatch: pytest.MonkeyPatch, disk_store: DiskStore
) -> None:
    calls = {"scan": 0, "read_json": 0}
    original_scan_disk = DiskStore._scan_disk
    original_read_json = pd.read_json

    def _wrapped_scan_disk(self: DiskStore, key: str) -> Any:  # noqa: ANN401
        calls["scan"] += 1
        return original_scan_disk(self, key)

    def _wrapped_read_json(*args: Any, **kwargs: Any) -> Any:
        calls["read_json"] += 1
        return original_read_json(*args, **kwargs)

    monkeypatch.setattr(DiskStore, "_scan_disk", _wrapped_scan_disk, raising=True)
    monkeypatch.setattr(
        "urim.store.disk_store.pd.read_json", _wrapped_read_json, raising=True
    )

    # hit disk
    assert disk_store.get("x") is None
    assert calls["scan"] == 1
    assert calls["read_json"] == 2

    # all pages are already in LRU or wp, read 0 files from disk
    assert disk_store.get("x") is None
    assert calls["scan"] == 2
    assert calls["read_json"] == 2
