from __future__ import annotations

import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from urim.store.disk_store import DiskStore


def _write(store: DiskStore, items: Iterable[tuple[str, Any]]) -> None:
    for k, v in items:
        store.put(k, v)

    store.flush()


def reload_disk_store(p: Path, page_size: int) -> DiskStore:
    return DiskStore(p, page_size=page_size)


@pytest.fixture()
def disk_store(request: pytest.FixtureRequest, tmp_path: Path) -> DiskStore:
    params = getattr(request, "param", {}) or {}
    page_size: int = params.get("page_size", 2)
    preload: list[tuple[str, Any]] = params.get("preload", [])

    store_path = tmp_path / "store.jsonl"
    store = reload_disk_store(store_path, page_size)
    _write(store, preload)
    store = reload_disk_store(store_path, page_size)

    return store


def with_disk_store(**kwargs: Any) -> Any:
    return pytest.mark.parametrize("disk_store", [kwargs], indirect=True)


@with_disk_store(preload=[("a", 1)])
def test_get_hit(disk_store: DiskStore) -> None:
    assert disk_store.get("a") == 1


@with_disk_store()
def test_get_miss(disk_store: DiskStore) -> None:
    assert disk_store.get("x") is None


@with_disk_store()
def test_put(
    disk_store: DiskStore,
) -> None:
    disk_store.put("a", 1)
    assert disk_store.get("a") == 1


@with_disk_store(page_size=2, preload=[("a", 1)])
def test_put_overflow(disk_store: DiskStore) -> None:
    disk_store.put("b", 2)

    assert disk_store.get("a") == 1
    assert disk_store.get("b") == 2
    assert disk_store.full()

    disk_store.put("c", 3)

    assert disk_store.get("a") is None
    assert disk_store.get("b") == 2
    assert disk_store.get("c") == 3
    assert disk_store.full()


@with_disk_store(page_size=100)
def test_put_multithreaded(disk_store: DiskStore) -> None:
    num_threads = 50

    start_barrier = threading.Barrier(num_threads)

    def worker(i: int) -> None:
        start_barrier.wait()
        disk_store.put(f"k{i}", i)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for i in range(num_threads):
        assert disk_store.get(f"k{i}") == i

    disk_store.flush()
    reloaded = reload_disk_store(disk_store.store_path, page_size=disk_store.page_size)
    for i in range(num_threads):
        assert reloaded.get(f"k{i}") == i


@with_disk_store(page_size=10)
def test_eviction_under_concurrent_put(disk_store: DiskStore) -> None:
    total_keys = 50

    start_barrier = threading.Barrier(total_keys)

    def writer(i: int) -> None:
        start_barrier.wait()
        disk_store.put(f"e{i}", i)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(total_keys)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    present = sum(1 for i in range(total_keys) if disk_store.get(f"e{i}") is not None)
    assert present == disk_store.page_size
    assert disk_store.full()


@with_disk_store(preload=[(f"f{i}", i) for i in range(20)], page_size=20)
def test_concurrent_flush_is_serialized(disk_store: DiskStore) -> None:
    num_flushers = 10
    start_barrier = threading.Barrier(num_flushers)
    errors: list[BaseException] = []

    def flusher() -> None:
        try:
            start_barrier.wait()
            disk_store.flush()
        except BaseException as exc:  # capture any unexpected concurrency errors
            errors.append(exc)

    threads = [threading.Thread(target=flusher) for _ in range(num_flushers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors

    # Data on disk should be a valid snapshot of the in-memory page
    reloaded = reload_disk_store(disk_store.store_path, page_size=disk_store.page_size)
    assert sum(1 for i in range(20) if reloaded.get(f"f{i}") == i) == 20
