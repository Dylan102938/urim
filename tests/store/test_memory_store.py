from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd
import pytest

from urim.store.memory_store import MemoryStore


def _df(items: Iterable[tuple[str, Any]]) -> pd.DataFrame:
    keys, values = zip(*items, strict=False) if items else ([], [])
    return pd.DataFrame({"key": list(keys), "value": list(values)})


@pytest.fixture()
def mem_store(request: pytest.FixtureRequest) -> MemoryStore:
    params = getattr(request, "param", {}) or {}
    capacity: int = params.get("capacity", 2)
    page_size: int = params.get("page_size", 2)
    preload: list[tuple[str, Any]] = params.get("preload", [])

    store = MemoryStore(capacity, page_size)
    for k, v in preload:
        store.put(k, v)

    return store


def with_mem_store(**kwargs: Any) -> Any:
    return pytest.mark.parametrize("mem_store", [kwargs], indirect=True)


@with_mem_store(preload=[("a", 1)])
def test_get_hit(mem_store: MemoryStore) -> None:
    assert mem_store.get("a") == 1
    assert "a" in mem_store._working_page["key"]


@with_mem_store()
def test_get_miss(mem_store: MemoryStore) -> None:
    assert mem_store.get("a") is None


@with_mem_store()
def test_put(mem_store: MemoryStore) -> None:
    mem_store.put("a", 1)
    assert mem_store.get("a") == 1


@with_mem_store(preload=[("a", 1), ("b", 2)])
def test_page_rotation(mem_store: MemoryStore) -> None:
    mem_store.put("c", 3)
    print(mem_store._working_page)

    assert mem_store._page_queue[0][1]["key"].tolist() == ["a", "b"]
    assert mem_store._working_page["key"] == ["c"]


@with_mem_store(capacity=1, preload=[("a", 1), ("b", 2), ("c", 3), ("d", 4)])
def test_page_eviction(mem_store: MemoryStore) -> None:
    assert len(mem_store._page_queue) == 1
    assert mem_store._page_queue[0][1]["key"].tolist() == ["a", "b"]

    mem_store.put("e", 5)

    assert len(mem_store._page_queue) == 1
    assert mem_store._page_queue[0][1]["key"].tolist() == ["c", "d"]
    assert mem_store._working_page["key"] == ["e"]
