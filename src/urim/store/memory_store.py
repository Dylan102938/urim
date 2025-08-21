import threading
from collections import defaultdict
from typing import Any

import pandas as pd

from urim.store.base import Store


class MemoryStore(Store):
    def __init__(self, capacity: int = 16, page_size: int = 10000) -> None:
        self.capacity = capacity
        self.page_size = page_size

        self._page_queue: list[tuple[int, pd.DataFrame]] = []
        self._working_page: dict[str, list[Any]] = defaultdict(list)
        self._next_page_id: int = 0
        self._lock = threading.RLock()

    def full(self) -> bool:
        return len(self._page_queue) >= self.capacity

    def get(self, key: str) -> Any | None:
        result: Any | None = None

        with self._lock:
            if self._working_page is not None:
                try:
                    idx = self._working_page["key"].index(key)
                except ValueError:
                    idx = None

                if idx is not None:
                    result = self._working_page["value"][idx]
                    return result

            found_page_idx: int | None = None
            for page_idx, (_pid, page) in enumerate(self._page_queue):
                filtered = page[page["key"] == key]
                if not filtered.empty:
                    result = filtered.iloc[0]["value"]
                    found_page_idx = page_idx
                    break

            if (
                found_page_idx is not None
                and found_page_idx != len(self._page_queue) - 1
            ):
                pid, page = self._page_queue.pop(found_page_idx)
                self._page_queue.append((pid, page))

        return result

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            if self._working_page is None:
                self._working_page = defaultdict(list)

            if len(self._working_page["key"]) >= self.page_size:
                self.insert_page(pd.DataFrame(self._working_page))
                self._working_page = defaultdict(list)

            self._working_page["key"].append(key)
            self._working_page["value"].append(value)

    def insert_page(self, page: pd.DataFrame) -> tuple[list[int], list[int]]:
        assert "key" in page.columns
        assert "value" in page.columns

        inserted_ids: list[int] = []
        evicted_ids: list[int] = []

        with self._lock:
            for i in range(0, len(page), self.page_size):
                subpage = page.iloc[i : i + self.page_size].copy()
                while self.full():
                    evicted_pid, _ = self._page_queue.pop(0)
                    evicted_ids.append(evicted_pid)

                pid = self._next_page_id
                self._next_page_id += 1
                self._page_queue.append((pid, subpage))
                inserted_ids.append(pid)

        return (inserted_ids, evicted_ids)
