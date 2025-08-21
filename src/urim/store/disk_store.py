import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import orjson
import pandas as pd

from urim.logging_utils import logger
from urim.store.base import Store
from urim.store.memory_store import MemoryStore


class _WorkingPage:
    def __init__(self, pages_dir: Path, idx: int, max_rows: int = 10000) -> None:
        self.idx = idx
        self.max_rows = max_rows
        self.pages_dir = pages_dir
        self._keys: set[str] = set()
        self._page: dict[str, list[Any]] = defaultdict(list)
        self._flushed_count: int = 0

        if self.fpath().exists():
            df = pd.read_json(
                self.fpath(), orient="records", lines=True, dtype={"value": "object"}
            )
            self._keys = set(df["key"])
            self._page = cast(dict[str, list[Any]], df.to_dict(orient="list"))
            self._flushed_count = len(self._page.get("key", []))

    @property
    def full(self) -> bool:
        return len(self._keys) >= self.max_rows

    def fname(self) -> str:
        return f"page_{self.idx:05d}.jsonl"

    def fpath(self) -> Path:
        return self.pages_dir / self.fname()

    def get(self, key: str) -> Any | None:
        try:
            key_idx = self._page["key"].index(key)
        except ValueError:
            key_idx = None

        if key_idx is None:
            return None

        return self._page["value"][key_idx]

    def put(self, key: str, value: Any) -> None:
        if key in self._keys:
            return

        assert len(self._keys) < self.max_rows, "Capacity exceeded in working page"

        self._keys.add(key)
        self._page["key"].append(key)
        self._page["value"].append(value)

    def flush(self) -> None:
        keys = self._page.get("key", [])
        values = self._page.get("value", [])

        if self._flushed_count >= len(keys):
            return

        with self.fpath().open("a", encoding="utf-8") as f:
            for key, value in zip(
                keys[self._flushed_count :],
                values[self._flushed_count :],
                strict=False,
            ):
                f.write(
                    orjson.dumps({"key": key, "value": value}).decode("utf-8") + "\n"
                )

        self._flushed_count = len(keys)


class DiskStore(Store):
    def __init__(
        self,
        base_dir: Path,
        *,
        page_capacity: int = 16,
        lru_capacity: int = 16,
        page_size: int = 10000,
    ) -> None:
        self.base_dir = base_dir
        self.capacity = page_capacity
        self._lru = MemoryStore(lru_capacity, page_size)
        self._pid_to_disk_idx: dict[int, int] = {}
        self._lock = threading.RLock()

        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._wp = _WorkingPage(base_dir, self._max_page_idx(), page_size)

    def full(self) -> bool:
        pages = self._list_pages()
        return len(pages) >= self.capacity

    def get(self, key: str) -> Any | None:
        with self._lock:
            if (result := self._wp.get(key)) is not None:
                return result

            if (result := self._lru.get(key)) is not None:
                return result

            return self._scan_disk(key)

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            try:
                self._wp.put(key, value)
            except AssertionError:
                self._lru_insert_df(pd.DataFrame(self._wp._page), self._wp.idx)
                self._wp.flush()
                self._wp = _WorkingPage(
                    self.base_dir, self._wp.idx + 1, self._wp.max_rows
                )
                self._wp.put(key, value)

    def _scan_disk(self, key: str) -> Any | None:
        with self._lock:
            logger.debug(f"Scanning disk for: {key}")
            pages = self._list_pages()
            pages = sorted(pages, key=lambda p: int(p.stem.split("_")[-1]))

            loaded_disk_idxs = set(self._pid_to_disk_idx.values()) | {self._wp.idx}
            for page in pages:
                page_idx = int(page.stem.split("_")[-1])
                if page_idx in loaded_disk_idxs:
                    continue

                df = pd.read_json(page, lines=True, dtype={"value": "object"})
                result: Any | None = None
                filtered = df[df["key"] == key]
                if not filtered.empty:
                    result = filtered.iloc[0]["value"]

                if result is not None or not self._lru.full():
                    self._lru_insert_df(df, page_idx)

                if result is not None:
                    return result

            return None

    def _max_page_idx(self) -> int:
        pages = self.base_dir.glob("page_*.jsonl")
        page_ids = [int(p.stem.split("_")[-1]) for p in pages]
        return max([0, *page_ids])

    def _list_pages(self) -> list[Path]:
        return list(self.base_dir.glob("page_*.jsonl"))

    def _lru_insert_df(self, df: pd.DataFrame, disk_page_idx: int) -> None:
        inserted_ids, evicted_ids = self._lru.insert_page(df)
        for pid in inserted_ids:
            self._pid_to_disk_idx[pid] = disk_page_idx
        for pid in evicted_ids:
            self._pid_to_disk_idx.pop(pid, None)
