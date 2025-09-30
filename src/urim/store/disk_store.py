import threading
from pathlib import Path
from typing import Any

from urim.store.base import Store


class DiskStore(Store):
    """Maintains a disk-backed jsonl page of key-value pairs.

    Caches up to `page_size` records on the page, and evicts the oldest records if the page is full.
    """

    def __init__(
        self,
        store_path: Path,
        *,
        page_size: int = 1_000_000,
    ) -> None:
        import pandas as pd

        self.store_path = store_path
        self.page_size = page_size

        if self.store_path.exists() and self.store_path.stat().st_size > 0:
            df = pd.read_json(self.store_path, orient="records", lines=True)
            # Handle empty or missing columns gracefully
            if df.empty or "key" not in df.columns:
                self._page = pd.DataFrame(columns=["key", "value"])
            else:
                self._page = df
        else:
            self._page = pd.DataFrame(columns=["key", "value"])

        # Ensure the page uses the key as index
        if "key" in self._page.columns:
            self._page.set_index("key", inplace=True)
        else:
            # If for any reason columns are missing, reset to an empty page
            self._page = pd.DataFrame(columns=["key", "value"]).set_index("key")

        self._lock = threading.RLock()
        self._flush_lock = threading.Lock()

    def full(self) -> bool:
        return len(self._page) >= self.page_size

    def get(self, key: str) -> Any | None:
        with self._lock:
            return self._page["value"].get(key)

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._page.drop(index=key, errors="ignore", inplace=True)
            self._page.loc[key, "value"] = value

            if not self.full():
                return

            overflow = len(self._page) - self.page_size
            if overflow <= 0:
                return

            to_drop = self._page.iloc[:overflow].index
            self._page.drop(index=to_drop, inplace=True)

    def flush(self) -> None:
        with self._flush_lock:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot = self._page.copy().reset_index(names="key")
            snapshot.to_json(self.store_path, orient="records", lines=True)
