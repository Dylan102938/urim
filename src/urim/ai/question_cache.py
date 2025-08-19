from __future__ import annotations

import atexit
import json
import multiprocessing as mp
import os
import queue
import signal
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

from urim.env import URIM_HOME

if TYPE_CHECKING:
    from urim.ai.question import Question


class Page:
    def __init__(self, pages_dir: Path, idx: int, max_rows: int = 10000) -> None:
        self.pages_dir = pages_dir
        self.max_rows = max_rows
        self.idx = idx

        self.cnt = self._count_lines(self.path())

    def name(self) -> str:
        return f"page_{self.idx:05d}.jsonl"

    def path(self) -> Path:
        return self.pages_dir / self.name()

    def scan(self, entry_id: str) -> Any | None:
        with self.path().open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("question_hash") == entry_id:
                    return obj.get("result")
        return None

    def write(self, rows: list[dict[str, Any]]) -> Page:
        remaining_rows = self.max_rows - self.cnt
        rows_to_write = rows[:remaining_rows]

        self.cnt += len(rows_to_write)
        with self.path().open("a", encoding="utf-8") as f:
            for row in rows_to_write:
                try:
                    serialized = json.dumps(row, ensure_ascii=False)
                except TypeError:
                    row = {**row, "result": str(row.get("result"))}
                    serialized = json.dumps(row, ensure_ascii=False)

                f.write(serialized + "\n")

        if len(rows) > remaining_rows:
            new_page = Page(self.pages_dir, self.idx + 1, self.max_rows)
            return new_page.write(rows[remaining_rows:])

        return self

    @classmethod
    def list(cls, pages_dir: Path, max_rows: int = 10000) -> list[Page]:
        pages = sorted(pages_dir.glob("page_*.jsonl"))
        return [cls(pages_dir, int(p.stem.split("_")[-1]), max_rows) for p in pages]

    @classmethod
    def working_page(cls, pages_dir: Path, max_rows: int = 10000) -> Page:
        pages = sorted(pages_dir.glob("page_*.jsonl"))
        if not pages:
            return cls(pages_dir, 0, max_rows)

        page_id = int(pages[-1].stem.split("_")[-1])
        return cls(pages_dir, page_id, max_rows)

    @staticmethod
    def _count_lines(path: Path) -> int:
        if not path.exists():
            return 0

        count = 0
        with path.open("r", encoding="utf-8") as f:
            for _ in f:
                count += 1

        return count


WRITE_LOOP_STARTED = False
lock = mp.Lock()


def _check_in_write_loop() -> None:
    with lock:
        global WRITE_LOOP_STARTED
        assert not WRITE_LOOP_STARTED, "Only one write process is allowed"

        WRITE_LOOP_STARTED = True


def _write_loop(
    cache_dir: Path, q: mp.Queue, page_size: int, batch_size: int, flush_interval: float
) -> None:
    _check_in_write_loop()
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    models = [m.name for m in cache_dir.iterdir() if (cache_dir / m).is_dir()]
    pending: dict[str, list[dict[str, Any]]] = {m: [] for m in models}
    models_curr_pages = {m: Page.working_page(cache_dir / m, page_size) for m in models}

    def _flush_model(model: str) -> None:
        rows = pending.get(model) or []
        if not rows:
            return

        (cache_dir / model).mkdir(parents=True, exist_ok=True)

        page = models_curr_pages.get(
            model, Page.working_page(cache_dir / model, page_size)
        )
        page = page.write(rows)
        pending[model] = []
        models_curr_pages[model] = page

    def _flush_all() -> None:
        for model in pending.keys():
            _flush_model(model)

    last_flush = time.monotonic()

    while True:
        try:
            item = q.get(timeout=flush_interval)
            if not isinstance(item, dict):
                continue
            if item.get("op") == "stop":
                _flush_all()
                break
            elif item.get("op") == "set":
                model = str(item["model"])
                pending.setdefault(model, []).append(
                    {"question_hash": item["question_hash"], "result": item["result"]}
                )
        except queue.Empty:
            pass

        now = time.monotonic()
        total_pending = sum(len(v) for v in pending.values())
        if total_pending >= batch_size or (now - last_flush) >= flush_interval:
            _flush_all()
            last_flush = now


class QuestionCache:
    def __init__(
        self,
        cache_dir: str | Path = URIM_HOME / "questions",
        *,
        page_size: int = 10000,
        max_threads: int = 2,
        batch_size: int = 64,
        flush_interval: float = 0.1,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.page_size = page_size
        self.max_threads = max_threads
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._proc: mp.Process | None = None
        self._q: mp.Queue | None = None

    def __del__(self) -> None:
        self.stop()

    def set(
        self, question: Question, model: str, result: Any, **fill_prompt_kwargs: Any
    ) -> None:
        if self._q is None:
            self.start()

        assert self._q is not None

        qhash = question.hash(**fill_prompt_kwargs)
        msg = {"op": "set", "model": model, "question_hash": qhash, "result": result}

        for _ in range(2):
            try:
                self._q.put(msg, block=False, timeout=0.1)
                break
            except Exception:
                # event dropped (not critical since this is a cache)
                pass

    def read(
        self,
        question: Question,
        model: str,
        *,
        executor: ThreadPoolExecutor | None = None,
        **fill_prompt_kwargs: Any,
    ) -> Any | None:
        qhash = question.hash(**fill_prompt_kwargs)
        pages = Page.list(self.cache_dir / model, max_rows=self.page_size)
        pages_rev = list(reversed(pages))

        if not pages_rev:
            return None

        if executor is not None:
            futures = [executor.submit(p.scan, qhash) for p in pages_rev]
        else:
            workers = self._compute_num_threads(len(pages_rev))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(p.scan, qhash) for p in pages_rev]

        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                return result

        return None

    def start(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            return

        self._q = mp.Queue(maxsize=10000)
        self._proc = mp.Process(
            target=_write_loop,
            args=(
                self.cache_dir,
                self._q,
                self.page_size,
                self.batch_size,
                self.flush_interval,
            ),
        )

        assert self._proc is not None
        self._proc.start()

        atexit.register(self._graceful_stop)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda sig, _: self._graceful_stop(sig))

    def stop(self, timeout: float | None = 5.0) -> None:
        if self._q is None or self._proc is None:
            return

        self._q.put({"op": "stop"}, block=False)
        self._proc.join(timeout=timeout)
        if self._proc.is_alive():
            self._proc.kill()

        self._proc = None
        self._q = None

    def _graceful_stop(self, sig: int | None = None) -> None:
        try:
            self.stop()
        finally:
            if sig is not None and (sig == signal.SIGINT or sig == signal.SIGTERM):
                raise KeyboardInterrupt

    def _compute_num_threads(self, num_pages: int) -> int:
        cpu = os.cpu_count() or 4
        return max(1, min(self.max_threads, cpu * 4, num_pages))
