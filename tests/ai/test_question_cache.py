import signal
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from urim.ai.question import FreeForm
from urim.ai.question_cache import Page, QuestionCache


def test_question_cache_roundtrip(tmp_path: Path) -> None:
    cache = QuestionCache(cache_dir=tmp_path)
    q = FreeForm(prompt="What is 2+2?")

    assert cache.read(q, "test-model") is None

    cache.set(q, "test-model", "4")
    cache.stop()

    assert cache.read(q, "test-model") == "4"


def test_question_cache_page_rotation(tmp_path: Path) -> None:
    cache = QuestionCache(cache_dir=tmp_path, page_size=2)
    qs = [FreeForm(prompt=f"Q{i}") for i in range(5)]

    for i, q in enumerate(qs):
        cache.set(q, "model-x", f"R{i}")

    cache.stop()

    pages = Page.list(tmp_path / "model-x", max_rows=2)
    assert len(pages) == 3

    for page in pages:
        assert page.path().exists()
        with page.path().open("r", encoding="utf-8") as f:
            assert sum(1 for _ in f) <= 2

    for i, q in enumerate(qs):
        assert cache.read(q, "model-x") == f"R{i}"


def test_model_namespace_isolation(tmp_path: Path) -> None:
    cache = QuestionCache(cache_dir=tmp_path)
    q = FreeForm(prompt="same")
    cache.set(q, "model-a", "A")
    cache.stop()
    assert cache.read(q, "model-a") == "A"
    assert cache.read(q, "model-b") is None


def test_read_miss_with_populated_cache(tmp_path: Path) -> None:
    cache = QuestionCache(cache_dir=tmp_path, page_size=2)
    qs = [FreeForm(prompt=f"Q{i}") for i in range(6)]
    for i, q in enumerate(qs):
        cache.set(q, "m", f"R{i}")
    cache.stop()
    missing = FreeForm(prompt="missing")
    assert cache.read(missing, "m") is None


def test_duplicate_key_newest_wins_across_pages(tmp_path: Path) -> None:
    cache = QuestionCache(cache_dir=tmp_path, page_size=1)
    q = FreeForm(prompt="dup")
    cache.set(q, "m", "old")
    cache.set(q, "m", "new")
    cache.stop()
    with ThreadPoolExecutor(max_workers=1) as ex:
        assert cache.read(q, "m", executor=ex) == "new"


def test_reverse_order_scan_hits_newest_page_first(tmp_path: Path) -> None:
    cache = QuestionCache(cache_dir=tmp_path, page_size=2)
    noise = [FreeForm(prompt=f"N{i}") for i in range(4)]
    for i, q in enumerate(noise):
        cache.set(q, "m", f"n{i}")
    target = FreeForm(prompt="target")
    cache.set(target, "m", "hit")
    cache.stop()
    with ThreadPoolExecutor(max_workers=1) as ex:
        assert cache.read(target, "m", executor=ex) == "hit"


def test_start_idempotency(tmp_path: Path) -> None:
    cache = QuestionCache(cache_dir=tmp_path)
    cache.start()
    assert cache._proc is not None and cache._proc.is_alive()
    pid1 = cache._proc.pid
    cache.start()
    assert cache._proc is not None and cache._proc.pid == pid1
    cache.stop()


def test_compute_num_threads_bounds() -> None:
    c = QuestionCache(cache_dir=Path("/tmp"), max_threads=3)
    assert c._compute_num_threads(0) == 1
    assert c._compute_num_threads(1) == 1
    assert c._compute_num_threads(2) == 2
    assert c._compute_num_threads(100) == 3


def test_read_respects_provided_executor(tmp_path: Path, monkeypatch) -> None:
    cache = QuestionCache(cache_dir=tmp_path)
    q = FreeForm(prompt="x")
    cache.set(q, "m", "v")
    cache.stop()

    def _boom(*args, **kwargs):  # noqa: ANN001, ANN002
        raise AssertionError("ThreadPoolExecutor should not be constructed")

    monkeypatch.setattr("urim.ai.question_cache.ThreadPoolExecutor", _boom)
    with ThreadPoolExecutor(max_workers=1) as ex:
        assert cache.read(q, "m", executor=ex) == "v"


def test_background_flush_without_stop(tmp_path: Path) -> None:
    cache = QuestionCache(cache_dir=tmp_path, batch_size=1, flush_interval=0.01)
    q = FreeForm(prompt="bf")
    cache.set(q, "m", "v")
    # Allow writer process to flush
    for _ in range(50):
        if cache.read(q, "m") == "v":
            break
        time.sleep(0.01)
    assert cache.read(q, "m") == "v"
    cache.stop()


def test_interleaved_multi_model_writes(tmp_path: Path) -> None:
    cache = QuestionCache(cache_dir=tmp_path, page_size=2)
    qa = [FreeForm(prompt=f"A{i}") for i in range(3)]
    qb = [FreeForm(prompt=f"B{i}") for i in range(3)]
    for i in range(3):
        cache.set(qa[i], "ma", f"A{i}")
        cache.set(qb[i], "mb", f"B{i}")
    cache.stop()
    for i in range(3):
        assert cache.read(qa[i], "ma") == f"A{i}"
        assert cache.read(qb[i], "mb") == f"B{i}"
    assert (tmp_path / "ma").exists() and (tmp_path / "mb").exists()


def test_persistence_across_cache_restarts(tmp_path: Path) -> None:
    cache1 = QuestionCache(cache_dir=tmp_path)
    q = FreeForm(prompt="persist")
    cache1.set(q, "m", "v")
    cache1.stop()
    cache2 = QuestionCache(cache_dir=tmp_path)
    assert cache2.read(q, "m") == "v"


def test_page_scan_skips_malformed_lines(tmp_path: Path) -> None:
    model_dir = tmp_path / "m"
    model_dir.mkdir(parents=True, exist_ok=True)
    q = FreeForm(prompt="mal")
    qhash = q.hash()
    page_path = model_dir / "page_00000.jsonl"
    with page_path.open("w", encoding="utf-8") as f:
        f.write("{not json}\n")
        f.write(f'{{"question_hash": "{qhash}", "result": "ok"}}\n')
    cache = QuestionCache(cache_dir=tmp_path)
    assert cache.read(q, "m") == "ok"


def test_page_write_stringifies_non_json_result(tmp_path: Path) -> None:
    class NonJSON:
        def __str__(self) -> str:
            return "NONJSON"

    model_dir = tmp_path / "m2"
    model_dir.mkdir(parents=True, exist_ok=True)
    page = Page.working_page(model_dir)
    page.write([{"question_hash": "x", "result": NonJSON()}])
    with page.path().open("r", encoding="utf-8") as f:
        line = f.readline()
    assert "NONJSON" in line


def test_working_page_empty_and_existing(tmp_path: Path) -> None:
    model_dir = tmp_path / "m3"
    model_dir.mkdir(parents=True, exist_ok=True)
    p0 = Page.working_page(model_dir)
    assert p0.idx == 0 and p0.name() == "page_00000.jsonl"
    (model_dir / "page_00000.jsonl").touch()
    (model_dir / "page_00002.jsonl").touch()
    plast = Page.working_page(model_dir)
    assert plast.idx == 2


def test_set_put_retry_does_not_raise(monkeypatch) -> None:
    cache = QuestionCache(cache_dir=Path("/tmp"))

    class FakeQ:
        def __init__(self) -> None:
            self.calls = 0

        def put(self, *_args, **_kwargs) -> None:  # noqa: ANN002
            self.calls += 1
            if self.calls == 1:
                raise Exception("boom")

    cache._q = FakeQ()  # type: ignore[assignment]
    q = FreeForm(prompt="retry")
    cache.set(q, "m", "v")
    calls = getattr(cache._q, "calls", None)  # type: ignore[union-attr]
    assert calls == 2


def test_stop_kills_when_process_stays_alive(monkeypatch) -> None:
    cache = QuestionCache(cache_dir=Path("/tmp"))

    class FakeProc:
        def __init__(self) -> None:
            self.killed = False

        def join(self, *_args, **_kwargs) -> None:  # noqa: ANN002
            return None

        def is_alive(self) -> bool:
            return True

        def kill(self) -> None:
            self.killed = True

    class FakeQ:
        def put(self, *_args, **_kwargs) -> None:  # noqa: ANN002
            return None

    cache._proc = FakeProc()  # type: ignore[assignment]
    cache._q = FakeQ()  # type: ignore[assignment]
    cache.stop(timeout=0.0)
    assert cache._proc is None and cache._q is None


def test_graceful_stop_raises_on_sigint_and_sigterm() -> None:
    cache = QuestionCache(cache_dir=Path("/tmp"))
    try:
        cache._graceful_stop(signal.SIGINT)
        raise AssertionError("should raise")
    except KeyboardInterrupt:
        pass
    try:
        cache._graceful_stop(signal.SIGTERM)
        raise AssertionError("should raise")
    except KeyboardInterrupt:
        pass


def test_write_loop_ignores_non_dict_items(tmp_path: Path) -> None:
    cache = QuestionCache(cache_dir=tmp_path)
    cache.start()
    assert cache._q is not None
    cache._q.put("noise", block=False)  # type: ignore[arg-type]
    cache.stop()


def test_write_loop_handles_queue_empty(tmp_path: Path) -> None:
    cache = QuestionCache(cache_dir=tmp_path, flush_interval=0.01)
    cache.start()
    # wait long enough to trigger at least one queue.Empty in the writer loop
    time.sleep(0.03)
    cache.stop()
