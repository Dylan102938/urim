import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from urim.ai.question import FreeForm, Question
from urim.ai.question_cache import Page, QuestionCache


@pytest.fixture()
def cache(request: pytest.FixtureRequest, tmp_path: Path) -> QuestionCache:
    params = getattr(request, "param", None)
    entries: Iterable[tuple[Question, str, Any]] = []
    page_size = 10000

    if isinstance(params, dict):
        entries = params.get("entries", entries)
        page_size = params.get("page_size", page_size)
    elif params is not None:
        entries = params

    cache = QuestionCache(tmp_path, page_size=page_size)
    for question, model, result in entries:
        cache.set(question, model, result)
    cache.stop()

    return cache


def with_cache(
    entries: Iterable[tuple[Question, str, Any]] | None = None,
    *,
    page_size: int = 10000,
):
    return pytest.mark.parametrize(
        "cache",
        [{"entries": list(entries or []), "page_size": page_size}],
        indirect=True,
    )


@with_cache()
def test_basic_read_write(cache: QuestionCache) -> None:
    q = FreeForm(prompt="What is 2+2?")

    assert cache.read(q, "test-model") is None

    cache.set(q, "test-model", "4")
    cache.stop()

    assert cache.read(q, "test-model") == "4"


@with_cache([(FreeForm(prompt="Q0"), "test-model", "R0")])
def test_read_miss(cache: QuestionCache) -> None:
    q = FreeForm(prompt="Q1")
    assert cache.read(q, "test-model") is None


@with_cache()
def test_flush_without_stop(cache: QuestionCache) -> None:
    q = FreeForm(prompt="Q0")

    cache.start()
    cache.set(q, "model-x", "R0")
    time.sleep(0.25)

    assert cache.read(q, "model-x") == "R0"
    cache.stop()


@with_cache(
    [
        (FreeForm(prompt="Q0"), "model-x", "R0"),
        (FreeForm(prompt="Q1"), "model-x", "R1"),
    ],
    page_size=2,
)
def test_page_rotation(cache: QuestionCache) -> None:
    cache.set(FreeForm(prompt="Q2"), "model-x", "R2")
    cache.stop()

    dir = cache.cache_dir / "model-x"
    assert dir.exists()
    assert len(list(dir.iterdir())) == 2

    q0_id = FreeForm(prompt="Q0").hash()
    q1_id = FreeForm(prompt="Q1").hash()
    q2_id = FreeForm(prompt="Q2").hash()

    page = Page(dir, 0, max_rows=2)
    assert page.path().exists()
    assert page.scan(q0_id)
    assert page.scan(q1_id)
    assert page.scan(q2_id) is None

    page = Page(dir, 1, max_rows=2)
    assert page.path().exists()
    assert page.scan(q0_id) is None
    assert page.scan(q1_id) is None
    assert page.scan(q2_id)

    assert cache.read(FreeForm(prompt="Q0"), "model-x") == "R0"
    assert cache.read(FreeForm(prompt="Q1"), "model-x") == "R1"
    assert cache.read(FreeForm(prompt="Q2"), "model-x") == "R2"


@with_cache()
def test_start_idempotency(cache: QuestionCache) -> None:
    cache.start()
    assert cache._proc is not None and cache._proc.is_alive()
    pid1 = cache._proc.pid
    cache.start()
    assert cache._proc is not None and cache._proc.pid == pid1
    cache.stop()
