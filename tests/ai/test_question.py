import time
from collections.abc import Iterator
from math import isclose
from typing import Any, cast

import pandas as pd  # type: ignore
import pytest

from urim.ai.client import ChatResult
from urim.ai.question import ExtractFunction, ExtractJSON, FreeForm, Rating
from urim.ai.question_cache import QuestionCache

DUMMY_FN = """\
def fn(x: pd.Series):
    return x["count"] + 1
"""


@pytest.fixture()
def temp_cache(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Iterator[QuestionCache]:
    import urim.ai.question as qmod

    cache = QuestionCache(cache_dir=tmp_path)
    monkeypatch.setattr(qmod, "_DEFAULT_CACHE", cache)

    yield cache

    cache.stop()


@pytest.fixture()
def chat_stub_calls() -> dict[str, Any]:
    return {"count": 0}


@pytest.fixture()
def completion(request: pytest.FixtureRequest) -> str:
    return getattr(request, "param", "answer-1")


@pytest.fixture(autouse=True)
def stub_chat_completion_default(
    monkeypatch: pytest.MonkeyPatch,
    chat_stub_calls: dict[str, Any],
    completion: str,
) -> None:
    def _stub(
        self,  # noqa: ANN001
        model: str,
        *,
        messages=None,
        prompt=None,
        **kwargs,
    ) -> ChatResult:
        chat_stub_calls["count"] += 1
        if kwargs.get("logprobs"):
            top = {"50": 0.6, "70": 0.3, "x": 0.1}
            return ChatResult(content=None, raw={"ok": True}, top_tokens=top)

        return ChatResult(content=completion, raw={"ok": True})

    monkeypatch.setattr("urim.ai.question.LLM.chat_completion", _stub, raising=True)


@pytest.mark.parametrize("use_cache", [True, False])
@pytest.mark.parametrize("completion", ["answer-1"])
def test_freeform(
    use_cache: bool,
    temp_cache: QuestionCache,
    chat_stub_calls: dict[str, Any],
) -> None:
    q1 = FreeForm(prompt="Hello", enable_cache=use_cache)
    answer, extra = q1.resolve("test-model")
    assert answer == "answer-1"
    assert extra == {}
    assert cast(int, chat_stub_calls["count"]) == 1

    time.sleep(0.25)

    if use_cache:
        result = temp_cache.read(q1, "test-model")
        assert result is not None
        retr_ans, retr_extra = result
        assert retr_ans == "answer-1"
        assert retr_extra == {}
    else:
        assert temp_cache.read(q1, "test-model") is None

    q2 = FreeForm(prompt="Hello", enable_cache=True)
    answer, extra = q2.resolve("test-model")
    assert answer == "answer-1"
    assert extra == {}

    assert chat_stub_calls["count"] == 1 + int(not use_cache)


@pytest.mark.parametrize("completion", ['{"foo": 1, "bar": "dummy"}'], indirect=True)
def test_extract_json(
    temp_cache: QuestionCache, chat_stub_calls: dict[str, Any]
) -> None:
    q = ExtractJSON(prompt="Give me a dummy json")
    answer, extra = q.resolve("test-model")
    assert answer == '{"foo": 1, "bar": "dummy"}'
    # extra is always a dict; JSON extractor may or may not include parsed obj
    assert isinstance(extra, dict)
    assert chat_stub_calls["count"] == 1

    time.sleep(0.25)

    json = q.json("test-model")
    assert isinstance(json, dict)
    assert json["foo"] == 1
    assert json["bar"] == "dummy"
    assert chat_stub_calls["count"] == 1


@pytest.mark.parametrize("completion", [DUMMY_FN], indirect=True)
def test_extract_function(
    temp_cache: QuestionCache, chat_stub_calls: dict[str, Any]
) -> None:
    q = ExtractFunction(prompt="Write a function")
    code, extra = q.resolve("test-model")
    assert isinstance(code, str) and code.strip().startswith("def fn(")
    assert extra == {}
    assert chat_stub_calls["count"] == 1

    time.sleep(0.25)

    fn_obj = q.fn("test-model")
    assert callable(fn_obj)

    df = pd.DataFrame({"count": [1, 2, 3]})
    assert df.apply(fn_obj, axis=1).eq(pd.Series([2, 3, 4])).all()
    assert chat_stub_calls["count"] == 1


def test_rating(
    temp_cache: QuestionCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _chat_stub_completion(self, model: str, **kwargs):
        return ChatResult(
            content="1",
            raw={"ok": True},
            top_tokens={"1": 0.6, "2": 0.3, "3": 0.1},
        )

    monkeypatch.setattr("urim.ai.question.LLM.chat_completion", _chat_stub_completion)

    q = Rating(
        prompt="Rate the following text: 'This is a test'", min_rating=0, max_rating=3
    )
    answer, extra = q.resolve("test-model")
    assert isclose(answer, 1.5)
    assert extra == {"raw": {"1": 0.6, "2": 0.3, "3": 0.1}}
