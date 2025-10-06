import time
from math import isclose
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from urim.ai.client import ChatResult
from urim.ai.question import ExtractFunction, ExtractJSON, FreeForm, NextToken, Rating

DUMMY_FN = """\
def fn(x: pd.Series):
    return x["count"] + 1
"""


@pytest.fixture()
def completion(request: pytest.FixtureRequest) -> str:
    return getattr(request, "param", "answer-1")


@pytest.fixture(autouse=True)
def force_question_cache_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    import urim.ai.question as qmod

    qmod._caches.clear()

    original_init = qmod.Question.__init__

    def _init(self: Any, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        original_init(self, *args, **kwargs)
        self.cache_dir = Path(tmp_path)

    monkeypatch.setattr(qmod.Question, "__init__", _init, raising=True)


@pytest.fixture(autouse=True)
def chat_stub_calls(monkeypatch: pytest.MonkeyPatch, completion: str) -> dict[str, Any]:
    call_counts: dict[str, Any] = {"count": 0}

    async def _stub(
        self: Any,  # noqa: ANN001
        model: str,
        *,
        messages: Any | None = None,
        prompt: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        call_counts["count"] += 1
        if kwargs.get("logprobs"):
            top = {"50": 0.6, "70": 0.3, "x": 0.1}
            return ChatResult(content=completion, raw={"ok": True}, top_tokens=top)

        return ChatResult(content=completion, raw={"ok": True})

    monkeypatch.setattr("urim.ai.client.LLM.chat_completion", _stub, raising=True)

    return call_counts


@pytest.mark.parametrize("use_cache", [True, False])
@pytest.mark.parametrize("completion", ["answer-1"])
async def test_freeform(use_cache: bool, chat_stub_calls: dict[str, Any]) -> None:
    q1 = FreeForm(prompt="Hello", enable_cache=use_cache)
    answer, extra = await q1.resolve("test-model")
    assert answer == "answer-1"
    assert extra == {}
    assert cast(int, chat_stub_calls["count"]) == 1

    time.sleep(0.25)

    if use_cache:
        result = q1.get_model_cache("test-model").get(q1.hash())
        assert result is not None
        retr_ans, retr_extra = result
        assert retr_ans == "answer-1"
        assert retr_extra == {}
    else:
        assert q1.get_model_cache("test-model").get(q1.hash()) is None

    q2 = FreeForm(prompt="Hello", enable_cache=True)
    answer, extra = await q2.resolve("test-model")
    assert answer == "answer-1"
    assert extra == {}

    assert chat_stub_calls["count"] == 1 + int(not use_cache)


@pytest.mark.parametrize("completion", ['{"foo": 1, "bar": "dummy"}'], indirect=True)
async def test_extract_json(chat_stub_calls: dict[str, Any]) -> None:
    q = ExtractJSON(prompt="Give me a dummy json")
    answer, extra = await q.resolve("test-model")
    assert answer == '{"foo": 1, "bar": "dummy"}'
    # extra is always a dict; JSON extractor may or may not include parsed obj
    assert isinstance(extra, dict)
    assert chat_stub_calls["count"] == 1

    time.sleep(0.25)

    json = await q.json("test-model")
    assert isinstance(json, dict)
    assert json["foo"] == 1
    assert json["bar"] == "dummy"
    assert chat_stub_calls["count"] == 1


@pytest.mark.parametrize("completion", [DUMMY_FN], indirect=True)
async def test_extract_function(chat_stub_calls: dict[str, Any]) -> None:
    q = ExtractFunction(prompt="Write a function")
    code, extra = await q.resolve("test-model")
    assert isinstance(code, str) and code.strip().startswith("def fn(")
    assert extra == {}
    assert chat_stub_calls["count"] == 1

    time.sleep(0.25)

    fn_obj = await q.fn("test-model")
    assert callable(fn_obj)

    df = pd.DataFrame({"count": [1, 2, 3]})
    assert df.apply(fn_obj, axis=1).eq(pd.Series([2, 3, 4])).all()
    assert chat_stub_calls["count"] == 1


async def test_rating(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _chat_stub_completion(_: Any, _model: str, **_kwargs: Any) -> ChatResult:
        return ChatResult(
            content="1",
            raw={"ok": True},
            top_tokens={"1": 0.6, "2": 0.3, "3": 0.1},
        )

    monkeypatch.setattr("urim.ai.client.LLM.chat_completion", _chat_stub_completion)

    q = Rating(prompt="Rate the following text: 'This is a test'", min_rating=0, max_rating=3)
    answer, extra = await q.resolve("test-model")
    assert isclose(answer, 1.5)
    assert extra == {"raw": {"1": 0.6, "2": 0.3, "3": 0.1}}


async def test_next_token(chat_stub_calls: dict[str, Any]) -> None:
    q = NextToken(prompt="Predict next token")
    answer, extra = await q.resolve("test-model")
    assert answer == "answer-1"
    assert extra == {"probs": {"50": 0.6, "70": 0.3, "x": 0.1}}
    assert chat_stub_calls["count"] == 1
