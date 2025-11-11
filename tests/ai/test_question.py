import random
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from urim.ai.client import ChatResult, TopTokens
from urim.ai.question import ExtractFunction, ExtractJSON, FreeForm, NextToken, Rating
from urim.env import set_storage_root

DUMMY_FN = """\
def fn(x: pd.Series):
    return x["count"] + 1
"""


@pytest.fixture(autouse=True)
def configure_question_storage(tmp_path: Path) -> None:
    import urim.ai.question as qmod

    set_storage_root(tmp_path)
    qmod._caches.clear()


@pytest.fixture()
def completion(request: pytest.FixtureRequest) -> str:
    return getattr(request, "param", "answer-1")


@pytest.fixture(autouse=True)
def completions_stub(monkeypatch: pytest.MonkeyPatch, completion: str) -> dict[str, Any]:
    call_counts: dict[str, Any] = {"count": 0}

    async def _stub(
        _model: str,
        *,
        _messages: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        call_counts["count"] += 1
        top: list[TopTokens] | None = None
        if kwargs.get("logprobs"):
            tokens = [tok if i == 0 else " " + tok for i, tok in enumerate(completion.split())]
            top = []
            for t in tokens:
                p = random.random()
                top.append(
                    TopTokens(token=t, value=p, top_scores={t: p, "20": 0.1, "30": 0.2, "60": 0.1})
                )

        return ChatResult(content=completion, top_tokens=top, raw={"ok": True})

    monkeypatch.setattr("urim.ai.client.chat_completion", _stub, raising=True)

    return call_counts


@pytest.mark.parametrize("use_cache", [True, False])
@pytest.mark.parametrize(
    "completion", ["answer-1", "<thinking>This is some COT.</thinking> answer-1"]
)
async def test_freeform(
    use_cache: bool,
    completion: str,
    completions_stub: dict[str, Any],
) -> None:
    enable_cot = "<thinking>" in completion

    q1 = FreeForm(prompt="Hello", enable_cache=use_cache, enable_cot=enable_cot)
    answer, extra = await q1.resolve("test-model")

    assert answer == "answer-1"
    assert completions_stub["count"] == 1
    if enable_cot:
        assert "cot" in extra
        assert extra["cot"] == "<thinking>This is some COT.</thinking>"
    else:
        assert extra == {}

    if use_cache:
        result = q1.get_model_cache("test-model").get(q1.hash())
        assert result is not None
        retr_ans, retr_extra = result
        assert retr_ans == "answer-1"
        assert retr_extra == (
            {"cot": "<thinking>This is some COT.</thinking>"} if enable_cot else {}
        )
    else:
        assert q1.get_model_cache("test-model").get(q1.hash()) is None

    q2 = FreeForm(prompt="Hello", enable_cache=True, enable_cot=enable_cot)
    answer, extra = await q2.resolve("test-model")

    assert completions_stub["count"] == 1 + int(not use_cache)
    assert answer == "answer-1"
    if enable_cot:
        assert "cot" in extra
        assert extra["cot"] == "<thinking>This is some COT.</thinking>"
    else:
        assert extra == {}


@pytest.mark.parametrize(
    "completion",
    [
        '{"foo": 1, "bar": "dummy"}',
        '<thinking>This is some COT.</thinking> {"foo": 1, "bar": "dummy"}',
    ],
)
async def test_extract_json(completion: str, completions_stub: dict[str, Any]) -> None:
    enable_cot = "<thinking>" in completion

    q = ExtractJSON(prompt="Give me a dummy json", enable_cot=enable_cot)
    answer, extra = await q.resolve("test-model")

    assert answer == '{"foo": 1, "bar": "dummy"}'
    assert completions_stub["count"] == 1
    if enable_cot:
        assert "cot" in extra
        assert extra["cot"] == "<thinking>This is some COT.</thinking>"
    else:
        assert extra == {}

    json = await q.json("test-model")

    assert completions_stub["count"] == 1
    assert isinstance(json, dict)
    assert json["foo"] == 1
    assert json["bar"] == "dummy"


@pytest.mark.parametrize(
    "completion", [DUMMY_FN, f"<thinking>This is some COT.</thinking> {DUMMY_FN}"], indirect=True
)
async def test_extract_function(completion: str, completions_stub: dict[str, Any]) -> None:
    enable_cot = "<thinking>" in completion

    q = ExtractFunction(prompt="Write a function", enable_cot=enable_cot)
    code, extra = await q.resolve("test-model")

    assert isinstance(code, str) and code.strip().startswith("def fn(")
    assert completions_stub["count"] == 1
    if enable_cot:
        assert "cot" in extra
        assert extra["cot"] == "<thinking>This is some COT.</thinking>"
    else:
        assert extra == {}

    fn_obj = await q.fn("test-model")
    assert callable(fn_obj)

    df = pd.DataFrame({"count": [1, 2, 3]})
    assert df.apply(fn_obj, axis=1).eq(pd.Series([2, 3, 4])).all()
    assert completions_stub["count"] == 1


@pytest.mark.parametrize("completion", ["1", "<thinking>This is some COT.</thinking> 1"])
async def test_rating(completion: str) -> None:
    enable_cot = "<thinking>" in completion

    q = Rating(
        prompt="Rate the following text: 'This is a test'",
        min_rating=0,
        max_rating=3,
        refusal_threshold=1.0,
        enable_cot=enable_cot,
    )
    answer, extra = await q.resolve("test-model")
    assert 0.0 <= answer <= 1.0

    assert "raw" in extra
    assert "1" in extra["raw"] or " 1" in extra["raw"]
    if enable_cot:
        assert "cot" in extra
        assert extra["cot"] == "<thinking>This is some COT.</thinking>"


@pytest.mark.parametrize("completion", ["x", "<thinking>This is some COT.</thinking> x"])
async def test_next_token(completion: str) -> None:
    enable_cot = "<thinking>" in completion

    q = NextToken(prompt="Predict next token", enable_cot=enable_cot)
    answer, extra = await q.resolve("test-model")
    assert answer == "x"
    assert "probs" in extra
    assert "x" in extra["probs"] or " x" in extra["probs"]
    if enable_cot:
        assert "cot" in extra
        assert extra["cot"] == "<thinking>This is some COT.</thinking>"
