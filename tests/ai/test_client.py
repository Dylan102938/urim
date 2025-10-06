from __future__ import annotations

import pytest

from urim.ai.client import LLM
from urim.env import collect_openai_keys

requires_llm = pytest.mark.requires_llm


def test_collect_openai_keys_order_and_dedupe(monkeypatch: pytest.MonkeyPatch) -> None:
    for i in range(0, 10):
        monkeypatch.delenv(f"OPENAI_API_KEY_{i}", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    monkeypatch.setenv("OPENAI_API_KEY", "A")
    monkeypatch.setenv("OPENAI_API_KEY_0", "B")
    monkeypatch.setenv("OPENAI_API_KEY_1", "C")
    monkeypatch.setenv("OPENAI_API_KEY_2", "A")

    keys = collect_openai_keys()
    assert keys == ["A", "B", "C"]


@requires_llm
async def test_client_calls_openai() -> None:
    client = LLM()
    keys = collect_openai_keys()

    if len(keys) == 0:
        pytest.skip("No OpenAI keys found")

    client = LLM()
    result = await client.chat_completion(
        model="gpt-4.1-nano",
        prompt="Hi there",
        max_tokens=1,
        temperature=0.0,
    )

    assert result.content is not None
    assert result.raw is not None
    assert result.top_tokens is None
