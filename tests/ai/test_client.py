from __future__ import annotations

import pytest

from urim.ai.client import _PROVIDER_CACHE, chat_completion
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
    keys = collect_openai_keys()

    if len(keys) == 0:
        pytest.skip("No OpenAI keys found")

    result = await chat_completion(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": "Hi there"}],
        max_tokens=1,
        temperature=0.0,
    )

    assert result.content is not None
    assert result.raw is not None
    assert result.top_tokens is None


@requires_llm
async def test_client_reuse_same_instance() -> None:
    keys = collect_openai_keys()

    if len(keys) == 0:
        pytest.skip("No OpenAI keys found")

    test_model = "gpt-4.1-nano"
    _PROVIDER_CACHE.clear()

    result1 = await chat_completion(
        model=test_model,
        messages=[{"role": "user", "content": "First prompt"}],
        max_tokens=1,
        temperature=0.0,
    )

    assert result1.content is not None

    first_client = _PROVIDER_CACHE.get(test_model)
    assert first_client is not None

    result2 = await chat_completion(
        model=test_model,
        messages=[{"role": "user", "content": "Second prompt"}],
        max_tokens=1,
        temperature=0.0,
    )

    assert result2.content is not None

    second_client = _PROVIDER_CACHE.get(test_model)
    assert second_client is not None
    assert first_client is second_client, "Client instance should be reused for the same model"

    _PROVIDER_CACHE.clear()
