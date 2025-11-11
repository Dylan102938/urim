from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import backoff
import openai
from openai.types.chat import ChatCompletion

from urim.env import OPENROUTER_API_KEY, OPENROUTER_BASE_URL
from urim.logging import get_logger

if TYPE_CHECKING:
    from openai import AsyncOpenAI

logger = get_logger("ai.client")
_PROVIDER_CACHE: dict[str, AsyncOpenAI] = {}


@dataclass(frozen=True)
class ChatResult:
    content: str | None
    raw: dict[str, Any]
    top_tokens: list[TopTokens] | None = None


@dataclass(frozen=True)
class TopTokens:
    token: str
    value: float | None
    top_scores: dict[str, float] | None = None


async def chat_completion(
    model: str,
    messages: list[dict[str, str]],
    convert_to_probs: bool = True,
    *,
    default_base_url: str | None = None,
    default_api_key: str | None = None,
    timeout: float = 60.0,
    **kwargs: Any,
) -> ChatResult:
    client = await _build_client(model, default_base_url, default_api_key, timeout)
    logger.debug(
        "Dispatching chat completion to provider: model=%s, messages=%s, extra_args=%s",
        model,
        messages,
        kwargs,
    )
    resp = await openai_chat_completion(
        client,
        model=model,
        messages=messages,
        **kwargs,
    )
    logger.debug(
        "Received chat completion response for model=%s: message=%s",
        model,
        resp.choices[0].message.content if resp.choices else None,
    )

    top_tokens: list[TopTokens] | None = None
    if (
        resp.choices
        and resp.choices[0].logprobs is not None
        and resp.choices[0].logprobs.content is not None
    ):
        top_tokens = []
        for token_info in resp.choices[0].logprobs.content:
            top_scores_dict: dict[str, float] | None = None
            if token_info.top_logprobs:
                top_scores_dict = {
                    top.token: top.logprob if not convert_to_probs else math.exp(top.logprob)
                    for top in token_info.top_logprobs
                }

            top_tokens.append(
                TopTokens(
                    token=token_info.token,
                    value=token_info.logprob,
                    top_scores=top_scores_dict,
                )
            )

    return ChatResult(
        raw=resp.model_dump(),
        content=resp.choices[0].message.content,
        top_tokens=top_tokens,
    )


async def _build_client(
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 60.0,
) -> AsyncOpenAI:
    from openai import AsyncOpenAI

    from urim.env import collect_openai_keys

    if model in _PROVIDER_CACHE:
        return _PROVIDER_CACHE[model]

    openai_keys = collect_openai_keys(explicit_key=api_key)
    openrouter_keys = [api_key or OPENROUTER_API_KEY]
    custom_keys = [api_key] if api_key else []

    openai_setup = list(
        zip(
            openai_keys,
            [None] * len(openai_keys),
            strict=False,
        )
    )
    openrouter_setup = list(
        zip(
            openrouter_keys,
            [OPENROUTER_BASE_URL] * len(openrouter_keys),
            strict=False,
        )
    )
    custom_setup = list(
        zip(
            custom_keys,
            [base_url or ""] * len(custom_keys),
            strict=False,
        )
    )

    for key, base_url in openai_setup + openrouter_setup + custom_setup:
        client = AsyncOpenAI(api_key=key, base_url=base_url, timeout=timeout)

        logger.debug(
            "Testing LLM client for model=%s using base_url=%s key_suffix=%s.",
            model,
            base_url or "openai",
            key[-4:] if key else "None",
        )
        is_client_valid = await _test_client(client, model)
        if not is_client_valid:
            logger.debug(
                "Skipping client for model=%s using base_url=%s; validation failed.",
                model,
                base_url or "openai",
            )
            continue

        logger.debug(
            "Selected client for model=%s using base_url=%s.",
            model,
            base_url or "openai",
        )
        _PROVIDER_CACHE[model] = client
        return client

    logger.error("No valid client configuration found for model=%s.", model)
    raise RuntimeError("No valid client found")


def _on_backoff(details: Any) -> None:
    exception_details = details["exception"]
    if not str(exception_details).startswith("Connection error."):
        logger.warning(
            "LLM backoff triggered by exception: %s (tries=%s, next_wait=%s).",
            exception_details,
            details.get("tries"),
            details.get("wait"),
        )


@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.InternalServerError,
    ),
    max_value=60,
    factor=1.5,
    on_backoff=_on_backoff,
)
async def openai_chat_completion(client: AsyncOpenAI, *args: Any, **kwargs: Any) -> ChatCompletion:
    return await client.chat.completions.create(*args, **kwargs)


async def _test_client(client: AsyncOpenAI, model: str) -> bool:
    try:
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
            "timeout": 5,
        }
        if not model.startswith("o") and not model.startswith("gpt-5"):
            kwargs["max_tokens"] = 1

        await openai_chat_completion(client, **kwargs)
        return True
    except (
        openai.NotFoundError,
        openai.BadRequestError,
        openai.PermissionDeniedError,
        openai.AuthenticationError,
    ):
        return False
