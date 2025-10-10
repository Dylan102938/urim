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


@dataclass(frozen=True)
class ChatResult:
    content: str | None
    raw: dict[str, Any]
    top_tokens: dict | None = None


class LLM:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout

    async def chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]] | None = None,
        prompt: str | None = None,
        convert_to_probs: bool = True,
        **kwargs: Any,
    ) -> ChatResult:
        if messages is None and not prompt:
            raise ValueError("Either messages or prompt must be provided")

        final_messages = messages if messages is not None else _prompt_to_messages(prompt or "")
        logger.debug(
            "LLM.chat_completion request: model=%s, messages=%s",
            model,
            final_messages,
        )
        return await self._request(
            model=model,
            messages=final_messages,
            convert_to_probs=convert_to_probs,
            **kwargs,
        )

    async def _build_client(self, model: str) -> AsyncOpenAI:
        from openai import AsyncOpenAI

        from urim.env import collect_openai_keys

        openai_keys = collect_openai_keys(explicit_key=self.api_key)
        openrouter_keys = [self.api_key or OPENROUTER_API_KEY]
        custom_keys = [self.api_key] if self.api_key else []

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
                [self.base_url or ""] * len(custom_keys),
                strict=False,
            )
        )

        for key, base_url in openai_setup + openrouter_setup + custom_setup:
            client = AsyncOpenAI(
                api_key=key,
                base_url=base_url,
                timeout=self.timeout,
            )

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
            return client

        logger.error("No valid client configuration found for model=%s.", model)
        raise RuntimeError("No valid client found")

    async def _request(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        convert_to_probs: bool = True,
        **kwargs: Any,
    ) -> ChatResult:
        client = await self._build_client(model)
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

        logprobs_dict: dict[str, float] | None = None
        if resp.choices[0].logprobs is not None:
            logprobs = resp.choices[0].logprobs.content[0].top_logprobs  # type: ignore
            logprobs_dict = {}
            for el in logprobs:
                logprobs_dict[el.token] = (
                    el.logprob if not convert_to_probs else math.exp(el.logprob)
                )

        return ChatResult(
            raw=resp.model_dump(),
            content=resp.choices[0].message.content,
            top_tokens=logprobs_dict,
        )


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


def _prompt_to_messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]
