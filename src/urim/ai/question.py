from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from urim.env import storage_subdir

if TYPE_CHECKING:
    from urim.store.disk_store import DiskStore

EvalType = TypeVar("EvalType", bound=str | int | float | bool | list | dict)
QuestionResult = tuple[EvalType, dict[str, Any]]

_caches: dict[str, DiskStore] = {}


def _to_hashable(value: Any) -> Any:
    import inspect

    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Question):
        return {"question_type": value.__class__.__name__, "hash": value.hash()}
    if isinstance(value, dict):
        return {str(k): _to_hashable(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, list | tuple | set):
        return [_to_hashable(v) for v in value]
    if callable(value):
        try:
            src = inspect.getsource(value)
        except Exception:
            src = getattr(value, "__qualname__", repr(value))
        return {"callable": src}

    return value


class Question(ABC, Generic[EvalType]):
    def __init__(
        self,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        system: str | None = None,
        enable_cache: bool = True,
        **kwargs: Any,
    ) -> None:
        assert not prompt or not messages, "Cannot specify both prompt and messages"

        self.prompt = prompt
        self.messages = messages
        self.system = system
        self.enable_cache = enable_cache
        self.kwargs = kwargs

    def __str__(self) -> str:
        wrapper = "{class_name}({insides})"
        insides = ", ".join(f"{k}={v}" for k, v in self.__dict__.items() if v is not None)
        return wrapper.format(class_name=self.__class__.__name__, insides=insides)

    def __repr__(self) -> str:
        return str(self)

    def hash(self) -> str:
        import hashlib
        import json

        ignore_fields = {"enable_cache"}
        semantic = {k: v for k, v in self.__dict__.items() if k not in ignore_fields}
        semantic["__type__"] = self.__class__.__name__

        normalized = _to_hashable(semantic)
        json_str = json.dumps(normalized, sort_keys=True)

        return hashlib.sha256(json_str.encode()).hexdigest()

    async def resolve(self, model: str, *, flush_cache: bool = True) -> QuestionResult[EvalType]:
        """Resolves the question with a response from the specified model.

        Cache is automatically flushed when the question is resolved. If you don't want this to
        happen, set `flush_cache` to `False` and manually flush with `Question.flush_cache(model)`
        to persist the result.
        """

        result: QuestionResult[EvalType] | None = None
        cache: DiskStore | None = None
        if self.enable_cache:
            cache = self.get_model_cache(model)
            result = cache.get(self.hash())

        if result is None:
            result = await self.fetch(model)
            if self.enable_cache and cache is not None:
                cache.put(self.hash(), result)
                if flush_cache:
                    await asyncio.to_thread(cache.flush)

        return result

    def resolve_sync(self, model: str, *, flush_cache: bool = True) -> QuestionResult[EvalType]:
        return asyncio.run(self.resolve(model, flush_cache=flush_cache))

    def get_model_cache(self, model: str) -> DiskStore:
        from urim.store.disk_store import DiskStore

        if model not in _caches:
            _caches[model] = DiskStore(storage_subdir("questions") / f"{model}.jsonl")

        return _caches[model]

    def remove_from_cache(self, model: str) -> None:
        if model not in _caches:
            return

        _caches[model].remove(self.hash())

    @classmethod
    async def flush_cache(cls, model: str) -> None:
        if model not in _caches:
            return

        await asyncio.to_thread(_caches[model].flush)

    @abstractmethod
    async def fetch(self, model: str) -> QuestionResult[EvalType]:
        """Ignores cache and always fetches a fresh response from LLM"""
        ...


class FreeForm(Question[str]):
    async def fetch(self, model: str) -> QuestionResult[str]:
        from urim.ai.client import LLM

        messages = self.resolve_to_messages()
        completion = await LLM().chat_completion(model, messages=messages, **self.kwargs)
        return (completion.content or "", {})

    def resolve_to_messages(self) -> list[dict]:
        if self.messages is None:
            assert self.prompt is not None, "Must provide either messages or prompt"
            messages = [{"role": "user", "content": self.prompt}]
            if self.system is not None:
                messages.insert(0, {"role": "system", "content": self.system})

            return messages

        return self.messages


class ExtractJSON(FreeForm):
    def __init__(
        self,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        system: str | None = None,
        enable_cache: bool = True,
        use_json_system: bool = True,
        **kwargs: Any,
    ) -> None:
        from urim.ai.prompts import OUTPUT_JSON_SYSTEM

        resolved_system = OUTPUT_JSON_SYSTEM if use_json_system else system
        super().__init__(prompt, messages, resolved_system, enable_cache, **kwargs)

    async def json(self, model: str) -> dict:
        import json

        result, _ = await self.resolve(model)
        if result.startswith("```json") and result.endswith("```"):
            result = result[len("```json") : -len("```")]

        return json.loads(result)

    def json_sync(self, model: str) -> dict:
        return asyncio.run(self.json(model))


class ExtractFunction(FreeForm):
    def __init__(
        self,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        system: str | None = None,
        enable_cache: bool = True,
        use_function_system: bool = True,
        **kwargs: Any,
    ) -> None:
        from urim.ai.prompts import OUTPUT_FUNCTION_SYSTEM

        resolved_system = OUTPUT_FUNCTION_SYSTEM if use_function_system else system
        super().__init__(prompt, messages, resolved_system, enable_cache, **kwargs)

    async def fn(self, model: str) -> Callable[..., Any]:
        import ast
        import inspect

        result, _ = await super().resolve(model)
        if result.startswith("```python") and result.endswith("```"):
            result = result[len("```python") : -len("```")]

        fn_obj = None
        fn_name: str | None = None

        tree = ast.parse(result)
        names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
        fn_name = names[0] if names else None

        namespace: dict[str, Any] = {}
        exec(result, namespace)

        if fn_name and inspect.isfunction(namespace.get(fn_name)):
            fn_obj = namespace[fn_name]
        else:
            raise ValueError("No function name found")

        return fn_obj

    def fn_sync(self, model: str) -> Callable[..., Any]:
        return asyncio.run(self.fn(model))


class Rating(Question[float]):
    def __init__(
        self,
        *args: Any,
        min_rating: float | None = None,
        max_rating: float | None = None,
        refusal_threshold: float = 0.75,
        top_logprobs: int = 20,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, top_logprobs=top_logprobs, **kwargs)
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.refusal_threshold = refusal_threshold

    async def fetch(self, model: str) -> QuestionResult[float]:
        from urim.ai.client import LLM

        completion = await LLM().chat_completion(
            model,
            messages=self.messages,
            prompt=self.prompt,
            **self.kwargs,
            max_tokens=1,
            temperature=0.0,
            logprobs=True,
            convert_to_probs=True,
        )

        assert (
            completion.top_tokens is not None
        ), "Looks like your provider doesn't support logprobs"

        score = self._agg_score(completion.top_tokens)
        assert score is not None, "No valid score found"

        return (score, {"raw": completion.top_tokens})

    def _agg_score(self, scores: dict[str, float]) -> float | None:
        total = 0.0
        sum_ = 0.0
        for key, val in scores.items():
            try:
                int_key = int(key)
            except ValueError:
                continue
            if self.min_rating and self.min_rating > int_key:
                continue
            if self.max_rating and self.max_rating < int_key:
                continue

            sum_ += int_key * val
            total += val

        refusal_weight = 1 - total
        if refusal_weight >= self.refusal_threshold:
            return None

        return sum_ / total


class NextToken(Question):
    def __init__(
        self,
        *args: Any,
        top_logprobs: int = 20,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, top_logprobs=top_logprobs, **kwargs)

    async def fetch(self, model: str) -> QuestionResult[str]:
        from urim.ai.client import LLM

        completion = await LLM().chat_completion(
            model,
            messages=self.messages,
            prompt=self.prompt,
            **self.kwargs,
            max_tokens=1,
            temperature=0.0,
            logprobs=True,
            convert_to_probs=True,
        )

        return completion.content or "", {"probs": completion.top_tokens or {}}
