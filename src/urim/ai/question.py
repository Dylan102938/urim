from __future__ import annotations

import ast
import hashlib
import inspect
import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generic, TypeAlias, TypeVar

from urim.ai.client import LLM
from urim.ai.prompts import OUTPUT_FUNCTION_SYSTEM, OUTPUT_JSON_SYSTEM
from urim.ai.question_cache import QuestionCache
from urim.env import URIM_HOME

EvalType = TypeVar("EvalType", bound=str | int | float | bool | list | dict)
QuestionResult = tuple[EvalType, dict[str, Any]]

_DEFAULT_CACHE = QuestionCache(cache_dir=URIM_HOME / "questions")


def _to_hashable(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Question):
        return {"question_type": value.__class__.__name__, "hash": value.hash()}
    if isinstance(value, dict):
        return {
            str(k): _to_hashable(v)
            for k, v in sorted(value.items(), key=lambda x: str(x[0]))
        }
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
        cache_dir: str | None = None,
        **kwargs,
    ):
        assert not prompt or not messages, "Cannot specify both prompt and messages"

        self.prompt = prompt
        self.messages = messages
        self.system = system
        self.enable_cache = enable_cache
        self.cache_dir = str(Path(cache_dir) if cache_dir else URIM_HOME / "questions")
        self.kwargs = kwargs

    def __str__(self) -> str:
        wrapper = "{class_name}({insides})"
        insides = ", ".join(
            f"{k}={v}" for k, v in self.__dict__.items() if v is not None
        )
        return wrapper.format(class_name=self.__class__.__name__, insides=insides)

    def __repr__(self) -> str:
        return str(self)

    def hash(self) -> str:
        ignore_fields = {"enable_cache", "cache_dir"}
        semantic = {k: v for k, v in self.__dict__.items() if k not in ignore_fields}
        semantic["__type__"] = self.__class__.__name__

        normalized = _to_hashable(semantic)
        json_str = json.dumps(normalized, sort_keys=True)

        return hashlib.sha256(json_str.encode()).hexdigest()

    def resolve(
        self,
        model: str,
        *,
        executor: ThreadPoolExecutor | None = None,
        **fill_prompt_kwargs,
    ) -> QuestionResult[EvalType]:
        with self.fill_template(**fill_prompt_kwargs) as filled_question:
            if filled_question.enable_cache:
                cached = _DEFAULT_CACHE.read(filled_question, model, executor=executor)
                if cached is not None:
                    return cached

            fresh = filled_question.fetch(model)
            if filled_question.enable_cache:
                _DEFAULT_CACHE.set(filled_question, model, fresh)

            return fresh

    def copy(self) -> Question:
        return self.__class__(
            prompt=self.prompt,
            messages=self.messages,
            system=self.system,
            enable_cache=self.enable_cache,
            cache_dir=self.cache_dir,
        )

    @contextmanager
    def fill_template(self, **templated_kwargs) -> Iterator[Question]:
        question = self.copy()
        if question.prompt is not None:
            question.prompt = question.prompt.format(**templated_kwargs)
        if question.system is not None:
            question.system = question.system.format(**templated_kwargs)
        if question.messages is not None:
            question.messages = [
                {
                    "role": m["role"],
                    "content": m["content"].format(**templated_kwargs),
                }
                for m in question.messages
            ]

        try:
            yield question
        finally:
            pass

    @abstractmethod
    def fetch(self, model: str) -> QuestionResult[EvalType]:
        """Ignores cache and always fetches a fresh response from LLM"""
        ...


JudgeQuestion: TypeAlias = (
    Question[str] | Question[int] | Question[float] | Question[bool]
)


class FreeForm(Question[str]):
    def __init__(
        self,
        *args,
        judges: dict[str, tuple[JudgeQuestion, str]] | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.judges = judges

    def fetch(self, model: str) -> QuestionResult[str]:
        if self.messages is None:
            assert self.prompt is not None, "Must provide either messages or prompt"
            messages = [{"role": "user", "content": self.prompt}]
            if self.system is not None:
                messages.insert(0, {"role": "system", "content": self.system})
        else:
            messages = self.messages

        completion = LLM().chat_completion(model, messages=messages, **self.kwargs)

        judge_results: dict[str, str | int | float | bool] = {}
        if self.judges is not None:
            for judge_name, (judge, judge_model) in self.judges.items():
                judge_result, _ = judge.resolve(
                    judge_model,
                    prompt=self.prompt,
                    messages=self.messages,
                )
                judge_results[judge_name] = judge_result

        return (completion.content or "", judge_results)


class ExtractJSON(FreeForm):
    def __init__(
        self,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        system: str | None = None,
        enable_cache: bool = True,
        cache_dir: str | None = None,
        use_json_system: bool = True,
        **kwargs,
    ):
        resolved_system = OUTPUT_JSON_SYSTEM if use_json_system else system
        super().__init__(
            prompt, messages, resolved_system, enable_cache, cache_dir, **kwargs
        )

    def json(self, model: str) -> dict:
        result, _ = self.resolve(model)
        return json.loads(result)


class ExtractFunction(FreeForm):
    def __init__(
        self,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        system: str | None = None,
        enable_cache: bool = True,
        cache_dir: str | None = None,
        use_function_system: bool = True,
        **kwargs,
    ):
        resolved_system = OUTPUT_FUNCTION_SYSTEM if use_function_system else system
        super().__init__(
            prompt, messages, resolved_system, enable_cache, cache_dir, **kwargs
        )

    def fn(self, model: str) -> Callable[..., Any]:
        result, _ = super().resolve(model)

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


class Rating(Question[float]):
    def __init__(
        self,
        *args,
        min_rating: float | None = None,
        max_rating: float | None = None,
        refusal_threshold: float = 0.75,
        top_logprobs: int = 20,
        **kwargs,
    ):
        super().__init__(*args, top_logprobs=top_logprobs, **kwargs)
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.refusal_threshold = refusal_threshold

    def fetch(self, model: str) -> QuestionResult[float]:
        completion = LLM().chat_completion(
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
