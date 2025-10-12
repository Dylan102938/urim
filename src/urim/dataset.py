from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable, Hashable, Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from urim.ai.question import Question

QuestionFactory = Callable[["pd.Series"], "Question"] | str

PRESET_FAST = "gpt-4.1-mini"
PRESET_BALANCED = "gpt-4.1"
PRESET_THOROUGH = "gpt-5"

GENERATION_SEMAPHORE = asyncio.Semaphore(100)


async def extract_op_kwargs(
    template: str,
    model: str,
    *,
    curr_df: pd.DataFrame | None = None,
    question_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    from urim.ai.question import ExtractJSON

    if curr_df is not None:
        if len(curr_df) < 5:
            rows = curr_df
        else:
            rows = curr_df.iloc[:: len(curr_df) // 5]
        prompt = template.format(**{"head": rows.head(), **kwargs})
    else:
        prompt = template.format(**kwargs)

    for _ in range(3):
        question: ExtractJSON
        try:
            question = ExtractJSON(prompt, **(question_kwargs or {}))
            return await question.json(model)
        except Exception:
            question.remove_from_cache(model)

    raise Exception("Failed to extract kwargs")


async def extract_op_fn(
    template: str,
    model: str,
    *,
    curr_df: pd.DataFrame | None = None,
    question_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    from urim.ai.question import ExtractFunction

    if curr_df is not None:
        if len(curr_df) < 5:
            rows = curr_df
        else:
            rows = curr_df.iloc[:: len(curr_df) // 5]
        prompt = template.format(**{"head": rows.head(), **kwargs})
    else:
        prompt = template.format(**kwargs)

    for _ in range(3):
        question: ExtractFunction
        try:
            question = ExtractFunction(prompt, **(question_kwargs or {}))
            return await question.fn(model)
        except Exception:
            question.remove_from_cache(model)

    raise Exception("Failed to extract fn")


class Dataset:
    def __init__(self, dataset: pd.DataFrame | str | Path, **kwargs: Any):
        import pandas as pd
        from datasets import load_dataset

        self._df: pd.DataFrame
        if isinstance(dataset, str | Path):
            path = Path(dataset)
            if path.exists():
                self._df = pd.read_json(path, orient="records", lines=True, **kwargs)
            elif isinstance(dataset, str):
                self._df = load_dataset(dataset, **kwargs).to_pandas()
            else:
                msg = f"Dataset path {path} does not exist"
                raise FileNotFoundError(msg)
        else:
            self._df = dataset

        self._init_dataset_kwargs = kwargs

    def __hash__(self) -> int:
        from blake3 import blake3
        from pandas.util import hash_pandas_object

        hasher = blake3()
        df = _normalize_dataframe(self._df)

        for key, mul in [
            ("0123456789ABCDEF", 0x9E3779B97F4A7C15),
            ("23456789ABCDEFGH", 0xBF58476D1CE4E5B9),
        ]:
            h = hash_pandas_object(df, index=False, hash_key=key).to_numpy(np.uint64)
            s = int(h.sum(dtype=np.uint64))
            sw = int((h * np.uint64(mul)).sum(dtype=np.uint64))

            hasher.update(s.to_bytes(8, "big", signed=False))
            hasher.update(sw.to_bytes(8, "big", signed=False))

        return int.from_bytes(hasher.digest(length=16), "big")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Dataset):
            return False

        return hash(self) == hash(other)

    def df(self) -> pd.DataFrame:
        return self._df

    def to_json(self, out_path: str | Path) -> None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        self._df.to_json(out_path, orient="records", lines=True)

    def sample(
        self,
        n: int | None = None,
        frac: float | None = None,
        *,
        inplace: bool = False,
        **kwargs: Any,
    ) -> Dataset:
        if frac is not None and frac > 1:
            frac = frac / 100

        df = self._df.sample(n=n, frac=frac, **kwargs)
        return self._maybe_inplace(df, inplace=inplace)

    async def rename(
        self,
        columns: Mapping[Any, Hashable] | None = None,
        hint: str | None = None,
        model: str = PRESET_FAST,
        *,
        inplace: bool = False,
        question_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Dataset:
        from urim.ai.prompts import DATASET_RENAME_PROMPT

        if columns is None:
            assert hint is not None
            kwargs = {
                **kwargs,
                "columns": await extract_op_kwargs(
                    DATASET_RENAME_PROMPT,
                    columns=", ".join(self._df.columns),
                    scheme=hint,
                    model=model,
                    curr_df=self._df,
                    question_kwargs=question_kwargs,
                ),
            }
        else:
            kwargs = {
                "columns": columns,
                **kwargs,
            }

        df = self._df.rename(**kwargs)
        return self._maybe_inplace(df, inplace=inplace)

    async def drop(
        self,
        columns: list[str] | str | None = None,
        hint: str | None = None,
        model: str = PRESET_FAST,
        *,
        inplace: bool = False,
        question_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Dataset:
        from urim.ai.prompts import (
            DATASET_DROP_NO_HINT_PROMPT,
            DATASET_DROP_WITH_HINT_PROMPT,
        )

        assert columns is not None or hint is not None, "Must provide either columns or hint"

        if columns is None:
            add_kwargs = await extract_op_kwargs(
                DATASET_DROP_WITH_HINT_PROMPT if hint else DATASET_DROP_NO_HINT_PROMPT,
                columns=", ".join(self._df.columns),
                scheme=hint,
                model=model,
                curr_df=self._df,
                question_kwargs=question_kwargs,
            )
            kwargs = {
                **kwargs,
                **add_kwargs,
            }
        else:
            kwargs = {
                "columns": columns,
                **kwargs,
            }

        df = self._df.drop(**kwargs)
        return self._maybe_inplace(df, inplace=inplace)

    async def filter(
        self,
        fn: Callable[[pd.Series], bool] | None = None,
        hint: str | None = None,
        model: str = PRESET_BALANCED,
        *,
        inplace: bool = False,
        question_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Dataset:
        from urim.ai.prompts import DATASET_FILTER_PROMPT

        assert fn is not None or hint is not None, "Must provide either fn or hint"

        if fn is None:
            fn = await extract_op_fn(
                DATASET_FILTER_PROMPT,
                columns=", ".join(self._df.columns),
                scheme=hint,
                model=model,
                curr_df=self._df,
                question_kwargs=question_kwargs,
            )

        df = self._df.loc[self._df.apply(fn, axis=1, **kwargs), :]
        return self._maybe_inplace(df, inplace=inplace)

    async def apply(
        self,
        fn: Callable[[pd.Series], Any] | None = None,
        column: str | None = None,
        hint: str | None = None,
        model: str = PRESET_BALANCED,
        *,
        inplace: bool = False,
        question_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Dataset:
        from urim.ai.prompts import DATASET_APPLY_PROMPT

        assert fn is not None or hint is not None, "Must provide either fn or hint"

        if fn is None:
            wrapper_fn = await extract_op_fn(
                DATASET_APPLY_PROMPT,
                columns=", ".join(self._df.columns),
                column_hint=f"The column's name should be {column}." if column is not None else "",
                scheme=hint,
                model=model,
                curr_df=self._df,
                question_kwargs=question_kwargs,
            )
            column, fn = wrapper_fn()

        df = self._df.copy()
        df[column] = df.apply(fn, axis=1, **kwargs)

        return self._maybe_inplace(df, inplace=inplace)

    async def reduce(
        self,
        by: str | list[str] | None = None,
        agg: dict[str, Literal["mean", "min", "max", "sum", "count"]] | None = None,
        hint: str | None = None,
        model: str = PRESET_BALANCED,
        *,
        inplace: bool = False,
        question_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Dataset:
        import json

        from urim.ai.prompts import DATASET_REDUCE_PROMPT, DATASET_REDUCE_WITH_HINT_PROMPT

        if by is None or agg is None:
            counts_series = self._df.nunique(dropna=False)
            counts_dict = {str(column): int(count) for column, count in counts_series.items()}
            add_kwargs = await extract_op_kwargs(
                DATASET_REDUCE_WITH_HINT_PROMPT if hint else DATASET_REDUCE_PROMPT,
                scheme=hint,
                model=model,
                columns=", ".join(self._df.columns),
                unique_summary=json.dumps(counts_dict, indent=2),
                column_categories=counts_dict,
                curr_df=self._df,
                question_kwargs=question_kwargs,
            )

            assert "groupby" in add_kwargs and "agg" in add_kwargs

            df = self._df.groupby(**add_kwargs["groupby"], **kwargs).agg(add_kwargs["agg"])
        else:
            df = self._df.groupby(by, **kwargs).agg(agg)

        df = df.reset_index()
        return self._maybe_inplace(df, inplace=inplace)

    async def generate(
        self,
        question_col: str | None = None,
        messages_col: str | None = None,
        system_col: str | None = None,
        out_col: str | None = None,
        salt_col: str | None = None,
        question_type: type[Question] | None = None,
        model: str = PRESET_BALANCED,
        *,
        judges: dict[str, QuestionFactory] | None = None,
        inplace: bool = False,
        **question_kwargs: Any,
    ) -> Dataset:
        from urim.ai.question import FreeForm, Rating

        ### Generate questions using dataframes and defaults ###

        question_col = question_col or "question"
        messages_col = messages_col or "messages"
        system_col = system_col or "system"
        out_col = out_col or "answer"
        salt_col = salt_col or "salt"

        input_col = (question_col in self._df.columns and question_col) or (
            messages_col in self._df.columns and messages_col
        )

        assert input_col in self._df.columns

        input_iter = self._df[input_col].to_list()
        questions: list[Question] = []
        question_type = question_type or FreeForm
        for i, inp in enumerate(input_iter):
            common_system = question_kwargs.pop("system", None)
            system = (
                str(self._df.iloc[i][system_col])
                if system_col in self._df.columns
                else common_system
            )
            common_salt = question_kwargs.pop("salt", None)
            salt = str(self._df.iloc[i][salt_col]) if salt_col in self._df.columns else common_salt
            question_input: dict[str, Any]
            if input_col == messages_col:
                question_input = {"messages": inp}
            else:
                question_input = {"prompt": inp}

            questions.append(
                question_type(
                    system=system,
                    salt=salt,
                    **question_input,
                    **question_kwargs,
                )
            )

        ### Resolve questions and add to dataframe ###

        results = await self._resolve_questions(model, questions)
        df = self._df.copy()
        df[out_col] = [result[0] for result in results]

        extra_columns: dict[str, list[Any]] = defaultdict(list)
        for extra in [result[1] for result in results]:
            for k, v in extra.items():
                extra_columns[k].append(v)

        for k, v in extra_columns.items():
            df[k] = v

        if judges is None:
            return self._maybe_inplace(df, inplace=inplace)

        ### Generate judge questions ###

        judge_questions: dict[str, list[Question]] = defaultdict(list)
        for _, row in df.iterrows():
            for k, factory in judges.items():
                judge_questions[k].append(
                    Rating(prompt=factory.format(**row.to_dict()), **question_kwargs)
                    if isinstance(factory, str)
                    else factory(row)
                )

        ### Resolve judge questions and add to dataframe ###

        judge_results = await self._resolve_judge_questions(model, judge_questions)
        for k, results in judge_results.items():
            df[k] = [result[0] for result in results]
            extra_columns = defaultdict(list)

            for extra in [result[1] for result in results]:
                for extra_k, v in extra.items():
                    extra_columns[extra_k].append(v)

            for extra_k, v in extra_columns.items():
                df[f"{k}_{extra_k}"] = v

        return self._maybe_inplace(df, inplace=inplace)

    @classmethod
    def concatenate(cls, *datasets: Dataset, **kwargs: Any) -> Dataset:
        return cls(pd.concat([ds._df for ds in datasets], axis=0, **kwargs))

    def _maybe_inplace(self, df: pd.DataFrame, *, inplace: bool) -> Dataset:
        if inplace:
            self._df = df
            return self
        else:
            return Dataset(df)

    async def _resolve_questions(
        self,
        model: str,
        questions: list[Question],
    ) -> list[tuple[Any, dict]]:
        from tqdm.auto import tqdm

        from urim.ai.question import Question

        num_questions = len(questions)
        results: list[tuple[Any, dict]] = [("", {}) for _ in range(num_questions)]

        async def _run_question(idx: int, question: Question) -> None:
            async with GENERATION_SEMAPHORE:
                try:
                    results[idx] = await question.resolve(model, flush_cache=False)
                except Exception as exc:
                    results[idx] = ("", {"error": str(exc)})

        tasks = [
            asyncio.create_task(_run_question(idx, question))
            for idx, question in enumerate(questions)
        ]

        for task in tqdm(
            asyncio.as_completed(tasks),
            total=num_questions,
            leave=False,
            desc=f"{model} - resolving questions",
        ):
            await task

        await Question.flush_cache(model)

        return results

    async def _resolve_judge_questions(
        self,
        model: str,
        judge_questions: dict[str, list[Question]],
    ) -> dict[str, list[tuple[Any, dict]]]:
        from tqdm.auto import tqdm

        from urim.ai.question import Question

        num_judge_questions = sum(len(questions) for questions in judge_questions.values())
        judge_results: dict[str, list[tuple[Any, dict]]] = {
            k: [("", {}) for _ in range(len(questions))] for k, questions in judge_questions.items()
        }

        async def _run_judge(label: str, idx: int, question: Question) -> None:
            async with GENERATION_SEMAPHORE:
                try:
                    judge_results[label][idx] = await question.resolve(
                        model,
                        flush_cache=False,
                    )
                except Exception as exc:
                    judge_results[label][idx] = ("", {"error": str(exc)})

        judge_tasks = [
            asyncio.create_task(_run_judge(label, idx, question))
            for label, questions in judge_questions.items()
            for idx, question in enumerate(questions)
        ]

        for task in tqdm(
            asyncio.as_completed(judge_tasks),
            total=num_judge_questions,
            leave=False,
            desc=f"{model} - resolving judge questions",
        ):
            await task

        await Question.flush_cache(model)

        return judge_results


def _normalize_cell(value: Any) -> Hashable | None:
    import datetime as dt
    import math

    if isinstance(value, np.generic):
        value = value.item()

    if value is None:
        return None

    if value is pd.NA:
        return None

    if isinstance(value, np.bool_ | bool):
        return bool(value)

    if isinstance(value, np.integer | int) and not isinstance(value, bool):
        return int(value)

    if isinstance(value, Real) and not isinstance(value, bool):
        float_value = float(value)
        if math.isnan(float_value):
            return None
        return float_value

    if isinstance(value, str | bytes):
        return value

    if isinstance(value, dt.datetime | dt.date | dt.time):
        return value.isoformat()

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, pd.Timedelta):
        return value.isoformat()

    if isinstance(value, np.datetime64 | dt.datetime):
        return pd.Timestamp(value).isoformat()

    if isinstance(value, np.timedelta64 | dt.timedelta):
        return pd.Timedelta(value).isoformat()

    if isinstance(value, pd.Series):
        return (
            "series",
            tuple(_normalize_cell(v) for v in value.tolist()),
        )

    if isinstance(value, pd.DataFrame):
        normalized_df = _normalize_dataframe(value)
        return (
            "dataframe",
            tuple(tuple(row) for row in normalized_df.to_numpy().tolist()),
        )

    if isinstance(value, np.ndarray):
        return (
            "ndarray",
            tuple(_normalize_cell(v) for v in value.tolist()),
        )

    if isinstance(value, Mapping):
        items = tuple(
            (str(k), _normalize_cell(v))
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        )
        return ("map", items)

    if isinstance(value, set | frozenset):
        normalized_members = [_normalize_cell(v) for v in value]
        sorted_members = tuple(sorted(normalized_members, key=lambda elem: repr(elem)))
        return ("set", sorted_members)

    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return ("seq", tuple(_normalize_cell(v) for v in value))

    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:  # pragma: no cover - fallback to repr
            pass

    if hasattr(value, "__dict__"):
        attrs = tuple(
            (str(k), _normalize_cell(v))
            for k, v in sorted(vars(value).items(), key=lambda item: str(item[0]))
        )
        return ("object", attrs)

    return repr(value)


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized_cols = sorted(df.columns, key=lambda col: str(col))
    normalized_df = df.reindex(columns=normalized_cols)
    return normalized_df.map(_normalize_cell)
