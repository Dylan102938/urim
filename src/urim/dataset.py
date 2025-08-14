import hashlib
import shutil
from collections import defaultdict
from collections.abc import Callable, Hashable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd  # type: ignore
from datasets import Dataset as HFDataset  # type: ignore
from datasets import load_dataset
from typing_extensions import Self

from urim.ai.prompts import (
    DATASET_APPLY_PROMPT,
    DATASET_CONCAT_HINT_PROMPT,
    DATASET_CONCAT_NO_HINT_PROMPT,
    DATASET_DROP_NO_HINT_PROMPT,
    DATASET_DROP_WITH_HINT_PROMPT,
    DATASET_FILTER_PROMPT,
    DATASET_MERGE_HINT_PROMPT,
    DATASET_MERGE_NO_HINT_PROMPT,
    DATASET_RENAME_PROMPT,
    GENERATE_DESCRIBE_CHAIN_PROMPT,
)
from urim.ai.question import ExtractFunction, ExtractJSON, FreeForm, Question
from urim.env import URIM_HOME
from urim.logging_utils import RichLogger

Axis = int | Literal["index", "columns", "rows"]
Renamer = Mapping[Any, Hashable]


def get_hf_dataset_local_id(**kwargs) -> str:
    sorted_keys = sorted(kwargs.keys())
    semantic_id = "_".join(f"{k}={kwargs[k]}" for k in sorted_keys)
    serialized_id = hashlib.sha256(semantic_id.encode()).hexdigest()

    return serialized_id


def get_dataset_local_id(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read(200 * 1024)).hexdigest()


class Dataset:
    def __init__(self, input_path: str | None = None, df: pd.DataFrame | None = None):
        assert (
            input_path is not None or df is not None
        ), "Must provide either a path or a DataFrame"
        self.path = input_path
        self._df = df

    def df(self) -> pd.DataFrame:
        if self._df is None:
            assert self.path is not None
            self._df = pd.read_json(self.path, lines=True)

        return self._df

    def to_json(self, output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self.df().to_json(output_path, orient="records", lines=True)

    def sample(self, n: int | None = None, frac: float | None = None, **kwargs) -> Self:
        assert n is not None or frac is not None, "Must provide either n or frac"

        self._df = self.df().sample(n=n, frac=frac, **kwargs)
        return self

    def rename(
        self,
        columns: Renamer | None = None,
        hint: str | None = None,
        model: str = "gpt-4.1-mini",
    ) -> Self:
        assert columns is not None or hint is not None
        df = self.df()
        if columns is None:
            assert hint is not None, "Must provide a hint if no columns are provided"
            question = ExtractJSON(
                prompt=DATASET_RENAME_PROMPT.format(
                    columns=", ".join(df.columns),
                    head=df.head(5),
                    scheme=hint,
                )
            )
            columns = question.json(model)

        self._df = df.rename(columns=columns)
        return self

    def drop(
        self,
        columns: list[str] | None = None,
        hint: str | None = None,
        model: str = "gpt-4.1-mini",
    ) -> Self:
        assert (
            columns is not None or hint is not None
        ), "Must provide either columns or hint"

        df = self.df()
        if columns is None:
            drop_prompt = (
                DATASET_DROP_NO_HINT_PROMPT
                if hint is None
                else DATASET_DROP_WITH_HINT_PROMPT
            )
            question = ExtractJSON(
                prompt=drop_prompt.format(
                    columns=", ".join(df.columns),
                    head=df.head(5),
                    scheme=hint,
                )
            )
            columns = question.json(model).get("columns", [])

        self._df = df.drop(columns=columns)
        return self

    def filter(
        self,
        fn: Callable[[pd.Series], bool] | None = None,
        hint: str | None = None,
        model: str = "gpt-4.1",
    ) -> Self:
        assert fn is not None or hint is not None, "Must provide either fn or hint"

        df = self.df()
        if fn is None:
            assert hint is not None, "Must provide a hint if no function is provided"
            question = ExtractFunction(
                DATASET_FILTER_PROMPT.format(
                    columns=", ".join(df.columns),
                    head=df.head(5),
                    scheme=hint,
                )
            )
            fn = question.fn(model)

        self._df = df[df.apply(fn, axis=1)]
        return self

    def apply(
        self,
        fn: Callable[[pd.Series], Any] | None = None,
        column: str | None = None,
        hint: str | None = None,
        model: str = "gpt-4.1",
    ) -> Self:
        assert fn is not None or hint is not None, "Must provide either fn or hint"

        df = self.df()
        if fn is None:
            assert hint is not None, "Must provide a hint if no function is provided"
            question = ExtractFunction(
                prompt=DATASET_APPLY_PROMPT.format(
                    columns=", ".join(df.columns),
                    head=df.head(5),
                    scheme=hint,
                    column_hint=(
                        f"The column's name should be {column}."
                        if column is not None
                        else ""
                    ),
                )
            )
            wrapper_fn = question.fn(model)
            column, fn = wrapper_fn()

        assert column is not None, "Must provide a column name"
        df[column] = df.apply(fn, axis=1)  # type: ignore

        return self

    def merge(
        self,
        other: Self,
        on: str | list[str] | None = None,
        left_on: str | list[str] | None = None,
        right_on: str | list[str] | None = None,
        how: Literal["left", "right", "inner", "outer", "cross"] | None = None,
        hint: str | None = None,
        model: str = "gpt-4.1",
        **kwargs,
    ) -> Self:
        df, other_df = self.df(), other.df()
        ons_none = on is None and left_on is None and right_on is None
        merge_args = {
            "on": on,
            "left_on": left_on,
            "right_on": right_on,
            "how": how,
        }
        if hint is not None:
            merge_hint = ""
            if hint is not None:
                merge_hint += hint
            if ons_none:
                if merge_hint:
                    merge_hint += " "

                defined_args = [
                    f"{k}={v}" for k, v in merge_args.items() if v is not None
                ]
                merge_hint += (
                    f"I've already set the following args: {', '.join(defined_args)}"
                )

            merge_template = (
                DATASET_MERGE_NO_HINT_PROMPT
                if hint is None
                else DATASET_MERGE_HINT_PROMPT
            )

            question = ExtractJSON(
                prompt=merge_template.format(
                    columns=", ".join(df.columns),
                    other_columns=", ".join(other_df.columns),
                    head=df.head(5),
                    other_head=other_df.head(5),
                    scheme=merge_hint,
                )
            )

            json = question.json(model)
            on = json.get("on")
            left_on = json.get("left_on")
            right_on = json.get("right_on")
            how = json.get("how", "left")

        assert how is not None

        self._df = df.merge(
            other_df,
            on=on,
            left_on=left_on,
            right_on=right_on,
            how=how,
            **kwargs,
        )

        return self

    def concat(
        self,
        other: Self,
        hint: str | None = None,
        model: str = "gpt-4.1",
    ) -> Self:
        df, other_df = self.df(), other.df()
        columns = set(df.columns)
        other_columns = set(other_df.columns)
        col_diff = columns - other_columns

        if len(col_diff) > 0 or hint is not None:
            concat_prompt = (
                DATASET_CONCAT_NO_HINT_PROMPT
                if hint is None
                else DATASET_CONCAT_HINT_PROMPT
            )
            question = ExtractJSON(
                prompt=concat_prompt.format(
                    columns=", ".join(df.columns),
                    other_columns=", ".join(other_df.columns),
                    head=df.head(5),
                    other_head=other_df.head(5),
                    scheme=hint,
                )
            )
            json: dict[str, str] = question.json(model)
            df1_columns, df2_columns = {}, {}
            for k, v in json.items():
                if k.startswith("df1_"):
                    df1_columns[k.removeprefix("df1_")] = v
                elif k.startswith("df2_"):
                    df2_columns[k.removeprefix("df2_")] = v

            df = df.rename(columns=df1_columns)
            other_df = other_df.rename(columns=df2_columns)

        self._df = pd.concat([df, other_df], axis=0)

        return self

    def describe(self, hint: str, model: str = "gpt-4.1") -> Self:
        question = FreeForm(
            prompt=GENERATE_DESCRIBE_CHAIN_PROMPT.format(
                columns=", ".join(self.df().columns),
                head=self.df().head(5),
                scheme=hint,
            ),
        )

        answer, _ = question.resolve(model)
        for fn in answer.split("\n"):
            self._execute_describe_fn(fn)

        return self

    def generate(
        self,
        question_col: str | None = None,
        messages_col: str | None = None,
        out_col: str | None = None,
        question_type: type[Question] = FreeForm,
        model: str = "gpt-4.1",
        max_workers: int = 100,
        **question_kwargs,
    ) -> Self:
        question_col = question_col or "question"
        messages_col = messages_col or "messages"
        out_col = out_col or "answer"

        df = self.df()

        if messages_col not in df and question_col not in df:
            self.rename(
                hint=(
                    "Pick one column that is the likeliest to contain either questions"
                    " or a list of messages that fit the OpenAI chat completion"
                    f" format. Rename that column to `{question_col}` if it contains"
                    f" strings and `{messages_col}` if it contains OpenAI-style chat"
                    " completion messages."
                )
            )
            df = self.df()

        if messages_col not in df:
            assert (
                question_col in df
            ), "Both question and messages columns are missing, need at least one"
            questions = [
                question_type(prompt=question, **question_kwargs)
                for question in df[question_col].to_list()
            ]
        else:
            questions = [
                question_type(messages=question, **question_kwargs)
                for question in df[messages_col].to_list()
            ]

        num_questions = len(questions)
        results: list[tuple[Any, dict]] = [("", {}) for _ in range(num_questions)]

        read_thread_pool = ThreadPoolExecutor(max_workers=40)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(
                    question.resolve,
                    model,
                    executor=read_thread_pool,
                ): idx
                for idx, question in enumerate(questions)
            }

            with RichLogger.progress_bar() as progress:
                task_id = progress.add_task("Generating…", total=num_questions)
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    try:
                        results[idx] = future.result()
                    except Exception as exc:
                        results[idx] = ("", {"error": str(exc)})
                    finally:
                        progress.advance(task_id, 1)

        df[out_col] = [result[0] for result in results]

        extra_columns: dict[str, list[Any]] = defaultdict(list)
        for extra in [result[1] for result in results]:
            for k, v in extra.items():
                extra_columns[k].append(v)

        for k, v in extra_columns.items():
            df[k] = v

        return self

    @classmethod
    def is_valid_id(cls, id: str) -> bool:
        path = URIM_HOME / "datasets" / f"{id}.jsonl"
        return path.exists()

    @classmethod
    def load_from_id(cls, id: str) -> tuple[str, Self]:
        assert cls.is_valid_id(id), f"Dataset {id} not found"
        return id, cls(input_path=str(URIM_HOME / "datasets" / f"{id}.jsonl"))

    @classmethod
    def load_from_local(cls, path: Path) -> tuple[str, Self]:
        if path.is_relative_to(URIM_HOME / "datasets"):
            ds_id = path.relative_to(URIM_HOME / "datasets").with_suffix("").name
        else:
            assert path.exists() and path.is_file(), f"Dataset {path} not found"
            ds_id = get_dataset_local_id(path)
            dest_dir = URIM_HOME / "datasets"
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest_dir / f"{ds_id}.jsonl")

        return cls.load_from_id(ds_id)

    @classmethod
    def load_from_hf(
        cls, name: str, subset: str | None = None, **kwargs
    ) -> tuple[str, Self]:
        ds_id = get_hf_dataset_local_id(name=name, subset=subset, **kwargs)
        if not cls.is_valid_id(ds_id):
            ds = cast(HFDataset, load_dataset(name, subset, **kwargs))
            ds.to_json(
                URIM_HOME / "datasets" / f"{ds_id}.jsonl", orient="records", lines=True
            )

        assert cls.is_valid_id(ds_id)
        return cls.load_from_id(ds_id)

    @classmethod
    def load(
        cls,
        name: str,
        *,
        data_dir: str | None = None,
        cache_dir: str | None = None,
        token: str | None = None,
        split: str = "train",
        num_proc: int | None = None,
        subset: str | None = None,
        **kwargs,
    ) -> tuple[str, Self]:
        path = Path(name)
        if cls.is_valid_id(name):
            return cls.load_from_id(name)
        elif path.exists() and path.is_file():
            return cls.load_from_local(path)
        else:
            return cls.load_from_hf(
                name,
                subset=subset,
                data_dir=data_dir,
                cache_dir=cache_dir,
                token=token,
                split=split,
                num_proc=num_proc,
                **kwargs,
            )

    def _execute_describe_fn(self, serialized_fn: str) -> Self:
        parts = serialized_fn.split("|")
        assert len(parts) > 0, f"Invalid function call: {serialized_fn}"

        fn_name = parts[0].strip()
        kwargs: dict[str, Any] = {}
        for part in parts[1:]:
            kwargs_parts = part.split("=")
            key, value = kwargs_parts[0].strip(), kwargs_parts[1].strip()
            if fn_name == "sample" and key == "n":
                kwargs["n"] = int(value)
            elif fn_name == "sample" and key == "frac":
                kwargs["frac"] = float(value)
            else:
                kwargs[key] = value

        if fn_name == "sample":
            self.sample(**kwargs)
        elif fn_name == "rename":
            self.rename(**kwargs)
        elif fn_name == "drop":
            self.drop(**kwargs)
        elif fn_name == "filter":
            self.filter(**kwargs)
        elif fn_name == "apply":
            self.apply(**kwargs)

        return self
