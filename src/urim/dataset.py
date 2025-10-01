from collections import defaultdict
from collections.abc import Callable, Hashable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from urim.ai.prompts import (
    DATASET_APPLY_PROMPT,
    DATASET_DROP_NO_HINT_PROMPT,
    DATASET_DROP_WITH_HINT_PROMPT,
    DATASET_FILTER_PROMPT,
    DATASET_RENAME_PROMPT,
)
from urim.ai.question import FreeForm

if TYPE_CHECKING:
    import pandas as pd

    from urim.ai.question import Question

PRESET_FAST = "gpt-4.1-mini"
PRESET_BALANCED = "gpt-4.1"
PRESET_THOROUGH = "gpt-5"


def extract_op_kwargs(
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
        prompt = template.format({"head": rows.head(), **kwargs})
    else:
        prompt = template.format(**kwargs)

    for _ in range(3):
        try:
            question = ExtractJSON(prompt, **(question_kwargs or {}))
            return question.json(model)
        except Exception:
            pass

    raise Exception("Failed to extract kwargs")


def extract_op_fn(
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
        prompt = template.format({"head": rows.head(), **kwargs})
    else:
        prompt = template.format(**kwargs)

    for _ in range(3):
        try:
            question = ExtractFunction(prompt, **(question_kwargs or {}))
            return question.fn(model)
        except Exception:
            pass

    raise Exception("Failed to extract fn")


class Dataset:
    def __init__(self, dataset: "pd.DataFrame" | str, **kwargs: Any):
        import pandas as pd
        from datasets import load_dataset

        self._df: pd.DataFrame
        if isinstance(dataset, str):
            if Path(dataset).exists():
                self._df = pd.read_json(dataset, orient="records", lines=True, **kwargs)
            else:
                self._df = load_dataset(dataset, **kwargs).to_pandas()
        else:
            self._df = dataset

        self._init_dataset_kwargs = kwargs

    def df(self) -> "pd.DataFrame":
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
    ) -> "Dataset":
        df = self._df.sample(n=n, frac=frac, **kwargs)
        return self._maybe_inplace(df, inplace=inplace)

    def rename(
        self,
        columns: Mapping[Any, Hashable] | None = None,
        hint: str | None = None,
        model: str = PRESET_FAST,
        *,
        inplace: bool = False,
        **kwargs: Any,
    ) -> "Dataset":
        assert (
            columns is not None or hint is not None
        ), "Must provide a hint if no columns are provided"

        if columns is None:
            assert hint is not None
            kwargs = {
                **kwargs,
                **extract_op_kwargs(
                    DATASET_RENAME_PROMPT,
                    columns=", ".join(self._df.columns),
                    scheme=hint,
                    model=model,
                    curr_df=self._df,
                ),
            }

        df = self._df.rename(**kwargs)
        return self._maybe_inplace(df, inplace=inplace)

    def drop(
        self,
        columns: list[str] | str | None = None,
        hint: str | None = None,
        model: str = PRESET_FAST,
        *,
        inplace: bool = False,
        **kwargs: Any,
    ) -> "Dataset":
        assert columns is not None or hint is not None, "Must provide either columns or hint"

        if columns is None:
            kwargs = {
                **kwargs,
                **extract_op_kwargs(
                    DATASET_DROP_WITH_HINT_PROMPT if hint else DATASET_DROP_NO_HINT_PROMPT,
                    columns=", ".join(self._df.columns),
                    scheme=hint,
                    model=model,
                    curr_df=self._df,
                ),
            }

        df = self._df.drop(**kwargs)
        return self._maybe_inplace(df, inplace=inplace)

    def filter(
        self,
        fn: Callable[[pd.Series], bool] | None = None,
        hint: str | None = None,
        model: str = PRESET_BALANCED,
        *,
        inplace: bool = False,
        **kwargs: Any,
    ) -> "Dataset":
        assert fn is not None or hint is not None, "Must provide either fn or hint"

        if fn is None:
            kwargs = {
                **kwargs,
                **extract_op_kwargs(
                    DATASET_FILTER_PROMPT,
                    columns=", ".join(self._df.columns),
                    scheme=hint,
                    model=model,
                    curr_df=self._df,
                ),
            }

        df = self._df.filter(**kwargs)
        return self._maybe_inplace(df, inplace=inplace)

    def apply(
        self,
        fn: Callable[[pd.Series], bool] | None = None,
        column: str | None = None,
        hint: str | None = None,
        model: str = PRESET_BALANCED,
        *,
        inplace: bool = False,
        **kwargs: Any,
    ) -> "Dataset":
        assert fn is not None or hint is not None, "Must provide either fn or hint"

        if fn is None:
            wrapper_fn = extract_op_fn(
                DATASET_APPLY_PROMPT,
                columns=", ".join(self._df.columns),
                column_hint=f"The column's name should be {column}." if column is not None else "",
                scheme=hint,
                model=model,
                curr_df=self._df,
            )

            column, fn = wrapper_fn()

        df = self._df.copy()
        df[column] = df.apply(fn, axis=1, **kwargs)

        return self._maybe_inplace(df, inplace=inplace)

    def generate(
        self,
        question_col: str | None = None,
        messages_col: str | None = None,
        system_col: str | None = None,
        out_col: str | None = None,
        question_type: type["Question"] = FreeForm,
        model: str = PRESET_BALANCED,
        max_workers: int = 100,
        inplace: bool = False,
        **question_kwargs: Any,
    ) -> "Dataset":
        from concurrent.futures import ThreadPoolExecutor, as_completed

        question_col = question_col or "question"
        messages_col = messages_col or "messages"
        system_col = system_col or "system"
        out_col = out_col or "answer"

        assert question_col is not None or messages_col is not None

        input_col = messages_col or question_col

        assert input_col in self._df.columns

        input_iter = self._df[input_col].to_list()
        questions: list[Question] = []
        for i, inp in enumerate(input_iter):
            common_system = question_kwargs.pop("system", None)
            system = (
                str(self._df.iloc[i][system_col])
                if system_col in self._df.columns
                else common_system
            )
            questions.append(
                question_type(
                    prompt=inp,  # TODO: fix bug here that only supports prompt vs messages
                    system=system,
                    **question_kwargs,
                )
            )

        num_questions = len(questions)
        results: list[tuple[Any, dict]] = [("", {}) for _ in range(num_questions)]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(question.resolve, model): idx
                for idx, question in enumerate(questions)
            }

            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    results[idx] = ("", {"error": str(exc)})

        df = self._df.copy()
        df[out_col] = [result[0] for result in results]

        extra_columns: dict[str, list[Any]] = defaultdict(list)
        for extra in [result[1] for result in results]:
            for k, v in extra.items():
                extra_columns[k].append(v)

        for k, v in extra_columns.items():
            df[k] = v

        return self._maybe_inplace(df, inplace=inplace)

    def concatenate(
        self, other: "Dataset" | list["Dataset"], *, inplace: bool = False, **kwargs: Any
    ) -> "Dataset":
        import pandas as pd

        if not isinstance(other, list):
            other = [other]

        return self._maybe_inplace(
            pd.concat([self._df, *[ds._df for ds in other]], axis=0, **kwargs), inplace=inplace
        )

    def _maybe_inplace(self, df: "pd.DataFrame", *, inplace: bool) -> "Dataset":
        if inplace:
            self._df = df
            return self
        else:
            return Dataset(df)
