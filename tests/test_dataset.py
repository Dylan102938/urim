from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore
import pytest

from urim.dataset import Dataset

requires_llm = pytest.mark.requires_llm


@pytest.fixture()
def df_small() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "a": [10, 20, 30, 40],
            "b": [100, 200, 300, 400],
        }
    )


def test_df_and_to_json_roundtrip(tmp_path: Path, df_small: pd.DataFrame) -> None:
    ds = Dataset(df=df_small.copy())
    out = tmp_path / "ds.jsonl"
    ds.to_json(str(out))

    assert out.exists()

    ds2 = Dataset(input_path=str(out))
    df2 = ds2.df()
    assert list(df2.columns) == list(df_small.columns)
    assert len(df2) == len(df_small)


def test_sample_by_n_and_frac(df_small: pd.DataFrame) -> None:
    ds = Dataset(df=df_small.copy())
    ds.sample(n=2, random_state=0)
    assert len(ds.df()) == 2

    ds = Dataset(df=df_small.copy())
    ds.sample(frac=0.5, random_state=0)
    assert len(ds.df()) == 2


def test_rename_with_columns(df_small: pd.DataFrame) -> None:
    ds = Dataset(df=df_small.copy())
    ds.rename(columns={"a": "x"})
    assert "x" in ds.df().columns and "a" not in ds.df().columns


@requires_llm
def test_rename_with_hint(df_small: pd.DataFrame) -> None:
    ds = Dataset(df=df_small.copy())
    ds.rename(columns=None, hint="rename columns")
    assert len(ds.df().columns) == len(df_small.columns)


def test_drop_with_columns(df_small: pd.DataFrame) -> None:
    ds = Dataset(df=df_small.copy())
    ds.drop(columns=["b"])
    assert list(ds.df().columns) == ["id", "a"]


@requires_llm
def test_drop_with_hint(df_small: pd.DataFrame) -> None:
    ds = Dataset(df=df_small.copy())
    ds.drop(columns=None, hint="drop noisy cols")
    assert len(ds.df().columns) < len(df_small.columns)


def test_filter_with_fn(df_small: pd.DataFrame) -> None:
    ds = Dataset(df=df_small.copy())
    ds.filter(fn=lambda row: row["a"] > 20)
    assert list(ds.df()["id"]) == [3, 4]


@requires_llm
def test_filter_with_hint(df_small: pd.DataFrame) -> None:
    ds = Dataset(df=df_small.copy())
    ds.filter(fn=None, hint="filter a > 20")
    assert len(ds.df()) <= len(df_small)


def test_apply_with_fn(df_small: pd.DataFrame) -> None:
    ds = Dataset(df=df_small.copy())
    ds.apply(fn=lambda row: row["a"] + row["b"], column="sum")
    assert list(ds.df()["sum"]) == [110, 220, 330, 440]


@requires_llm
def test_apply_with_hint(df_small: pd.DataFrame) -> None:
    ds = Dataset(df=df_small.copy())
    ds.apply(fn=None, column=None, hint="create column z = 2*a")
    assert len(ds.df().columns) == len(df_small.columns) + 1


def test_merge_simple() -> None:
    left = Dataset(df=pd.DataFrame({"id": [1, 2], "a": [10, 20]}))
    right = Dataset(df=pd.DataFrame({"id": [2, 3], "b": [200, 300]}))

    out = left.merge(right, on="id", how="inner")
    # should mutate self and return self
    assert out is left
    df = out.df()
    # Expect only id=2
    assert list(df["id"]) == [2]
    assert list(df["a"]) == [20]
    assert list(df["b"]) == [200]


@requires_llm
def test_merge_with_hint() -> None:
    left = Dataset(df=pd.DataFrame({"id": [1, 2], "a": [10, 20]}))
    right = Dataset(df=pd.DataFrame({"id": [2, 3], "b": [200, 300]}))

    left.merge(right, hint="figure out join")
    df = left.df()
    assert len(df) >= 2


def test_concat_simple() -> None:
    top = Dataset(df=pd.DataFrame({"id": [1, 2], "a": [10, 20]}))
    bottom = Dataset(df=pd.DataFrame({"id": [3, 4], "a": [30, 40]}))

    top.concat(bottom)
    df = top.df()
    assert list(df["id"]) == [1, 2, 3, 4]
    assert list(df["a"]) == [10, 20, 30, 40]


@requires_llm
def test_concat_with_hint() -> None:
    df1 = pd.DataFrame({"id": [1], "a": [10]})
    df2 = pd.DataFrame({"A": [2], "B": [20]})

    left = Dataset(df=df1.copy())
    right = Dataset(df=df2.copy())

    left.concat(right, hint="align columns")
    df = left.df()
    assert len(df) == 2


@requires_llm
def test_describe_noop(df_small: pd.DataFrame) -> None:
    ds = Dataset(df=df_small.copy())
    out = ds.describe("some hint")
    assert out is ds


from urim.ai.question import Question  # noqa: E402


class DummyQuestion(Question[str]):
    def __init__(
        self,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(prompt=prompt, messages=messages, **kwargs)

    def fetch(self, model: str):  # noqa: ANN001, ANN201
        if self.prompt is not None:
            return (f"ans:{self.prompt}", {"meta": "ok"})
        return (f"ans:{len(self.messages or [])}", {"meta": "ok"})


def test_generate_with_question_col() -> None:
    df = pd.DataFrame({"question": ["Q1", "Q2"]})
    ds = Dataset(df=df)
    # Disable cache in generated questions to avoid cache writer signals in threads
    ds.generate(
        question_col="question",
        out_col="answer",
        question_type=DummyQuestion,
        max_workers=2,
        enable_cache=False,
    )

    out_df = ds.df()
    assert list(out_df["answer"]) == ["ans:Q1", "ans:Q2"]
    # extra column propagated
    assert list(out_df["meta"]) == ["ok", "ok"]


def test_generate_with_messages_col() -> None:
    df = pd.DataFrame(
        {
            "messages": [
                [{"role": "user", "content": "hi"}],
                [{"role": "user", "content": "bye"}],
            ]
        }
    )
    ds = Dataset(df=df)
    # Disable cache in generated questions to avoid cache writer signals in threads
    ds.generate(
        messages_col="messages",
        out_col="out",
        question_type=DummyQuestion,
        max_workers=2,
        enable_cache=False,
    )

    out_df = ds.df()
    assert list(out_df["out"]) == ["ans:1", "ans:1"]
    assert list(out_df["meta"]) == ["ok", "ok"]


def test_load_path_exists(tmp_path: Path) -> None:
    df = pd.DataFrame({"id": [1], "a": [10]})
    path = tmp_path / "exists.jsonl"
    df.to_json(path, orient="records", lines=True)

    _, ds = Dataset.load(str(path))
    assert list(ds.df()["id"]) == [1]


def test_load_hf_branch_monkeypatched(monkeypatch: pytest.MonkeyPatch) -> None:
    def _stub(name: str, subset: str | None = None, **kwargs):  # noqa: ANN001, ANN201
        _ = (name, subset, kwargs)
        return "stubbed", Dataset(df=pd.DataFrame({"id": [42]}))

    monkeypatch.setattr(
        Dataset,
        "load_from_hf",
        classmethod(
            lambda cls, name, subset=None, **kwargs: _stub(name, subset, **kwargs)
        ),
    )

    _, ds = Dataset.load("definitely-not-a-path")
    assert list(ds.df()["id"]) == [42]
