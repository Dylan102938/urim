from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import urim.ai.client as client_mod
from urim.ai.client import ChatResult
from urim.ai.question_cache import QuestionCache
from urim.dataset import Dataset

requires_llm = pytest.mark.requires_llm


@pytest.fixture()
def temp_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Iterator[QuestionCache]:
    import urim.ai.question as qmod

    cache = QuestionCache(cache_dir=tmp_path)
    monkeypatch.setattr(qmod, "_DEFAULT_CACHE", cache)

    yield cache

    cache.stop()


@pytest.fixture()
def dataset(temp_cache: QuestionCache) -> Dataset:
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "a": [10, 20, 30, 40],
            "b": [100, 200, 300, 400],
        }
    )

    return Dataset(df=df)


def test_df(dataset: Dataset) -> None:
    assert dataset.df() is not None
    assert len(dataset.df()) == 4
    assert list(dataset.df()["id"]) == [1, 2, 3, 4]
    assert list(dataset.df()["a"]) == [10, 20, 30, 40]
    assert list(dataset.df()["b"]) == [100, 200, 300, 400]


def test_to_json(dataset: Dataset, tmp_path: Path) -> None:
    out = tmp_path / "ds.jsonl"
    dataset.to_json(out.as_posix())

    assert out.exists()
    assert out.read_text() == "".join(
        [
            f'{{"id":{i},"a":{a},"b":{b}}}\n'
            for i, a, b in zip(range(1, 5), [10, 20, 30, 40], [100, 200, 300, 400], strict=False)
        ]
    )


@pytest.mark.parametrize("n_or_frac", [2, 0.5])
def test_sample(dataset: Dataset, n_or_frac: int | float) -> None:
    if isinstance(n_or_frac, int):
        ds = dataset.sample(n=n_or_frac)
    else:
        ds = dataset.sample(frac=n_or_frac)

    df = ds.df()
    assert len(df) == 2


def test_rename_no_llm(dataset: Dataset) -> None:
    ds = dataset.rename(columns={"a": "x"})
    assert "x" in ds.df().columns and "a" not in ds.df().columns

    ds = dataset.rename(columns={"a": "x", "b": "y"})
    assert "x" in ds.df().columns and "a" not in ds.df().columns
    assert "y" in ds.df().columns and "b" not in ds.df().columns


@requires_llm
def test_rename(dataset: Dataset) -> None:
    ds = dataset.rename(columns=None, hint="Rename a to x, b to y")
    assert "x" in ds.df().columns and "a" not in ds.df().columns
    assert "y" in ds.df().columns and "b" not in ds.df().columns


def test_drop_no_llm(dataset: Dataset) -> None:
    ds = Dataset(df=dataset.df().copy())
    ds = ds.drop(columns=["a"])
    assert "a" not in ds.df().columns

    ds = Dataset(df=dataset.df().copy())
    ds = ds.drop(columns=["a", "b"])
    assert "a" not in ds.df().columns
    assert "b" not in ds.df().columns


@requires_llm
def test_drop_with_llm(dataset: Dataset) -> None:
    ds = dataset.drop(columns=None, hint="Drop a and b")
    assert "a" not in ds.df().columns
    assert "b" not in ds.df().columns


def test_filter_no_llm(dataset: Dataset) -> None:
    ds = dataset.filter(fn=lambda row: row["a"] > 20)
    assert len(ds.df()) == 2


@requires_llm
def test_filter_with_llm(dataset: Dataset) -> None:
    ds = dataset.filter(fn=None, hint="Filter a > 20")
    assert len(ds.df()) == 2


def test_apply_no_llm(dataset: Dataset) -> None:
    ds = dataset.apply(fn=lambda row: row["a"] + row["b"], column="sum")
    assert list(ds.df()["sum"]) == [110, 220, 330, 440]


@requires_llm
def test_apply_with_llm(dataset: Dataset) -> None:
    ds = Dataset(df=dataset.df().copy())
    ds = ds.apply(column="sum", hint="create column sum = a + b")
    assert len(ds.df().columns) == len(dataset.df().columns) + 1
    assert list(ds.df()["sum"]) == [110, 220, 330, 440]


def test_merge_no_llm(dataset: Dataset) -> None:
    ds = dataset.merge(dataset, on="id", how="inner")
    assert len(ds.df()) == 4


@requires_llm
def test_merge_with_llm_simple(dataset: Dataset) -> None:
    ds1 = Dataset(df=dataset.df().copy())
    ds1 = ds1.rename(columns={"a": "a_id"})
    merged = dataset.merge(ds1)
    assert len(merged.df()) == 4
    assert list(merged.df()["a_id"]) == [10, 20, 30, 40]
    assert list(merged.df()["a"]) == [10, 20, 30, 40]


@requires_llm
def test_merge_with_llm_mixed(dataset: Dataset) -> None:
    ds1 = Dataset(df=dataset.df().copy())
    ds1 = ds1.rename(columns={"a": "a_id"})
    merged = dataset.merge(ds1, left_on="a", how="left", hint="merge on a")
    assert len(merged.df()) == 4

    assert list(merged.df()["a_id"]) == [10, 20, 30, 40]
    assert list(merged.df()["a"]) == [10, 20, 30, 40]


def test_concat_no_llm(dataset: Dataset) -> None:
    ds1 = Dataset(df=dataset.df().copy())
    concat = ds1.concat(ds1)
    assert len(concat.df()) == 8


@requires_llm
def test_concat_with_llm_no_hint(dataset: Dataset) -> None:
    ds1 = Dataset(df=dataset.df().copy())
    ds1 = ds1.rename(columns={"a": "a_alt"})

    concat = dataset.concat(ds1)

    assert len(concat.df()) == 8
    assert "a" in concat.df().columns
    assert "a_id" not in concat.df().columns


@requires_llm
def test_concat_with_llm_with_hint(dataset: Dataset) -> None:
    ds1 = Dataset(df=dataset.df().copy())

    concat = dataset.concat(
        ds1,
        hint=(
            "You should concatenate column a from the original dataset to column b in"
            " the other dataset and column b in the original dataset to column a in the"
            " other dataset."
        ),
    )

    assert len(concat.df()) == 8
    assert "a" in concat.df().columns
    assert "b" in concat.df().columns

    a_col = concat.df()["a"]
    b_col = concat.df()["b"]

    assert list(a_col) == [10, 20, 30, 40, 100, 200, 300, 400]
    assert list(b_col) == [100, 200, 300, 400, 10, 20, 30, 40]


def test_describe_sample_only(monkeypatch: pytest.MonkeyPatch, dataset: Dataset) -> None:
    original = client_mod.LLM.chat_completion
    call_count = {"n": 0}

    def stub_first_call(self: Any, model: str, *args: Any, **kwargs: Any) -> ChatResult:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return ChatResult(content="sample | n=2", raw={})
        return original(self, model, *args, **kwargs)

    monkeypatch.setattr(client_mod.LLM, "chat_completion", stub_first_call)

    ds = dataset.describe(hint="take a small sample")
    assert len(ds.df()) == 2


@requires_llm
def test_describe_sample_then_drop(monkeypatch: pytest.MonkeyPatch, dataset: Dataset) -> None:
    original = client_mod.LLM.chat_completion
    call_count = {"n": 0}

    def stub_first_call(self: Any, model: str, *args: Any, **kwargs: Any) -> ChatResult:
        call_count["n"] += 1
        if call_count["n"] == 1:
            content = "\n".join(
                [
                    "sample | n=4",
                    "drop | hint=Drop b",
                ]
            )
            return ChatResult(content=content, raw={})
        return original(self, model, *args, **kwargs)

    monkeypatch.setattr(client_mod.LLM, "chat_completion", stub_first_call)

    ds = dataset.describe(hint="drop a column after sampling")
    assert set(ds.df().columns) == {"id", "a"}
    assert len(ds.df()) == 4


@requires_llm
def test_describe_rename_then_apply(monkeypatch: pytest.MonkeyPatch, dataset: Dataset) -> None:
    original = client_mod.LLM.chat_completion
    call_count = {"n": 0}

    def stub_first_call(self: Any, model: str, *args: Any, **kwargs: Any) -> ChatResult:
        call_count["n"] += 1
        if call_count["n"] == 1:
            content = "\n".join(
                [
                    "rename | hint=Rename a to x",
                    "apply | column=total | hint=create column z = x + b",
                ]
            )
            return ChatResult(content=content, raw={})
        return original(self, model, *args, **kwargs)

    monkeypatch.setattr(client_mod.LLM, "chat_completion", stub_first_call)

    ds = dataset.describe(hint="rename then compute a total")
    assert "x" in ds.df().columns and "a" not in ds.df().columns
    assert "total" in ds.df().columns
    assert list(ds.df()["total"]) == [110, 220, 330, 440]
