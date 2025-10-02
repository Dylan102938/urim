from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from urim.dataset import Dataset

requires_llm = pytest.mark.requires_llm


@pytest.fixture()
def dataset() -> Dataset:
    return Dataset(
        pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "a": [10, 20, 30, 40],
                "b": [100, 200, 300, 400],
            }
        )
    )


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
    ds = dataset.rename(
        hint="Rename a to x, b to y",
        question_kwargs={"enable_cache": False},
    )
    assert "x" in ds.df().columns and "a" not in ds.df().columns
    assert "y" in ds.df().columns and "b" not in ds.df().columns


def test_drop_no_llm(dataset: Dataset) -> None:
    ds = Dataset(dataset.df().copy())
    ds = ds.drop(columns=["a"])
    assert "a" not in ds.df().columns

    ds = Dataset(dataset.df().copy())
    ds = ds.drop(columns=["a", "b"])
    assert "a" not in ds.df().columns
    assert "b" not in ds.df().columns


@requires_llm
def test_drop_with_llm(dataset: Dataset) -> None:
    ds = dataset.drop(
        hint="Drop a and b",
        question_kwargs={"enable_cache": False},
    )
    assert "a" not in ds.df().columns
    assert "b" not in ds.df().columns


def test_filter_no_llm(dataset: Dataset) -> None:
    ds = dataset.filter(fn=lambda row: row["a"] > 20)
    assert len(ds.df()) == 2


@requires_llm
def test_filter_with_llm(dataset: Dataset) -> None:
    ds = dataset.filter(
        fn=None,
        hint="Filter a > 20",
        question_kwargs={"enable_cache": False},
    )
    assert len(ds.df()) == 2


def test_apply_no_llm(dataset: Dataset) -> None:
    ds = dataset.apply(fn=lambda row: row["a"] + row["b"], column="sum")
    assert list(ds.df()["sum"]) == [110, 220, 330, 440]


@requires_llm
def test_apply_with_llm(dataset: Dataset) -> None:
    ds = Dataset(dataset.df().copy())
    ds = ds.apply(
        column="sum",
        hint="create column sum = a + b",
        question_kwargs={"enable_cache": False},
    )
    assert len(ds.df().columns) == len(dataset.df().columns) + 1
    assert list(ds.df()["sum"]) == [110, 220, 330, 440]


def test_concat_no_llm(dataset: Dataset) -> None:
    ds1 = Dataset(dataset.df().copy())
    concat = ds1.concatenate(ds1)

    assert len(concat.df()) == 8
