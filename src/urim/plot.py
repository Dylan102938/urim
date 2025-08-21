from collections.abc import Mapping

from urim.ai.prompts import GET_KWARGS_PROMPT
from urim.ai.question import ExtractJSON
from urim.dataset import Dataset


def _serialize_kwargs(kwargs: Mapping[str, str | None]) -> str:
    return ", ".join(f"{key}={value}" for key, value in kwargs.items() if value is not None)


def _reorder_columns(guessed_order: list, actual_order: list) -> list:
    for i, item in enumerate(actual_order):
        if item is None:
            continue
        if item in guessed_order:
            guessed_order.remove(item)
            guessed_order.insert(i, item)

    return guessed_order


def _segment_columns(prioritized_columns: list, other_columns: list) -> tuple[list, list]:
    for item in reversed(prioritized_columns):
        if item in other_columns:
            other_columns.remove(item)

    return prioritized_columns, other_columns


def concat_datasets(
    ds_ids: list[str],
    hint: str | None = None,
) -> Dataset:
    concat_ds: Dataset | None = None
    for ds_id in ds_ids:
        _, ds = Dataset.load_from_id(ds_id)
        df = ds.df()
        df["id"] = ds_id
        if concat_ds is None:
            concat_ds = ds
        else:
            concat_ds.concat(ds, hint=hint)

    assert concat_ds is not None

    return concat_ds


def get_categorical_columns(ds: Dataset) -> list[str]:
    df = ds.df()
    candidate_cats: list[tuple[str, int]] = []
    for col in df.columns:
        series = df[col]
        nunique = int(series.nunique(dropna=True))
        threshold = max(10, min(50, len(series) * 0.05))
        if 2 <= nunique <= threshold:
            candidate_cats.append((col, nunique))

    ordered_groups = sorted(candidate_cats, key=lambda x: (x[1], x[0]))
    return [col for col, _ in ordered_groups]


def get_data_columns(ds: Dataset) -> list[str]:
    df = ds.df()
    numeric_cols = list(df.select_dtypes(include="number").columns)
    candidate_data: list[tuple[str, int]] = []
    for col in numeric_cols:
        series = df[col]
        nunique = int(series.nunique(dropna=True))
        candidate_data.append((col, nunique))

    ordered_data = sorted(candidate_data, key=lambda x: (x[1], x[0]), reverse=True)
    return [col for col, _ in ordered_data]


def infer_distribution_kwargs(
    ds: Dataset,
    x: str | None = None,
    hue: str | None = None,
    column: str | None = None,
    hint: str | None = None,
    graph_type: str = "hist",
    model: str = "gpt-4.1",
) -> dict[str, str]:
    categorical_columns = get_categorical_columns(ds)
    data_columns = get_data_columns(ds)
    assert len(data_columns) > 0, "No numeric columns found in the dataset"

    data_cols = _reorder_columns(data_columns, [x])
    cat_cols = _reorder_columns(categorical_columns, [column, hue])

    data_cols, cat_cols = _segment_columns(data_cols[:1], cat_cols)
    cat_cols = cat_cols[:2]

    kwargs = {
        "x": data_cols[0],
        "hue": cat_cols[-1] if len(cat_cols) > 0 else None,
        "col": cat_cols[-2] if len(cat_cols) > 1 else None,
    }

    if hint is not None:
        kwargs = ExtractJSON(
            prompt=GET_KWARGS_PROMPT.format(
                columns=ds.df().dtypes.to_string(),
                graph_type=graph_type,
                scheme=hint,
                guessed_columns=_serialize_kwargs(kwargs),
            ),
        ).json(model)

    return {k: v for k, v in kwargs.items() if v is not None}


def infer_categorical_kwargs(
    ds: Dataset,
    x: str | None = None,
    y: str | None = None,
    hue: str | None = None,
    column: str | None = None,
    hint: str | None = None,
    graph_type: str = "bar",
    model: str = "gpt-4.1",
) -> dict[str, str]:
    data_columns = get_data_columns(ds)
    categorical_columns = get_categorical_columns(ds)
    assert len(categorical_columns) > 0, "No categorical columns found in the dataset"

    cat_cols = _reorder_columns(categorical_columns, [column, hue, x])
    data_cols = _reorder_columns(data_columns, [y])

    data_cols, cat_cols = _segment_columns(data_cols[:1], cat_cols)
    cat_cols = cat_cols[:3]

    kwargs = {
        "x": cat_cols[-1] if len(cat_cols) > 0 else None,
        "hue": (
            cat_cols[-2] if len(cat_cols) >= 2 else cat_cols[-1] if len(cat_cols) > 0 else None
        ),
        "col": cat_cols[-3] if len(cat_cols) >= 3 else None,
        "y": data_cols[0] if len(data_cols) > 0 else None,
    }

    if hint is not None:
        kwargs = ExtractJSON(
            prompt=GET_KWARGS_PROMPT.format(
                columns=ds.df().dtypes.to_string(),
                graph_type=graph_type,
                scheme=hint,
                guessed_columns=_serialize_kwargs(kwargs),
            ),
        ).json(model)

    return {k: v for k, v in kwargs.items() if v is not None}


def infer_relational_kwargs(
    ds: Dataset,
    x: str | None = None,
    y: str | None = None,
    hue: str | None = None,
    column: str | None = None,
    hint: str | None = None,
    graph_type: str = "scatter",
    model: str = "gpt-4.1",
) -> dict[str, str]:
    numeric_cols = get_data_columns(ds)
    categorical_cols = get_categorical_columns(ds)
    assert len(numeric_cols) > 1, "No numeric columns found in the dataset"

    categorical_cols = _reorder_columns(categorical_cols, [column, hue])
    numeric_cols = _reorder_columns(numeric_cols, [y, x])

    data_cols, cat_cols = _segment_columns(numeric_cols[:2], categorical_cols)
    cat_cols = cat_cols[:2]

    kwargs = {
        "x": data_cols[1],
        "y": data_cols[0],
        "hue": cat_cols[-1] if len(cat_cols) > 0 else data_cols[-1],
        "col": cat_cols[-2] if len(cat_cols) > 1 else None,
    }

    if hint is not None:
        kwargs = ExtractJSON(
            prompt=GET_KWARGS_PROMPT.format(
                columns=ds.df().dtypes.to_string(),
                graph_type=graph_type,
                scheme=hint,
                guessed_columns=_serialize_kwargs(kwargs),
            ),
        ).json(model)

    return {k: v for k, v in kwargs.items() if v is not None}
