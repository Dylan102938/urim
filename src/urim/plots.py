from urim.dataset import Dataset


def concat_with_groups(
    ds_ids: list[str],
    hint: str | None = None,
) -> tuple[tuple[str, ...], Dataset]:
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

    df = concat_ds.df()
    candidate_cats: list[tuple[str, int]] = []
    for col in df.columns:
        series = df[col]
        nunique = int(series.nunique(dropna=True))
        threshold = max(2, min(50, int(len(series) * 0.05)))
        if 2 <= nunique <= threshold:
            candidate_cats.append((col, nunique))

    ordered_groups = tuple(
        [name for name, _ in sorted(candidate_cats, key=lambda x: (x[1], x[0]))]
    )

    return ordered_groups, concat_ds
