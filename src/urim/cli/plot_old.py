from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd  # type: ignore[import-untyped]
import typer

from urim.ai.question import ExtractJSON
from urim.cli.utils import parse_kv, random_filestub
from urim.dataset import Dataset
from urim.env import URIM_HOME, UrimDatasetGraph
from urim.logging_utils import RichLogger

import seaborn as sns

plot_app = typer.Typer(help="Plot utilities: visualize one or more datasets.")


def _resolve_output_path(
    output: Path | None,
    *,
    filename: str | None = None,
    suffix: str = ".png",
) -> Path:
    if output is None:
        base = URIM_HOME / "plots"
        base.mkdir(parents=True, exist_ok=True)
        name = (
            filename
            or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random_filestub()}"
        )
        return base / f"{name}{suffix}"

    # If output is a directory, use provided filename or derive one
    if output.exists() and output.is_dir():
        name = (
            filename
            or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random_filestub()}"
        )
        return output / f"{name}{suffix}"

    # Treat as file path (ensure parent exists)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


PLOT_SPEC_PROMPT = (
    "I have one or more tabular datasets I want to visualize. You will output a JSON "
    "spec describing exactly one plot to generate.\n\n"
    "Input datasets (name, columns, head):\n{datasets}\n\n"
    "Requirements/hint:\n{hint}\n\n"
    "If some parameters are already fixed by the user, honor them: {fixed}.\n\n"
    "Output a JSON object with keys chosen from the following, depending on plot type:"
    "\n- type: one of [line, bar, hist, scatter, corr]\n"
    "- x: name of x-axis column (if applicable)\n"
    "- y: name of y-axis column (if applicable)\n"
    "- column: column for hist/value_counts (if applicable)\n"
    "- bins: integer for hist\n"
    "- top_n: integer for bar with counts\n"
    "- normalize: boolean for frequency vs count in bar/value_counts\n"
    "- aggregate: aggregation for line/bar when combining duplicates (mean/sum/count)\n"
    "- title: string plot title\n"
    "- limit: integer to limit number of points\n"
    "- method: for corr (pearson/spearman/kendall).\n\n"
    "Choose sensible defaults if not specified, based on the data and hint."
)


def _format_datasets_for_prompt(datasets: list[tuple[str, Dataset]]) -> str:
    parts: list[str] = []
    for label, ds in datasets[:3]:
        df = ds.df()
        head = df.head(5).to_string()
        parts.append(f"- {label}: columns=[{', '.join(map(str, df.columns))}]\n{head}")
    return "\n\n".join(parts)


def _infer_plot_spec(
    datasets: list[tuple[str, Dataset]],
    hint: str,
    fixed: dict[str, Any],
    *,
    model: str = "gpt-4.1",
) -> dict[str, Any]:
    prompt = PLOT_SPEC_PROMPT.format(
        datasets=_format_datasets_for_prompt(datasets),
        hint=hint,
        fixed=fixed,
    )
    question = ExtractJSON(prompt=prompt)
    spec = question.json(model)
    for k, v in fixed.items():
        if v is not None:
            spec[k] = v
    return spec


def _auto_plot(
    ctx: typer.Context,
    *,
    hint: str | None,
    plot_type: str | None,
    x: str | None,
    y: str | None,
    column: str | None,
    bins: int | None,
    top_n: int | None,
    normalize: bool | None,
    title: str | None,
    output: Path | None,
    limit: int | None,
    method: Literal["pearson", "spearman", "kendall"] | None,
    model: str,
) -> None:
    ctx_obj: PlotContext = ctx.obj
    datasets = ctx_obj.datasets
    assert len(datasets) > 0

    fixed: dict[str, Any] = {
        "type": plot_type,
        "x": x,
        "y": y,
        "column": column,
        "bins": bins,
        "top_n": top_n,
        "normalize": normalize,
        "title": title,
        "limit": limit,
        "method": method,
    }

    if hint:
        spec = _infer_plot_spec(datasets, hint, fixed, model=model)
    else:
        spec = {k: v for k, v in fixed.items() if v is not None}

    plot_kind = (spec.get("type") or "").lower()
    if plot_kind in {"", None}:
        if spec.get("x") and spec.get("y"):
            plot_kind = "scatter"
        elif spec.get("column"):
            plot_kind = "hist"
        else:
            ctx.invoke(status, ctx)
            return

    if plot_kind == "hist":
        ctx.invoke(
            hist,
            ctx,
            column=spec.get("column"),
            bins=int(spec.get("bins") or 30),
            output=output,
            title=spec.get("title"),
        )
        return

    if plot_kind in {"bar", "value_counts"}:
        if spec.get("y") is None:
            ctx.invoke(
                bar,
                ctx,
                x=spec.get("x") or spec.get("column"),
                y=None,
                aggregate="count",
                top_n=int(spec.get("top_n") or 20),
                normalize=bool(spec.get("normalize") or False),
                output=output,
                title=spec.get("title"),
            )
        else:
            ctx.invoke(
                bar,
                ctx,
                x=spec.get("x"),
                y=spec.get("y"),
                aggregate=str(spec.get("aggregate") or "mean"),
                top_n=int(spec.get("top_n") or 20),
                normalize=False,
                output=output,
                title=spec.get("title"),
            )
        return

    if plot_kind == "line":
        ctx.invoke(
            line,
            ctx,
            x=spec.get("x"),
            y=spec.get("y"),
            aggregate=str(spec.get("aggregate") or "mean"),
            output=output,
            title=spec.get("title"),
            limit=int(spec.get("limit") or 0) or None,
        )
        return

    if plot_kind == "scatter":
        ctx.invoke(
            scatter,
            ctx,
            x=spec.get("x"),
            y=spec.get("y"),
            output=output,
            title=spec.get("title"),
            limit=int(spec.get("limit") or 0) or None,
        )
        return

    if plot_kind == "corr":
        ctx.invoke(
            corr,
            ctx,
            output=output,
            method=spec.get("method") or "pearson",
        )
        return

    RichLogger.warning(f"Unknown plot type '{plot_kind}'. Showing status instead.")
    ctx.invoke(status, ctx)


def _load_dataset_named(
    name: str,
    *,
    data_dir: str | None = None,
    cache_dir: str | None = None,
    token: str | None = None,
    split: str = "train",
    num_proc: int | None = None,
    subset: str | None = None,
    **kwargs: Any,
) -> tuple[str, Dataset]:
    ds_id, ds = Dataset.load(
        name,
        data_dir=data_dir,
        cache_dir=cache_dir,
        token=token,
        split=split,
        num_proc=num_proc,
        subset=subset,
        **kwargs,
    )
    label = kwargs.get("label") or ds_id
    return (str(label), ds)


@plot_app.callback(invoke_without_command=True)
def setup_plot_context(
    ctx: typer.Context,
    dataset: list[str] | None = typer.Option(
        None,
        "-d",
        "--dataset",
        "-n",
        "--name",
        help=(
            "Dataset(s) to plot: HF id(s), urim dataset id(s), or local JSONL path(s)."
        ),
    ),
    split: str = typer.Option(
        "train",
        "--split",
        help="Split to use when loading from HF.",
    ),
    subset: str | None = typer.Option(
        None,
        "--subset",
        help="Subset to use from the dataset.",
    ),
    data_dir: str | None = typer.Option(
        None,
        "--data-dir",
        help="Directory to store HF dataset data.",
    ),
    cache_dir: str | None = typer.Option(
        None,
        "--cache-dir",
        help="Directory to store HF dataset cache.",
    ),
    token: str | None = typer.Option(
        None,
        "--token",
        help="Hugging Face token.",
    ),
    num_proc: int | None = typer.Option(
        None,
        "--num-proc",
        help="Number of processes to use for dataset loading.",
    ),
    labels: list[str] = typer.Option(
        [],
        "-l",
        "--label",
        help="Optional display label(s) for the provided dataset(s) (repeatable).",
    ),
    kwargs: list[str] = typer.Option(
        [],
        "--kw",
        help="Additional keyword args as key=value (repeatable).",
    ),
    hint: str | None = typer.Option(
        None,
        "-h",
        "--hint",
        help="Natural language hint to auto-select plot type and parameters.",
    ),
    type: str | None = typer.Option(  # noqa: A002
        None,
        "--type",
        help="Fixed plot type (line, bar, hist, scatter, corr).",
    ),
    x: str | None = typer.Option(None, "-x", "--x", help="X-axis column."),
    y: str | None = typer.Option(None, "-y", "--y", help="Y-axis column."),
    column: str | None = typer.Option(
        None, "-c", "--column", help="Single column for hist or counts."
    ),
    bins: int | None = typer.Option(None, "--bins", help="Bins for histogram."),
    top_n: int | None = typer.Option(
        None, "--top-n", help="Top N categories for bar/counts."
    ),
    normalize: bool | None = typer.Option(
        None, "--normalize", help="Use frequency instead of counts for bar/counts."
    ),
    output: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help=(
            "Output file (png/pdf) or directory. Defaults to"
            " $URIM_HOME/plots/<file>.png"
        ),
        dir_okay=True,
        file_okay=True,
        resolve_path=True,
    ),
    title: str | None = typer.Option(None, "--title", help="Optional plot title."),
    limit: int | None = typer.Option(
        None, "--limit", help="Optional cap on number of plotted points per dataset."
    ),
    method: Literal["pearson", "spearman", "kendall"] = typer.Option(
        "pearson", "--method", help="Correlation method for corr plot."
    ),
    model: str = typer.Option(
        "gpt-4.1", "-m", "--model", help="Model to use for hint-based plotting."
    ),
) -> None:
    graph = UrimDatasetGraph.from_file()

    datasets: list[tuple[str, Dataset]] = []
    if not dataset or len(dataset) == 0:
        working_ds_id = graph.working_dataset
        if ctx.invoked_subcommand not in {None, "status"}:
            assert (
                working_ds_id is not None
            ), "No dataset loaded, either set a working dataset or pass one via -n."
        if working_ds_id is not None:
            label = labels[0] if labels else working_ds_id
            _, ds = Dataset.load_from_id(working_ds_id)
            datasets = [(label, ds)]
        else:
            ctx.obj = PlotContext(datasets=[])
            if ctx.invoked_subcommand is None:
                ctx.invoke(status, ctx)
            return
    else:
        parsed_kwargs: dict[str, Any] = parse_kv(kwargs or [])
        for idx, name in enumerate(dataset):
            ds_label = labels[idx] if idx < len(labels) else None
            label_kwargs = {**parsed_kwargs}
            if ds_label is not None:
                label_kwargs["label"] = ds_label
            datasets.append(
                _load_dataset_named(
                    name,
                    data_dir=data_dir,
                    cache_dir=cache_dir,
                    token=token,
                    split=split,
                    num_proc=num_proc,
                    subset=subset,
                    **label_kwargs,
                )
            )

    ctx.obj = PlotContext(datasets=datasets)
    if ctx.invoked_subcommand is None:
        wants_auto = any(
            v is not None
            for v in [hint, type, x, y, column, bins, top_n, normalize, title, limit]
        )
        if wants_auto:
            _auto_plot(
                ctx,
                hint=hint,
                plot_type=type,
                x=x,
                y=y,
                column=column,
                bins=bins,
                top_n=top_n,
                normalize=normalize,
                title=title,
                output=output,
                limit=limit,
                method=method,
                model=model,
            )
        else:
            ctx.invoke(status, ctx)


@plot_app.command()
def status(ctx: typer.Context) -> None:
    ctx_obj: PlotContext = ctx.obj
    if not ctx_obj or len(ctx_obj.datasets) == 0:
        RichLogger.info("No dataset(s) selected. Use -n/--name to load one or more.")
        return

    names = [label for label, _ in ctx_obj.datasets]
    sizes = [len(ds.df()) for _, ds in ctx_obj.datasets]
    cols = [", ".join(map(str, ds.df().columns)) for _, ds in ctx_obj.datasets]
    RichLogger.table(Labels=names, Rows=sizes, Columns=cols)


@plot_app.command()
def hist(
    ctx: typer.Context,
    column: str = typer.Option(..., "-c", "--column", help="Column to plot."),
    bins: int = typer.Option(30, "--bins", help="Number of bins."),
    output: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help=(
            "Output file (png/pdf) or directory. Defaults to"
            " $URIM_HOME/plots/<file>.png"
        ),
        dir_okay=True,
        file_okay=True,
        resolve_path=True,
    ),
    title: str | None = typer.Option(None, "--title", help="Optional plot title."),
) -> None:
    datasets = [Dataset.load_from_id(ds_id) for ds_id in dataset_ids]
    for ds_id in dataset_ids:

    ctx_obj: PlotContext = ctx.obj
    assert ctx_obj and len(ctx_obj.datasets) > 0
    sns, plt = _import_seaborn()

    combined_rows: list[dict[str, Any]] = []
    for label, ds in ctx_obj.datasets:
        df = ds.df()
        if column not in df:
            RichLogger.warning(f"Column '{column}' not in dataset '{label}', skipping.")
            continue
        series = df[column].dropna()
        if not pd.api.types.is_numeric_dtype(series):
            RichLogger.warning(
                f"Column '{column}' in dataset '{label}' is not numeric; skipping."
            )
            continue
        for v in series.tolist():
            combined_rows.append({"value": float(v), "dataset": label})

    assert len(combined_rows) > 0, f"No numeric data found for column '{column}'."
    plot_df = pd.DataFrame(combined_rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        data=plot_df,
        x="value",
        hue="dataset",
        bins=bins,
        multiple="layer",
        element="step",
        alpha=0.4,
        ax=ax,
    )
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    if title:
        ax.set_title(title)
    ax.legend(title="dataset")

    path = _resolve_output_path(output, filename=f"hist_{column}")
    fig.tight_layout()
    fig.savefig(path.as_posix())
    RichLogger.success(f"Saved histogram to {path}")


@plot_app.command()
def value_counts(
    ctx: typer.Context,
    column: str = typer.Option(..., "-c", "--column", help="Categorical column."),
    top_n: int = typer.Option(20, "--top-n", help="Top N categories to show."),
    normalize: bool = typer.Option(
        False, "--normalize", help="Plot frequency instead of counts."
    ),
    output: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output file or directory.",
        dir_okay=True,
        file_okay=True,
        resolve_path=True,
    ),
    title: str | None = typer.Option(None, "--title", help="Optional plot title."),
) -> None:
    ctx_obj: PlotContext = ctx.obj
    assert ctx_obj and len(ctx_obj.datasets) > 0
    sns, plt = _import_seaborn()

    data: list[dict[str, Any]] = []
    for label, ds in ctx_obj.datasets:
        df = ds.df()
        if column not in df:
            RichLogger.warning(f"Column '{column}' not in dataset '{label}', skipping.")
            continue
        vc = df[column].astype("string").value_counts(normalize=normalize).head(top_n)
        for k, v in vc.items():
            data.append({"dataset": label, "category": str(k), "value": float(v)})

    assert len(data) > 0, f"No data found for column '{column}'."
    plot_df = pd.DataFrame(data)

    categories = plot_df["category"].unique().tolist()
    _ = plot_df["dataset"].unique().tolist()
    fig, ax = plt.subplots(figsize=(max(8, len(categories) * 0.6), 6))
    sns.barplot(data=plot_df, x="category", y="value", hue="dataset", ax=ax)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_ylabel("Frequency" if normalize else "Count")
    if title:
        ax.set_title(title)
    ax.legend(title="dataset")

    path = _resolve_output_path(output, filename=f"value_counts_{column}")
    fig.tight_layout()
    fig.savefig(path.as_posix())
    RichLogger.success(f"Saved value counts to {path}")


@plot_app.command()
def scatter(
    ctx: typer.Context,
    x: str = typer.Option(..., "--x", help="X-axis column."),
    y: str = typer.Option(..., "--y", help="Y-axis column."),
    output: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output file or directory.",
        dir_okay=True,
        file_okay=True,
        resolve_path=True,
    ),
    title: str | None = typer.Option(None, "--title", help="Optional plot title."),
    limit: int | None = typer.Option(None, "--limit", help="Max points per dataset."),
) -> None:
    ctx_obj: PlotContext = ctx.obj
    assert ctx_obj and len(ctx_obj.datasets) > 0
    sns, plt = _import_seaborn()

    rows: list[dict[str, Any]] = []
    for label, ds in ctx_obj.datasets:
        df = ds.df()
        if x not in df or y not in df:
            RichLogger.warning(
                f"Columns '{x}', '{y}' not in dataset '{label}', skipping."
            )
            continue
        sub = df[[x, y]].dropna()
        if limit is not None and len(sub) > limit:
            sub = sub.sample(n=limit, random_state=0)
        for xv, yv in sub[[x, y]].itertuples(index=False):
            rows.append({"x": xv, "y": yv, "dataset": label})

    assert len(rows) > 0, f"No valid numeric data for '{x}' vs '{y}'."
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(data=plot_df, x="x", y="y", hue="dataset", ax=ax, s=18)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if title:
        ax.set_title(title)
    ax.legend(title="dataset")

    path = _resolve_output_path(output, filename=f"scatter_{x}_vs_{y}")
    fig.tight_layout()
    fig.savefig(path.as_posix())
    RichLogger.success(f"Saved scatter to {path}")


@plot_app.command()
def line(
    ctx: typer.Context,
    x: str = typer.Option(..., "-x", "--x", help="X-axis column."),
    y: str | None = typer.Option(
        None, "-y", "--y", help="Y column. If omitted, will count per x."
    ),
    aggregate: str = typer.Option(
        "mean",
        "--aggregate",
        help="Aggregation when multiple rows share the same x (mean/sum/count).",
    ),
    output: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output file or directory.",
        dir_okay=True,
        file_okay=True,
        resolve_path=True,
    ),
    title: str | None = typer.Option(None, "--title", help="Optional plot title."),
    limit: int | None = typer.Option(None, "--limit", help="Max points per dataset."),
) -> None:
    ctx_obj: PlotContext = ctx.obj
    assert ctx_obj and len(ctx_obj.datasets) > 0
    sns, plt = _import_seaborn()

    long_rows: list[dict[str, Any]] = []
    for label, ds in ctx_obj.datasets:
        df = ds.df()
        if x not in df or (y is not None and y not in df):
            RichLogger.warning(
                f"Columns '{x}'{'' if y is None else f', {y}'} not in dataset"
                f" '{label}', skipping."
            )
            continue

        sub = df[[x, y]].dropna() if y is not None else df[[x]].dropna()
        if limit is not None and len(sub) > limit:
            sub = sub.sample(n=limit, random_state=0)

        if y is None:
            grouped = sub.groupby(x).size()
            grouped = grouped.reset_index()
            grouped.rename(columns={0: "value", x: x}, inplace=True)
        else:
            if aggregate == "sum":
                grouped = sub.groupby(x)[y].sum()
            elif aggregate == "count":
                grouped = sub.groupby(x)[y].count()
            else:
                grouped = sub.groupby(x)[y].mean()
            grouped = grouped.reset_index()
            grouped.rename(columns={y: "value"}, inplace=True)
        for xv, val in grouped[[x, "value"]].itertuples(index=False):
            long_rows.append({x: xv, "value": val, "dataset": label})

    assert len(long_rows) > 0, "No valid series to plot."
    plot_df = pd.DataFrame(long_rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=plot_df, x=x, y="value", hue="dataset", ax=ax)
    ax.set_xlabel(x)
    if y is not None:
        ax.set_ylabel(y)
    if title:
        ax.set_title(title)
    ax.legend(title="dataset")

    path = _resolve_output_path(
        output, filename=f"line_{x}{'' if y is None else f'_{y}'}"
    )
    fig.tight_layout()
    fig.savefig(path.as_posix())
    RichLogger.success(f"Saved line plot to {path}")


@plot_app.command()
def bar(
    ctx: typer.Context,
    x: str = typer.Option(..., "-x", "--x", help="X-axis category column."),
    y: str | None = typer.Option(
        None, "-y", "--y", help="Y column. If omitted, counts per category are shown."
    ),
    aggregate: str = typer.Option(
        "mean",
        "--aggregate",
        help="Aggregation for y when present (mean/sum/count).",
    ),
    top_n: int = typer.Option(20, "--top-n", help="Top-N categories to display."),
    normalize: bool = typer.Option(
        False, "--normalize", help="Normalize counts to frequencies when y is None."
    ),
    output: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output file or directory.",
        dir_okay=True,
        file_okay=True,
        resolve_path=True,
    ),
    title: str | None = typer.Option(None, "--title", help="Optional plot title."),
) -> None:
    ctx_obj: PlotContext = ctx.obj
    assert ctx_obj and len(ctx_obj.datasets) > 0
    sns, plt = _import_seaborn()

    data: list[dict[str, Any]] = []
    for label, ds in ctx_obj.datasets:
        df = ds.df()
        if x not in df or (y is not None and y not in df):
            RichLogger.warning(
                f"Columns '{x}'{'' if y is None else f', {y}'} not in dataset"
                f" '{label}', skipping."
            )
            continue

        sub = df[[x, y]].dropna() if y is not None else df[[x]].dropna()
        if y is None:
            vc = sub[x].astype("string").value_counts(normalize=normalize).head(top_n)
            for k, v in vc.items():
                data.append({"dataset": label, "category": str(k), "value": float(v)})
        else:
            if aggregate == "sum":
                grouped = sub.groupby(x)[y].sum()
            elif aggregate == "count":
                grouped = sub.groupby(x)[y].count()
            else:
                grouped = sub.groupby(x)[y].mean()
            grouped = grouped.sort_values(ascending=False).head(top_n)
            for k, v in grouped.items():
                data.append({"dataset": label, "category": str(k), "value": float(v)})

    assert len(data) > 0, "No data to plot."
    plot_df = pd.DataFrame(data)
    categories = plot_df["category"].unique().tolist()
    _ = plot_df["dataset"].unique().tolist()

    fig, ax = plt.subplots(figsize=(max(8, len(categories) * 0.6), 6))
    sns.barplot(data=plot_df, x="category", y="value", hue="dataset", ax=ax)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_ylabel("Frequency" if y is None and normalize else (y or "Value"))
    if title:
        ax.set_title(title)
    ax.legend(title="dataset")

    path = _resolve_output_path(
        output, filename=f"bar_{x}{'' if y is None else f'_{y}'}"
    )
    fig.tight_layout()
    fig.savefig(path.as_posix())
    RichLogger.success(f"Saved bar chart to {path}")


@plot_app.command()
def corr(
    ctx: typer.Context,
    output: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help=(
            "Output file or directory. For multiple datasets, files are suffixed by"
            " label."
        ),
        dir_okay=True,
        file_okay=True,
        resolve_path=True,
    ),
    method: Literal["pearson", "spearman", "kendall"] = typer.Option(
        "pearson",
        "--method",
        help="Correlation method: pearson/spearman/kendall.",
    ),
) -> None:
    ctx_obj: PlotContext = ctx.obj
    assert ctx_obj and len(ctx_obj.datasets) > 0
    sns, plt = _import_seaborn()

    for label, ds in ctx_obj.datasets:
        df = ds.df().select_dtypes(include=["number"]).copy()
        assert not df.empty, f"Dataset '{label}' has no numeric columns."
        corr_df = df.corr(method=method)

        fig, ax = plt.subplots(figsize=(max(6, len(corr_df.columns) * 0.6), 6))
        sns.heatmap(corr_df, cmap="coolwarm", vmin=-1, vmax=1, cbar=True, ax=ax)
        ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr_df.columns)
        ax.set_title(f"Correlation ({method}) - {label}")

        fname = f"corr_{method}_{label}"
        path = _resolve_output_path(output, filename=fname)
        fig.tight_layout()
        fig.savefig(path.as_posix())
        RichLogger.success(f"Saved correlation heatmap to {path}")


@plot_app.command()
def head(
    ctx: typer.Context,
    n: int = typer.Option(10, "-n", "--n", help="Number of rows to show."),
) -> None:
    ctx_obj: PlotContext = ctx.obj
    assert ctx_obj and len(ctx_obj.datasets) > 0
    for label, ds in ctx_obj.datasets:
        RichLogger.decorated_print(subject="Dataset", body=label)
        RichLogger.print_dataframe(ds.df().head(n))


@plot_app.command()
def export(
    ctx: typer.Context,
    out: Path = typer.Argument(
        ..., help="Export first dataset to JSONL at this path.", dir_okay=False
    ),
) -> None:
    ctx_obj: PlotContext = ctx.obj
    assert ctx_obj and len(ctx_obj.datasets) > 0
    _, ds = ctx_obj.datasets[0]
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_json(out.as_posix())
    RichLogger.success(f"Exported dataset to {out}")
