from pathlib import Path

import seaborn as sns  # type: ignore
import typer
from matplotlib import pyplot as plt

from urim.plots import concat_with_groups

plot_app = typer.Typer(help="Plot utilities: visualize one or more datasets.")


def _out_results(output: Path | None) -> None:
    if output is None:
        plt.show()
    else:
        plt.savefig(output.as_posix())
    plt.close()


@plot_app.command()
def hist(
    dataset_ids: list[str] = typer.Argument(
        ..., help="Dataset ids to pass in to plot function."
    ),
    column: str = typer.Option(
        ..., "-c", "--column", help="Column to plot for y-axis."
    ),
    bins: int | None = typer.Option(None, "--bins", help="Number of bins."),
    output: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output file or directory.",
    ),
) -> None:
    cat_columns, concat_ds = concat_with_groups(dataset_ids)
    hue: str | None = None
    if len(cat_columns) > 0:
        hue = cat_columns[0]

    sns.histplot(
        concat_ds.df(),
        x=column,
        hue=column if hue is None else hue,
        multiple="stack",
        bins=bins if bins is not None else "auto",
        palette="pastel",
    )

    _out_results(output)


@plot_app.command()
def scatter(
    dataset_ids: list[str] = typer.Argument(
        ..., help="Dataset ids to pass in to plot function."
    ),
    x: str = typer.Option(..., "-x", "--x", help="X-axis column."),
    y: str = typer.Option(..., "-y", "--y", help="Y-axis column."),
    output: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output file or directory.",
    ),
) -> None:
    cat_columns, concat_ds = concat_with_groups(dataset_ids)
    hue: str | None = None
    if len(cat_columns) > 0:
        hue = cat_columns[0]

    sns.scatterplot(concat_ds.df(), x=x, y=y, hue=hue, palette="pastel")

    _out_results(output)


@plot_app.command()
def bar(
    dataset_ids: list[str] = typer.Argument(
        ..., help="Dataset ids to pass in to plot function."
    ),
    column: str = typer.Option(
        None, "-c", "--column", help="Column to plot for y-axis."
    ),
    output: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output file or directory.",
    ),
) -> None:
    cat_columns, concat_ds = concat_with_groups(dataset_ids)
    assert (
        len(cat_columns) > 0
    ), "At least one categorical column is required for bar plots."

    hue: str | None = None
    cats: str = ""

    if len(cat_columns) > 1:
        hue, cats = cat_columns[:2]
    else:
        cats = cat_columns[0]

    sns.barplot(
        concat_ds.df(),
        x=cats,
        y=column,
        errorbar=("pi", 95),
        palette="deep",
        hue=cats if hue is None else hue,
    )
    _out_results(output)


@plot_app.command()
def line(
    dataset_ids: list[str] = typer.Argument(
        ..., help="Dataset ids to pass in to plot function."
    ),
    x: str = typer.Option(..., "-x", "--x", help="X-axis column."),
    y: str = typer.Option(..., "-y", "--y", help="Y-axis column."),
    output: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output file or directory.",
    ),
) -> None:
    cat_columns, concat_ds = concat_with_groups(dataset_ids)
    assert (
        len(cat_columns) > 0
    ), "At least one categorical column is required for line plots."

    hue: str | None = None
    if len(cat_columns) > 1: