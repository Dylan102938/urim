from pathlib import Path

import seaborn as sns  # type: ignore
import typer
from matplotlib import pyplot as plt

from urim.plot import (
    concat_datasets,
    infer_categorical_kwargs,
    infer_distribution_kwargs,
    infer_relational_kwargs,
)

plot_app = typer.Typer(help="Plot utilities: visualize one or more datasets.")

DatasetIds = typer.Argument(..., help="Dataset ids to pass in to plot function.")
XCol = typer.Option(None, "-x", "--x", help="X-axis column.")
YCol = typer.Option(None, "-y", "--y", help="Y-axis column.")
Column = typer.Option(
    None, "-c", "--column", help="Column used to group for separate plots."
)
Hue = typer.Option(
    None, "-h", "--hue", help="Column used to group for different colors"
)
Output = typer.Option(None, "-o", "--output", help="Output file or directory.")
Multiple = typer.Option("stack", "--multiple", help="How to handle multiple groups.")
Palette = typer.Option("deep", "--palette", help="Color palette.")
ErrorBar = typer.Option("pi", "--error", help="Error bar type.")
ErrorBarConfidence = typer.Option(95, "--error-ci", help="Error bar confidence.")


def _out_results(output: Path | None) -> None:
    if output is None:
        plt.show()
    else:
        plt.savefig(output.as_posix())
    plt.close()


@plot_app.command()
def hist(
    dataset_ids: list[str] = DatasetIds,
    x: str | None = Column,
    hue: str | None = Hue,
    bins: int | None = typer.Option(None, "--bins", help="Number of bins."),
    output: Path | None = Output,
    multiple: str = Multiple,
    palette: str = Palette,
) -> None:
    concat_ds = concat_datasets(dataset_ids)
    kwargs = infer_distribution_kwargs(
        concat_ds,
        x=x,
        hue=hue,
    )

    sns.displot(
        concat_ds.df(),
        **kwargs,  # type: ignore
        multiple=multiple,  # type: ignore
        bins=bins if bins is not None else "auto",
        palette=palette,
    )

    _out_results(output)


@plot_app.command()
def bar(
    dataset_ids: list[str] = DatasetIds,
    x: str | None = Column,
    y: str | None = YCol,
    hue: str | None = Hue,
    output: Path | None = Output,
    palette: str = Palette,
    error: str | None = ErrorBar,
    error_conf: int = ErrorBarConfidence,
) -> None:
    concat_ds = concat_datasets(dataset_ids)
    kwargs = infer_categorical_kwargs(
        concat_ds,
        x=x,
        y=y,
        hue=hue,
        graph_type="bar",
    )

    sns.catplot(
        concat_ds.df(),
        **kwargs,  # type: ignore
        kind="bar",
        errorbar=None if error is None else (error, error_conf),
        palette=palette,
    )

    _out_results(output)


@plot_app.command()
def line(
    dataset_ids: list[str] = DatasetIds,
    x: str | None = XCol,
    y: str | None = YCol,
    hue: str | None = Hue,
    column: str | None = None,
    output: Path | None = Output,
    palette: str = Palette,
    error: str | None = ErrorBar,
    error_conf: int = ErrorBarConfidence,
) -> None:
    concat_ds = concat_datasets(dataset_ids)
    kwargs = infer_relational_kwargs(
        concat_ds,
        graph_type="line",
        x=x,
        y=y,
        hue=hue,
        column=column,
    )

    sns.relplot(
        concat_ds.df(),
        **kwargs,  # type: ignore
        kind="line",
        palette=palette,
        errorbar=None if error is None else (error, error_conf),
        markers=True,
    )

    _out_results(output)


@plot_app.command()
def scatter(
    dataset_ids: list[str] = DatasetIds,
    x: str = XCol,
    y: str = YCol,
    hue: str | None = Hue,
    column: str | None = Column,
    output: Path | None = Output,
    palette: str = Palette,
) -> None:
    concat_ds = concat_datasets(dataset_ids)
    kwargs = infer_relational_kwargs(
        concat_ds,
        graph_type="scatter",
        x=x,
        y=y,
        hue=hue,
        column=column,
    )

    sns.relplot(
        concat_ds.df(),
        **kwargs,  # type: ignore
        kind="scatter",
        palette=palette,
        markers=True,
    )

    _out_results(output)
