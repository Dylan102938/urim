from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from urim.ai.question import Rating
from urim.cli.utils import (
    parse_kv,
    random_filestub,
)
from urim.dataset import Dataset
from urim.env import URIM_HOME, UrimState
from urim.logging_utils import RichLogger

dataset_app = typer.Typer(help="Dataset utilities: creation and inspection.")


def get_output_path(out: Path | None) -> Path:
    if out is None:
        return URIM_HOME / "datasets" / f"{random_filestub()}.jsonl"

    return out


@dataset_app.callback()
def setup_local_dataset(
    ctx: typer.Context,
    dataset: str | None = typer.Option(
        None,
        "-n",
        "--name",
        help="HuggingFace dataset id (e.g., 'imdb') or a local JSONL path.",
    ),
    split: str = typer.Option(
        "train",
        "--split",
        help="Split to use from the dataset.",
    ),
    subset: str | None = typer.Option(
        None,
        "--subset",
        help="Subset to use from the dataset.",
    ),
    data_dir: str | None = typer.Option(
        None,
        "--data-dir",
        help="Directory to store the dataset.",
    ),
    cache_dir: str | None = typer.Option(
        None,
        "--cache-dir",
        help="Directory to store the dataset cache.",
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
    kwargs: list[str] = typer.Option(
        [],
        "--kw",
        help="Additional keyword args as key=value (repeatable).",
    ),
) -> None:
    if dataset is None:
        dataset = str(UrimState.from_state_file().working_dataset)
        assert (
            dataset is not None
        ), "You need to pass in a dataset name since there is no working dataset."

    parsed_kwargs = parse_kv(kwargs or [])
    ds = Dataset.load(
        dataset,
        data_dir=data_dir,
        cache_dir=cache_dir,
        token=token,
        split=split,
        num_proc=num_proc,
        subset=subset,
        **parsed_kwargs,
    )

    ctx.obj = ds


@dataset_app.command()
def head(
    ctx: typer.Context,
    n: int = typer.Option(
        10,
        "-n",
        "--n",
        help="Number of rows to show.",
    ),
) -> None:
    ds: Dataset = ctx.obj

    RichLogger.print_dataframe(ds.df().head(n))


@dataset_app.command()
def sample(
    ctx: typer.Context,
    n_or_frac: float = typer.Argument(
        ..., help="Absolute number or fraction of rows to sample."
    ),
    out: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output filepath. Defaults to a random filestub if not provided.",
    ),
) -> None:
    assert n_or_frac > 0, "Must take a postive number of samples from the dataset"

    ds: Dataset = ctx.obj
    output_path = get_output_path(out)

    if n_or_frac > 1:
        ds.sample(n=int(n_or_frac))
    else:
        ds.sample(frac=n_or_frac)

    ds.to_json(str(output_path))

    RichLogger.print_output_to_filepath(output_path)
    RichLogger.update_working_dataset(output_path)


@dataset_app.command()
def rename(
    ctx: typer.Context,
    columns: list[str] | None = typer.Option(
        None,
        "-c",
        "--columns",
        help="Columns to rename (repeatable).",
    ),
    hint: str | None = typer.Option(
        None,
        "-h",
        "--hint",
        help="Autogenerate rename map using this hint.",
    ),
    out: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output filepath. Defaults to a random filestub if not provided.",
        dir_okay=False,
        file_okay=True,
        resolve_path=True,
    ),
) -> None:
    ds: Dataset = ctx.obj

    output_path = get_output_path(out)
    rename_map = parse_kv(columns or []) or None

    ds.rename(columns=rename_map, hint=hint)
    ds.to_json(str(output_path))

    RichLogger.print_output_to_filepath(output_path)
    RichLogger.update_working_dataset(output_path)


@dataset_app.command()
def drop(
    ctx: typer.Context,
    columns: list[str] | None = typer.Option(
        None,
        "-c",
        "--columns",
        help="Columns to drop (repeatable).",
    ),
    hint: str | None = typer.Option(
        None,
        "-h",
        "--hint",
        help="Autogenerate drop list using this hint.",
    ),
    out: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output filepath. Defaults to a random filestub if not provided.",
    ),
) -> None:
    ds: Dataset = ctx.obj

    output_path = get_output_path(out)

    ds.drop(columns=columns, hint=hint)
    ds.to_json(str(output_path))

    RichLogger.print_output_to_filepath(output_path)
    RichLogger.update_working_dataset(output_path)


@dataset_app.command()
def filter(
    ctx: typer.Context,
    hint: str | None = typer.Option(
        ...,
        "-h",
        "--hint",
        help="Autogenerate filter function using this hint.",
    ),
    out: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output filepath. Defaults to a random filestub if not provided.",
    ),
) -> None:
    ds: Dataset = ctx.obj

    output_path = get_output_path(out)

    ds.filter(hint=hint)
    ds.to_json(str(output_path))

    RichLogger.print_output_to_filepath(output_path)
    RichLogger.update_working_dataset(output_path)


@dataset_app.command()
def apply(
    ctx: typer.Context,
    column: str | None = typer.Option(
        None,
        "-c",
        "--column",
        help="Column to apply the function to.",
    ),
    hint: str | None = typer.Option(
        None,
        "-h",
        "--hint",
        help="Autogenerate apply function using this hint.",
    ),
    out: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output filepath. Defaults to a random filestub if not provided.",
    ),
) -> None:
    ds: Dataset = ctx.obj

    output_path = get_output_path(out)

    ds.apply(column=column, hint=hint)
    ds.to_json(str(output_path))

    RichLogger.print_output_to_filepath(output_path)
    RichLogger.update_working_dataset(output_path)


@dataset_app.command()
def merge(
    ctx: typer.Context,
    other: str = typer.Option(
        ..., "-b", "--other", help="Other dataset (HF id or local JSONL path)."
    ),
    on: list[str] | None = typer.Option(
        None, "--on", help="Column(s) to join on (repeatable)."
    ),
    left_on: list[str] | None = typer.Option(
        None, "--left-on", help="Column(s) from left dataset to join on (repeatable)."
    ),
    right_on: list[str] | None = typer.Option(
        None, "--right-on", help="Column(s) from right dataset to join on (repeatable)."
    ),
    how: str = typer.Option(
        "left",
        "--how",
        help="Type of merge to be performed (left, right, inner, outer, cross).",
        case_sensitive=False,
    ),
    hint: str | None = typer.Option(
        None, "-h", "--hint", help="Autogenerate merge args using this hint."
    ),
    out: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output filepath. Defaults to a random filestub if not provided.",
    ),
) -> None:
    ds: Dataset = ctx.obj
    other_ds = Dataset.load(other)

    valid_hows = {"left", "right", "inner", "outer", "cross"}
    how_lower = how.lower() if how else None
    if how_lower is not None:
        assert how_lower in valid_hows, f"--how must be one of {sorted(valid_hows)}"

    output_path = get_output_path(out)

    ds.merge(
        other_ds,
        on=on if on else None,
        left_on=left_on,
        right_on=right_on,
        how=how_lower,  # type: ignore[arg-type]
        hint=hint,
    )
    ds.to_json(str(output_path))

    RichLogger.print_output_to_filepath(output_path)
    RichLogger.update_working_dataset(output_path)


@dataset_app.command()
def concat(
    ctx: typer.Context,
    other: str = typer.Option(
        ..., "-b", "--other", help="Other dataset (HF id or local JSONL path)."
    ),
    hint: str | None = typer.Option(
        None, "-h", "--hint", help="Autogenerate concat rename map using this hint."
    ),
    out: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output filepath. Defaults to a random filestub if not provided.",
    ),
) -> None:
    ds: Dataset = ctx.obj
    other_ds = Dataset.load(other)

    output_path = get_output_path(out)

    ds.concat(other_ds, hint=hint)
    ds.to_json(str(output_path))

    RichLogger.print_output_to_filepath(output_path)
    RichLogger.update_working_dataset(output_path)


@dataset_app.command()
def generate(
    ctx: typer.Context,
    question_col: str | None = typer.Option(
        None, "--question-col", help="Name of question text column."
    ),
    messages_col: str | None = typer.Option(
        None, "--messages-col", help="Name of messages (chat) column."
    ),
    out_col: str | None = typer.Option(
        None, "-c", "--out-col", help="Name of output column to write answers to."
    ),
    model: str = typer.Option(
        "gpt-4.1", "-m", "--model", help="Model to use for generation."
    ),
    max_workers: int = typer.Option(
        100, "--max-workers", help="Maximum concurrent workers for generation."
    ),
    system_prompt: str | None = typer.Option(
        None,
        "-s",
        "--system",
        help="System prompt to use for generation. Reads from file if provided.",
    ),
    judges: list[str] = typer.Option(
        [],
        "--judge",
        help="Pass in templates for a Generation Rating judge (repeatable).",
    ),
    out: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output filepath. Defaults to a random filestub if not provided.",
        dir_okay=False,
        file_okay=True,
    ),
    kwargs: list[str] = typer.Option(
        [], "--kw", help="Additional question kwargs as key=value (repeatable)."
    ),
) -> None:
    ds: Dataset = ctx.obj
    output_path = get_output_path(out)
    question_kwargs: dict[str, Any] = parse_kv(kwargs or [])

    if system_prompt and Path(system_prompt).exists():
        system_prompt = Path(system_prompt).read_text()

    judge_dict = parse_kv(judges or [])
    judge_dict = {
        k: (Rating(template), "gpt-4.1") for k, template in judge_dict.items()
    }
    ds.generate(
        question_col=question_col,
        messages_col=messages_col,
        out_col=out_col,
        model=model,
        max_workers=max_workers,
        system=system_prompt,
        judges=judge_dict,
        **question_kwargs,
    )
    ds.to_json(str(output_path))

    RichLogger.print_output_to_filepath(output_path)
    RichLogger.update_working_dataset(output_path)


@dataset_app.command()
def describe(
    ctx: typer.Context,
    hint: str = typer.Option(
        ..., "-h", "--hint", help="Autogenerate describe args using this hint."
    ),
    model: str = typer.Option(
        "gpt-4.1", "-m", "--model", help="Model to use for generation."
    ),
    out: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output filepath. Defaults to a random filestub if not provided.",
    ),
) -> None:
    ds: Dataset = ctx.obj
    ouptut_path = get_output_path(out)

    ds.describe(hint=hint, model=model)
    ds.to_json(str(ouptut_path))

    RichLogger.print_output_to_filepath(ouptut_path)
    RichLogger.update_working_dataset(ouptut_path)
