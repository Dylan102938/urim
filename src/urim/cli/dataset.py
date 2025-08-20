from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd
import typer
from rich.text import Text

from urim.ai.question import QuestionFactory, Rating
from urim.cli.utils import parse_kv, random_filestub
from urim.dataset import Dataset
from urim.env import URIM_HOME, UrimDatasetGraph
from urim.logging_utils import Colors, logger

dataset_app = typer.Typer(help="Dataset utilities: creation and inspection.")


def get_ds_id(explicit_id: str | None = None) -> str:
    if explicit_id is not None:
        assert not Path(
            explicit_id
        ).exists(), f"Dataset with id={explicit_id} already exists"
        return explicit_id

    for _ in range(5):
        id = random_filestub()
        if not Path(id).exists():
            return id

    raise RuntimeError(
        "Failed to find a unique dataset id. You may want to consider running urim"
        " clean."
    )


def get_ds_path(id: str) -> Path:
    return URIM_HOME / "datasets" / f"{id}.jsonl"


def create_next_wd(
    graph: UrimDatasetGraph,
    command: str,
    new_dataset: Dataset,
    new_dataset_id: str | None = None,
) -> None:
    assert graph.working_dataset is not None

    new_ds_id = new_dataset_id or get_ds_id()
    new_ds_path = get_ds_path(new_ds_id)

    new_dataset.to_json(new_ds_path.as_posix())
    graph.add_child_and_set_wd(graph.working_dataset, new_ds_id, command=command)
    logger.success(
        f"New dataset created: {new_ds_id}. Working dataset set to"
        f" {URIM_HOME / 'datasets' / new_ds_id}.jsonl."
    )


class DatasetContext:
    graph: UrimDatasetGraph
    dataset: Dataset

    def __init__(
        self,
        graph: UrimDatasetGraph,
        dataset: Dataset,
    ):
        self.graph = graph
        self.dataset = dataset


@dataset_app.callback(invoke_without_command=True)
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
    graph = UrimDatasetGraph.from_file()
    if dataset is None:
        working_ds_id = graph.working_dataset
        if ctx.invoked_subcommand not in {None, "status"}:
            assert working_ds_id is not None, (
                "No dataset loaded, either set a working dataset or pass in a"
                " dataset name via -n."
            )
            _, ds = Dataset.load_from_id(working_ds_id)
        else:
            ctx.obj = DatasetContext(graph=graph, dataset=Dataset(df=pd.DataFrame()))
            if ctx.invoked_subcommand in {None, "status"}:
                ctx.invoke(status, ctx)

            return
    else:
        parsed_kwargs = parse_kv(kwargs or [])
        working_ds_id, ds = Dataset.load(
            dataset,
            data_dir=data_dir,
            cache_dir=cache_dir,
            token=token,
            split=split,
            num_proc=num_proc,
            subset=subset,
            **parsed_kwargs,
        )
        graph.set_working_dataset(working_ds_id)

    ctx.obj = DatasetContext(graph=graph, dataset=ds)


@dataset_app.command()
def status(ctx: typer.Context) -> None:
    ctx_obj: DatasetContext = ctx.obj
    forest: dict[str, set[str]] = defaultdict(set)
    for node_id in ctx_obj.graph.graph:
        path_from_root = ctx_obj.graph.path_from_root(node_id)
        for i, node_id in enumerate(path_from_root):
            if i == 0:
                continue

            forest[path_from_root[i - 1]].add(node_id)

    wd = ctx_obj.graph.working_dataset
    logger.print(
        Text("Current Working Dataset:", style=f"bold {Colors.PRIMARY.value}"),
        wd or "None",
    )
    logger.forest(
        "Dataset Mutations Graph",
        {k: list(v) for k, v in forest.items()},
        transform_fn=lambda text: (
            Text(text, style="bold cyan") if text == wd else Text(text, style="dim")
        ),
    )


@dataset_app.command()
def history(ctx: typer.Context) -> None:
    ctx_obj: DatasetContext = ctx.obj
    assert ctx_obj.graph.working_dataset is not None

    path_from_root = ctx_obj.graph.path_from_root(ctx_obj.graph.working_dataset)
    history: dict[str, Any] = defaultdict(list)
    for i, node_id in enumerate(path_from_root):
        node = ctx_obj.graph.get_node(node_id)
        assert node is not None, f"Invalid dataset id: {node_id}"

        history["order"].append(i)
        history["dataset id"].append(node_id)
        history["command"].append(node.command)

    logger.table("Dataset Mutation History", **history)


@dataset_app.command()
def goto(
    ctx: typer.Context,
    id: str = typer.Argument(
        ..., help="Dataset id or step index from root to navigate to."
    ),
) -> None:
    ctx_obj: DatasetContext = ctx.obj

    try:
        assert ctx_obj.graph.working_dataset is not None

        idx, path_from_root = (
            int(id),
            ctx_obj.graph.path_from_root(ctx_obj.graph.working_dataset),
        )
        assert 0 <= idx < len(path_from_root), f"Invalid step index: {idx}"
        new_wd_id = path_from_root[idx]
    except IndexError as e:
        raise typer.BadParameter(f"Invalid step index: {id}") from e
    except (ValueError, AssertionError):
        node = ctx_obj.graph.get_node(id)
        assert node is not None, f"Invalid dataset id: {id}"
        new_wd_id = id

    ctx_obj.graph.set_working_dataset(new_wd_id)

    logger.success(f"Working dataset set to {new_wd_id}.")


@dataset_app.command()
def back(ctx: typer.Context) -> None:
    ctx_obj: DatasetContext = ctx.obj
    assert ctx_obj.graph.working_dataset is not None

    path_from_root = ctx_obj.graph.path_from_root(ctx_obj.graph.working_dataset)
    if len(path_from_root) <= 1:
        logger.warning("Current working dataset is a root node. No action taken.")
    else:
        ctx.invoke(goto, ctx, id=path_from_root[-2])


@dataset_app.command()
def root(ctx: typer.Context) -> None:
    ctx.invoke(goto, ctx, id=0)


@dataset_app.command()
def export(
    ctx: typer.Context,
    out: Path = typer.Argument(
        ...,
        help="Output filepath. Defaults to a random filestub if not provided.",
        dir_okay=False,
        file_okay=True,
        resolve_path=True,
    ),
) -> None:
    ctx_obj: DatasetContext = ctx.obj

    ds = ctx_obj.dataset
    ds.to_json(out.as_posix())

    logger.success(f"Exported dataset to {out}.")


@dataset_app.command()
def prune(
    ctx: typer.Context,
    root: bool = typer.Option(
        False,
        "-r",
        "--root",
        help="Prune from the root dataset.",
    ),
    from_id: str | None = typer.Option(
        None,
        "-f",
        "--from",
        help="Prune dataset mutation tree from this id.",
    ),
) -> None:
    ctx_obj: DatasetContext = ctx.obj
    assert ctx_obj.graph.working_dataset is not None

    path_from_root = ctx_obj.graph.path_from_root(ctx_obj.graph.working_dataset)
    if root:
        ctx_obj.graph.prune_from_node(path_from_root[0])
    elif from_id is not None:
        try:
            node_idx = int(from_id)
            node_id = path_from_root[node_idx]
        except ValueError:
            node_id = from_id

        ctx_obj.graph.prune_from_node(node_id)
    else:
        ctx_obj.graph.prune_from_node(path_from_root[-1])

    logger.success(
        f"Pruned dataset(s). Set working dataset to {ctx_obj.graph.working_dataset}"
    )


@dataset_app.command()
def print(
    ctx: typer.Context,
    n: int = typer.Option(
        10,
        "-n",
        "--n",
        help="Number of rows to show.",
    ),
    strategy: str = typer.Option(
        "head",
        "-s",
        "--strategy",
        help="Strategy to use for printing the dataset. One of: head, tail, sample.",
    ),
) -> None:
    ctx_obj: DatasetContext = ctx.obj
    assert ctx_obj.graph.working_dataset is not None
    assert strategy in {"head", "tail", "sample"}, f"Invalid strategy: {strategy}"

    logger.df(
        ctx_obj.graph.working_dataset,
        ctx_obj.dataset.df(),
        overflow_strategy=cast(Literal["head", "tail", "sample"], strategy),
        max_rows=n,
    )


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
    ctx_obj: DatasetContext = ctx.obj
    assert n_or_frac > 0, "Must take a postive number of samples from the dataset"

    ds = ctx_obj.dataset

    if n_or_frac > 1:
        ds.sample(n=int(n_or_frac))
    else:
        ds.sample(frac=n_or_frac)

    create_next_wd(ctx_obj.graph, " ".join(sys.argv), ds)

    if out is not None:
        ctx.invoke(export, out)


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
    ctx_obj: DatasetContext = ctx.obj
    ds = ctx_obj.dataset

    ds.rename(columns=parse_kv(columns or []), hint=hint)

    create_next_wd(ctx_obj.graph, " ".join(sys.argv), ds)

    if out is not None:
        ctx.invoke(export, out)


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
    ctx_obj: DatasetContext = ctx.obj

    ds = ctx_obj.dataset
    ds.drop(columns=columns, hint=hint)

    create_next_wd(ctx_obj.graph, " ".join(sys.argv), ds)

    if out is not None:
        ctx.invoke(export, out)


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
    ctx_obj: DatasetContext = ctx.obj

    ds = ctx_obj.dataset
    ds.filter(hint=hint)

    create_next_wd(ctx_obj.graph, " ".join(sys.argv), ds)

    if out is not None:
        ctx.invoke(export, out)


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
    ctx_obj: DatasetContext = ctx.obj

    ds = ctx_obj.dataset

    ds.apply(column=column, hint=hint)

    create_next_wd(ctx_obj.graph, " ".join(sys.argv), ds)

    if out is not None:
        ctx.invoke(export, out)


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
    ctx_obj: DatasetContext = ctx.obj

    ds = ctx_obj.dataset
    _, other_ds = Dataset.load(other)

    valid_hows = {"left", "right", "inner", "outer", "cross"}
    how_lower = how.lower() if how else None
    if how_lower is not None:
        assert how_lower in valid_hows, f"--how must be one of {sorted(valid_hows)}"

    ds.merge(
        other_ds,
        on=on if on else None,
        left_on=left_on,
        right_on=right_on,
        how=how_lower,  # type: ignore[arg-type]
        hint=hint,
    )

    create_next_wd(ctx_obj.graph, " ".join(sys.argv), ds)

    if out is not None:
        ctx.invoke(export, out)


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
    ctx_obj: DatasetContext = ctx.obj

    ds = ctx_obj.dataset
    _, other_ds = Dataset.load(other)

    ds.concat(other_ds, hint=hint)

    create_next_wd(ctx_obj.graph, " ".join(sys.argv), ds)

    if out is not None:
        ctx.invoke(export, out)


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
    ctx_obj: DatasetContext = ctx.obj

    ds = ctx_obj.dataset
    question_kwargs: dict[str, Any] = parse_kv(kwargs or [])

    try:
        if system_prompt and Path(system_prompt).exists():
            system_prompt = Path(system_prompt).read_text()
    except OSError:
        pass

    judge_dict = parse_kv(judges or [])
    for k, template in judge_dict.items():
        judge_dict[k] = (
            QuestionFactory(type=Rating, prompt=template),
            "gpt-4.1",
        )

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

    create_next_wd(ctx_obj.graph, " ".join(sys.argv), ds)

    if out is not None:
        ctx.invoke(export, out)


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
    ctx_obj: DatasetContext = ctx.obj

    ds = ctx_obj.dataset
    ds.describe(hint=hint, model=model)

    create_next_wd(ctx_obj.graph, " ".join(sys.argv), ds)

    if out is not None:
        ctx.invoke(export, out)
