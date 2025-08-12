from __future__ import annotations

from enum import Enum
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]
from rich import box
from rich import print as prettyprint
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)
from rich.table import Column, Table
from rich.tree import Tree

from urim.env import UrimDatasetGraph


class Colors(Enum):
    SUCCESS = "green"
    ERROR = "red"
    WARNING = "yellow"
    DEBUG = "cyan"
    INFO = "blue"
    QUESTION = "magenta"
    DEFAULT = "white"


class LogLevel(Enum):
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3


class RichLogger:
    level: LogLevel = LogLevel.INFO

    @classmethod
    def decorated_print(
        cls,
        subject: str = "",
        subject_color: Colors = Colors.SUCCESS,
        body: str = "",
        body_color: Colors = Colors.DEFAULT,
        first_emoji: str = "",
        last_emoji: str = "",
    ) -> None:
        parts: list[str] = []
        if first_emoji:
            parts.append(first_emoji)
        if subject:
            parts.append(
                f"[bold {subject_color.value}]{subject}[/bold {subject_color.value}]"
            )
        if body:
            parts.append(f"[{body_color.value}]{body}[/{body_color.value}]")
        if last_emoji:
            parts.append(last_emoji)

        prettyprint(" ".join(parts))

    @classmethod
    def info(cls, message: str) -> None:
        if cls.level.value > LogLevel.INFO.value:
            return

        cls.decorated_print(
            subject="INFO",
            body=message,
            subject_color=Colors.INFO,
        )

    @classmethod
    def warning(cls, message: str) -> None:
        if cls.level.value > LogLevel.WARNING.value:
            return

        cls.decorated_print(
            subject="WARNING",
            body=message,
            subject_color=Colors.WARNING,
        )

    @classmethod
    def error(cls, message: str) -> None:
        if cls.level.value > LogLevel.ERROR.value:
            return

        cls.decorated_print(
            subject="ERROR",
            body=message,
            subject_color=Colors.ERROR,
        )

    @classmethod
    def success(cls, message: str) -> None:
        cls.decorated_print(
            subject="SUCCESS",
            body=message,
            subject_color=Colors.SUCCESS,
            first_emoji="✅",
        )

    @classmethod
    def debug(cls, message: str) -> None:
        if cls.level.value > LogLevel.DEBUG.value:
            return

        cls.decorated_print(
            subject="DEBUG",
            body=message,
            subject_color=Colors.DEBUG,
        )

    @classmethod
    def progress_bar(cls) -> Progress:
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=80, table_column=Column(ratio=10)),
            TextColumn("{task.completed}/{task.total}"),
        )

    @classmethod
    def table(
        cls,
        title: str | None = None,
        max_width: int | None = None,
        **columns,
    ) -> None:
        column_names = list(columns.keys())
        table = Table(title=title, box=box.SIMPLE_HEAVY, show_lines=False)

        for name in column_names:
            table.add_column(str(name), justify="left", overflow="fold")

        max_rows = 0
        for values in columns.values():
            max_rows = max(max_rows, len(values))

        for row_idx in range(max_rows):
            row_values: list[str] = []
            for name in column_names:
                values = columns.get(name, [])
                value = values[row_idx] if row_idx < len(values) else ""
                cell = "" if value is None else str(value)
                if max_width is not None and len(cell) > max_width:
                    cell = cell[: max_width - 1] + "…"
                row_values.append(cell)
            table.add_row(*row_values)

        prettyprint(table)

    @classmethod
    def print_dataframe(
        cls, df: pd.DataFrame, title: str | None = None, show_index: bool = True
    ) -> None:
        columns: dict[str, list[Any]] = {}
        if show_index:
            index_name = df.index.name or "index"
            columns[index_name] = ["" if v is None else v for v in df.index.tolist()]

        for col in df.columns:
            columns[col] = ["" if v is None else v for v in df[col].tolist()]

        cls.table(title, max_width=None, **columns)

    @classmethod
    def print_ds_history(
        cls, graph: UrimDatasetGraph | None = None, node: str | None = None
    ) -> None:
        graph = graph or UrimDatasetGraph.from_file()
        node = node or graph.working_dataset

        assert node is not None

        idx, ds, cmd = cast(tuple[list, list, list], ([], [], []))
        path_from_root = graph.path_from_root(node)
        for i, node_id in enumerate(path_from_root):
            node_obj = graph.get_node(node_id)
            assert node_obj is not None

            idx.append(i)
            ds.append(node_id)
            cmd.append(node_obj.command or "")

        cls.table(
            max_width=100,
            Index=idx,
            Dataset=ds,
            Command=cmd,
        )

    @classmethod
    def print_ds_status(
        cls,
        graph: UrimDatasetGraph | None = None,
        *,
        rich: bool = False,
    ) -> None:
        graph = graph or UrimDatasetGraph.from_file()

        wd = graph.working_dataset
        assert (
            wd is not None
        ), "No working dataset is set. Run with -n first to load a new dataset."

        cls.decorated_print(
            first_emoji=":star:",
            subject="Working Dataset",
            body=wd,
            subject_color=Colors.INFO,
        )

        if not rich:
            return

        roots = [
            node_id for node_id, node in graph.graph.items() if node.parent is None
        ]

        forest = Tree(f":deciduous_tree: [{Colors.QUESTION.value}]Datasets")

        def add_children(parent_tree: Tree, parent_id: str) -> None:
            for child_id in graph.children(parent_id):
                child_tree = parent_tree.add(child_id)
                add_children(child_tree, child_id)

        for root_id in sorted(roots):
            root_tree = forest.add(root_id)
            add_children(root_tree, root_id)

        prettyprint(forest)
