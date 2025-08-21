from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import Enum
from typing import Any, Literal, cast

import pandas as pd
from rich import box
from rich.console import Console, JustifyMethod, RenderableType
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.style import StyleType
from rich.table import Table
from rich.text import Text
from rich.tree import Tree


class Colors(Enum):
    SUCCESS = "green"
    ERROR = "red"
    WARNING = "yellow"
    DEBUG = "cyan"
    INFO = "blue"
    QUESTION = "magenta"
    DEFAULT = "white"
    PRIMARY = "#87AFA3"
    SECONDARY = "#B5A46D"


class RichLogger:
    def __init__(
        self,
        level: logging._Level = logging.INFO,
    ) -> None:
        self.console = Console()
        self.handler = RichHandler(
            console=self.console,
            show_path=False,
            markup=False,
            show_time=True,
            rich_tracebacks=True,
        )

        logging.getLogger().disabled = True

        self._logger = logging.getLogger("urim")
        self._logger.setLevel(level)
        self._logger.addHandler(self.handler)
        self._logger.propagate = False

        self._pbar_ctx: Progress | None = None

    def setLevel(self, level: logging._Level) -> None:
        self._logger.setLevel(level)

    def print(self, *args: Any, **kwargs: Any) -> None:
        self.console.print(*args, **kwargs)
        if self._pbar_ctx is not None:
            self._pbar_ctx.refresh()

    def info(self, message: str, **kwargs: Any) -> None:
        self._logger.info(message, **kwargs)
        if self._pbar_ctx is not None:
            self._pbar_ctx.refresh()

    def warning(self, message: str, **kwargs: Any) -> None:
        self._logger.warning(message, **kwargs)
        if self._pbar_ctx is not None:
            self._pbar_ctx.refresh()

    def error(self, message: str, **kwargs: Any) -> None:
        self._logger.error(message, **kwargs)
        if self._pbar_ctx is not None:
            self._pbar_ctx.refresh()

    def success(self, message: str, **kwargs: Any) -> None:
        self._logger.info(message, **kwargs)
        if self._pbar_ctx is not None:
            self._pbar_ctx.refresh()

    def debug(self, message: str, **kwargs: Any) -> None:
        self._logger.debug(message, **kwargs)
        if self._pbar_ctx is not None:
            self._pbar_ctx.refresh()

    @contextmanager
    def pbar(self) -> Iterator[Progress]:
        progress = Progress(
            TimeElapsedColumn(),
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=self.console,
            transient=True,
            refresh_per_second=12,
        )
        self._pbar_ctx = progress
        try:
            with progress:
                yield progress
        finally:
            self._pbar_ctx = None

    def table(
        self,
        title: str,
        *,
        max_cell_width: int = 100,
        **columns: Any,
    ) -> None:
        column_names = list(columns.keys())
        assert len(column_names) > 0, "Must provide at least one column"
        n_rows = len(columns[column_names[0]])
        assert all(len(columns[name]) == n_rows for name in column_names), (
            "All columns must have the same number of rows"
        )

        table = Table(
            title=title,
            box=box.ASCII_DOUBLE_HEAD,
            title_style=f"bold {Colors.PRIMARY.value}",
            title_justify="left",
        )

        for name in column_names:
            justify: JustifyMethod = "left"
            style: StyleType | None = None
            if isinstance(columns[name][0], int | float):
                justify = "right"
                style = "bold cyan"

            table.add_column(str(name), overflow="fold", justify=justify, style=style)

        for row_idx in range(len(columns[column_names[0]])):
            row_values: list[str] = []
            for name in column_names:
                value = columns.get(name, [])[row_idx]
                cell = "" if value is None else str(value)
                if max_cell_width and len(cell) > max_cell_width:
                    side_len = max_cell_width // 2
                    cell = cell[:side_len] + " … " + cell[-side_len:]
                row_values.append(cell)

            table.add_row(*row_values)

        self.console.print(table)

    def df(
        self,
        title: str,
        df: pd.DataFrame,
        *,
        overflow_strategy: Literal["head", "tail", "sample"] = "head",
        max_rows: int = 10,
        max_cell_width: int = 100,
    ) -> None:
        if overflow_strategy == "head":
            print_df = df.head(n=max_rows)
        elif overflow_strategy == "tail":
            print_df = df.tail(n=max_rows)
        elif overflow_strategy == "sample":
            print_df = df.sample(n=max_rows)

        columns = cast(dict[str, list[Any]], print_df.to_dict(orient="list"))
        self.table(
            title,
            max_cell_width=max_cell_width,
            index=print_df.index.tolist(),
            **columns,
        )

    def forest(
        self,
        title: str,
        forest: dict[str, list[str | tuple[str, Any]]],
        *,
        transform_fn: Callable[[str], RenderableType] | None = None,
    ) -> None:
        roots = set(forest.keys())
        for children in forest.values():
            for child in children:
                key = child if isinstance(child, str) else child[0]
                if key in roots:
                    roots.remove(key)

        f = Tree(Text(title, style=f"bold {Colors.PRIMARY.value}"))

        def add_children(curr_forest_node: Tree, node_id: str) -> None:
            children = forest.get(node_id, [])
            for child in children:
                key = child if isinstance(child, str) else child[0]
                value = child if isinstance(child, str) else child[1]
                if transform_fn is not None:
                    value = transform_fn(value)

                next_node = curr_forest_node.add(value)
                add_children(next_node, key)

        for root in roots:
            root_tree = f.add(Text(root, style=f"bold {Colors.SECONDARY.value}"))
            add_children(root_tree, root)

        self.console.print(f)


logger = RichLogger(level=logging.INFO)


def setup_rich_logging(verbosity: int = 0) -> None:
    global logger

    if verbosity <= -1:
        logger.setLevel(logging.DEBUG)
    elif verbosity == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
