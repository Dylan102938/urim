from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

import pandas as pd  # type: ignore[import-untyped]
from rich import box
from rich import print as prettyprint
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)
from rich.prompt import Prompt
from rich.table import Column, Table

from urim.env import UrimState


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
    def update_working_dataset(cls, new_dataset_path: Path) -> None:
        update_working_dataset = Prompt.ask(
            f":pushpin: [{Colors.QUESTION.value}]Update working"
            f" dataset?[/{Colors.QUESTION.value}]",
            choices=["y", "n"],
            default="y",
        )

        if update_working_dataset == "y":
            UrimState.update("working_dataset", new_dataset_path)

    @classmethod
    def progress_bar(cls) -> Progress:
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=80, table_column=Column(ratio=10)),
            TextColumn("{task.completed}/{task.total}"),
        )

    @classmethod
    def print_output_to_filepath(cls, out: Path) -> None:
        cls.decorated_print(
            subject="Success!",
            body=(
                f"Output written to [bold {Colors.INFO.value}]{out}.[/bold"
                f" {Colors.INFO.value}]"
            ),
            first_emoji=":white_check_mark:",
        )

    @classmethod
    def print_dataframe(
        cls, df: pd.DataFrame, title: str | None = None, show_index: bool = True
    ) -> None:
        table = Table(title=title, box=box.SIMPLE_HEAVY, show_lines=False)
        if show_index:
            table.add_column(df.index.name or "index", justify="right", style="dim")

        for col in df.columns:
            justify: Literal["right", "left"] = (
                "right" if pd.api.types.is_numeric_dtype(df[col]) else "left"
            )
            table.add_column(str(col), justify=justify, overflow="fold")

        for row in df.itertuples(index=show_index, name=None):
            table.add_row(*[("" if v is None else str(v)) for v in row])

        prettyprint(table)
