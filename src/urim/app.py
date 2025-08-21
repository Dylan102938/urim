from __future__ import annotations

import typer

from urim.ai.question import _caches
from urim.cli.dataset import dataset_app
from urim.version import __version__

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Urim: CLI utilities for LLM research.",
)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    def cleanup() -> None:
        for cache in _caches.values():
            cache._wp.flush()

    ctx.call_on_close(cleanup)


@app.command()
def version() -> None:
    """Print version and exit."""
    typer.echo(__version__)


# Subcommands
app.add_typer(dataset_app, name="dataset")
