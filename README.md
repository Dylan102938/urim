# urim

CLI utilities for LLM research: quick dataset creation, evaluations, chat, and inference.

## Features
- Dataset: create and inspect datasets
- Eval: run evaluation tasks and summarize results
- Chat: quick single-turn chat
- Infer: single and batch inference stubs

This is an initial scaffold with best practices: `src/` layout, `typer` CLI, `ruff` lint/format, `pytest` tests, and `uv` package management.

## Quickstart (using `uv`)

Prerequisites: Python 3.10+ and `uv` installed.

```bash
# Install dev dependencies
uv sync --dev

# Run CLI help
uv run urim --help

# Run tests
uv run pytest -q

# Lint and format
uv run ruff check .
uv run ruff format .

# Build wheel
uv build
```

## CLI Overview

```bash
urim --help

urim version
urim dataset create <name> --source <spec> [--limit N]
urim dataset inspect <path>
urim eval run <task> --model <id>
urim eval summarize <run-id>
urim chat quick "hello" --model <id>
urim infer single "prompt" --model <id>
urim infer batch <path>
```

## Project Structure

```
src/urim/
  cli.py           # Typer root app and subcommand wiring
  commands/        # Subcommands: dataset, eval, chat, infer
  logging_utils.py # Rich logging setup
  config.py        # Global CLI options container
  version.py       # __version__ source of truth
```

## Contributing
- Use `uv sync --dev` to install dev deps
- Ensure `ruff check .` passes
- Ensure `pytest` is green
- Submit small, focused PRs

## License
MIT