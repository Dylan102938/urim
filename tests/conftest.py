from __future__ import annotations

import os
from collections.abc import Iterator

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:  # type: ignore[name-defined]
    parser.addoption(
        "--requires-llm",
        action="store_true",
        default=False,
        help="Enable tests that require LLM (network/prompt).",
    )


def _is_llm_enabled_from_config(config: pytest.Config) -> bool:  # type: ignore[name-defined]
    env_flag = os.getenv("URIM_TEST_REQUIRES_LLM")
    cli_flag = bool(config.getoption("--requires-llm"))
    return cli_flag or (env_flag == "1" or (env_flag or "").lower() == "true")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:  # type: ignore[name-defined]
    if _is_llm_enabled_from_config(config):
        return

    skip_marker = pytest.mark.skip(
        reason=(
            "requires_llm disabled. Enable with --requires-llm or"
            " URIM_TEST_REQUIRES_LLM=1"
        )
    )
    for item in items:
        if "requires_llm" in item.keywords:
            item.add_marker(skip_marker)


def requires_llm_test(func):  # noqa: ANN001
    """Decorator to mark tests that require LLM.

    Usage:
        @requires_llm
        def test_something(): ...
    """
    return pytest.mark.requires_llm(func)


# Back-compat: keep the fixture if some tests still import it
@pytest.fixture(scope="session")
def requires_llm(request: pytest.FixtureRequest) -> Iterator[bool]:
    yield _is_llm_enabled_from_config(request.config)
