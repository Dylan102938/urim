from __future__ import annotations

import logging
import os
import sys
from collections.abc import Mapping
from typing import TextIO

__all__ = ["configure_logger", "get_logger", "set_module_level"]

_DEFAULT_LOGGER_NAME = "urim"
_CONFIGURED = False


class _ColorFormatter(logging.Formatter):
    _COLORS = {
        logging.DEBUG: "\033[36m",  # cyan
        logging.INFO: "\033[32m",  # green
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",  # red
        logging.CRITICAL: "\033[35m",  # magenta
    }
    _RESET = "\033[0m"

    def __init__(self, *, use_color: bool) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        message = super().format(record)
        if not self._use_color:
            return message
        color = self._COLORS.get(record.levelno)
        if not color:
            return message
        return f"{color}{message}{self._RESET}"


def _normalize_level(level: int | str) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        upper = level.upper()
        if upper.isdigit():
            return int(upper)
        resolved = logging.getLevelName(upper)
        if isinstance(resolved, str):
            raise ValueError(f"Unknown logging level: {level}")
        return resolved
    raise TypeError("Logging level must be an int or str.")


def configure_logger(
    *,
    level: int | str = logging.INFO,
    stream: TextIO | None = None,
    color: bool | None = None,
    force: bool = False,
    module_levels: Mapping[str | None, int | str] | None = None,
) -> None:
    """Configure the root urim logger.

    Parameters
    ----------
    level:
        Logging level as int or name. Defaults to INFO.
    stream:
        Stream to write logs to. Defaults to stdout.
    color:
        Force enable/disable ANSI colors. Defaults to auto (enabled for TTYs).
    force:
        If True, reconfigure even if a configuration already exists.
    """

    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    logger = logging.getLogger(_DEFAULT_LOGGER_NAME)
    logger.handlers.clear()

    stream = stream or sys.stdout
    use_color = color
    if use_color is None:
        is_tty = getattr(stream, "isatty", lambda: False)()
        use_color = is_tty and os.name != "nt"

    handler = logging.StreamHandler(stream)
    handler.setFormatter(_ColorFormatter(use_color=bool(use_color)))

    logger.addHandler(handler)
    logger.setLevel(_normalize_level(level))
    logger.propagate = False
    _CONFIGURED = True

    if module_levels:
        for module_name, module_level in module_levels.items():
            set_module_level(module_name, module_level)


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a namespaced logger, ensuring default configuration exists."""
    if not _CONFIGURED:
        configure_logger()

    full_name = _DEFAULT_LOGGER_NAME if not name else f"{_DEFAULT_LOGGER_NAME}.{name}"
    return logging.getLogger(full_name)


def set_module_level(name: str | None, level: int | str) -> None:
    """Set logging level for a specific urim module."""
    logger = get_logger(name)
    logger.setLevel(_normalize_level(level))
