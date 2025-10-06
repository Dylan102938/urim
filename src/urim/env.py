import os
from collections.abc import Iterable
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

URIM_HOME = Path(os.environ.get("URIM_HOME", os.path.expanduser("~/.urim")))
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
CUSTOM_BASE_URL = os.environ.get("CUSTOM_BASE_URL")

_STORAGE_ROOT: Path = URIM_HOME


def collect_openai_keys(*, explicit_key: str | None = None) -> list[str]:
    keys: list[str] = []
    if explicit_key:
        keys.append(explicit_key)
    primary = os.environ.get("OPENAI_API_KEY")
    if primary:
        keys.append(primary)
    for i in range(0, 10):
        k = os.environ.get(f"OPENAI_API_KEY_{i}")
        if k:
            keys.append(k)
    seen: set[str] = set()
    ordered: list[str] = []
    for k in keys:
        if k not in seen:
            ordered.append(k)
            seen.add(k)

    return ordered


def get_storage_root() -> Path:
    return _STORAGE_ROOT


def set_storage_root(path: str | Path) -> Path:
    global _STORAGE_ROOT
    _STORAGE_ROOT = Path(path)

    return _STORAGE_ROOT


def storage_subdir(*parts: str | Path) -> Path:
    root = get_storage_root()
    dir_path = root.joinpath(*_flatten(parts))
    dir_path.mkdir(parents=True, exist_ok=True)

    return dir_path


def storage_file(*parts: str | Path) -> Path:
    root = get_storage_root()
    file_path = root.joinpath(*_flatten(parts))
    file_path.parent.mkdir(parents=True, exist_ok=True)

    return file_path


def _flatten(parts: Iterable[str | Path]) -> list[str | Path]:
    flattened: list[str | Path] = []
    for part in parts:
        if isinstance(part, list | tuple):
            flattened.extend(part)
        else:
            flattened.append(part)

    return flattened
