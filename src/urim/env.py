import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

URIM_HOME = Path(os.environ.get("URIM_HOME", os.path.expanduser("~/.urim")))
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
CUSTOM_BASE_URL = os.environ.get("CUSTOM_BASE_URL")


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
