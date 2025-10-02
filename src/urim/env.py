import os
from pathlib import Path

URIM_HOME = Path(os.environ.get("URIM_HOME", os.path.expanduser("~/.urim")))
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
CUSTOM_BASE_URL = os.environ.get("CUSTOM_BASE_URL")
