import os
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel

URIM_HOME = Path(os.environ.get("URIM_HOME", os.path.expanduser("~/.urim")))
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENROUTER_BASE_URL = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
CUSTOM_BASE_URL = os.environ.get("CUSTOM_BASE_URL")


class UrimState(BaseModel):
    working_dataset: Path | None = None

    def serialize(self) -> None:
        state_filepath = URIM_HOME / "state.json"
        with open(state_filepath, "w") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def from_state_file(cls) -> Self:
        state_filepath = URIM_HOME / "state.json"
        if not state_filepath.exists():
            state_filepath.touch()
            return cls()

        with open(state_filepath) as f:
            return cls.model_validate_json(f.read())

    @classmethod
    def update(cls, attr: str, value: Any) -> None:
        assert attr in cls.model_fields, f"Invalid attribute: {attr}"

        state = cls.from_state_file()
        setattr(state, attr, value)
        state.serialize()
