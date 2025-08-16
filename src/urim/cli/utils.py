from __future__ import annotations

import json
import random
import string
from typing import Any

from urim.env import UrimDatasetGraph


def coerce_value(text: str) -> Any:
    lower = text.strip().lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"null", "none"}:
        return None

    if (
        (text.startswith("[") and text.endswith("]"))
        or (text.startswith("{") and text.endswith("}"))
        or (text.startswith('"') and text.endswith('"'))
    ):
        try:
            return json.loads(text)
        except Exception:
            pass

    try:
        if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
            return int(text)
    except Exception:
        pass

    try:
        return float(text)
    except Exception:
        return text


def parse_args(values: list[str]) -> list[Any]:
    return [coerce_value(v) for v in values]


def parse_kv(pairs: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid --kw value '{pair}'. Expected key=value.")
        key, value = pair.split("=", 1)
        key = key.strip()
        result[key] = coerce_value(value)
    return result


ADJECTIVES = [
    "agile",
    "angry",
    "brave",
    "calm",
    "clever",
    "curious",
    "daring",
    "eager",
    "fancy",
    "fast",
    "fierce",
    "fine",
    "gentle",
    "giant",
    "glad",
    "glossy",
    "grand",
    "happy",
    "harsh",
    "humble",
    "jolly",
    "keen",
    "lazy",
    "light",
    "lively",
    "lucky",
    "merry",
    "mighty",
    "noble",
    "plain",
    "proud",
    "quick",
    "quiet",
    "rapid",
    "roaring",
    "sharp",
    "shiny",
    "silent",
    "silly",
    "smart",
    "snappy",
    "solid",
    "swift",
    "tender",
    "tiny",
    "tough",
    "witty",
    "zany",
    "bold",
    "bright",
]
ANIMALS = [
    "aardvark",
    "albatross",
    "antelope",
    "badger",
    "beaver",
    "bison",
    "bobcat",
    "buffalo",
    "camel",
    "cheetah",
    "cougar",
    "coyote",
    "cricket",
    "crow",
    "dolphin",
    "donkey",
    "eagle",
    "falcon",
    "ferret",
    "fox",
    "gazelle",
    "giraffe",
    "goose",
    "gorilla",
    "heron",
    "ibis",
    "jaguar",
    "lemur",
    "leopard",
    "lion",
    "lizard",
    "lynx",
    "marmoset",
    "mole",
    "moose",
    "narwhal",
    "otter",
    "owl",
    "panda",
    "panther",
    "pelican",
    "penguin",
    "phoenix",
    "rabbit",
    "raccoon",
    "raven",
    "rhino",
    "tiger",
    "walrus",
    "wombat",
]


def random_filestub() -> str:
    return (
        f"{random.choice(ADJECTIVES)}_"
        f"{random.choice(ANIMALS)}_"
        f"{''.join(random.choices(string.ascii_letters + string.digits, k=4))}"
    )


def list_dataset_ids(graph: UrimDatasetGraph | None = None) -> list[str]:
    graph = graph or UrimDatasetGraph.from_file()

    return sorted(graph.graph.keys())
