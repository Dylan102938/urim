from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class GlobalOptions:
    """Holds process-wide CLI options.

    Extend as needed (e.g., config path, profile, seed, etc.).
    """

    verbose: int = 0


GLOBAL_OPTIONS = GlobalOptions()
