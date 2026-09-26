"""Path helpers for locating the task data.

All dataset locations are resolved lazily from the ``WUAS_DATA_ROOT``
environment variable; no absolute paths are hardcoded in this codebase.
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = ["data_root"]


def data_root() -> Path:
    """Return the root directory that holds the task data files."""
    value = os.environ.get("WUAS_DATA_ROOT")
    if not value:
        raise FileNotFoundError(
            "WUAS_DATA_ROOT is not set. Point it at the directory containing "
            "the task data (e.g. data/searchqa_split, data/livemath, "
            "data/docvqa, ...). Data files are prepared separately; see README.")
    return Path(value)
