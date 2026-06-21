"""Platform-aware cache directory utility."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def get_cache_dir(override: str | None = None) -> Path:
    """Return the cache directory for arxiv-interest-sort.

    Uses the platform-appropriate default unless *override* is given:

    - **Linux**:   ``$XDG_CACHE_HOME/arxiv-interest-sort`` or
      ``~/.cache/arxiv-interest-sort``
    - **macOS**:   ``~/Library/Caches/arxiv-interest-sort``
    - **Windows**: ``%LOCALAPPDATA%/arxiv-interest-sort``
    """
    if override is not None:
        return Path(override).resolve()

    if sys.platform == "linux":
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path.home() / ".cache"

    return base / "arxiv-interest-sort"


def topic_embeddings_path(cache_dir: str | Path) -> Path:
    """Return the path to the topic embeddings cache file."""
    return Path(cache_dir) / "topic_embeddings.npz"
