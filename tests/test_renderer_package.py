from __future__ import annotations

import importlib.util


def test_renderer_is_installed_package() -> None:
    assert importlib.util.find_spec("arxiv_renderer") is not None
