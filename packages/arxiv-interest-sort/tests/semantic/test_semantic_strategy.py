"""Tests for the semantic scoring strategy."""

from __future__ import annotations

import builtins
import json
from pathlib import Path

import pytest

# ── fixtures ─────────────────────────────────────────────────────
FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def papers() -> list[dict]:
    with open(FIXTURES / "sample_input.json") as f:
        return json.load(f)


@pytest.fixture
def interests() -> list:
    import yaml

    with open(FIXTURES / "sample_interests.yaml") as f:
        return yaml.safe_load(f)


# ── semantic strategy ────────────────────────────────────────────
class TestSemanticScore:
    def test_missing_dependency(
        self, capsys: pytest.CaptureFixture[str], monkeypatch
    ) -> None:
        """Without sentence-transformers, the strategy exits with an error."""
        original_import = builtins.__import__

        def raise_for_sentence_transformers(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("simulated missing optional dependency")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", raise_for_sentence_transformers)
        from arxiv_interest_sort.strategies import semantic_score

        with pytest.raises(SystemExit) as exc_info:
            semantic_score([], [])

        assert exc_info.value.code == 1
        assert "requires sentence-transformers" in capsys.readouterr().err

    @pytest.mark.slow
    def test_semantic_with_model(self, papers: list[dict], interests: list) -> None:
        """Semantic scoring produces annotations (requires model download)."""
        pytest.importorskip(
            "sentence_transformers",
            reason=(
                "requires the semantic extra: "
                "uv sync --package arxiv-interest-sort --extra semantic"
            ),
        )
        from arxiv_interest_sort.strategies import semantic_score

        result = semantic_score(
            papers, interests, model="all-MiniLM-L6-v2", threshold=0.3
        )
        for p in result:
            assert "match_score" in p
            assert "match_count" in p
            assert "matched" in p
            assert "semantic_topic" in p
            assert "raw_cosim" in p
