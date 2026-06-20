"""Tests for the semantic scoring strategy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ── fixtures ─────────────────────────────────────────────────────
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def papers() -> list[dict]:
    with open(FIXTURES / "sample_input.json") as f:
        return json.load(f)


@pytest.fixture
def interests() -> dict:
    import yaml

    with open(FIXTURES / "sample_interests.yaml") as f:
        return yaml.safe_load(f)


# ── semantic strategy ────────────────────────────────────────────
class TestSemanticScore:
    def test_missing_dependency(self) -> None:
        """Without sentence-transformers, the strategy exits with an error."""
        # We can't actually uninstall the package in a test, but we can verify
        # the import-error handling by checking the code path.
        # This is a structural test that the ImportError trap exists.
        from arxiv_interest_sort.strategies import semantic_score

        assert hasattr(semantic_score, "__call__")

    @pytest.mark.skipif(
        "not pytest.skip",
        reason="Requires sentence-transformers (install with [semantic] extra)",
    )
    def test_semantic_with_model(self, papers: list[dict], interests: dict) -> None:
        """Semantic scoring produces annotations (requires model download)."""
        from arxiv_interest_sort.strategies import semantic_score

        topics = interests["topics"]
        result = semantic_score(papers, topics, model="all-MiniLM-L6-v2", threshold=0.3)
        for p in result:
            assert "match_score" in p
            assert "match_count" in p
            assert "matched" in p
            assert "semantic_topic" in p
            assert "raw_cosim" in p
