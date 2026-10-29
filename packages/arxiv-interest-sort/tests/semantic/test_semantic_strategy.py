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
    @pytest.mark.parametrize("topics", [[], [{"name": "quantum", "keywords": ["qubit"], "weight": 2}]])
    def test_empty_papers_do_not_import_model_dependencies(self, topics: list, monkeypatch) -> None:
        from arxiv_interest_sort import score_and_sort
        from arxiv_interest_sort.strategies import semantic_score

        original_import = builtins.__import__

        def forbid_model_imports(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "sentence_transformers"}:
                raise AssertionError(f"Empty input attempted model dependency import: {name}")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", forbid_model_imports)
        empty: list[dict] = []
        assert semantic_score(empty, topics) is empty
        assert score_and_sort(empty, topics, strategy="semantic") == []

    def test_nonempty_scoring_preserves_weighted_annotations(self, monkeypatch) -> None:
        import sys
        import types

        import torch
        from arxiv_interest_sort import score_and_sort

        class FakeEncoder:
            def encode(self, texts, **kwargs):
                if texts == ["heavy: alpha", "light: beta"]:
                    return torch.eye(2)
                assert texts == ["Light. beta", "Heavy. alpha"]
                return torch.tensor([[0.0, 1.0], [1.0, 0.0]])

        fake_module = types.ModuleType("sentence_transformers")
        monkeypatch.setattr(fake_module, "SentenceTransformer", lambda name: FakeEncoder(), raising=False)
        monkeypatch.setattr(fake_module, "util", types.SimpleNamespace(cos_sim=lambda a, b: a @ b.T), raising=False)
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
        papers: list[dict] = [
            {"id": "light", "title": "Light", "abstract": "beta"},
            {"id": "heavy", "title": "Heavy", "abstract": "alpha"},
        ]
        topics: list[dict] = [
            {"name": "heavy", "keywords": ["alpha"], "weight": 2},
            {"name": "light", "keywords": ["beta"], "weight": 1},
        ]
        result = score_and_sort(papers, topics, strategy="semantic", threshold=0.75)
        assert [paper["id"] for paper in result] == ["heavy", "light"]
        for paper, matched, score, topic in zip(result, [True, False], [1.0, 0.5], ["heavy", "light"], strict=True):
            assert paper["matched"] is matched
            assert paper["match_score"] == score
            assert paper["match_count"] == int(score * 100)
            assert paper["semantic_topic"] == topic
            assert paper["raw_cosim"] == 1.0

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
            semantic_score(
                [{"id": "test", "title": "Quantum network", "abstract": "A qubit link."}],
                [{"name": "quantum", "keywords": ["qubit"], "weight": 2}],
            )

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
