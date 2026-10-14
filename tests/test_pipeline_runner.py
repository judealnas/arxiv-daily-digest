from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from scripts.run_test_pipeline import load_papers, run_pipeline


class IdentitySummarizer:
    def summarize_items(self, items: list[dict[str, str]]) -> list[dict[str, Any]]:
        return [
            {"id": item["id"], "status": "ok", "summary": f"Summary for {item['id']}", "error": None}
            for item in reversed(items)
        ]


class PositionalSummarizer:
    def summarize_many(self, texts: list[str]) -> list[dict[str, Any]]:
        return [
            {"id": str(index), "status": "ok", "summary": f"Summary for {text}", "error": None}
            for index, text in enumerate(texts)
        ]


@pytest.fixture
def interests_file(tmp_path: Path) -> Path:
    path = tmp_path / "interests.yaml"
    path.write_text(
        "- name: quantum networking\n"
        "  weight: 2\n"
        "  keywords:\n"
        "    - quantum network\n"
        "  description: Quantum networking papers\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def papers() -> list[dict[str, Any]]:
    return [
        {"id": "2401.00002", "title": "Other work", "abstract": "Unrelated abstract."},
        {"id": "2401.00001", "title": "Quantum network protocol", "abstract": "A quantum network method."},
    ]


def test_pipeline_sorts_all_and_summarizes_only_matches_with_identity(
    interests_file: Path, papers: list[dict[str, Any]]
) -> None:
    output = run_pipeline(papers, interests_file, IdentitySummarizer())

    assert [paper["id"] for paper in output] == ["2401.00001", "2401.00002"]
    assert output[0]["matched"] is True
    assert output[0]["summary"] == "Summary for 2401.00001"
    assert "summary" not in output[1]


def test_pipeline_restores_positional_bart_style_ids(interests_file: Path, papers: list[dict[str, Any]]) -> None:
    output = run_pipeline(papers, interests_file, PositionalSummarizer())
    matched = next(paper for paper in output if paper["matched"])
    assert matched["summary"] == "Summary for A quantum network method."


def test_load_papers_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "papers.json"
    paper = {"id": "2401.00001", "title": "T", "abstract": "A."}
    path.write_text(json.dumps([paper, paper]), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate paper ID"):
        load_papers(path)
