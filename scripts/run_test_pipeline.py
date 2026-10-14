#!/usr/bin/env python3
"""Run interest sorting and summarization against a local arXiv catchup JSON file.

Examples
--------
    uv run python scripts/run_test_pipeline.py papers.json arxiv-interests.yaml
    uv run python scripts/run_test_pipeline.py papers.json interests.yaml --backend bart
    uv run python scripts/run_test_pipeline.py papers.json interests.yaml \
        --backend llm --provider openrouter --model provider/model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from arxiv_summarizer import BartSummarizer, LlmSummarizer, OpenRouterSummarizer
from arxiv_summarizer.models import SourceItem

from arxiv_interest_sort import STRATEGIES, load_interest_config, score_and_sort
from arxiv_interest_sort._cache import get_cache_dir

Paper = dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("papers_json", type=Path, help="arxiv-catchup JSON file (array of paper objects)")
    parser.add_argument("interests_yaml", type=Path, help="Interest-sort YAML configuration")
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), default="keyword", help="Sorting strategy")
    parser.add_argument("--sort-model", help="Sentence-transformers model (semantic strategy only)")
    parser.add_argument("--threshold", type=float, help="Match threshold (semantic strategy only)")
    parser.add_argument("--cache-dir", help="Semantic topic-embedding cache directory")
    parser.add_argument("--backend", choices=["bart", "llm", "openrouter"], default="bart")
    parser.add_argument("--model", help="Summarizer model ID; BART defaults to facebook/bart-large-cnn")
    parser.add_argument(
        "--provider",
        choices=["opencode-zen", "openrouter"],
        default="opencode-zen",
        help="Provider for --backend llm (default: opencode-zen)",
    )
    parser.add_argument("--papers-per-prompt", type=int, help="LLM/OpenRouter: papers per request")
    parser.add_argument("--user-prompt", help="LLM/OpenRouter qualitative summary guidance")
    parser.add_argument("--max-length", type=int, default=130, help="BART maximum summary length")
    parser.add_argument("--min-length", type=int, default=40, help="BART minimum summary length")
    parser.add_argument("--batch-size", type=int, default=4, help="BART inference batch size")
    return parser


def load_papers(path: Path) -> list[Paper]:
    """Load and validate the arxiv-catchup paper array."""
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Cannot read papers JSON: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError("Papers JSON must be an array of paper objects")
    seen_ids: set[str] = set()
    for index, paper in enumerate(data):
        if not isinstance(paper, dict):
            raise ValueError(f"Paper at index {index} must be a JSON object")
        paper_id: Any = paper.get("id")
        if not isinstance(paper_id, str) or not paper_id.strip():
            raise ValueError(f"Paper at index {index} must have a nonempty string 'id'")
        if paper_id in seen_ids:
            raise ValueError(f"Duplicate paper ID: {paper_id}")
        seen_ids.add(paper_id)
        for field in ("title", "abstract"):
            if not isinstance(paper.get(field), str):
                raise ValueError(f"Paper {paper_id} must have a string '{field}'")
    return data


def build_summarizer(args: argparse.Namespace) -> Any:
    """Construct the requested summarizer without making a provider request."""
    if args.backend == "bart":
        return BartSummarizer(
            model_name=args.model or "facebook/bart-large-cnn",
            max_length=args.max_length,
            min_length=args.min_length,
            batch_size=args.batch_size,
        )
    if args.backend == "openrouter":
        return OpenRouterSummarizer(
            model=args.model,
            papers_per_prompt=args.papers_per_prompt,
            user_prompt=args.user_prompt,
        )
    return LlmSummarizer(
        provider=args.provider,
        model=args.model,
        papers_per_prompt=args.papers_per_prompt,
        user_prompt=args.user_prompt,
    )


def run_pipeline(
    papers: list[Paper],
    interests_path: Path,
    summarizer: Any,
    *,
    strategy: str = "keyword",
    sort_model: str | None = None,
    threshold: float | None = None,
    cache_dir: str | None = None,
) -> list[Paper]:
    """Score/sort all papers and summarize matched papers, preserving IDs."""
    config = load_interest_config(interests_path)
    kwargs: dict[str, Any] = {}
    if strategy == "semantic":
        if sort_model:
            kwargs["model"] = sort_model
        if threshold is not None:
            kwargs["threshold"] = threshold
        kwargs["cache_dir"] = get_cache_dir(cache_dir)
    sorted_papers: list[Paper] = score_and_sort(papers, list(config), strategy=strategy, **kwargs)

    matched_papers: list[Paper] = [paper for paper in sorted_papers if paper.get("matched", False)]
    items: list[SourceItem] = [SourceItem(id=paper["id"], text=paper["abstract"]) for paper in matched_papers]
    if hasattr(summarizer, "summarize_items"):
        results = summarizer.summarize_items(items)
        results_by_id: dict[str, dict[str, Any]] = {result["id"]: result for result in results}
    else:
        positional_results = summarizer.summarize_many([item["text"] for item in items])
        if len(positional_results) != len(items):
            raise ValueError("Summarizer returned a different number of results than requested")
        results_by_id = {
            item["id"]: {**result, "id": item["id"]}
            for item, result in zip(items, positional_results, strict=True)
        }
    expected_ids: set[str] = {paper["id"] for paper in matched_papers}
    if set(results_by_id) != expected_ids:
        raise ValueError("Summarizer results did not match the requested paper IDs")
    for paper in matched_papers:
        result: dict[str, Any] = results_by_id[paper["id"]]
        paper["summary_status"] = result["status"]
        paper["summary"] = result["summary"]
        if result.get("error"):
            paper["summary_error"] = result["error"]
    return sorted_papers


def main(argv: list[str] | None = None) -> int:
    """Run the selected local pipeline and print its JSON result."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        papers = load_papers(args.papers_json)
        output = run_pipeline(
            papers,
            args.interests_yaml,
            build_summarizer(args),
            strategy=args.strategy,
            sort_model=args.sort_model,
            threshold=args.threshold,
            cache_dir=args.cache_dir,
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
