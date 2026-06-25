"""CLI entry point for arxiv-interest-sort.

Registered as the ``arxiv-interest-sort`` console script in ``pyproject.toml``::

    $ arxiv-interest-sort < input.json
    $ arxiv-interest-sort papers.json
    $ arxiv-interest-sort --strategy semantic --model all-MiniLM-L6-v2 papers.json
    $ arxiv-interest-sort --include-metadata < input.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

from arxiv_interest_sort import STRATEGIES, load_interest_config, score_and_sort
from arxiv_interest_sort._cache import get_cache_dir


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (separated for testability)."""
    parser = argparse.ArgumentParser(
        description="Sort arXiv papers by interest profile.",
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default=None,
        help="Path to a JSON file containing a list of paper objects "
        "(omit to read from stdin)",
    )
    parser.add_argument(
        "--strategy",
        default="keyword",
        help="Scoring strategy to use (default: keyword)",
    )
    parser.add_argument(
        "--config",
        default="arxiv-interests.yaml",
        help="Path to the interests YAML file (default: arxiv-interests.yaml)",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Wrap output in an envelope with date, counts, and papers list",
    )

    # ── semantic-strategy options ─────────────────────────────────
    parser.add_argument(
        "--model",
        help="sentence-transformers model name (default: allenai/specter2_base)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Weighted-score threshold for 'matched' (default: 0.79)",
    )
    parser.add_argument(
        "--cache-dir",
        help="Directory for topic embedding cache (default: platform-appropriate)",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point.

    Parameters
    ----------
    argv:
        Command-line arguments (defaults to ``sys.argv[1:]``).  Exposed as a
        parameter so the function can be tested without subprocess calls.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # ── load config ─────────────────────────────────────────────────
    try:
        config = load_interest_config(args.config)
    except FileNotFoundError:
        print(
            json.dumps({"error": f"Config file not found: {args.config}"}),
            file=sys.stderr,
        )
        sys.exit(1)

    # ── read input ──────────────────────────────────────────────────
    if args.input_file:
        try:
            with open(args.input_file) as f:
                papers = json.load(f)
        except FileNotFoundError:
            print(
                json.dumps({"error": f"Input file not found: {args.input_file}"}),
                file=sys.stderr,
            )
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(
                json.dumps({"error": f"Invalid JSON in input file: {e}"}),
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        try:
            papers = json.load(sys.stdin)
        except json.JSONDecodeError as e:
            print(
                json.dumps({"error": f"Invalid JSON on stdin: {e}"}),
                file=sys.stderr,
            )
            sys.exit(1)

    if not isinstance(papers, list):
        print(
            json.dumps({"error": "Input must be a JSON array of paper objects"}),
            file=sys.stderr,
        )
        sys.exit(1)

    # ── resolve strategy ────────────────────────────────────────────
    strategy: str = args.strategy
    if strategy not in STRATEGIES:
        print(
            json.dumps(
                {
                    "error": f"Unknown strategy: {strategy}",
                    "available": list(STRATEGIES.keys()),
                }
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    # ── build strategy kwargs ───────────────────────────────────────
    kwargs: dict = {}
    if strategy == "semantic":
        if args.model:
            kwargs["model"] = args.model
        if args.threshold is not None:
            kwargs["threshold"] = args.threshold
        kwargs["cache_dir"] = get_cache_dir(args.cache_dir)

    # ── score & sort ────────────────────────────────────────────────
    papers = score_and_sort(papers, config, strategy=strategy, **kwargs)

    # ── render output ───────────────────────────────────────────────
    if args.include_metadata:
        matched = [p for p in papers if p.get("matched", False)]
        output = {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "count": len(papers),
            "matched_count": len(matched),
            "papers": papers,
        }
        print(json.dumps(output, indent=2))
    else:
        print(json.dumps(papers, indent=2))
