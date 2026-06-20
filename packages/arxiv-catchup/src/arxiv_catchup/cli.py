"""CLI entry point for the arXiv fetcher.

Registered as the ``arxiv-catchup`` console script in ``pyproject.toml``::

    $ arxiv-catchup
    $ arxiv-catchup quant-ph
    $ arxiv-catchup quant-ph 2026-06-01
    $ arxiv-catchup quant-ph --days-ago 1
    $ arxiv-catchup cs.LG --no-abstracts
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone, timedelta

from arxiv_catchup import build_catchup_url, fetch_page, extract_papers


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (separated for testability)."""
    parser = argparse.ArgumentParser(
        description="Extract papers from an arXiv catchup page as JSON.",
    )
    parser.add_argument(
        "category",
        nargs="?",
        default="quant-ph",
        help=(
            "arXiv category (e.g. quant-ph, cs.LG). "
            "See https://arxiv.org/category_taxonomy for options"
        ),
    )
    parser.add_argument(
        "date",
        nargs="?",
        default=None,
        help="Date in YYYY-MM-DD format (default: today)",
    )
    parser.add_argument(
        "--no-abstracts",
        action="store_true",
        help="Fetch without abstracts (smaller payload)",
    )
    parser.add_argument(
        "--days-ago",
        type=int,
        default=0,
        help="Fetch listings from N days before today (default: 0 = today)",
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

    # ── resolve date ────────────────────────────────────────────────
    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        target_date = (datetime.now(timezone.utc) - timedelta(days=args.days_ago)).date()

    url = build_catchup_url(args.category, target_date, args.no_abstracts)

    # ── fetch ───────────────────────────────────────────────────────
    try:
        html = fetch_page(url)
    except Exception as e:
        print(json.dumps({"error": str(e), "url": url}))
        sys.exit(1)

    if html is None:
        print(json.dumps({"error": "Failed to fetch after retries", "url": url}))
        sys.exit(1)

    # ── parse ───────────────────────────────────────────────────────
    papers = extract_papers(html)
    print(json.dumps(papers, indent=2))
