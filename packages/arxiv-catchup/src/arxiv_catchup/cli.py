"""CLI entry point for the arXiv fetcher.

Registered as the ``arxiv-fetch`` console script in ``pyproject.toml``::

    $ arxiv-fetch
    $ arxiv-fetch quant-ph
    $ arxiv-fetch quant-ph 2026-06-01
    $ arxiv-fetch quant-ph --yesterday
    $ arxiv-fetch cs.LG --no-abstracts
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
        help="arXiv category (e.g. quant-ph, cs.LG)"
        "See https://arxiv.org/category_taxonomy for options",
    )
    parser.add_argument(
        "date",
        nargs="?",
        help="Date in YYYY-MM-DD format (default: today)",
    )
    parser.add_argument(
        "--no-abstracts",
        action="store_true",
        help="Fetch without abstracts (smaller payload)",
    )
    parser.add_argument(
        "--yesterday",
        action="store_true",
        help="Fetch yesterday's catchup (overrides default to today)",
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
    elif args.yesterday:
        target_date = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    else:
        target_date = datetime.now(timezone.utc).date()

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
