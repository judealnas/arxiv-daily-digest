#!/usr/bin/env python3
"""Pipeline: fetch arXiv papers and sort them by interest profile.

Usage
-----
Fetch from arXiv and sort::

    python arxiv-daily-digest.py quant-ph
    python arxiv-daily-digest.py quant-ph --days-ago 1
    python arxiv-daily-digest.py cs.LG 2026-06-19

Parse a local HTML file and sort::

    python arxiv-daily-digest.py --from-html catchup-page.html
    python arxiv-daily-digest.py --from-html catchup-page.html --html-category quant-ph

Options
-------
--config PATH
    Path to interests YAML (default: arxiv-interests.yaml).
--strategy {keyword,semantic}
    Scoring strategy (default: keyword).
--model NAME
    sentence-transformers model (semantic strategy only).
--threshold FLOAT
    Match threshold (semantic strategy only).
--cache-dir PATH
    Embedding cache directory (semantic strategy only).
--include-metadata
    Wrap output in an envelope with date and counts.
--output PATH
    Write JSON to file instead of stdout.
--html-category CATEGORY
    arXiv category to record when using --from-html (default: quant-ph).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from arxiv_catchup import build_catchup_url, extract_papers, fetch_page
from arxiv_interest_sort import load_interest_config, score_and_sort
from arxiv_interest_sort._cache import get_cache_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── source: fetch or local HTML ─────────────────────────────────
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "category",
        nargs="?",
        default=None,
        help="arXiv category to fetch (e.g. quant-ph, cs.LG). "
        "Omit when using --from-html.",
    )
    source.add_argument(
        "--from-html",
        metavar="PATH",
        help="Parse papers from a local arXiv catchup HTML file instead of fetching.",
    )

    # ── fetch options ───────────────────────────────────────────────
    fetch_group = parser.add_argument_group("fetch options")
    fetch_group.add_argument(
        "date_positional",
        nargs="?",
        default=None,
        metavar="DATE",
        help="Date in YYYY-MM-DD format (default: today).",
    )
    fetch_group.add_argument(
        "--days-ago",
        type=int,
        default=0,
        help="Fetch listings from N days before today (default: 0).",
    )
    fetch_group.add_argument(
        "--no-abstracts",
        action="store_true",
        help="Fetch without abstracts (smaller payload).",
    )
    fetch_group.add_argument(
        "--html-category",
        default="quant-ph",
        help="Category label recorded when using --from-html (default: quant-ph).",
    )

    # ── sort options ────────────────────────────────────────────────
    sort_group = parser.add_argument_group("sort options")
    sort_group.add_argument(
        "--config",
        default="arxiv-interests.yaml",
        help="Path to interests YAML (default: arxiv-interests.yaml).",
    )
    sort_group.add_argument(
        "--strategy",
        default="keyword",
        choices=["keyword", "semantic"],
        help="Scoring strategy (default: keyword).",
    )
    sort_group.add_argument(
        "--model",
        help="sentence-transformers model name (semantic only).",
    )
    sort_group.add_argument(
        "--threshold",
        type=float,
        help="Match threshold (semantic only).",
    )
    sort_group.add_argument(
        "--cache-dir",
        help="Embedding cache directory (semantic only).",
    )
    sort_group.add_argument(
        "--include-metadata",
        action="store_true",
        help="Wrap output in an envelope with date and counts.",
    )

    # ── output ──────────────────────────────────────────────────────
    parser.add_argument(
        "--output",
        "-o",
        metavar="PATH",
        help="Write JSON output to file instead of stdout.",
    )

    return parser


def fetch_papers(args: argparse.Namespace) -> tuple[list[dict], str | None]:
    """Fetch papers from arXiv or parse from a local HTML file.

    Returns ``(papers, category)`` where *category* is the arXiv category
    label (useful for metadata when reading from a local file).
    """
    if args.from_html:
        html_path = Path(args.from_html)
        if not html_path.exists():
            print(json.dumps({"error": f"HTML file not found: {html_path}"}))
            sys.exit(1)
        html = html_path.read_text(encoding="utf-8")
        papers = extract_papers(html)
        return papers, args.html_category

    # ── fetch from arXiv ───────────────────────────────────────────
    category = args.category or "quant-ph"

    from datetime import datetime, timedelta

    if args.date_positional:
        target_date = datetime.strptime(args.date_positional, "%Y-%m-%d").date()
    else:
        target_date = (datetime.now() - timedelta(days=args.days_ago)).date()

    url = build_catchup_url(category, target_date, args.no_abstracts)

    try:
        html = fetch_page(url)
    except Exception as e:
        print(json.dumps({"error": str(e), "url": url}))
        sys.exit(1)

    if html is None:
        print(json.dumps({"error": "Failed to fetch after retries", "url": url}))
        sys.exit(1)

    papers = extract_papers(html)
    return papers, category


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # ── validate ────────────────────────────────────────────────────
    if not args.from_html and not args.category and not args.date_positional:
        # No positional args at all — default to quant-ph today
        args.category = "quant-ph"

    # ── fetch / parse ───────────────────────────────────────────────
    papers, category = fetch_papers(args)

    if not papers:
        print(json.dumps({"error": "No papers found", "category": category}))
        sys.exit(1)

    # ── load interests config ───────────────────────────────────────
    try:
        config = load_interest_config(args.config)
    except FileNotFoundError:
        print(
            json.dumps({"error": f"Config file not found: {args.config}"}),
            file=sys.stderr,
        )
        sys.exit(1)

    # ── build strategy kwargs ───────────────────────────────────────
    kwargs: dict = {}
    if args.strategy == "semantic":
        if args.model:
            kwargs["model"] = args.model
        if args.threshold is not None:
            kwargs["threshold"] = args.threshold
        kwargs["cache_dir"] = get_cache_dir(args.cache_dir)

    # ── score & sort ────────────────────────────────────────────────
    papers = score_and_sort(papers, list(config), strategy=args.strategy, **kwargs)

    # ── render output ───────────────────────────────────────────────
    from datetime import datetime

    if args.include_metadata:
        matched = [p for p in papers if p.get("matched", False)]
        output = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "category": category,
            "count": len(papers),
            "matched_count": len(matched),
            "strategy": args.strategy,
            "papers": papers,
        }
    else:
        output = papers

    json_text = json.dumps(output, indent=2)

    if args.output:
        Path(args.output).write_text(json_text + "\n", encoding="utf-8")
        print(f"Wrote {len(papers)} papers to {args.output}")
    else:
        print(json_text)


if __name__ == "__main__":
    main()
