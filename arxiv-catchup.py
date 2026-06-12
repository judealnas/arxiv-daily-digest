#!/usr/bin/env python3
"""
Extract papers from an arXiv catchup page and output as JSON.

Usage:
    python3 arxiv-catchup.py quant-ph 2026-06-01
    python3 arxiv-catchup.py quant-ph --yesterday
    python3 arxiv-catchup.py  (default: category=quant-ph, date=today)

URL pattern: https://arxiv.org/catchup/{category}/{YYYY-MM-DD}?abs=True|False
"""

import json
import re
import sys
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone
from html import unescape

USER_AGENT = "HermesArxivBot/1.0 (arxiv-catchup extractor)"
REQUEST_TIMEOUT = 30
RETRY_DELAY = 5
MAX_RETRIES = 3


def fetch_page(url):
    """Fetch URL with retry on rate limit."""
    for attempt in range(MAX_RETRIES):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < MAX_RETRIES - 1:
                import time
                time.sleep(RETRY_DELAY * (2 ** attempt))
                continue
            raise
    return None


def strip_html(text):
    """Remove HTML tags and unescape entities."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return unescape(text).strip()


def extract_papers(html_content):
    """
    Parse an arXiv catchup page and return a list of paper dicts.

    Each paper is structured as:
        <dt>
          <a href="/abs/2605.30425" ...>arXiv:2605.30425</a>
          [<a href="/pdf/...">pdf</a>, ...]
        </dt>
        <dd>
          <div class='meta'>
            <div class='list-title mathjax'><span>Title:</span> TITLE</div>
            <div class='list-authors'>AUTHOR LINKS</div>
            <div class='list-comments mathjax'>...</div>
            <div class='list-subjects'>...</div>
            <p class='mathjax'>ABSTRACT</p>
          </div>
        </dd>
    """
    papers = []

    # Match each <dt>...</dt>\n<dd>...</dd> block
    # The <dd> is the last child of each paper block (no nesting of <dd> here)
    block_pattern = re.compile(
        r'<dt\b[^>]*>(.*?)</dt>\s*<dd\b[^>]*>(.*?)</dd>',
        re.DOTALL
    )

    for dt_match, dd_content in block_pattern.findall(html_content):
        # arXiv ID from the dt section
        id_match = re.search(r'/abs/(\d+\.\d+v?\d*)', dt_match)
        if not id_match:
            continue
        arxiv_id = id_match.group(1)

        # Title
        title_match = re.search(
            r"<div\s+class='list-title\s*mathjax'\s*>.*?"
            r"<span\s+class='descriptor'>\s*Title:\s*</span>\s*(.*?)</div>",
            dd_content, re.DOTALL
        )
        title = strip_html(title_match.group(1)) if title_match else ""

        # Authors
        authors_match = re.search(
            r"<div\s+class='list-authors'\s*>(.*?)</div>",
            dd_content, re.DOTALL
        )
        authors = []
        if authors_match:
            author_links = re.findall(r'<a\b[^>]*>([^<]+)</a>', authors_match.group(1))
            authors = [unescape(a.strip()) for a in author_links]

        # Comments (optional)
        comments_match = re.search(
            r"<div\s+class='list-comments\s*mathjax'\s*>.*?"
            r"<span\s+class='descriptor'>\s*Comments:\s*</span>\s*(.*?)</div>",
            dd_content, re.DOTALL
        )
        comments = strip_html(comments_match.group(1)) if comments_match else ""

        # Subjects
        subjects_match = re.search(
            r"<div\s+class='list-subjects'\s*>.*?"
            r"<span\s+class='descriptor'>\s*Subjects:\s*</span>\s*(.*?)</div>",
            dd_content, re.DOTALL
        )
        subjects = strip_html(subjects_match.group(1)) if subjects_match else ""

        # Abstract
        abstract_match = re.search(
            r"<p\s+class='mathjax'\s*>(.*?)</p>",
            dd_content, re.DOTALL
        )
        abstract = strip_html(abstract_match.group(1)) if abstract_match else ""

        # Strip version suffix for clean links
        arxiv_id_clean = re.sub(r'v\d+$', '', arxiv_id)

        papers.append({
            "id": arxiv_id,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "abstract_link": f"https://arxiv.org/abs/{arxiv_id_clean}",
            "pdf_link": f"https://arxiv.org/pdf/{arxiv_id_clean}",
            "comments": comments,
            "subjects": subjects,
        })

    return papers


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract papers from an arXiv catchup page as JSON."
    )
    parser.add_argument("category", nargs="?", default="quant-ph",
                        help="arXiv category (e.g. quant-ph, cs.LG)")
    parser.add_argument("date", nargs="?",
                        help="Date in YYYY-MM-DD format (default: today)")
    parser.add_argument("--no-abstracts", action="store_true",
                        help="Fetch without abstracts (smaller payload)")
    parser.add_argument("--yesterday", action="store_true",
                        help="Fetch yesterday's catchup (overrides default to today)")

    args = parser.parse_args()

    # Determine date
    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    elif args.yesterday:
        target_date = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    else:
        target_date = datetime.now(timezone.utc).date()

    date_str = target_date.strftime("%Y-%m-%d")
    abs_param = "False" if args.no_abstracts else "True"

    url = f"https://arxiv.org/catchup/{args.category}/{date_str}?abs={abs_param}"

    try:
        html = fetch_page(url)
        if html is None:
            print(json.dumps({"error": "Failed to fetch after retries", "url": url}))
            sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": str(e), "url": url}))
        sys.exit(1)

    papers = extract_papers(html)

    print(json.dumps(papers, indent=2))


if __name__ == "__main__":
    main()
