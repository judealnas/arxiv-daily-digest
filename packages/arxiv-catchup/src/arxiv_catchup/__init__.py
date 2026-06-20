"""arXiv catchup page fetcher and parser.

Public API:
    fetch_page(url: str, ...) -> str | None
    extract_papers(html: str) -> list[PaperDict]
    get_papers(category: str, ...) -> list[PaperDict]
"""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone, date
from html import unescape
from typing import Any

# ── constants ──────────────────────────────────────────────────────────────

# USER_AGENT: str = "HermesArxivBot/1.0 (arxiv-catchup extractor)"
DEFAULT_TIMEOUT: int = 30
DEFAULT_RETRY_DELAY: int = 5
DEFAULT_MAX_RETRIES: int = 3

# ── type aliases ───────────────────────────────────────────────────────────

PaperDict = dict[str, Any]
"""Shape returned by extract_papers:
    id, title, authors, abstract, abstract_link, pdf_link, comments, subjects
"""

# ── network ────────────────────────────────────────────────────────────────


def fetch_page(
    url: str,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: int = DEFAULT_RETRY_DELAY,
    timeout: int = DEFAULT_TIMEOUT,
    headers: dict[str, Any] = {},
) -> str | None:
    """Fetch *url* with exponential-backoff retry on HTTP 429.

    Returns the response body as UTF-8 text, or ``None`` if all retries are
    exhausted.  Re-raises any non-429 exception.
    """
    import time

    for attempt in range(max_retries):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                time.sleep(retry_delay * (2**attempt))
                continue
            raise
    return None


# ── HTML parsing ───────────────────────────────────────────────────────────


def _strip_html(text: str) -> str:
    """Remove HTML tags and unescape entities, collapsing whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return unescape(text).strip()


def extract_papers(html_content: str) -> list[PaperDict]:
    """Parse an arXiv catchup HTML page into a list of paper dicts.

    Each ``<dt>…</dt>\\n<dd>…</dd>`` pair in the listing corresponds to one
    paper.  The expected HTML structure is::

        <dt>
          <a href="/abs/2605.30425" …>arXiv:2605.30425</a>
          [<a href="/pdf/…">pdf</a>, …]
        </dt>
        <dd>
          <div class='meta'>
            <div class='list-title mathjax'>
              <span class='descriptor'>Title:</span> TITLE
            </div>
            <div class='list-authors'>
              <a href="…">Author Name</a>, …
            </div>
            <div class='list-comments mathjax'>…</div>
            <div class='list-subjects'>…</div>
            <p class='mathjax'>ABSTRACT</p>
          </div>
        </dd>

    The ``list-comments`` and ``list-subjects`` blocks are optional — they are
    missing from some cross-listed entries.
    """
    papers: list[PaperDict] = []

    # Match each <dt>…</dt>\n<dd>…</dd> block (the <dd> is never nested here)
    block_pattern = re.compile(
        r"<dt\b[^>]*>(.*?)</dt>\s*<dd\b[^>]*>(.*?)</dd>",
        re.DOTALL,
    )

    for dt_match, dd_content in block_pattern.findall(html_content):
        # ── arXiv ID ────────────────────────────────────────────────
        id_match = re.search(r"/abs/(\d+\.\d+v?\d*)", dt_match)
        if not id_match:
            continue
        arxiv_id: str = id_match.group(1)

        # ── Title ───────────────────────────────────────────────────
        title_match = re.search(
            r"<div\s+class='list-title\s*mathjax'\s*>.*?"
            r"<span\s+class='descriptor'>\s*Title:\s*</span>\s*(.*?)</div>",
            dd_content,
            re.DOTALL,
        )
        title: str = _strip_html(title_match.group(1)) if title_match else ""

        # ── Authors ─────────────────────────────────────────────────
        authors_match = re.search(
            r"<div\s+class='list-authors'\s*>(.*?)</div>",
            dd_content,
            re.DOTALL,
        )
        authors: list[str] = []
        if authors_match:
            author_links = re.findall(r"<a\b[^>]*>([^<]+)</a>", authors_match.group(1))
            authors = [unescape(a.strip()) for a in author_links]

        # ── Comments (optional) ─────────────────────────────────────
        comments_match = re.search(
            r"<div\s+class='list-comments\s*mathjax'\s*>.*?"
            r"<span\s+class='descriptor'>\s*Comments:\s*</span>\s*(.*?)</div>",
            dd_content,
            re.DOTALL,
        )
        comments: str = _strip_html(comments_match.group(1)) if comments_match else ""

        # ── Subjects (optional in rare cross-lists) ─────────────────
        subjects_match = re.search(
            r"<div\s+class='list-subjects'\s*>.*?"
            r"<span\s+class='descriptor'>\s*Subjects:\s*</span>\s*(.*?)</div>",
            dd_content,
            re.DOTALL,
        )
        subjects: str = _strip_html(subjects_match.group(1)) if subjects_match else ""

        # ── Abstract ────────────────────────────────────────────────
        abstract_match = re.search(
            r"<p\s+class='mathjax'\s*>(.*?)</p>",
            dd_content,
            re.DOTALL,
        )
        abstract: str = _strip_html(abstract_match.group(1)) if abstract_match else ""

        # ── Clean links (strip version suffix) ──────────────────────
        arxiv_id_clean: str = re.sub(r"v\d+$", "", arxiv_id)

        papers.append(
            {
                "id": arxiv_id,
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "abstract_link": f"https://arxiv.org/abs/{arxiv_id_clean}",
                "pdf_link": f"https://arxiv.org/pdf/{arxiv_id_clean}",
                "comments": comments,
                "subjects": subjects,
            }
        )

    return papers


# ── high-level convenience ─────────────────────────────────────────────────


def get_papers(
    category: str = "quant-ph",
    date: date | str | None = None,
    *,
    yesterday: bool = False,
    with_abstracts: bool = True,
) -> list[PaperDict]:
    """Fetch and parse a single arXiv catchup page in one call.

    Parameters
    ----------
    category:
        arXiv category, e.g. ``"quant-ph"``, ``"cs.LG"`` (default ``"quant-ph"``).
    date:
        Explicit date as a ``date`` object or ``"YYYY-MM-DD"`` string.
        Mutually exclusive with *yesterday*.
    yesterday:
        If ``True``, fetch yesterday's listings.  Ignored when *date* is set.
    with_abstracts:
        If ``False``, request the listing without abstract text (smaller
        payload).

    Returns
    -------
    list[PaperDict]
        Parsed paper dicts, one per entry on the page.

    Raises
    ------
    SystemExit
        If the HTTP request fails after exhausting all retries.
    """
    if date:
        if isinstance(date, str):
            target = datetime.strptime(date, "%Y-%m-%d").date()
        else:
            target = date
    elif yesterday:
        target = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    else:
        target = datetime.now(timezone.utc).date()

    date_str = target.strftime("%Y-%m-%d")
    abs_param = "False" if not with_abstracts else "True"
    url = f"https://arxiv.org/catchup/{category}/{date_str}?abs={abs_param}"

    html = fetch_page(url)
    if html is None:
        print(
            json.dumps({"error": "Failed to fetch after retries", "url": url}),
            file=sys.stderr,
        )
        sys.exit(1)

    return extract_papers(html)


__all__ = ["fetch_page", "extract_papers", "get_papers"]
