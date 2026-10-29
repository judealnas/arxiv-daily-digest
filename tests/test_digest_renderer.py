from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from arxiv_digest_renderer import render_digest

FIXTURES = Path(__file__).parent / "fixtures" / "digest"


def load_json(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text())


def load_papers() -> dict[str, object]:
    return load_json("papers.json")


def test_render_digest_preserves_ranked_visual_format() -> None:
    html = render_digest(load_papers())

    assert '<html data-schema-version="1" data-digest-date="2026-09-15" data-source-run-id="rank-20260916T120000Z" data-build-stage="ranked">' in html
    assert "<h1>arXiv quant-ph — 2026-09-15</h1>" in html
    assert "<p>3 papers, ⭐2 matched</p>" in html
    assert '<p>⭐1. <a href="https://arxiv.org/abs/2609.00002">Matched &lt;Title&gt; &amp; Z_2</a><br>A matched abstract with x^2 and useful details. Second sentence.</p>' in html
    assert '<p>⭐2. <a href="https://arxiv.org/abs/2609.00003">Another matched paper</a><br>Second matched abstract.</p>' in html
    assert '<p>3. <a href="https://arxiv.org/abs/2609.00001">Unmatched &amp; &lt;Paper&gt;</a><br>First line with LaTeX q.</p>' in html
    assert "authors" not in html


def test_render_digest_shows_first_three_authors_with_etal_and_escapes_names() -> None:
    data = load_papers()
    papers = data["papers"]
    assert isinstance(papers, list)
    paper = papers[0]
    assert isinstance(paper, dict)
    paper["authors"] = ["Alice <A>", "Bob & Co.", "Carol", "David"]

    html = render_digest(data)

    assert (
        "<a href=\"https://arxiv.org/abs/2609.00002\">Matched &lt;Title&gt; &amp; Z_2</a><br>"
        "<small><i>Authors: Alice &lt;A&gt;, Bob &amp; Co., Carol, et al.</i></small><br>"
        "A matched abstract with x^2 and useful details. Second sentence."
    ) in html
    assert "David" not in html


def test_render_digest_shows_all_authors_when_three_or_fewer() -> None:
    data = load_papers()
    papers = data["papers"]
    assert isinstance(papers, list)
    paper = papers[0]
    assert isinstance(paper, dict)
    paper["authors"] = ["Alice", "Bob", "Carol"]

    html = render_digest(data)

    assert "<small><i>Authors: Alice, Bob, Carol</i></small><br>" in html
    assert "Carol, et al." not in html


def test_render_digest_omits_missing_or_empty_author_lists() -> None:
    data = load_papers()
    papers = data["papers"]
    assert isinstance(papers, list)
    papers[0]["authors"] = []
    papers[1].pop("authors", None)

    html = render_digest(data)

    assert "Authors:" not in html