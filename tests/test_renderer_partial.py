from __future__ import annotations

import pytest
from arxiv_renderer import render_digest


def digest() -> dict:
    return {
        "date": "2026-10-23",
        "run_id": "test",
        "count": 3,
        "matched_count": 2,
        "papers": [
            {
                "id": "a",
                "matched": True,
                "title": "Good",
                "abstract": "Original. More.",
                "abstract_link": "https://arxiv.org/abs/a",
                "summary_status": "ok",
                "summary": "Useful summary.",
            },
            {
                "id": "b",
                "matched": True,
                "title": "Failed",
                "abstract": "Full fallback. Keep this too.",
                "url": "https://arxiv.org/abs/b",
                "summary_status": "error",
                "summary": "Untrusted summary.",
                "summary_error": "HTTP <429> & quota",
            },
            {
                "id": "c",
                "matched": False,
                "title": "Other",
                "abstract": "First. Second.",
                "url": "https://arxiv.org/abs/c",
            },
        ],
    }


def test_error_without_status_is_still_reported() -> None:
    data = digest()
    for paper in data["papers"]:
        paper.pop("summary_status", None)
        paper.pop("summary", None)
    rendered = render_digest(data)
    assert "HTTP &lt;429&gt; &amp; quota" in rendered
    assert "2 summaries unavailable" in rendered


def test_summary_map_rejects_unknown_ids() -> None:
    with pytest.raises(ValueError, match="unknown"):
        render_digest(digest(), summaries={"invented": "Summary."})


def test_partial_summaries_keep_success_and_show_errors() -> None:
    rendered = render_digest(digest())
    assert "Useful summary." in rendered
    assert "Original. More." not in rendered
    assert "Full fallback. Keep this too." in rendered
    assert "Untrusted summary." not in rendered
    assert "HTTP &lt;429&gt; &amp; quota" in rendered
    assert "1 summary unavailable" in rendered
    assert 'data-build-stage="partial"' in rendered
    assert "Second." not in rendered
    assert rendered.count("<a href=") == 3


@pytest.mark.parametrize("status,summary", [("ok", ""), ("ok", None), ("error", "text"), (None, "text")])
def test_invalid_summary_falls_back_visibly(status, summary) -> None:
    data = digest()
    data["papers"][0].update(summary_status=status, summary=summary)
    rendered = render_digest(data)
    assert "Original. More." in rendered
    assert "2 summaries unavailable" in rendered


def test_partial_explicit_summary_map_falls_back_visibly() -> None:
    rendered = render_digest(digest(), summaries={"a": "Override."}, notice="Warning <unsafe>")
    assert "Override." in rendered
    assert "Full fallback. Keep this too." in rendered
    assert "Warning &lt;unsafe&gt;" in rendered
