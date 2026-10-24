"""Validate the rendering boundary without fetching or inferring listing dates."""

from __future__ import annotations

from datetime import date as calendar_date
from typing import Any
from urllib.parse import urlsplit


def prepare_digest(data: Any, *, date: str | None = None, run_id: str | None = None) -> dict[str, Any]:
    """Accept pipeline arrays or digest objects, preserving supplied metadata."""
    if isinstance(data, list):
        digest: dict[str, Any] = {"papers": data}
    elif isinstance(data, dict):
        digest = data.copy()
    else:
        raise ValueError("JSON root must be an array or object")
    if date is not None and digest.get("date") not in (None, date):
        raise ValueError("--date conflicts with input date")
    listing_date: Any = date if date is not None else digest.get("date")
    if not isinstance(listing_date, str):
        raise ValueError("--date YYYY-MM-DD is required when input has no date")
    try:
        if calendar_date.fromisoformat(listing_date).isoformat() != listing_date:
            raise ValueError
    except ValueError as exc:
        raise ValueError("date must be a valid YYYY-MM-DD calendar date") from exc
    papers: Any = digest.get("papers")
    if not isinstance(papers, list):
        raise ValueError("papers must be an array")
    ids: set[str] = set()
    for index, paper in enumerate(papers):
        if not isinstance(paper, dict):
            raise ValueError(f"paper {index} must be an object")
        pid: Any = paper.get("id")
        if not isinstance(pid, str) or not pid.strip() or pid in ids:
            raise ValueError("paper IDs must be unique nonempty strings")
        ids.add(pid)
        for key in ("title", "abstract"):
            if not isinstance(paper.get(key), str):
                raise ValueError(f"paper {pid}: {key} must be a string")
        if not isinstance(paper.get("matched"), bool):
            raise ValueError(f"paper {pid}: matched must be a boolean")
        url: Any = paper.get("url", paper.get("abstract_link"))
        if not isinstance(url, str):
            raise ValueError(f"paper {pid}: url or abstract_link is required")
        parsed = urlsplit(url)
        if parsed.scheme not in ("https", "http") or not parsed.netloc:
            raise ValueError(f"paper {pid}: link must be an HTTP(S) URL")
        code: Any = paper.get("summary_error_code")
        if code is not None and not isinstance(code, str):
            raise ValueError(f"paper {pid}: summary_error_code must be a string or null")
        for key in ("summary_metadata", "summarizer_config"):
            if key in paper and paper[key] is not None and not isinstance(paper[key], dict):
                raise ValueError(f"paper {pid}: {key} must be an object or null")
    matched_count: int = sum(paper["matched"] for paper in papers)
    for key, expected in (("count", len(papers)), ("matched_count", matched_count)):
        if key in digest and (type(digest[key]) is not int or digest[key] != expected):
            raise ValueError(f"{key} does not match paper entries")
        digest[key] = expected
    if run_id is not None and digest.get("run_id") not in (None, run_id):
        raise ValueError("--run-id conflicts with input run_id")
    digest["run_id"] = run_id if run_id is not None else digest.get("run_id", "")
    if not isinstance(digest["run_id"], str):
        raise ValueError("run_id must be a string")
    digest["date"] = listing_date
    return digest
