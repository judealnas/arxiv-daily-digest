"""Shared type aliases for the summarizer pipeline."""

from __future__ import annotations

from typing import NotRequired, TypedDict


class SourceItem(TypedDict):
    """One entry in the input JSONL stream.

    Parameters
    ----------
    id : str
        Opaque identifier echoed back in the output so client processes
        can match results to their source data (e.g. an arXiv paper ID).
    text : str
        The raw text to summarise (e.g. a paper abstract on a single line).
    """

    id: str
    text: str


class SummaryMetadata(TypedDict):
    requested_model: str | None
    returned_model: str | None
    finish_reason: str | None


class ResultItem(TypedDict):
    """One entry in the output JSONL stream.

    ``status`` is ``"ok"`` when *summary* is populated, ``"error"`` when
    *error* is populated.

    Parameters
    ----------
    id : str
        Mirrors the ``id`` of the corresponding input item.
    status : str
        ``"ok"`` or ``"error"``.
    summary : str | None
        The LLM-generated summary, or ``None`` on failure.
    error : str | None
        Error message on failure, or ``None`` on success.
    """

    id: str
    status: str  # "ok" | "error"
    summary: str | None
    error: str | None
    summary_metadata: NotRequired[SummaryMetadata]
    summary_error_code: NotRequired[str]


# ── internal helpers ───────────────────────────────────────────────────────


def dump_result(item: ResultItem) -> str:
    """Serialize a ResultItem as a single JSON line (no trailing newline)."""
    import json

    return json.dumps(item, ensure_ascii=False, sort_keys=True)
