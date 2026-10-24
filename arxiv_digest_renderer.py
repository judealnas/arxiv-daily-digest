from __future__ import annotations

import html
import re
from typing import Any

Paper = dict[str, Any]
Digest = dict[str, Any]


def clean_text(text: str) -> str:
    value: str = text.replace("\\n", " ").replace("\n", " ")
    value = re.sub(r"\$\$.*?\$\$", "", value, flags=re.DOTALL)
    value = re.sub(r"\$([^$]*)\$", r"\1", value)
    value = re.sub(r"\\\((.*?)\\\)", r"\1", value)
    value = re.sub(r"\\\[(.*?)\\\]", r"\1", value)
    value = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\([,_%#&])", r"\1", value)
    value = re.sub(r"\\[a-zA-Z]+", "", value)
    value = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", value)
    return " ".join(value.replace("~", " ").split())


def first_sentence(text: str) -> str:
    cleaned: str = clean_text(text)
    match: re.Match[str] | None = re.match(r"^(.*?[.!?])(?:\s|$)", cleaned)
    return match.group(1) if match else cleaned


def render_digest(papers_data: Digest, summaries: dict[str, str] | None = None, notice: str | None = None) -> str:
    papers: list[Paper] = papers_data["papers"]
    date: str = papers_data["date"]
    run_id: str = papers_data["run_id"]
    count: int = papers_data["count"]
    matched_count: int = papers_data["matched_count"]
    stage: str = "summarized" if summaries is not None else "ranked"
    entries: list[str] = []
    for number, paper in enumerate(papers, start=1):
        matched: bool = paper["matched"]
        prefix: str = "⭐" if matched else ""
        title: str = html.escape(clean_text(str(paper["title"])))
        url: str = html.escape(str(paper["url"]), quote=True)
        body: str = summaries[str(paper["id"])] if summaries is not None and matched else (
            clean_text(str(paper["abstract"])) if matched else first_sentence(str(paper["abstract"]))
        )
        entries.append(f'<p>{prefix}{number}. <a href="{url}">{title}</a><br>{html.escape(body)}</p>')
    notice_html: str = f"<p>{html.escape(notice)}</p>" if notice else ""
    return (
        f'<html data-schema-version="1" data-digest-date="{html.escape(date, quote=True)}" '
        f'data-source-run-id="{html.escape(run_id, quote=True)}" data-build-stage="{stage}">\n'
        f"<body>\n<h1>arXiv quant-ph — {html.escape(date)}</h1>\n"
        f"<p>{count} papers, ⭐{matched_count} matched</p>\n{notice_html}"
        + "\n".join(entries)
        + "\n</body>\n</html>\n"
    )
