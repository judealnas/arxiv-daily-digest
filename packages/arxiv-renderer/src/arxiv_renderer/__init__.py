from __future__ import annotations

import html
import re
from typing import Any

from .errors import short_error
from .input import prepare_digest

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
    papers_data = prepare_digest(papers_data)
    papers: list[Paper] = papers_data["papers"]
    date: str = papers_data["date"]
    run_id: str = papers_data["run_id"]
    count: int = papers_data["count"]
    matched_count: int = papers_data["matched_count"]
    if summaries is not None and set(summaries) - {p["id"] for p in papers if p["matched"]}:
        raise ValueError("summaries contain unknown or unmatched paper IDs")
    attempted: bool = summaries is not None or any(
        any(key in p for key in ("summary_status", "summary", "summary_error", "summary_error_code"))
        for p in papers
        if p["matched"]
    )
    failures: int = 0
    successes: int = 0
    entries: list[str] = []
    for number, paper in enumerate(papers, start=1):
        matched: bool = paper["matched"]
        prefix: str = "⭐" if matched else ""
        title: str = html.escape(clean_text(str(paper["title"])))
        authors_value: Any = paper.get("authors")
        authors: list[str] = (
            [name.strip() for name in authors_value if isinstance(name, str) and name.strip()]
            if isinstance(authors_value, list)
            else []
        )
        author_line: str = ""
        if authors:
            displayed_authors: list[str] = [html.escape(name) for name in authors[:3]]
            if len(authors) > 3:
                displayed_authors.append("et al.")
            author_line = f"<br><small><i>Authors: {', '.join(displayed_authors)}</i></small>"
        url: str = html.escape(str(paper.get("url", paper.get("abstract_link", ""))), quote=True)
        summary: Any = summaries.get(str(paper["id"])) if summaries is not None else paper.get("summary")
        valid: bool = (
            isinstance(summary, str)
            and bool(summary.strip())
            and (summaries is not None or paper.get("summary_status") == "ok")
        )
        body: str = (
            clean_text(summary)
            if matched and valid
            else (clean_text(str(paper["abstract"])) if matched else first_sentence(str(paper["abstract"])))
        )
        extra: str = ""
        diagnostic: str = ""
        if matched and valid:
            successes += 1
            metadata: dict[str, Any] = paper.get("summary_metadata") or {}
            model: Any = metadata.get("returned_model")
            if isinstance(model, str) and model.strip():
                extra = f"<br><small>Summarized by: {html.escape(model)}</small>"
            elif metadata.get("requested_model"):
                extra = f"<br><small>Requested model: {html.escape(str(metadata['requested_model']))}; actual model unavailable</small>"
        elif matched and attempted:
            failures += 1
            error: str = str(paper.get("summary_error") or "")
            message: str = short_error(paper.get("summary_error_code"), error)
            extra = f"<br><strong>Summary unavailable:</strong> {html.escape(message)}"
            if error:
                escaped_error: str = html.escape(error).replace("\n", "<br>")
                diagnostic = (
                    '\n<blockquote style="margin:8px 0 16px 16px;padding:10px 12px;'
                    "border-left:3px solid #9ca3af;background-color:#f3f4f6;color:#374151;"
                    'font-size:12px;line-height:1.4;word-wrap:break-word;word-break:break-word;">'
                    "<strong>Technical details — raw diagnostic</strong><br>"
                    f'<span style="font-family:Consolas,monospace;">{escaped_error}</span></blockquote>'
                )
        entries.append(
            f'<p>{prefix}{number}. <a href="{url}">{title}</a>{author_line}<br>{html.escape(body)}{extra}</p>{diagnostic}'
        )
    stage: str = "partial" if failures and successes else ("summarized" if attempted and not failures else "ranked")
    notice_html: str = f"<p>{html.escape(notice)}</p>" if notice else ""
    if failures:
        label: str = "summary" if failures == 1 else "summaries"
        notice_html += (
            f'<p role="alert"><strong>{failures} {label} unavailable; full abstracts shown below.</strong></p>'
        )
    configs: list[str] = []
    allowed: tuple[str, ...] = (
        "backend",
        "provider",
        "model",
        "max_length",
        "min_length",
        "batch_size",
        "num_beams",
        "papers_per_prompt",
        "chunk_count",
        "temperature",
        "max_tokens",
    )
    for config in [papers_data.get("summarizer_config")] + [p.get("summarizer_config") for p in papers]:
        if not isinstance(config, dict):
            continue
        description: str = "; ".join(f"{key}: {config[key]}" for key in allowed if config.get(key) is not None)
        if description and description not in configs:
            configs.append(description)
    config_html: str = ""
    if configs:
        config_html = (
            "<p><strong>Summarizer configuration</strong><br>" + "<br>".join(html.escape(c) for c in configs) + "</p>\n"
        )
    return (
        f'<html data-schema-version="1" data-digest-date="{html.escape(date, quote=True)}" '
        f'data-source-run-id="{html.escape(run_id, quote=True)}" data-build-stage="{stage}">\n'
        f"<body>\n<h1>arXiv quant-ph — {html.escape(date)}</h1>\n"
        f"<p>{count} papers, ⭐{matched_count} matched</p>\n{config_html}{notice_html}"
        + "\n".join(entries)
        + "\n</body>\n</html>\n"
    )
