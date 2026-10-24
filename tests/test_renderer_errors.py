from __future__ import annotations

import html
from html.parser import HTMLParser

import pytest
from arxiv_renderer import render_digest
from test_renderer_partial import digest


class DiagnosticParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_quote = False
        self.in_paragraph = False
        self.quoted = []
        self.outside = []
        self.quote_attrs = []

    def handle_starttag(self, tag, attrs):
        if tag == "p":
            self.in_paragraph = True
        if tag == "blockquote":
            assert not self.in_paragraph, "blockquote must not be nested in a paragraph"
            self.in_quote = True
            self.quote_attrs.append(dict(attrs))
        if tag == "br" and self.in_quote:
            self.quoted.append("\n")

    def handle_endtag(self, tag):
        if tag == "p":
            self.in_paragraph = False
        if tag == "blockquote":
            self.in_quote = False

    def handle_data(self, data):
        (self.quoted if self.in_quote else self.outside).append(data)


@pytest.mark.parametrize(
    "code,message",
    [
        ("invalid_json", "The model returned invalid JSON."),
        ("id_mismatch", "The response did not match the requested papers."),
        ("http_429", "The provider rejected the request: HTTP 429."),
        ("http_503", "The provider rejected the request: HTTP 503."),
        ("timeout", "The summarization request timed out."),
        ("output_token_limit", "The model reached its output-token limit without returning a usable summary."),
        ("invalid_summary", "The response contained an invalid ID or summary."),
        ("empty_response", "No usable summary was returned."),
        ("generation_error", "Local summary generation failed."),
        ("request_failed", "The summarization request failed."),
        ("invalid_response", "The model returned an invalid response format."),
        ("unknown", "Summarization failed; see technical details."),
    ],
)
def test_short_error_and_expanded_raw_quote(code, message) -> None:
    data = digest()
    raw = 'Full error <script> & "quotes"\nSecond line. ' + "long diagnostic " * 200
    data["papers"][1].update(summary_error_code=code, summary_error=raw)
    rendered = render_digest(data)
    parser = DiagnosticParser()
    parser.feed(rendered)
    assert message in "".join(parser.outside)
    assert raw not in "".join(parser.outside)
    assert raw in "".join(parser.quoted)
    assert len(parser.quote_attrs) == 1
    assert "border-left:" in parser.quote_attrs[0]["style"]
    assert "background-color:" in parser.quote_attrs[0]["style"]
    assert "font-size:" in parser.quote_attrs[0]["style"]
    assert "Technical details — raw diagnostic" in "".join(parser.quoted)
    assert "<script>" not in rendered
    assert "<details" not in rendered and "display:none" not in rendered
    assert "Useful summary." in rendered and "Full fallback. Keep this too." in rendered
    assert rendered.count("<a href=") == 3


@pytest.mark.parametrize(
    "raw,message",
    [
        ("Invalid summary response: response is not JSON", "The model returned invalid JSON."),
        ("HTTPError: HTTP Error 429: quota body", "The provider rejected the request: HTTP 429."),
        ("TimeoutError: details", "The summarization request timed out."),
        ("Provider body mentions 429 and quota", "Summarization failed; see technical details."),
    ],
)
def test_legacy_diagnostics_use_only_known_prefixes(raw, message) -> None:
    data = digest()
    data["papers"][1]["summary_error"] = raw
    rendered = render_digest(data)
    assert message in rendered
    assert html.escape(raw) in rendered


def test_missing_error_uses_short_missing_summary_message() -> None:
    data = digest()
    data["papers"][1].pop("summary_error")
    rendered = render_digest(data)
    assert "No usable summary was returned." in rendered
    assert "<blockquote" not in rendered


@pytest.mark.parametrize("value", [[], {}, 17])
def test_malformed_error_code_is_rejected(value) -> None:
    data = digest()
    data["papers"][1]["summary_error_code"] = value
    with pytest.raises(ValueError, match="summary_error_code"):
        render_digest(data)


def test_error_code_alone_is_a_failed_summary_attempt() -> None:
    data = digest()
    for paper in data["papers"]:
        paper.pop("summary", None)
        paper.pop("summary_status", None)
        paper.pop("summary_error", None)
    data["papers"][1]["summary_error_code"] = "timeout"
    assert "The summarization request timed out." in render_digest(data)


def test_explicit_unknown_code_does_not_guess_from_body() -> None:
    data = digest()
    data["papers"][1].update(summary_error_code="new_unknown_code", summary_error="TimeoutError: text")
    rendered = render_digest(data)
    assert "Summarization failed; see technical details." in rendered
    assert "request timed out" not in rendered


def test_email_html_and_plain_parts_keep_short_error_and_full_details(tmp_path, monkeypatch) -> None:
    import importlib.util
    import json
    import sys
    from pathlib import Path

    from arxiv_renderer.cli import main as render_main

    spec = importlib.util.spec_from_file_location(
        "test_email_sender", Path(__file__).parents[1] / "arxiv-email-sender.py"
    )
    assert spec is not None and spec.loader is not None
    sender = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sender)
    data = digest()
    raw = "Provider diagnostic <unsafe> & details\nSecond diagnostic line."
    data["papers"][1].update(summary_error_code="invalid_json", summary_error=raw)
    source = tmp_path / "papers.json"
    source.write_text(json.dumps(data))
    output = tmp_path / "digest.html"
    assert render_main([str(source), "--output", str(output)]) == 0
    sent = []
    monkeypatch.setattr(
        sender, "load_env", lambda _: {"EMAIL_ADDRESS": "sender@example.test", "EMAIL_PASSWORD": "test-only"}
    )
    monkeypatch.setattr(sender, "send_email", lambda **kwargs: sent.append(kwargs))
    monkeypatch.setattr(sys, "argv", ["arxiv-email-sender.py", "--html", str(output), "--to", "recipient@example.test"])
    assert sender.main() == 0
    assert len(sent) == 1
    parser = DiagnosticParser()
    parser.feed(sent[0]["html_body"])
    assert raw in "".join(parser.quoted)
    assert "The model returned invalid JSON." in "".join(parser.outside)
    assert "Full fallback. Keep this too." in sent[0]["plain_body"]
    assert "The model returned invalid JSON." in sent[0]["plain_body"]
    assert raw in sent[0]["plain_body"]
    assert "JSON.\nTechnical details — raw diagnostic\n" in sent[0]["plain_body"]


def test_success_never_renders_stale_diagnostics() -> None:
    data = digest()
    data["papers"][1].update(summary_status="ok", summary="Good.")
    rendered = render_digest(data)
    assert "<blockquote" not in rendered
    assert "Summary unavailable:" not in rendered
