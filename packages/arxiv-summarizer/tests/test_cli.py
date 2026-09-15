"""CLI behaviour tests: JSONL in -> JSONL out, backend selection.

Backend constructors are monkeypatched to fakes — no model load, no API
calls. These pin the CLI's plumbing contract.
"""

from __future__ import annotations

import io
import json
import sys

import pytest

import arxiv_summarizer.cli as cli
from arxiv_summarizer.models import ResultItem

SAMPLE_TEXT = (
    "Quantum repeaters are essential for long-distance quantum communication. "
    "We propose an architecture achieving 9 km repeater spacing."
)


def make_source_line(pid: str, text: str) -> str:
    return json.dumps({"id": pid, "text": text})


def run_cli(monkeypatch, stdin_text: str, argv: list[str] | None = None) -> str:
    """Run cli.main with stdin/stdout redirected; return stdout text."""
    buf = io.StringIO()
    old_stdin, old_stdout = sys.stdin, sys.stdout
    sys.stdin = io.StringIO(stdin_text)
    sys.stdout = buf
    try:
        cli.main(argv)
    finally:
        sys.stdin, sys.stdout = old_stdin, old_stdout
    return buf.getvalue()


class RecordingCLI:
    """Installs fake constructors on the cli module and records which fired."""

    def __init__(self, monkeypatch):
        self.constructed: dict[str, dict] = {}
        rec = self

        class FakeBart:
            def summarize_many(self, texts):
                rec.constructed["bart"] = True
                return [ResultItem(id=str(i), status="ok", summary="FAKE", error=None) for i, _ in enumerate(texts)]

        class FakeLlm:
            def summarize_many(self, texts):
                rec.constructed["llm"] = True
                return [ResultItem(id=str(i), status="ok", summary="FAKE", error=None) for i, _ in enumerate(texts)]

        monkeypatch.setattr(cli, "BartSummarizer", lambda **kw: FakeBart())
        monkeypatch.setattr(cli, "LlmSummarizer", lambda **kw: FakeLlm())


def test_default_backend_is_bart(monkeypatch):
    """Default invocation (no --backend) routes to BartSummarizer."""
    rec = RecordingCLI(monkeypatch)
    out = run_cli(monkeypatch, make_source_line("p1", SAMPLE_TEXT) + "\n")
    assert rec.constructed == {"bart": True}
    result = json.loads(out.strip())
    assert result["id"] == "p1"
    assert result["status"] == "ok"


def test_backend_llm_flag_routes_to_llm(monkeypatch):
    rec = RecordingCLI(monkeypatch)
    out = run_cli(monkeypatch, make_source_line("p1", SAMPLE_TEXT) + "\n", ["--backend", "llm"])
    assert rec.constructed == {"llm": True}
    result = json.loads(out.strip())
    assert result["id"] == "p1"


def test_cli_emits_valid_jsonl_lines(monkeypatch):
    """Multiple inputs -> one parseable JSON line per input, ids preserved."""
    RecordingCLI(monkeypatch)
    stdin = "\n".join([
        make_source_line("a", SAMPLE_TEXT),
        make_source_line("b", SAMPLE_TEXT),
    ]) + "\n"
    out = run_cli(monkeypatch, stdin)
    lines = [line for line in out.strip().splitlines() if line.strip()]
    assert len(lines) == 2
    assert [json.loads(line)["id"] for line in lines] == ["a", "b"]
