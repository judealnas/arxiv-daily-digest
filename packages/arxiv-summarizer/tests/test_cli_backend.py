"""CLI behaviour tests: JSONL in -> JSONL out, backend selection.

Backend constructors are monkeypatched to fakes — no model load, no API
calls. These pin the CLI's plumbing contract.
"""

from __future__ import annotations

import io
import json
import sys

import arxiv_summarizer.cli as cli
import pytest
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
            def summarize_items(self, source):
                rec.constructed["llm"] = True
                return [ResultItem(id=item["id"], status="ok", summary="FAKE", error=None) for item in source]

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


def test_llm_cli_emits_identity_results_without_positional_restoration(monkeypatch):
    class FakeLlm:
        def summarize_items(self, source):
            return [
                ResultItem(id=source[1]["id"], status="ok", summary="second", error=None),
                ResultItem(id=source[0]["id"], status="error", summary=None, error="bad group"),
            ]

    monkeypatch.setattr(cli, "LlmSummarizer", lambda **kw: FakeLlm())
    out = run_cli(monkeypatch, make_source_line("a", "first") + "\n" + make_source_line("b", "second") + "\n", ["--backend", "llm"])

    assert [json.loads(line) for line in out.splitlines()] == [
        {"id": "b", "status": "ok", "summary": "second", "error": None},
        {"id": "a", "status": "error", "summary": None, "error": "bad group"},
    ]


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--backend", "llm", "--chunk", "2", "--papers-per-prompt", "2"], "cannot be combined"),
        (["--backend", "openrouter", "--chunk", "2"], "not supported"),
        (["--backend", "bart", "--papers-per-prompt", "2"], "not supported"),
        (["--backend", "llm", "--papers-per-prompt", "0"], "must be positive"),
    ],
)
def test_cli_rejects_invalid_grouping_options(monkeypatch, argv, message, capsys):
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, "", argv)
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    "line",
    [
        "[]",
        "null",
        '{"id":7,"text":"abstract"}',
        '{"id":"","text":"abstract"}',
        '{"id":"paper","text":42}',
    ],
)
def test_invalid_jsonl_record_is_reported_without_traceback(monkeypatch, capsys, line):
    with pytest.raises(SystemExit) as exc:
        run_cli(monkeypatch, line + "\n", ["--backend", "llm"])
    assert exc.value.code == 1
    error = json.loads(capsys.readouterr().err.strip())
    assert error["status"] == "error"
    assert "Invalid JSONL line 1" in error["error"]


def test_duplicate_source_ids_rejected_before_backend_is_built(monkeypatch, capsys):
    def should_not_build(**kwargs):
        raise AssertionError("backend must not be built")

    monkeypatch.setattr(cli, "LlmSummarizer", should_not_build)
    stdin = make_source_line("a", "first") + "\n" + make_source_line("a", "second") + "\n"
    with pytest.raises(SystemExit) as exc:
        run_cli(monkeypatch, stdin, ["--backend", "llm"])
    assert exc.value.code == 1
    error = json.loads(capsys.readouterr().err.strip())
    assert "Duplicate paper ID" in error["error"]


def test_empty_user_prompt_is_argparse_error_without_request(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        run_cli(monkeypatch, make_source_line("a", "text") + "\n", ["--backend", "llm", "--user-prompt", "   "])
    assert exc.value.code == 2
    assert "user_prompt" in capsys.readouterr().err.lower()


def test_user_prompt_file_is_forwarded_as_exact_multiline_guidance(monkeypatch, tmp_path):
    guidance = "Use five sentences for quantum networking readers.\nEmphasize numbers.\n"
    prompt_file = tmp_path / "summary-prompt.txt"
    prompt_file.write_text(guidance, encoding="utf-8")
    passed = []

    class FakeLlm:
        def summarize_items(self, source):
            return [ResultItem(id=item["id"], status="ok", summary="FAKE", error=None) for item in source]

    monkeypatch.setattr(cli, "LlmSummarizer", lambda **kwargs: passed.append(kwargs) or FakeLlm())
    out = run_cli(
        monkeypatch,
        make_source_line("paper-1", SAMPLE_TEXT) + "\n",
        ["--backend", "llm", "--user-prompt-file", str(prompt_file)],
    )
    assert passed[0]["user_prompt"] == guidance
    assert json.loads(out)["id"] == "paper-1"


def test_user_prompt_file_reaches_grouped_prompt_with_real_backend(monkeypatch, tmp_path):
    from arxiv_summarizer.llm import LlmSummarizer

    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Use five sentences for physicists.\nEmphasize numbers.", encoding="utf-8")
    calls = []

    class FakeClient:
        def chat(self, messages, **kwargs):
            calls.append(messages)
            return json.dumps([
                {"id": "b", "summary": "Second summary."},
                {"id": "a", "summary": "First summary."},
            ])

    monkeypatch.setattr(cli, "LlmSummarizer", lambda **kwargs: LlmSummarizer(client=FakeClient(), **kwargs))
    source = make_source_line("a", "First abstract") + "\n" + make_source_line("b", "Second abstract") + "\n"
    output = run_cli(monkeypatch, source, ["--backend", "llm", "--user-prompt-file", str(prompt_file)])

    assert "Use five sentences for physicists." in calls[0][1]["content"]
    assert '"id": "a"' in calls[0][1]["content"]
    assert "exactly once" in calls[0][0]["content"]
    assert [json.loads(line)["id"] for line in output.splitlines()] == ["a", "b"]


def test_user_prompt_file_cannot_be_combined_with_inline_prompt(monkeypatch, tmp_path, capsys):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Five sentences.", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        run_cli(monkeypatch, "", ["--backend", "llm", "--user-prompt", "Emphasize numbers", "--user-prompt-file", str(prompt_file)])
    assert exc.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


def test_missing_user_prompt_file_is_controlled_error(monkeypatch, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        run_cli(
            monkeypatch,
            make_source_line("paper-1", SAMPLE_TEXT) + "\n",
            ["--backend", "llm", "--user-prompt-file", str(tmp_path / "missing.txt")],
        )
    assert exc.value.code == 2
    assert "--user-prompt-file" in capsys.readouterr().err


def test_bart_rejects_llm_user_prompt_file(monkeypatch, tmp_path, capsys):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("{text}", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        run_cli(monkeypatch, "", ["--backend", "bart", "--user-prompt-file", str(prompt_file)])
    assert exc.value.code == 2
    assert "--user-prompt-file" in capsys.readouterr().err


def test_bart_num_beams_flag_is_forwarded(monkeypatch):
    passed = []
    monkeypatch.setattr(cli, "BartSummarizer", lambda **kwargs: passed.append(kwargs) or object())
    cli.build_backend(cli.build_parser().parse_args(["--backend", "bart", "--num-beams", "7"]))
    assert passed[0]["num_beams"] == 7
