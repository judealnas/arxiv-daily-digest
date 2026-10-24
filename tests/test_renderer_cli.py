from __future__ import annotations

import json
import subprocess
import sys

import pytest
from test_renderer_partial import digest


def invoke(*args, text=""):
    return subprocess.run([sys.executable, "-m", "arxiv_renderer", *args], input=text, text=True, capture_output=True)


def test_stdin_stdout_partial_digest() -> None:
    result = invoke("--date", "2026-10-23", text=json.dumps(digest()["papers"]))
    assert result.returncode == 0, result.stderr
    assert "Useful summary." in result.stdout
    assert "1 summary unavailable" in result.stdout
    assert result.stdout.startswith("<html")
    assert result.stderr == ""


def test_file_input_output_and_canonical_metadata(tmp_path) -> None:
    source = tmp_path / "input.json"
    output = tmp_path / "output.html"
    source.write_text(json.dumps(digest()))
    result = invoke(str(source), "--output", str(output))
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert 'data-source-run-id="test"' in output.read_text()


@pytest.mark.parametrize(
    "text,args,error",
    [
        ("[]", [], "--date"),
        ("[]", ["--date", "2026-02-30"], "date"),
        ("{", ["--date", "2026-10-23"], "JSON"),
        ("42", ["--date", "2026-10-23"], "array or object"),
        (json.dumps({**digest(), "count": 99}), [], "count"),
        (json.dumps(digest()), ["--date", "2026-10-22"], "conflict"),
    ],
)
def test_invalid_input_does_not_overwrite_output(tmp_path, text, args, error) -> None:
    output = tmp_path / "existing.html"
    output.write_text("preserve")
    result = invoke(*args, "--output", str(output), text=text)
    assert result.returncode != 0
    assert error in result.stderr
    assert result.stdout == ""
    assert "Traceback" not in result.stderr
    assert output.read_text() == "preserve"


@pytest.mark.parametrize(
    "change",
    [
        {"id": ""},
        {"matched": "false"},
        {"title": None},
        {"summary_metadata": []},
        {"url": "javascript:alert(1)"},
    ],
)
def test_malformed_paper_is_rejected(change) -> None:
    data = digest()
    data["papers"][0].update(change)
    result = invoke(text=json.dumps(data))
    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    assert not result.stdout


def test_duplicate_paper_ids_rejected() -> None:
    data = digest()
    data["papers"][1]["id"] = "a"
    result = invoke(text=json.dumps(data))
    assert result.returncode != 0
    assert "unique" in result.stderr


def test_missing_file_reports_read_error(tmp_path) -> None:
    result = invoke(str(tmp_path / "absent.json"), "--date", "2026-10-23")
    assert result.returncode != 0
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr
