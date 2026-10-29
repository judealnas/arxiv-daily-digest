from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_fetch_boundary(
    tmp_path: Path, *, today: str = "2026-10-26", listing_date: str | None = None,
    payload: str = "[]", fetch_status: int = 0,
) -> subprocess.CompletedProcess[str]:
    """Run an isolated wrapper copy; forbid every downstream stage."""
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=test-only\n")
    wrapper = tmp_path / "openrouter_free_pipeline.sh"
    wrapper.write_text((ROOT / wrapper.name).read_text())
    stub = tmp_path / "uv"
    stub.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[4:]
with Path(os.environ["TEST_CALLS"]).open("a") as stream:
    stream.write(json.dumps(args) + "\\n")
if args[0] == "arxiv-catchup":
    print(os.environ["TEST_PAYLOAD"])
    raise SystemExit(int(os.environ["TEST_FETCH_STATUS"]))
raise SystemExit(17)
''')
    stub.chmod(0o700)
    clock = tmp_path / "date"
    clock.write_text('''#!/usr/bin/env python3
import os, sys
args = sys.argv[1:]
if "-d" not in args:
    args = ["-d", os.environ["TEST_TODAY"], *args]
else:
    index = args.index("-d") + 1
    if args[index] == "yesterday":
        args[index] = os.environ["TEST_TODAY"] + " -1 day"
os.execv("/usr/bin/date", ["date", *args])
''')
    clock.chmod(0o700)
    env = dict(os.environ, HERMES_HOME=str(tmp_path), TMPDIR=str(tmp_path),
               UV=str(stub), PATH=f"{tmp_path}:{os.environ['PATH']}",
               TEST_TODAY=today, TEST_PAYLOAD=payload, TEST_FETCH_STATUS=str(fetch_status),
               TEST_CALLS=str(tmp_path / "calls.jsonl"))
    return subprocess.run(["sh", str(wrapper), *([] if listing_date is None else [listing_date])],
                          env=env, capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize("today,expected", [
    ("2026-10-26", "2026-10-23"),
    ("2026-10-27", "2026-10-26"),
    ("2026-10-28", "2026-10-27"),
    ("2026-10-29", "2026-10-28"),
    ("2026-10-30", "2026-10-29"),
    ("2026-11-02", "2026-10-30"),
    ("2027-01-01", "2026-12-31"),
])
def test_openrouter_default_date_is_previous_weekday(tmp_path: Path, today: str, expected: str) -> None:
    import json

    run_fetch_boundary(tmp_path, today=today, payload='[{"id":"test"}]')
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert calls[0] == ["arxiv-catchup", "quant-ph", expected]


@pytest.mark.parametrize("listing_date", ["2026-10-24", "2026-10-25", "not-a-date"])
def test_openrouter_explicit_date_is_preserved(tmp_path: Path, listing_date: str) -> None:
    import json

    run_fetch_boundary(tmp_path, listing_date=listing_date, payload='[{"id":"test"}]')
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert calls[0] == ["arxiv-catchup", "quant-ph", listing_date]


@pytest.mark.parametrize("payload", ["[]", " \n[ ]\n"])
def test_openrouter_empty_fetch_is_successful_skip(tmp_path: Path, payload: str) -> None:
    import json

    result = run_fetch_boundary(tmp_path, listing_date="2026-10-25", payload=payload)
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert result.returncode == 0, result.stderr
    assert calls == [["arxiv-catchup", "quant-ph", "2026-10-25"]]
    assert "SKIPPED" in result.stderr and "2026-10-25" in result.stderr
    assert "no email sent" in result.stderr
    assert not list(tmp_path.glob("arxiv-digest.*"))


@pytest.mark.parametrize("payload", ["", "not json", "{}", "null", "0", '"[]"'])
def test_openrouter_invalid_fetch_fails_before_scoring(tmp_path: Path, payload: str) -> None:
    import json

    result = run_fetch_boundary(tmp_path, listing_date="2026-10-25", payload=payload)
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert result.returncode != 0
    assert calls == [["arxiv-catchup", "quant-ph", "2026-10-25"]]
    assert "Invalid fetch output" in result.stderr
    assert "SKIPPED" not in result.stderr
    artifacts = list(tmp_path.glob("arxiv-digest.*"))
    assert len(artifacts) == 1
    assert (artifacts[0] / "fetched.json").read_text().strip() == payload.strip()


def test_openrouter_failed_fetch_is_not_an_empty_skip(tmp_path: Path) -> None:
    import json

    result = run_fetch_boundary(tmp_path, payload="[]", fetch_status=9)
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert result.returncode == 9
    assert len(calls) == 1
    assert "SKIPPED" not in result.stderr
    assert "FAILED" in result.stderr


@pytest.mark.parametrize("model", ["recursive/flint-v1.0:free", "cohere/north-mini-code:free"])
@pytest.mark.parametrize("script", ["openrouter_free_pipeline.sh", "bart_pipeline.sh"])
@pytest.mark.parametrize("failure", [False, True])
def test_wrapper_artifacts_and_failure_guard(tmp_path: Path, failure: bool, script: str, model: str) -> None:
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=test-only\n")
    stub = tmp_path / "uv"
    stub.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[4:]
if args[0] == "arxiv-catchup":
    assert args[2] == "2026-10-22"
    print(json.dumps([dict(id="test", title="Title", abstract="Abstract.")]))
elif args[:2] == ["python", "scripts/run_test_pipeline.py"]:
    assert "--debug" in args
    if os.environ["TEST_BACKEND"] == "openrouter":
        prompt = args[args.index("--user-prompt") + 1]
        assert prompt == (
            "Summarize the following scientific abstracts in up to 5 sentences. "
            "Capture significant contributions and quantitative results."
        )
        assert Path(args[args.index("--diagnostics-dir") + 1]).name == "diagnostics"
        assert args[args.index("--model") + 1] == os.environ["TEST_MODEL"]
        assert args[args.index("--papers-per-prompt") + 1] == "1"
    else:
        assert "--diagnostics-dir" not in args
        assert args[args.index("--model") + 1] == "facebook/bart-large-cnn"
    assert args[args.index("--backend") + 1] == os.environ["TEST_BACKEND"]
    bad = os.environ["TEST_FAILURE"] == "1"
    print(json.dumps([dict(id="test", title="Title", abstract="Abstract.",
        abstract_link="https://arxiv.org/abs/test", matched=True,
        summary_status="error" if bad else "ok", summary=None if bad else "Summary.",
        summary_error="LLM HTTP 429" if bad else None)]))
elif args[0] == "arxiv-render":
    assert args[args.index("--date") + 1] == "2026-10-22"
    assert args[args.index("--run-id") + 1].startswith("arxiv-digest.")
    import subprocess
    subprocess.run([os.environ["TEST_PYTHON"], "-m", "arxiv_renderer", *args[1:]], check=True)
    Path(os.environ["TEST_RENDERED"]).touch()
elif args[:2] == ["python", "-"]:
    os.execv(os.environ["TEST_PYTHON"], [os.environ["TEST_PYTHON"], *args[1:]])
elif args[:2] == ["python", "arxiv-email-sender.py"]:
    assert args[args.index("--subject") + 1] == "arXiv quant-ph daily digest — 2026-10-22"
    rendered = Path(args[args.index("--html") + 1]).read_text()
    if os.environ["TEST_FAILURE"] == "1":
        assert "Abstract." in rendered and "LLM HTTP 429" in rendered
        assert "1 summary unavailable" in rendered
    else:
        assert "Summary." in rendered
    Path(os.environ["TEST_SENT"]).touch()
else:
    raise AssertionError(args)
''')
    stub.chmod(0o700)
    env = dict(os.environ, HERMES_HOME=str(tmp_path), TMPDIR=str(tmp_path),
               UV=str(stub), TEST_FAILURE=str(int(failure)),
               TEST_PYTHON=str(ROOT / ".venv/bin/python"), TEST_SENT=str(tmp_path / "sent"),
               TEST_RENDERED=str(tmp_path / "rendered"))
    env["TEST_MODEL"] = model
    env.pop("OPENROUTER_MODEL", None)
    if model != "recursive/flint-v1.0:free":
        env["OPENROUTER_MODEL"] = model
    env["TEST_BACKEND"] = "bart" if script == "bart_pipeline.sh" else "openrouter"
    if env["TEST_BACKEND"] == "bart":
        (tmp_path / ".env").unlink()
        env.pop("OPENROUTER_API_KEY", None)
    wrapper = tmp_path / script
    wrapper.write_text((ROOT / script).read_text())
    result = subprocess.run(["sh", str(wrapper), "2026-10-22"],
                            cwd=ROOT, env=env, capture_output=True, text=True)
    assert (tmp_path / "rendered").exists(), result.stderr
    artifacts = list(tmp_path.glob("arxiv-digest.*"))
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "sent").exists()
    assert not artifacts
    assert "test-only" not in result.stdout + result.stderr
