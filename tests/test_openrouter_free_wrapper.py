from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("model", ["openrouter/free", "cohere/north-mini-code:free"])
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
    print("[]")
elif args[:2] == ["python", "scripts/run_test_pipeline.py"]:
    assert "--debug" in args
    if os.environ["TEST_BACKEND"] == "openrouter":
        assert Path(args[args.index("--diagnostics-dir") + 1]).name == "diagnostics"
        assert args[args.index("--model") + 1] == os.environ["TEST_MODEL"]
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
    if model != "openrouter/free":
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
