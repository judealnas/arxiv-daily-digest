"""Regression tests for the top-level daily digest CLI."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / "arxiv-daily-digest.py"
HTML_FIXTURE = ROOT / "packages/arxiv-catchup/tests/fixtures/arxiv-catchup-sample.html"
INTERESTS_FIXTURE = ROOT / "packages/arxiv-interest-sort/tests/fixtures/sample_interests.yaml"


def test_from_html_scores_and_ranks_papers_with_list_interest_config() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--from-html",
            str(HTML_FIXTURE),
            "--config",
            str(INTERESTS_FIXTURE),
        ],
        capture_output=True,
        check=False,
        cwd=ROOT,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    papers = json.loads(result.stdout)

    assert [paper["id"] for paper in papers] == [
        "2601.03779",
        "2601.03772",
        "2501.12345v2",
        "2601.03750",
        "2601.03778",
        "2601.03810",
    ]
    assert [
        (paper["matched"], paper["match_score"], paper["match_count"])
        for paper in papers
    ] == [
        (True, 5, 5),
        (True, 3, 3),
        (False, 0, 0),
        (False, 0, 0),
        (False, 0, 0),
        (False, 0, 0),
    ]
    assert [paper["matched_topics"] for paper in papers] == [
        ["quantum networking"],
        ["quantum simulation"],
        [],
        [],
        [],
        [],
    ]


def _load_daily_digest_module():
    """Load the hyphenated top-level script as an importable test module."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("daily_digest_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixed_datetime(module, value):
    class FixedDatetime(module.datetime):
        @classmethod
        def now(cls, tz=None):
            assert tz is module.timezone.utc
            return cls.fromisoformat(value).replace(tzinfo=module.timezone.utc)

    return FixedDatetime


def test_from_html_metadata_uses_explicit_date_for_stdout(capsys) -> None:
    module = _load_daily_digest_module()

    module.main(
        [
            "--from-html",
            str(HTML_FIXTURE),
            "--date",
            "2026-01-15",
            "--include-metadata",
            "--config",
            str(INTERESTS_FIXTURE),
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert output["date"] == "2026-01-15"
    assert output["category"] == "quant-ph"


def test_from_html_metadata_uses_explicit_date_for_file(tmp_path, capsys) -> None:
    module = _load_daily_digest_module()
    output_path = tmp_path / "digest.json"

    module.main(
        [
            "--from-html",
            str(HTML_FIXTURE),
            "--date",
            "2026-01-15",
            "--include-metadata",
            "--output",
            str(output_path),
            "--config",
            str(INTERESTS_FIXTURE),
        ]
    )

    assert json.loads(output_path.read_text(encoding="utf-8"))["date"] == "2026-01-15"
    assert "Wrote 6 papers" in capsys.readouterr().out


def test_from_html_metadata_requires_an_explicit_date(capsys) -> None:
    import pytest

    module = _load_daily_digest_module()

    with pytest.raises(SystemExit) as exit_info:
        module.main(
            [
                "--from-html",
                str(HTML_FIXTURE),
                "--include-metadata",
                "--config",
                str(INTERESTS_FIXTURE),
            ]
        )

    assert exit_info.value.code == 2
    assert "--date is required with --from-html and --include-metadata" in capsys.readouterr().err


def test_positional_date_drives_fetch_url_and_metadata(monkeypatch, capsys) -> None:
    module = _load_daily_digest_module()
    urls: list[str] = []
    fixture_html = HTML_FIXTURE.read_text(encoding="utf-8")
    monkeypatch.setattr(module, "fetch_page", lambda url: urls.append(url) or fixture_html)

    module.main(
        [
            "quant-ph",
            "2026-01-15",
            "--include-metadata",
            "--config",
            str(INTERESTS_FIXTURE),
        ]
    )

    assert urls == ["https://arxiv.org/catchup/quant-ph/2026-01-15?abs=True"]
    assert json.loads(capsys.readouterr().out)["date"] == "2026-01-15"


def test_date_option_drives_fetch_url_and_metadata(monkeypatch, capsys) -> None:
    module = _load_daily_digest_module()
    urls: list[str] = []
    fixture_html = HTML_FIXTURE.read_text(encoding="utf-8")
    monkeypatch.setattr(module, "fetch_page", lambda url: urls.append(url) or fixture_html)

    module.main(
        [
            "quant-ph",
            "--date",
            "2026-01-15",
            "--include-metadata",
            "--config",
            str(INTERESTS_FIXTURE),
        ]
    )

    assert urls == ["https://arxiv.org/catchup/quant-ph/2026-01-15?abs=True"]
    assert json.loads(capsys.readouterr().out)["date"] == "2026-01-15"


def test_default_date_uses_utc_calendar_date(monkeypatch, capsys) -> None:
    module = _load_daily_digest_module()
    monkeypatch.setattr(module, "datetime", _fixed_datetime(module, "2026-01-01T00:30:00"))
    fixture_html = HTML_FIXTURE.read_text(encoding="utf-8")
    monkeypatch.setattr(module, "fetch_page", lambda url: fixture_html)

    module.main(["--include-metadata", "--config", str(INTERESTS_FIXTURE)])

    assert json.loads(capsys.readouterr().out)["date"] == "2026-01-01"


def test_days_ago_uses_calendar_day_subtraction_across_weekend_and_year(monkeypatch, capsys) -> None:
    module = _load_daily_digest_module()
    fixture_html = HTML_FIXTURE.read_text(encoding="utf-8")
    monkeypatch.setattr(module, "fetch_page", lambda url: fixture_html)

    monkeypatch.setattr(module, "datetime", _fixed_datetime(module, "2026-01-05T12:00:00"))
    module.main(["--days-ago", "2", "--include-metadata", "--config", str(INTERESTS_FIXTURE)])
    assert json.loads(capsys.readouterr().out)["date"] == "2026-01-03"

    monkeypatch.setattr(module, "datetime", _fixed_datetime(module, "2026-01-01T12:00:00"))
    module.main(["--days-ago", "1", "--include-metadata", "--config", str(INTERESTS_FIXTURE)])
    assert json.loads(capsys.readouterr().out)["date"] == "2025-12-31"


def test_invalid_or_conflicting_date_controls_fail_before_fetch(monkeypatch, capsys) -> None:
    import pytest

    module = _load_daily_digest_module()
    fetch_called = False

    def fake_fetch(url):
        nonlocal fetch_called
        fetch_called = True
        return ""

    monkeypatch.setattr(module, "fetch_page", fake_fetch)
    cases = [
        ["quant-ph", "2026-02-30"],
        ["quant-ph", "2026-1-05"],
        ["quant-ph", "2026-01-15", "--date", "2026-01-16"],
        ["quant-ph", "--date", "2026-01-15", "--days-ago", "0"],
        ["quant-ph", "--date", "2026-01-15", "--days-ago", "1"],
        ["quant-ph", "--days-ago", "-1"],
        ["--from-html", str(HTML_FIXTURE), "--days-ago", "0"],
    ]

    for argv in cases:
        with pytest.raises(SystemExit) as exit_info:
            module.main(argv)
        assert exit_info.value.code == 2

    assert not fetch_called
    assert capsys.readouterr().err


def test_days_ago_overflow_fails_before_fetch_or_output(monkeypatch, tmp_path, capsys) -> None:
    import pytest

    module = _load_daily_digest_module()
    output_path = tmp_path / "digest.json"
    fetch_called = False

    def fake_fetch(url):
        nonlocal fetch_called
        fetch_called = True
        return ""

    monkeypatch.setattr(module, "fetch_page", fake_fetch)

    with pytest.raises(SystemExit) as exit_info:
        module.main(["--days-ago", "1000000000", "--output", str(output_path)])

    assert exit_info.value.code == 2
    assert not fetch_called
    assert not output_path.exists()
    assert "--days-ago is outside the supported date range" in capsys.readouterr().err


def test_days_ago_underflow_fails_before_fetch_or_output(monkeypatch, tmp_path, capsys) -> None:
    import pytest

    module = _load_daily_digest_module()
    output_path = tmp_path / "digest.json"
    fetch_called = False

    def fake_fetch(url):
        nonlocal fetch_called
        fetch_called = True
        return ""

    monkeypatch.setattr(module, "datetime", _fixed_datetime(module, "0001-01-01T00:00:00"))
    monkeypatch.setattr(module, "fetch_page", fake_fetch)

    with pytest.raises(SystemExit) as exit_info:
        module.main(["--days-ago", "1", "--output", str(output_path)])

    assert exit_info.value.code == 2
    assert not fetch_called
    assert not output_path.exists()
    assert "--days-ago is outside the supported date range" in capsys.readouterr().err


def test_fetch_papers_normalizes_legacy_positional_date_and_zero_days(monkeypatch) -> None:
    module = _load_daily_digest_module()
    urls: list[str] = []
    fixture_html = HTML_FIXTURE.read_text(encoding="utf-8")
    legacy_args = argparse.Namespace(
        category="quant-ph",
        date_positional="2026-01-15",
        days_ago=0,
        from_html=None,
        no_abstracts=False,
    )
    monkeypatch.setattr(module, "fetch_page", lambda url: urls.append(url) or fixture_html)

    papers, category = module.fetch_papers(legacy_args)

    assert category == "quant-ph"
    assert len(papers) == 6
    assert not hasattr(legacy_args, "date")
    assert not hasattr(legacy_args, "listing_date")
    assert urls == ["https://arxiv.org/catchup/quant-ph/2026-01-15?abs=True"]


def test_fetch_papers_rejects_legacy_nonzero_days_with_positional_date() -> None:
    import pytest

    module = _load_daily_digest_module()
    legacy_args = argparse.Namespace(
        category="quant-ph",
        date_positional="2026-01-15",
        days_ago=1,
        from_html=None,
        no_abstracts=False,
    )

    with pytest.raises(ValueError, match="--days-ago cannot be used with an explicit date"):
        module.fetch_papers(legacy_args)


def test_fetch_papers_rejects_invalid_legacy_positional_date() -> None:
    import pytest

    module = _load_daily_digest_module()
    legacy_args = argparse.Namespace(
        category="quant-ph",
        date_positional="2026-02-30",
        days_ago=0,
        from_html=None,
        no_abstracts=False,
    )

    with pytest.raises(ValueError, match="must be a valid YYYY-MM-DD date"):
        module.fetch_papers(legacy_args)
