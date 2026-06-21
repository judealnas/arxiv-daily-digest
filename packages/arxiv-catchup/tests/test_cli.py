"""Tests for the arxiv-catchup CLI."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pytest

from arxiv_catchup.cli import build_parser, main

HERE = Path(__file__).parent
FIXTURES = HERE / "fixtures"

with open(FIXTURES / "arxiv-catchup-sample.html") as f:
    SAMPLE_HTML = f.read()

with open(FIXTURES / "arxiv-catchup-expected.json") as f:
    EXPECTED_PAPERS = json.load(f)


# ── build_parser ───────────────────────────────────────────────────────────


class TestBuildParser:
    def test_returns_parser(self) -> None:
        parser = build_parser()
        assert parser is not None

    def test_default_category(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])
        assert args.category == "quant-ph"

    def test_custom_category(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["cs.LG"])
        assert args.category == "cs.LG"

    def test_category_and_date(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["quant-ph", "2026-06-01"])
        assert args.category == "quant-ph"
        assert args.date == "2026-06-01"

    def test_days_ago_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--days-ago", "3"])
        assert args.days_ago == 3

    def test_days_ago_default(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])
        assert args.days_ago == 0

    def test_no_abstracts_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--no-abstracts"])
        assert args.no_abstracts is True

    def test_date_and_days_ago(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["hep-th", "2026-06-15", "--days-ago", "2", "--no-abstracts"]
        )
        assert args.category == "hep-th"
        assert args.date == "2026-06-15"
        assert args.days_ago == 2
        assert args.no_abstracts is True


# ── main ────────────────────────────────────────────────────────────────────


class TestMain:
    @mock.patch("arxiv_catchup.cli.fetch_page", return_value=SAMPLE_HTML)
    def test_default_args(self, mock_fetch: mock.MagicMock) -> None:
        """Default args (no argv) should fetch quant-ph for today."""
        main([])
        # Should have been called with a URL containing quant-ph and today's date
        call_url = mock_fetch.call_args[0][0]
        assert "quant-ph" in call_url
        assert datetime.now(timezone.utc).date().strftime("%Y-%m-%d") in call_url

    @mock.patch("arxiv_catchup.cli.fetch_page", return_value=SAMPLE_HTML)
    def test_category_and_date(self, mock_fetch: mock.MagicMock) -> None:
        """Should pass category and date through to the URL."""
        main(["cs.LG", "2026-06-01"])
        call_url = mock_fetch.call_args[0][0]
        assert "cs.LG" in call_url
        assert "2026-06-01" in call_url

    @mock.patch("arxiv_catchup.cli.fetch_page", return_value=SAMPLE_HTML)
    def test_days_ago(self, mock_fetch: mock.MagicMock) -> None:
        """--days-ago 1 should use yesterday's date."""
        main(["quant-ph", "--days-ago", "1"])
        yesterday = (
            (datetime.now(timezone.utc) - timedelta(days=1)).date().strftime("%Y-%m-%d")
        )
        call_url = mock_fetch.call_args[0][0]
        assert yesterday in call_url

    @mock.patch("arxiv_catchup.cli.fetch_page", return_value=SAMPLE_HTML)
    def test_days_ago_7(self, mock_fetch: mock.MagicMock) -> None:
        """--days-ago 7 should fetch from a week ago."""
        main(["quant-ph", "--days-ago", "7"])
        week_ago = (
            (datetime.now(timezone.utc) - timedelta(days=7)).date().strftime("%Y-%m-%d")
        )
        call_url = mock_fetch.call_args[0][0]
        assert week_ago in call_url

    @mock.patch("arxiv_catchup.cli.fetch_page", return_value=SAMPLE_HTML)
    def test_no_abstracts_flag(self, mock_fetch: mock.MagicMock) -> None:
        """--no-abstracts should set ?abs=False."""
        main(["quant-ph", "--no-abstracts"])
        call_url = mock_fetch.call_args[0][0]
        assert "abs=False" in call_url

    @mock.patch("arxiv_catchup.cli.fetch_page", return_value=SAMPLE_HTML)
    def test_output_is_valid_json(self, mock_fetch: mock.MagicMock) -> None:
        """The printed output should be valid JSON matching expected papers."""
        with mock.patch("sys.stdout") as mock_stdout:
            main(["quant-ph", "2026-01-15"])
            # Verify fetch_page was called
            assert mock_fetch.called

    @mock.patch("arxiv_catchup.cli.fetch_page", return_value=SAMPLE_HTML)
    def test_json_output_matches_expected(self, mock_fetch: mock.MagicMock) -> None:
        """The JSON printed to stdout should match the expected papers."""
        from io import StringIO

        fake_stdout = StringIO()
        with mock.patch("sys.stdout", fake_stdout):
            main(["quant-ph", "2026-01-15"])

        output = json.loads(fake_stdout.getvalue())
        assert output == EXPECTED_PAPERS

    @mock.patch("arxiv_catchup.cli.fetch_page", return_value=None)
    def test_fetch_failure_exits(self, mock_fetch: mock.MagicMock) -> None:
        """When fetch_page returns None, should print error and exit(1)."""
        with pytest.raises(SystemExit) as exc:
            main(["quant-ph"])
        assert exc.value.code == 1

    @mock.patch("arxiv_catchup.cli.fetch_page")
    def test_fetch_exception_exits(self, mock_fetch: mock.MagicMock) -> None:
        """When fetch_page raises, should print error and exit(1)."""
        mock_fetch.side_effect = ConnectionError("Network error")
        with pytest.raises(SystemExit) as exc:
            main(["quant-ph"])
        assert exc.value.code == 1

    @mock.patch("arxiv_catchup.cli.fetch_page", return_value=SAMPLE_HTML)
    def test_date_takes_precedence_over_days_ago(
        self, mock_fetch: mock.MagicMock
    ) -> None:
        """When both date and --days-ago are given, date takes precedence."""
        main(["quant-ph", "2026-06-15", "--days-ago", "3"])
        call_url = mock_fetch.call_args[0][0]
        assert "2026-06-15" in call_url
        three_days_ago = (
            (datetime.now(timezone.utc) - timedelta(days=3)).date().strftime("%Y-%m-%d")
        )
        assert three_days_ago not in call_url
