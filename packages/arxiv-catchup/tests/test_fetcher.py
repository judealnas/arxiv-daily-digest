"""Tests for the arxiv-catchup fetcher module."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pytest

from arxiv_catchup import fetch_page, extract_papers, get_papers, _strip_html

HERE = Path(__file__).parent
FIXTURES = HERE / "fixtures"

with open(FIXTURES / "arxiv-catchup-sample.html") as f:
    SAMPLE_HTML = f.read()

with open(FIXTURES / "arxiv-catchup-expected.json") as f:
    EXPECTED_PAPERS = json.load(f)


# ── _strip_html ────────────────────────────────────────────────────────────


class TestStripHtml:
    def test_strips_tags(self) -> None:
        assert _strip_html("<b>Hello</b> <i>world</i>") == "Hello world"

    def test_unescapes_entities(self) -> None:
        assert _strip_html("foo &amp; bar") == "foo & bar"

    def test_collapses_whitespace(self) -> None:
        assert _strip_html("  hello   world\n  test  ") == "hello world test"

    def test_empty_string(self) -> None:
        assert _strip_html("") == ""

    def test_no_html(self) -> None:
        assert _strip_html("plain text") == "plain text"


# ── extract_papers ─────────────────────────────────────────────────────────


class TestExtractPapers:
    def test_returns_expected_papers(self) -> None:
        """Parsing the sample HTML should produce the expected JSON output."""
        result = extract_papers(SAMPLE_HTML)
        assert result == EXPECTED_PAPERS

    def test_count(self) -> None:
        """The sample HTML has 6 paper entries."""
        result = extract_papers(SAMPLE_HTML)
        assert len(result) == 6

    def test_paper_fields(self) -> None:
        """Each paper should have all required keys with correct types."""
        result = extract_papers(SAMPLE_HTML)
        for paper in result:
            assert isinstance(paper["id"], str) and paper["id"]
            assert isinstance(paper["title"], str) and paper["title"]
            assert isinstance(paper["authors"], list)
            assert isinstance(paper["abstract"], str) and paper["abstract"]
            assert paper["abstract_link"].startswith("https://arxiv.org/abs/")
            assert paper["pdf_link"].startswith("https://arxiv.org/pdf/")
            assert isinstance(paper["comments"], str)
            assert isinstance(paper["subjects"], str)

    def test_versioned_id_clean_links(self) -> None:
        """A paper with version suffix v2 should have clean links without the suffix."""
        result = extract_papers(SAMPLE_HTML)
        paper = next(p for p in result if "v2" in p["id"])
        assert paper["id"] == "2501.12345v2"
        assert paper["abstract_link"] == "https://arxiv.org/abs/2501.12345"
        assert paper["pdf_link"] == "https://arxiv.org/pdf/2501.12345"

    def test_paper_without_comments(self) -> None:
        """Paper 2 (2601.03779) has no comments block -> empty string."""
        result = extract_papers(SAMPLE_HTML)
        paper = next(p for p in result if p["id"] == "2601.03779")
        assert paper["comments"] == ""

    def test_single_author(self) -> None:
        """Paper 3 (2601.03810) has a single author."""
        result = extract_papers(SAMPLE_HTML)
        paper = next(p for p in result if p["id"] == "2601.03810")
        assert paper["authors"] == ["Reyhaneh Khasseh"]

    def test_cross_listed_paper(self) -> None:
        """Cross-listed papers should be parsed with correct subjects."""
        result = extract_papers(SAMPLE_HTML)
        paper = next(p for p in result if p["id"] == "2601.03772")
        assert "Quantum Gases (cond-mat.quant-gas)" in paper["subjects"]

    def test_empty_html(self) -> None:
        assert extract_papers("") == []

    def test_html_with_no_paper_blocks(self) -> None:
        assert extract_papers("<html><body>No papers here</body></html>") == []

    def test_preserves_paper_order(self) -> None:
        """Papers should appear in the same order as in the HTML."""
        result = extract_papers(SAMPLE_HTML)
        expected_ids = [p["id"] for p in EXPECTED_PAPERS]
        assert [p["id"] for p in result] == expected_ids


# ── fetch_page ─────────────────────────────────────────────────────────────


class TestFetchPage:
    @mock.patch("urllib.request.urlopen")
    def test_success(self, mock_urlopen: mock.MagicMock) -> None:
        """A successful request returns the decoded response body."""
        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = b"hello world"
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        result = fetch_page("https://example.com")
        assert result == "hello world"

    @mock.patch("urllib.request.urlopen")
    def test_retry_on_429_then_succeeds(
        self, mock_urlopen: mock.MagicMock
    ) -> None:
        """Should retry on 429 and succeed on the second attempt."""
        from urllib.error import HTTPError

        mock_resp_ok = mock.MagicMock()
        mock_resp_ok.read.return_value = b"ok"

        mock_urlopen.side_effect = [
            HTTPError("url", 429, "Too Many Requests", {}, None),
            mock_resp_ok,
        ]
        mock_resp_ok.__enter__.return_value = mock_resp_ok

        with mock.patch("time.sleep"):
            result = fetch_page("https://example.com", max_retries=2)

        assert result == "ok"
        assert mock_urlopen.call_count == 2

    @mock.patch("urllib.request.urlopen")
    def test_exhaust_retries_raises_on_last_attempt(
        self, mock_urlopen: mock.MagicMock
    ) -> None:
        """After exhausting all retries on 429, the last error is re-raised."""
        from urllib.error import HTTPError

        error = HTTPError("url", 429, "Too Many Requests", {}, None)
        mock_urlopen.side_effect = [error, error, error]

        with (
            mock.patch("time.sleep"),
            pytest.raises(HTTPError),
        ):
            fetch_page("https://example.com", max_retries=3)

    @mock.patch("urllib.request.urlopen")
    def test_raises_on_non_429(self, mock_urlopen: mock.MagicMock) -> None:
        """Non-429 HTTP errors should be re-raised immediately."""
        from urllib.error import HTTPError

        error = HTTPError("url", 403, "Forbidden", {}, None)
        mock_urlopen.side_effect = error

        with pytest.raises(HTTPError):
            fetch_page("https://example.com")

    @mock.patch("urllib.request.urlopen")
    def test_headers_are_passed(self, mock_urlopen: mock.MagicMock) -> None:
        """Custom headers should be forwarded to the request."""
        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = b"data"
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        headers = {"User-Agent": "test-bot/1.0"}
        with mock.patch("urllib.request.Request") as mock_req:
            mock_req.return_value = mock.MagicMock()
            fetch_page("https://example.com", headers=headers)
            mock_req.assert_called_with(
                "https://example.com", headers=headers
            )


# ── get_papers ──────────────────────────────────────────────────────────────


class TestGetPapers:
    def test_accepts_date_object(self) -> None:
        """A date object should be accepted without error."""
        with mock.patch.object(
            __import__("arxiv_catchup"), "fetch_page", return_value=SAMPLE_HTML
        ):
            result = get_papers(date=date(2026, 1, 15))
        assert len(result) == 6

    def test_accepts_date_string(self) -> None:
        """A 'YYYY-MM-DD' string should be accepted."""
        with mock.patch.object(
            __import__("arxiv_catchup"), "fetch_page", return_value=SAMPLE_HTML
        ):
            result = get_papers(date="2026-01-15")
        assert len(result) == 6

    @mock.patch("arxiv_catchup.fetch_page", return_value=SAMPLE_HTML)
    def test_default_date_is_today(self, mock_fetch: mock.MagicMock) -> None:
        """When no date or yesterday is given, should use today's date."""
        today = datetime.now(timezone.utc).date().strftime("%Y-%m-%d")
        get_papers("quant-ph")
        expected_url = f"https://arxiv.org/catchup/quant-ph/{today}?abs=True"
        mock_fetch.assert_called_once_with(expected_url)

    @mock.patch("arxiv_catchup.fetch_page", return_value=SAMPLE_HTML)
    def test_days_ago(self, mock_fetch: mock.MagicMock) -> None:
        """days_ago=1 should use yesterday's date."""
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date()
        yesterday_str = yesterday.strftime("%Y-%m-%d")
        get_papers("quant-ph", days_ago=1)
        expected_url = (
            f"https://arxiv.org/catchup/quant-ph/{yesterday_str}?abs=True"
        )
        mock_fetch.assert_called_once_with(expected_url)

    @mock.patch("arxiv_catchup.fetch_page", return_value=SAMPLE_HTML)
    def test_days_ago_7(self, mock_fetch: mock.MagicMock) -> None:
        """days_ago=7 should fetch from a week ago."""
        week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).date()
        week_ago_str = week_ago.strftime("%Y-%m-%d")
        get_papers("quant-ph", days_ago=7)
        expected_url = (
            f"https://arxiv.org/catchup/quant-ph/{week_ago_str}?abs=True"
        )
        mock_fetch.assert_called_once_with(expected_url)

    @mock.patch("arxiv_catchup.fetch_page", return_value=SAMPLE_HTML)
    def test_without_abstracts(self, mock_fetch: mock.MagicMock) -> None:
        """with_abstracts=False should set ?abs=False in the URL."""
        today = datetime.now(timezone.utc).date().strftime("%Y-%m-%d")
        get_papers("cs.LG", with_abstracts=False)
        expected_url = f"https://arxiv.org/catchup/cs.LG/{today}?abs=False"
        mock_fetch.assert_called_once_with(expected_url)

    @mock.patch("arxiv_catchup.fetch_page", return_value=None)
    def test_fetch_failure_exits(self, mock_fetch: mock.MagicMock) -> None:
        """When fetch_page returns None, the function should sys.exit(1)."""
        with pytest.raises(SystemExit) as exc:
            get_papers()
        assert exc.value.code == 1

    @mock.patch("arxiv_catchup.fetch_page", return_value=SAMPLE_HTML)
    def test_category_default(self, mock_fetch: mock.MagicMock) -> None:
        """Default category should be 'quant-ph'."""
        today = datetime.now(timezone.utc).date().strftime("%Y-%m-%d")
        get_papers()
        expected_url = f"https://arxiv.org/catchup/quant-ph/{today}?abs=True"
        mock_fetch.assert_called_once_with(expected_url)
