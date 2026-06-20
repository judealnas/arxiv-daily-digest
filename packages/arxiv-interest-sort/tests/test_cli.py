"""Tests for the CLI entry point."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arxiv_interest_sort.cli import build_parser, main

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_INPUT = str(FIXTURES / "sample_input.json")
SAMPLE_CONFIG = str(FIXTURES / "sample_interests.yaml")


# ── parser ───────────────────────────────────────────────────────
class TestBuildParser:
    def test_defaults(self) -> None:
        """Parser provides sensible defaults."""
        parser = build_parser()
        args = parser.parse_args([])
        assert args.input_file is None
        assert args.strategy == "keyword"
        assert args.config == "arxiv-interests.yaml"
        assert args.include_metadata is False
        assert args.model is None
        assert args.threshold is None
        assert args.cache_dir is None


# ── main ─────────────────────────────────────────────────────────
class TestMain:
    def test_with_file(self, capsys: pytest.CaptureFixture) -> None:
        """Reads from a file and outputs sorted JSON array."""
        main([SAMPLE_INPUT, "--config", SAMPLE_CONFIG])
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert isinstance(output, list)
        assert len(output) == 4
        # Papers should be sorted: matched first
        for p in output:
            assert "matched" in p
            assert "match_score" in p

    def test_include_metadata(self, capsys: pytest.CaptureFixture) -> None:
        """--include-metadata wraps output in an envelope."""
        main(
            [
                SAMPLE_INPUT,
                "--config",
                SAMPLE_CONFIG,
                "--include-metadata",
            ]
        )
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert isinstance(output, dict)
        assert "date" in output
        assert "count" in output
        assert "matched_count" in output
        assert "papers" in output
        assert output["count"] == 4
        assert len(output["papers"]) == 4

    def test_strategy_override(self, capsys: pytest.CaptureFixture) -> None:
        """Explicit --strategy flag selects the correct strategy."""
        main(
            [
                SAMPLE_INPUT,
                "--config",
                SAMPLE_CONFIG,
                "--strategy",
                "keyword",
            ]
        )
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert isinstance(output, list)

    def test_invalid_strategy(self) -> None:
        """Unknown strategy prints an error and exits."""
        with pytest.raises(SystemExit):
            main(
                [
                    SAMPLE_INPUT,
                    "--config",
                    SAMPLE_CONFIG,
                    "--strategy",
                    "nonexistent",
                ]
            )

    def test_missing_config(self) -> None:
        """Missing config file prints an error and exits."""
        with pytest.raises(SystemExit):
            main([SAMPLE_INPUT, "--config", "/nonexistent/path.yaml"])

    def test_missing_input_file(self) -> None:
        """Missing input file prints an error and exits."""
        with pytest.raises(SystemExit):
            main(["/nonexistent/input.json", "--config", SAMPLE_CONFIG])

    def test_invalid_input_file(self) -> None:
        """Invalid JSON in input file prints an error and exits."""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json")

        try:
            with pytest.raises(SystemExit):
                main([f.name, "--config", SAMPLE_CONFIG])
        finally:
            Path(f.name).unlink()

    def test_non_list_input(self) -> None:
        """Non-array input prints an error and exits."""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"not": "a list"}, f)

        try:
            with pytest.raises(SystemExit):
                main([f.name, "--config", SAMPLE_CONFIG])
        finally:
            Path(f.name).unlink()

    def test_sorting_order(self, capsys: pytest.CaptureFixture) -> None:
        """Matched papers appear before non-matched papers."""
        main([SAMPLE_INPUT, "--config", SAMPLE_CONFIG])
        captured = capsys.readouterr()
        output = json.loads(captured.out)

        found_non_matched = False
        for p in output:
            if not p["matched"]:
                found_non_matched = True
            else:
                # Once we see a non-matched, we shouldn't see matched again
                assert not found_non_matched, "Matched paper appears after non-matched"
