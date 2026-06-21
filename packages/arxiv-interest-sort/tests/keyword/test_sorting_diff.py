"""Tests that produce a readable diff showing how the sorter orders input papers.

These tests use the ``filecmp``/``difflib`` approach: run ``score_and_sort``
on the fixture input, then compare the annotated, sorted output against an
expected snapshot.  When the test fails, pytest prints a unified diff so you
can see exactly how the sorting changed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arxiv_interest_sort import load_config, score_and_sort

# ── fixtures ─────────────────────────────────────────────────────
FIXTURES = Path(__file__).parent.parent / "fixtures"
SNAPSHOTS = Path(__file__).parent / "snapshots"


@pytest.fixture
def papers() -> list[dict]:
    with open(FIXTURES / "sample_input.json") as f:
        return json.load(f)


@pytest.fixture
def interests() -> dict:
    import yaml

    with open(FIXTURES / "sample_interests.yaml") as f:
        return yaml.safe_load(f)


def _sort_to_snapshot(papers: list[dict], topics: list[dict]) -> list[dict]:
    """Score, sort, and strip volatile fields for a stable snapshot."""
    result = score_and_sort(papers, topics, strategy="keyword")
    # Keep only fields that are stable across runs
    return [
        {
            "id": p["id"],
            "title": p["title"],
            "matched": p["matched"],
            "match_score": p["match_score"],
            "match_count": p["match_count"],
            "matched_topics": p["matched_topics"],
        }
        for p in result
    ]


class TestKeywordSortingDiff:
    def test_sorted_output_matches_snapshot(
        self, papers: list[dict], interests: dict
    ) -> None:
        """The keyword strategy produces the expected sort order and annotations."""
        snapshot_path = SNAPSHOTS / "keyword_sorted_expected.json"

        actual = _sort_to_snapshot(papers, interests["topics"])
        actual_json = json.dumps(actual, indent=2, sort_keys=True) + "\n"

        if not snapshot_path.exists():
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_path.write_text(actual_json)
            pytest.fail(
                f"Snapshot created at {snapshot_path}. "
                "Re-run the test to verify it passes."
            )

        expected = json.loads(snapshot_path.read_text())
        assert actual == expected, (
            f"Sorted output differs from snapshot.\n"
            f"Expected order: {[p['id'] for p in expected]}\n"
            f"Actual order:   {[p['id'] for p in actual]}"
        )

    def test_sorted_output_unified_diff(
        self, papers: list[dict], interests: dict, capsys: pytest.CaptureFixture
    ) -> None:
        """Print a unified diff on failure for human-readable output."""
        snapshot_path = SNAPSHOTS / "keyword_sorted_expected.json"

        actual = _sort_to_snapshot(papers, interests["topics"])
        actual_json = json.dumps(actual, indent=2, sort_keys=True) + "\n"

        if snapshot_path.exists():
            expected_json = snapshot_path.read_text()
            if actual_json != expected_json:
                import difflib

                diff = difflib.unified_diff(
                    expected_json.splitlines(keepends=True),
                    actual_json.splitlines(keepends=True),
                    fromfile="expected (snapshot)",
                    tofile="actual",
                )
                pytest.fail("Sorted output changed:\n" + "".join(diff))
        else:
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_path.write_text(actual_json)
            pytest.skip(f"Created initial snapshot at {snapshot_path}")

    def test_sort_order_matched_first(
        self, papers: list[dict], interests: dict
    ) -> None:
        """Matched papers appear before unmatched papers."""
        result = score_and_sort(papers, interests["topics"], strategy="keyword")

        matched_done = False
        for p in result:
            if not p["matched"]:
                matched_done = True
            elif matched_done:
                pytest.fail(
                    f"Matched paper {p['id']} appears after an unmatched paper. "
                    f"Order: {[p['id'] for p in result]}"
                )

    def test_sort_order_score_descending(
        self, papers: list[dict], interests: dict
    ) -> None:
        """Within the matched group, papers are sorted by score descending."""
        result = score_and_sort(papers, interests["topics"], strategy="keyword")

        matched = [p for p in result if p["matched"]]
        for i in range(len(matched) - 1):
            assert matched[i]["match_score"] >= matched[i + 1]["match_score"], (
                f"Score order violated: {matched[i]['id']} ({matched[i]['match_score']}) "
                f"before {matched[i + 1]['id']} ({matched[i + 1]['match_score']})"
            )
