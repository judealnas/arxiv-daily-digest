"""Tests for interests.py file I/O."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from arxiv_interest_sort.interests import (
    InterestTopic,
    read_interests_from_file,
    write_interests_to_file,
)

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_YAML = FIXTURES / "sample_interests.yaml"


def _make_sample_interests() -> list[InterestTopic]:
    return [
        InterestTopic(
            name="quantum networking",
            weight=5,
            keywords=["entanglement", "quantum network", "distributed quantum"],
            description="",
        ),
        InterestTopic(
            name="quantum simulation",
            weight=3,
            keywords=["Hamiltonian simulation", "quantum simulation"],
            description="",
        ),
    ]


# ── read_interests_from_file ─────────────────────────────────────
class TestReadInterestsFromFile:
    def test_read_yaml(self) -> None:
        interests = read_interests_from_file(SAMPLE_YAML)
        assert len(interests) == 2
        assert interests[0].name == "quantum networking"
        assert interests[0].weight == 5
        assert "entanglement" in interests[0].keywords

    def test_read_json(self, tmp_path: Path) -> None:
        data = [
            {
                "name": "test topic",
                "weight": 2,
                "keywords": ["kw1", "kw2"],
                "description": "desc",
            }
        ]
        json_file = tmp_path / "interests.json"
        json_file.write_text(json.dumps(data))

        interests = read_interests_from_file(json_file)
        assert len(interests) == 1
        assert interests[0].name == "test topic"
        assert interests[0].weight == 2
        assert interests[0].keywords == ["kw1", "kw2"]
        assert interests[0].description == "desc"

    def test_unsupported_format_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "interests.txt"
        bad_file.write_text("not yaml or json")

        with pytest.raises(ValueError, match="Unsupported file format"):
            read_interests_from_file(bad_file)

    def test_returns_interest_topic_instances(self) -> None:
        interests = read_interests_from_file(SAMPLE_YAML)
        assert all(isinstance(i, InterestTopic) for i in interests)


# ── write_interests_to_file ──────────────────────────────────────
class TestWriteInterestsToFile:
    def test_write_yaml(self, tmp_path: Path) -> None:
        interests = _make_sample_interests()
        out = tmp_path / "out.yaml"

        write_interests_to_file(interests, out)

        loaded = yaml.safe_load(out.read_text())
        assert isinstance(loaded, list)
        assert len(loaded) == 2
        assert loaded[0]["name"] == "quantum networking"
        assert loaded[1]["weight"] == 3

    def test_write_json(self, tmp_path: Path) -> None:
        interests = _make_sample_interests()
        out = tmp_path / "out.json"

        write_interests_to_file(interests, out)

        loaded = json.loads(out.read_text())
        assert len(loaded) == 2
        assert loaded[0]["name"] == "quantum networking"
        assert loaded[1]["keywords"] == ["Hamiltonian simulation", "quantum simulation"]

    def test_unsupported_format_raises(self, tmp_path: Path) -> None:
        interests = _make_sample_interests()
        bad_file = tmp_path / "out.txt"

        with pytest.raises(ValueError, match="Unsupported file format"):
            write_interests_to_file(interests, bad_file)

    def test_roundtrip_yaml(self, tmp_path: Path) -> None:
        """Writing then reading should preserve data."""
        interests = _make_sample_interests()
        out = tmp_path / "roundtrip.yaml"

        write_interests_to_file(interests, out)
        loaded = read_interests_from_file(out)

        assert len(loaded) == len(interests)
        for original, restored in zip(interests, loaded):
            assert original.name == restored.name
            assert original.weight == restored.weight
            assert original.keywords == restored.keywords
            assert original.description == restored.description

    def test_roundtrip_json(self, tmp_path: Path) -> None:
        """Writing then reading should preserve data."""
        interests = _make_sample_interests()
        out = tmp_path / "roundtrip.json"

        write_interests_to_file(interests, out)
        loaded = read_interests_from_file(out)

        assert len(loaded) == len(interests)
        for original, restored in zip(interests, loaded):
            assert original.name == restored.name
            assert original.weight == restored.weight
            assert original.keywords == restored.keywords
            assert original.description == restored.description

    def test_write_empty_list(self, tmp_path: Path) -> None:
        out = tmp_path / "empty.yaml"
        write_interests_to_file([], out)

        loaded = yaml.safe_load(out.read_text())
        assert loaded == []

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        interests = _make_sample_interests()
        out = str(tmp_path / "str_path.yaml")

        write_interests_to_file(interests, out)
        loaded = read_interests_from_file(out)

        assert len(loaded) == 2
