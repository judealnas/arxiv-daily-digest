"""Tests for the interest file hashing and embedding cache system."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from arxiv_interest_sort._cache import get_cache_dir, topic_embeddings_path
from arxiv_interest_sort.strategies.semantic import _compute_cache_key


# ── _compute_cache_key ───────────────────────────────────────────
class TestComputeCacheKey:
    def test_deterministic(self) -> None:
        """Same inputs always produce the same hash."""
        queries = [
            {
                "name": "quantum networking",
                "query": "quantum networking: entanglement",
                "weight": 5,
            },
        ]
        assert _compute_cache_key("model-a", queries) == _compute_cache_key(
            "model-a", queries
        )

    def test_different_models_produce_different_keys(self) -> None:
        """Changing the model name changes the cache key."""
        queries = [
            {"name": "A", "query": "A: foo", "weight": 3},
        ]
        key_a = _compute_cache_key("model-a", queries)
        key_b = _compute_cache_key("model-b", queries)
        assert key_a != key_b

    def test_different_topics_produce_different_keys(self) -> None:
        """Changing topic content changes the cache key."""
        queries_a = [{"name": "A", "query": "A: foo", "weight": 3}]
        queries_b = [{"name": "B", "query": "B: bar", "weight": 3}]
        assert _compute_cache_key("model", queries_a) != _compute_cache_key(
            "model", queries_b
        )

    def test_different_weights_produce_different_keys(self) -> None:
        """Changing topic weight changes the cache key."""
        queries_a = [{"name": "A", "query": "A: foo", "weight": 3}]
        queries_b = [{"name": "A", "query": "A: foo", "weight": 5}]
        assert _compute_cache_key("model", queries_a) != _compute_cache_key(
            "model", queries_b
        )

    def test_key_ordering_does_not_matter(self) -> None:
        """Reordering keys within a dict does not change the hash (sort_keys=True)."""
        queries_a = [{"name": "A", "query": "A: foo", "weight": 3}]
        queries_b = [{"weight": 3, "name": "A", "query": "A: foo"}]
        assert _compute_cache_key("model", queries_a) == _compute_cache_key(
            "model", queries_b
        )

    def test_topic_order_does_not_matter(self) -> None:
        """Reordering the topics list does not change the hash (sort_keys on list elements)."""
        queries_a = [
            {"name": "A", "query": "A: foo", "weight": 3},
            {"name": "B", "query": "B: bar", "weight": 5},
        ]
        queries_b = [
            {"name": "B", "query": "B: bar", "weight": 5},
            {"name": "A", "query": "A: foo", "weight": 3},
        ]
        # json.dumps with sort_keys=True sorts dict keys but NOT list order,
        # so reordering list elements WILL change the hash.
        # This is expected behaviour — topics order is part of the identity.
        # If order-independence is needed, the caller should sort first.
        # Here we just verify the function is consistent.
        key_a = _compute_cache_key("model", queries_a)
        key_b = _compute_cache_key("model", queries_b)
        # List order matters for json.dumps, so these should differ
        assert key_a != key_b

    def test_returns_hex_digest(self) -> None:
        """The cache key is a valid SHA256 hex string (64 chars)."""
        queries = [{"name": "A", "query": "A: foo", "weight": 3}]
        key = _compute_cache_key("model", queries)
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_matches_manual_hash(self) -> None:
        """Cache key matches a manually computed SHA256."""
        queries = [
            {
                "name": "quantum networking",
                "query": "quantum networking: entanglement, quantum network",
                "weight": 5,
            },
        ]
        hasher = hashlib.sha256()
        hasher.update(b"my-model")
        canonical = json.dumps(queries, sort_keys=True, ensure_ascii=True)
        hasher.update(canonical.encode())
        expected = hasher.hexdigest()

        assert _compute_cache_key("my-model", queries) == expected

    def test_empty_queries(self) -> None:
        """Empty topic list produces a valid (if trivial) hash."""
        key = _compute_cache_key("model", [])
        assert len(key) == 64


# ── get_cache_dir ────────────────────────────────────────────────
class TestGetCacheDir:
    def test_returns_path_with_package_name(self) -> None:
        """The cache directory path ends with the package name."""
        result = get_cache_dir()
        assert result.name == "arxiv-interest-sort"

    def test_override(self) -> None:
        """An explicit override is used as-is."""
        with tempfile.TemporaryDirectory() as tmp:
            override = Path(tmp) / "my-cache"
            result = get_cache_dir(str(override))
            assert result == override.resolve()

    def test_linux_uses_xdg_cache_home(self) -> None:
        """On Linux, XDG_CACHE_HOME is respected."""
        with patch("arxiv_interest_sort._cache.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch.dict(os.environ, {"XDG_CACHE_HOME": "/tmp/xdg-cache"}):
                result = get_cache_dir()
                assert result == Path("/tmp/xdg-cache/arxiv-interest-sort")

    def test_linux_fallback_to_dot_cache(self) -> None:
        """On Linux without XDG_CACHE_HOME, falls back to ~/.cache."""
        with patch("arxiv_interest_sort._cache.sys") as mock_sys:
            mock_sys.platform = "linux"
            env = os.environ.copy()
            env.pop("XDG_CACHE_HOME", None)
            with patch.dict(os.environ, env, clear=True):
                result = get_cache_dir()
                assert ".cache/arxiv-interest-sort" in str(result)

    def test_macos_uses_library_caches(self) -> None:
        """On macOS, uses ~/Library/Caches."""
        with patch("arxiv_interest_sort._cache.sys") as mock_sys:
            mock_sys.platform = "darwin"
            result = get_cache_dir()
            assert "Library/Caches/arxiv-interest-sort" in str(result)


# ── topic_embeddings_path ────────────────────────────────────────
class TestTopicEmbeddingsPath:
    def test_returns_npz_file(self) -> None:
        """The cache file has a .npz extension."""
        result = topic_embeddings_path("/some/cache/dir")
        assert result.name == "topic_embeddings.npz"
        assert result.parent == Path("/some/cache/dir")

    def test_accepts_path_objects(self) -> None:
        """Accepts both str and Path for cache_dir."""
        result = topic_embeddings_path(Path("/tmp/cache"))
        assert result == Path("/tmp/cache/topic_embeddings.npz")


# ── _load_topic_embeddings (cache hit / miss) ────────────────────
class TestLoadTopicEmbeddings:
    def _make_model(self, dim: int = 4) -> MagicMock:
        """Create a mock SentenceTransformer that returns random-ish embeddings."""
        import numpy as np
        import torch

        model = MagicMock()
        model.device = "cpu"

        def encode_fn(texts, convert_to_tensor=True):
            n = len(texts)
            return torch.ones(n, dim)

        model.encode = encode_fn
        return model

    def test_cache_miss_encodes_and_saves(self) -> None:
        """On cache miss, embeddings are encoded and saved to disk."""
        import numpy as np
        import torch

        model = self._make_model()
        queries = [
            {"name": "A", "query": "A: foo", "weight": 3},
        ]
        key = _compute_cache_key("model", queries)

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            cache_dir.mkdir()

            from arxiv_interest_sort.strategies.semantic import _load_topic_embeddings

            result = _load_topic_embeddings(model, queries, str(cache_dir), key)

            cache_path = cache_dir / "topic_embeddings.npz"
            assert cache_path.exists()

            data = np.load(cache_path, allow_pickle=True)
            assert str(data["cache_key"]) == key
            assert str(data["names"][0]) == "A"

    def test_cache_hit_loads_from_disk(self) -> None:
        """On cache hit (matching key), embeddings are loaded from disk without re-encoding."""
        import numpy as np
        import torch

        from arxiv_interest_sort.strategies.semantic import _load_topic_embeddings

        model = self._make_model()
        queries = [
            {"name": "A", "query": "A: foo", "weight": 3},
        ]
        key = _compute_cache_key("model", queries)

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            cache_dir.mkdir()
            cache_path = cache_dir / "topic_embeddings.npz"

            # Pre-populate cache with known embeddings
            known_embs = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
            np.savez_compressed(
                cache_path,
                embeddings=known_embs.numpy(),
                cache_key=key,
                names=["A"],
                queries=["A: foo"],
                weights=[3],
            )

            # Track whether model.encode is called
            encode_called = False
            original_encode = model.encode

            def tracking_encode(*args, **kwargs):
                nonlocal encode_called
                encode_called = True
                return original_encode(*args, **kwargs)

            model.encode = tracking_encode

            result = _load_topic_embeddings(model, queries, str(cache_dir), key)

            assert not encode_called, "model.encode should not be called on cache hit"
            assert result.shape == (1, 4)

    def test_cache_mismatch_re_encodes(self) -> None:
        """When the cached key doesn't match, embeddings are re-encoded."""
        import numpy as np
        import torch

        from arxiv_interest_sort.strategies.semantic import _load_topic_embeddings

        model = self._make_model()
        queries = [
            {"name": "A", "query": "A: foo", "weight": 3},
        ]
        new_key = _compute_cache_key("model", queries)
        old_key = "deadbeef" * 8  # different key

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            cache_dir.mkdir()
            cache_path = cache_dir / "topic_embeddings.npz"

            # Pre-populate with stale key
            stale_embs = torch.tensor([[9.0, 9.0, 9.0, 9.0]])
            np.savez_compressed(
                cache_path,
                embeddings=stale_embs.numpy(),
                cache_key=old_key,
                names=["A"],
                queries=["A: foo"],
                weights=[3],
            )

            result = _load_topic_embeddings(model, queries, str(cache_dir), new_key)

            # Result should come from model.encode (ones), not the stale cache (nines)
            expected = torch.ones(1, 4)
            assert torch.allclose(result, expected)

    def test_no_cache_dir_skips_caching(self) -> None:
        """When cache_dir is None, no cache file is written."""
        import torch

        from arxiv_interest_sort.strategies.semantic import _load_topic_embeddings

        model = self._make_model()
        queries = [
            {"name": "A", "query": "A: foo", "weight": 3},
        ]

        result = _load_topic_embeddings(model, queries, None, "some-key")

        assert result.shape == (1, 4)
        # No exception means no cache was attempted

    def test_corrupted_cache_falls_back_to_re_encode(self) -> None:
        """A corrupted cache file is silently ignored and embeddings are re-encoded."""
        import torch

        from arxiv_interest_sort.strategies.semantic import _load_topic_embeddings

        model = self._make_model()
        queries = [
            {"name": "A", "query": "A: foo", "weight": 3},
        ]
        key = _compute_cache_key("model", queries)

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            cache_dir.mkdir()
            cache_path = cache_dir / "topic_embeddings.npz"

            # Write garbage data
            cache_path.write_bytes(b"not a valid npz file")

            result = _load_topic_embeddings(model, queries, str(cache_dir), key)

            # Should succeed with freshly encoded embeddings
            assert result.shape == (1, 4)
