"""Semantic scoring strategy.

Uses sentence-transformers to embed papers and interest topics, then
computes weighted cosine similarity to find the best-matching topic
for each paper.
"""

from __future__ import annotations

import hashlib
import json
import sys
from typing import Any

# ── type aliases ─────────────────────────────────────────────────
PaperDict = dict[str, Any]
TopicDict = dict[str, Any]


def semantic_score(
    papers: list[PaperDict],
    topics: list[TopicDict],
    **kwargs: Any,
) -> list[PaperDict]:
    """Score papers by semantic similarity using sentence-transformers.

    Empty paper lists are returned unchanged without importing model dependencies.
    Nonempty input requires the ``sentence-transformers`` package (install with
    ``pip install arxiv-interest-sort[semantic]``).

    Parameters
    ----------
    papers, topics:
        Standard inputs.
    **kwargs:
        Strategy-specific options passed from the CLI:

        * ``model`` — model name (default ``"allenai/specter2_base"``)
        * ``threshold`` — weighted-score threshold (default ``0.79``)
        * ``cache_dir`` — cache directory override

    Annotations added
    -----------------
    * ``matched`` — ``True`` if weighted cosine ≥ threshold
    * ``match_score`` — weighted cosine similarity (0–1)
    * ``match_count`` — ``int(round(match_score * 100))`` for backward compat
    * ``semantic_topic`` — name of the best-matching topic
    * ``raw_cosim`` — unweighted cosine similarity (for debugging)
    """
    if not papers:
        return papers

    try:
        import torch
        from sentence_transformers import SentenceTransformer, util
    except ImportError:
        print(
            "The 'semantic' strategy requires sentence-transformers.\n"
            "Install it with:  pip install arxiv-interest-sort[semantic]",
            file=sys.stderr,
        )
        sys.exit(1)

    model_name: str = kwargs.get("model", "allenai/specter2_base")
    threshold: float = kwargs.get("threshold", 0.79)

    # ── load model ────────────────────────────────────────────────
    print(f"Loading model '{model_name}'...", file=sys.stderr)
    model: SentenceTransformer = SentenceTransformer(model_name)

    # ── build topic queries ───────────────────────────────────────
    topic_queries: list[dict[str, Any]] = []
    for t in topics:
        q: str = t["name"] + ": " + ", ".join(t["keywords"])
        topic_queries.append(
            {"name": t["name"], "query": q, "weight": t.get("weight", 3)}
        )

    # ── encode topics (with caching) ──────────────────────────────
    cache_key: str = _compute_cache_key(model_name, topic_queries)
    topic_embs = _load_topic_embeddings(
        model, topic_queries, kwargs.get("cache_dir"), cache_key
    )

    # ── encode papers ─────────────────────────────────────────────
    texts: list[str] = [
        p["title"] + ". " + p.get("abstract", "")[:1000] for p in papers
    ]
    print(f"Encoding {len(papers)} papers...", file=sys.stderr)
    paper_embs: torch.Tensor = model.encode(texts, convert_to_tensor=True)

    # ── score ─────────────────────────────────────────────────────
    max_weight: float = float(max(t["weight"] for t in topic_queries))
    cos: torch.Tensor = util.cos_sim(paper_embs, topic_embs)  # (N × T)

    weight_vec: torch.Tensor = torch.tensor(
        [t["weight"] / max_weight for t in topic_queries],
        device=cos.device,
    )
    weighted: torch.Tensor = cos * weight_vec  # (N × T)

    best_weighted, best_indices = weighted.max(dim=1)

    for i, p in enumerate(papers):
        ws: float = float(best_weighted[i])
        topic_idx: int = int(best_indices[i])
        p["match_score"] = round(ws, 4)
        p["match_count"] = int(round(ws * 100))
        p["matched"] = ws >= threshold
        p["semantic_topic"] = topic_queries[topic_idx]["name"]
        p["raw_cosim"] = round(float(cos[i][topic_idx]), 4)

    return papers


# ── internal helpers ─────────────────────────────────────────────
def _compute_cache_key(model_name: str, topic_queries: list[dict[str, Any]]) -> str:
    """Return a SHA256 hash of the model name + canonical topic data.

    The topic queries are serialised with ``sort_keys=True`` so that
    cosmetic YAML changes (whitespace, comments, key ordering) do
    **not** invalidate the cache — only meaningful semantic changes
    (topic names, keywords, weights) produce a different hash.
    """
    hasher = hashlib.sha256()
    hasher.update(model_name.encode())
    canonical = json.dumps(topic_queries, sort_keys=True, ensure_ascii=True)
    hasher.update(canonical.encode())
    return hasher.hexdigest()


def _load_topic_embeddings(
    model: Any,
    topic_queries: list[dict[str, Any]],
    cache_dir: str | None,
    cache_key: str | None = None,
) -> Any:
    """Load topic embeddings from cache or compute them.

    Cache validity is determined by comparing a SHA256 hash
    (``cache_key``) of the model name + canonical topic data.
    If the key matches, cached embeddings are reused.
    """
    import numpy as np
    import torch

    # Try loading from cache
    if cache_dir and cache_key:
        from arxiv_interest_sort._cache import topic_embeddings_path

        cache_path = topic_embeddings_path(cache_dir)
        if cache_path.exists():
            try:
                data = np.load(cache_path, allow_pickle=True)
                stored_key: str = str(data.get("cache_key", ""))
                if stored_key == cache_key:
                    emb = torch.from_numpy(data["embeddings"]).to(model.device)
                    print(
                        f"  [cached topic embeddings from {cache_path.name}]",
                        file=sys.stderr,
                    )
                    return emb
                # Cache key mismatch → fall through to re-encode
            except Exception:
                pass  # fall through to re-encode

    print(f"  Encoding {len(topic_queries)} interest topics...", file=sys.stderr)
    embs = model.encode([t["query"] for t in topic_queries], convert_to_tensor=True)

    # Cache
    if cache_dir and cache_key:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            embeddings=embs.cpu().numpy(),
            cache_key=cache_key,
            names=[t["name"] for t in topic_queries],
            queries=[t["query"] for t in topic_queries],
            weights=[t["weight"] for t in topic_queries],
        )

    return embs
