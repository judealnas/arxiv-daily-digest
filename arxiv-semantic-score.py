#!/usr/bin/env python3
"""Semantic interest scoring for arXiv papers using sentence-transformers.

Replaces the keyword-occurrence approach in arxiv-fetch-and-score.py with
cosine-similarity scoring against topic embeddings. Outputs the same JSON
shape so the downstream pipeline (summarize + email) works unchanged.

Usage:
    uv run python arxiv-semantic-score.py
    uv run python arxiv-semantic-score.py --model BAAI/bge-small-en-v1.5
    uv run python arxiv-semantic-score.py --threshold 0.35
    uv run python arxiv-semantic-score.py --compare   # show keyword vs semantic side-by-side
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from typing import Any

import numpy as np
import torch
import yaml
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.util import cos_sim

# ── paths ────────────────────────────────────────────────────────
HERE: str = os.path.dirname(os.path.abspath(__file__))
CATCHUP_SCRIPT: str = os.path.join(HERE, "arxiv-catchup.py")
INTERESTS_FILE: str = os.path.join(HERE, "arxiv-interests.yaml")
TOPIC_CACHE: str = os.path.join(HERE, ".topic_embeddings.npz")

DEFAULT_MODEL: str = "allenai/specter2_base"
DEFAULT_THRESHOLD: float = 0.79


# ── type aliases ─────────────────────────────────────────────────
TopicDict = dict[str, Any]         # {"name": str, "keywords": list[str], "weight": int}
TopicQuery = dict[str, Any]        # {"name": str, "query": str, "weight": float}
PaperDict = dict[str, Any]         # arXiv paper with id, title, abstract, …
ConfigDict = dict[str, Any]        # full YAML contents


# ── loading ──────────────────────────────────────────────────────
def fetch_papers() -> list[PaperDict]:
    """Run the catchup script and return the paper list."""
    result = subprocess.run(
        [sys.executable, CATCHUP_SCRIPT, "quant-ph", "--yesterday"],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        print(json.dumps({"error": f"Fetch failed: {result.stderr.strip()}"}), file=sys.stderr)
        sys.exit(1)
    try:
        data: Any = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"JSON parse error: {e}"}), file=sys.stderr)
        sys.exit(1)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "papers" in data:
        return data["papers"]
    print(json.dumps({"error": "Unexpected JSON shape from catchup script"}), file=sys.stderr)
    sys.exit(1)


def load_interests() -> ConfigDict:
    """Return full config dict from arxiv-interests.yaml."""
    with open(INTERESTS_FILE) as f:
        cfg: ConfigDict = yaml.safe_load(f)
    if "topics" not in cfg:
        cfg["topics"] = []
    return cfg


def build_topic_queries(topics: list[TopicDict]) -> list[TopicQuery]:
    """Build a search-query string for each interest topic."""
    queries: list[TopicQuery] = []
    for t in topics:
        q: str = t["name"] + ": " + ", ".join(t["keywords"])
        queries.append({
            "name": t["name"],
            "query": q,
            "weight": t.get("weight", 3),
        })
    return queries


# ── embedding ────────────────────────────────────────────────────
def encode_topics(
    model: SentenceTransformer,
    topic_queries: list[TopicQuery],
    cache_path: str | None,
) -> tuple[torch.Tensor, list[TopicQuery]]:
    """Return (embeddings_tensor, list_of_topic_dicts).

    Caches to .npz so subsequent runs skip re-encoding topics.
    Result tensor is placed on the same device as *model*.
    """
    device: torch.device = model.device
    # Check cache freshness against interests file mtime
    if cache_path and os.path.exists(cache_path):
        cache_mtime: float = os.path.getmtime(cache_path)
        interests_mtime: float = os.path.getmtime(INTERESTS_FILE)
        if cache_mtime > interests_mtime:
            try:
                data: Any = np.load(cache_path, allow_pickle=True)
                stored: list[TopicQuery] = [
                    {"name": n, "query": q, "weight": float(w)}
                    for n, q, w in zip(data["names"], data["queries"], data["weights"])
                ]
                if stored == topic_queries:
                    emb: torch.Tensor = torch.from_numpy(data["embeddings"]).to(device)
                    print(f"  [cached topic embeddings from {os.path.basename(cache_path)}]",
                          file=sys.stderr)
                    return emb, topic_queries
            except Exception:
                pass  # fall through to re-encode

    print(f"  Encoding {len(topic_queries)} interest topics...", file=sys.stderr)
    embs: torch.Tensor = model.encode([t["query"] for t in topic_queries], convert_to_tensor=True)

    # Cache — store weight too
    if cache_path:
        np.savez_compressed(
            cache_path,
            embeddings=embs.cpu().numpy(),
            names=[t["name"] for t in topic_queries],
            queries=[t["query"] for t in topic_queries],
            weights=[t["weight"] for t in topic_queries],
        )
    return embs, topic_queries


def encode_papers(model: SentenceTransformer, papers: list[PaperDict]) -> torch.Tensor:
    """Encode title+abstract for each paper, return tensor."""
    texts: list[str] = [
        p["title"] + ". " + p.get("abstract", "")[:1000]
        for p in papers
    ]
    return model.encode(texts, convert_to_tensor=True)


# ── scoring ──────────────────────────────────────────────────────
def score_papers(
    papers: list[PaperDict],
    paper_embs: torch.Tensor,
    topic_embs: torch.Tensor,
    topic_queries: list[TopicQuery],
    threshold: float,
) -> list[PaperDict]:
    """Annotate papers with match_score, match_count, matched, semantic_topic.

    Scoring incorporates topic weight:
      weighted = cos_sim * topic.weight / max_weight
      match_score = max(weighted) across topics
    """
    max_weight: float = float(max(t["weight"] for t in topic_queries))
    cos: torch.Tensor = util.cos_sim(paper_embs, topic_embs)          # (N × T)

    # Weighted scores
    weight_vec: torch.Tensor = torch.tensor(
        [t["weight"] / max_weight for t in topic_queries],
        device=cos.device,
    )
    weighted: torch.Tensor = cos * weight_vec                          # (N × T)

    best_weighted: torch.Tensor
    best_indices: torch.Tensor
    best_weighted, best_indices = weighted.max(dim=1)

    for i, p in enumerate(papers):
        ws: float = float(best_weighted[i])
        topic_idx: int = int(best_indices[i])
        p["match_score"] = round(ws, 4)
        p["match_count"] = int(round(ws * 100))          # backward compat
        p["matched"] = ws >= threshold
        p["semantic_topic"] = topic_queries[topic_idx]["name"]
        p["raw_cosim"] = round(float(cos[i][topic_idx]), 4)  # unweighted, for debugging

    return papers


def sort_papers(papers: list[PaperDict]) -> list[PaperDict]:
    """Matched first (by match_score desc, then id asc), then non-matched (id asc)."""
    def sort_key(p: PaperDict) -> tuple[int, float, str]:
        arxiv_id: str = p.get("id", "")
        arxiv_id_clean: str = re.sub(r'v\d+$', '', arxiv_id)
        return (
            0 if p["matched"] else 1,
            -p["match_score"],
            arxiv_id_clean,
        )
    return sorted(papers, key=sort_key)


# ── compare mode ─────────────────────────────────────────────────
def keyword_score(papers: list[PaperDict], topics: list[TopicDict]) -> list[PaperDict]:
    """Apply keyword-based scoring for comparison (includes weight)."""
    for p in papers:
        text: str = (p.get("title", "") + " " + p.get("abstract", "")).lower()
        total: int = 0
        for topic in topics:
            w: int = topic.get("weight", 3)
            for kw in topic["keywords"]:
                if kw.lower() in text:
                    total += w
        p["kw_matched"] = total > 0
        p["kw_score"] = total
    return papers


# ── main ─────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic arXiv interest scoring")
    parser.add_argument("--model",
                        help="sentence-transformers model name (overrides arxiv-interests.yaml)")
    parser.add_argument("--threshold", type=float,
                        help="Weighted-score threshold for 'matched' (overrides arxiv-interests.yaml)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Skip topic embedding cache")
    parser.add_argument("--compare", action="store_true",
                        help="Show keyword vs semantic comparison (no JSON output)")
    args: argparse.Namespace = parser.parse_args()

    # Load config from YAML (topics, model, threshold)
    cfg: ConfigDict = load_interests()
    topics: list[TopicDict] = cfg.get("topics", [])

    # Resolve model: CLI flag → YAML config → built-in default
    model_name: str = args.model or cfg.get("model") or DEFAULT_MODEL
    # Resolve threshold: CLI flag → YAML config → built-in default
    threshold: float = args.threshold if args.threshold is not None else cfg.get("threshold") or DEFAULT_THRESHOLD

    # Load data
    papers: list[PaperDict] = fetch_papers()
    if not papers:
        output: dict[str, Any] = {
            "date": (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d"),
            "count": 0,
            "matched_count": 0,
            "papers": [],
        }
        print(json.dumps(output, indent=2))
        return

    topic_queries: list[TopicQuery] = build_topic_queries(topics)

    # Load model
    print(f"Loading model '{model_name}'...", file=sys.stderr)
    model: SentenceTransformer = SentenceTransformer(model_name)

    # Encode
    topic_embs: torch.Tensor
    topic_embs, topic_queries = encode_topics(model, topic_queries,
                                               None if args.no_cache else TOPIC_CACHE)
    print(f"Encoding {len(papers)} papers...", file=sys.stderr)
    paper_embs: torch.Tensor = encode_papers(model, papers)

    # Score & sort
    papers = score_papers(papers, paper_embs, topic_embs, topic_queries, threshold)
    papers = sort_papers(papers)

    matched: list[PaperDict] = [p for p in papers if p["matched"]]
    target_date: str = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

    # ── compare mode ──
    if args.compare:
        papers = keyword_score(papers, topics)
        print(f"\n{'Title':<72} {'Weighted':>8} {'Raw':>6} {'KW':>5}  Topic")
        print("─" * 115)
        for p in papers:
            if not p["matched"] and not p["kw_matched"]:
                continue
            title: str = p["title"][:69] + ".." if len(p["title"]) > 69 else p["title"]
            kw_hit: str = "✓" if p["kw_matched"] else "✗"
            sem_hit: str = "✓" if p["matched"] else "✗"
            print(f"{sem_hit} {title:<70} {p['match_score']:>8.4f} {p['raw_cosim']:>6.4f} {kw_hit:>5}  {p['semantic_topic']}")

        # Papers semantic catches but keyword misses
        print(f"\n── Semantic-matched but keyword-missed (weighted >= {threshold}) ──")
        missed: list[PaperDict] = [p for p in papers if p["matched"] and not p["kw_matched"]]
        for p in missed:
            title = p["title"][:69] + ".." if len(p["title"]) > 69 else p["title"]
            print(f"  {p['match_score']:.4f} (raw {p['raw_cosim']:.4f})  {title}  →  {p['semantic_topic']}")
        print(f"\nTotal semantic-matched: {len(matched)}")
        print(f"  of which keyword would have missed: {len(missed)}")
        print(f"Total keyword-matched: {len([p for p in papers if p['kw_matched']])}")
        return

    # ── normal output ──
    output = {
        "date": target_date,
        "count": len(papers),
        "matched_count": len(matched),
        "papers": papers,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
