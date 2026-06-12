#!/usr/bin/env python3
"""Run the semantic scorer on the fixture input file and output sorted JSON."""
import json
import os
import sys

# Add project root to path
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)

import yaml
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util

# ── paths ──
INPUT_FILE = os.path.join(HERE, "fixtures", "arxiv-catchup-input.json")
OUTPUT_FILE = os.path.join(HERE, "fixtures", "arxiv-catchup-sorted.json")
INTERESTS_FILE = os.path.join(PROJECT, "arxiv-interests.yaml")

# ── load config from YAML ──
with open(INTERESTS_FILE) as f:
    cfg = yaml.safe_load(f)

MODEL_NAME = cfg.get("model", "all-MiniLM-L6-v2")
THRESHOLD = cfg.get("threshold", 0.79)
topics = cfg.get("topics", [])

print(f"Model: {MODEL_NAME}", file=sys.stderr)
print(f"Threshold: {THRESHOLD}", file=sys.stderr)

# ── load input ──
with open(INPUT_FILE) as f:
    papers = json.load(f)
print(f"Loaded {len(papers)} papers from input", file=sys.stderr)

# ── build topic queries ──
topic_queries = []
for t in topics:
    q = t["name"] + ": " + ", ".join(t["keywords"])
    topic_queries.append({"name": t["name"], "query": q, "weight": t.get("weight", 3)})
print(f"Topics: {[t['name'] for t in topic_queries]}", file=sys.stderr)

# ── load model ──
model = SentenceTransformer(MODEL_NAME)
device = model.device

# ── encode topics ──
topic_texts = [t["query"] for t in topic_queries]
topic_embs = model.encode(topic_texts, convert_to_tensor=True)

# ── encode papers ──
paper_texts = [p["title"] + ". " + p.get("abstract", "")[:1000] for p in papers]
paper_embs = model.encode(paper_texts, convert_to_tensor=True)
print(f"Encoded {len(paper_texts)} papers", file=sys.stderr)

# ── score ──
max_weight = max(t["weight"] for t in topic_queries)
cos = util.cos_sim(paper_embs, topic_embs)  # (N × T)
weight_vec = torch.tensor(
    [t["weight"] / max_weight for t in topic_queries],
    device=cos.device,
)
weighted = cos * weight_vec
best_weighted, best_indices = weighted.max(dim=1)

for i, p in enumerate(papers):
    ws = float(best_weighted[i])
    topic_idx = int(best_indices[i])
    p["match_score"] = round(ws, 4)
    p["match_count"] = int(round(ws * 100))
    p["matched"] = ws >= THRESHOLD
    p["semantic_topic"] = topic_queries[topic_idx]["name"]
    p["raw_cosim"] = round(float(cos[i][topic_idx]), 4)

# ── sort ──
import re
def sort_key(p):
    arxiv_id = p.get("id", "")
    arxiv_id_clean = re.sub(r'v\d+$', '', arxiv_id)
    return (0 if p["matched"] else 1, -p["match_score"], arxiv_id_clean)

papers = sorted(papers, key=sort_key)

matched = [p for p in papers if p["matched"]]
output = {
    "date": "2026-06-11",
    "count": len(papers),
    "matched_count": len(matched),
    "papers": papers,
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2)
print(f"Wrote {OUTPUT_FILE}", file=sys.stderr)
print(f"  Total papers: {len(papers)}")
print(f"  Matched:      {len(matched)} ({len(matched)/len(papers)*100:.1f}%)")
print(f"  Threshold:    {THRESHOLD}")
