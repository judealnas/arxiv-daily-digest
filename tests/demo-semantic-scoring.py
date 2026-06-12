#!/usr/bin/env python3
"""Demo: semantic interest scoring with sentence-transformers vs keyword matching."""

import json, sys, yaml
from sentence_transformers import SentenceTransformer, util

FIXTURE = "tests/fixtures/arxiv-catchup-sample.json"
INTERESTS = "arxiv-interests.yaml"

# ── 1. Load data ────────────────────────────────────────────────
with open(FIXTURE) as f:
    papers = json.load(f)
with open(INTERESTS) as f:
    topics = yaml.safe_load(f)["topics"]

# ── 2. Build interest queries ───────────────────────────────────
topic_queries = []
for t in topics:
    q = t["name"] + ": " + ", ".join(t["keywords"])
    topic_queries.append({"name": t["name"], "query": q, "weight": t.get("weight", 3)})

max_weight = max(t["weight"] for t in topic_queries)
print(f"Loaded {len(papers)} papers, {len(topic_queries)} interest topics\n")
for t in topic_queries:
    print(f"  {t['name']:>40}  weight={t['weight']}")
print()

# ── 3. Encode everything ────────────────────────────────────────
model = SentenceTransformer("all-MiniLM-L6-v2")

topic_embs = model.encode(
    [t["query"] for t in topic_queries],
    convert_to_tensor=True,
)

paper_texts = [
    p["title"] + ". " + p["abstract"][:1000] for p in papers
]
paper_embs = model.encode(paper_texts, convert_to_tensor=True)

# ── 4. Score: weighted cosine similarity ────────────────────────
cos = util.cos_sim(paper_embs, topic_embs)
weight_vec = topic_embs.new_tensor([t["weight"] / max_weight for t in topic_queries])
weighted = cos * weight_vec

best_weighted, best_indices = weighted.max(dim=1)
best_raw, _ = cos.max(dim=1)

for i, p in enumerate(papers):
    p["weighted_score"] = round(float(best_weighted[i]), 4)
    p["raw_cosim"] = round(float(best_raw[i]), 4)
    p["semantic_topic"] = topic_queries[int(best_indices[i])]["name"]
    p["matched_semantic"] = p["weighted_score"] >= 0.35
    # Keyword scoring w/ weight
    text = (p["title"] + " " + p["abstract"]).lower()
    kw_total = 0
    for t in topics:
        w = t.get("weight", 3)
        for kw in t["keywords"]:
            if kw.lower() in text:
                kw_total += w
    p["kw_matched"] = kw_total > 0
    p["kw_score"] = kw_total

# ── 5. Compare — top papers by weighted score ───────────────────
papers_sorted = sorted(papers, key=lambda p: -p["weighted_score"])

print(f"{'Title':<72} {'WScore':>7} {'Raw':>6}  Topic")
print("─" * 115)
for p in papers_sorted[:15]:
    title = p["title"][:69] + ".." if len(p["title"]) > 69 else p["title"]
    print(f"{title:<72} {p['weighted_score']:>7.4f} {p['raw_cosim']:>6.4f}  {p['semantic_topic']}")

# ── 6. Papers keyword misses but semantic catches ──────────────
print(f"\n═══ Keyword-missed papers (weighted >= 0.35) ═══")
missed = [p for p in papers_sorted if not p["kw_matched"] and p["weighted_score"] >= 0.35]
for p in missed[:8]:
    title = p["title"][:69] + ".." if len(p["title"]) > 69 else p["title"]
    print(f"  [{p['weighted_score']:.4f}] {title}  →  topic: {p['semantic_topic']}")

print(f"\nTotal semantic-matched (weighted >= 0.35): {len([p for p in papers if p['matched_semantic']])}")
print(f"  of which keyword would have missed: {len(missed)}")
print(f"Total keyword-matched: {len([p for p in papers if p['kw_matched']])}")

# Show weight effect: compare before/after for trapped-ions papers
print(f"\n═══ Weight effect on trapped-ions papers ═══")
ion_papers = [p for p in papers if p["semantic_topic"] == "trapped ions"]
for p in sorted(ion_papers, key=lambda x: -x["weighted_score"])[:5]:
    title = p["title"][:69] + ".." if len(p["title"]) > 69 else p["title"]
    print(f"  raw={p['raw_cosim']:.4f} → weighted={p['weighted_score']:.4f}  {title}")
