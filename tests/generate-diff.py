#!/usr/bin/env python3
"""Generate a clean diff between input and sorted output JSON."""
import json
import re
import os

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(HERE, "fixtures", "arxiv-catchup-input.json")
SORTED_FILE = os.path.join(HERE, "fixtures", "arxiv-catchup-sorted.json")

with open(INPUT_FILE) as f:
    input_papers = json.load(f)
with open(SORTED_FILE) as f:
    output = json.load(f)
sorted_papers = output["papers"]

# Build lookup by ID for the sorted output
sorted_by_id = {p["id"]: p for p in sorted_papers}
input_by_id = {p["id"]: p for p in input_papers}

# Build a comparison table: input rank (unsorted id order) -> sorted rank
# Input is already in arxiv ID order (from the catchup script)
input_order = [p["id"] for p in input_papers]

# For the sorted papers, we track match_score, rank, topic
print("═" * 120)
print("  SEMANTIC SORTING DIFF — Input (arxiv-catchup) vs Sorted (semantic-score)")
print(f"  Total papers: {len(input_papers)}  |  Matched: {output['matched_count']} ({output['matched_count']/len(input_papers)*100:.1f}%)")
print("═" * 120)

# Print matched papers first (with ranking)
print("\n── MATCHED PAPERS (sorted by weighted score, descending) ──\n")
print(f"{'Rank':<5} {'Match':<7} {'Raw':<7} {'Topic':<28} {'Title / ID'}")
print("─" * 120)

matched = [p for p in sorted_papers if p["matched"]]
non_matched = [p for p in sorted_papers if not p["matched"]]

for rank, p in enumerate(matched, 1):
    title = p["title"][:75] + ".." if len(p["title"]) > 75 else p["title"]
    arxiv_id_short = p["id"].replace("http://arxiv.org/abs/", "")
    print(f"{rank:<5} {p['match_score']:<7.4f} {p['raw_cosim']:<7.4f} {p['semantic_topic']:<28} {title}")
    print(f"     {'':<7} {'':<7} {'':<28} {arxiv_id_short}")

print(f"\n── NON-MATCHED PAPERS ({len(non_matched)}) ──\n")
for p in non_matched:
    print(f"  {p['match_score']:.4f}  {p['semantic_topic']:<28} {p['title'][:70]}")

# ── JSON structural diff ──
print("\n\n═══ JSON STRUCTURAL DIFF ═══\n")

# Create simplified JSON files for diffing: each paper as {id, title, match_score, semantic_topic, matched}
input_simple = []
for pid in input_order:
    p = input_by_id[pid]
    entry = {
        "id": p["id"],
        "title": p["title"],
        "match_score": 0,
        "raw_cosim": 0,
        "matched": False,
        "semantic_topic": "",
        "_input_rank": input_order.index(pid),
    }
    input_simple.append(entry)

sorted_simple = []
for rank, p in enumerate(sorted_papers):
    entry = {
        "id": p["id"],
        "title": p["title"],
        "match_score": p["match_score"],
        "raw_cosim": p["raw_cosim"],
        "matched": p["matched"],
        "semantic_topic": p["semantic_topic"],
        "_sorted_rank": rank,
    }
    sorted_simple.append(entry)

# Write simplified JSON files for diffing
diff_dir = os.path.join(HERE, "fixtures")
with open(os.path.join(diff_dir, "_input_simple.json"), "w") as f:
    json.dump(input_simple, f, indent=2)
with open(os.path.join(diff_dir, "_sorted_simple.json"), "w") as f:
    json.dump(sorted_simple, f, indent=2)

print("Simplified diff files written:")
print(f"  tests/fixtures/_input_simple.json   — {len(input_simple)} papers (ID order, no scores)")
print(f"  tests/fixtures/_sorted_simple.json  — {len(sorted_simple)} papers (sorted order, with scores)")
