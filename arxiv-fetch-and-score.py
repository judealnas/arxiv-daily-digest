#!/usr/bin/env python3
"""
Fetch quant-ph papers from arXiv catchup for yesterday, score by interest topics,
sort matched first, and output scored JSON. Designed for no_agent cron jobs.

Scoring uses weighted keyword matching — each match contributes its topic's
weight value instead of a flat count of 1.
"""
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone, timedelta

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "arxiv-catchup.py")
INTERESTS = os.path.join(HERE, "arxiv-interests.yaml")


def fetch_papers():
    """Run the catchup script and return paper list."""
    result = subprocess.run(
        [sys.executable, SCRIPT, "quant-ph", "--yesterday"],
        capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        print(json.dumps({"error": f"Fetch failed: {result.stderr.strip()}"}), file=sys.stderr)
        sys.exit(1)
    try:
        papers = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"JSON parse error: {e}"}), file=sys.stderr)
        sys.exit(1)
    return papers


def load_interests():
    with open(INTERESTS) as f:
        return yaml.safe_load(f)["topics"]


def score_papers(papers, topics):
    """Add matched, match_count fields using weighted keyword matching."""
    for p in papers:
        text = (p.get("title", "") + " " + p.get("abstract", "")).lower()
        score = 0
        for topic in topics:
            weight = topic.get("weight", 3)
            for kw in topic["keywords"]:
                if kw.lower() in text:
                    score += weight
        p["matched"] = score > 0
        p["match_count"] = score
    return papers


def sort_papers(papers):
    """Matched first (by match_count desc, then id asc), non-matched by id asc."""
    def sort_key(p):
        arxiv_id = p.get("id", "")
        arxiv_id_clean = re.sub(r'v\d+$', '', arxiv_id)
        return (0 if p["matched"] else 1, -p["match_count"], arxiv_id_clean)
    return sorted(papers, key=sort_key)


def main():
    papers = fetch_papers()
    if not papers:
        print(json.dumps({"date": "", "count": 0, "matched_count": 0, "papers": []}))
        return

    topics = load_interests()
    papers = score_papers(papers, topics)
    papers = sort_papers(papers)

    target_date = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    matched = [p for p in papers if p["matched"]]

    output = {
        "date": target_date,
        "count": len(papers),
        "matched_count": len(matched),
        "papers": papers,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
