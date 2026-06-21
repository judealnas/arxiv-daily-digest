"""Keyword-based scoring strategy.

Matches papers by case-insensitive substring search of topic keywords
in the title and abstract.
"""

from __future__ import annotations

from typing import Any

# ── type aliases ─────────────────────────────────────────────────
PaperDict = dict[str, Any]
TopicDict = dict[str, Any]


def keyword_score(
    papers: list[PaperDict],
    topics: list[TopicDict],
    **kwargs: Any,
) -> list[PaperDict]:
    """Score papers by substring keyword matching (case-insensitive).

    Each matching keyword contributes its topic's weight to the total
    ``match_count``.  A paper is considered ``matched`` when
    ``match_count > 0``.

    Annotations added
    -----------------
    * ``matched`` — ``True`` if any keyword matched
    * ``match_score`` — same as ``match_count`` (raw weighted sum)
    * ``match_count`` — cumulative weighted sum of matched keywords
    * ``matched_topics`` — list of topic names that contributed a match
    """
    for p in papers:
        text: str = (p.get("title", "") + " " + p.get("abstract", "")).lower()
        total: int = 0
        matched_topics: list[str] = []

        for topic in topics:
            w: int = topic.get("weight", 3)
            topic_hit: bool = False
            for kw in topic["keywords"]:
                if kw.lower() in text:
                    total += w
                    topic_hit = True
            if topic_hit:
                matched_topics.append(topic["name"])

        p["match_score"] = total
        p["match_count"] = total
        p["matched"] = total > 0
        p["matched_topics"] = matched_topics

    return papers
