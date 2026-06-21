"""Scoring strategies for arxiv-interest-sort.

Each strategy is a function with the signature::

    def strategy_fn(
        papers: list[PaperDict],
        topics: list[TopicDict],
        **kwargs: Any,
    ) -> list[PaperDict]:

The function mutates each paper dict in-place, adding scoring annotations
(``matched``, ``match_score``, ``match_count``, …), and returns the list.
"""

from __future__ import annotations

from typing import Any

from arxiv_interest_sort.strategies.keyword import keyword_score
from arxiv_interest_sort.strategies.semantic import semantic_score

# ── type aliases ─────────────────────────────────────────────────
PaperDict = dict[str, Any]
TopicDict = dict[str, Any]

STRATEGIES: dict[str, Any] = {
    "keyword": keyword_score,
    "semantic": semantic_score,
}
"""Dispatch dict mapping strategy names to scoring functions."""
