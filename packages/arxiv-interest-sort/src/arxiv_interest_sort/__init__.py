"""arxiv-interest-sort: sort arXiv papers by interest profile.

Usage
-----
    from arxiv_interest_sort import load_config, STRATEGIES, sort_papers

    papers = load_json(...)
    config = load_config("arxiv-interests.yaml")
    kwargs = {"model": "allenai/specter2_base", "threshold": 0.79}
    papers = STRATEGIES["keyword"](papers, config["topics"], **kwargs)
    papers = sort_papers(papers)
"""

from __future__ import annotations

import re
from typing import Any

import yaml

from arxiv_interest_sort.strategies import STRATEGIES, PaperDict, TopicDict

# ── public API ───────────────────────────────────────────────────

ConfigDict = dict[str, Any]


def load_config(path: str) -> ConfigDict:
    """Load interests configuration from a YAML file.

    Returns a dict with the ``topics`` key containing the list of topic dicts
    (``name``, ``weight``, ``keywords``).
    """
    with open(path) as f:
        cfg: ConfigDict = yaml.safe_load(f)
    if "topics" not in cfg:
        cfg["topics"] = []
    return cfg


def sort_papers(papers: list[PaperDict]) -> list[PaperDict]:
    """Sort papers: matched first, then by score descending and arXiv ID ascending.

    Within each group, sort by ``match_score`` descending.
    Non-matched papers (``matched == False``) are sorted by arXiv ID ascending.
    """

    def _key(p: PaperDict) -> tuple[int, float, str]:
        arxiv_id: str = p.get("id", "")
        arxiv_id_clean: str = re.sub(r"v\d+$", "", arxiv_id)
        return (
            0 if p.get("matched", False) else 1,
            -p.get("match_score", 0.0),
            arxiv_id_clean,
        )

    return sorted(papers, key=_key)


def score_and_sort(
    papers: list[PaperDict],
    topics: list[TopicDict],
    strategy: str = "keyword",
    **kwargs: Any,
) -> list[PaperDict]:
    """Convenience: apply a strategy then sort the papers in-place.

    Parameters
    ----------
    papers:
        List of paper dicts (will be mutated).
    topics:
        List of topic dicts from :func:`load_config`.
    strategy:
        Strategy name to look up in :data:`STRATEGIES`.
    **kwargs:
        Forwarded to the strategy function.
    """
    fn = STRATEGIES[strategy]
    fn(papers, topics, **kwargs)
    return sort_papers(papers)
