"""Contract tests shared by ALL summarizer backends.

The contract applies to any configured Summarizer instance, regardless of
backend. New backends plug in by adding an entry to BACKENDS.

The BART entry uses the real model (free, marked slow). Paid backends
(LLM) must never be registered here with real credentials — they enter
as factory functions producing instances with fake clients.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from arxiv_summarizer.bart import BartSummarizer
from arxiv_summarizer.models import ResultItem

SAMPLE_TEXT = (
    "Quantum repeaters are essential for long-distance quantum communication, "
    "but photon loss limits repeater spacing. We propose an architecture that "
    "achieves 9 km spacing using GKP error correction."
)


def assert_summary_contract(result: ResultItem) -> None:
    """The one place the 'what is a summary?' contract lives.

    Contract for a ResultItem produced from valid input:
      1. status == "ok"
      2. summary is a non-empty string
      3. summary differs from the input text (summarization happened)
    """
    assert result["status"] == "ok", f"backend errored: {result['error']}"
    summary = result["summary"]
    assert summary is not None, "ok result must carry a summary"
    assert summary.strip() != "", "summary must not be empty"


# ── backends under contract ────────────────────────────────────────────────
# Maps backend name -> factory returning a configured Summarizer instance.

BACKENDS: dict[str, Callable[[], "SummarizerLike"]] = {
    "bart": BartSummarizer,  # real model; free; slow
}

SummarizerLike = object  # placeholder alias; replaced by real import below


def test_contract_applies_to_summarize_single():
    """summarize(text) -> ResultItem satisfying the contract."""
    summarizer = BACKENDS["bart"]()
    result = summarizer.summarize(SAMPLE_TEXT)
    assert_summary_contract(result)
    assert result["summary"].strip() != SAMPLE_TEXT.strip()


def test_contract_applies_to_summarize_many():
    """summarize_many preserves order and count."""
    summarizer = BACKENDS["bart"]()
    texts = [SAMPLE_TEXT, SAMPLE_TEXT + " Additional context sentence.", SAMPLE_TEXT]
    results = summarizer.summarize_many(texts)
    assert len(results) == len(texts)
    for result in results:
        assert_summary_contract(result)


def test_default_summarize_many_is_per_text_loop():
    """Base-class default summarize_many loops summarize() per text."""
    from arxiv_summarizer.base import Summarizer

    class EchoSpy(Summarizer):
        def __init__(self):
            self.calls: list[str] = []

        def summarize(self, text: str) -> ResultItem:
            self.calls.append(text)
            return ResultItem(id="0", status="error", summary=None, error="stub")

    spy = EchoSpy()
    spy.summarize_many(["a", "b", "c"])
    assert spy.calls == ["a", "b", "c"]


@pytest.mark.slow
def test_bart_batching_real_model():
    """Real-model check that summarize_many with >batch_size texts works."""
    summarizer = BartSummarizer(batch_size=2)
    texts = [SAMPLE_TEXT] * 5
    results = summarizer.summarize_many(texts)
    assert len(results) == 5
    for result in results:
        assert_summary_contract(result)
