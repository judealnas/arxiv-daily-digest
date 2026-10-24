"""Contract compliance for the LLM backend (fake client, zero API cost).

The LLM backend must satisfy the same Summarizer contract as BART, tested
through a fake LLMClient — no network, no tokens billed.
"""

from __future__ import annotations

import json
from typing import Any

SAMPLE_TEXT = (
    "Quantum repeaters are essential for long-distance quantum communication, "
    "but photon loss limits repeater spacing. We propose an architecture that "
    "achieves 9 km spacing using GKP error correction."
)


class FakeLLMClient:
    """Stands in for LLMClient: returns canned chat responses, records calls."""

    def __init__(self, response: str | Exception = "A concise summary of the text.") -> None:
        self._response = response
        self.calls: list[list[dict[str, Any]]] = []

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        self.calls.append(messages)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def test_llm_summarizer_exists_and_satisfies_contract():
    """LlmSummarizer constructed with a fake client satisfies the contract."""
    from arxiv_summarizer.llm import LlmSummarizer

    summarizer = LlmSummarizer(client=FakeLLMClient(json.dumps([{"id": "0", "summary": "We propose a 9 km repeater spacing via GKP codes."}])))
    result = summarizer.summarize(SAMPLE_TEXT)
    assert result["status"] == "ok", f"backend errored: {result['error']}"
    assert result["summary"]
    assert result["summary"].strip() != SAMPLE_TEXT.strip()


def test_llm_client_error_becomes_error_result():
    """A client failure must surface as status='error', not an exception."""
    from arxiv_summarizer.llm import LlmSummarizer

    summarizer = LlmSummarizer(client=FakeLLMClient(RuntimeError("401 unauthorized")))
    result = summarizer.summarize(SAMPLE_TEXT)
    assert result["status"] == "error"
    assert result["summary"] is None
    assert result["error"] == "RuntimeError: 401 unauthorized"


def test_llm_summarize_many_batches_by_default_with_request_id_mapping():
    """The compatibility adapter uses grouped JSON and restores input order."""
    from arxiv_summarizer.llm import LlmSummarizer

    client = FakeLLMClient(
        json.dumps(
            [
                {"id": "2", "summary": "S3"},
                {"id": "0", "summary": "S1"},
                {"id": "1", "summary": "S2"},
            ]
        )
    )
    results = LlmSummarizer(client=client).summarize_many(["t1", "t2", "t3"])

    assert [result["id"] for result in results] == ["0", "1", "2"]
    assert [result["summary"] for result in results] == ["S1", "S2", "S3"]
    assert len(client.calls) == 1
    prompt = client.calls[0][1]["content"]
    assert '"id": "0"' in prompt
    assert '"id": "1"' in prompt
    assert '"id": "2"' in prompt


def test_llm_summarize_many_rejects_duplicate_numbered_output():
    """Legacy numbered-line parsing must never recover ambiguous summaries."""
    from arxiv_summarizer.llm import LlmSummarizer

    client = FakeLLMClient("1. S1\n1. S2")
    results = LlmSummarizer(client=client).summarize_many(["t1", "t2"])

    assert [result["status"] for result in results] == ["error", "error"]
