"""Identity-preserving grouped LLM request tests (all clients are fakes)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from arxiv_summarizer.llm import LlmSummarizer


class FakeClient:
    def __init__(self, responses: list[str | Exception]) -> None:
        self._responses = iter(responses)
        self.calls: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        self.calls.append((messages, kwargs))
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        return response


def items() -> list[dict[str, str]]:
    return [
        {"id": "2609.00002", "text": 'First abstract with a "quoted" phrase.'},
        {"id": "2609.00003", "text": "Second abstract.\nWith another line."},
    ]


def valid_response(records: list[dict[str, str]] | None = None) -> str:
    return json.dumps(
        records
        or [
            {"id": "2609.00002", "summary": "First summary."},
            {"id": "2609.00003", "summary": "Second summary."},
        ]
    )


def test_grouped_request_embeds_id_text_json_and_requires_json_response() -> None:
    source = items()
    client = FakeClient([valid_response()])

    results = LlmSummarizer(client=client).summarize_items(source)

    assert [result["id"] for result in results] == ["2609.00002", "2609.00003"]
    assert len(client.calls) == 1
    prompt = client.calls[0][0][1]["content"]
    assert json.loads(prompt) == {"guidance": "Summarize each paper concisely.", "papers": source}
    system_prompt = client.calls[0][0][0]["content"]
    assert '"id" and "summary"' in system_prompt
    assert "exactly once" in system_prompt
    assert "JSON array of objects shaped" not in prompt


def test_grouped_request_uses_custom_user_guidance_without_losing_id_contract() -> None:
    client = FakeClient([valid_response()])
    guidance = "Explain each quantum paper for a general audience. Keep it brief."

    results = LlmSummarizer(client=client, user_prompt=guidance).summarize_items(items())

    assert all(result["status"] == "ok" for result in results)
    prompt = client.calls[0][0][1]["content"]
    assert json.loads(prompt) == {"guidance": guidance, "papers": items()}
    assert "exactly once" in client.calls[0][0][0]["content"]


def test_grouped_json_response_is_reordered_to_input_ids() -> None:
    client = FakeClient(
        [
            valid_response(
                [
                    {"id": "2609.00003", "summary": "Second summary."},
                    {"id": "2609.00002", "summary": "First summary."},
                ]
            )
        ]
    )

    results = LlmSummarizer(client=client).summarize_items(items())

    assert results == [
        {"id": "2609.00002", "status": "ok", "summary": "First summary.", "error": None, "summary_metadata": {"requested_model": "big-pickle", "returned_model": None, "finish_reason": None}},
        {"id": "2609.00003", "status": "ok", "summary": "Second summary.", "error": None, "summary_metadata": {"requested_model": "big-pickle", "returned_model": None, "finish_reason": None}},
    ]


@pytest.mark.parametrize(
    "response",
    [
        valid_response(
            [
                {"id": "2609.00002", "summary": "First summary."},
                {"id": "2609.00002", "summary": "Second summary."},
            ]
        ),
        valid_response([{"id": "2609.00002", "summary": "First summary."}]),
        valid_response(
            [
                {"id": "2609.00002", "summary": "First summary."},
                {"id": "unknown", "summary": "Unknown summary."},
            ]
        ),
        valid_response(
            [
                {"id": "2609.00002", "summary": "   "},
                {"id": "2609.00003", "summary": "Second summary."},
            ]
        ),
        valid_response(
            [
                {"id": "2609.00002", "summary": 42},
                {"id": "2609.00003", "summary": "Second summary."},
            ]
        ),
        "not JSON",
        "Here is the JSON:\n" + valid_response(),
        "```json\n" + valid_response() + "\n```",
        '[{"id":"2609.00003","id":"2609.00002","summary":"Ambiguous"},'
        '{"id":"2609.00003","summary":"Second"}]',
        json.dumps({"id": "2609.00002", "summary": "Not a list."}),
        json.dumps(["not an object", {"id": "2609.00003", "summary": "Second summary."}]),
        valid_response(
            [
                {"id": "2609.00002", "summary": "First summary.", "extra": "nope"},
                {"id": "2609.00003", "summary": "Second summary."},
            ]
        ),
    ],
)
def test_ambiguous_grouped_responses_fail_the_entire_group(response: str) -> None:
    client = FakeClient([response])

    results = LlmSummarizer(client=client).summarize_items(items())

    assert [result["id"] for result in results] == ["2609.00002", "2609.00003"]
    assert all(result["status"] == "error" for result in results)
    assert all(result["summary"] is None for result in results)
    assert all(result["error"] for result in results)
    # A placeholder like "not JSON" can overlap the fixed rejection wording.
    if response != "not JSON":
        assert all(response not in (result["error"] or "") for result in results)


def test_invalid_group_does_not_prevent_later_group_from_succeeding() -> None:
    source = [
        {"id": "a", "text": "First."},
        {"id": "b", "text": "Second."},
        {"id": "c", "text": "Third."},
        {"id": "d", "text": "Fourth."},
    ]
    client = FakeClient(
        [
            "not JSON",
            json.dumps([{"id": "c", "summary": "Third summary."}, {"id": "d", "summary": "Fourth summary."}]),
        ]
    )

    results = LlmSummarizer(client=client, chunk_count=2).summarize_items(source)

    assert [result["status"] for result in results] == ["error", "error", "ok", "ok"]
    assert [result["id"] for result in results] == ["a", "b", "c", "d"]
    assert len(client.calls) == 2


def test_one_item_group_uses_same_json_contract() -> None:
    client = FakeClient([json.dumps([{"id": "paper-1", "summary": "A summary."}])])

    results = LlmSummarizer(client=client).summarize_items([{"id": "paper-1", "text": "An abstract."}])

    assert results == [{"id": "paper-1", "status": "ok", "summary": "A summary.", "error": None, "summary_metadata": {"requested_model": "big-pickle", "returned_model": None, "finish_reason": None}}]
    assert json.loads(client.calls[0][0][1]["content"])["papers"] == [{"id": "paper-1", "text": "An abstract."}]
    assert "exactly once" in client.calls[0][0][0]["content"]


@pytest.mark.parametrize("bad_id", ["", 1, None])
def test_invalid_input_ids_are_rejected_without_calling_client(bad_id: object) -> None:
    client = FakeClient([])

    with pytest.raises(ValueError, match="nonempty strings"):
        LlmSummarizer(client=client).summarize_items([{"id": bad_id, "text": "An abstract."}])  # type: ignore[typeddict-item]

    assert client.calls == []


def test_duplicate_input_ids_are_rejected_without_calling_client() -> None:
    client = FakeClient([])

    with pytest.raises(ValueError, match="unique"):
        LlmSummarizer(client=client).summarize_items(
            [{"id": "same", "text": "First."}, {"id": "same", "text": "Second."}]
        )

    assert client.calls == []


def test_papers_per_prompt_groups_by_fixed_size():
    source = [{"id": str(i), "text": f"Text {i}"} for i in range(5)]
    client = FakeClient(
        [
            json.dumps([{"id": "0", "summary": "S0"}, {"id": "1", "summary": "S1"}]),
            json.dumps([{"id": "2", "summary": "S2"}, {"id": "3", "summary": "S3"}]),
            json.dumps([{"id": "4", "summary": "S4"}]),
        ]
    )

    results = LlmSummarizer(client=client, chunk_count=99, papers_per_prompt=2).summarize_items(source)

    assert [result["summary"] for result in results] == ["S0", "S1", "S2", "S3", "S4"]
    assert len(client.calls) == 3


def test_failed_group_does_not_prevent_later_group_from_succeeding() -> None:
    source = [{"id": str(i), "text": f"Text {i}"} for i in range(3)]
    client = FakeClient(
        [
            RuntimeError("rate limited; response contained secret-token-123"),
            json.dumps([{"id": "2", "summary": "Third summary."}]),
        ]
    )

    results = LlmSummarizer(client=client, papers_per_prompt=2).summarize_items(source)

    assert [result["status"] for result in results] == ["error", "error", "ok"]
    assert results[0]["error"] == "RuntimeError: rate limited; response contained secret-token-123"
    assert results[2]["summary"] == "Third summary."


def test_http_error_preserves_reported_status_and_message() -> None:
    import urllib.error

    error = urllib.error.HTTPError("https://example.test", 429, "sensitive response", None, None)
    results = LlmSummarizer(client=FakeClient([error])).summarize_items([{"id": "a", "text": "Abstract"}])
    assert results == [{"id": "a", "status": "error", "summary": None, "error": "HTTPError: HTTP Error 429: sensitive response", "summary_error_code": "http_429"}]


@pytest.mark.parametrize("papers_per_prompt", [0, -1])
def test_papers_per_prompt_must_be_positive(papers_per_prompt: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        LlmSummarizer(client=FakeClient([]), papers_per_prompt=papers_per_prompt)


def test_empty_items_do_not_resolve_or_call_client() -> None:
    client = FakeClient([])
    assert LlmSummarizer(client=client, papers_per_prompt=1).summarize_items([]) == []
    assert client.calls == []


@pytest.mark.parametrize("response", ["", "   ", 1])
def test_one_item_group_rejects_non_json_or_blank_response(response: object) -> None:
    client = FakeClient([response])
    results = LlmSummarizer(client=client).summarize_items([{"id": "paper", "text": "Text"}])
    assert results == [{"id": "paper", "status": "error", "summary": None, "error": "Invalid summary response: response is not JSON", "summary_error_code": "invalid_json", "summary_metadata": {"requested_model": "big-pickle", "returned_model": None, "finish_reason": None}}]
