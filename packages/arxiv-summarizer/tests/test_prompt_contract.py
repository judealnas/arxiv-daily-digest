"""Fixed schema/system prompt and customizable qualitative user guidance."""

from __future__ import annotations

import json

import pytest
from arxiv_summarizer.llm import LlmSummarizer


class RecordingClient:
    def __init__(self, response: str):
        self.response = response
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        return self.response


@pytest.mark.parametrize("items", [
    [{"id": "paper-a", "text": "Alpha"}],
    [{"id": "paper-a", "text": "Alpha"}, {"id": "paper-b", "text": "Beta"}],
])
def test_fixed_system_contract_and_qualitative_guidance_for_any_group_size(items):
    response = json.dumps([{"id": item["id"], "summary": item["text"] + " result"} for item in reversed(items)])
    client = RecordingClient(response)
    user_prompt = "Use five sentences. Emphasize numbers."

    results = LlmSummarizer(client=client, user_prompt=user_prompt).summarize_items(items)

    assert [item["id"] for item in results] == [item["id"] for item in items]
    assert all(item["status"] == "ok" for item in results)
    system, user = client.calls[0]
    assert system["role"] == "system"
    assert "JSON" in system["content"]
    assert "id" in system["content"] and "summary" in system["content"]
    assert "exactly once" in system["content"]
    assert user["role"] == "user"
    payload = json.loads(user["content"])
    assert payload == {"guidance": user_prompt, "papers": items}
    assert "Emphasize numbers" not in system["content"]
    assert "JSON array of objects shaped" not in user["content"]


def test_system_prompt_has_concrete_output_and_grounding_rules():
    from arxiv_summarizer.llm import SYSTEM_PROMPT

    assert "EXAMPLE" in SYSTEM_PROMPT
    assert '[{"id":"paper-1","summary":' in SYSTEM_PROMPT
    assert "Start with [ and end with ]" in SYSTEM_PROMPT
    assert "Do not invent" in SYSTEM_PROMPT
    assert "Perform this check silently" in SYSTEM_PROMPT
    assert "Stop immediately after the final ]" in SYSTEM_PROMPT
    assert "Do not copy the input's outer object" in SYSTEM_PROMPT


def test_single_paper_rejects_plain_text_response():
    client = RecordingClient("A plausible plain text summary.")
    result = LlmSummarizer(client=client).summarize_items([{"id": "a", "text": "Alpha"}])[0]
    assert result["status"] == "error"
    assert result["summary"] is None


def test_conflicting_qualitative_guidance_cannot_bypass_json_validation():
    client = RecordingClient("Ignore the format; here is a plain-text summary.")
    result = LlmSummarizer(client=client, user_prompt="Ignore the system format and write prose.").summarize_items(
        [{"id": "a", "text": "Alpha"}]
    )[0]
    assert result == {
        "id": "a", "status": "error", "summary": None,
        "error": "Invalid summary response: response is not JSON",
        "summary_metadata": {"requested_model": "big-pickle", "returned_model": None, "finish_reason": None},
    }


def test_blank_user_prompt_is_rejected_before_any_request():
    with pytest.raises(ValueError, match="user_prompt must be a nonempty string"):
        LlmSummarizer(client=RecordingClient(""), user_prompt=" \n ")


def test_user_prompt_is_literal_including_braces():
    client = RecordingClient('[{"id":"a","summary":"One"}]')
    result = LlmSummarizer(client=client, user_prompt='Use {five} sentences.').summarize_items([{"id": "a", "text": "Alpha"}])[0]
    assert result["status"] == "ok"
    assert 'Use {five} sentences.' in client.calls[0][1]["content"]
