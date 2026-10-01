"""OpenRouter backend tests: no network or paid API calls."""

from __future__ import annotations

import json

from arxiv_summarizer import OpenRouterSummarizer, Summarizer


class FakeClient:
    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = iter(responses)
        self.calls: list[tuple[list[dict[str, str]], dict]] = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return result


def test_openrouter_is_drop_in_summarizer_for_single_and_many() -> None:
    client = FakeClient([
        json.dumps([{"id": "0", "summary": "Concise single summary."}]),
        json.dumps([{"id": "0", "summary": "First summary."}]),
        json.dumps([{"id": "1", "summary": "Second summary."}]),
    ])
    summarizer: Summarizer = OpenRouterSummarizer(client=client)

    single = summarizer.summarize("First abstract. Some detail.")
    results = summarizer.summarize_many(["First abstract.", "Second abstract."])

    assert single == {"id": "0", "status": "ok", "summary": "Concise single summary.", "error": None}
    assert results == [
        {"id": "0", "status": "ok", "summary": "First summary.", "error": None},
        {"id": "1", "status": "ok", "summary": "Second summary.", "error": None},
    ]
    assert len(client.calls) == 3
    assert "First abstract." in client.calls[1][0][1]["content"]
    assert "Second abstract." in client.calls[2][0][1]["content"]


def test_failed_request_does_not_discard_other_summaries() -> None:
    client = FakeClient([
        json.dumps([{"id": "0", "summary": "First summary."}]),
        RuntimeError("rate limit"),
        json.dumps([{"id": "2", "summary": "Third summary."}]),
    ])
    results = OpenRouterSummarizer(client=client).summarize_many(["a", "b", "c"])

    assert [item["status"] for item in results] == ["ok", "error", "ok"]
    assert [item["id"] for item in results] == ["0", "1", "2"]
    assert results[1]["summary"] is None
    assert results[1]["error"] == "LLM request failed"


def test_empty_response_is_an_error_not_a_successful_summary() -> None:
    result = OpenRouterSummarizer(client=FakeClient(["  "])).summarize("An abstract")

    assert result["status"] == "error"
    assert result["summary"] is None
    assert result["error"]


def test_openrouter_resolves_its_endpoint_key_and_explicit_model(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    summarizer = OpenRouterSummarizer(model="example/model")

    assert summarizer._resolve_provider() == (
        "https://openrouter.ai/api/v1", "test-key", "example/model"
    )


def test_cli_selects_openrouter_backend(monkeypatch) -> None:
    import arxiv_summarizer.cli as cli

    constructed = []
    monkeypatch.setattr(cli, "OpenRouterSummarizer", lambda **kw: constructed.append(kw) or FakeSummarizer())
    args = cli.build_parser().parse_args(["--backend", "openrouter", "--model", "example/model"])

    backend = cli.build_backend(args)

    assert isinstance(backend, FakeSummarizer)
    assert constructed == [{
        "model": "example/model",
        "rpm_limit": 0,
        "temperature": 0.3,
        "max_tokens": 2048,
        "user_prompt": None,
        "papers_per_prompt": None,
        "base_url": None,
        "api_key": None,
    }]


class FakeSummarizer:
    pass


def test_console_entrypoint_forwards_backend_flag_and_restores_paper_ids(monkeypatch, capsys):
    import io
    import json
    import sys

    import arxiv_summarizer.cli as cli

    monkeypatch.setattr(sys, "argv", ["arxiv-summarize", "--backend", "openrouter"])
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO('{"id":"paper-z","text":"First abstract"}\n{"id":"paper-a","text":"Second abstract"}\n'),
    )
    client = FakeClient([
        json.dumps([{"id": "paper-z", "summary": "First summary."}]),
        json.dumps([{"id": "paper-a", "summary": "Second summary."}]),
    ])
    monkeypatch.setattr(cli, "OpenRouterSummarizer", lambda **kw: OpenRouterSummarizer(client=client))
    monkeypatch.setattr(cli, "BartSummarizer", lambda **kw: (_ for _ in ()).throw(AssertionError("selected BART")))

    cli.console_main()

    assert len(client.calls) == 2
    assert [json.loads(line) for line in capsys.readouterr().out.splitlines()] == [
        {"id": "paper-z", "status": "ok", "summary": "First summary.", "error": None},
        {"id": "paper-a", "status": "ok", "summary": "Second summary.", "error": None},
    ]


def test_openrouter_http_request_matches_chat_completions_contract(monkeypatch):
    import json
    import urllib.request

    from arxiv_summarizer.llm import SYSTEM_PROMPT

    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({"choices": [{"message": {"content": '[{"id":"0","summary":"A short summary."}]'}}]}).encode()

    def fake_urlopen(request, timeout):
        requests.append(request)
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = OpenRouterSummarizer(
        model="example/model",
        api_key="test-key",
        user_prompt="Emphasize numerical results.",
        temperature=0.1,
        max_tokens=175,
    ).summarize("An abstract")

    assert result["summary"] == "A short summary."
    assert len(requests) == 1
    request = requests[0]
    assert request.full_url == "https://openrouter.ai/api/v1/chat/completions"
    assert request.get_method() == "POST"
    headers = {name.lower(): value for name, value in request.header_items()}
    assert headers["authorization"] == "Bearer test-key"
    assert headers["content-type"] == "application/json"
    assert json.loads(request.data) == {
        "model": "example/model",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"guidance": "Emphasize numerical results.", "papers": [{"id": "0", "text": "An abstract"}]})},
        ],
        "temperature": 0.1,
        "max_tokens": 175,
    }


def test_openrouter_papers_per_prompt_uses_grouped_json():
    import json

    client = FakeClient([json.dumps([{"id": "1", "summary": "B"}, {"id": "0", "summary": "A"}]), json.dumps([{"id": "2", "summary": "C"}])])
    results = OpenRouterSummarizer(client=client, papers_per_prompt=2).summarize_many(["a", "b", "c"])

    assert [item["summary"] for item in results] == ["A", "B", "C"]
    assert len(client.calls) == 2
