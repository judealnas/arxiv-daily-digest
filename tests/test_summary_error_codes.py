from __future__ import annotations

import io
import json
import urllib.error

import pytest
from arxiv_summarizer import LlmSummarizer
from arxiv_summarizer.client import CompletionText
from arxiv_summarizer.models import dump_result


class Reply:
    def __init__(self, value):
        self.value = value

    def chat(self, messages, **kwargs):
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


@pytest.mark.parametrize(
    "response,code",
    [
        ("not JSON", "invalid_json"),
        ('[{"id":"wrong","summary":"text"}]', "id_mismatch"),
        ('[{"id":"a","summary":""}]', "invalid_summary"),
        ('[{"id":"a","text":"bad shape"}]', "invalid_response"),
        ("[]", "invalid_response"),
        (CompletionText("truncated", "model", "length"), "output_token_limit"),
    ],
)
def test_parser_failures_have_codes_without_changing_raw_errors(response, code) -> None:
    result = LlmSummarizer(client=Reply(response)).summarize_items([{"id": "a", "text": "Abstract."}])[0]
    assert result["status"] == "error"
    assert result["summary_error_code"] == code
    assert result["error"].startswith("Invalid summary response: ")
    assert json.loads(dump_result(result))["summary_error_code"] == code


@pytest.mark.parametrize(
    "error,code",
    [
        (TimeoutError("original timeout detail"), "timeout"),
        (urllib.error.URLError(TimeoutError("nested timeout")), "timeout"),
        (RuntimeError("original failure detail"), "request_failed"),
        (urllib.error.HTTPError("https://example.test", 429, "original", None, io.BytesIO(b"raw body")), "http_429"),
    ],
)
def test_request_error_codes_retain_original_diagnostics(error, code) -> None:
    result = LlmSummarizer(client=Reply(error)).summarize("Abstract.")
    assert result["summary_error_code"] == code
    assert str(error) in result["error"]
    if code == "http_429":
        assert result["error"].endswith("\nraw body")


@pytest.mark.parametrize(
    "body,code",
    [
        ({"choices": [{"message": {"content": None}, "finish_reason": "length"}]}, "output_token_limit"),
        ({"choices": [{"message": {"content": None}, "finish_reason": "stop"}]}, "empty_response"),
        ({"choices": []}, "invalid_response"),
    ],
)
def test_api_errors_have_structured_codes_and_preserve_envelope(monkeypatch, body, code) -> None:
    import urllib.request

    from arxiv_summarizer.client import LLMClient

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps(body).encode()

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: Response())
    result = LlmSummarizer(client=LLMClient("https://example.test", "test-only", "model")).summarize("Abstract.")
    assert result["summary_error_code"] == code
    assert result["error"].startswith("ValueError: ")
    assert json.dumps(body) in result["error"]


def test_bart_generation_error_is_coded_without_loading_weights(monkeypatch) -> None:
    from arxiv_summarizer import BartSummarizer, bart

    def broken_tokenizer(*args, **kwargs):
        raise RuntimeError("Original BART diagnostic")

    monkeypatch.setattr(bart, "_get_model", lambda _: (broken_tokenizer, object()))
    result = BartSummarizer().summarize("Abstract.")
    assert result["summary_error_code"] == "generation_error"
    assert result["error"] == "Original BART diagnostic"


def test_pipeline_retains_error_code_and_raw_diagnostic(tmp_path) -> None:
    from scripts.run_test_pipeline import run_pipeline

    config = tmp_path / "interests.yaml"
    config.write_text("- name: quantum\n  keywords: [quantum]\n  weight: 1\n")
    result = run_pipeline(
        [{"id": "a", "title": "quantum", "abstract": "quantum"}], config, LlmSummarizer(client=Reply("not JSON"))
    )[0]
    assert result["summary_error_code"] == "invalid_json"
    assert result["summary_error"] == "Invalid summary response: response is not JSON"


@pytest.mark.parametrize("backend", ["bart", "llm"])
def test_cli_preserves_codes_when_restoring_ids(monkeypatch, capsys, backend) -> None:
    from arxiv_summarizer import cli

    class Backend:
        def summarize_many(self, texts):
            return [
                {
                    "id": "0",
                    "status": "error",
                    "summary": None,
                    "error": "original",
                    "summary_error_code": "generation_error",
                }
            ]

        def summarize_items(self, source):
            return [
                {
                    "id": source[0]["id"],
                    "status": "error",
                    "summary": None,
                    "error": "original",
                    "summary_error_code": "invalid_json",
                }
            ]

    monkeypatch.setattr(cli, "build_backend", lambda args: Backend())
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO('{"id":"real-id","text":"abstract"}\n'))
    cli.main(["--backend", backend])
    result = json.loads(capsys.readouterr().out)
    assert result["id"] == "real-id"
    assert result["error"] == "original"
    assert result["summary_error_code"] == ("generation_error" if backend == "bart" else "invalid_json")


def test_success_does_not_have_an_error_code() -> None:
    result = LlmSummarizer(client=Reply('[{"id":"0","summary":"text"}]')).summarize("Abstract.")
    assert result["status"] == "ok"
    assert "summary_error_code" not in result
