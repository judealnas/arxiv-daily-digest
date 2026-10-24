from __future__ import annotations

import json

import pytest
from arxiv_summarizer import BartSummarizer, LlmSummarizer, OpenRouterSummarizer
from scripts import run_test_pipeline


@pytest.mark.parametrize(
    "summarizer,expected",
    [
        (BartSummarizer(), {"backend": "bart", "model": "facebook/bart-large-cnn", "batch_size": 4}),
        (LlmSummarizer(api_key="SECRET"), {"backend": "llm", "provider": "opencode-zen", "model": "big-pickle"}),
        (
            OpenRouterSummarizer(api_key="SECRET"),
            {"backend": "openrouter", "provider": "openrouter", "model": "google/gemma-2-27b-it"},
        ),
    ],
)
def test_safe_configuration_without_model_loading_or_requests(summarizer, expected) -> None:
    config = summarizer.configuration
    assert expected.items() <= config.items()
    assert "SECRET" not in json.dumps(config)
    assert "api_key" not in config
    assert "base_url" not in config


def test_runner_retains_config_and_completion_metadata(tmp_path) -> None:
    path = tmp_path / "interests.yaml"
    path.write_text("- name: quantum\n  keywords: [quantum]\n  weight: 1\n")

    class Fake:
        configuration = {"backend": "test", "model": "configured"}

        def summarize_items(self, items):
            return [
                {
                    "id": item["id"],
                    "status": "ok",
                    "summary": "Summary.",
                    "error": None,
                    "summary_metadata": {"requested_model": "configured", "returned_model": "actual"},
                }
                for item in items
            ]

    output = run_test_pipeline.run_pipeline([{"id": "a", "title": "quantum", "abstract": "quantum"}], path, Fake())
    assert output[0]["summarizer_config"] == Fake.configuration
    assert output[0]["summary_metadata"]["returned_model"] == "actual"
