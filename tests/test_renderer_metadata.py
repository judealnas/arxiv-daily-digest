from __future__ import annotations

from arxiv_renderer import render_digest
from test_renderer_partial import digest


def test_configuration_header_and_actual_models_are_escaped() -> None:
    data = digest()
    data["papers"][0]["summarizer_config"] = {"backend": "openrouter", "model": "openrouter/free", "api_key": "SECRET"}
    data["papers"][0]["summary_metadata"] = {"requested_model": "openrouter/free", "returned_model": "actual/<model>"}
    data["papers"][1].update(summary_status="ok", summary="Second summary.")
    data["papers"][1]["summary_metadata"] = {"requested_model": "openrouter/free", "returned_model": "other/model"}
    rendered = render_digest(data)
    assert "Summarizer configuration" in rendered
    assert "backend: openrouter" in rendered
    assert "model: openrouter/free" in rendered
    assert "Summarized by: actual/&lt;model&gt;" in rendered
    assert "Summarized by: other/model" in rendered
    assert "SECRET" not in rendered
    assert rendered.index("Summarizer configuration") < rendered.index("Useful summary.")


def test_unavailable_actual_model_is_not_claimed_as_executing_model() -> None:
    data = digest()
    data["papers"][0]["summary_metadata"] = {"requested_model": "router/free", "returned_model": None}
    rendered = render_digest(data)
    assert "Requested model: router/free" in rendered
    assert "Summarized by: router/free" not in rendered


def test_bart_configuration_and_no_metadata_are_supported() -> None:
    data = digest()
    data["papers"][0]["summarizer_config"] = {"backend": "bart", "model": "facebook/bart-large-cnn", "batch_size": 4}
    rendered = render_digest(data)
    assert "backend: bart" in rendered
    assert "facebook/bart-large-cnn" in rendered
    assert "batch_size: 4" in rendered
    assert "None" not in rendered
