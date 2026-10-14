"""Regression tests for the ordinary-test external-load guard."""

from __future__ import annotations

import socket
import sys
import types
import urllib.request

import conftest
import pytest
from arxiv_summarizer import bart
from arxiv_summarizer.bart import BartSummarizer


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        return {"input_ids": list(texts)}

    def batch_decode(self, output_ids, **kwargs):
        return [f"Summary of {text}" for text in output_ids]


class FakeModel:
    def eval(self):
        return self

    def generate(self, input_ids, **kwargs):
        return input_ids


def test_bart_fake_cache_is_reused_within_one_test(monkeypatch):
    """The guard clears cross-test state but preserves normal cache behavior."""
    calls = 0

    def load_tokenizer(model_name):
        nonlocal calls
        calls += 1
        return FakeTokenizer()

    monkeypatch.setattr(bart.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(
        bart.AutoModelForSeq2SeqLM, "from_pretrained", lambda model_name: FakeModel()
    )

    BartSummarizer().summarize("First abstract.")
    BartSummarizer().summarize("Second abstract.")

    assert calls == 1


def test_unmocked_bart_loaders_are_blocked_in_ordinary_tests():
    """The real BART loaders cannot use locally cached weights in ordinary tests."""
    with pytest.raises(RuntimeError, match="Blocked model loading"):
        bart.AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    with pytest.raises(RuntimeError, match="Blocked model loading"):
        bart.AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")


def test_socket_connections_are_blocked_in_ordinary_tests():
    """The guard rejects real network connections before a request is sent."""
    with pytest.raises(RuntimeError, match="Blocked external network access"):
        socket.create_connection(("example.com", 443))


def test_socket_connect_ex_is_blocked_in_ordinary_tests():
    """The alternate TCP connection API is rejected before it can connect."""
    with pytest.raises(RuntimeError, match="Blocked external network access"):
        socket.socket().connect_ex(("example.com", 443))


def test_outbound_udp_and_dns_resolution_are_blocked_in_ordinary_tests():
    """UDP and DNS APIs are rejected before any packet or lookup is attempted."""
    with pytest.raises(RuntimeError, match="Blocked external network access"):
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b"test", ("example.com", 53))
    with pytest.raises(RuntimeError, match="Blocked external network access"):
        socket.getaddrinfo("example.com", 443)


def test_socket_reverse_dns_is_guarded_without_attempting_a_lookup():
    """Reverse DNS is replaced before this test could ask the network anything."""
    assert socket.gethostbyaddr is conftest._blocked_network


def test_mocked_urllib_calls_remain_usable(monkeypatch):
    """Tests may still replace urllib with a fake transport."""
    sentinel = object()
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: sentinel)

    assert urllib.request.urlopen("https://example.test") is sentinel


def test_fake_bart_loaders_remain_usable(monkeypatch):
    """Tests may replace the expensive BART loaders with fake components."""
    fake_tokenizer = FakeTokenizer()
    fake_model = FakeModel()
    monkeypatch.setattr(bart.AutoTokenizer, "from_pretrained", lambda model_name: fake_tokenizer)
    monkeypatch.setattr(bart.AutoModelForSeq2SeqLM, "from_pretrained", lambda model_name: fake_model)

    result = BartSummarizer().summarize("A test abstract.")

    assert result["status"] == "ok"
    assert result["summary"] == "Summary of A test abstract."


def test_sentence_transformer_loading_is_blocked_even_with_an_injected_module(monkeypatch):
    """The semantic strategy cannot construct a model from a cached fake module."""
    from arxiv_interest_sort.strategies import semantic

    calls = 0

    class LoaderSpy:
        def __init__(self, *args, **kwargs):
            nonlocal calls
            calls += 1

    fake_sentence_transformers = types.ModuleType("sentence_transformers")
    fake_sentence_transformers.SentenceTransformer = LoaderSpy
    fake_sentence_transformers.util = object()
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_sentence_transformers)
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))

    with pytest.raises(RuntimeError, match="Blocked model loading"):
        semantic.semantic_score([], [])

    assert calls == 0


def test_sentence_transformer_guard_restores_cached_alias_constructor(monkeypatch):
    """Constructor patching blocks a pre-import alias and is undone after teardown."""

    class SyntheticSentenceTransformer:
        def __init__(self, model_name):
            self.model_name = model_name

    fake_sentence_transformers = types.ModuleType("sentence_transformers")
    fake_sentence_transformers.SentenceTransformer = SyntheticSentenceTransformer
    cached_alias = SyntheticSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_sentence_transformers)

    def import_fake_module(*args, **kwargs):
        return fake_sentence_transformers

    with pytest.MonkeyPatch.context() as guard_monkeypatch:
        guarded_import = conftest._guarded_import(guard_monkeypatch, import_fake_module)
        guarded_import("sentence_transformers")

        with pytest.raises(RuntimeError, match="Blocked model loading"):
            cached_alias("synthetic-model")

    instance = cached_alias("synthetic-model")
    assert instance.model_name == "synthetic-model"
    assert fake_sentence_transformers.SentenceTransformer is SyntheticSentenceTransformer
