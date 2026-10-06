"""Global isolation for tests that are not explicitly integration tests."""

from __future__ import annotations

import builtins
import socket
import sys

import pytest


def _blocked_network(*args, **kwargs):
    raise RuntimeError("Blocked external network access in an ordinary test")


def _blocked_model_load(*args, **kwargs):
    raise RuntimeError("Blocked model loading in an ordinary test")


def _block_sentence_transformer_constructor(monkeypatch, sentence_transformers):
    """Block construction without replacing the module or invalidating class aliases."""
    sentence_transformer = getattr(sentence_transformers, "SentenceTransformer", None)
    if isinstance(sentence_transformer, type):
        monkeypatch.setattr(sentence_transformer, "__init__", _blocked_model_load)


def _guarded_import(monkeypatch, original_import):
    """Guard a newly imported optional model library for this test only."""

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        module = original_import(name, globals, locals, fromlist, level)
        if name == "sentence_transformers" or name.startswith("sentence_transformers."):
            sentence_transformers = sys.modules.get("sentence_transformers")
            if sentence_transformers is not None:
                _block_sentence_transformer_constructor(monkeypatch, sentence_transformers)
        return module

    return guarded_import


@pytest.fixture(autouse=True)
def block_external_loads_in_ordinary_tests(monkeypatch, request):
    """Block real transport and model loading unless a test opts in explicitly."""
    if request.node.get_closest_marker("slow") or request.node.get_closest_marker("live"):
        yield
        return

    from arxiv_summarizer import bart

    # A model loaded by an earlier test must not make an ordinary test pass.
    bart._MODEL_CACHE.clear()

    monkeypatch.setattr(socket, "create_connection", _blocked_network)
    monkeypatch.setattr(socket.socket, "connect", _blocked_network)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked_network)
    monkeypatch.setattr(socket.socket, "sendto", _blocked_network)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked_network)
    monkeypatch.setattr(socket, "gethostbyname", _blocked_network)
    monkeypatch.setattr(socket, "gethostbyname_ex", _blocked_network)
    monkeypatch.setattr(socket, "gethostbyaddr", _blocked_network)
    monkeypatch.setattr(socket, "getnameinfo", _blocked_network)
    monkeypatch.setattr(bart.AutoTokenizer, "from_pretrained", _blocked_model_load)
    monkeypatch.setattr(bart.AutoModelForSeq2SeqLM, "from_pretrained", _blocked_model_load)
    sentence_transformers = sys.modules.get("sentence_transformers")
    if sentence_transformers is not None:
        _block_sentence_transformer_constructor(monkeypatch, sentence_transformers)
    monkeypatch.setattr(
        builtins,
        "__import__",
        _guarded_import(monkeypatch, builtins.__import__),
    )
    try:
        yield
    finally:
        bart._MODEL_CACHE.clear()
