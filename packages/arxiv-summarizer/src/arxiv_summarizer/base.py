"""Summarizer base class: the core abstraction of this package.

A Summarizer turns one text into one ResultItem. Configuration happens at
construction; summarize() takes no options.

Contract (enforced by tests/contract/test_contract.py):
  - status == "ok" with a non-empty summary, OR
  - status == "error" with a message and summary=None
  - a summary never equals its input verbatim
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .models import ResultItem


class Summarizer(ABC):
    """Base class for all summarizer backends."""

    @abstractmethod
    def summarize(self, text: str) -> ResultItem:
        """Summarize a single text."""

    def summarize_many(self, texts: list[str]) -> list[ResultItem]:
        """Summarize a batch of texts, in order.

        Default: loop over :meth:`summarize`. Backends override this when
        they can exploit batching (BART: padded batches; LLM: chunked
        requests) or need shared state across the run.
        """
        return [self.summarize(text) for text in texts]
