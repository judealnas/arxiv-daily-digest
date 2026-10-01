"""arxiv-summarizer: batch summarizer with pluggable backends (local BART + LLM providers)."""

from __future__ import annotations

from .bart import BartSummarizer
from .base import Summarizer
from .client import LLMClient
from .llm import LlmSummarizer
from .models import ResultItem, SourceItem, dump_result
from .openrouter import OpenRouterSummarizer

__all__ = [
    "BartSummarizer",
    "OpenRouterSummarizer",
    "LlmSummarizer",
    "LLMClient",
    "Summarizer",
    "SourceItem",
    "ResultItem",
    "dump_result",
]
