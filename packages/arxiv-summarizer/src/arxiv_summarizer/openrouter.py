"""OpenRouter summarizer with the same summarize API as BartSummarizer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .llm import LlmSummarizer


class OpenRouterSummarizer(LlmSummarizer):
    """Summarize through OpenRouter.

    By default each text is a separate request; ``papers_per_prompt`` groups
    papers in synchronous requests. All responses must use ID-keyed JSON.
    The fixed system prompt defines the schema; ``user_prompt`` controls only
    qualitative summary guidance. Configure at construction, then call
    ``summarize(text)`` or ``summarize_many(texts)`` like ``BartSummarizer``.
    Credentials come from ``OPENROUTER_API_KEY`` unless ``api_key`` is explicitly
    supplied. ``client`` can be injected for tests; construction makes no request.
    """

    def __init__(
        self,
        *,
        client: Any | None = None,
        model: str | None = None,
        user_prompt: str | None = None,
        papers_per_prompt: int | None = None,
        rpm_limit: int = 0,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        base_url: str | None = None,
        api_key: str | None = None,
        diagnostics_dir: str | Path | None = None,
    ) -> None:
        super().__init__(
            client=client,
            provider="openrouter",
            model=model,
            user_prompt=user_prompt,
            chunk_count=-1,
            papers_per_prompt=papers_per_prompt,
            rpm_limit=rpm_limit,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            api_key=api_key,
            diagnostics_dir=diagnostics_dir,
        )
