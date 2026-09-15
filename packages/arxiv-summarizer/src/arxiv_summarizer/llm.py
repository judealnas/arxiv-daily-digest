"""LLM backend: provider-based summarization as a configured instance.

Speaks the OpenAI-compatible ``/v1/chat/completions`` wire format via an
injectable client (``LLMClient`` or any object with ``chat(messages,
**kw) -> str``). Tests inject fakes — real API calls are never made from
tests.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Protocol

from .base import Summarizer
from .client import LLMClient
from .models import ResultItem

# ── provider registry ──────────────────────────────────────────────────────

_PROVIDER_CONFIGS: dict[str, dict[str, str]] = {
    "opencode-zen": {
        "base_url": "https://opencode.ai/zen/v1",
        "env_key": "OPENCODE_ZEN_API_KEY",
        "default_model": "big-pickle",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
        "default_model": "google/gemma-2-27b-it",
    },
}

DEFAULT_SYSTEM_PROMPT: str = (
    "You are a helpful assistant that summarizes text concisely. Respond with only the summary, no extra commentary."
)

DEFAULT_TEMPLATE: str = "Summarize the following:\n\n{text}"


class ChatClient(Protocol):
    """Minimal protocol the LLM backend needs (satisfied by LLMClient)."""

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str: ...


class LlmSummarizer(Summarizer):
    """LLM backend as a configured instance.

    Configuration happens at construction; ``summarize()`` takes no options.
    ``summarize_many`` issues chunked multi-output requests —
    ``chunk_count=1`` (default) sends one API call for the whole batch.

    The HTTP client may be injected (``client=``) for testing; otherwise
    an ``LLMClient`` is built lazily from the provider config on first use.
    """

    def __init__(
        self,
        *,
        client: Any | None = None,
        provider: str = "opencode-zen",
        model: str | None = None,
        system_prompt: str | None = None,
        template: str | None = None,
        chunk_count: int = 1,
        rpm_limit: int = 0,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._client = client
        self._provider = provider
        self._model = model
        self._system_prompt = system_prompt
        self._template = template
        self._chunk_count = chunk_count
        self._rpm_limit = rpm_limit
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._base_url = base_url
        self._api_key = api_key

    # ── client resolution ───────────────────────────────────────────

    def _get_client(self) -> Any:
        """Return the injected client, or build one from provider config."""
        if self._client is None:
            url, key, model = self._resolve_provider()
            self._client = LLMClient(url, key, model)
        return self._client

    def _resolve_provider(self) -> tuple[str, str, str]:
        """Resolve (base_url, api_key, model) from provider shorthand + overrides.

        Raises
        ------
        ValueError
            If the provider is unknown or no API key can be found.
        """
        cfg = _PROVIDER_CONFIGS.get(self._provider)
        if cfg is None:
            raise ValueError(f"Unknown provider {self._provider!r}.  Known: {list(_PROVIDER_CONFIGS)}")

        resolved_url = self._base_url or cfg["base_url"]
        resolved_key = self._api_key or os.environ.get(cfg["env_key"]) or ""
        resolved_model = self._model or cfg["default_model"]

        if not resolved_key:
            raise ValueError(
                f"No API key for provider {self._provider!r}. "
                f"Set the {cfg['env_key']!r} environment variable or pass ``api_key=``."
            )

        return resolved_url, resolved_key, resolved_model

    # ── Summarizer interface ────────────────────────────────────────

    def summarize(self, text: str) -> ResultItem:
        """Summarize a single text (one chat request)."""
        return self.summarize_many([text])[0]

    def summarize_many(self, texts: list[str]) -> list[ResultItem]:
        """Summarize texts via chunked chat requests (default: 1 call for all).

        Per-chunk failures mark only that chunk's items ``status="error"``;
        other chunks continue.
        """
        client = self._get_client()
        sys_prompt = self._system_prompt or DEFAULT_SYSTEM_PROMPT
        tmpl = self._template or DEFAULT_TEMPLATE

        results: list[ResultItem] = []
        for pos, chunk in enumerate(self._build_chunks(texts)):
            # ── rate-limit pacing ───────────────────────────────────
            if self._rpm_limit > 0 and pos > 0:
                time.sleep(60.0 / self._rpm_limit)

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": self._build_user_message(tmpl, chunk)},
            ]
            try:
                response = client.chat(messages, temperature=self._temperature, max_tokens=self._max_tokens)
            except Exception as exc:
                results.extend(
                    ResultItem(id=f"chunk{pos}-{i}", status="error", summary=None, error=str(exc))
                    for i in range(len(chunk))
                )
                continue
            for i, summary in enumerate(self._parse_multi_output(response, len(chunk))):
                results.append(ResultItem(id=f"chunk{pos}-{i}", status="ok", summary=summary, error=None))

        return results

    # ── chunking ────────────────────────────────────────────────────

    def _build_chunks(self, texts: list[str]) -> list[list[str]]:
        """Divide *texts* into ``chunk_count`` roughly-equal chunks.

        ``chunk_count <= 0`` or ``> len(texts)`` ⇒ one text per chunk.
        ``chunk_count == 1`` (default) ⇒ all texts in a single chunk.
        """
        n = len(texts)
        if n == 0:
            return []

        chunk_count = self._chunk_count
        if chunk_count <= 0 or chunk_count > n:
            chunk_count = n

        base = n // chunk_count
        remainder = n % chunk_count
        chunks: list[list[str]] = []
        start = 0
        for i in range(chunk_count):
            size = base + (1 if i < remainder else 0)
            chunks.append(texts[start : start + size])
            start += size
        return chunks

    # ── prompt building ─────────────────────────────────────────────

    @staticmethod
    def _build_user_message(template: str, chunk_texts: list[str]) -> str:
        """Render the user message from *template* and one chunk of texts.

        Single-text chunks format ``{text}`` directly; multi-text chunks
        get numbered blocks plus per-line output instructions so the
        parser can split the response.
        """
        count = len(chunk_texts)
        if count == 1:
            return template.format(text=chunk_texts[0], count=count)

        numbered = "\n---\n".join(f"Text {i + 1}: {t}" for i, t in enumerate(chunk_texts))
        body = template.format(text=numbered, count=count)
        body += (
            f"\n\nFor each of the {count} texts above, write a concise summary "
            f"(1-2 sentences) on a separate line. "
            f"Output exactly {count} summaries, one per line, in order."
        )
        return body

    # ── output parsing ──────────────────────────────────────────────

    @staticmethod
    def _parse_multi_output(response: str, expected_count: int) -> list[str]:
        """Extract *expected_count* summaries from a single LLM response.

        Strategy (tried in order):
        1. numbered lines (``Summary N: …``, ``N. …``, ``N: …``)
        2. exact line-count match
        3. take first N lines / pad with empties
        """
        lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
        if not lines:
            return [""] * expected_count

        numbered_pattern = re.compile(r"^(?:Summary\s*)?(\d+)[:.)]\s*(.*)", re.IGNORECASE | re.DOTALL)
        parsed: list[tuple[int, str]] = []
        for line in lines:
            m = numbered_pattern.match(line)
            if m:
                parsed.append((int(m.group(1)), m.group(2).strip()))

        if len(parsed) >= expected_count:
            parsed.sort(key=lambda x: x[0])
            return [s for _, s in parsed[:expected_count]]

        if len(lines) == expected_count:
            return lines

        if len(lines) > expected_count:
            return lines[:expected_count]
        return lines + [""] * (expected_count - len(lines))
