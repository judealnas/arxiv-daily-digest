"""Thin HTTP client for OpenAI-compatible ``/v1/chat/completions``."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from .errors import CodedResponseError


class CompletionText(str):
    """String-compatible completion with allow-listed diagnostic metadata."""

    returned_model: str | None
    finish_reason: str | None

    def __new__(cls, content: str, model: str | None, finish_reason: str | None) -> CompletionText:
        instance = super().__new__(cls, content)
        instance.returned_model = model
        instance.finish_reason = finish_reason
        return instance


class LLMClient:
    """Minimal client for ``POST /v1/chat/completions``.

    Both OpenRouter and opencode-zen speak this wire format, so one client
    parameterised by *base_url* / *api_key* / *model* covers both.

    Parameters
    ----------
    base_url:
        Provider base URL (e.g. ``"https://openrouter.ai/api/v1"``).
    api_key:
        Bearer token for the ``Authorization`` header.
    model:
        Model ID passed in the request body.
    """

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model

    # ── public API ─────────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        max_retries: int = 3,
    ) -> str:
        """Send a chat completion request.

        Returns the response *content* string.

        Raises
        ------
        ValueError
            If the API response is missing expected fields.
        urllib.error.HTTPError
            On HTTP 401, 5xx, or persistent 429 after retries.
        urllib.error.URLError
            On network failures that persist after retries.
        """
        url = f"{self._base_url}/chat/completions"
        payload = self._build_payload(messages, temperature, max_tokens)

        for attempt in range(max_retries):
            req = urllib.request.Request(
                url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < max_retries - 1:
                    time.sleep(min(60, 4 * (2**attempt)))
                    continue
                raise
            except (urllib.error.URLError, OSError):
                if attempt < max_retries - 1:
                    time.sleep(min(30, 2 * (2**attempt)))
                    continue
                raise

            # ── unpack response ─────────────────────────────────────
            choices = body.get("choices")
            if not choices or not isinstance(choices, list):
                raise CodedResponseError(f"Unexpected API response (no choices): {json.dumps(body)}", "invalid_response")
            content = choices[0].get("message", {}).get("content")
            if content is None:
                finish = choices[0].get("finish_reason", "unknown")
                raise CodedResponseError(
                    f"LLM returned empty content (finish_reason={finish}): {json.dumps(body)}",
                    "output_token_limit" if finish == "length" else "empty_response",
                )

            return CompletionText(
                content,
                body.get("model") if isinstance(body.get("model"), str) else None,
                choices[0].get("finish_reason") if isinstance(choices[0].get("finish_reason"), str) else None,
            )

        # Exhausted retries — this path is reached only for network errors.
        raise RuntimeError("Failed after all retries")

    # ── internals ──────────────────────────────────────────────────────────

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        return {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
