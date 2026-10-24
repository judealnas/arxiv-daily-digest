"""Short deterministic presentation messages; never rewrite raw diagnostics."""

from __future__ import annotations

import re

MESSAGES: dict[str, str] = {
    "invalid_json": "The model returned invalid JSON.",
    "id_mismatch": "The response did not match the requested papers.",
    "invalid_response": "The model returned an invalid response format.",
    "invalid_summary": "The response contained an invalid ID or summary.",
    "timeout": "The summarization request timed out.",
    "output_token_limit": "The model reached its output-token limit without returning a usable summary.",
    "empty_response": "No usable summary was returned.",
    "generation_error": "Local summary generation failed.",
    "request_failed": "The summarization request failed.",
}
LEGACY_REASONS: dict[str, str] = {
    "response is not JSON": "invalid_json",
    "response IDs do not match the requested IDs": "id_mismatch",
    "response contains duplicate IDs": "id_mismatch",
    "response record contains an invalid ID or summary": "invalid_summary",
    "response record has the wrong shape": "invalid_response",
    "response does not contain the requested record count": "invalid_response",
    "duplicate JSON key": "invalid_response",
}


def short_error(code: str | None, raw: str) -> str:
    """Prefer explicit codes; recognize only known legacy diagnostic formats."""
    if code is None:
        prefix: str = "Invalid summary response: "
        if raw.startswith(prefix):
            code = LEGACY_REASONS.get(raw[len(prefix) :])
        elif raw.startswith("TimeoutError:"):
            code = "timeout"
        else:
            status: re.Match[str] | None = re.match(r"^HTTPError: HTTP Error ([1-5][0-9]{2}):", raw)
            if status:
                code = f"http_{status.group(1)}"
        if not raw or raw == "Missing or empty successful summary":
            code = "empty_response"
    if isinstance(code, str) and re.fullmatch(r"http_[1-5][0-9]{2}", code):
        return f"The provider rejected the request: HTTP {code[5:]}."
    return MESSAGES.get(code or "", "Summarization failed; see technical details.")
