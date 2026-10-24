"""Machine-readable failure categories, independent of raw diagnostics."""

from __future__ import annotations

import urllib.error


class CodedResponseError(ValueError):
    """A rejected response with an explicit category and original message."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code: str = code


def request_error_code(exc: Exception) -> str:
    """Classify exception types, never infer causes from arbitrary body text."""
    if isinstance(exc, CodedResponseError):
        return exc.code
    if isinstance(exc, urllib.error.HTTPError):
        return f"http_{exc.code}"
    if isinstance(exc, TimeoutError) or (
        isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, TimeoutError)
    ):
        return "timeout"
    return "request_failed"
