"""BART backend: local seq2seq summarization via transformers.

``AutoModelForSeq2SeqLM`` + ``AutoTokenizer`` run locally — no API key,
no network at run time, no chunking or rate limits. One ``generate()``
call per batch; a failing batch marks only its own items
``status="error"`` and the remaining batches continue.

Usage
-----
    from arxiv_summarizer import BartSummarizer

    summarizer = BartSummarizer(max_length=160, min_length=80)
    item = summarizer.summarize("Some abstract ...")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .base import Summarizer
from .models import ResultItem

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# ── defaults ───────────────────────────────────────────────────────────────

DEFAULT_BART_MODEL: str = "facebook/bart-large-cnn"

# ── lazy model cache ───────────────────────────────────────────────────────

_MODEL_CACHE: dict[str, tuple[PreTrainedTokenizerBase, PreTrainedModel]] = {}


def _get_model(model_name: str) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Return the ``(tokenizer, model)`` pair for *model_name*.

    The model is downloaded and loaded on the first call for a given
    *model_name*, ``eval()``-ed, and cached at module level so package
    imports stay cheap and tests can monkeypatch the loaders or the cache.
    """
    if model_name not in _MODEL_CACHE:
        logger.info("Loading BART model %r (first use)", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.eval()
        _MODEL_CACHE[model_name] = (tokenizer, model)
    return _MODEL_CACHE[model_name]


# ── class-based backend ────────────────────────────────────────────────────


class BartSummarizer(Summarizer):
    """BART backend as a configured instance.

    Configuration happens at construction; ``summarize()``/``summarize_many()``
    take no options. ``summarize_many`` overrides the base-class loop with
    true batched generation (``batch_size`` texts per ``generate()`` call).

    Defaults mirror the tuning in ``experiments/summarizer/``:
    ``max_length=130``, ``min_length=40``, ``num_beams=4``, deterministic
    beam search (``do_sample=False``).

    Example
    -------
        summarizer = BartSummarizer(max_length=160, min_length=80)
        item = summarizer.summarize("Some abstract ...")
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_BART_MODEL,
        max_length: int = 130,
        min_length: int = 40,
        num_beams: int = 4,
        batch_size: int = 4,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.min_length = min_length
        self.num_beams = num_beams
        self.batch_size = batch_size

    @property
    def configuration(self) -> dict[str, str | int]:
        """Return safe, serializable provenance without loading model weights."""
        return {"backend": "bart", "model": self.model_name, "max_length": self.max_length,
                "min_length": self.min_length, "num_beams": self.num_beams, "batch_size": self.batch_size}

    def summarize(self, text: str) -> ResultItem:
        """Summarize a single text (one generate() call, batch of one)."""
        return self.summarize_many([text])[0]

    def summarize_many(self, texts: list[str]) -> list[ResultItem]:
        """Summarize texts with batched generation.

        Returns one ResultItem per input, in order. A per-batch failure
        marks only that batch's items ``status="error"``; other batches
        continue.
        """
        tokenizer, model = _get_model(self.model_name)

        results: list[ResultItem] = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start : start + self.batch_size]

            try:
                encoded = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=1024,
                )
                with torch.no_grad():
                    output_ids = model.generate(
                        **encoded,
                        max_length=self.max_length,
                        min_length=self.min_length,
                        num_beams=self.num_beams,
                        do_sample=False,
                    )
                summaries = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            except Exception as exc:
                logger.exception("BART batch at offset %d failed", start)
                results.extend(
                    ResultItem(id=str(start + i), status="error", summary=None, error=str(exc), summary_error_code="generation_error")
                    for i in range(len(batch_texts))
                )
                continue

            results.extend(
                ResultItem(
                    id=str(start + i),
                    status="ok",
                    summary=summary,
                    error=None,
                )
                for i, summary in enumerate(summaries)
            )

        return results
