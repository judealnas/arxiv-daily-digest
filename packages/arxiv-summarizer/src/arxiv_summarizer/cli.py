"""CLI entry point: JSONL in → JSONL out summarizer pipeline.

Reads ``SourceItem`` lines from **stdin**, constructs the configured
backend, and writes ``ResultItem`` lines to **stdout**.

Backends
--------
    bart (default): local seq2seq generation, no API key.
    llm:            OpenAI-compatible provider (opencode-zen, openrouter).

Examples
--------
    # BART backend (default; local, free):
    cat papers.jsonl | arxiv-summarize > summaries.jsonl

    # LLM backend, fully batched (default chunking):
    cat papers.jsonl | arxiv-summarize --backend llm --provider openrouter

    # One API call per paper (llm):
    cat papers.jsonl | arxiv-summarize --backend llm --chunk -1
"""

from __future__ import annotations

import argparse
import json
import sys

from .bart import BartSummarizer
from .llm import LlmSummarizer
from .models import ResultItem, SourceItem, dump_result


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (testable without subprocess)."""
    parser = argparse.ArgumentParser(
        description="Batch summarizer — JSONL in, JSONL out.",
    )
    parser.add_argument(
        "--backend",
        default="bart",
        choices=["bart", "llm"],
        help="Summarizer backend (default: bart, local; llm = hosted provider)",
    )
    # ── BART options ────────────────────────────────────────────────────
    parser.add_argument(
        "--bart-model",
        default="facebook/bart-large-cnn",
        help="HF model ID for the bart backend (default: facebook/bart-large-cnn)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=130,
        help="bart: max generated summary tokens (default: 130)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=40,
        help="bart: min generated summary tokens (default: 40)",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=4,
        help="bart: beam-search width (default: 4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="bart: texts per generate() call (default: 4)",
    )
    # ── LLM options ─────────────────────────────────────────────────────
    parser.add_argument(
        "--provider",
        default="opencode-zen",
        choices=["opencode-zen", "openrouter"],
        help="llm: provider (default: opencode-zen)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="llm: model ID override (default depends on provider)",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=1,
        help=("llm: number of API requests: 1 (default, fully batched), N chunks, -1 (one per text)"),
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=0,
        metavar="N",
        help="llm: max requests per minute (0 = no limit, default)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="llm: temperature (default: 0.3)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="llm: max completion tokens per request (default: 2048)",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="llm: system prompt override (otherwise built-in default)",
    )
    parser.add_argument(
        "--template",
        default=None,
        help=('llm: user message template with {text} and {count}. Default: "Summarize the following:\\n\\n{text}"'),
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="llm: override provider base URL",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="llm: override API key (otherwise read from env)",
    )
    return parser


def build_backend(args: argparse.Namespace):
    """Construct the configured Summarizer instance from parsed CLI args."""
    if args.backend == "bart":
        return BartSummarizer(
            model_name=args.bart_model,
            max_length=args.max_length,
            min_length=args.min_length,
            num_beams=4,  # not CLI-exposed; experiments fixed it at 4
            batch_size=args.batch_size,
        )
    return LlmSummarizer(
        provider=args.provider,
        model=args.model,
        system_prompt=args.system_prompt,
        template=args.template,
        chunk_count=args.chunk,
        rpm_limit=args.rpm,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        base_url=args.base_url,
        api_key=args.api_key,
    )


def main(argv: list[str] | None = None) -> None:
    """Read JSONL from stdin, emit JSONL to stdout.

    ``argv=None`` (default) means "no CLI arguments" — NOT "read sys.argv".
    This keeps in-process callers (tests, pipelines) isolated from the
    host process's argv. The console-script wrapper passes real argv.
    """
    parser = build_parser()
    args = parser.parse_args([] if argv is None else argv)

    # ── read input ──────────────────────────────────────────────────
    source: list[SourceItem] = []
    for line_no, raw in enumerate(sys.stdin, 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            item = json.loads(raw)
            source.append(
                SourceItem(
                    id=item["id"],
                    text=item["text"],
                )
            )
        except (json.JSONDecodeError, KeyError) as e:
            print(
                json.dumps(
                    {
                        "id": str(line_no - 1),
                        "status": "error",
                        "summary": None,
                        "error": f"Invalid JSONL line {line_no}: {e}",
                    }
                ),
                file=sys.stderr,
            )
            sys.exit(1)

    if not source:
        # Nothing to do — still valid
        return

    # ── construct backend and run ───────────────────────────────────
    backend = build_backend(args)
    results = backend.summarize_many([item["text"] for item in source])

    # ── emit output (ids restored from the source stream) ───────────
    for item, result in zip(source, results, strict=True):
        if result["id"] != item["id"]:
            result = ResultItem(
                id=item["id"],
                status=result["status"],
                summary=result["summary"],
                error=result["error"],
            )
        print(dump_result(result))


if __name__ == "__main__":
    main()
