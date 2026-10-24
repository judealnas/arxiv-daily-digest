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
from pathlib import Path

from .bart import BartSummarizer
from .llm import LlmSummarizer
from .models import SourceItem, dump_result
from .openrouter import OpenRouterSummarizer


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (testable without subprocess)."""
    parser = argparse.ArgumentParser(
        description="Batch summarizer — JSONL in, JSONL out.",
    )
    parser.add_argument(
        "--backend",
        default="bart",
        choices=["bart", "llm", "openrouter"],
        help="Summarizer backend (default: bart; openrouter = one hosted request per text)",
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
        help="llm/openrouter: model ID override (default depends on provider)",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=None,
        help="llm: number of API requests (default: 1, fully batched; -1 = one per text)",
    )
    parser.add_argument(
        "--papers-per-prompt",
        type=int,
        default=None,
        metavar="N",
        help="llm/openrouter: positive number of papers in each request",
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
    user_prompt_options = parser.add_mutually_exclusive_group()
    user_prompt_options.add_argument(
        "--user-prompt",
        default=None,
        help="llm/openrouter: qualitative summary guidance (format remains fixed)",
    )
    user_prompt_options.add_argument(
        "--user-prompt-file",
        default=None,
        metavar="PATH",
        help="llm/openrouter: read qualitative guidance from a UTF-8 file",
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


def _validate_backend_options(args: argparse.Namespace) -> None:
    """Reject backend-specific option combinations before construction."""
    if args.papers_per_prompt is not None and args.papers_per_prompt <= 0:
        raise ValueError("--papers-per-prompt must be positive")
    if args.chunk is not None and args.papers_per_prompt is not None:
        raise ValueError("--chunk cannot be combined with --papers-per-prompt")
    if args.backend == "openrouter" and args.chunk is not None:
        raise ValueError("--chunk is not supported by the dedicated openrouter backend")
    if args.backend == "bart" and args.papers_per_prompt is not None:
        raise ValueError("--papers-per-prompt is not supported by the bart backend")
    if args.backend == "bart" and (args.user_prompt is not None or args.user_prompt_file is not None):
        raise ValueError("--user-prompt/--user-prompt-file are not supported by the bart backend")


def _resolve_user_prompt(args: argparse.Namespace) -> str | None:
    if args.user_prompt_file is None:
        return args.user_prompt
    try:
        return Path(args.user_prompt_file).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError("--user-prompt-file must be a readable UTF-8 file") from exc


def build_backend(args: argparse.Namespace):
    """Construct the configured Summarizer instance from parsed CLI args."""
    _validate_backend_options(args)
    if args.backend == "bart":
        return BartSummarizer(
            model_name=args.bart_model,
            max_length=args.max_length,
            min_length=args.min_length,
            num_beams=args.num_beams,
            batch_size=args.batch_size,
        )
    if args.backend == "openrouter":
        return OpenRouterSummarizer(
            model=args.model,
            user_prompt=_resolve_user_prompt(args),
            papers_per_prompt=args.papers_per_prompt,
            rpm_limit=args.rpm,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            base_url=args.base_url,
            api_key=args.api_key,
        )
    return LlmSummarizer(
        provider=args.provider,
        model=args.model,
        user_prompt=_resolve_user_prompt(args),
        chunk_count=1 if args.chunk is None else args.chunk,
        papers_per_prompt=args.papers_per_prompt,
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
    try:
        _validate_backend_options(args)
    except ValueError as exc:
        parser.error(str(exc))

    # ── read input ──────────────────────────────────────────────────
    source: list[SourceItem] = []
    seen_ids: set[str] = set()
    for line_no, raw in enumerate(sys.stdin, 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            item = json.loads(raw)
            if not isinstance(item, dict):
                raise ValueError("expected a JSON object")
            item_id = item["id"]
            text = item["text"]
            if not isinstance(item_id, str) or not item_id:
                raise ValueError("id must be a nonempty string")
            if not isinstance(text, str):
                raise ValueError("text must be a string")
            if item_id in seen_ids:
                raise ValueError("Duplicate paper ID")
            seen_ids.add(item_id)
            source.append(SourceItem(id=item_id, text=text))
        except (json.JSONDecodeError, KeyError, ValueError) as e:
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
    try:
        backend = build_backend(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.backend in {"llm", "openrouter"}:
        results = backend.summarize_items(source)
        for result in results:
            print(dump_result(result))
        return

    results = backend.summarize_many([item["text"] for item in source])

    # BART returns positional IDs, so restore IDs from the source stream.
    for item, result in zip(source, results, strict=True):
        if result["id"] != item["id"]:
            result = result.copy()
            result["id"] = item["id"]
        print(dump_result(result))


def console_main() -> None:
    """Console-script adapter: pass the actual shell arguments to main()."""
    main(sys.argv[1:])


if __name__ == "__main__":
    console_main()
