"""Render sort-and-summarize JSON as HTML; no inference, network, or email."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import render_digest
from .input import prepare_digest


def main(argv: list[str] | None = None) -> int:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="-", help="JSON file, or - for stdin (default)")
    parser.add_argument("--date", help="Listing date YYYY-MM-DD; required for array input")
    parser.add_argument("--run-id", help="Optional source run identifier")
    parser.add_argument("--output", "-o", default="-", help="HTML file, or - for stdout (default)")
    parser.add_argument("--notice", help="Additional visible diagnostic notice")
    args: argparse.Namespace = parser.parse_args(argv)
    try:
        text: str = sys.stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf-8")
        digest = prepare_digest(json.loads(text), date=args.date, run_id=args.run_id)
        rendered: str = render_digest(digest, notice=args.notice)
        if args.output == "-":
            sys.stdout.write(rendered)
        else:
            Path(args.output).write_text(rendered, encoding="utf-8")
    except json.JSONDecodeError as exc:
        print(f"error: Invalid JSON: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, UnicodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
