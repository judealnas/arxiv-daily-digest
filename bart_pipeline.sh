#!/bin/sh
# Fetch, sort, summarize locally with BART, render, and email a digest.
# Requires a synced uv environment and BART model weights (downloaded on first use).
# SMTP credentials are read by arxiv-email-sender.py from the Hermes .env.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT"

UV=${UV:-uv}
TO=${TO:-jude.alnas@duke.edu}
LISTING_DATE=${1:-$(date -u -d yesterday +%F)}
exec 9>"$ROOT/.digest.lock"
flock -n 9 || {
    printf '%s\n' "Another digest run is already active." >&2
    exit 1
}

CACHE_ROOT=${TMPDIR:-$HOME/.cache}
mkdir -p "$CACHE_ROOT"
umask 077
WORK=$(mktemp -d "$CACHE_ROOT/arxiv-digest.XXXXXXXX")
STAGE=initialization
log() {
    printf '%s [%s] %s\n' "$(date -u +%FT%TZ)" "$STAGE" "$*" | tee -a "$WORK/debug.log" >&2
}
cleanup() {
    status=$?
    trap - 0
    if [ "$status" -eq 0 ]; then
        rm -rf "$WORK"
    else
        log "FAILED (exit $status). Artifacts preserved at $WORK"
    fi
    exit "$status"
}
trap cleanup 0
trap 'exit 1' HUP INT TERM

run() {
    log "Starting: $*"
    status=0
    "$UV" run --frozen --no-sync "$@" 2>"$WORK/$STAGE.stderr.log" || status=$?
    cat "$WORK/$STAGE.stderr.log" >&2
    log "Finished (exit $status); stderr: $WORK/$STAGE.stderr.log"
    return "$status"
}
log "Listing date=$LISTING_DATE; strategy=semantic; backend=bart; model=facebook/bart-large-cnn"
STAGE=fetch

printf 'Fetching listings for %s\n' "$LISTING_DATE"
run arxiv-catchup quant-ph "$LISTING_DATE" > "$WORK/fetched.json"

STAGE=summarize
printf '%s\n' "Sorting papers and summarizing matches locally with BART"
run python scripts/run_test_pipeline.py \
    "$WORK/fetched.json" \
    "$ROOT/arxiv-interests.yaml" \
    --strategy semantic \
    --backend bart \
    --model facebook/bart-large-cnn \
    --debug \
    > "$WORK/summarized.json"

STAGE=render
printf '%s\n' "Validating results and rendering HTML"
run python - "$WORK" "$LISTING_DATE" <<'PY'
import json
import sys
from pathlib import Path

from arxiv_digest_renderer import render_digest

work = Path(sys.argv[1])
listing_date = sys.argv[2]
papers = json.loads((work / "summarized.json").read_text(encoding="utf-8"))

if not papers:
    raise SystemExit("No papers found; no email sent.")

summaries = {}
failures = []
print(f"DEBUG: {len(papers)} papers; {sum(bool(p['matched']) for p in papers)} matched", file=sys.stderr)
for paper in papers:
    paper["url"] = paper["abstract_link"]
    if paper["matched"]:
        summary = paper.get("summary")
        if (
            paper.get("summary_status") != "ok"
            or not isinstance(summary, str)
            or not summary.strip()
        ):
            failures.append({"id": paper["id"], "status": paper.get("summary_status"),
                             "error": paper.get("summary_error", "Missing or empty summary")})
            continue
        summaries[paper["id"]] = summary

if failures:
    (work / "summary_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    for failure in failures:
        print(f"Summary failed: {json.dumps(failure)}", file=sys.stderr)
    raise SystemExit(f"{len(failures)} summary failures; no email sent.")

digest = {
    "date": listing_date,
    "run_id": work.name,
    "count": len(papers),
    "matched_count": sum(bool(p["matched"]) for p in papers),
    "papers": papers,
}
(work / "digest.html").write_text(
    render_digest(digest, summaries=summaries), encoding="utf-8"
)
PY

STAGE=send
printf '%s\n' "Sending email"
run python arxiv-email-sender.py \
    --html "$WORK/digest.html" \
    --to "$TO" \
    --subject "arXiv quant-ph daily digest — $LISTING_DATE"
