#!/bin/sh
# Fetch, sort, summarize with openrouter/free, render, and email a digest.
# Requires a synced uv environment; loads OPENROUTER_API_KEY from Hermes .env.
# SMTP credentials are read by arxiv-email-sender.py from the Hermes .env.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT"

UV=${UV:-uv}
# Fixed-model default; an explicit OPENROUTER_MODEL override remains available.
OPENROUTER_MODEL=${OPENROUTER_MODEL:-recursive/flint-v1.0:free}
TO=${TO:-jude.alnas@duke.edu}
# Resolve once in UTC; explicit dates (including weekends) are passed through.
if [ "$#" -gt 0 ]; then
    LISTING_DATE=$1
else
    TODAY=$(date -u +%F)
    case $(date -u -d "$TODAY" +%u) in
        1) DAYS_AGO=3 ;;
        7) DAYS_AGO=2 ;;
        *) DAYS_AGO=1 ;;
    esac
    LISTING_DATE=$(date -u -d "$TODAY -$DAYS_AGO days" +%F)
fi
# Read only the requested key as data; never execute/source the .env file.
OPENROUTER_API_KEY=$(python3 - <<'PY'
import os
from pathlib import Path

path = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))) / ".env"
try:
    lines = path.read_text(encoding="utf-8").splitlines()
except OSError:
    raise SystemExit(f"Cannot read OpenRouter credentials from {path}")
key = ""
for line in lines:
    line = line.strip()
    if line.startswith("export "):
        line = line[7:].lstrip()
    name, separator, value = line.partition("=")
    if separator and name.strip() == "OPENROUTER_API_KEY":
        key = value.strip()
        if len(key) >= 2 and key[0] == key[-1] and key[0] in "\"'":
            key = key[1:-1]
if not key:
    raise SystemExit(f"OPENROUTER_API_KEY is missing or empty in {path}")
print(key)
PY
)
export OPENROUTER_API_KEY

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
log "Listing date=$LISTING_DATE; strategy=semantic; backend=openrouter; model=$OPENROUTER_MODEL"
STAGE=fetch

printf 'Fetching listings for %s\n' "$LISTING_DATE"
run arxiv-catchup quant-ph "$LISTING_DATE" > "$WORK/fetched.json"

# Validate before loading semantic-scoring or summarization models.
STAGE=fetch-validation
PAPER_COUNT=$(python3 - "$WORK/fetched.json" <<'PY'
import json
import sys
from pathlib import Path
from typing import Any

try:
    papers: Any = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, ValueError) as exc:
    raise SystemExit(f"Invalid fetch output: {exc}") from exc
if not isinstance(papers, list):
    raise SystemExit("Invalid fetch output: expected a JSON array of papers.")
print(len(papers))
PY
)
log "Listing date=$LISTING_DATE; fetched $PAPER_COUNT papers"
if [ "$PAPER_COUNT" -eq 0 ]; then
    log "SKIPPED: no papers found for $LISTING_DATE; no email sent."
    exit 0
fi

STAGE=summarize
printf '%s\n' "Sorting papers and summarizing matches with $OPENROUTER_MODEL"
run python scripts/run_test_pipeline.py \
    "$WORK/fetched.json" \
    "$ROOT/arxiv-interests.yaml" \
    --strategy semantic\
    --backend openrouter \
    --model "$OPENROUTER_MODEL" \
    --papers-per-prompt 1 \
    --user-prompt 'Summarize the following scientific abstracts in up to 5 sentences. Capture significant contributions and quantitative results.' \
    --debug \
    --diagnostics-dir "$WORK/diagnostics" \
    > "$WORK/summarized.json"

STAGE=render
printf '%s\n' "Validating results and rendering HTML"
run arxiv-render "$WORK/summarized.json" \
    --date "$LISTING_DATE" \
    --run-id "${WORK##*/}" \
    --output "$WORK/digest.html"

# Keep delivery policy separate: rendering may succeed with partial summaries.
run python - "$WORK" <<'PY'
import json
import sys
from pathlib import Path


work = Path(sys.argv[1])
papers = json.loads((work / "summarized.json").read_text(encoding="utf-8"))

if not papers:
    raise SystemExit("No papers found; no email sent.")

failures = []
print(f"DEBUG: {len(papers)} papers; {sum(bool(p['matched']) for p in papers)} matched", file=sys.stderr)
for paper in papers:
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

if failures:
    (work / "summary_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    for failure in failures:
        print(f"Summary failed: {json.dumps(failure)}", file=sys.stderr)
    print(f"{len(failures)} summary failures; emailing full abstracts with diagnostics.", file=sys.stderr)

PY

STAGE=send
printf '%s\n' "Sending email"
run python arxiv-email-sender.py \
    --html "$WORK/digest.html" \
    --to "$TO" \
    --subject "arXiv quant-ph daily digest — $LISTING_DATE"
