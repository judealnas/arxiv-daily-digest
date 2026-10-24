# Product specification: arXiv daily digest

## Purpose and status

Help researchers find relevant arXiv papers and read short summaries without losing track of which paper each summary describes.

This document describes the project as it exists, not a finished product we hope to build. It was written retrospectively against commit `b305805`. The repository contains usable components, an incomplete top-level pipeline, and a packaged renderer. It is not yet a verified replacement for the separately operated daily digest service.

For technical details, see [the design document](design.md). Future changes should update these two documents rather than create a separate permanent specification for every feature.

## Who it is for

- Researchers who want papers ranked against their own research interests.
- Users comfortable running Python commands and connecting tools with files.
- Maintainers who need to understand failures before trusting automated delivery.

## What users can do now

### 1. Retrieve paper information

The catchup component can fetch an arXiv catchup listing or parse saved listing HTML. It returns paper IDs, titles, authors, abstracts, and links, with optional comments and subject information.

Saved HTML allows repeatable testing without contacting arXiv. The parser depends on the listing's HTML structure; it is not a general-purpose web scraper. The top-level fetch-and-rank command accepts either a fetched listing or saved HTML and ranks it using the same list-based interests configuration as the sorter.

Listing dates are resolved once. A fetched listing defaults to the current UTC calendar date, and `--days-ago N` subtracts N calendar days, including weekends. Users may supply the existing positional date for fetches or `--date YYYY-MM-DD`. Metadata records that resolved listing date. Saved HTML has no trusted page-date extraction: it keeps the existing plain-paper output without a date, while metadata output requires explicit `--date`.

### 2. Rank papers by interest

Users describe topics in a YAML interests file. The sorter supports:

- **Keyword ranking:** find configured words or phrases in titles and abstracts and apply topic weights.
- **Meaning-based ranking:** use a local embedding model to compare papers with topic descriptions built from topic names and keywords.

The sorter retains paper records and adds scores and match information. Matched papers come first, then higher scores, with paper IDs used to break ties. Ranking does not discard unmatched papers.

Meaning-based ranking requires additional model dependencies and may need a model download.

### 3. Summarize supplied text

Users can choose local BART summarization or a hosted language-model backend, including a dedicated OpenRouter backend.

The summarizer command reads JSONL: one JSON record per line, containing a paper `id` and its `text`. It writes one result per input, with the ID, a success/error status, and either a summary or an error message.

The caller currently chooses which papers to summarize. The command does not directly consume the sorter's full paper array or automatically select matched papers.

### 4. Control summary style without controlling the output format

For hosted models, users may provide plain-language guidance such as:

> Use five sentences. Emphasize numerical results. Explain the main limitation.

Guidance can be supplied inline or through a UTF-8 text file. It is literal text, not a template with placeholders. Users cannot replace the structural system prompt through the supported interface.

The fixed system prompt defines the input and response formats. The application still validates the response: instructions alone are not a guarantee that a model will obey.

BART does not support this guidance interface.

### 5. Put several papers in one hosted request

**Prompt grouping** means sending several papers in one model request. Users can set a maximum number of papers per prompt.

Every paper retains its exact ID. A valid response may arrive in a different order; the application restores input order by matching IDs.

If a response omits a paper, repeats an ID, adds an unknown ID, or has an invalid format, the entire group receives error results. Later groups can still succeed. The application does not guess which summary belongs to which paper.

This is different from asynchronous provider batch jobs, which are not implemented.

### 6. Render HTML with visible summary failures

`uv run arxiv-render summarized.json --date YYYY-MM-DD --output digest.html`
formats the sort-and-summarize runner's output. Stdin/stdout and dated digest
objects are also supported. All papers remain in supplied order, with match
stars and no author lists. Failed summaries use full abstracts, show a short error
below the paper, and preserve the complete raw diagnostic in an expanded,
smaller quote block with inline shading and a border. These blocks do not
require collapsible-element support. A warning at the top reports unavailable
summaries; successful summaries are retained. Short messages come from structured
error codes, not LLM summarization. Saved results without codes remain readable. Unmatched papers show their first abstract sentence.

When retained, summarizer backend/model/configuration appears at the top and
actual returned models appear below successful summaries. Requested aliases
are distinguished from verified returned models. Missing metadata is not guessed.
The renderer does not send email or change live scheduling. Existing shell
wrappers email partial digests with visible fallback diagnostics.

## Important limits

- Correct IDs and valid JSON do not prove that a summary is factually correct. Scientific claims and numerical results still need evaluation.
- Hosted summarization sends supplied text to the configured external service. Costs, availability, and provider data policies are outside this project's control. No free model is promised by this specification.
- There is no automatic input-token budget for hosted groups. Large groups can exceed model limits or leave too little room for complete answers.
- BART truncates long inputs. It needs model weights and local compute; it is not equivalent to the hosted models in quality or speed.
- Configuration failures can stop a run before per-paper results are produced. Not every failure is a per-paper error.
- The top-level command rejects invalid or conflicting date controls before it fetches or writes. A standalone email sender exists, but it does not validate digest freshness or prevent duplicate delivery. Its dry run currently still requires credentials.
- The renderer consumes sorted/summarized arrays or dated digest objects; it does not establish email freshness, duplicate-send protection, or scientific summary accuracy.

## Acceptance criteria for existing components

These are observable behaviors to preserve. They are not claims that the future complete pipeline has been built.

- **AC-1 — Parsing:** given a supported saved catchup page, parsing produces the expected paper records and links.
- **AC-2 — Ranking:** given papers and interests, ranking retains the papers, adds match information, and puts matched papers before unmatched papers.
- **AC-3 — Summary results:** given valid summarizer input, results retain the input IDs and order and expose success or failure explicitly.
- **AC-4 — Fixed structure:** changing hosted summary guidance does not replace the fixed structural system message; single-paper and grouped requests use the same response format.
- **AC-5 — No ambiguous assignment:** missing, duplicated, unknown, empty, or malformed hosted results are rejected rather than attached to papers by position.
- **AC-6 — Failure isolation:** a hosted request or response failure marks its group as failed without discarding successful independent groups.
- **AC-7 — Early input checks:** malformed CLI records, duplicate IDs, invalid grouping options, and unreadable guidance files fail before a hosted request is sent.
- **AC-8 — Provider request construction:** mocked HTTP tests verify the configured model, messages, endpoint, and request parameters without paid inference.
- **AC-9 — HTML rendering:** file and stdin inputs produce complete, escaped HTML; partial summary failures remain visible without losing successful summaries or paper entries. Invalid inputs fail without overwriting existing output.
- **AC-10 — Provenance:** available safe summarizer configuration is displayed at the top; actual returned models are attributed per summary without misrepresenting requested aliases.

Test locations and the limits of the evidence are listed in the design document.

## Future direction — not implemented promises

The desired complete digest workflow is:

1. Fetch and rank papers, then save a dated source file and a complete readable fallback.
2. Optionally summarize matched papers without endangering that fallback.
3. Validate that summaries belong to the same run and cover all required papers.
4. Render a complete digest or clearly explain why the fallback was used.
5. Deliver only the intended dated result, with explicit protection against stale and duplicate email.

Before implementation, decide the scheduling approach, default summary backend/model, and what a zero-paper day should produce. A live free-model smoke test is proposed, not completed evidence.

## Non-goals for the current version

- Asynchronous provider batch submission and polling.
- Reading full PDFs or independently checking scientific claims.
- Guaranteed summary quality from any model.
- A graphical user interface.
- Automatic migration of the existing scheduled service.

Changing live jobs, selecting a paid model, or sending real email requires a separate, explicit operational decision. This documentation does not authorize those actions.
