# Design: arXiv daily digest

## Scope and reading guide

This is a retrospective description of the code at commit `b305805`, with local draft files clearly identified. It explains how the current components work and where they do not yet connect. It does not describe the live scheduled service as if it were part of this repository.

Read [the product specification](product.md) for user-facing behavior and acceptance criteria.

**Terms:** a backend is an interchangeable implementation; JSON is structured text; JSONL stores one JSON record per line; an artifact is a saved output file; a run ID identifies one execution.

## 1. Overall structure

The project is a Python 3.11+ uv workspace containing three packages:

- `packages/arxiv-catchup`: download and parse catchup listings.
- `packages/arxiv-interest-sort`: load research interests, score papers, and sort them.
- `packages/arxiv-summarizer`: provide local and hosted summarization behind a shared result interface.

The connections currently look like this:

```text
arXiv catchup HTML or saved HTML
    → catchup parser
    → paper records
    → interest sorter
    → ranked paper records

Caller-supplied {id, text} records
    → summarizer CLI
    → local BART OR hosted model
    → {id, status, summary, error} records

Separately supplied HTML file
    → standalone email sender
    → SMTP email
```

There is no complete connection from ranked papers to summaries, validated HTML, and safe delivery. The top-level `arxiv-daily-digest.py` supports fixture-backed fetch or saved-HTML parsing, list-based interest configuration, and JSON output, but it is not an end-to-end delivery workflow.

## 2. Retrieval and ranking

### Paper records

The parser returns a JSON-compatible list of records with these main fields:

```text
id, title, authors, abstract, abstract_link, pdf_link, comments, subjects
```

It preserves a version suffix in the paper ID when present, but removes it from generated abstract/PDF links. Consumers must not assume that the ID and the last part of a link are identical.

The parser uses patterns tailored to catchup HTML. Missing optional fields can become empty values. An empty parse is not by itself proof that there were no papers; it may also indicate unsupported page structure.

The fetcher has bounded retry behavior for HTTP 429. It is not a general recovery layer for all HTTP failures.

The top-level command resolves a listing date before it reads, fetches, or writes output. Fetched runs use an explicit positional date or `--date`; otherwise they use the current UTC calendar date minus `--days-ago` calendar days. It rejects negative offsets and conflicting date controls before transport. The resolved date is used both in the catchup URL and in metadata. Saved HTML does not have a trusted date source: plain output remains undated, and `--include-metadata` requires explicit `--date`.

### Interest scoring

`load_interest_config()` returns a list of topic dictionaries. Keyword scoring searches titles and abstracts, adding each matching keyword's topic weight. Meaning-based scoring computes model embeddings, compares papers with topics, and uses weighted similarity. Topic embeddings can be cached; paper embeddings are computed for the supplied papers.

Sorting keeps matched records first, then sorts by descending score and version-stripped ID. The scoring functions add fields to the existing records rather than replacing paper identity.

## 3. Summarizer interfaces

All backends expose `summarize(text)` and `summarize_many(texts)`. Hosted backends also expose `summarize_items(items)` so callers can supply real paper IDs directly.

The CLI input is JSONL, not the full paper-array format emitted by retrieval and sorting:

```json
{"id":"paper-a","text":"The abstract to summarize."}
```

The public output is also JSONL:

```json
{"id":"paper-a","status":"ok","summary":"A short summary.","error":null}
```

On a failed hosted group, each affected result has `status: "error"`, `summary: null`, and an error message. Configuration and input-validation failures can instead stop execution before a request.

For text-only library callers, IDs are generated from input positions. The CLI sends real IDs directly to hosted backends. For BART, it restores source IDs from the ordered, separately generated outputs.

**Decision:** preserve a shared result interface without forcing BART to generate JSON. BART produces separate outputs in order; a hosted model produces one response that may contain several summaries. Those two cases need different identity handling.

## 4. Fixed structural prompt and user guidance

Hosted requests contain exactly two chat messages:

1. A hard-coded system message describes the input fields and required response structure.
2. A user message contains a JSON object with qualitative guidance and paper records.

Illustrative user-message content:

```json
{
  "guidance": "Use five sentences. Emphasize numbers.",
  "papers": [
    {"id": "paper-a", "text": "First abstract."},
    {"id": "paper-b", "text": "Second abstract."}
  ]
}
```

The model must return only an array of `{id, summary}` objects. The application then converts these into the richer public result records containing status and error fields.

The same response contract applies to one-paper and multi-paper requests. User guidance is literal text: braces are not substituted, and there is no supported system-prompt override. CLI guidance comes from `--user-prompt` or the mutually exclusive `--user-prompt-file`.

**Decision:** separate style from structure. An unrestricted formatting template makes downstream parsing fragile. Serializing guidance and papers together also avoids relying on text delimiters that user guidance might imitate. Neither measure guarantees obedience; validation remains necessary.

## 5. Response validation and failures

The hosted parser reads the entire response as JSON. It rejects:

- Prose before or after JSON, Markdown fences, and invalid JSON.
- Wrong root types, wrong record fields, and duplicate JSON object keys.
- Missing, repeated, or unknown paper IDs.
- Non-string or blank summaries.

The returned ID set must exactly equal the requested set. Valid summaries are returned in input order, regardless of response order.

**Decision:** reject an invalid group as a whole. Guessing from numbered lines, trimming extra results, or filling gaps can silently assign a summary to the wrong paper. This design prefers an explicit failure over a plausible but untrustworthy assignment.

Request exceptions inside the hosted group loop become safe diagnostics, such as an HTTP status or a generic request failure. Later groups continue. Initial provider/key resolution occurs before that loop and can raise a configuration error.

The low-level HTTP client still has weaknesses: some malformed response shapes are not handled consistently, and its direct exceptions may contain response data. The higher-level sanitizer does not make every direct use of that client safe.

## 6. Three different meanings of batching

- **Prompt grouping:** several papers inside one synchronous hosted request. `papers_per_prompt` sets the group-size cap.
- **Local inference batching:** BART processes several separate inputs in one model-generation call.
- **Provider batch processing:** submit asynchronous jobs and retrieve results later. This is not implemented.

The generic LLM backend retains its legacy default: all supplied papers in one request. The dedicated OpenRouter backend defaults to one paper per request. Explicit `papers_per_prompt` overrides library grouping behavior. The CLI rejects combining it with legacy `--chunk`; the dedicated OpenRouter backend rejects `--chunk` entirely.

There is no token-aware input sizing. A completion-token limit constrains the answer, not the input. Large groups and truncated responses remain operational risks.

## 7. Local and remote execution

### BART

BART uses Transformers and PyTorch. Model weights are loaded lazily and cached in the process; first use may download them. Inputs are truncated at the configured tokenizer call's 1,024-token limit. Generation is deterministic beam search rather than sampling. A generation failure affects its local batch; initial model-loading failures occur outside that batch handler.

### Hosted models

`LLMClient` sends synchronous chat-completions HTTP requests. Provider configuration supplies the endpoint, credential environment-variable name, and a default model; callers may select an explicit model.

The client retries HTTP 429 and selected network failures with bounded waits. It does not currently retry every server error or provide a durable request record. Retrying after a network failure may repeat work already received by a provider.

Use environment-based credentials rather than command-line secrets. Hosted input leaves the machine. Availability, prices, and model quality must be checked separately for any real evaluation; this document does not endorse a current free model.

## 8. Draft rendering and standalone delivery

The untracked `arxiv_digest_renderer.py` draft expects a different envelope containing `date`, `run_id`, counts, and `papers`, with each paper using a `url` field. It renders records in supplied order, marks matches, uses full cleaned matched abstracts when no summaries are supplied, and uses short excerpts for unmatched papers.

This is not the parser's direct output format. There is no integrated adapter, and the draft does not validate run identity, counts, summary completeness, or safe URL schemes. HTML escaping is not a substitute for URL validation. Its heading is also fixed to quant-ph.

The committed `arxiv-email-sender.py` reads an existing HTML file and sends HTML plus plain text using SMTP with STARTTLS. It is a standalone helper, not a safe publication stage. It lacks freshness checks and duplicate-send protection, derives its default subject date independently, and checks credentials even for a dry run. This date work does not change the sender because it is not integrated with the top-level command. No live email was sent for this documentation work.

## 9. Confirmed gaps, not new requirements silently implemented

1. **Missing integration:** ranked paper arrays need an explicit adapter to summarizer records and a validated renderer input.
2. **Missing publication safety:** versioned artifacts, atomic writes, full summary coverage checks, and safe email selection are not an integrated workflow.
3. **Ordinary-test isolation boundary:** root-level ordinary tests block network sockets and model loading; tests requiring either must opt in with `slow` or `live`. This guard does not make untested integrations safe.

These findings are documented, not fixed by this change.

## 10. Verification and requirements traceability

Existing evidence maps to the product criteria as follows:

- **AC-1:** `packages/arxiv-catchup/tests/test_fetcher.py` and `test_cli.py`.
- **AC-2:** `packages/arxiv-interest-sort/tests/keyword/`, `semantic/`, and `test_cli.py`.
- **AC-3:** summarizer `tests/contract/`, `test_cli_backend.py`, and `test_openrouter.py`.
- **AC-4:** summarizer `tests/test_prompt_contract.py` and guidance-file tests in `test_cli_backend.py`.
- **AC-5:** summarizer `tests/test_llm_grouping.py` and `test_prompt_contract.py`.
- **AC-6:** failed-request and failed-group tests in `test_openrouter.py` and `test_llm_grouping.py`.
- **AC-7:** summarizer `tests/test_cli_backend.py` and input-ID tests in `test_llm_grouping.py`.
- **AC-8:** `test_openrouter_http_request_matches_chat_completions_contract` in summarizer `tests/test_openrouter.py`.

Most paths above are relative to their named package. These tests establish component behavior, not scientific accuracy or end-to-end delivery safety.

### Test isolation and offline coverage

Root-level ordinary tests use `conftest.py` to block real socket connections and model loading. Tests that need either must be explicitly marked `slow` or `live`; mocks and fake transports remain usable in ordinary tests.

`tests/test_daily_digest_cli.py` exercises the top-level command with saved catchup HTML and fake transport. It covers list-based interest configuration, saved-HTML metadata dates, fetched positional dates, UTC defaults, calendar-day offsets, and invalid controls before transport. This is fixture-backed evidence only: it does not establish a live arXiv, model, renderer, or delivery workflow.

## 11. Direction for the next design revision

Before claiming an end-to-end product, define versioned saved-data formats with matching date and run identity, validate required summary coverage, and make the deterministic fallback independent of summarization success. Add a fixture-backed runner before live delivery.

Future operational rollout should use shadow runs without sending, then a separately approved switch with a rollback path. Existing live jobs remain outside the scope of this documentation change. Scheduling, zero-paper behavior, default model, and delivery retry policy remain decisions to make, not current guarantees.
