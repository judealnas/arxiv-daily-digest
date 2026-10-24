# arxiv-renderer

Dependency-free HTML renderer for the JSON array produced by
`scripts/run_test_pipeline.py`. Rendering never fetches papers, runs a model,
or sends email.

## CLI

From the workspace after `uv sync`:

```sh
uv run arxiv-render summarized.json --date 2026-10-23 --output digest.html
uv run arxiv-render --date 2026-10-23 < summarized.json > digest.html
uv run python -m arxiv_renderer summarized.json --date 2026-10-23 -o digest.html
```

`--run-id` supplies an optional source identifier; `--notice` adds escaped text
near the top. Input/output default to `-` (stdin/stdout). Output file parent
directories must already exist. Successful rendering exits 0, including partial
summary failures; invalid input or I/O errors exit 1 with diagnostics on stderr.
Do not use renderer success as a signal that every summary succeeded.

A digest object containing `papers` and `date` is also accepted. Optional
`run_id`, `count`, `matched_count`, and `summarizer_config` are retained; supplied
counts must agree with the entries. CLI date/run-ID overrides cannot conflict
with supplied metadata. Array input requires an explicit date. No date or run
identity is guessed from the clock. Empty arrays produce a zero-paper digest.

Each paper needs unique nonempty `id`, string `title` and `abstract`, boolean
`matched`, and an HTTP(S) `url` or `abstract_link`. Papers stay in supplied order.
The renderer does not rank them. Authors are omitted.

## Summary fallback and provenance

- A matched paper uses its summary only when `summary_status` is `ok` and
  `summary` is a nonempty string.
- When summaries were attempted, unsuccessful/missing summaries use the full
  cleaned abstract and a short message derived from `summary_error_code`.
  The complete `summary_error` is shown in a separate, expanded `<blockquote>`
  labeled **Technical details — raw diagnostic**, with inline shading, border,
  and smaller text. It is outside the paper paragraph, HTML-escaped, and retains
  line breaks. No collapsible controls, JavaScript, hidden text, or attachment
  are used. A top warning reports the fallback count; successful summaries stay.
- The summarizer emits optional `summary_error_code` values for parser failures,
  HTTP statuses (`http_429`, etc.), timeouts, empty responses, output-token limits,
  and local generation errors. The runner and JSONL CLI retain this field while
  leaving raw diagnostics unchanged. Classification does not use another LLM,
  change validation, or add retries. Output-token limits are identified only
  from an explicit `finish_reason=length` on an unusable response.
- Saved results without a code still work: a small set of exact parser messages
  and anchored HTTP/timeout prefixes are recognized. Unknown codes/errors use
  **Summarization failed; see technical details.** No cause is guessed from raw
  provider prose. Missing diagnostics produce a short explanation without an
  empty quote block.
- Without summary fields, matched papers use full cleaned abstracts without a
  failure warning. Unmatched papers use their first cleaned sentence.
- Distinct `summarizer_config` values appear at the top, using an explicit
  allowlist: backend, provider, model, length/batch/beam settings, prompt grouping,
  temperature, and token budget. The pipeline runner retains this configuration
  for BART, generic LLM, and OpenRouter backends on matched records.
- `summary_metadata.returned_model`, when available, is displayed below each
  successful summary. If only `requested_model` exists, it is explicitly labeled
  as requested, with the actual model unavailable. Router aliases are never
  presented as verified executing models. Missing historical metadata cannot
  be reconstructed.

Text and attributes are HTML-escaped. **Error strings are intentionally shown
in full as escaped text in the quote block and may contain sensitive provider details. Review
input errors before distributing the HTML.** Configuration credentials and
endpoint URLs are not collected or included in the configuration header.

## Library

```python
from arxiv_renderer import prepare_digest, render_digest

html = render_digest(prepare_digest(papers, date="2026-10-23"))
```

The existing `arxiv_digest_renderer` module re-exports `render_digest`,
`clean_text`, and `first_sentence` for compatibility. The optional legacy
`summaries={paper_id: text}` argument remains supported; missing entries fall
back visibly and unknown/unmatched IDs are rejected.

Both shell wrappers invoke `arxiv-render` and email partial digests with full
abstracts and errors for failed summaries. Empty listings and failed stages
still stop delivery. This does not deploy or modify live cron jobs.
