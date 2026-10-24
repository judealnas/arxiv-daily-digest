# arxiv-summarizer

Batch summarizer for the arXiv daily digest pipeline. JSONL in → JSONL out.

Part of the `arxiv-daily-digest` uv workspace. Feeds the HTML digest formatter
(see repo root `FEATURE-PARITY-ANALYSIS.md`).

## Rejected-response diagnostics

`LlmSummarizer` and `OpenRouterSummarizer` accept an optional
`diagnostics_dir=Path(...)`. By default no responses are written to disk.
The local pipeline runner exposes this as `--diagnostics-dir PATH`;
`openrouter_free_pipeline.sh` enables it inside its private run directory.

Rejected completions and request failures are saved as raw diagnostic
artifacts when enabled. Independently of diagnostics, every received LLM completion adds
`summary_metadata` to its result and the pipeline paper record: `requested_model`,
`returned_model`, and `finish_reason`. This applies to successful and invalid-format
responses. Plain-text injected clients use null for unavailable returned-model
and finish-reason fields. Transport failures have no completion metadata; BART
results are unchanged. Metadata is serialized in JSON, not rendered in email.
Successful wrapper runs still delete their temporary output.

Each uniquely
named JSON file contains requested paper IDs, requested model, returned model,
finish reason, exact rejection reason, and original completion text. Returned
model and finish reason are null when an injected client supplies plain text
without metadata. The built-in HTTP client still returns a string-compatible
value. Request failures now retain the original exception type/message and HTTP
error body, rather than collapsing to `LLM request failed`. Null-content API
responses include the API envelope in the exception message. With diagnostics
enabled, `request-error-*.json` additionally records requested IDs/model, exception
type/message, HTTP status, and response body. Outgoing headers, keys, and prompts
are not deliberately captured, but raw provider errors can contain sensitive data
and also appear in result JSON and validation logs. Keep artifacts and logs private
(directory 0700, files 0600) and inspect before sharing.

For a fixed-model run without changing the wrapper's default router alias:
`OPENROUTER_MODEL=cohere/north-mini-code:free ./openrouter_free_pipeline.sh`

Validation remains strict. The public error includes the specific safe rejection
reason, but never the response text. No format-repair retries or structured-output
request settings are added. `--debug` writes runner logs to stderr, keeping JSON
stdout clean; avoid `--verbose` when redirecting stdout to a JSON file.
Failed wrapper runs retain their artifacts; successful runs delete them. Scratch
artifacts are subject to automatic pruning.

## Architecture

At the core: a `Summarizer` base class. Configuration happens at
construction; the `summarize` call takes no options.

```
                 ┌──────────────────────────────────┐
 stdin JSONL ──▶ │  cli.py                          │
 {id, text}      │  parse args → build_backend()    │
                 └─────────┬────────────────────────┘
                           │ constructs ONE configured instance
              ┌────────────┴─────────────┐
              ▼                          ▼
   ┌─────────────────────┐    ┌─────────────────────┐
   │ BartSummarizer      │    │ LlmSummarizer       │
   │ (base.Summarizer)   │    │ (base.Summarizer)   │
   │ local seq2seq,      │    │ OpenAI-compatible   │
   │ no API key; batched │    │ provider; chunked   │
   │ generate() calls    │    │ requests (1 by def.)│
   └─────────┬───────────┘    └─────────┬───────────┘
             └────────────┬─────────────┘
                          ▼
            stdout JSONL {id, status, summary, error}
```

### The `Summarizer` interface (`base.py`)

```python
class Summarizer(ABC):
    @abstractmethod
    def summarize(self, text: str) -> ResultItem: ...   # one text in, one result out

    def summarize_many(self, texts: list[str]) -> list[ResultItem]:
        return [self.summarize(t) for t in texts]        # backends override for batching
```

- **`BartSummarizer`** — `facebook/bart-large-cnn` via `AutoModelForSeq2SeqLM`.
  Overrides `summarize_many` with true batched generation
  (`batch_size` texts per `generate()` call, beam search, deterministic).
  No network at run time; weights cached in `~/.cache/huggingface`
  (~1.6GB first download).
- **`OpenRouterSummarizer`** — hosted OpenRouter backend, implementing the
  same `Summarizer` interface as BART. By default it makes one request per
  text; `papers_per_prompt=N` enables strict JSON groups of N records.
  Individual request failures return error results without aborting the batch.
  Uses `OPENROUTER_API_KEY` and the existing OpenRouter provider model default
  (`google/gemma-2-27b-it`), or an explicit `model=` override. Requests may
  incur provider charges; tests inject a fake client and never call the API.
- **`LlmSummarizer`** — provider-based (`opencode-zen`, `openrouter`).
  Its `summarize_many` compatibility API creates synthetic IDs and delegates
  to `summarize_items`, which uses the same strict ID-keyed JSON contract for
  both single-paper and multi-paper prompts.
  Without `papers_per_prompt`, legacy `chunk_count=1` sends one request for
  the whole batch. `papers_per_prompt=N` instead makes fixed N-item groups.
  The HTTP client is injectable (`client=`) so tests run against fakes — real
  API calls are never made from tests (cost rule).

### Why classes, not a string-dispatched function

An earlier design passed `backend="bart" | "llm"` into one function with
a fat shared signature (LLM-only and BART-only params mixed). The class
hierarchy removes that: each constructor validates its own config, the
call site stays `summarize(text)`, and the contract test applies to any
instance.

### Result contract

`ResultItem(id, status, summary, error)`:

- `status="ok"` → non-empty `summary`, `error=None`
- `status="error"` → `summary=None`, human-readable `error`
- a summary never equals its input verbatim

Enforced by `tests/contract/` — shared by ALL backends. New backends add
themselves to the contract test's registry. Ordinary BART contract tests use
the real `BartSummarizer` with fake tokenizer/model components; LLM runs
against `FakeLLMClient` (free, fast). A separate real BART generation check
is marked `slow`.

## Defaults (from experiments/summarizer/)

`max_length=130`, `min_length=40`, `num_beams=4`, `do_sample=False`,
`batch_size=4`. See `bart_param_sweep.md` there for the tuning evidence.
Note: BART has no sentence-count or detail dial — `min_length` is the only
lever that adds content.

## Usage

```bash
# BART backend (default; local, free)
cat papers.jsonl | arxiv-summarize > summaries.jsonl

# BART with tuning
cat papers.jsonl | arxiv-summarize --max-length 160 --min-length 80

# Dedicated OpenRouter backend (one hosted request per abstract)
# Set OPENROUTER_API_KEY in your environment; choose an available model.
cat papers.jsonl | arxiv-summarize --backend openrouter --model YOUR_MODEL_ID

# Group hosted OpenRouter requests in pairs; each multi-paper response must be
# a JSON array of {"id", "summary"} records with every requested ID once.
cat papers.jsonl | arxiv-summarize --backend openrouter --model YOUR_MODEL_ID --papers-per-prompt 2

# Generic LLM backend: default legacy --chunk behavior is one request for all
# papers. Use --papers-per-prompt for fixed-size strict JSON groups instead.
cat papers.jsonl | arxiv-summarize --backend llm --provider opencode-zen --papers-per-prompt 2

# --chunk remains generic-LLM-only and cannot be combined with
# --papers-per-prompt. It is rejected for --backend openrouter and BART does
# not support --papers-per-prompt.

# Set the qualitative summary guidance inline:
cat papers.jsonl | arxiv-summarize --backend llm \
  --user-prompt 'Use five sentences. Emphasize numerical results.'

# Or read multiline guidance from a UTF-8 file (llm/openrouter):
cat papers.jsonl | arxiv-summarize --backend openrouter --model YOUR_MODEL_ID \
  --papers-per-prompt 2 --user-prompt-file prompts/quantum-summary.txt
```

Example `prompts/quantum-summary.txt`:

```text
Write five sentences per paper for quantum networking researchers.
Highlight the technical contribution, quantitative results, and stated limitations.
```

The **system message is fixed in code**: it describes a JSON user message
containing `guidance` and `papers` (`{id, text}` records) and requires a JSON
array of `{id, summary}` responses with each ID exactly once. User guidance
is treated literally; it needs no `{text}` placeholder and cannot replace the
format contract. Missing or malformed IDs, extra records, and invalid JSON
produce error results for that group. The parser enforces the output shape,
though it cannot guarantee a model will follow the prompt or summarize the
correct paper semantically.

`--user-prompt` and `--user-prompt-file` are mutually exclusive. The default
guidance is "Summarize each paper concisely." Library callers use
`user_prompt="Use five sentences..."`. The former `--system-prompt`,
`--template`, and `--template-file` options (and `system_prompt=`/`template=`
constructor parameters) have been removed; convert a template's qualitative
instructions to user guidance without `{text}` or `{count}` placeholders.
Both one-paper and grouped LLM calls now require the same ID-keyed JSON
response; plain-text responses for one paper are treated as errors.

Input and output are one JSON object per line:

```jsonl
{"id":"2609.00002","text":"First paper abstract."}
{"id":"2609.00003","text":"Second paper abstract."}
```

```jsonl
{"error":null,"id":"2609.00002","status":"ok","summary":"First concise summary."}
{"error":null,"id":"2609.00003","status":"ok","summary":"Second concise summary."}
```

For the generic LLM and dedicated OpenRouter backends, CLI calls
`summarize_items` internally, so returned IDs are emitted directly rather than
being restored by position. A malformed grouped response marks every paper in
that group as an error; other groups still run.

Library API:

```python
from arxiv_summarizer import BartSummarizer, OpenRouterSummarizer, LlmSummarizer, ResultItem, Summarizer

summarizer: Summarizer = OpenRouterSummarizer(model="YOUR_MODEL_ID")  # or BartSummarizer()
item: ResultItem = summarizer.summarize("Some abstract ...")
items = summarizer.summarize_many(["abstract 1", "abstract 2"])
```

## Testing

```bash
# Ordinary tests: no real model loading and no external network.
uv run pytest -m "not slow and not live" -q

# Real local-model integration tests. These may load cached local weights.
# The semantic integration test additionally needs the optional dependency:
uv sync --package arxiv-interest-sort --extra semantic
uv run pytest -m slow -q

# Opt-in hosted-service group. No hosted live smoke test exists yet.
uv run pytest -m live -q
```

- The root test guard blocks real network connections and BART or
  SentenceTransformer model loading in ordinary tests, while allowing mocked
  transports and fake BART model loaders.
- Unit/contract tests never load models or call APIs (fake clients,
  monkeypatched constructors).
- One `@pytest.mark.slow` integration test exercises real BART generation.
- CLI tests pin: default→bart, `--backend llm`→llm, JSONL round-trip.

## transformers v5 notes

- `pipeline("summarization")` was removed — use `AutoModelForSeq2SeqLM`
  directly.
- PEGASUS tokenizer needs `sentencepiece` + `protobuf` (falls back to a
  broken tiktoken path without them).
