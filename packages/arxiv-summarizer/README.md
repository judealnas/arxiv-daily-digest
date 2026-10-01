# arxiv-summarizer

Batch summarizer for the arXiv daily digest pipeline. JSONL in → JSONL out.

Part of the `arxiv-daily-digest` uv workspace. Feeds the HTML digest formatter
(see repo root `FEATURE-PARITY-ANALYSIS.md`).

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
themselves to the contract test's registry. BART runs it against the real
model (marked `slow`); LLM runs it against `FakeLLMClient` (free, fast).

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
uv run pytest packages/arxiv-summarizer/tests/ -m "not slow"   # fast (~12s)
uv run pytest packages/arxiv-summarizer/tests/                 # + real-model integration
```

- Unit/contract tests never load models or call APIs (fake clients,
  monkeypatched constructors).
- One `@pytest.mark.slow` integration test exercises real BART generation.
- CLI tests pin: default→bart, `--backend llm`→llm, JSONL round-trip.

## transformers v5 notes

- `pipeline("summarization")` was removed — use `AutoModelForSeq2SeqLM`
  directly.
- PEGASUS tokenizer needs `sentencepiece` + `protobuf` (falls back to a
  broken tiktoken path without them).
