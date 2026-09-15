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
- **`LlmSummarizer`** — provider-based (`opencode-zen`, `openrouter`).
  Overrides `summarize_many` with chunked multi-output requests
  (`chunk_count=1` → one API call for the whole batch). The HTTP client
  is injectable (`client=`) so tests run against fakes — real API calls
  are never made from tests (cost rule).

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

# LLM backend
cat papers.jsonl | arxiv-summarize --backend llm --provider openrouter
```

Library API:

```python
from arxiv_summarizer import BartSummarizer, LlmSummarizer, ResultItem, Summarizer

summarizer = BartSummarizer(max_length=160, min_length=80)
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
