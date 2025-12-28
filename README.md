## Parse, don’t prompt — `web_agent_parser`

Parser-level guardrails for agentic web browsers: **reduce prompt-injection attack surface by transforming raw HTML into a structured, annotated representation before any LLM reasoning**.

This repo scaffolds the “parse, don’t prompt” premise described in `Parse, don't prompt.pdf`, including a minimal ingestion pipeline, a risk-annotation pass, and an internal schema you can extend for experiments.

### Repo structure

- **`src/web_agent_parser/`**: library code (HTML extraction, structuring, risk annotations, storage).
- **`scripts/`**: runnable utilities (ingest URL / HTML into a local store).
- **`data/`**: input/output artifacts (ignored by git except README + `.gitkeep`).
  - **`data/raw/`**: raw HTML snapshots (or downloaded pages).
  - **`data/processed/`**: structured JSON / SQLite DB exports.
- **`scratch/`**: notebooks, ad-hoc experiments (ignored by git except README + `.gitkeep`).
- **`utils/`**: one-off helpers not part of the core library.

### Core idea (pipeline)

1. **Ingest**: fetch or load HTML.
2. **Parse**: extract meaningful text blocks with provenance (tag path, attributes, position).
3. **Normalize**: language detection + (optional) translation to English, dropping low-confidence fragments.
4. **Annotate**: flag patterns correlated with prompt injection (imperatives, instruction hierarchy, coercion).
5. **Store**: persist a structured representation (JSONL / SQLite).
6. **Query**: the agent queries the structured store and receives **only** relevant, annotated snippets.

### Quickstart

Create a venv and install deps:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Try parsing a URL (saves a JSONL file under `data/processed/`):

```bash
python -m web_agent_parser.cli ingest-url "https://example.com"
```

### Notes / citations

- Prompt-injection in agentic browsers and mitigation framing: [Brave blog: “Agentic Browser Security: Indirect Prompt Injection in Perplexity Comet”】【](https://brave.com/blog/comet-prompt-injection/)


