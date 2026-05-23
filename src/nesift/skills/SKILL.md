---
name: nesift
description: |
  Token-efficient web research for AI agents. Fetches, indexes, and queries
  web content locally via hybrid BM25 + embeddings, returning only the
  chunks that fit a token budget. Replaces raw `web_fetch` for any task
  involving more than one source. Use when the user asks you to research,
  compare, summarize, or fact-check anything across multiple URLs or
  search results.
---

# nesift skill

This skill exposes `nesift` (https://github.com/scottgl9/nesift), a local
semantic-search layer that sits between your `web_fetch` tool and the
LLM. Indexing and querying happen entirely on CPU; no API keys.

## When to invoke

- Any multi-page web research task.
- When raw `web_fetch` would burn >5k tokens per page on prose that is
  90% irrelevant to the question.
- When the user has a SearXNG instance and wants search → fetch →
  answer in a single shot.

## Mechanics

The `nesift` CLI is available in PATH. State lives in
`/tmp/nesift-<session>.json` keyed on parent PID by default; you can pin
it with `NESIFT_SESSION=<name>` if you want the index to persist across
shells.

### Typical flow

```bash
# 1. Pre-score search results BEFORE fetching to skip irrelevant ones.
nesift score "<question>" "<snippet1>" "<snippet2>" ... --json

# 2. Index the URLs that survived scoring.
nesift add-batch "<url1>" "<url2>" "<url3>" --json

# 3. Ask the question. The --budget flag bounds the cost.
nesift answer "<question>" --budget 2000

# 4. Or, if a local SearXNG instance is configured, do it all at once:
NESIFT_SEARXNG_URL=http://127.0.0.1:8888 \
  nesift search "<question>" --top 5 --budget 2000

# 5. Clear the session when the task is done.
nesift clear
```

### MCP server (alternative)

Instead of shelling out, agents can connect via MCP:

```bash
nesift-mcp
# or
uvx --from "nesift[mcp]" nesift-mcp
```

Tools: `score_snippets`, `add_page`, `add_batch`, `query`, `answer`,
`list_pages`, `clear`, `search`.

## Tips

- Always `score` snippets before fetching when you have ≥3 candidate
  URLs — it's near-free and routinely drops 40-60% of the queue.
- Use `--budget` aggressively; 1500-2500 tokens is enough for most
  answers and saves ~10× over raw fetch.
- For non-English content, add `--lang` (loads the multilingual model).
- For PDFs, `nesift add` auto-detects the content type and dispatches
  to the PDF extractor.
