# CLI reference

All commands accept:

| Flag | Default | Description |
|------|---------|-------------|
| `--session <id>` | `pid<ppid>` or `$NESIFT_SESSION` | Pick a session id |
| `--json` | off | Emit JSON instead of human-readable output |
| `--fast` | off (where applicable) | Use the smaller `potion-base-8M` model |
| `--no-embed` | off (where applicable) | Skip embeddings entirely (BM25-only mode) |

## `nesift add <url>`

Fetch, extract, chunk, embed (unless `--no-embed`), and store one URL.

```bash
nesift add https://en.wikipedia.org/wiki/Retrieval-augmented_generation
nesift add --json --fast https://example.com/post
```

JSON output:

```json
{ "url": "...", "title": "...", "chunks": 14, "triage": "..." }
```

## `nesift add-batch <url> [<url> ...]`

Sequentially ingest multiple URLs. Continues past per-URL failures and reports each result.

## `nesift query "<q>"`

Hybrid BM25 + embeddings retrieval against the session.

| Flag | Default | Description |
|------|---------|-------------|
| `--top-k N` | 5 | Number of chunks (ignored if `--budget` is set) |
| `--budget N` | unset | Trim greedily by score until N tokens used |
| `--url URL` | unset | Restrict to a single previously-indexed URL |

JSON output:

```json
{
  "results": [
    { "chunk": "...", "url": "...", "section": "...", "score": 0.95, "sources": 2, "token_count": 412 }
  ],
  "budget_total": 1500,
  "budget_used": 1421
}
```

## `nesift answer "<question>"`

Like `query`, but produces an extractive answer paragraph with `[N]` citations and a trailing `Sources:` block.

## `nesift score "<query>" "<snippet>" [<snippet> ...]`

Rank snippets by relevance to a query **without** fetching anything. Useful for pre-filtering search results before deciding which pages to ingest.

```bash
nesift score "vector database" \
  "Pinecone is a managed vector DB." \
  "How to bake sourdough."
```

## `nesift search "<q>"`

One-shot SearXNG bridge: search → pre-score snippets → fetch top hits → index → answer.

| Flag | Default | Description |
|------|---------|-------------|
| `--top N` | 5 | Top N search hits to fetch + ingest |
| `--budget N` | 2000 | Token budget for the answer |
| `--instance URL` | `$NESIFT_SEARXNG_URL` or `http://127.0.0.1:8888` | SearXNG instance |
| `--via-searxng/--no-via-searxng` | on | Backend selector (only SearXNG implemented) |

Requires a SearXNG instance with the JSON API enabled (`formats: [html, json]` in `settings.yml`).

## `nesift list`

Show indexed URLs with title, chunk count, and triage summary.

## `nesift clear`

Drop the active session's index file.

## `nesift save -o <path>`

Copy the active session's index JSON to `<path>` for later reuse / inspection.

## `nesift version`

Print the installed nesift version.
