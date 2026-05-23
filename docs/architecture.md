# Architecture

```
URL ──► fetcher ──► trafilatura extractor ──► heading-aware chunker ──► triage summary
                                                  │
                                                  ▼
                                       ┌─── BM25 index (bm25s)
                                       │
                                       └─── potion-retrieval-32M embedder ──► vector index
                                                                                  │
query ──► tokenize ──► BM25 scores ┐                                              │
         + embed ──► cosine scores ┼──► RRF fusion ──► diversity + heading rerank
                                   │                            │
                                   │                            ▼
                                   │                  cross-page dedup
                                   │                            │
                                   │                            ▼
                                   │                    budget trimmer
                                   │                            │
                                   ▼                            ▼
                            top_k chunks                  answer synthesizer (citations)
```

## Module map

| Module | Responsibility |
|--------|----------------|
| `nesift.fetcher` | HTTP GET via httpx, sets UA, raises `FetchError` |
| `nesift.extractor` | `trafilatura.extract` → `ExtractedDoc(title, sections)` |
| `nesift.chunker` | Splits doc into chunks bounded by `MIN/TARGET/MAX_CHUNK_TOKENS`, never crossing headings |
| `nesift.summarizer` | Extractive 1–3 sentence triage summary (position + title overlap + heading overlap) |
| `nesift.tokens` | tiktoken `cl100k_base`; falls back to word-count × 1.3 if tiktoken absent |
| `nesift.embedder` | Lazy-loaded `model2vec.StaticModel` (`potion-retrieval-32M`); `FakeEmbedder` for tests |
| `nesift.index.bm25` | Wrapper over `bm25s.BM25` with rebuild-on-dirty |
| `nesift.index.vector` | Numpy matrix of L2-normalized vectors; cosine via matmul |
| `nesift.index.hybrid` | `rrf(...)` + greedy `rerank(...)` with diversity penalty + heading boost |
| `nesift.dedup` | Pairwise cosine ≥ `NESIFT_DEDUP_THRESHOLD` collapses results, summing `sources` |
| `nesift.budget` | Greedy token-budget trim, preserves rank order |
| `nesift.answer` | Extractive concat + numbered citations |
| `nesift.scorer` | Pre-fetch snippet scoring without downloading pages |
| `nesift.searxng` | SearXNG `format=json` client; instance URL via `NESIFT_SEARXNG_URL` |
| `nesift.session` | Per-session JSON store under `tempfile.gettempdir()` |
| `nesift.pipeline` | High-level orchestrators (`ingest_url`, `run_query`, `run_answer`, `run_search`) |
| `nesift.cli` | Typer command surface |

## Session store format

`/tmp/nesift-<session>.json`:

```json
{
  "version": 1,
  "pages": [
    {
      "id": "abc123def456",
      "url": "https://...",
      "title": "...",
      "fetched_at": 1718000000.0,
      "triage": "...",
      "chunks": [
        {
          "id": "abc123def456:0",
          "page_id": "abc123def456",
          "url": "...",
          "section": "Motivation",
          "text": "...",
          "token_count": 412,
          "embedding": "<base64 float32>",
          "embedding_dim": 512
        }
      ]
    }
  ]
}
```

Session id resolution order:

1. `--session <id>` flag on the CLI
2. `$NESIFT_SESSION` env var
3. `f"pid{os.getppid()}"` (so each parent shell gets its own index)

## Embedding model

Default: [`minishlab/potion-retrieval-32M`](https://huggingface.co/minishlab/potion-retrieval-32M) — a static `model2vec` distillation of `baai/bge-base-en-v1.5` fine-tuned for retrieval. ~30–60 MB; CPU-only; millisecond-latency per chunk; ~82% of MiniLM-L6's MTEB Retrieval score.

`--fast` swaps in `minishlab/potion-base-8M` (smaller, general-purpose, ~4× faster at indexing time).
