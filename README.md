# nesift

**Fast, local semantic search over web content for AI agents.** Sifts the net for signal — uses ~90% fewer tokens than raw `web_fetch`.

[github.com/scottgl9/nesift](https://github.com/scottgl9/nesift)

---

## What it does

When an AI agent researches the web, the usual flow is: search → fetch 10 pages → drown in 100k+ tokens of irrelevant prose. `nesift` sits between the web and the agent: it ingests pages on the fly, indexes them with hybrid BM25 + embeddings, deduplicates redundant content across sources, and returns only the chunks that fit your token budget.

- **Local** — runs on CPU, no API keys, no cloud calls (other than the page fetch itself).
- **Zero setup** — `pip install -e .`, no database, no daemon.
- **Session-scoped** — index lives in `/tmp` and is per-session by default.
- **Hybrid retrieval** — BM25 + `potion-retrieval-32M` embeddings fused via RRF.
- **Context budget mode** — `--budget N` trims results to N tokens.
- **Cross-page dedup** — collapses near-identical chunks, notes source count.
- **SearXNG bridge** — `nesift search "..."` does search + filter + fetch + index + answer in one command.

## Install

```bash
git clone git@github.com:scottgl9/nesift.git
cd nesift
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python 3.11+.

## Quickstart

```bash
# Index a page and ask about it
nesift add https://en.wikipedia.org/wiki/Retrieval-augmented_generation
nesift query "what is RAG used for" --budget 1500
nesift answer "how does RAG reduce hallucinations"

# Pre-fetch scoring — rank snippets before downloading
nesift score "vector database" "Pinecone is a vector DB" "How to bake bread"

# One-shot SearXNG search + ingest + answer
NESIFT_SEARXNG_URL=http://127.0.0.1:8888 \
  nesift search "retry logic in distributed systems" --top 5 --budget 2000

nesift list
nesift clear
```

See [`docs/cli.md`](docs/cli.md) for every command and flag.

## How it works

```
URL → trafilatura extract → heading-aware chunker → triage summary
         → BM25 index + potion-retrieval-32M embeddings (CPU)
         → query: RRF fusion + dedup + budget trim → ranked chunks or synthesized answer
```

See [`docs/architecture.md`](docs/architecture.md).

## Status

Implemented (this release):
- Phase 1: CLI MVP (`add`, `query`, `list`, `clear`)
- Phase 2: hybrid retrieval, dedup, `--fast`
- Phase 3: triage summaries, `--budget`, `answer`, `score`
- Phase 5 (partial): SearXNG bridge (`nesift search`, `NESIFT_SEARXNG_URL`)

Deferred:
- Phase 4: MCP server (`nesift[mcp]` extras)
- Phase 5 remainder: multilingual model, PDF ingestion, OpenClaw skill scaffolding

## License

GPL-2.0-only — see [`LICENSE`](LICENSE).
