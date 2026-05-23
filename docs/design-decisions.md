# Design decisions

## Dedup threshold: 0.88

The PRD called for 0.85–0.90. We picked **0.88** as the midpoint. Override at runtime with `NESIFT_DEDUP_THRESHOLD`:

```bash
NESIFT_DEDUP_THRESHOLD=0.92 nesift query "..."
```

Why a midpoint default: 0.85 collapsed paraphrases that still added information; 0.90 missed verbatim mirrors with minor whitespace differences. 0.88 catches both extractive-quote reuse and full mirrors without merging genuinely distinct paragraphs.

## Answer mode is extractive

The PRD left "pure extractive concatenation vs. light rewriting pass" open. We chose **extractive only** for v0.1 because:

- It is faithful: every word of the answer appears verbatim in a source chunk.
- It requires no LLM at the synthesis step (keeping nesift entirely local).
- It composes cleanly with citation numbering — chunks map 1:1 to `[N]` markers.

A future rewriting pass can be added behind an opt-in flag without changing the extractive path.

## tiktoken for budget reasoning

`tiktoken`'s `cl100k_base` encoding is the de-facto unit in which agents reason about context budgets, even though the embedding model is different. If `tiktoken` is unavailable we fall back to `len(text.split()) * 1.3` — a coarse but stable approximation.

## Lazy model loading

`Embedder._load()` defers the `model2vec.StaticModel.from_pretrained` call until the first `embed(...)` call. This keeps CLI startup snappy for commands that do not need embeddings (`list`, `clear`, `save`, `version`). Tests bypass the download entirely by substituting `FakeEmbedder`.

## Session storage in `tempfile.gettempdir()`

Sessions are scoped to a task: an agent that runs for the duration of a research session, then exits. `/tmp` is appropriate because:

- Indices are ephemeral by design.
- Cleanup happens at machine reboot for free.
- No PII handling concerns (the agent's own scratch space).

For long-lived archival, `nesift save -o <path>` writes the index anywhere on disk.

## Flat module layout

The package is mostly flat under `src/nesift/`. Only `index/` is a subpackage, because BM25 / vector / hybrid evolve together and share a scoring interface. Premature subpackaging (e.g. `nesift.core`, `nesift.io`) would force readers to chase imports across boundaries for no gain at this size.

## Diversity penalty: 0.9 per repeat hit

After the top result is selected, every subsequent chunk from the same page is multiplied by `0.9` per prior occurrence. This means a same-page second hit needs to beat a different-page second hit by ~11% before diversity kicks in. Strong enough to spread sources, weak enough not to push obviously-best chunks off the list.

## SearXNG instead of direct search providers

SearXNG is a search aggregator the user already runs locally; it removes the API-key requirement entirely. The `--via-searxng/--no-via-searxng` flag is a stub for future direct-provider backends.
