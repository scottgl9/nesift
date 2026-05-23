"""Pre-fetch snippet scorer: cosine-rank snippets without downloading pages."""

from __future__ import annotations

import numpy as np

from nesift.embedder import EmbedderProtocol
from nesift.schema import ScoredSnippet


def score_snippets(
    query: str, snippets: list[str], embedder: EmbedderProtocol
) -> list[ScoredSnippet]:
    """Rank ``snippets`` by cosine similarity to ``query``.

    Returned list is sorted by descending score with the original index
    preserved on each entry.
    """

    if not snippets:
        return []
    q = embedder.embed(query)
    mat = embedder.embed_many(snippets)
    if mat.size == 0:
        return []
    sims = mat @ q
    out = [
        ScoredSnippet(index=i, text=snippets[i], score=float(sims[i]))
        for i in range(len(snippets))
    ]
    out.sort(key=lambda s: -s.score)
    return out


def topk(scored: list[ScoredSnippet], k: int) -> list[ScoredSnippet]:
    return scored[: max(0, k)]


__all__ = ["score_snippets", "topk", "np"]
