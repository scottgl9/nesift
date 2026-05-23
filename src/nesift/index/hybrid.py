"""Hybrid fusion: BM25 + cosine via Reciprocal Rank Fusion, with reranking."""

from __future__ import annotations

import numpy as np

from nesift.config import RRF_K
from nesift.index.bm25 import tokenize


def ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    """Return 1-based ranks for ``scores`` (highest score → rank 1)."""

    n = scores.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    order = np.argsort(-scores, kind="stable")
    ranks = np.zeros(n, dtype=np.int32)
    for r, idx in enumerate(order, start=1):
        ranks[idx] = r
    return ranks


def rrf(score_lists: list[np.ndarray], k: int = RRF_K) -> np.ndarray:
    """Reciprocal Rank Fusion of ``score_lists`` (one score vector per channel).

    All input vectors must be the same length. Returns a fused score
    where higher is better.
    """

    if not score_lists:
        return np.zeros(0, dtype=np.float32)
    n = score_lists[0].shape[0]
    for s in score_lists:
        if s.shape[0] != n:
            raise ValueError("score lists must have matching length")
    fused = np.zeros(n, dtype=np.float32)
    for scores in score_lists:
        ranks = ranks_from_scores(scores)
        # Channels with all-zero scores get zero contribution.
        if not scores.any():
            continue
        fused += 1.0 / (k + ranks)
    return fused


def rerank(
    fused_scores: np.ndarray,
    *,
    page_ids: list[str],
    sections: list[str | None],
    query: str,
    top_k: int,
    diversity_penalty: float = 0.9,
    heading_boost: float = 1.05,
) -> list[tuple[int, float]]:
    """Greedy rerank: pick best, then penalize same-page repeats; boost on heading match.

    Returns ``[(chunk_index, adjusted_score), ...]`` in chosen order.
    """

    n = fused_scores.shape[0]
    if n == 0 or top_k <= 0:
        return []
    q_tokens = set(tokenize(query))
    # Pre-compute heading boost factors.
    boosts = np.ones(n, dtype=np.float32)
    for i, sec in enumerate(sections):
        if sec and set(tokenize(sec)) & q_tokens:
            boosts[i] = heading_boost
    boosted = fused_scores * boosts

    chosen: list[tuple[int, float]] = []
    seen_pages: dict[str, int] = {}
    candidates = np.argsort(-boosted, kind="stable")
    for idx in candidates:
        if len(chosen) >= top_k:
            break
        if boosted[idx] <= 0:
            break
        page = page_ids[idx]
        penalty = diversity_penalty ** seen_pages.get(page, 0)
        adj = float(boosted[idx]) * penalty
        chosen.append((int(idx), adj))
        seen_pages[page] = seen_pages.get(page, 0) + 1
    # Sort chosen by adjusted score desc (greedy picks may need reordering).
    chosen.sort(key=lambda x: -x[1])
    return chosen
