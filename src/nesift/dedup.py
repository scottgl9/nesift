"""Cross-page deduplication of near-identical chunks."""

from __future__ import annotations

import numpy as np

from nesift.config import dedup_threshold
from nesift.schema import QueryResult


def collapse(
    results: list[QueryResult],
    embeddings: list[np.ndarray | None],
    *,
    threshold: float | None = None,
) -> list[QueryResult]:
    """Collapse near-identical chunks across pages.

    Walks results in order; for each, compares cosine similarity against
    every already-kept result. If ≥ ``threshold``, merge into the kept
    result (incrementing ``sources``) and drop this one.

    Embeddings are expected to be L2-normalized (so dot product == cosine).
    Entries with ``None`` embedding never collapse.
    """

    thr = dedup_threshold() if threshold is None else threshold
    kept: list[QueryResult] = []
    kept_vecs: list[np.ndarray | None] = []
    for res, emb in zip(results, embeddings):
        merged_into: int | None = None
        if emb is not None:
            for i, kv in enumerate(kept_vecs):
                if kv is None:
                    continue
                sim = float(np.dot(emb, kv))
                if sim >= thr:
                    merged_into = i
                    break
        if merged_into is not None:
            kept[merged_into].sources += 1
        else:
            kept.append(res)
            kept_vecs.append(emb)
    return kept
