"""Thin wrapper over ``bm25s`` that fits our chunk-at-a-time workflow."""

from __future__ import annotations

import re

import bm25s
import numpy as np

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Index:
    """Holds tokenized chunks and (re)builds a ``bm25s.BM25`` on demand."""

    def __init__(self) -> None:
        self._tokens: list[list[str]] = []
        self._engine: bm25s.BM25 | None = None
        self._dirty = True

    def __len__(self) -> int:
        return len(self._tokens)

    def add(self, text: str) -> None:
        self._tokens.append(tokenize(text))
        self._dirty = True

    def add_many(self, texts: list[str]) -> None:
        for t in texts:
            self.add(t)

    def _ensure_built(self) -> None:
        if not self._dirty and self._engine is not None:
            return
        if not self._tokens:
            self._engine = None
            self._dirty = False
            return
        engine = bm25s.BM25()
        engine.index(self._tokens, show_progress=False)
        self._engine = engine
        self._dirty = False

    def scores(self, query: str) -> np.ndarray:
        """Return BM25 score per indexed chunk (zeros if index is empty)."""

        self._ensure_built()
        n = len(self._tokens)
        if n == 0 or self._engine is None:
            return np.zeros(0, dtype=np.float32)
        q_tokens = tokenize(query)
        if not q_tokens:
            return np.zeros(n, dtype=np.float32)
        # bm25s API: retrieve(queries, k) → (ids, scores), shape (1, k).
        ids, scores = self._engine.retrieve([q_tokens], k=n, show_progress=False)
        out = np.zeros(n, dtype=np.float32)
        for doc_id, score in zip(ids[0], scores[0]):
            out[int(doc_id)] = float(score)
        return out
