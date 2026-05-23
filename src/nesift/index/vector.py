"""Dense vector index: cosine similarity over a stacked float32 matrix."""

from __future__ import annotations

import numpy as np


class VectorIndex:
    """Stacks L2-normalized embeddings; cosine = dot product."""

    def __init__(self, dim: int | None = None) -> None:
        self._dim = dim
        self._matrix: np.ndarray | None = None

    @property
    def dim(self) -> int | None:
        return self._dim

    def __len__(self) -> int:
        return 0 if self._matrix is None else int(self._matrix.shape[0])

    def add_many(self, vectors: np.ndarray) -> None:
        if vectors.size == 0:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        if self._dim is None:
            self._dim = int(vectors.shape[1])
        if vectors.shape[1] != self._dim:
            raise ValueError(
                f"embedding dim mismatch: index={self._dim} new={vectors.shape[1]}"
            )
        self._matrix = (
            vectors if self._matrix is None else np.vstack([self._matrix, vectors])
        )

    def scores(self, query_vec: np.ndarray) -> np.ndarray:
        n = len(self)
        if n == 0 or self._matrix is None:
            return np.zeros(0, dtype=np.float32)
        q = np.asarray(query_vec, dtype=np.float32)
        if q.ndim > 1:
            q = q.ravel()
        return self._matrix @ q
