"""Embedding model wrapper around model2vec's StaticModel."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from nesift.config import DEFAULT_MODEL, FAST_MODEL, MULTILINGUAL_MODEL


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Minimal interface required of any embedder."""

    dim: int

    def embed(self, text: str) -> np.ndarray: ...

    def embed_many(self, texts: list[str]) -> np.ndarray: ...


class Embedder:
    """Lazy-loading wrapper around a model2vec ``StaticModel``.

    The model is only loaded on first call; tests can substitute a
    :class:`FakeEmbedder` to avoid the ~30-60MB download.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        fast: bool = False,
        lang: bool = False,
    ) -> None:
        if model_name is None:
            if fast:
                model_name = FAST_MODEL
            elif lang:
                model_name = MULTILINGUAL_MODEL
            else:
                model_name = DEFAULT_MODEL
        self.model_name = model_name
        self._model = None
        self._dim: int | None = None

    def _load(self) -> object:
        if self._model is None:
            from model2vec import StaticModel  # imported lazily

            self._model = StaticModel.from_pretrained(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            v = self.embed("dimension probe")
            self._dim = int(v.shape[0])
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        model = self._load()
        v = np.asarray(model.encode([text])[0], dtype=np.float32)
        self._dim = int(v.shape[0])
        return _l2_normalize(v)

    def embed_many(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim if self._dim else 1), dtype=np.float32)
        model = self._load()
        m = np.asarray(model.encode(texts), dtype=np.float32)
        self._dim = int(m.shape[1])
        # L2-normalize per row for cosine via dot product.
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (m / norms).astype(np.float32)


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0:
        return v
    return (v / n).astype(np.float32)


class FakeEmbedder:
    """Deterministic, dependency-free embedder for tests.

    Hashes word n-grams into a fixed-dimension vector and L2-normalizes
    it. Same text → same vector across runs.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        return _hash_embed(text, self.dim)

    def embed_many(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([_hash_embed(t, self.dim) for t in texts])


def _hash_embed(text: str, dim: int) -> np.ndarray:
    import hashlib

    vec = np.zeros(dim, dtype=np.float32)
    tokens = [t.lower() for t in text.split() if t]
    if not tokens:
        return vec
    for tok in tokens:
        h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "little") % dim
        sign = 1.0 if (h[4] & 1) else -1.0
        vec[idx] += sign
    n = float(np.linalg.norm(vec))
    if n == 0:
        return vec
    return (vec / n).astype(np.float32)
