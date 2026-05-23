"""Session store: in-memory page list backed by a JSON file in tempdir."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from nesift.config import session_path
from nesift.index.bm25 import BM25Index
from nesift.index.vector import VectorIndex
from nesift.schema import Chunk, Page


def page_id_for(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


class SessionStore:
    """Holds the active session's pages plus its rebuilt BM25/vector indices."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else session_path()
        self.pages: list[Page] = []
        self.bm25: BM25Index = BM25Index()
        self.vectors: VectorIndex = VectorIndex()
        self._flat_chunks: list[Chunk] = []

    # ---------- persistence ----------

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        self.pages = [Page.from_json(p) for p in data.get("pages", [])]
        self._rebuild_indices()

    def save(self, target: Path | None = None) -> Path:
        tgt = Path(target) if target is not None else self.path
        tgt.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "pages": [p.to_json() for p in self.pages]}
        tgt.write_text(json.dumps(payload), encoding="utf-8")
        return tgt

    def clear(self) -> None:
        self.pages = []
        self.bm25 = BM25Index()
        self.vectors = VectorIndex()
        self._flat_chunks = []
        if self.path.exists():
            self.path.unlink()

    # ---------- mutation ----------

    def has_url(self, url: str) -> bool:
        pid = page_id_for(url)
        return any(p.id == pid for p in self.pages)

    def add_page(self, page: Page) -> None:
        # Replace existing page with same id (re-ingest).
        self.pages = [p for p in self.pages if p.id != page.id]
        self.pages.append(page)
        self._rebuild_indices()

    # ---------- read ----------

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._flat_chunks)

    def page_for(self, url: str) -> Page | None:
        pid = page_id_for(url)
        for p in self.pages:
            if p.id == pid:
                return p
        return None

    # ---------- internals ----------

    def _rebuild_indices(self) -> None:
        self.bm25 = BM25Index()
        self.vectors = VectorIndex()
        self._flat_chunks = []
        embeddings: list[np.ndarray] = []
        for page in self.pages:
            for chunk in page.chunks:
                self._flat_chunks.append(chunk)
                self.bm25.add(chunk.text)
                if chunk.embedding is not None:
                    embeddings.append(chunk.embedding)
        if embeddings:
            self.vectors.add_many(np.vstack(embeddings))
