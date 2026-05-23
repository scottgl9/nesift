"""Dataclasses describing pages, chunks, query results, and external snippets."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Chunk:
    """A chunk of text from a page, with optional embedding."""

    id: str
    page_id: str
    url: str
    section: str | None
    text: str
    token_count: int
    embedding: np.ndarray | None = None

    def to_json(self) -> dict[str, Any]:
        emb: str | None = None
        if self.embedding is not None:
            arr = np.asarray(self.embedding, dtype=np.float32)
            emb = base64.b64encode(arr.tobytes()).decode("ascii")
        return {
            "id": self.id,
            "page_id": self.page_id,
            "url": self.url,
            "section": self.section,
            "text": self.text,
            "token_count": self.token_count,
            "embedding": emb,
            "embedding_dim": None if self.embedding is None else int(self.embedding.shape[0]),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Chunk:
        emb = None
        if data.get("embedding"):
            raw = base64.b64decode(data["embedding"])
            emb = np.frombuffer(raw, dtype=np.float32)
        return cls(
            id=data["id"],
            page_id=data["page_id"],
            url=data["url"],
            section=data.get("section"),
            text=data["text"],
            token_count=int(data["token_count"]),
            embedding=emb,
        )


@dataclass
class Page:
    """A fetched + indexed web page."""

    id: str
    url: str
    title: str
    fetched_at: float
    triage: str
    chunks: list[Chunk] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "url": self.url,
            "title": self.title,
            "fetched_at": self.fetched_at,
            "triage": self.triage,
            "chunks": [c.to_json() for c in self.chunks],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Page:
        return cls(
            id=data["id"],
            url=data["url"],
            title=data["title"],
            fetched_at=float(data["fetched_at"]),
            triage=data.get("triage", ""),
            chunks=[Chunk.from_json(c) for c in data.get("chunks", [])],
        )


@dataclass
class QueryResult:
    """A single ranked + (optionally) deduplicated query hit."""

    chunk: str
    url: str
    section: str | None
    score: float
    sources: int = 1
    token_count: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "chunk": self.chunk,
            "url": self.url,
            "section": self.section,
            "score": self.score,
            "sources": self.sources,
            "token_count": self.token_count,
        }


@dataclass
class ScoredSnippet:
    """A snippet scored by the pre-fetch scorer."""

    index: int
    text: str
    score: float

    def to_json(self) -> dict[str, Any]:
        return {"index": self.index, "text": self.text, "score": self.score}


@dataclass
class SearxResult:
    """A search result returned by a SearXNG instance."""

    title: str
    url: str
    snippet: str
    score: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet, "score": self.score}
