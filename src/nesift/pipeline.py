"""High-level orchestration: ingest URLs, run queries, run SearXNG-backed searches."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from nesift import answer as answer_mod
from nesift import dedup as dedup_mod
from nesift import searxng as searxng_mod
from nesift.budget import trim as budget_trim
from nesift.chunker import chunk_document
from nesift.config import DEFAULT_TOP_K
from nesift.embedder import EmbedderProtocol
from nesift.extractor import extract
from nesift.fetcher import fetch_raw
from nesift.index.hybrid import rerank, rrf
from nesift.pdf import extract_pdf, is_pdf_bytes, is_pdf_url
from nesift.schema import Chunk, Page, QueryResult, ScoredSnippet
from nesift.scorer import score_snippets
from nesift.session import SessionStore, page_id_for
from nesift.summarizer import triage
from nesift.tokens import count_tokens


@dataclass
class QueryResponse:
    """Wraps query results with budget metadata for downstream consumers."""

    results: list[QueryResult]
    budget_total: int | None
    budget_used: int


# ---------- ingestion ----------


def ingest_url(
    url: str,
    store: SessionStore,
    *,
    embedder: EmbedderProtocol | None = None,
    html: str | None = None,
    pdf_bytes: bytes | None = None,
) -> Page:
    """Fetch (if needed), extract, chunk, embed, and store one URL.

    Returns the resulting :class:`Page`. Replaces any existing page with
    the same URL. Dispatches to the PDF extractor when the URL ends in
    ``.pdf`` or the response body has a PDF signature; otherwise uses
    the trafilatura HTML extractor.
    """

    if pdf_bytes is not None:
        doc = extract_pdf(pdf_bytes, url=url)
    elif html is not None:
        doc = extract(html, url=url)
    elif is_pdf_url(url):
        body, _ = fetch_raw(url)
        doc = extract_pdf(body, url=url)
    else:
        body, ctype = fetch_raw(url)
        if "application/pdf" in ctype.lower() or is_pdf_bytes(body[:5]):
            doc = extract_pdf(body, url=url)
        else:
            doc = extract(body.decode("utf-8", errors="replace"), url=url)
    summary = triage(doc)
    pid = page_id_for(url)
    chunk_specs = chunk_document(doc)

    texts = [text for _, text in chunk_specs]
    embeddings: list[np.ndarray | None]
    if embedder is not None and texts:
        mat = embedder.embed_many(texts)
        embeddings = [mat[i] for i in range(mat.shape[0])]
    else:
        embeddings = [None] * len(texts)

    chunks: list[Chunk] = []
    for i, ((section, text), emb) in enumerate(zip(chunk_specs, embeddings, strict=False)):
        chunks.append(
            Chunk(
                id=f"{pid}:{i}",
                page_id=pid,
                url=url,
                section=section,
                text=text,
                token_count=count_tokens(text),
                embedding=emb,
            )
        )
    page = Page(
        id=pid,
        url=url,
        title=doc.title,
        fetched_at=time.time(),
        triage=summary,
        chunks=chunks,
    )
    store.add_page(page)
    return page


# ---------- query ----------


def run_query(
    query: str,
    store: SessionStore,
    *,
    embedder: EmbedderProtocol | None = None,
    top_k: int = DEFAULT_TOP_K,
    budget: int | None = None,
    url_filter: str | None = None,
    dedup: bool = True,
) -> QueryResponse:
    """Hybrid query against the session store."""

    chunks = store.chunks
    if not chunks:
        return QueryResponse(results=[], budget_total=budget, budget_used=0)

    mask: np.ndarray | None = None
    if url_filter is not None:
        target_pid = page_id_for(url_filter)
        mask = np.array([c.page_id == target_pid for c in chunks], dtype=bool)
        if not mask.any():
            return QueryResponse(results=[], budget_total=budget, budget_used=0)

    bm25_scores = store.bm25.scores(query)
    score_channels: list[np.ndarray] = [bm25_scores]

    vec_scores: np.ndarray | None = None
    if embedder is not None and len(store.vectors) == len(chunks) and len(chunks) > 0:
        try:
            q_vec = embedder.embed(query)
            vec_scores = store.vectors.scores(q_vec)
            score_channels.append(vec_scores)
        except Exception:
            vec_scores = None

    if mask is not None:
        for i, channel in enumerate(score_channels):
            score_channels[i] = np.where(mask, channel, 0.0)

    fused = rrf(score_channels)
    page_ids = [c.page_id for c in chunks]
    sections = [c.section for c in chunks]
    # Generate enough candidates to allow dedup pruning.
    candidate_k = max(top_k * 3, top_k + 5)
    ranked = rerank(
        fused,
        page_ids=page_ids,
        sections=sections,
        query=query,
        top_k=candidate_k,
    )
    if not ranked:
        return QueryResponse(results=[], budget_total=budget, budget_used=0)

    results: list[QueryResult] = []
    embeddings: list[np.ndarray | None] = []
    for idx, score in ranked:
        c = chunks[idx]
        results.append(
            QueryResult(
                chunk=c.text,
                url=c.url,
                section=c.section,
                score=float(score),
                sources=1,
                token_count=c.token_count,
            )
        )
        embeddings.append(c.embedding)

    if dedup:
        results = _zip_dedup(results, embeddings)

    if budget is not None:
        results, used = budget_trim(results, budget)
        return QueryResponse(results=results, budget_total=budget, budget_used=used)

    results = results[:top_k]
    used = sum(r.token_count for r in results)
    return QueryResponse(results=results, budget_total=None, budget_used=used)


def _zip_dedup(
    results: list[QueryResult], embeddings: list[np.ndarray | None]
) -> list[QueryResult]:
    return dedup_mod.collapse(results, embeddings)


# ---------- answer ----------


def run_answer(
    question: str,
    store: SessionStore,
    *,
    embedder: EmbedderProtocol | None = None,
    budget: int | None = 2000,
    url_filter: str | None = None,
    top_k: int = 6,
) -> tuple[str, QueryResponse]:
    """Run a query and return ``(answer_text, query_response)``."""

    response = run_query(
        question,
        store,
        embedder=embedder,
        top_k=top_k,
        budget=budget,
        url_filter=url_filter,
    )
    text = answer_mod.synthesize(question, response.results)
    return text, response


# ---------- pre-fetch scoring ----------


def run_score(
    query: str, snippets: list[str], embedder: EmbedderProtocol
) -> list[ScoredSnippet]:
    return score_snippets(query, snippets, embedder)


# ---------- searxng-backed one-shot search ----------


def run_search(
    query: str,
    store: SessionStore,
    *,
    embedder: EmbedderProtocol,
    top_n: int = 5,
    budget: int | None = 2000,
    instance_url: str | None = None,
    fetch_fn=None,
) -> tuple[str, QueryResponse, list[ScoredSnippet]]:
    """End-to-end SearXNG bridge: search → score → fetch+ingest → answer.

    ``fetch_fn`` lets callers (or tests) override the per-URL ingestion
    function. Defaults to :func:`ingest_url`.
    """

    raw_results = searxng_mod.search(query, top_n=top_n * 3, instance_url=instance_url)
    if not raw_results:
        return f"SearXNG returned no results for {query!r}", QueryResponse(
            results=[], budget_total=budget, budget_used=0
        ), []

    snippets = [r.snippet or r.title for r in raw_results]
    scored = score_snippets(query, snippets, embedder)
    selected_indices = [s.index for s in scored[:top_n]]
    selected = [raw_results[i] for i in selected_indices]

    ingestor = fetch_fn or (lambda u, s=store: ingest_url(u, s, embedder=embedder))
    for r in selected:
        if store.has_url(r.url):
            continue
        try:
            ingestor(r.url)
        except Exception:
            continue

    text, response = run_answer(query, store, embedder=embedder, budget=budget)
    return text, response, scored[:top_n]
