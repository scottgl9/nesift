"""High-level orchestration: ingest URLs, run queries, run SearXNG-backed searches."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from nesift import answer as answer_mod
from nesift import cache as cache_mod
from nesift import dedup as dedup_mod
from nesift import searxng as searxng_mod
from nesift.budget import trim as budget_trim
from nesift.chunker import chunk_document
from nesift.config import DEFAULT_TOP_K
from nesift.embedder import EmbedderProtocol
from nesift.extractor import extract
from nesift.fetcher import FetchResult, fetch_many, fetch_raw
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
    use_cache: bool = True,
) -> Page:
    """Fetch (if needed), extract, chunk, embed, and store one URL.

    Returns the resulting :class:`Page`. Replaces any existing page with
    the same URL. Dispatches to the PDF extractor when the URL ends in
    ``.pdf`` or the response body has a PDF signature; otherwise uses
    the trafilatura HTML extractor.

    A persistent cache under ``$NESIFT_CACHE_DIR`` (default
    ``~/.cache/nesift/pages``) skips the entire fetch/extract/chunk/embed
    pipeline on a hit. Pass ``html=`` or ``pdf_bytes=`` (e.g. for tests)
    to bypass the network and also bypass the cache.
    """

    inline_body = html is not None or pdf_bytes is not None
    model = _model_name(embedder) if not inline_body else None
    if use_cache and not inline_body:
        cached = cache_mod.get(url, model)
        if cached is not None and _cache_matches(cached, embedder):
            store.add_page(cached)
            return cached

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
    if use_cache and not inline_body:
        cache_mod.put(page, model)
    return page


def _model_name(embedder: EmbedderProtocol | None) -> str | None:
    if embedder is None:
        return None
    return getattr(embedder, "model_name", None)


def _cache_matches(page: Page, embedder: EmbedderProtocol | None) -> bool:
    """If an embedder is configured, the cached page must have embeddings."""

    if embedder is None:
        return True
    return all(c.embedding is not None for c in page.chunks)


@dataclass
class BatchResult:
    """Per-URL outcome from :func:`ingest_urls`."""

    url: str
    ok: bool
    page: Page | None = None
    error: str | None = None


def ingest_urls(
    urls: list[str],
    store: SessionStore,
    *,
    embedder: EmbedderProtocol | None = None,
    concurrency: int = 8,
    use_cache: bool = True,
) -> list[BatchResult]:
    """Fetch many URLs concurrently and index them with a single batched embed.

    Network I/O is parallelized via :func:`fetch_many`; per-URL failures
    are reported in the returned :class:`BatchResult` list rather than
    raised. The embedding model is invoked **once** across every chunk
    of every successful page, which avoids per-URL model dispatch
    overhead. Cache hits short-circuit before any network call.
    """

    if not urls:
        return []

    model = _model_name(embedder)
    cached_pages: dict[str, Page] = {}
    if use_cache:
        for u in urls:
            page = cache_mod.get(u, model)
            if page is not None and _cache_matches(page, embedder):
                cached_pages[u] = page

    misses = [u for u in urls if u not in cached_pages]
    fetched = fetch_many(misses, concurrency=concurrency) if misses else []
    fetched_by_url = {fr.url: fr for fr in fetched}

    extracted: list[tuple[FetchResult, object | None, str | None]] = []
    for fr in fetched:
        if not fr.ok:
            extracted.append((fr, None, fr.error))
            continue
        try:
            doc = _extract_for(fr)
        except Exception as exc:
            extracted.append((fr, None, str(exc)))
            continue
        extracted.append((fr, doc, None))

    # Flatten all chunks across all successful pages for one batched embed call.
    flat_texts: list[str] = []
    flat_owner: list[int] = []  # index into extracted/page-builder list
    per_page_chunks: list[list[tuple[str | None, str]]] = []
    for i, (_fr, doc, err) in enumerate(extracted):
        if err or doc is None:
            per_page_chunks.append([])
            continue
        chunks = chunk_document(doc)
        per_page_chunks.append(chunks)
        for _, text in chunks:
            flat_texts.append(text)
            flat_owner.append(i)

    flat_embeddings: list[np.ndarray | None]
    if embedder is not None and flat_texts:
        mat = embedder.embed_many(flat_texts)
        flat_embeddings = [mat[i] for i in range(mat.shape[0])]
    else:
        flat_embeddings = [None] * len(flat_texts)

    # Distribute embeddings back to their owning pages, indexed by URL.
    cursor = 0
    built: dict[str, BatchResult] = {}
    for i, (fr, doc, err) in enumerate(extracted):
        if err or doc is None:
            built[fr.url] = BatchResult(url=fr.url, ok=False, error=err or "extraction failed")
            continue
        chunks_spec = per_page_chunks[i]
        n = len(chunks_spec)
        page_embs = flat_embeddings[cursor : cursor + n]
        cursor += n
        pid = page_id_for(fr.url)
        chunks = [
            Chunk(
                id=f"{pid}:{j}",
                page_id=pid,
                url=fr.url,
                section=section,
                text=text,
                token_count=count_tokens(text),
                embedding=page_embs[j],
            )
            for j, (section, text) in enumerate(chunks_spec)
        ]
        page = Page(
            id=pid,
            url=fr.url,
            title=doc.title,  # type: ignore[union-attr]
            fetched_at=time.time(),
            triage=triage(doc),  # type: ignore[arg-type]
            chunks=chunks,
        )
        store.add_page(page)
        if use_cache:
            cache_mod.put(page, model)
        built[fr.url] = BatchResult(url=fr.url, ok=True, page=page)

    # Assemble final result in input order; promote cache hits into the store.
    out: list[BatchResult] = []
    for u in urls:
        if u in cached_pages:
            page = cached_pages[u]
            store.add_page(page)
            out.append(BatchResult(url=u, ok=True, page=page))
        elif u in built:
            out.append(built[u])
        else:
            # Should not happen: every URL is either cached or fetched.
            fr = fetched_by_url.get(u)
            err = fr.error if fr else "missing"
            out.append(BatchResult(url=u, ok=False, error=err))
    return out


def _extract_for(fr: FetchResult):
    """Pick the right extractor based on URL or response body."""

    assert fr.body is not None
    ctype = (fr.content_type or "").lower()
    if is_pdf_url(fr.url) or "application/pdf" in ctype or is_pdf_bytes(fr.body[:5]):
        return extract_pdf(fr.body, url=fr.url)
    return extract(fr.body.decode("utf-8", errors="replace"), url=fr.url)


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
