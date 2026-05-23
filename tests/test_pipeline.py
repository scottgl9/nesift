from __future__ import annotations

import httpx
import respx

from nesift.pipeline import ingest_url, run_answer, run_query, run_search


def test_ingest_url_offline(session, fake_embedder, fixture_html):
    page = ingest_url(
        "https://example.com/blog",
        session,
        embedder=fake_embedder,
        html=fixture_html["blog_post"],
    )
    assert page.chunks
    # Triage should not be empty for a real blog post.
    assert page.triage
    # Every chunk should have an embedding now.
    assert all(c.embedding is not None for c in page.chunks)


def test_query_finds_relevant_chunk(session, fake_embedder, fixture_html):
    ingest_url(
        "https://example.com/blog",
        session,
        embedder=fake_embedder,
        html=fixture_html["blog_post"],
    )
    resp = run_query(
        "exponential backoff with jitter",
        session,
        embedder=fake_embedder,
        top_k=3,
    )
    assert resp.results
    joined = " ".join(r.chunk for r in resp.results).lower()
    assert "backoff" in joined or "jitter" in joined


def test_query_url_filter(session, fake_embedder, fixture_html):
    ingest_url("https://a.test/", session, embedder=fake_embedder, html=fixture_html["blog_post"])
    ingest_url("https://b.test/", session, embedder=fake_embedder, html=fixture_html["docs_page"])
    resp = run_query(
        "retry",
        session,
        embedder=fake_embedder,
        top_k=5,
        url_filter="https://b.test/",
    )
    assert all(r.url == "https://b.test/" for r in resp.results)


def test_query_dedups_across_pages(session, fake_embedder, fixture_html):
    ingest_url("https://orig.test/", session, embedder=fake_embedder, html=fixture_html["blog_post"])
    ingest_url(
        "https://mirror.test/", session, embedder=fake_embedder, html=fixture_html["duplicate_para"]
    )
    resp = run_query(
        "exponential backoff jitter",
        session,
        embedder=fake_embedder,
        top_k=10,
    )
    # At least one result should have sources >= 2 (the duplicate paragraph).
    assert any(r.sources >= 2 for r in resp.results)


def test_query_budget(session, fake_embedder, fixture_html):
    ingest_url(
        "https://wiki.test/RAG",
        session,
        embedder=fake_embedder,
        html=fixture_html["wikipedia_article"],
    )
    resp = run_query("RAG", session, embedder=fake_embedder, top_k=10, budget=80)
    assert resp.budget_total == 80
    assert resp.budget_used <= 80 or len(resp.results) == 1


def test_answer_produces_citations(session, fake_embedder, fixture_html):
    ingest_url(
        "https://wiki.test/RAG",
        session,
        embedder=fake_embedder,
        html=fixture_html["wikipedia_article"],
    )
    text, resp = run_answer("what is RAG", session, embedder=fake_embedder, budget=500)
    assert "[1]" in text
    assert "https://wiki.test/RAG" in text
    assert resp.results


@respx.mock
def test_run_search_orchestration(session, fake_embedder, fixture_html):
    # Mock SearXNG search.
    respx.get("http://127.0.0.1:8888/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Designing Resilient Retry Logic",
                        "url": "https://blog.test/retries",
                        "content": "exponential backoff with jitter circuit breaker",
                        "score": 1.0,
                    },
                    {
                        "title": "Cake recipes",
                        "url": "https://bake.test/cake",
                        "content": "flour eggs butter sugar oven",
                        "score": 0.1,
                    },
                ]
            },
        )
    )
    # Mock the page fetch for the only URL we expect to ingest.
    respx.get("https://blog.test/retries").mock(
        return_value=httpx.Response(200, text=fixture_html["blog_post"])
    )
    respx.get("https://bake.test/cake").mock(
        return_value=httpx.Response(200, text="<html><body><p>cake stuff</p></body></html>")
    )
    text, resp, scored = run_search(
        "retry logic in distributed systems",
        session,
        embedder=fake_embedder,
        top_n=1,
        budget=1000,
    )
    assert scored
    assert resp.results
    assert "https://blog.test/retries" in text
