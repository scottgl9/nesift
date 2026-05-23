from __future__ import annotations

import httpx
import respx

from nesift.fetcher import fetch_many
from nesift.pipeline import ingest_urls


@respx.mock
def test_fetch_many_preserves_order_and_reports_errors():
    respx.get("https://a.test/").mock(return_value=httpx.Response(200, text="A"))
    respx.get("https://b.test/").mock(return_value=httpx.Response(500, text="oops"))
    respx.get("https://c.test/").mock(side_effect=httpx.ConnectError("dns"))
    out = fetch_many(["https://a.test/", "https://b.test/", "https://c.test/"], concurrency=4)
    assert [r.url for r in out] == ["https://a.test/", "https://b.test/", "https://c.test/"]
    assert out[0].ok and out[0].body == b"A"
    assert not out[1].ok and "500" in (out[1].error or "")
    assert not out[2].ok and out[2].body is None


@respx.mock
def test_ingest_urls_concurrent(session, fake_embedder, fixture_html):
    urls = [
        "https://blog.test/r",
        "https://wiki.test/RAG",
        "https://docs.test/auth",
        "https://broken.test/",
    ]
    respx.get(urls[0]).mock(return_value=httpx.Response(200, text=fixture_html["blog_post"]))
    respx.get(urls[1]).mock(return_value=httpx.Response(200, text=fixture_html["wikipedia_article"]))
    respx.get(urls[2]).mock(return_value=httpx.Response(200, text=fixture_html["docs_page"]))
    respx.get(urls[3]).mock(return_value=httpx.Response(503, text="down"))

    out = ingest_urls(urls, session, embedder=fake_embedder, concurrency=4)
    assert [r.url for r in out] == urls
    assert out[0].ok and out[0].page is not None and out[0].page.chunks
    assert out[1].ok and out[2].ok
    assert not out[3].ok and "503" in (out[3].error or "")
    # All three good pages landed in the store.
    assert len(session.pages) == 3


@respx.mock
def test_ingest_urls_empty():
    from nesift.session import SessionStore

    s = SessionStore.__new__(SessionStore)
    s.__init__()
    assert ingest_urls([], s) == []
