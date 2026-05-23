from __future__ import annotations

import httpx
import respx

from nesift import cache as cache_mod
from nesift.pipeline import ingest_url, ingest_urls


def test_cache_roundtrip(tmp_path, monkeypatch, fake_embedder, fixture_html, session):
    monkeypatch.setenv("NESIFT_CACHE_DIR", str(tmp_path / "cache"))
    with respx.mock() as router:
        route = router.get("https://blog.test/r").mock(
            return_value=httpx.Response(200, text=fixture_html["blog_post"])
        )
        ingest_url("https://blog.test/r", session, embedder=fake_embedder)
        assert route.call_count == 1

    # Fresh store; second ingest should NOT hit the network because the cache is warm.
    from nesift.session import SessionStore

    s2 = SessionStore(tmp_path / "s2.json")
    with respx.mock() as router2:
        # No respx route registered → if pipeline hit the network, respx would error.
        page = ingest_url("https://blog.test/r", s2, embedder=fake_embedder)
        assert router2.calls.call_count == 0
    assert page.chunks


def test_cache_no_cache_flag_bypasses(tmp_path, monkeypatch, fake_embedder, fixture_html, session):
    monkeypatch.setenv("NESIFT_CACHE_DIR", str(tmp_path / "cache"))
    with respx.mock() as router:
        router.get("https://blog.test/r").mock(
            return_value=httpx.Response(200, text=fixture_html["blog_post"])
        )
        ingest_url("https://blog.test/r", session, embedder=fake_embedder)

    from nesift.session import SessionStore

    s2 = SessionStore(tmp_path / "s2.json")
    with respx.mock() as router2:
        route = router2.get("https://blog.test/r").mock(
            return_value=httpx.Response(200, text=fixture_html["blog_post"])
        )
        ingest_url("https://blog.test/r", s2, embedder=fake_embedder, use_cache=False)
        assert route.call_count == 1


def test_cache_disabled_env(tmp_path, monkeypatch, fake_embedder, fixture_html, session):
    monkeypatch.setenv("NESIFT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("NESIFT_NO_CACHE", "1")
    with respx.mock() as router:
        router.get("https://blog.test/r").mock(
            return_value=httpx.Response(200, text=fixture_html["blog_post"])
        )
        ingest_url("https://blog.test/r", session, embedder=fake_embedder)
    # Cache dir should remain empty.
    assert not list((tmp_path / "cache").glob("*.json")) if (tmp_path / "cache").exists() else True


def test_cache_keyed_on_model(tmp_path, monkeypatch, fake_embedder, fixture_html, session):
    """Different embedder model names → distinct cache entries."""

    monkeypatch.setenv("NESIFT_CACHE_DIR", str(tmp_path / "cache"))
    with respx.mock() as router:
        router.get("https://blog.test/r").mock(
            return_value=httpx.Response(200, text=fixture_html["blog_post"])
        )
        ingest_url("https://blog.test/r", session, embedder=fake_embedder)

    files = list((tmp_path / "cache").glob("*.json"))
    assert len(files) == 1

    # Different fake embedder with a different model_name → separate entry.
    class _Other:
        model_name = "minishlab/potion-other"
        dim = 64

        def embed(self, t):
            return fake_embedder.embed(t)

        def embed_many(self, ts):
            return fake_embedder.embed_many(ts)

    from nesift.session import SessionStore

    s2 = SessionStore(tmp_path / "s2.json")
    with respx.mock() as router2:
        router2.get("https://blog.test/r").mock(
            return_value=httpx.Response(200, text=fixture_html["blog_post"])
        )
        ingest_url("https://blog.test/r", s2, embedder=_Other())
    assert len(list((tmp_path / "cache").glob("*.json"))) == 2


def test_cache_clear_and_stats(tmp_path, monkeypatch, fake_embedder, fixture_html, session):
    monkeypatch.setenv("NESIFT_CACHE_DIR", str(tmp_path / "cache"))
    with respx.mock() as router:
        router.get("https://a.test/").mock(
            return_value=httpx.Response(200, text=fixture_html["blog_post"])
        )
        router.get("https://b.test/").mock(
            return_value=httpx.Response(200, text=fixture_html["docs_page"])
        )
        ingest_urls(["https://a.test/", "https://b.test/"], session, embedder=fake_embedder)
    s = cache_mod.stats()
    assert s["entries"] == 2
    assert s["bytes"] > 0
    n = cache_mod.clear()
    assert n == 2
    assert cache_mod.stats()["entries"] == 0


@respx.mock
def test_ingest_urls_mixes_cache_and_fetch(
    tmp_path, monkeypatch, fake_embedder, fixture_html
):
    monkeypatch.setenv("NESIFT_CACHE_DIR", str(tmp_path / "cache"))
    # Pre-warm the cache for one URL.
    from nesift.session import SessionStore

    prime = SessionStore(tmp_path / "prime.json")
    respx.get("https://cached.test/").mock(
        return_value=httpx.Response(200, text=fixture_html["blog_post"])
    )
    ingest_url("https://cached.test/", prime, embedder=fake_embedder)

    # Now ingest_urls with one cached + one fresh URL.
    respx.get("https://fresh.test/").mock(
        return_value=httpx.Response(200, text=fixture_html["docs_page"])
    )
    s2 = SessionStore(tmp_path / "s2.json")
    results = ingest_urls(
        ["https://cached.test/", "https://fresh.test/"], s2, embedder=fake_embedder
    )
    assert all(r.ok for r in results)
    assert [r.url for r in results] == ["https://cached.test/", "https://fresh.test/"]
    assert len(s2.pages) == 2
