"""Network-gated end-to-end checks.

Run with::

    pytest -m validation

Skips automatically when the network is unreachable or
``NESIFT_SKIP_NETWORK=1`` is set.
"""

from __future__ import annotations

import os

import pytest

from nesift.embedder import FakeEmbedder
from nesift.fetcher import FetchError, fetch
from nesift.pipeline import ingest_url, run_answer, run_query
from nesift.session import SessionStore

pytestmark = pytest.mark.validation

LIVE_URL = "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"


def _network_ok() -> bool:
    if os.environ.get("NESIFT_SKIP_NETWORK"):
        return False
    try:
        fetch("https://example.com/")
    except FetchError:
        return False
    return True


@pytest.fixture(scope="module")
def online() -> bool:
    if not _network_ok():
        pytest.skip("network unavailable or NESIFT_SKIP_NETWORK set")
    return True


def test_live_wikipedia_ingest_and_query(tmp_path, online):
    store = SessionStore(tmp_path / "live.json")
    embedder = FakeEmbedder(dim=64)

    page = ingest_url(LIVE_URL, store, embedder=embedder)
    assert page.chunks, "expected non-empty chunks"
    assert page.title
    assert page.triage, "expected a triage summary"

    resp = run_query(
        "what is retrieval-augmented generation",
        store,
        embedder=embedder,
        top_k=5,
        budget=2000,
    )
    assert resp.results
    joined = " ".join(r.chunk for r in resp.results).lower()
    assert "retrieval" in joined


def test_live_answer_has_citation(tmp_path, online):
    store = SessionStore(tmp_path / "live-ans.json")
    embedder = FakeEmbedder(dim=64)
    ingest_url(LIVE_URL, store, embedder=embedder)
    text, _ = run_answer("how does RAG reduce hallucinations", store, embedder=embedder, budget=1500)
    assert "[1]" in text
    assert LIVE_URL in text


@pytest.mark.skipif(
    not os.environ.get("NESIFT_SEARXNG_URL"),
    reason="set NESIFT_SEARXNG_URL to run the live SearXNG smoke test",
)
def test_live_searxng_bridge(tmp_path):
    from nesift.pipeline import run_search

    store = SessionStore(tmp_path / "live-sx.json")
    embedder = FakeEmbedder(dim=64)
    try:
        text, resp, scored = run_search(
            "retry logic in distributed systems",
            store,
            embedder=embedder,
            top_n=3,
            budget=1500,
        )
    except Exception as exc:
        pytest.skip(f"SearXNG instance unreachable: {exc}")
    assert scored
    assert resp.results or "no results" in text.lower()
