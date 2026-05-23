"""Smoke tests for the MCP server: list_tools + a couple of call_tool routes."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import respx

mcp_server = pytest.importorskip("nesift.mcp_server")


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch, tmp_path, fake_embedder):
    """Force the MCP module to use a clean store and the fake embedder."""

    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("NESIFT_SESSION", f"mcp-{tmp_path.name}")
    mcp_server._state.clear()
    # Inject fake embedder under all keys it may look up.
    monkeypatch.setattr(mcp_server, "_embedder", lambda fast=False, lang=False: fake_embedder)
    yield
    mcp_server._state.clear()


def _call(name: str, args: dict | None = None) -> dict | list:
    out = asyncio.run(mcp_server.call_tool(name, args or {}))
    assert len(out) == 1
    return json.loads(out[0].text)


def test_list_tools_includes_expected():
    tools = asyncio.run(mcp_server.list_tools())
    names = {t.name for t in tools}
    assert {
        "score_snippets",
        "add_page",
        "add_batch",
        "query",
        "answer",
        "list_pages",
        "clear",
        "search",
    }.issubset(names)


def test_score_snippets_route():
    out = _call(
        "score_snippets",
        {
            "query": "vector database",
            "snippets": ["Pinecone vector DB", "How to bake bread"],
            "top_k": 1,
        },
    )
    assert isinstance(out, list)
    assert len(out) == 1


@respx.mock
def test_add_page_and_query_routes(fixture_html):
    respx.get("https://blog.test/r").mock(
        return_value=httpx.Response(200, text=fixture_html["blog_post"])
    )
    page = _call("add_page", {"url": "https://blog.test/r"})
    assert page["chunks"] > 0

    listed = _call("list_pages", {})
    assert listed and listed[0]["url"] == "https://blog.test/r"

    q = _call("query", {"q": "exponential backoff", "top_k": 3})
    assert q["results"]


@respx.mock
def test_answer_route(fixture_html):
    respx.get("https://blog.test/r").mock(
        return_value=httpx.Response(200, text=fixture_html["blog_post"])
    )
    _call("add_page", {"url": "https://blog.test/r"})
    ans = _call("answer", {"q": "what is exponential backoff", "budget": 800})
    assert "[1]" in ans["answer"]


def test_clear_route():
    out = _call("clear", {})
    assert out["status"] == "cleared"
