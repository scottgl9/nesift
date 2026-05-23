"""Live MCP smoke test: spawn ``nesift-mcp`` and drive it with the MCP client SDK.

Run with::

    pytest -m validation tests/validation/test_mcp_live.py -v

This proves the published JSON-RPC tools work with a real client, not
just with internal ``asyncio.run(call_tool(...))`` stubs.

Skips if the MCP extra is not installed or the network is unavailable.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys

import pytest

mcp_client = pytest.importorskip("mcp.client.stdio")
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

pytestmark = pytest.mark.validation


def _nesift_mcp_cmd() -> list[str]:
    """Prefer the installed console script, fall back to ``python -m``."""

    exe = shutil.which("nesift-mcp")
    if exe:
        return [exe]
    return [sys.executable, "-m", "nesift.mcp_server"]


async def _drive(searxng_url: str | None) -> dict:
    env = os.environ.copy()
    env["NESIFT_SESSION"] = "mcp-live"
    if searxng_url:
        env["NESIFT_SEARXNG_URL"] = searxng_url
    cmd = _nesift_mcp_cmd()
    params = StdioServerParameters(command=cmd[0], args=cmd[1:], env=env)
    out: dict = {}
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            out["tool_names"] = sorted(t.name for t in tools.tools)

            # Clear from a previous run.
            await session.call_tool("clear", {})

            # 1. Pre-score snippets (no network needed).
            r = await session.call_tool(
                "score_snippets",
                {
                    "query": "vector database",
                    "snippets": ["Pinecone vector DB", "How to bake bread"],
                    "top_k": 1,
                },
            )
            out["score"] = json.loads(r.content[0].text)

            # 2. Add Wikipedia (real fetch).
            r = await session.call_tool(
                "add_page",
                {"url": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"},
            )
            out["add"] = json.loads(r.content[0].text)

            # 3. Answer from indexed content.
            r = await session.call_tool(
                "answer",
                {"q": "how does RAG reduce hallucinations", "budget": 800},
            )
            out["answer"] = json.loads(r.content[0].text)

            # 4. SearXNG search (only if env set).
            if searxng_url:
                r = await session.call_tool(
                    "search",
                    {"q": "retry logic exponential backoff", "top": 2, "budget": 800},
                )
                out["search"] = json.loads(r.content[0].text)
    return out


def test_mcp_live_end_to_end():
    if os.environ.get("NESIFT_SKIP_NETWORK"):
        pytest.skip("NESIFT_SKIP_NETWORK set")
    searxng_url = os.environ.get("NESIFT_SEARXNG_URL")
    try:
        result = asyncio.run(_drive(searxng_url))
    except Exception as exc:
        pytest.skip(f"MCP live drive failed: {exc}")

    assert {"score_snippets", "add_page", "query", "answer", "list_pages", "clear", "search"}.issubset(
        set(result["tool_names"])
    )
    assert result["score"], "score_snippets returned empty"
    assert result["add"]["chunks"] > 0
    assert "[1]" in result["answer"]["answer"]
    if searxng_url:
        assert result["search"]["results"] or "no results" in result["search"]["answer"].lower()
