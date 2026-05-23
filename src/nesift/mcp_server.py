"""MCP server exposing nesift as tools for AI agents.

Run via ``uvx --from "nesift[mcp]" nesift-mcp`` or
``python -m nesift.mcp_server``. Communicates over stdio.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from nesift import __version__
from nesift.embedder import Embedder
from nesift.pipeline import (
    ingest_url,
    run_answer,
    run_query,
    run_score,
    run_search,
)
from nesift.session import SessionStore

log = logging.getLogger("nesift.mcp")

server = Server("nesift")

# Lazily-created shared state for the MCP process. A single embedder + store
# is reused across all tool calls in a session, so the model only loads once.
_state: dict[str, Any] = {}


def _embedder(fast: bool = False, lang: bool = False) -> Embedder:
    key = ("fast" if fast else ("lang" if lang else "default"))
    emb = _state.setdefault("embedders", {}).get(key)
    if emb is None:
        from nesift.config import FAST_MODEL, MULTILINGUAL_MODEL, DEFAULT_MODEL

        name = FAST_MODEL if fast else (MULTILINGUAL_MODEL if lang else DEFAULT_MODEL)
        emb = Embedder(name)
        _state["embedders"][key] = emb
    return emb


def _store() -> SessionStore:
    s = _state.get("store")
    if s is None:
        s = SessionStore()
        s.load()
        _state["store"] = s
    return s


def _persist(s: SessionStore) -> None:
    s.save()


def _text(payload: object) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps(payload, indent=2, default=str))]


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="score_snippets",
            description=(
                "Rank text snippets by semantic relevance to a query, without fetching "
                "any pages. Use this BEFORE downloading search results to skip "
                "irrelevant pages and save tokens."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "snippets": {"type": "array", "items": {"type": "string"}},
                    "top_k": {"type": "integer", "default": 0, "description": "0 = return all"},
                },
                "required": ["query", "snippets"],
            },
        ),
        Tool(
            name="add_page",
            description=(
                "Fetch a URL, extract clean text, chunk by heading, and index it into "
                "the active session. Returns title + chunk count + triage summary."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "fast": {"type": "boolean", "default": False},
                    "lang": {"type": "boolean", "default": False, "description": "Use multilingual model"},
                },
                "required": ["url"],
            },
        ),
        Tool(
            name="add_batch",
            description="Ingest multiple URLs sequentially.",
            inputSchema={
                "type": "object",
                "properties": {
                    "urls": {"type": "array", "items": {"type": "string"}},
                    "fast": {"type": "boolean", "default": False},
                    "lang": {"type": "boolean", "default": False},
                },
                "required": ["urls"],
            },
        ),
        Tool(
            name="query",
            description=(
                "Hybrid BM25 + embedding semantic search over indexed pages. "
                "Use --budget to fit results within a token budget."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "q": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                    "budget": {"type": "integer", "default": 0, "description": "0 = no budget"},
                    "url_filter": {"type": "string", "default": ""},
                },
                "required": ["q"],
            },
        ),
        Tool(
            name="answer",
            description="Synthesize an extractive answer with [N] citations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "q": {"type": "string"},
                    "budget": {"type": "integer", "default": 2000},
                    "url_filter": {"type": "string", "default": ""},
                },
                "required": ["q"],
            },
        ),
        Tool(
            name="list_pages",
            description="Show indexed URLs with title, chunk count, and triage summary.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="clear",
            description="Drop all indexed pages from the active session.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="search",
            description=(
                "One-shot SearXNG bridge: search → score → fetch top results → ingest → answer."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "q": {"type": "string"},
                    "top": {"type": "integer", "default": 5},
                    "budget": {"type": "integer", "default": 2000},
                    "instance_url": {"type": "string", "default": ""},
                },
                "required": ["q"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any] | None) -> list[TextContent]:
    args = arguments or {}
    if name == "score_snippets":
        emb = _embedder()
        ranked = run_score(args["query"], list(args.get("snippets", [])), emb)
        if args.get("top_k"):
            ranked = ranked[: int(args["top_k"])]
        return _text([s.to_json() for s in ranked])

    if name == "add_page":
        store = _store()
        emb = _embedder(fast=args.get("fast", False), lang=args.get("lang", False))
        page = ingest_url(args["url"], store, embedder=emb)
        _persist(store)
        return _text({
            "url": page.url,
            "title": page.title,
            "chunks": len(page.chunks),
            "triage": page.triage,
        })

    if name == "add_batch":
        store = _store()
        emb = _embedder(fast=args.get("fast", False), lang=args.get("lang", False))
        rows: list[dict[str, Any]] = []
        for u in args.get("urls", []):
            try:
                page = ingest_url(u, store, embedder=emb)
                rows.append({"url": u, "ok": True, "chunks": len(page.chunks)})
            except Exception as exc:
                rows.append({"url": u, "ok": False, "error": str(exc)})
        _persist(store)
        return _text({"pages": rows})

    if name == "query":
        store = _store()
        emb = _embedder()
        budget = args.get("budget") or None
        url = args.get("url_filter") or None
        resp = run_query(
            args["q"],
            store,
            embedder=emb,
            top_k=int(args.get("top_k", 5)),
            budget=budget if budget else None,
            url_filter=url,
        )
        return _text({
            "results": [r.to_json() for r in resp.results],
            "budget_total": resp.budget_total,
            "budget_used": resp.budget_used,
        })

    if name == "answer":
        store = _store()
        emb = _embedder()
        budget = args.get("budget", 2000) or None
        url = args.get("url_filter") or None
        text, resp = run_answer(args["q"], store, embedder=emb, budget=budget, url_filter=url)
        return _text({
            "answer": text,
            "results": [r.to_json() for r in resp.results],
            "budget_total": resp.budget_total,
            "budget_used": resp.budget_used,
        })

    if name == "list_pages":
        store = _store()
        return _text([
            {"url": p.url, "title": p.title, "chunks": len(p.chunks), "triage": p.triage}
            for p in store.pages
        ])

    if name == "clear":
        store = _store()
        store.clear()
        _state["store"] = store
        return _text({"status": "cleared"})

    if name == "search":
        store = _store()
        emb = _embedder()
        instance = args.get("instance_url") or None
        text, resp, scored = run_search(
            args["q"],
            store,
            embedder=emb,
            top_n=int(args.get("top", 5)),
            budget=int(args.get("budget", 2000)) or None,
            instance_url=instance,
        )
        _persist(store)
        return _text({
            "answer": text,
            "results": [r.to_json() for r in resp.results],
            "snippets": [s.to_json() for s in scored],
        })

    return _text({"error": f"unknown tool: {name}"})


async def _serve() -> None:
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> None:
    """Console-script entrypoint for ``nesift-mcp``."""

    logging.basicConfig(level=logging.WARNING)
    log.info("starting nesift MCP server v%s", __version__)
    asyncio.run(_serve())


if __name__ == "__main__":  # pragma: no cover
    main()
