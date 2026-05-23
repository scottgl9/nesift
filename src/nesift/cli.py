"""Typer-based CLI surface for nesift."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from nesift import __version__
from nesift.config import DEFAULT_TOP_K, session_path
from nesift.embedder import Embedder
from nesift.pipeline import (
    ingest_url,
    run_answer,
    run_query,
    run_score,
    run_search,
)
from nesift.session import SessionStore

app = typer.Typer(
    help="Fast, local semantic search over web content for AI agents.",
    no_args_is_help=True,
    add_completion=False,
)


def _load_store(session: str | None) -> SessionStore:
    store = SessionStore(session_path(session) if session else None)
    store.load()
    return store


def _persist(store: SessionStore) -> None:
    store.save()


def _embedder(fast: bool, lang: bool = False) -> Embedder:
    return Embedder(fast=fast, lang=lang)


def _emit(payload: object, *, as_json: bool) -> None:
    if as_json:
        typer.echo(json.dumps(payload, indent=2))
    else:
        typer.echo(payload)


# ---------- ingestion commands ----------


@app.command("add")
def cmd_add(
    url: str = typer.Argument(..., help="URL to fetch and index."),
    fast: bool = typer.Option(False, "--fast", help="Use the smaller potion-base-8M model."),
    lang: bool = typer.Option(False, "--lang", help="Use potion-multilingual-128M (101 languages)."),
    no_embed: bool = typer.Option(False, "--no-embed", help="Skip embeddings (BM25 only)."),
    session: str | None = typer.Option(None, "--session"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Fetch a URL, extract, chunk, and index it into the session."""

    store = _load_store(session)
    embedder = None if no_embed else _embedder(fast, lang)
    try:
        page = ingest_url(url, store, embedder=embedder)
    except Exception as exc:
        typer.secho(f"add failed: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    _persist(store)
    payload = {
        "url": page.url,
        "title": page.title,
        "chunks": len(page.chunks),
        "triage": page.triage,
    }
    if as_json:
        _emit(payload, as_json=True)
    else:
        typer.echo(f"Indexed: {page.title}")
        typer.echo(f"  url:    {page.url}")
        typer.echo(f"  chunks: {len(page.chunks)}")
        if page.triage:
            typer.echo(f"  triage: {page.triage}")


@app.command("add-batch")
def cmd_add_batch(
    urls: list[str] = typer.Argument(..., help="URLs to ingest."),
    fast: bool = typer.Option(False, "--fast"),
    lang: bool = typer.Option(False, "--lang"),
    no_embed: bool = typer.Option(False, "--no-embed"),
    session: str | None = typer.Option(None, "--session"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Ingest multiple URLs sequentially; continues past per-URL failures."""

    store = _load_store(session)
    embedder = None if no_embed else _embedder(fast, lang)
    summary: list[dict] = []
    for url in urls:
        try:
            page = ingest_url(url, store, embedder=embedder)
            summary.append({"url": url, "ok": True, "chunks": len(page.chunks)})
        except Exception as exc:
            summary.append({"url": url, "ok": False, "error": str(exc)})
    _persist(store)
    if as_json:
        _emit({"pages": summary}, as_json=True)
    else:
        for row in summary:
            if row["ok"]:
                typer.echo(f"OK  {row['url']} ({row['chunks']} chunks)")
            else:
                typer.secho(f"ERR {row['url']}: {row['error']}", fg=typer.colors.YELLOW)


# ---------- query / answer / score ----------


@app.command("query")
def cmd_query(
    q: str = typer.Argument(..., help="Natural-language query."),
    top_k: int = typer.Option(DEFAULT_TOP_K, "--top-k"),
    budget: int | None = typer.Option(None, "--budget", help="Trim to N tokens."),
    url: str | None = typer.Option(None, "--url", help="Restrict to a single indexed URL."),
    fast: bool = typer.Option(False, "--fast"),
    lang: bool = typer.Option(False, "--lang"),
    no_embed: bool = typer.Option(False, "--no-embed"),
    session: str | None = typer.Option(None, "--session"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Retrieve relevant chunks across indexed pages."""

    store = _load_store(session)
    embedder = None if no_embed else _embedder(fast, lang)
    resp = run_query(
        q,
        store,
        embedder=embedder,
        top_k=top_k,
        budget=budget,
        url_filter=url,
    )
    if as_json:
        _emit(
            {
                "results": [r.to_json() for r in resp.results],
                "budget_total": resp.budget_total,
                "budget_used": resp.budget_used,
            },
            as_json=True,
        )
        return
    if not resp.results:
        typer.echo("(no results)")
        return
    for i, r in enumerate(resp.results, start=1):
        header = f"[{i}] score={r.score:.3f}  sources={r.sources}  {r.url}"
        if r.section:
            header += f"  §{r.section}"
        typer.echo(header)
        typer.echo(r.chunk)
        typer.echo("")
    if resp.budget_total is not None:
        typer.echo(f"-- budget: {resp.budget_used}/{resp.budget_total} tokens")


@app.command("answer")
def cmd_answer(
    q: str = typer.Argument(..., help="Question to answer from indexed pages."),
    budget: int | None = typer.Option(2000, "--budget"),
    url: str | None = typer.Option(None, "--url"),
    fast: bool = typer.Option(False, "--fast"),
    lang: bool = typer.Option(False, "--lang"),
    no_embed: bool = typer.Option(False, "--no-embed"),
    session: str | None = typer.Option(None, "--session"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Produce an extractive answer with numbered citations."""

    store = _load_store(session)
    embedder = None if no_embed else _embedder(fast, lang)
    text, resp = run_answer(q, store, embedder=embedder, budget=budget, url_filter=url)
    if as_json:
        _emit(
            {
                "answer": text,
                "results": [r.to_json() for r in resp.results],
                "budget_total": resp.budget_total,
                "budget_used": resp.budget_used,
            },
            as_json=True,
        )
        return
    typer.echo(text)


@app.command("score")
def cmd_score(
    query: str = typer.Argument(..., help="Reference query."),
    snippets: list[str] = typer.Argument(..., help="Snippets to rank."),
    top_k: int | None = typer.Option(None, "--top-k"),
    fast: bool = typer.Option(False, "--fast"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Rank snippets by relevance to a query without fetching anything."""

    embedder = _embedder(fast)
    ranked = run_score(query, snippets, embedder)
    if top_k is not None:
        ranked = ranked[: max(0, top_k)]
    if as_json:
        _emit([s.to_json() for s in ranked], as_json=True)
        return
    for s in ranked:
        typer.echo(f"[{s.index}] score={s.score:.3f}  {s.text[:120]}")


@app.command("search")
def cmd_search(
    q: str = typer.Argument(..., help="Search query."),
    top: int = typer.Option(5, "--top", help="Top N search results to ingest."),
    budget: int | None = typer.Option(2000, "--budget"),
    via_searxng: bool = typer.Option(
        True,
        "--via-searxng/--no-via-searxng",
        help="Use SearXNG as the search backend (currently the only mode).",
    ),
    instance: str | None = typer.Option(
        None, "--instance", help="SearXNG instance URL (overrides $NESIFT_SEARXNG_URL)."
    ),
    fast: bool = typer.Option(False, "--fast"),
    session: str | None = typer.Option(None, "--session"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """One-shot: search SearXNG → score snippets → fetch+ingest top hits → answer."""

    if not via_searxng:
        typer.secho(
            "--no-via-searxng is reserved for future backends and not yet implemented.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)
    store = _load_store(session)
    embedder = _embedder(fast)
    try:
        text, resp, scored = run_search(
            q,
            store,
            embedder=embedder,
            top_n=top,
            budget=budget,
            instance_url=instance,
        )
    except Exception as exc:
        typer.secho(f"search failed: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    _persist(store)
    if as_json:
        _emit(
            {
                "answer": text,
                "results": [r.to_json() for r in resp.results],
                "budget_total": resp.budget_total,
                "budget_used": resp.budget_used,
                "snippets": [s.to_json() for s in scored],
            },
            as_json=True,
        )
        return
    typer.echo(text)


# ---------- maintenance ----------


@app.command("list")
def cmd_list(
    session: str | None = typer.Option(None, "--session"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Show indexed URLs with their triage summaries."""

    store = _load_store(session)
    if as_json:
        _emit(
            [
                {
                    "url": p.url,
                    "title": p.title,
                    "chunks": len(p.chunks),
                    "triage": p.triage,
                }
                for p in store.pages
            ],
            as_json=True,
        )
        return
    if not store.pages:
        typer.echo("(empty session)")
        return
    for p in store.pages:
        typer.echo(f"- {p.title}")
        typer.echo(f"    url: {p.url}")
        typer.echo(f"    chunks: {len(p.chunks)}")
        if p.triage:
            typer.echo(f"    triage: {p.triage}")


@app.command("clear")
def cmd_clear(
    session: str | None = typer.Option(None, "--session"),
) -> None:
    """Drop all indexed pages from the active session."""

    store = _load_store(session)
    store.clear()
    typer.echo("cleared")


@app.command("save")
def cmd_save(
    output: Path = typer.Option(..., "-o", "--output", help="Destination path."),
    session: str | None = typer.Option(None, "--session"),
) -> None:
    """Copy the active session index to ``output``."""

    store = _load_store(session)
    path = store.save(output)
    typer.echo(f"saved {len(store.pages)} pages → {path}")


_AGENTS_SNIPPET = """\
## nesift — token-efficient web research

Use `nesift` instead of raw `web_fetch` when researching multiple sources:

```bash
# 1. Pre-score snippets to avoid fetching irrelevant pages
nesift score "<query>" "<snippet1>" "<snippet2>" ...

# 2. Index pages on-the-fly
nesift add <url>
nesift add-batch <url1> <url2> ...

# 3. Ask questions across everything indexed (token-budgeted)
nesift query "<question>" --budget 2000
nesift answer "<question>" --budget 2000

# 4. One-shot SearXNG-backed research
nesift search "<query>" --top 5 --budget 2000

# 5. Session hygiene
nesift list
nesift clear
```

Session state is per-shell (keyed on parent PID); override with `NESIFT_SESSION`.
"""


@app.command("init")
def cmd_init(
    file: str = typer.Option(
        "AGENTS.md",
        "--file",
        help="Target file (e.g. AGENTS.md, CLAUDE.md).",
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace file if it exists."),
) -> None:
    """Drop a nesift usage snippet into ``AGENTS.md`` (or another agent file)."""

    target = Path(file)
    snippet = _AGENTS_SNIPPET
    if target.exists() and not overwrite:
        existing = target.read_text(encoding="utf-8")
        if "nesift" in existing.lower():
            typer.echo(f"{target} already mentions nesift; not modifying. Use --overwrite to replace.")
            return
        target.write_text(existing.rstrip() + "\n\n" + snippet, encoding="utf-8")
        typer.echo(f"appended nesift section to {target}")
    else:
        target.write_text(snippet, encoding="utf-8")
        typer.echo(f"wrote {target}")


@app.command("mcp")
def cmd_mcp() -> None:
    """Start the MCP server (stdio). Requires the ``[mcp]`` extra."""

    try:
        from nesift.mcp_server import main as mcp_main
    except ImportError as exc:
        typer.secho(
            f"MCP extra not installed: {exc}. Install with `pip install 'nesift[mcp]'`.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2) from exc
    mcp_main()


@app.command("version")
def cmd_version() -> None:
    """Print the installed nesift version."""

    typer.echo(__version__)


def main(argv: list[str] | None = None) -> None:
    """Console-script entrypoint."""

    if argv is None:
        app()
    else:
        app(args=argv, prog_name="nesift", standalone_mode=True)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
