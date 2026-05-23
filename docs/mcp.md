# MCP server

`nesift[mcp]` ships a [Model Context Protocol](https://modelcontextprotocol.io) server that exposes nesift's tools to MCP-capable agents (Claude Code, Claude Desktop, Cursor, etc.) over stdio.

## Install + run

```bash
pip install "nesift[mcp]"
nesift-mcp                  # console-script entrypoint
# or
nesift mcp                  # CLI subcommand
# or
python -m nesift.mcp_server
# or, with uv:
uvx --from "nesift[mcp]" nesift-mcp
```

The server reads JSON-RPC from stdin and writes to stdout. Logging goes to stderr.

## Exposed tools

| Tool | Description |
|------|-------------|
| `score_snippets` | Rank text snippets by relevance to a query. Use BEFORE fetching to skip irrelevant pages. |
| `add_page` | Fetch + extract + chunk + embed + index one URL. |
| `add_batch` | Same, multiple URLs. |
| `query` | Hybrid BM25 + embedding search with optional `--budget`. |
| `answer` | Extractive answer with `[N]` citations. |
| `list_pages` | Indexed URLs with titles and triage summaries. |
| `clear` | Drop the active session. |
| `search` | One-shot SearXNG bridge: search → score → ingest top hits → answer. |

## Configuring agents

### Claude Code

`~/.config/claude/mcp.json` (or per-project `.mcp.json`):

```json
{
  "mcpServers": {
    "nesift": {
      "command": "nesift-mcp",
      "env": {
        "NESIFT_SEARXNG_URL": "http://127.0.0.1:8888"
      }
    }
  }
}
```

### Claude Desktop / Cursor

Same shape — different settings file. Restart the host after editing.

## Verifying

The bundled validation test drives the server with the real MCP client SDK:

```bash
NESIFT_SEARXNG_URL=http://127.0.0.1:8888 \
  pytest -m validation tests/validation/test_mcp_live.py -v
```

It spawns `nesift-mcp`, calls `list_tools`, scores snippets, adds a real Wikipedia page, runs `answer`, and (when SearXNG is reachable) runs `search`.
