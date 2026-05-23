# SearXNG integration

`nesift search` performs an end-to-end retrieval over the live web by chaining a SearXNG instance with the rest of the pipeline:

1. Query the SearXNG `/search?format=json` endpoint.
2. Embed the query and every returned snippet; rank by cosine similarity (`nesift.scorer`).
3. Fetch + ingest the top-N most relevant URLs (skipping any already in the session).
4. Run `nesift answer` against the freshly augmented session.

## Configuring the instance

Set `NESIFT_SEARXNG_URL` (or pass `--instance`):

```bash
export NESIFT_SEARXNG_URL=http://127.0.0.1:8888
nesift search "retry logic in distributed systems" --top 5 --budget 2000
```

Default: `http://127.0.0.1:8888`.

## Enabling the JSON API

Most SearXNG installs serve HTML only by default. Edit `searxng/settings.yml`:

```yaml
search:
  formats:
    - html
    - json
```

Restart SearXNG and confirm with:

```bash
curl 'http://127.0.0.1:8888/search?q=test&format=json' | head -c 200
```

If the response is HTML or 403, the JSON API is still disabled.

## Troubleshooting

| Symptom | Likely cause |
|---------|-------------|
| `SearXNG ... returned HTTP 403` | JSON API not enabled in `settings.yml` |
| `SearXNG request to ... failed: [Errno 111]` | Instance not running, or wrong port |
| Empty results | Query yielded nothing on configured engines; try the same query in the browser |
| Top hit is irrelevant | `nesift score` is filtering, but bad snippets can still survive — try `--top 10` to widen the pool |
