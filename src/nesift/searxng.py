"""SearXNG JSON API client."""

from __future__ import annotations

import httpx

from nesift.config import HTTP_TIMEOUT_SECONDS, USER_AGENT, searxng_url
from nesift.schema import SearxResult


class SearxNGError(RuntimeError):
    """Raised when a SearXNG instance cannot be reached or returns an error."""


def search(
    query: str,
    *,
    top_n: int = 10,
    instance_url: str | None = None,
    timeout: float = HTTP_TIMEOUT_SECONDS,
) -> list[SearxResult]:
    """Query a SearXNG instance and return parsed results.

    Instance URL resolution: explicit ``instance_url`` →
    ``$NESIFT_SEARXNG_URL`` → default ``http://127.0.0.1:8888``.
    """

    base = (instance_url or searxng_url()).rstrip("/")
    url = f"{base}/search"
    params = {"q": query, "format": "json"}
    try:
        resp = httpx.get(
            url,
            params=params,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        )
    except httpx.HTTPError as exc:
        raise SearxNGError(f"SearXNG request to {base} failed: {exc}") from exc
    if resp.status_code >= 400:
        raise SearxNGError(
            f"SearXNG {base} returned HTTP {resp.status_code}. "
            "Many SearXNG instances disable the JSON API by default — enable "
            "`formats: [html, json]` in settings.yml."
        )
    try:
        data = resp.json()
    except ValueError as exc:
        raise SearxNGError(f"SearXNG {base} returned non-JSON response") from exc

    out: list[SearxResult] = []
    for item in data.get("results", [])[: max(0, top_n)]:
        u = item.get("url")
        if not u:
            continue
        out.append(
            SearxResult(
                title=str(item.get("title", "")).strip(),
                url=str(u),
                snippet=str(item.get("content", "")).strip(),
                score=float(item.get("score", 0.0) or 0.0),
            )
        )
    return out
