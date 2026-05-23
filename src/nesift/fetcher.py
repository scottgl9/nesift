"""HTTP fetcher with sane defaults for web pages."""

from __future__ import annotations

import httpx

from nesift.config import HTTP_TIMEOUT_SECONDS, USER_AGENT


class FetchError(RuntimeError):
    """Raised when a URL cannot be fetched or returns a non-2xx status."""


def fetch(url: str, *, timeout: float = HTTP_TIMEOUT_SECONDS) -> str:
    """Fetch ``url`` and return the response body as text.

    Follows redirects, sets a descriptive User-Agent, and raises
    :class:`FetchError` on network or HTTP failures.
    """

    try:
        resp = httpx.get(
            url,
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*;q=0.8"},
        )
    except httpx.HTTPError as exc:
        raise FetchError(f"failed to fetch {url}: {exc}") from exc
    if resp.status_code >= 400:
        raise FetchError(f"failed to fetch {url}: HTTP {resp.status_code}")
    return resp.text
