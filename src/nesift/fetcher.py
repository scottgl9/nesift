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

    resp = _get(url, timeout)
    return resp.text


def fetch_raw(url: str, *, timeout: float = HTTP_TIMEOUT_SECONDS) -> tuple[bytes, str]:
    """Fetch ``url`` and return ``(body_bytes, content_type)``.

    Use this when the caller needs to dispatch on content type, e.g. to
    feed PDFs to a different extractor.
    """

    resp = _get(url, timeout)
    return resp.content, resp.headers.get("content-type", "")


def _get(url: str, timeout: float) -> httpx.Response:
    try:
        resp = httpx.get(
            url,
            timeout=timeout,
            follow_redirects=True,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/pdf,*/*;q=0.8",
            },
        )
    except httpx.HTTPError as exc:
        raise FetchError(f"failed to fetch {url}: {exc}") from exc
    if resp.status_code >= 400:
        raise FetchError(f"failed to fetch {url}: HTTP {resp.status_code}")
    return resp
