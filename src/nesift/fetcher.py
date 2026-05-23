"""HTTP fetcher with sane defaults for web pages.

Provides both synchronous (:func:`fetch`, :func:`fetch_raw`) and
asynchronous (:func:`fetch_many`) interfaces. The async path is used by
:func:`nesift.pipeline.ingest_urls` to parallelize multi-URL ingestion.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx

from nesift.config import HTTP_TIMEOUT_SECONDS, USER_AGENT


class FetchError(RuntimeError):
    """Raised when a URL cannot be fetched or returns a non-2xx status."""


@dataclass
class FetchResult:
    """One result from a concurrent fetch."""

    url: str
    body: bytes | None
    content_type: str
    error: str | None

    @property
    def ok(self) -> bool:
        return self.error is None and self.body is not None


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


async def _fetch_one_async(
    client: httpx.AsyncClient, url: str, timeout: float
) -> FetchResult:
    try:
        resp = await client.get(url, timeout=timeout)
    except httpx.HTTPError as exc:
        return FetchResult(url=url, body=None, content_type="", error=str(exc))
    if resp.status_code >= 400:
        return FetchResult(
            url=url, body=None, content_type="", error=f"HTTP {resp.status_code}"
        )
    return FetchResult(
        url=url,
        body=resp.content,
        content_type=resp.headers.get("content-type", ""),
        error=None,
    )


async def _fetch_many_async(
    urls: list[str], *, concurrency: int, timeout: float
) -> list[FetchResult]:
    sem = asyncio.Semaphore(max(1, concurrency))
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/pdf,*/*;q=0.8",
    }
    async with httpx.AsyncClient(
        headers=headers, follow_redirects=True, timeout=timeout
    ) as client:

        async def _bounded(u: str) -> FetchResult:
            async with sem:
                return await _fetch_one_async(client, u, timeout)

        return await asyncio.gather(*[_bounded(u) for u in urls])


def fetch_many(
    urls: list[str],
    *,
    concurrency: int = 8,
    timeout: float = HTTP_TIMEOUT_SECONDS,
) -> list[FetchResult]:
    """Fetch multiple URLs in parallel via :class:`httpx.AsyncClient`.

    Results are returned in input order, never raising — each entry
    carries an ``error`` string for failures so callers can decide
    whether to skip or surface.
    """

    if not urls:
        return []
    return asyncio.run(_fetch_many_async(urls, concurrency=concurrency, timeout=timeout))
