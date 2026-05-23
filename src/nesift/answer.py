"""Extractive answer synthesis with numbered citations."""

from __future__ import annotations

from nesift.schema import QueryResult


def synthesize(question: str, results: list[QueryResult]) -> str:
    """Concatenate ``results`` as a single answer block with ``[N]`` citations.

    Citation numbers are assigned in first-appearance order per URL.
    Trailing ``Sources:`` block lists each unique URL once.
    """

    if not results:
        return f"No indexed content matches: {question!r}"

    citations: dict[str, int] = {}
    body_parts: list[str] = []
    for r in results:
        if r.url not in citations:
            citations[r.url] = len(citations) + 1
        num = citations[r.url]
        body_parts.append(f"{r.chunk.strip()} [{num}]")
    body = "\n\n".join(body_parts)
    sources = "\n".join(f"[{n}] {url}" for url, n in sorted(citations.items(), key=lambda x: x[1]))
    return f"{body}\n\nSources:\n{sources}"
