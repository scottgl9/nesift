"""Token-budget trimming for query results."""

from __future__ import annotations

from nesift.schema import QueryResult


def trim(results: list[QueryResult], max_tokens: int) -> tuple[list[QueryResult], int]:
    """Greedily take results in order until the cumulative token count exceeds budget.

    Returns ``(kept, used_tokens)``. Order is preserved. A single
    over-budget result is kept only if it's the first (otherwise an
    empty budget would always return nothing).
    """

    if max_tokens <= 0:
        return [], 0
    kept: list[QueryResult] = []
    used = 0
    for r in results:
        cost = max(1, r.token_count)
        if not kept and cost > max_tokens:
            # Always include at least one result so the caller gets *something*.
            kept.append(r)
            used = cost
            break
        if used + cost > max_tokens:
            break
        kept.append(r)
        used += cost
    return kept, used
