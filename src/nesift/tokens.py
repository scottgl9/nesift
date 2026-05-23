"""Token counting. tiktoken cl100k by default; word-count fallback if unavailable."""

from __future__ import annotations

from functools import lru_cache

try:
    import tiktoken

    _HAS_TIKTOKEN = True
except Exception:  # pragma: no cover - exercised only when tiktoken missing
    _HAS_TIKTOKEN = False


@lru_cache(maxsize=1)
def _encoding() -> object | None:
    if not _HAS_TIKTOKEN:
        return None
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover
        return None


def count_tokens(text: str) -> int:
    """Return an approximate token count for ``text``.

    Uses tiktoken's cl100k encoding when available (the same tokenizer
    OpenAI/Anthropic context budgets are reasoned about with); otherwise
    falls back to ``len(text.split()) * 1.3``.
    """

    if not text:
        return 0
    enc = _encoding()
    if enc is not None:
        return len(enc.encode(text))
    return int(len(text.split()) * 1.3)
