"""Persistent extract + embedding cache.

Keyed on ``(url, embedding_model)`` because cached embeddings are
invalid if the model changes. One JSON file per entry under
``$NESIFT_CACHE_DIR`` (default ``~/.cache/nesift/pages``).

Cache hits skip the fetch → extract → chunk → embed pipeline entirely.
Set ``NESIFT_NO_CACHE=1`` to bypass globally; pass ``use_cache=False`` to
an individual call to bypass per-invocation.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from nesift.schema import Page


def cache_dir() -> Path:
    """Resolve the cache directory, honoring ``$NESIFT_CACHE_DIR``."""

    override = os.environ.get("NESIFT_CACHE_DIR")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "nesift" / "pages"


def disabled() -> bool:
    return os.environ.get("NESIFT_NO_CACHE") == "1"


def _key(url: str, model: str | None) -> str:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    tag = (model or "noembed").replace("/", "_").replace(":", "_")
    return f"{digest}__{tag}.json"


def _path(url: str, model: str | None) -> Path:
    return cache_dir() / _key(url, model)


def get(url: str, model: str | None) -> Page | None:
    """Return the cached :class:`Page` for ``(url, model)`` or ``None``."""

    if disabled():
        return None
    p = _path(url, model)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    try:
        return Page.from_json(data)
    except (KeyError, ValueError):
        return None


def put(page: Page, model: str | None) -> Path | None:
    """Write ``page`` to the cache. Returns the path on success or ``None``."""

    if disabled():
        return None
    p = _path(page.url, model)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(page.to_json()), encoding="utf-8")
    except OSError:
        return None
    return p


def clear() -> int:
    """Delete every cached page. Returns the number of files removed."""

    base = cache_dir()
    if not base.exists():
        return 0
    n = 0
    for p in base.glob("*.json"):
        try:
            p.unlink()
            n += 1
        except OSError:
            pass
    return n


def stats() -> dict:
    """Lightweight stats for ``nesift cache stats``."""

    base = cache_dir()
    if not base.exists():
        return {"dir": str(base), "entries": 0, "bytes": 0}
    entries = list(base.glob("*.json"))
    total = sum(p.stat().st_size for p in entries)
    return {"dir": str(base), "entries": len(entries), "bytes": total}
