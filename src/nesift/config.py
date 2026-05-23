"""Runtime configuration: paths, model names, thresholds, env vars."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

DEFAULT_MODEL = "minishlab/potion-retrieval-32M"
FAST_MODEL = "minishlab/potion-base-8M"

DEFAULT_TOP_K = 5
DEFAULT_DEDUP_THRESHOLD = 0.88
RRF_K = 60
MIN_CHUNK_TOKENS = 80
TARGET_CHUNK_TOKENS = 500
MAX_CHUNK_TOKENS = 800

DEFAULT_SEARXNG_URL = "http://127.0.0.1:8888"
HTTP_TIMEOUT_SECONDS = 15.0
USER_AGENT = "nesift/0.1 (+https://github.com/scottgl9/nesift)"


def dedup_threshold() -> float:
    """Cosine similarity threshold for cross-page dedup; overridable via env."""
    raw = os.environ.get("NESIFT_DEDUP_THRESHOLD")
    if raw is None:
        return DEFAULT_DEDUP_THRESHOLD
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_DEDUP_THRESHOLD


def session_id() -> str:
    """Active session id: $NESIFT_SESSION or `pid<parent-pid>`."""
    return os.environ.get("NESIFT_SESSION") or f"pid{os.getppid()}"


def session_path(session: str | None = None) -> Path:
    """Filesystem path for the session JSON store."""
    sid = session or session_id()
    return Path(tempfile.gettempdir()) / f"nesift-{sid}.json"


def searxng_url() -> str:
    return os.environ.get("NESIFT_SEARXNG_URL") or DEFAULT_SEARXNG_URL
