"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from nesift.embedder import FakeEmbedder
from nesift.session import SessionStore

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    """Deterministic embedder that avoids the model2vec download."""

    return FakeEmbedder(dim=64)


@pytest.fixture
def fixture_html() -> dict[str, str]:
    """All HTML fixtures by short name."""

    return {p.stem: p.read_text(encoding="utf-8") for p in FIXTURE_DIR.glob("*.html")}


@pytest.fixture
def session(tmp_path: Path) -> SessionStore:
    """Per-test SessionStore backed by a tmp_path file."""

    return SessionStore(tmp_path / "session.json")
