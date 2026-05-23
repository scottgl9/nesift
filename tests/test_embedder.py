import numpy as np
import pytest

from nesift.embedder import Embedder, FakeEmbedder


def test_fake_embedder_deterministic():
    e = FakeEmbedder(dim=32)
    v1 = e.embed("hello world")
    v2 = e.embed("hello world")
    assert np.allclose(v1, v2)
    assert v1.shape == (32,)


def test_fake_embedder_differs_for_different_text():
    e = FakeEmbedder(dim=32)
    v1 = e.embed("hello world")
    v2 = e.embed("completely unrelated content here")
    assert not np.allclose(v1, v2)


def test_fake_embedder_many():
    e = FakeEmbedder(dim=16)
    m = e.embed_many(["a", "b", "c"])
    assert m.shape == (3, 16)


@pytest.mark.slow
def test_real_embedder_smoke():
    """Downloads the real model. Run with `-m slow` (or set NESIFT_RUN_SLOW=1)."""
    import os

    if not os.environ.get("NESIFT_RUN_SLOW"):
        pytest.skip("set NESIFT_RUN_SLOW=1 to run")
    e = Embedder()
    v = e.embed("retrieval-augmented generation")
    assert v.shape[0] > 0
    # Normalized.
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-3
