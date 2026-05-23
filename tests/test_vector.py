import numpy as np

from nesift.index.vector import VectorIndex


def test_vector_cosine_identical():
    idx = VectorIndex()
    v = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    idx.add_many(v)
    scores = idx.scores(np.array([1.0, 0.0, 0.0]))
    assert abs(scores[0] - 1.0) < 1e-6


def test_vector_cosine_orthogonal():
    idx = VectorIndex()
    idx.add_many(np.array([[1.0, 0.0]], dtype=np.float32))
    scores = idx.scores(np.array([0.0, 1.0]))
    assert abs(scores[0]) < 1e-6


def test_vector_dim_mismatch_raises():
    idx = VectorIndex()
    idx.add_many(np.array([[1.0, 0.0]], dtype=np.float32))
    try:
        idx.add_many(np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for dim mismatch")
