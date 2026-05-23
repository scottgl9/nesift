import numpy as np

from nesift.dedup import collapse
from nesift.schema import QueryResult


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


def test_collapse_merges_near_duplicates():
    a = _unit([1, 0, 0])
    b = _unit([0.99, 0.01, 0])
    c = _unit([0, 1, 0])
    results = [
        QueryResult(chunk="X", url="u1", section=None, score=1.0, token_count=10),
        QueryResult(chunk="X again", url="u2", section=None, score=0.9, token_count=10),
        QueryResult(chunk="Y", url="u3", section=None, score=0.8, token_count=10),
    ]
    kept = collapse(results, [a, b, c], threshold=0.95)
    assert len(kept) == 2
    assert kept[0].sources == 2
    assert kept[1].sources == 1


def test_collapse_skips_when_no_embedding():
    results = [
        QueryResult(chunk="X", url="u1", section=None, score=1.0, token_count=10),
        QueryResult(chunk="X", url="u2", section=None, score=0.9, token_count=10),
    ]
    kept = collapse(results, [None, None], threshold=0.5)
    assert len(kept) == 2
