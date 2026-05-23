from nesift.budget import trim
from nesift.schema import QueryResult


def _r(tok, score=1.0, url="u"):
    return QueryResult(chunk="x", url=url, section=None, score=score, token_count=tok)


def test_trim_respects_budget():
    results = [_r(300), _r(300), _r(300), _r(300)]
    kept, used = trim(results, 700)
    assert len(kept) == 2
    assert used == 600


def test_trim_preserves_order():
    results = [_r(100, score=1.0, url="a"), _r(100, score=0.9, url="b")]
    kept, _ = trim(results, 1000)
    assert [r.url for r in kept] == ["a", "b"]


def test_trim_keeps_at_least_one_oversize():
    results = [_r(5000)]
    kept, used = trim(results, 100)
    assert len(kept) == 1
    assert used == 5000


def test_trim_zero_budget_empty():
    assert trim([_r(10)], 0) == ([], 0)
