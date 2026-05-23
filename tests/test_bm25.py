from nesift.index.bm25 import BM25Index, tokenize


def test_tokenize_lowercase():
    assert tokenize("Hello World!") == ["hello", "world"]


def test_bm25_ranks_matches_higher():
    idx = BM25Index()
    idx.add("Exponential backoff with jitter is the canonical retry strategy.")
    idx.add("Cooking pasta requires boiling water and salt.")
    scores = idx.scores("exponential backoff retry")
    assert scores[0] > scores[1]


def test_bm25_empty():
    idx = BM25Index()
    scores = idx.scores("anything")
    assert scores.shape == (0,)


def test_bm25_no_matches_returns_zeros():
    idx = BM25Index()
    idx.add("alpha beta gamma")
    scores = idx.scores("nonexistent words elsewhere")
    assert float(scores[0]) == 0.0
