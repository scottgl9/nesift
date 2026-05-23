import numpy as np

from nesift.index.hybrid import ranks_from_scores, rerank, rrf


def test_ranks_from_scores():
    r = ranks_from_scores(np.array([0.1, 0.5, 0.3]))
    assert list(r) == [3, 1, 2]


def test_rrf_combines_channels():
    bm25 = np.array([1.0, 0.5, 0.0])
    vec = np.array([0.0, 0.9, 0.8])
    fused = rrf([bm25, vec])
    # Doc index 1 ranks highly in both — should win.
    assert int(np.argmax(fused)) == 1


def test_rerank_diversity_penalty():
    scores = np.array([1.0, 0.95, 0.9])
    pages = ["A", "A", "B"]  # two chunks from same page; diversity pushes B above 2nd A
    sections = [None, None, None]
    out = rerank(scores, page_ids=pages, sections=sections, query="x", top_k=3)
    chosen = [i for i, _ in out]
    assert chosen[0] == 0  # top hit
    # Second-best from a different page should appear before the second A-page chunk.
    rank_of = {i: pos for pos, (i, _) in enumerate(out)}
    assert rank_of[2] < rank_of[1]


def test_rerank_heading_boost():
    scores = np.array([0.5, 0.5])
    pages = ["A", "B"]
    sections = [None, "retry logic"]
    out = rerank(scores, page_ids=pages, sections=sections, query="retry", top_k=2)
    assert out[0][0] == 1
