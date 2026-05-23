from nesift.answer import synthesize
from nesift.schema import QueryResult


def test_synthesize_basic():
    results = [
        QueryResult(chunk="Alpha.", url="https://a", section=None, score=1.0, token_count=2),
        QueryResult(chunk="Beta.", url="https://b", section=None, score=0.5, token_count=2),
    ]
    out = synthesize("q?", results)
    assert "Alpha." in out and "Beta." in out
    assert "[1]" in out and "[2]" in out
    assert "https://a" in out and "https://b" in out


def test_synthesize_same_url_same_citation():
    results = [
        QueryResult(chunk="A1.", url="https://a", section=None, score=1.0, token_count=2),
        QueryResult(chunk="A2.", url="https://a", section=None, score=0.9, token_count=2),
    ]
    out = synthesize("q", results)
    # Only one source line for the single URL.
    assert out.count("https://a") == 1
    # Both chunks cite [1].
    assert out.count("[1]") == 3  # two inline citations + one in sources block


def test_synthesize_empty():
    out = synthesize("q?", [])
    assert "q?" in out
