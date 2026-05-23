from nesift.tokens import count_tokens


def test_count_tokens_empty():
    assert count_tokens("") == 0


def test_count_tokens_basic():
    n = count_tokens("Hello world this is a sentence about retrieval-augmented generation.")
    assert n > 5
    assert n < 30


def test_count_tokens_monotonic():
    short = count_tokens("one two three")
    long = count_tokens("one two three " * 50)
    assert long > short
