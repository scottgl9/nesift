from nesift.scorer import score_snippets


def test_score_orders_relevant_first(fake_embedder):
    query = "vector database"
    snippets = [
        "How to bake sourdough bread at home.",
        "Pinecone is a managed vector database for embeddings.",
        "The history of the Roman Empire.",
    ]
    out = score_snippets(query, snippets, fake_embedder)
    assert out
    # Relevant snippet should be index 1 in the original list.
    assert out[0].index == 1 or "vector" in out[0].text.lower()


def test_score_empty(fake_embedder):
    assert score_snippets("anything", [], fake_embedder) == []
