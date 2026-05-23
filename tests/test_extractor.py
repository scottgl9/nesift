import pytest

from nesift.extractor import ExtractionError, extract


def test_extract_blog_post(fixture_html):
    doc = extract(fixture_html["blog_post"], url="https://example.com/blog")
    assert "retry" in doc.title.lower() or "resilient" in doc.title.lower()
    assert any(s.paragraphs for s in doc.sections)
    text = doc.text.lower()
    assert "exponential backoff" in text
    # Boilerplate stripped:
    assert "site footer (ignored)" not in text


def test_extract_wikipedia(fixture_html):
    doc = extract(fixture_html["wikipedia_article"], url="https://wiki.test/RAG")
    assert "retrieval" in doc.title.lower()
    headings = [s.heading for s in doc.sections if s.heading]
    # At least one of the expected H2 headings survives.
    assert any(h and ("motivation" in h.lower() or "architecture" in h.lower() or "applications" in h.lower()) for h in headings)


def test_extract_docs(fixture_html):
    doc = extract(fixture_html["docs_page"], url="https://docs.test/auth")
    text = doc.text.lower()
    assert "bearer" in text
    assert "scope" in text


def test_extract_empty():
    with pytest.raises(ExtractionError):
        extract("")


def test_extract_garbage():
    with pytest.raises(ExtractionError):
        extract("<html><body></body></html>")
