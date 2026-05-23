from nesift.chunker import chunk_document
from nesift.extractor import ExtractedDoc, Section


def test_chunker_respects_headings():
    doc = ExtractedDoc(
        title="t",
        sections=[
            Section(heading="A", paragraphs=["para a1 " * 30, "para a2 " * 30]),
            Section(heading="B", paragraphs=["para b1 " * 30]),
        ],
    )
    chunks = chunk_document(doc)
    headings = {h for h, _ in chunks}
    assert headings == {"A", "B"}
    for h, text in chunks:
        # No chunk should contain text from a different heading.
        if h == "A":
            assert "b1" not in text
        else:
            assert "a1" not in text and "a2" not in text


def test_chunker_packs_within_max():
    long_para = "word " * 50
    doc = ExtractedDoc(
        title="t",
        sections=[Section(heading="H", paragraphs=[long_para] * 30)],
    )
    chunks = chunk_document(doc)
    assert len(chunks) >= 2


def test_chunker_handles_no_heading():
    doc = ExtractedDoc(
        title="t",
        sections=[Section(heading=None, paragraphs=["A small paragraph of text."])],
    )
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    h, text = chunks[0]
    assert h is None
    assert "small paragraph" in text
