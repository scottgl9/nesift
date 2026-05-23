from __future__ import annotations

import io

import pytest

from nesift.extractor import ExtractionError
from nesift.pdf import extract_pdf, is_pdf_bytes, is_pdf_url


def _make_pdf(pages: list[str]) -> bytes:
    """Build a tiny multi-page PDF in-memory using pypdf."""

    pypdf = pytest.importorskip("pypdf")
    # pypdf can't render fresh PDFs from raw text, but reportlab is heavy.
    # Use a minimal hand-rolled PDF generator: a single page per string,
    # with Helvetica text. This pattern is well documented in pypdf tests.
    from pypdf import PdfWriter

    writer = PdfWriter()
    for text in pages:
        writer.add_blank_page(width=612, height=792)
        writer.pages[-1].merge_page  # ensure attribute exists
    # Now overlay text via a separate PDF built with pypdf's content streams.
    # Easier: write a tiny PDF by hand for each page.
    return _minimal_pdf(pages)


def _minimal_pdf(pages: list[str]) -> bytes:
    """Construct the smallest viable PDF that pypdf can parse text from."""

    # Build a 1-page PDF per chunk and merge — keeps the byte construction simple.
    from pypdf import PdfWriter, PdfReader

    writer = PdfWriter()
    for text in pages:
        single = _one_page_pdf(text)
        reader = PdfReader(io.BytesIO(single))
        writer.add_page(reader.pages[0])
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _one_page_pdf(text: str) -> bytes:
    """A single page PDF containing ``text`` using Helvetica."""

    # Reference: minimal PDF following the spec, with one stream of text-showing
    # operators. Built so pypdf's text extractor will find it.
    safe = text.replace("(", r"\(").replace(")", r"\)")
    stream = (
        "BT /F1 12 Tf 72 720 Td "
        + " ".join(f"({line}) Tj 0 -14 Td" for line in safe.splitlines() or [safe])
        + " ET"
    ).encode("latin-1")
    objs: list[bytes] = []

    def add(obj: bytes) -> int:
        objs.append(obj)
        return len(objs)

    add(b"<< /Type /Catalog /Pages 2 0 R >>")
    add(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    add(
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
    )
    add(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
    add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_pos = len(out)
    out += b"xref\n0 " + str(len(objs) + 1).encode() + b"\n"
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        b"trailer\n<< /Size " + str(len(objs) + 1).encode()
        + b" /Root 1 0 R >>\nstartxref\n" + str(xref_pos).encode() + b"\n%%EOF\n"
    )
    return bytes(out)


def test_pdf_signature_detection():
    assert is_pdf_bytes(b"%PDF-1.4 ...")
    assert not is_pdf_bytes(b"<html>")


def test_pdf_url_detection():
    assert is_pdf_url("https://example.com/foo.pdf")
    assert is_pdf_url("https://example.com/foo.PDF?x=1")
    assert not is_pdf_url("https://example.com/foo.html")


def test_extract_pdf_smoke():
    pdf = _minimal_pdf([
        "Introduction\nThis paper describes a fast local search index.",
        "Conclusion\nThe approach reduces token consumption substantially.",
    ])
    doc = extract_pdf(pdf, url="https://example.com/paper.pdf")
    assert doc.sections
    text = doc.text.lower()
    assert "search" in text or "token" in text


def test_extract_pdf_garbage():
    with pytest.raises(ExtractionError):
        extract_pdf(b"not a pdf", url="https://x")
