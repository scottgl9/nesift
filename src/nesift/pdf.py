"""PDF → ExtractedDoc.

Uses ``pypdf`` (pure-Python, no system libs) to pull text out of PDFs and
reshape it into the same :class:`ExtractedDoc` structure the HTML
pipeline produces. One section per page; the first non-empty line is
treated as a heading hint.
"""

from __future__ import annotations

import io
import re

from nesift.extractor import ExtractedDoc, ExtractionError, Section


def is_pdf_bytes(data: bytes) -> bool:
    return data[:5] == b"%PDF-"


def is_pdf_url(url: str) -> bool:
    return url.lower().split("?", 1)[0].endswith(".pdf")


def extract_pdf(data: bytes, *, url: str | None = None) -> ExtractedDoc:
    """Parse a PDF and return a structured document."""

    try:
        import pypdf  # type: ignore
    except ImportError as exc:  # pragma: no cover - extra not installed
        raise ExtractionError(
            "pypdf is required for PDF ingestion. Install with `pip install pypdf`."
        ) from exc

    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
    except Exception as exc:
        raise ExtractionError(f"failed to open PDF: {exc}") from exc

    title = ""
    try:
        meta = reader.metadata
        if meta and getattr(meta, "title", None):
            title = str(meta.title).strip()
    except Exception:
        pass

    sections: list[Section] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""
        if not raw.strip():
            continue
        paragraphs = _split_paragraphs(raw)
        if not paragraphs:
            continue
        # First short paragraph that looks like a header becomes the section heading.
        heading: str | None = f"Page {i}"
        cand = paragraphs[0].strip()
        if 0 < len(cand) <= 80 and cand[0].isupper() and not cand.endswith("."):
            heading = cand
            paragraphs = paragraphs[1:] or [cand]
        sections.append(Section(heading=heading, paragraphs=paragraphs))

    if not sections:
        raise ExtractionError("PDF contained no extractable text")

    if not title:
        title = url or (sections[0].heading or "(untitled PDF)")
    return ExtractedDoc(title=title, sections=sections)


_BLANK_LINE = re.compile(r"\n\s*\n+")


def _split_paragraphs(text: str) -> list[str]:
    parts = _BLANK_LINE.split(text)
    out: list[str] = []
    for p in parts:
        # Collapse internal newlines into spaces (PDF line breaks are noise).
        flat = " ".join(line.strip() for line in p.splitlines() if line.strip())
        if flat:
            out.append(flat)
    return out
