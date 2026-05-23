"""HTML → structured text via trafilatura, preserving headings."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import trafilatura


@dataclass
class Section:
    """A heading and the paragraphs that follow it."""

    heading: str | None
    paragraphs: list[str] = field(default_factory=list)


@dataclass
class ExtractedDoc:
    """The result of stripping a page down to its meaningful prose."""

    title: str
    sections: list[Section]

    @property
    def text(self) -> str:
        out: list[str] = []
        for sec in self.sections:
            if sec.heading:
                out.append(f"# {sec.heading}")
            out.extend(sec.paragraphs)
        return "\n\n".join(out)


class ExtractionError(RuntimeError):
    """Raised when trafilatura returns no usable content for a page."""


_HEADING_RE = re.compile(r"^(#{1,6})\s*(.+)$")


def extract(html: str, *, url: str | None = None) -> ExtractedDoc:
    """Strip boilerplate from ``html`` and return a structured document.

    Uses trafilatura's markdown output mode so that headings are
    preserved as ``# Heading`` lines, which we then split into
    :class:`Section` groups.
    """

    if not html or not html.strip():
        raise ExtractionError("empty HTML")

    md = trafilatura.extract(
        html,
        url=url,
        output_format="markdown",
        include_comments=False,
        include_tables=True,
        favor_recall=True,
    )
    if not md:
        raise ExtractionError("trafilatura returned no content")

    # Title: prefer trafilatura's metadata; fall back to first heading.
    title = ""
    try:
        meta = trafilatura.extract_metadata(html)
        if meta and meta.title:
            title = meta.title.strip()
    except Exception:
        pass

    sections: list[Section] = []
    current = Section(heading=None)
    buf: list[str] = []

    def flush_paragraph() -> None:
        if buf:
            para = "\n".join(buf).strip()
            if para:
                current.paragraphs.append(para)
            buf.clear()

    for line in md.splitlines():
        stripped = line.rstrip()
        m = _HEADING_RE.match(stripped)
        if m:
            flush_paragraph()
            if current.heading is not None or current.paragraphs:
                sections.append(current)
            current = Section(heading=m.group(2).strip())
        elif not stripped:
            flush_paragraph()
        else:
            buf.append(stripped)
    flush_paragraph()
    if current.heading is not None or current.paragraphs:
        sections.append(current)

    if not title and sections and sections[0].heading:
        title = sections[0].heading
    if not title and url:
        title = url

    if not any(s.paragraphs for s in sections):
        raise ExtractionError("extracted document has no paragraphs")

    return ExtractedDoc(title=title or "(untitled)", sections=sections)
