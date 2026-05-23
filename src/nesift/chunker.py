"""Paragraph- and heading-aware chunking."""

from __future__ import annotations

from nesift.config import MAX_CHUNK_TOKENS, MIN_CHUNK_TOKENS, TARGET_CHUNK_TOKENS
from nesift.extractor import ExtractedDoc, Section
from nesift.tokens import count_tokens


def chunk_document(doc: ExtractedDoc) -> list[tuple[str | None, str]]:
    """Split ``doc`` into ``(section_heading, chunk_text)`` tuples.

    Rules:
    - Never merge across section headings.
    - Within a section, accumulate paragraphs into windows of
      ``TARGET_CHUNK_TOKENS`` (soft) up to ``MAX_CHUNK_TOKENS`` (hard).
    - Tiny trailing chunks below ``MIN_CHUNK_TOKENS`` are merged into the
      previous chunk in the same section.
    """

    chunks: list[tuple[str | None, str]] = []
    for sec in doc.sections:
        chunks.extend(_chunk_section(sec))
    return chunks


def _chunk_section(section: Section) -> list[tuple[str | None, str]]:
    heading = section.heading
    out: list[tuple[str | None, str]] = []
    buf: list[str] = []
    buf_tokens = 0

    def flush() -> None:
        nonlocal buf, buf_tokens
        if not buf:
            return
        text = "\n\n".join(buf).strip()
        if text:
            out.append((heading, text))
        buf = []
        buf_tokens = 0

    for para in section.paragraphs:
        ptoks = count_tokens(para)
        if buf_tokens + ptoks > MAX_CHUNK_TOKENS and buf:
            flush()
        buf.append(para)
        buf_tokens += ptoks
        if buf_tokens >= TARGET_CHUNK_TOKENS:
            flush()
    flush()

    # Merge tiny trailing chunks forward within this section.
    merged: list[tuple[str | None, str]] = []
    for h, text in out:
        toks = count_tokens(text)
        if merged and toks < MIN_CHUNK_TOKENS and merged[-1][0] == h:
            prev_h, prev_text = merged[-1]
            merged[-1] = (prev_h, prev_text + "\n\n" + text)
        else:
            merged.append((h, text))
    return merged
