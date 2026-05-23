"""Extractive triage summarizer: lede + top-scored sentences."""

from __future__ import annotations

import re

from nesift.extractor import ExtractedDoc

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_WORD = re.compile(r"[A-Za-z0-9']+")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def _tokens(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text)}


def triage(doc: ExtractedDoc, *, max_sentences: int = 3) -> str:
    """Return a 1-3 sentence extractive summary of ``doc``.

    Sentences are scored by:
    - position decay (early sentences favored)
    - overlap with title tokens
    - overlap with heading tokens
    Lede always wins ties.
    """

    title_toks = _tokens(doc.title)
    heading_toks: set[str] = set()
    for sec in doc.sections:
        if sec.heading:
            heading_toks |= _tokens(sec.heading)

    # Flatten paragraphs to sentences with provenance.
    sentences: list[str] = []
    for sec in doc.sections:
        for para in sec.paragraphs:
            sentences.extend(_sentences(para))
    if not sentences:
        return ""

    scored: list[tuple[float, int, str]] = []
    for idx, sent in enumerate(sentences):
        # Skip very short fragments (likely bullets/labels).
        if len(sent.split()) < 4:
            continue
        toks = _tokens(sent)
        position = 1.0 / (1.0 + idx * 0.15)
        title_overlap = len(toks & title_toks) * 0.4
        heading_overlap = len(toks & heading_toks) * 0.2
        score = position + title_overlap + heading_overlap
        scored.append((score, idx, sent))

    if not scored:
        return sentences[0]

    # Always include the lede if it qualifies.
    chosen_idx = {scored[0][1]}
    scored.sort(key=lambda x: (-x[0], x[1]))
    for _, idx, _ in scored:
        if len(chosen_idx) >= max_sentences:
            break
        chosen_idx.add(idx)

    return " ".join(sentences[i] for i in sorted(chosen_idx))
