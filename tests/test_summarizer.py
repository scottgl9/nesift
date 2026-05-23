from nesift.extractor import ExtractedDoc, Section
from nesift.summarizer import triage


def test_triage_picks_lede():
    doc = ExtractedDoc(
        title="Retry Logic",
        sections=[
            Section(
                heading=None,
                paragraphs=[
                    "Retry logic is one of the most misused patterns in distributed systems.",
                    "Most network requests fail for transient reasons that retry can absorb.",
                    "Operations that are not idempotent should not be blindly retried.",
                ],
            )
        ],
    )
    out = triage(doc)
    assert out
    assert "Retry logic" in out


def test_triage_deterministic():
    doc = ExtractedDoc(
        title="Topic",
        sections=[Section(heading=None, paragraphs=["First sentence about the topic at hand. Second sentence here. Third also."])],
    )
    assert triage(doc) == triage(doc)


def test_triage_empty():
    doc = ExtractedDoc(title="", sections=[])
    assert triage(doc) == ""
