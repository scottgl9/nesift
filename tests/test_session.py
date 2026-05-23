import numpy as np

from nesift.schema import Chunk, Page
from nesift.session import SessionStore, page_id_for


def _page(url, embedding=None):
    pid = page_id_for(url)
    return Page(
        id=pid,
        url=url,
        title="t",
        fetched_at=0.0,
        triage="",
        chunks=[
            Chunk(
                id=f"{pid}:0",
                page_id=pid,
                url=url,
                section=None,
                text="hello world",
                token_count=2,
                embedding=embedding,
            )
        ],
    )


def test_session_roundtrip(tmp_path):
    path = tmp_path / "s.json"
    s1 = SessionStore(path)
    emb = np.array([0.6, 0.8], dtype=np.float32)
    s1.add_page(_page("https://a", emb))
    s1.save()

    s2 = SessionStore(path)
    s2.load()
    assert len(s2.pages) == 1
    rec = s2.pages[0].chunks[0].embedding
    assert rec is not None
    assert np.allclose(rec, emb, atol=1e-6)


def test_session_clear(tmp_path):
    s = SessionStore(tmp_path / "s.json")
    s.add_page(_page("https://a"))
    s.save()
    s.clear()
    assert s.pages == []
    assert not (tmp_path / "s.json").exists()


def test_session_isolation(tmp_path):
    a = SessionStore(tmp_path / "a.json")
    b = SessionStore(tmp_path / "b.json")
    a.add_page(_page("https://a"))
    assert b.pages == []
