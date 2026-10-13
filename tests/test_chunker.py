"""Chunker edge cases: empty text, long paragraphs, overlap, metadata."""

from src.ingest.chunker import chunk_pages
from src.ingest.parsers import Page


def _page(text: str, source="doc.md", section="S1") -> Page:
    return Page(source=source, section=section, text=text)


def test_empty_text_yields_no_chunks():
    assert chunk_pages([_page("")], chunk_size=100, overlap=10) == []
    assert chunk_pages([_page("   \n  ")], chunk_size=100, overlap=10) == []


def test_short_text_is_one_chunk_with_metadata():
    chunks = chunk_pages([_page("A short section.")], chunk_size=100, overlap=10)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.text == "A short section."
    assert c.source == "doc.md"
    assert c.section == "S1"
    assert c.chunk_index == 0
    assert len(c.id) == 16  # stable sha1 prefix


def test_long_paragraph_is_split_under_chunk_size():
    # A single paragraph with no good separators, longer than chunk_size.
    long_text = "word " * 400  # 2000 chars
    chunks = chunk_pages([_page(long_text)], chunk_size=200, overlap=0)
    assert len(chunks) > 1
    # No chunk should greatly exceed the limit (overlap adds a little slack).
    assert all(len(c.text) <= 220 for c in chunks)


def test_overlap_shares_text_between_consecutive_chunks():
    text = "\n\n".join(f"Paragraph number {i} with some filler content." for i in range(20))
    no_overlap = chunk_pages([_page(text)], chunk_size=120, overlap=0)
    with_overlap = chunk_pages([_page(text)], chunk_size=120, overlap=40)
    assert len(with_overlap) >= 2
    # With overlap, a later chunk should start with a tail of the previous one.
    tail = no_overlap[0].text[-20:]
    assert any(tail[:10] in c.text for c in with_overlap[1:])


def test_chunk_index_counts_across_sections_per_source():
    pages = [_page("first section text", section="A"), _page("second section text", section="B")]
    chunks = chunk_pages(pages, chunk_size=100, overlap=0)
    assert [c.chunk_index for c in chunks] == [0, 1]
    assert chunks[0].section == "A" and chunks[1].section == "B"


def test_ids_are_unique():
    text = "\n\n".join(f"Distinct paragraph {i}." for i in range(10))
    chunks = chunk_pages([_page(text)], chunk_size=40, overlap=0)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
