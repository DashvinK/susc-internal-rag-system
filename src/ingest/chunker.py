"""Recursive character-based chunker.

Why write this by hand instead of using a framework? Chunking is the single
biggest lever on retrieval quality, and the behaviour is simple enough to own:
try to split on the most semantic boundary available (paragraph, then line,
then sentence, then word), never exceed ``chunk_size`` characters, and carry a
small ``overlap`` so a fact that straddles a boundary survives in both chunks.

Every chunk keeps its source + section metadata and a stable ``id`` so the API
can cite it and the evaluation harness can match it against gold sources.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field

from src.ingest.parsers import Page

# Ordered from most to least semantic. The empty string is the hard fallback:
# split mid-token only if a single "word" is itself longer than chunk_size.
_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    section: str
    chunk_index: int  # position of this chunk within its source file
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _split_recursive(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Split ``text`` into pieces no larger than ``chunk_size`` (pre-overlap)."""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    separator = separators[0]
    rest = separators[1:]

    if separator == "":
        # Hard character split -- last resort for a single giant token.
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    parts = text.split(separator)
    chunks: list[str] = []
    buffer = ""
    for part in parts:
        candidate = part if not buffer else f"{buffer}{separator}{part}"
        if len(candidate) <= chunk_size:
            buffer = candidate
            continue
        # ``candidate`` is too big. Emit what we have, then handle ``part``.
        if buffer:
            chunks.append(buffer)
        if len(part) > chunk_size:
            # A single part still exceeds the limit -> recurse with a finer separator.
            chunks.extend(_split_recursive(part, chunk_size, rest))
            buffer = ""
        else:
            buffer = part
    if buffer:
        chunks.append(buffer)
    return [c.strip() for c in chunks if c.strip()]


def _apply_overlap(pieces: list[str], overlap: int) -> list[str]:
    """Prepend the tail of the previous piece to each piece (character overlap)."""
    if overlap <= 0 or len(pieces) <= 1:
        return pieces
    out = [pieces[0]]
    for i in range(1, len(pieces)):
        tail = pieces[i - 1][-overlap:]
        out.append(f"{tail} {pieces[i]}".strip())
    return out


def _chunk_id(source: str, section: str, index: int, text: str) -> str:
    h = hashlib.sha1(f"{source}|{section}|{index}|{text}".encode()).hexdigest()
    return h[:16]


def chunk_page(page: Page, chunk_size: int, overlap: int, start_index: int) -> list[Chunk]:
    pieces = _split_recursive(page.text, chunk_size, _SEPARATORS)
    pieces = _apply_overlap(pieces, overlap)
    chunks: list[Chunk] = []
    for offset, piece in enumerate(pieces):
        idx = start_index + offset
        chunks.append(
            Chunk(
                id=_chunk_id(page.source, page.section, idx, piece),
                text=piece,
                source=page.source,
                section=page.section,
                chunk_index=idx,
                metadata={"char_len": len(piece)},
            )
        )
    return chunks


def chunk_pages(pages: list[Page], chunk_size: int, overlap: int) -> list[Chunk]:
    """Chunk a list of pages. ``chunk_index`` is per-source, counting across sections."""
    chunks: list[Chunk] = []
    per_source_counter: dict[str, int] = {}
    for page in pages:
        start = per_source_counter.get(page.source, 0)
        page_chunks = chunk_page(page, chunk_size, overlap, start)
        per_source_counter[page.source] = start + len(page_chunks)
        chunks.extend(page_chunks)
    return chunks
