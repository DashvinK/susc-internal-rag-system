"""Document parsers: file -> list[Page].

Each parser extracts text while preserving a coarse ``section`` label (a heading
or page number) and the ``source`` filename. That metadata is what later powers
citations like ``constitution.md > Article 3``.

OFFLINE ONLY. Imports pypdf / python-docx, which are not deployed to Vercel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Page:
    """A contiguous span of text from one logical section of one file."""

    source: str  # filename, e.g. "event_sop.md"
    section: str  # heading text or "page N"
    text: str


SUPPORTED_SUFFIXES = {".pdf", ".docx", ".md", ".markdown", ".txt"}


def parse_pdf(path: Path) -> list[Page]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[Page] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(Page(source=path.name, section=f"page {i}", text=text))
    return pages


def parse_docx(path: Path) -> list[Page]:
    from docx import Document

    doc = Document(str(path))
    pages: list[Page] = []
    current_section = "body"
    buffer: list[str] = []

    def flush() -> None:
        joined = "\n".join(buffer).strip()
        if joined:
            pages.append(Page(source=path.name, section=current_section, text=joined))

    for para in doc.paragraphs:
        style = (para.style.name or "").lower() if para.style else ""
        if style.startswith("heading") or style == "title":
            # A heading starts a new section.
            flush()
            buffer = []
            current_section = para.text.strip() or current_section
        else:
            buffer.append(para.text)
    flush()
    return pages


def parse_markdown(path: Path) -> list[Page]:
    """Split markdown on ATX headings (``#`` .. ``######``), one Page per section."""
    lines = path.read_text(encoding="utf-8").splitlines()
    pages: list[Page] = []
    current_section = path.stem
    buffer: list[str] = []

    def flush() -> None:
        joined = "\n".join(buffer).strip()
        if joined:
            pages.append(Page(source=path.name, section=current_section, text=joined))

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("#") and stripped.lstrip("#").startswith(" "):
            flush()
            buffer = []
            current_section = stripped.lstrip("#").strip() or current_section
        else:
            buffer.append(line)
    flush()
    return pages


def parse_txt(path: Path) -> list[Page]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    return [Page(source=path.name, section="body", text=text)]


def parse_file(path: Path) -> list[Page]:
    """Dispatch by file extension. Unknown types raise ValueError."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path)
    if suffix == ".docx":
        return parse_docx(path)
    if suffix in {".md", ".markdown"}:
        return parse_markdown(path)
    if suffix == ".txt":
        return parse_txt(path)
    raise ValueError(f"Unsupported file type: {path.name} ({suffix})")


def parse_directory(root: Path) -> list[Page]:
    """Parse every supported file under ``root`` (recursively), sorted for determinism."""
    pages: list[Page] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            pages.extend(parse_file(path))
    return pages
