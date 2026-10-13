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


SUPPORTED_SUFFIXES = {".pdf", ".docx", ".md", ".markdown", ".txt", ".xlsx", ".xlsm"}


def parse_pdf(path: Path) -> list[Page]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[Page] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(Page(source=path.name, section=f"page {i}", text=text))
    return pages


def _iter_docx_blocks(doc):
    """Yield paragraphs and tables in the order they appear in the document body.

    python-docx exposes paragraphs and tables separately, which loses their
    interleaving and drops table text entirely from a naive paragraph-only read.
    Walking the body XML in order preserves both -- important for form/template
    documents (reimbursement forms, meeting minutes) that are mostly tables.
    """
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    body = doc.element.body
    for child in body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, doc)
        elif isinstance(child, CT_Tbl):
            yield Table(child, doc)


def _table_to_text(table) -> str:
    """Render a table as newline-separated rows, cells joined by ' | '."""
    lines: list[str] = []
    for row in table.rows:
        cells = [cell.text.strip() for cell in row.cells]
        # Collapse the common case of a cell repeated across merged columns.
        deduped: list[str] = []
        for cell in cells:
            if cell and (not deduped or deduped[-1] != cell):
                deduped.append(cell)
        if deduped:
            lines.append(" | ".join(deduped))
    return "\n".join(lines)


def parse_docx(path: Path) -> list[Page]:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(str(path))
    pages: list[Page] = []
    current_section = "body"
    buffer: list[str] = []

    def flush() -> None:
        joined = "\n".join(buffer).strip()
        if joined:
            pages.append(Page(source=path.name, section=current_section, text=joined))

    for block in _iter_docx_blocks(doc):
        if isinstance(block, Paragraph):
            style = (block.style.name or "").lower() if block.style else ""
            if style.startswith("heading") or style == "title":
                # A heading starts a new section.
                flush()
                buffer = []
                current_section = block.text.strip() or current_section
            else:
                buffer.append(block.text)
        elif isinstance(block, Table):
            table_text = _table_to_text(block)
            if table_text:
                buffer.append(table_text)
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


def parse_xlsx(path: Path) -> list[Page]:
    """One Page per worksheet; rows rendered as 'cell | cell | ...'.

    ``data_only=True`` returns computed values instead of raw formulas, and
    ``read_only=True`` keeps memory low on large workbooks. Empty cells and
    empty rows are dropped so sparse sheets don't produce noise.
    """
    from openpyxl import load_workbook

    wb = load_workbook(str(path), read_only=True, data_only=True)
    pages: list[Page] = []
    try:
        for ws in wb.worksheets:
            lines: list[str] = []
            for row in ws.iter_rows(values_only=True):
                cells = [str(c).strip() for c in row if c is not None and str(c).strip()]
                if cells:
                    lines.append(" | ".join(cells))
            text = "\n".join(lines)
            if text.strip():
                pages.append(Page(source=path.name, section=ws.title, text=text))
    finally:
        wb.close()
    return pages


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
    if suffix in {".xlsx", ".xlsm"}:
        return parse_xlsx(path)
    raise ValueError(f"Unsupported file type: {path.name} ({suffix})")


def parse_directory(root: Path) -> list[Page]:
    """Parse every supported file under ``root`` (recursively), sorted for determinism."""
    pages: list[Page] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            pages.extend(parse_file(path))
    return pages
