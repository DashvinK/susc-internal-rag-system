"""Text cleaning: normalize whitespace and strip extraction noise.

Deliberately conservative -- it fixes mechanical artifacts (de-hyphenation,
bare page numbers, runaway blank lines) without rewording content, so chunk
text stays faithful to the source for honest citations.
"""

from __future__ import annotations

import re

_MULTISPACE = re.compile(r"[ \t]+")
_MULTINEWLINE = re.compile(r"\n{3,}")
# A line that is nothing but a number (a stray page number from PDF extraction).
_PAGE_NUMBER_LINE = re.compile(r"^\s*\d{1,4}\s*$")
# A word split across a line break by hyphenation: "requi-\nrement" -> "requirement".
_DEHYPHENATE = re.compile(r"(\w)-\n(\w)")


def clean_text(text: str) -> str:
    if not text:
        return ""

    # Normalize line endings and non-breaking spaces first.
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")

    # Re-join hyphenated line breaks before we touch newlines.
    text = _DEHYPHENATE.sub(r"\1\2", text)

    cleaned_lines: list[str] = []
    for line in text.split("\n"):
        line = _MULTISPACE.sub(" ", line).strip()
        if _PAGE_NUMBER_LINE.match(line):
            continue  # drop standalone page numbers
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = _MULTINEWLINE.sub("\n\n", text)
    return text.strip()
