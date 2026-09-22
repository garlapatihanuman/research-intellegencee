"""
PDF parsing: text extraction with lightweight section detection, using
PyMuPDF (fitz). This module is intentionally dependency-light so it can run
without any external API calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import fitz  # PyMuPDF

from app.logging_config import get_logger

logger = get_logger(__name__)

# Common research-paper section headers, used for lightweight section detection.
_SECTION_PATTERNS = [
    r"abstract",
    r"introduction",
    r"related work",
    r"background",
    r"methodology|methods|method",
    r"experiments?|experimental setup",
    r"results?",
    r"discussion",
    r"limitations?",
    r"conclusion",
    r"references|bibliography",
    r"appendix",
]
_SECTION_REGEX = re.compile(
    r"^\s*(?:\d+\.?\s*)?(" + "|".join(_SECTION_PATTERNS) + r")\s*$",
    re.IGNORECASE,
)


@dataclass
class PageText:
    page_number: int  # 1-indexed
    text: str
    section: str = "Unknown"


@dataclass
class ParsedDocument:
    filename: str
    num_pages: int
    pages: list[PageText] = field(default_factory=list)


def validate_pdf(file_path: str) -> None:
    """Raise ValueError if the file is not a readable PDF."""
    try:
        doc = fitz.open(file_path)
        if doc.page_count == 0:
            raise ValueError("PDF has no pages.")
        doc.close()
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid or unreadable PDF: {exc}") from exc


def _detect_section(line: str, current_section: str) -> str:
    match = _SECTION_REGEX.match(line.strip())
    if match:
        return match.group(1).title()
    return current_section


def parse_pdf(file_path: str, filename: str) -> ParsedDocument:
    """
    Extract text from every page, doing a lightweight pass to tag each page
    with the most recently seen section heading.
    """
    logger.info("Parsing PDF: %s", filename)
    doc = fitz.open(file_path)
    pages: list[PageText] = []
    current_section = "Unknown"

    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        raw_text = page.get_text("text")

        for line in raw_text.splitlines():
            current_section = _detect_section(line, current_section)

        pages.append(
            PageText(
                page_number=page_index + 1,
                text=raw_text.strip(),
                section=current_section,
            )
        )

    doc.close()
    logger.info("Parsed %d pages from %s", len(pages), filename)
    return ParsedDocument(filename=filename, num_pages=len(pages), pages=pages)
