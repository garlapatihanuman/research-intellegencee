"""
Table extraction using pdfplumber, which handles ruled and semi-ruled tables
reasonably well for typical research-paper PDFs.
"""

from __future__ import annotations

import re
import uuid

import pdfplumber

from app.logging_config import get_logger
from app.models.schemas import TableRecord

logger = get_logger(__name__)

_CAPTION_REGEX = re.compile(r"^(table\s*\d+)[:.\-]?\s*(.*)", re.IGNORECASE)


def _find_caption(page_text: str) -> str:
    for line in page_text.splitlines():
        match = _CAPTION_REGEX.match(line.strip())
        if match:
            return line.strip()
    return ""


def _table_to_markdown(rows: list[list[str | None]]) -> str:
    clean_rows = [[cell.strip() if cell else "" for cell in row] for row in rows]
    if not clean_rows:
        return ""
    header, *body = clean_rows
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in body:
        # Pad/truncate rows to header length for a well-formed markdown table.
        row = (row + [""] * len(header))[: len(header)]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def extract_tables(file_path: str, document_id: str, filename: str) -> list[TableRecord]:
    """Extract tables from every page and render them as markdown."""
    logger.info("Extracting tables from %s", filename)
    tables: list[TableRecord] = []
    table_counter = 0

    with pdfplumber.open(file_path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            try:
                raw_tables = page.extract_tables()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Table extraction failed on page %d: %s", page_index + 1, exc)
                continue

            if not raw_tables:
                continue

            page_text = page.extract_text() or ""
            caption = _find_caption(page_text)

            for raw_table in raw_tables:
                if not raw_table or len(raw_table) < 2:
                    continue  # skip noise (single-row "tables")
                table_counter += 1
                markdown = _table_to_markdown(raw_table)
                tables.append(
                    TableRecord(
                        document_id=document_id,
                        filename=filename,
                        page_number=page_index + 1,
                        table_number=table_counter,
                        table_id=f"tbl_{uuid.uuid4().hex[:12]}",
                        caption=caption,
                        structured_content=markdown,
                    )
                )

    logger.info("Extracted %d tables from %s", len(tables), filename)
    return tables
