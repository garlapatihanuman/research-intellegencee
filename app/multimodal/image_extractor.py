"""
Figure/image extraction from PDF pages using PyMuPDF.

For each embedded image we also try to recover a nearby caption (a line
starting with "Figure N" / "Fig. N") by scanning the page's text blocks.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

import fitz  # PyMuPDF

from app.config import settings
from app.logging_config import get_logger
from app.models.schemas import FigureRecord

logger = get_logger(__name__)

_CAPTION_REGEX = re.compile(r"^(fig(?:ure)?\.?\s*\d+)[:.\-]?\s*(.*)", re.IGNORECASE)


def _find_caption(page: fitz.Page) -> str:
    """Best-effort caption lookup: scan page text for a 'Figure N: ...' line."""
    text = page.get_text("text")
    for line in text.splitlines():
        match = _CAPTION_REGEX.match(line.strip())
        if match:
            return line.strip()
    return ""


def extract_figures(file_path: str, document_id: str, filename: str) -> list[FigureRecord]:
    """Extract every embedded raster image from the PDF as a FigureRecord."""
    logger.info("Extracting figures from %s", filename)
    doc = fitz.open(file_path)
    figures: list[FigureRecord] = []
    figure_counter = 0

    out_dir = Path(settings.images_dir) / document_id
    out_dir.mkdir(parents=True, exist_ok=True)

    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        image_list = page.get_images(full=True)
        if not image_list:
            continue

        caption = _find_caption(page)

        for img in image_list:
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping unreadable image xref=%s: %s", xref, exc)
                continue

            image_bytes = base_image["image"]
            ext = base_image.get("ext", "png")

            # Skip tiny images (icons, decorative lines) — heuristics only.
            if len(image_bytes) < 3000:
                continue

            figure_counter += 1
            image_path = out_dir / f"figure_{figure_counter:03d}.{ext}"
            image_path.write_bytes(image_bytes)

            figures.append(
                FigureRecord(
                    document_id=document_id,
                    filename=filename,
                    page_number=page_index + 1,
                    figure_number=figure_counter,
                    figure_id=f"fig_{uuid.uuid4().hex[:12]}",
                    caption=caption,
                    image_path=str(image_path),
                )
            )

    doc.close()
    logger.info("Extracted %d figures from %s", len(figures), filename)
    return figures
