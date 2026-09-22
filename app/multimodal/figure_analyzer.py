"""
Generates a textual description of an extracted figure using a vision-capable
chat model — OpenAI (default) or a local Ollama vision model, selected via
LLM_PROVIDER in .env.

llama3.2 (1B/3B, text-only) cannot see images. If you want real figure
descriptions locally, pull a vision-capable model, e.g.:
    ollama pull llama3.2-vision
If that model isn't available, this module falls back to using the figure's
caption (or a generic placeholder) so ingestion never hard-fails on this step.
"""

from __future__ import annotations

import base64
from pathlib import Path

from app.config import settings
from app.logging_config import get_logger
from app.models.schemas import FigureRecord

logger = get_logger(__name__)

_DESCRIPTION_PROMPT = (
    "You are analyzing a figure extracted from a research paper. Describe, in 2-4 "
    "sentences, what the figure shows: chart type, axes/variables if visible, "
    "the general trend or architecture depicted, and any labels or entities you can "
    "read. Be factual and specific — this description will be used for semantic "
    "search, so include concrete terms a reader might search for. If the image is "
    "unclear, say so briefly instead of guessing."
)


def _encode_image(image_path: str) -> str:
    return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")


def _describe_openai(figure: FigureRecord) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    image_b64 = _encode_image(figure.image_path)
    ext = Path(figure.image_path).suffix.lstrip(".") or "png"

    response = client.chat.completions.create(
        model=settings.openai_vision_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": _DESCRIPTION_PROMPT
                        + (f"\n\nCaption found near this figure: {figure.caption}" if figure.caption else ""),
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{image_b64}"}},
                ],
            }
        ],
        max_tokens=250,
    )
    return (response.choices[0].message.content or "").strip()


def _describe_ollama(figure: FigureRecord) -> str:
    import requests

    image_b64 = _encode_image(figure.image_path)
    prompt = _DESCRIPTION_PROMPT + (
        f"\n\nCaption found near this figure: {figure.caption}" if figure.caption else ""
    )

    response = requests.post(
        f"{settings.ollama_base_url}/api/chat",
        json={
            "model": settings.ollama_vision_model,
            "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("message", {}).get("content", "").strip()


def describe_figure(figure: FigureRecord) -> str:
    """
    Produce a description for one figure using the configured provider.
    Falls back to the caption (or a generic placeholder) if the call fails,
    no vision model is available, or no API key is configured, so ingestion
    never hard-fails on this step.
    """
    fallback = figure.caption or f"Figure {figure.figure_number} on page {figure.page_number}."

    if settings.llm_provider == "ollama":
        try:
            return _describe_ollama(figure) or fallback
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Ollama figure description failed for %s (is '%s' pulled? "
                "`ollama pull %s`). Falling back to caption. Error: %s",
                figure.figure_id,
                settings.ollama_vision_model,
                settings.ollama_vision_model,
                exc,
            )
            return fallback

    if not settings.openai_api_key:
        logger.warning("No OPENAI_API_KEY configured; using caption as figure description.")
        return fallback

    try:
        return _describe_openai(figure) or fallback
    except Exception as exc:  # noqa: BLE001
        logger.error("Figure description failed for %s: %s", figure.figure_id, exc)
        return fallback


def describe_figures(figures: list[FigureRecord]) -> list[FigureRecord]:
    """Populate `generated_description` for a batch of figures in place."""
    for fig in figures:
        fig.generated_description = describe_figure(fig)
    return figures
