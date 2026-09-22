"""
Shared chat-completion helper used by every agent.

Supports two providers, selected via LLM_PROVIDER in .env:
  - "openai" (default): calls the OpenAI API, requires OPENAI_API_KEY
  - "ollama": calls a local Ollama server, no API key required

Every agent module calls chat()/chat_json() from here rather than talking to
either backend directly, so the provider can be swapped in one place.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)


def _openai_client():
    from openai import OpenAI

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")
    return OpenAI(api_key=settings.openai_api_key)


def _chat_openai(system_prompt: str, user_prompt: str, temperature: float, json_mode: bool, model: Optional[str]) -> str:
    response = _openai_client().chat.completions.create(
        model=model or settings.openai_chat_model,
        temperature=temperature,
        response_format={"type": "json_object"} if json_mode else {"type": "text"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or ""


def _chat_ollama(system_prompt: str, user_prompt: str, temperature: float, json_mode: bool, model: Optional[str]) -> str:
    import requests

    payload: dict[str, Any] = {
        "model": model or settings.ollama_chat_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }
    if json_mode:
        payload["format"] = "json"

    try:
        response = requests.post(f"{settings.ollama_base_url}/api/chat", json=payload, timeout=180)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {settings.ollama_base_url}. Is `ollama serve` running "
            f"and have you pulled '{payload['model']}' (`ollama pull {payload['model']}`)? "
            f"Original error: {exc}"
        ) from exc

    data = response.json()
    return data.get("message", {}).get("content", "")


def chat(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    json_mode: bool = False,
    model: Optional[str] = None,
) -> str:
    """Single-turn chat completion. Returns the raw text content."""
    if settings.llm_provider == "ollama":
        return _chat_ollama(system_prompt, user_prompt, temperature, json_mode, model)
    return _chat_openai(system_prompt, user_prompt, temperature, json_mode, model)


def chat_json(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> dict[str, Any]:
    """Chat completion that expects and parses a JSON object response."""
    raw = chat(system_prompt, user_prompt, temperature=temperature, json_mode=True)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.error("Failed to parse JSON from LLM response: %.200s", raw)
        return {}
