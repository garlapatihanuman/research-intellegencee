"""
Optional web search tool. Only invoked by the Tool Agent when a question
explicitly requires information beyond the uploaded papers (e.g. "search
for additional information about X").

Uses SerpAPI if configured; otherwise returns a clear message rather than
silently failing, so the agent can report that web search is unavailable.
"""

from __future__ import annotations

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)


def search_web(query: str, num_results: int = 5) -> list[dict[str, str]]:
    """Perform a web search and return a list of {title, link, snippet} dicts."""
    if not settings.serpapi_api_key:
        logger.warning("SERPAPI_API_KEY not configured; web search unavailable.")
        return [
            {
                "title": "Web search unavailable",
                "link": "",
                "snippet": (
                    "No SERPAPI_API_KEY is configured, so external web search could not "
                    "be performed. Configure it in .env to enable this tool."
                ),
            }
        ]

    try:
        import requests

        response = requests.get(
            "https://serpapi.com/search",
            params={"q": query, "api_key": settings.serpapi_api_key, "num": num_results},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        results = []
        for item in data.get("organic_results", [])[:num_results]:
            results.append(
                {
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                }
            )
        return results
    except Exception as exc:  # noqa: BLE001
        logger.error("Web search failed: %s", exc)
        return [{"title": "Web search error", "link": "", "snippet": str(exc)}]
