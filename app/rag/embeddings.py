"""
Embedding generation. Supports two providers, selected via LLM_PROVIDER in
.env (same switch used by app/agents/llm.py):

  - "openai" (default): OpenAI embeddings API, requires OPENAI_API_KEY
  - "ollama": a local Ollama embedding model (NOT the chat model — llama3.2
    itself cannot produce embeddings). Pull one first, e.g.:
        ollama pull nomic-embed-text
"""

from __future__ import annotations

from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Thin wrapper around whichever embedding backend is configured."""

    def __init__(self, model: str | None = None) -> None:
        self._explicit_model = model
        self._client = None

    @property
    def model(self) -> str:
        if self._explicit_model:
            return self._explicit_model
        if settings.llm_provider == "ollama":
            return settings.ollama_embedding_model
        return settings.openai_embedding_model

    @property
    def _openai_client(self):
        if self._client is None:
            from openai import OpenAI

            if not settings.openai_api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Add it to your .env file before "
                    "running ingestion or retrieval, or set LLM_PROVIDER=ollama."
                )
            self._client = OpenAI(api_key=settings.openai_api_key)
        return self._client

    @retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3))
    def _embed_texts_openai(self, texts: list[str]) -> list[list[float]]:
        response = self._openai_client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    @retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3))
    def _embed_texts_ollama(self, texts: list[str]) -> list[list[float]]:
        import requests

        # Ollama's /api/embeddings endpoint embeds one prompt per call.
        vectors: list[list[float]] = []
        for text in texts:
            try:
                response = requests.post(
                    f"{settings.ollama_base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                    timeout=60,
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                raise RuntimeError(
                    f"Could not reach Ollama embeddings at {settings.ollama_base_url}. "
                    f"Have you pulled '{self.model}'? Run: ollama pull {self.model}. "
                    f"Original error: {exc}"
                ) from exc
            data = response.json()
            embedding = data.get("embedding")
            if not embedding:
                raise RuntimeError(f"Ollama returned no embedding for model '{self.model}': {data}")
            vectors.append(embedding)
        return vectors

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per input string."""
        if not texts:
            return []
        if settings.llm_provider == "ollama":
            return self._embed_texts_ollama(texts)
        return self._embed_texts_openai(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


embedding_service = EmbeddingService()
