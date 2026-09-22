"""
Centralized application configuration.

All configuration is loaded from environment variables (via a .env file in
development). Nothing sensitive is ever hardcoded here — see .env.example
for the full list of supported variables.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM provider ---
    # "openai" (default) or "ollama" (fully local, no API key required)
    llm_provider: str = "openai"

    # --- OpenAI ---
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    openai_vision_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # --- Ollama (local) ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3.2"
    # llama3.2 (1B/3B) is text-only. Pull "llama3.2-vision" separately for figure
    # descriptions, or leave as-is and figures will fall back to their captions.
    ollama_vision_model: str = "llama3.2-vision"
    # llama3.2 cannot generate embeddings. Pull a dedicated embedding model, e.g.:
    #   ollama pull nomic-embed-text
    ollama_embedding_model: str = "nomic-embed-text"

    # --- Vector store ---
    vector_store_provider: str = "chroma"
    chroma_persist_dir: str = "./vectorstore"
    chroma_collection_name: str = "research_papers"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""

    # --- PostgreSQL ---
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "research_intelligence"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    use_postgres: bool = False

    # --- Storage paths ---
    data_dir: str = "./data"
    papers_dir: str = "./data/papers"
    images_dir: str = "./data/images"
    processed_dir: str = "./data/processed"

    # --- Web search ---
    serpapi_api_key: str = ""

    # --- Langfuse ---
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    enable_langfuse: bool = False

    # --- App behaviour ---
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k_retrieval: int = 6
    max_review_retries: int = 2
    log_level: str = "INFO"

    # --- FastAPI ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- MCP ---
    mcp_server_host: str = "127.0.0.1"
    mcp_calculator_port: int = 8765
    mcp_document_port: int = 8766
    mcp_search_port: int = 8767

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    def ensure_directories(self) -> None:
        """Create all required data directories if they do not exist."""
        for path in (self.data_dir, self.papers_dir, self.images_dir, self.processed_dir, self.chroma_persist_dir):
            Path(path).mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
