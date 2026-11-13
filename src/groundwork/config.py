"""Central configuration.

Every tunable knob of the pipeline lives here so you can change behaviour from a
`.env` file (or environment variables) without touching code.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    # --- Generation (the "G" in RAG) ---
    llm_provider: Literal["ollama", "anthropic"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:4b"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-5-5"

    # --- Retrieval (the "R" in RAG) ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # BGE models are trained to embed short search queries with this prefix and
    # documents without it (asymmetric retrieval). See notes, chapter 8.
    query_instruction: str = "Represent this sentence for searching relevant passages: "
    chroma_dir: Path = PROJECT_ROOT / "data" / "chroma"
    collection: str = "groundwork"
    chunk_size: int = 350  # measured in embedding-model tokens
    chunk_overlap: int = 60
    top_k: int = 5
    # Below this cosine similarity, the best chunk is "not relevant enough" and the
    # app refuses instead of guessing. First estimate from exploring real scores
    # (in-scope questions ~0.69-0.77, out-of-scope <= 0.53); calibrated in Phase 6.
    min_similarity: float = 0.60

    # --- Corpus ---
    sources_file: Path = PROJECT_ROOT / "data" / "sources.yaml"


@lru_cache
def get_settings() -> Settings:
    return Settings()
