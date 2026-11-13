"""Central configuration.

Every tunable knob of the pipeline lives here so you can change behaviour from a
`.env` file (or environment variables) without touching code.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    # --- Generation (the "G" in RAG), with a local model through Ollama ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:4b"
    # Ollama's default context window is small and longer prompts are cut silently.
    # 5 chunks x 350 tokens + instructions is ~2,500 tokens, so give it room.
    ollama_num_ctx: int = 8192
    llm_timeout: float = 180.0  # seconds; a local model on CPU/small GPU can be slow

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
    # The gate: a chunk is relevant enough to answer from if its cosine similarity
    # reaches min_similarity OR its keyword match reaches min_keyword. Both values
    # were calibrated on eval/questions.yaml with scripts/evaluate.py (docs/evaluation.md).
    min_similarity: float = 0.645
    # Hybrid search: also rank chunks by keyword match (BM25), and let a strong keyword
    # match pass the gate when the dense score is low (short questions like "What is QLoRA?").
    hybrid: bool = True
    min_keyword: float = 0.70
    candidates: int = 20  # how many results each search contributes before fusion
    rrf_k: int = 60  # standard Reciprocal Rank Fusion constant

    # --- Corpus ---
    sources_file: Path = PROJECT_ROOT / "data" / "sources.yaml"


@lru_cache
def get_settings() -> Settings:
    return Settings()
