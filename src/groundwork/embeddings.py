"""Turn text into vectors (embeddings).

An embedding model maps a text to a fixed-length list of numbers (384 for bge-small)
such that texts with similar meaning end up close together. Vectors are normalized to
length 1, so the cosine similarity of two texts is just the dot product of their vectors.
"""

from functools import lru_cache
from typing import Protocol

import numpy as np

from groundwork.config import Settings, get_settings


class Embedder(Protocol):
    """Anything that can embed documents and queries. Tests use a fake one."""

    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, query_instruction: str = ""):
        self.name = model_name
        self.query_instruction = query_instruction
        self._model = None

    @property
    def model(self):
        # Loaded on first use, so commands that end up embedding nothing stay fast.
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            try:
                # Use the copy already on disk without asking Hugging Face for updates:
                # ~0.4 s instead of 20-40 s of network checks, and it works offline
                # (handy for a live demo on unreliable Wi-Fi).
                self._model = SentenceTransformer(self.name, local_files_only=True)
            except OSError:  # not on disk yet: first run, download it
                self._model = SentenceTransformer(self.name)
        return self._model

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 64,
        )

    def embed_query(self, text: str) -> np.ndarray:
        # Queries get the instruction prefix, documents don't (see config.query_instruction).
        return self.model.encode(
            self.query_instruction + text, normalize_embeddings=True, convert_to_numpy=True
        )


@lru_cache
def _cached(model_name: str, query_instruction: str) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(model_name, query_instruction)


def get_embedder(settings: Settings | None = None) -> SentenceTransformerEmbedder:
    settings = settings or get_settings()
    return _cached(settings.embedding_model, settings.query_instruction)
