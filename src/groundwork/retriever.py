"""Semantic search: embed the question, find the nearest chunks in the vector store."""

from dataclasses import dataclass

from groundwork.config import Settings, get_settings
from groundwork.documents import Chunk
from groundwork.embeddings import Embedder, get_embedder
from groundwork.store import VectorStore


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # cosine similarity between the question and the chunk, -1 to 1


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore):
        self.embedder = embedder
        self.store = store

    def search(self, query: str, k: int) -> list[Hit]:
        vector = self.embedder.embed_query(query)
        return [Hit(chunk, score) for chunk, score in self.store.query(vector, k)]


def open_store(settings: Settings | None = None) -> VectorStore:
    settings = settings or get_settings()
    return VectorStore(settings.chroma_dir, settings.collection, settings.embedding_model)


def get_retriever(settings: Settings | None = None) -> Retriever:
    settings = settings or get_settings()
    return Retriever(get_embedder(settings), open_store(settings))
