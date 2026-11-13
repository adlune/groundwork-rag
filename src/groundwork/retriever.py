"""Retrieval: dense (embeddings) search, optionally combined with keyword (BM25) search.

Hybrid search runs both searches, then merges the two rankings with Reciprocal Rank
Fusion (RRF): every chunk gets 1 / (60 + its rank) from each list it appears in. A chunk
ranked well by both searches rises to the top; one found by only one search can still
make it. RRF only uses ranks, so the two very different score scales never get mixed.
"""

from dataclasses import dataclass

from groundwork.config import Settings, get_settings
from groundwork.documents import Chunk
from groundwork.embeddings import Embedder, get_embedder
from groundwork.keyword import KeywordIndex
from groundwork.store import VectorStore


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # cosine similarity between the question and the chunk, -1 to 1
    keyword: float | None = None  # BM25 match strength 0-1 (None: keyword search off)


def passes_gate(hit: Hit, min_similarity: float, min_keyword: float) -> bool:
    """Is this chunk relevant enough to send to the model?

    Either signal is enough: similar in meaning (dense score), or containing the
    question's rare words (keyword score). The second rescues short questions like
    "What is QLoRA?", whose dense score is too low to tell apart from off-topic ones.
    """
    if hit.score >= min_similarity:
        return True
    return hit.keyword is not None and hit.keyword >= min_keyword


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        keyword_index: KeywordIndex | None = None,
        candidates: int = 20,
        rrf_k: int = 60,
    ):
        self.embedder = embedder
        self.store = store
        self.keyword_index = keyword_index
        self.candidates = candidates
        self.rrf_k = rrf_k

    @property
    def hybrid(self) -> bool:
        return self.keyword_index is not None

    def search(self, query: str, k: int) -> list[Hit]:
        vector = self.embedder.embed_query(query)
        if not self.hybrid:
            return [Hit(chunk, score) for chunk, score in self.store.query(vector, k)]

        dense = self.store.query(vector, self.candidates)
        keyword = self.keyword_index.search(query, self.candidates)
        keyword_score = dict(keyword)

        fused: dict[str, float] = {}
        for rank, (chunk, _) in enumerate(dense, start=1):
            fused[chunk.id] = fused.get(chunk.id, 0) + 1 / (self.rrf_k + rank)
        for rank, (cid, _) in enumerate(keyword, start=1):
            fused[cid] = fused.get(cid, 0) + 1 / (self.rrf_k + rank)
        top = sorted(fused, key=fused.get, reverse=True)[:k]

        # Every hit needs its dense score too; fetch the ones only keyword search found.
        dense_score = {chunk.id: (chunk, score) for chunk, score in dense}
        missing = [cid for cid in top if cid not in dense_score]
        dense_score.update({chunk.id: (chunk, score) for chunk, score in self.store.fetch(missing, vector)})
        return [
            Hit(dense_score[cid][0], dense_score[cid][1], keyword_score.get(cid, 0.0))
            for cid in top
        ]


def open_store(settings: Settings | None = None) -> VectorStore:
    settings = settings or get_settings()
    return VectorStore(settings.chroma_dir, settings.collection, settings.embedding_model)


def get_retriever(settings: Settings | None = None) -> Retriever:
    settings = settings or get_settings()
    store = open_store(settings)
    index = KeywordIndex(store.texts()) if settings.hybrid else None
    return Retriever(get_embedder(settings), store, index, settings.candidates, settings.rrf_k)
