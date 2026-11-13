"""The vector database: ChromaDB, stored on disk in data/chroma/.

Chroma keeps, for every chunk: its id, its vector, its text, and metadata (source,
page, section...). Searching returns the chunks whose vectors are closest to the query
vector, using an HNSW index so it doesn't have to compare against every vector.
"""

from pathlib import Path

import chromadb
import numpy as np

from groundwork.documents import Chunk

# Bump this when the way chunks are turned into embedding text changes
# (see Chunk.embedding_text). Old vectors would no longer match, so the store refuses
# to mix them and asks for a rebuild.
EMBEDDING_FORMAT = "title-header-v1"

_BATCH = 500


class StoreMismatchError(RuntimeError):
    pass


def _to_metadata(chunk: Chunk) -> dict:
    meta = {
        "source_id": chunk.source_id,
        "title": chunk.title,
        "path": chunk.path,
        "index": chunk.index,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
        "n_tokens": chunk.n_tokens,
    }
    # Chroma metadata can't hold None, so optional fields are only stored when present.
    if chunk.page is not None:
        meta["page"] = chunk.page
    if chunk.section is not None:
        meta["section"] = chunk.section
    return meta


def _from_record(chunk_id: str, text: str, meta: dict) -> Chunk:
    return Chunk(
        id=chunk_id,
        source_id=meta["source_id"],
        title=meta["title"],
        path=meta["path"],
        text=text,
        page=meta.get("page"),
        section=meta.get("section"),
        index=meta["index"],
        start_char=meta["start_char"],
        end_char=meta["end_char"],
        n_tokens=meta["n_tokens"],
    )


class VectorStore:
    def __init__(self, path: Path, collection: str, embedding_model: str):
        self.client = chromadb.PersistentClient(
            path=str(path),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self.name = collection
        self.embedding_model = embedding_model
        self.collection = self._open()

    def _open(self):
        collection = self.client.get_or_create_collection(
            self.name,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"embedding_model": self.embedding_model, "embedding_format": EMBEDDING_FORMAT},
            embedding_function=None,  # we pass our own vectors; never let Chroma embed
        )
        meta = collection.metadata or {}
        expected = (self.embedding_model, EMBEDDING_FORMAT)
        found = (meta.get("embedding_model"), meta.get("embedding_format"))
        if found != expected:
            raise StoreMismatchError(
                f"The index was built with {found}, but the app now uses {expected}. "
                "Vectors from different models can't be compared. "
                "Rebuild it with: uv run python scripts/ingest.py --rebuild"
            )
        return collection

    def reset(self) -> None:
        """Delete everything and start an empty collection."""
        self.client.delete_collection(self.name)
        self.collection = self._open()

    def count(self) -> int:
        return self.collection.count()

    def ids(self) -> set[str]:
        return set(self.collection.get(include=[])["ids"])

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        for i in range(0, len(chunks), _BATCH):
            batch = chunks[i : i + _BATCH]
            self.collection.upsert(
                ids=[c.id for c in batch],
                embeddings=vectors[i : i + _BATCH].tolist(),
                documents=[c.text for c in batch],
                metadatas=[_to_metadata(c) for c in batch],
            )

    def delete(self, ids: set[str]) -> None:
        ids = list(ids)
        for i in range(0, len(ids), _BATCH):
            self.collection.delete(ids=ids[i : i + _BATCH])

    def query(self, vector: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        """The k nearest chunks with their cosine similarity (1 = same direction)."""
        k = min(k, self.count())
        if k == 0:
            return []
        res = self.collection.query(
            query_embeddings=[vector.tolist()],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        return [
            # Chroma returns cosine *distance* = 1 - cosine similarity.
            (_from_record(cid, doc, meta), 1.0 - dist)
            for cid, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    def chunks(self) -> list[Chunk]:
        """Every stored chunk."""
        got = self.collection.get(include=["documents", "metadatas"])
        return [_from_record(*rec) for rec in zip(got["ids"], got["documents"], got["metadatas"])]

    def texts(self) -> dict[str, str]:
        """Every stored chunk's text by id (used to build the keyword index)."""
        got = self.collection.get(include=["documents"])
        return dict(zip(got["ids"], got["documents"]))

    def fetch(self, ids: list[str], vector: np.ndarray) -> list[tuple[Chunk, float]]:
        """Specific chunks with their cosine similarity to `vector`.

        Used for chunks that keyword search found but vector search didn't rank, so
        every result still has a semantic score.
        """
        if not ids:
            return []
        got = self.collection.get(ids=ids, include=["documents", "metadatas", "embeddings"])
        return [
            (_from_record(cid, doc, meta), float(np.dot(np.asarray(emb), vector)))
            for cid, doc, meta, emb in zip(
                got["ids"], got["documents"], got["metadatas"], got["embeddings"]
            )
        ]

    def documents(self) -> list[dict]:
        """One row per source document: id, title, path and number of chunks."""
        rows: dict[str, dict] = {}
        for meta in self.collection.get(include=["metadatas"])["metadatas"]:
            row = rows.setdefault(
                meta["source_id"],
                {"source_id": meta["source_id"], "title": meta["title"], "path": meta["path"], "chunks": 0},
            )
            row["chunks"] += 1
        return sorted(rows.values(), key=lambda r: r["source_id"])
