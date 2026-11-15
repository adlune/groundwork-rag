"""The vector database: ChromaDB, stored on disk in data/chroma/.

Chroma keeps, for every chunk: its id, its vector, its text, and metadata (source,
page, section...). Searching compares the query vector with every stored vector
(exact search), not through Chroma's approximate HNSW index. With a few hundred chunks
that is one small matrix product, and it returns the same ranking on every run. An
HNSW index is rebuilt each time the program starts and can miss some of the true
nearest chunks, so results moved between runs.
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
        self._matrix: tuple[list[str], np.ndarray] | None = None  # loaded on first query

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
        self._matrix = None

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
        self._matrix = None

    def delete(self, ids: set[str]) -> None:
        ids = list(ids)
        for i in range(0, len(ids), _BATCH):
            self.collection.delete(ids=ids[i : i + _BATCH])
        self._matrix = None

    def _vectors(self) -> tuple[list[str], np.ndarray]:
        """All ids and vectors, sorted by id so ties always break the same way."""
        if self._matrix is None:
            got = self.collection.get(include=["embeddings"])
            if not got["ids"]:
                self._matrix = ([], np.zeros((0, 0), dtype=np.float32))
            else:
                order = sorted(range(len(got["ids"])), key=lambda i: got["ids"][i])
                vectors = np.asarray(got["embeddings"], dtype=np.float32)
                self._matrix = ([got["ids"][i] for i in order], vectors[order])
        return self._matrix

    def query(self, vector: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        """The k nearest chunks with their cosine similarity (1 = same direction)."""
        ids, vectors = self._vectors()
        k = min(k, len(ids))
        if k == 0:
            return []
        # Vectors are normalized, so the dot product is the cosine similarity.
        sims = vectors @ np.asarray(vector, dtype=np.float32)
        top = np.argsort(-sims, kind="stable")[:k]
        records = self._records([ids[i] for i in top])
        return [(records[ids[i]], float(sims[i])) for i in top]

    def _records(self, ids: list[str]) -> dict[str, Chunk]:
        got = self.collection.get(ids=ids, include=["documents", "metadatas"])
        return {
            cid: _from_record(cid, doc, meta)
            for cid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"])
        }

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
        all_ids, vectors = self._vectors()
        row = {cid: i for i, cid in enumerate(all_ids)}
        sims = vectors @ np.asarray(vector, dtype=np.float32)
        records = self._records(ids)
        return [(records[cid], float(sims[row[cid]])) for cid in ids if cid in records]

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
