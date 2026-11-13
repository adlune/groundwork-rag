import hashlib

import numpy as np
import pytest

from groundwork.documents import Chunk
from groundwork.store import VectorStore


class FakeEmbedder:
    """Bag-of-words embedder: each word is hashed into one of 64 dimensions.

    Texts that share words get similar vectors, which is enough to test the store and
    retriever without downloading a real model.
    """

    name = "fake-bow-64"
    dim = 64

    def _embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for word in text.lower().split():
            word = word.strip(".,:;!?()")
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._embed(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(text)


def make_chunk(text: str, source_id: str = "doc", page: int | None = None, section: str | None = "S") -> Chunk:
    cid = hashlib.sha1(f"{source_id}|{page}|{section}|{text}".encode()).hexdigest()[:16]
    return Chunk(
        id=cid, source_id=source_id, title=f"Title of {source_id}", path=f"{source_id}.md",
        text=text, page=page, section=section, index=0, start_char=0, end_char=len(text),
        n_tokens=len(text.split()),
    )


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def store(tmp_path) -> VectorStore:
    return VectorStore(tmp_path / "chroma", "test", FakeEmbedder.name)
