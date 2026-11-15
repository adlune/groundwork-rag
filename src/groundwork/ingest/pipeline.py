"""Ingestion pipeline: sources -> sections -> chunks -> vectors in the store."""

from dataclasses import dataclass

from groundwork.config import Settings, get_settings
from groundwork.documents import Chunk, Section
from groundwork.embeddings import Embedder
from groundwork.ingest.chunker import chunk_sections
from groundwork.ingest.loaders import load_source, looks_like_noise, looks_like_references
from groundwork.sources import Source, load_sources
from groundwork.store import VectorStore
from groundwork.tokens import get_token_counter


def load_corpus(sources: list[Source] | None = None) -> list[Section]:
    sections: list[Section] = []
    for src in sources or load_sources():
        if not src.file.exists():
            raise FileNotFoundError(
                f"{src.file} is missing. Run: uv run python scripts/download_corpus.py"
            )
        sections.extend(load_source(src))
    return sections


def build_chunks(settings: Settings | None = None) -> list[Chunk]:
    settings = settings or get_settings()
    length = get_token_counter(settings.embedding_model, settings.embedding_revision)
    chunks = chunk_sections(load_corpus(), length, settings.chunk_size, settings.chunk_overlap)
    # Quality filters for papers (PDFs): drop bibliographies, flattened result tables and
    # garbled text. The Markdown notes keep their short reference sections, which name
    # the papers behind each technique.
    return [
        c
        for c in chunks
        if c.page is None or not (looks_like_references(c.text) or looks_like_noise(c.text))
    ]


@dataclass(frozen=True)
class IngestReport:
    added: int
    removed: int
    unchanged: int

    @property
    def total(self) -> int:
        return self.added + self.unchanged


def ingest(chunks: list[Chunk], embedder: Embedder, store: VectorStore) -> IngestReport:
    """Make the store contain exactly these chunks, embedding only what is new.

    Chunk ids are content hashes, so an unchanged chunk keeps its id and is skipped,
    an edited chunk gets a new id (embedded and added) and its old id disappears from
    the corpus (deleted from the store). Running this twice in a row does nothing the
    second time.
    """
    unique = list({c.id: c for c in chunks}.values())  # identical text in one place: keep one
    existing = store.ids()
    current = {c.id for c in unique}

    new = [c for c in unique if c.id not in existing]
    stale = existing - current
    if new:
        store.upsert(new, embedder.embed_documents([c.embedding_text for c in new]))
    if stale:
        store.delete(stale)
    return IngestReport(added=len(new), removed=len(stale), unchanged=len(current) - len(new))
