"""Ingestion pipeline: sources -> sections -> chunks.

Phase 3 extends this with embedding and storing the chunks.
"""

from groundwork.config import Settings, get_settings
from groundwork.documents import Chunk, Section
from groundwork.ingest.chunker import chunk_sections
from groundwork.ingest.loaders import load_source, looks_like_references
from groundwork.sources import Source, load_sources
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
    length = get_token_counter(settings.embedding_model)
    chunks = chunk_sections(load_corpus(), length, settings.chunk_size, settings.chunk_overlap)
    # Drop bibliography chunks from papers (PDFs). The Markdown notes keep their short
    # reference sections, which name the papers behind each technique.
    return [c for c in chunks if c.page is None or not looks_like_references(c.text)]
