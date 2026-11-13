"""The two data types that flow through ingestion.

Section: a natural unit of a document as it was loaded (one PDF page, one Markdown
         section under a heading, one TXT file). Chunks never cross a section, so
         every chunk can be cited with an exact page or heading.
Chunk:   a piece of a section small enough to embed. This is what gets stored in
         the vector database and retrieved at question time.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Section:
    source_id: str  # id from data/sources.yaml
    title: str  # document title, shown in citations
    path: str  # file path relative to the project root
    text: str
    page: int | None = None  # 1-based page number (PDFs)
    section: str | None = None  # heading text (Markdown)


@dataclass(frozen=True)
class Chunk:
    id: str  # content hash: same text from the same place always gets the same id
    source_id: str
    title: str
    path: str
    text: str
    page: int | None
    section: str | None
    index: int  # position of this chunk inside its section
    start_char: int  # where the chunk starts and ends inside the section text
    end_char: int
    n_tokens: int

    @property
    def location(self) -> str:
        """Human-readable place in the document, e.g. 'p. 4' or 'Security lens'."""
        if self.page is not None:
            return f"p. {self.page}"
        return self.section or ""
