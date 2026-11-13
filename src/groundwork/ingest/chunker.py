"""Recursive chunking with overlap.

The idea, in two steps:

1. Split. Cut the text at the most meaningful boundary available: paragraphs first,
   then lines, then sentences, then clauses, then words. Only a piece that is still
   too big gets split again with the next, finer separator. The result is a list of
   small "atomic" pieces that each fit in one chunk.

2. Merge. Walk through the pieces and pack consecutive ones into a chunk until the
   next piece would go over the size limit. Start the next chunk a little earlier,
   repeating the last pieces of the previous chunk (the overlap), so a sentence near
   a boundary appears whole in at least one chunk.

Pieces are tracked as (start, end) character positions instead of copied strings,
so every chunk is an exact slice of the original text and can be traced back to it.
"""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

from groundwork.documents import Chunk, Section

SEPARATORS = ("\n\n", "\n", ". ", "; ", ", ", " ")


@dataclass(frozen=True)
class _Piece:
    start: int
    end: int
    tokens: int


def _hard_split(text: str, start: int, end: int, length, size: int) -> list[_Piece]:
    """Last resort for text with no separators at all (e.g. a very long URL)."""
    tokens = length(text[start:end])
    step = max(1, (end - start) * size // max(tokens, 1))
    return [
        _Piece(s, min(s + step, end), length(text[s : min(s + step, end)]))
        for s in range(start, end, step)
    ]


def _split(text: str, start: int, end: int, length, size: int, separators) -> list[_Piece]:
    tokens = length(text[start:end])
    if tokens <= size:
        return [_Piece(start, end, tokens)]

    # Use the coarsest separator that actually occurs in this stretch of text.
    for i, sep in enumerate(separators):
        if sep in text[start:end]:
            finer = separators[i + 1 :]
            break
    else:
        return _hard_split(text, start, end, length, size)

    pieces: list[_Piece] = []
    pos = start
    while pos < end:
        found = text.find(sep, pos, end)
        cut = end if found == -1 else found + len(sep)  # the separator stays with its piece
        pieces.extend(_split(text, pos, cut, length, size, finer))
        pos = cut
    return pieces


def _merge(pieces: list[_Piece], size: int, overlap: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    i, n = 0, len(pieces)
    while i < n:
        total, j = 0, i
        while j < n and total + pieces[j].tokens <= size:
            total += pieces[j].tokens
            j += 1
        j = max(j, i + 1)  # always make progress, even past an oversized piece
        spans.append((pieces[i].start, pieces[j - 1].end))
        if j >= n:
            break
        # Step back from j to repeat up to `overlap` tokens in the next chunk.
        k, carried = j, 0
        while k - 1 > i and carried + pieces[k - 1].tokens <= overlap:
            carried += pieces[k - 1].tokens
            k -= 1
        i = k
    return spans


def split_text(
    text: str,
    length: Callable[[str], int],
    size: int,
    overlap: int,
    separators: tuple[str, ...] = SEPARATORS,
) -> list[tuple[int, int]]:
    """Return (start, end) character spans of chunks of at most ~`size` tokens."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than the chunk size")
    if not text:
        return []
    return _merge(_split(text, 0, len(text), length, size, separators), size, overlap)


def chunk_id(section: Section, text: str) -> str:
    key = f"{section.source_id}|{section.page}|{section.section}|{text}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def chunk_sections(
    sections: list[Section],
    length: Callable[[str], int],
    size: int,
    overlap: int,
    min_tokens: int = 15,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in sections:
        index = 0
        for start, end in split_text(section.text, length, size, overlap):
            raw = section.text[start:end]
            text = raw.strip()
            n_tokens = length(text) if text else 0
            if n_tokens < min_tokens:
                continue  # page numbers, stray captions, empty title pages
            start += len(raw) - len(raw.lstrip())
            chunks.append(
                Chunk(
                    id=chunk_id(section, text),
                    source_id=section.source_id,
                    title=section.title,
                    path=section.path,
                    text=text,
                    page=section.page,
                    section=section.section,
                    index=index,
                    start_char=start,
                    end_char=start + len(text),
                    n_tokens=n_tokens,
                )
            )
            index += 1
    return chunks
