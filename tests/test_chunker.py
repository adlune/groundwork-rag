from itertools import pairwise

import pytest

from groundwork.documents import Section
from groundwork.ingest.chunker import chunk_sections, split_text


def words(text: str) -> int:
    """Stand-in tokenizer: one token per word. Keeps the tests fast and readable."""
    return len(text.split())


SENTENCES = " ".join(f"Sentence number {i} talks about topic {i}." for i in range(40))
# 6 paragraphs of 30 words each ("Para 0 word0" counts as 3 words).
PARAGRAPHS = "\n\n".join(" ".join(f"Para {p} word{w}" for w in range(10)) for p in range(6))


def test_every_chunk_fits_the_size_limit():
    for start, end in split_text(SENTENCES, words, size=30, overlap=5):
        assert words(SENTENCES[start:end]) <= 30


def test_chunks_cover_the_whole_text_in_order():
    spans = split_text(SENTENCES, words, size=30, overlap=5)
    assert spans[0][0] == 0
    assert spans[-1][1] == len(SENTENCES)
    for (s1, e1), (s2, e2) in pairwise(spans):
        assert s1 < s2  # always moving forward
        assert s2 <= e1  # no gaps: next chunk starts before the previous one ends


def test_overlap_repeats_text_between_neighbours():
    spans = split_text(SENTENCES, words, size=30, overlap=10)
    assert all(s2 < e1 for (_, e1), (s2, _) in pairwise(spans))


def test_no_overlap_when_disabled():
    spans = split_text(SENTENCES, words, size=30, overlap=0)
    assert all(s2 == e1 for (_, e1), (s2, _) in pairwise(spans))


def test_prefers_paragraph_boundaries():
    # Each paragraph has 30 words, so with room for 35 every chunk should be one paragraph.
    spans = split_text(PARAGRAPHS, words, size=35, overlap=0)
    texts = [PARAGRAPHS[s:e].strip() for s, e in spans]
    assert texts == [p.strip() for p in PARAGRAPHS.split("\n\n")]


def test_text_without_separators_is_still_split():
    blob = "x" * 5000
    spans = split_text(blob, lambda t: len(t) // 4, size=100, overlap=0)
    assert len(spans) > 1
    assert "".join(blob[s:e] for s, e in spans) == blob


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        split_text(SENTENCES, words, size=10, overlap=10)


def test_chunks_point_back_to_their_exact_source_text():
    section = Section("doc", "Doc", "doc.md", SENTENCES, section="Intro")
    chunks = chunk_sections([section], words, size=30, overlap=5, min_tokens=1)
    for c in chunks:
        assert section.text[c.start_char : c.end_char] == c.text
        assert c.section == "Intro"
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunk_ids_are_stable_and_unique():
    section = Section("doc", "Doc", "doc.md", SENTENCES)
    first = chunk_sections([section], words, size=30, overlap=5, min_tokens=1)
    second = chunk_sections([section], words, size=30, overlap=5, min_tokens=1)
    assert [c.id for c in first] == [c.id for c in second]
    assert len({c.id for c in first}) == len(first)


def test_tiny_chunks_are_dropped():
    section = Section("doc", "Doc", "doc.pdf", "12", page=3)
    assert chunk_sections([section], words, size=30, overlap=5) == []
