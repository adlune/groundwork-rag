"""Turn files into clean Sections of text.

PDF text extraction is the messiest step of any RAG pipeline. A PDF stores glyphs at
positions on a page, not paragraphs, so the extracted text needs cleaning:
  - ligatures ("ﬁ" as a single character) are normalized to plain letters,
  - words hyphenated across lines are joined back,
  - layout line breaks are removed, keeping breaks only where a sentence ends,
  - headers and footers that repeat on most pages are dropped.
"""

import logging
import re
import unicodedata
from collections import Counter

from pypdf import PdfReader

from groundwork.config import PROJECT_ROOT
from groundwork.documents import Section
from groundwork.sources import Source

logging.getLogger("pypdf").setLevel(logging.ERROR)  # malformed-xref warnings are noise

HEADING = re.compile(r"^(#{1,6})\s+(.*)")
SENTENCE_END = (".", "!", "?", ":")
# Table-of-contents lines: "Prevention and Mitigation Strategies . . . . . 36"
TOC_LINE = re.compile(r"(?:\.\s*){4,}\d+\s*$")
# The vertical stamp arXiv prints on the first page of every paper.
ARXIV_STAMP = re.compile(r"arXiv:\d{4}\.\d{4,5}v\d+\s*\[[^\]]+\]\s*\d{1,2}\s+\w{3}\s+\d{4}")


def load_source(src: Source) -> list[Section]:
    suffix = src.file.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(src)
    if suffix in (".md", ".markdown"):
        return load_markdown(src)
    if suffix == ".txt":
        return load_text(src)
    raise ValueError(f"Unsupported file type: {src.file}")


def _rel(src: Source) -> str:
    """Path shown in citations: relative to the project when possible."""
    try:
        return src.file.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return src.file.as_posix()


# --- PDF ---------------------------------------------------------------------


def _line_key(line: str) -> str:
    """Compare lines ignoring digits, so 'Page 3' and 'Page 4' count as the same line."""
    return re.sub(r"\d+", "", line).strip().lower()


def _repeated_lines(pages: list[str], min_share: float = 0.5) -> set[str]:
    """Lines that appear on at least half of the pages are headers or footers."""
    if len(pages) < 3:
        return set()
    counts: Counter[str] = Counter()
    for page in pages:
        counts.update({_line_key(line) for line in page.splitlines()})
    threshold = max(3, len(pages) * min_share)
    return {key for key, n in counts.items() if key and n >= threshold}


def clean_pdf_text(raw: str, boilerplate: set[str] = frozenset()) -> str:
    text = unicodedata.normalize("NFKC", raw)  # "ﬁ" -> "fi", full-width chars -> ASCII
    text = ARXIV_STAMP.sub("", text)
    lines = [line.strip() for line in text.splitlines()]
    # Drop empty lines, bare page numbers (empty key), repeated headers/footers and
    # table-of-contents entries.
    lines = [
        line
        for line in lines
        if line
        and _line_key(line)
        and _line_key(line) not in boilerplate
        and not TOC_LINE.search(line)
    ]

    out: list[str] = []
    for line in lines:
        if out and out[-1].endswith("-") and line[:1].islower():
            out[-1] = out[-1][:-1] + line  # "av-" + "eraged" -> "averaged"
        elif out and not out[-1].endswith(SENTENCE_END):
            out[-1] += " " + line  # same sentence continues on the next line
        else:
            out.append(line)
    return "\n".join(out)


CITATION = re.compile(
    r"In Proceedings|arXiv preprint|Advances in Neural|URL https?://"
    r"|\. (?:19|20)\d{2}[a-z]?\. "  # ACL style: "Author. 2019. Title."
    r"|\bIn (?:ACL|EMNLP|NAACL|EACL|ICLR|ICML|NeurIPS|TACL)\b",
    re.IGNORECASE,
)


def looks_like_references(text: str, min_citations: int = 2) -> bool:
    """True for chunks that are mostly a bibliography.

    Reference lists are full of paper titles, so they match many questions by keyword
    while containing no answers. Prose that cites a paper or two stays below the limit.
    """
    return len(CITATION.findall(text)) >= min_citations


WORD = re.compile(r"^[A-Za-z][A-Za-z'-]*[A-Za-z]$")


def word_ratio(text: str) -> float:
    """Share of whitespace-separated tokens that are ordinary words."""
    tokens = text.split()
    words = sum(1 for t in tokens if WORD.match(t.strip(".,;:()[]\"“”‘’!?*")))
    return words / max(len(tokens), 1)


def looks_like_noise(text: str, min_word_ratio: float = 0.4) -> bool:
    """True for result tables flattened into numbers and text from broken font encodings.

    Such chunks carry no meaning, so their vectors sit in a "neutral" region of the
    embedding space where unrelated questions also land. Prose is ~90% words; a chunk
    under 40% is almost all numbers or garbage. Formula-heavy prose scores ~45% and stays.
    """
    return word_ratio(text) < min_word_ratio


def load_pdf(src: Source) -> list[Section]:
    reader = PdfReader(src.file)
    raw_pages = [unicodedata.normalize("NFKC", p.extract_text() or "") for p in reader.pages]
    boilerplate = _repeated_lines(raw_pages)
    title = src.title or (reader.metadata.title if reader.metadata else None) or src.file.stem

    sections = []
    for number, raw in enumerate(raw_pages, start=1):
        text = clean_pdf_text(raw, boilerplate)
        if text:
            sections.append(Section(src.id, title, _rel(src), text, page=number))
    return sections


# --- Markdown ----------------------------------------------------------------


def load_markdown(src: Source) -> list[Section]:
    """One Section per heading, so chunks can be cited as 'document > section'."""
    lines = src.file.read_text(encoding="utf-8").splitlines()
    title = src.title or next(
        (m.group(2).strip() for line in lines if (m := HEADING.match(line)) and m.group(1) == "#"),
        src.file.stem,
    )

    sections: list[Section] = []
    heading: str | None = None
    buffer: list[str] = []
    in_code = False

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            sections.append(Section(src.id, title, _rel(src), text, section=heading))

    for line in lines:
        if line.startswith("```"):
            in_code = not in_code
        match = None if in_code else HEADING.match(line)
        if match:
            flush()
            heading, buffer = match.group(2).strip(), [line]
        elif not in_code and line.lstrip().startswith("!["):
            continue  # image embeds carry no text; their captions are separate lines
        else:
            buffer.append(line)
    flush()
    return sections


# --- Plain text --------------------------------------------------------------


def load_text(src: Source) -> list[Section]:
    text = src.file.read_text(encoding="utf-8").strip()
    return [Section(src.id, src.title or src.file.stem, _rel(src), text)] if text else []
