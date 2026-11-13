from groundwork.ingest.loaders import (
    _repeated_lines,
    clean_pdf_text,
    load_markdown,
    looks_like_references,
)
from groundwork.sources import Source


def test_ligatures_are_normalized():
    assert clean_pdf_text("signiﬁcant efﬁciency.") == "significant efficiency."


def test_hyphenated_words_are_joined():
    assert clean_pdf_text("results av-\neraged over trials.") == "results averaged over trials."


def test_layout_line_breaks_are_removed_but_sentence_breaks_kept():
    raw = "This sentence is\nsplit over two lines.\nThis one starts fresh."
    assert clean_pdf_text(raw) == "This sentence is split over two lines.\nThis one starts fresh."


def test_page_numbers_and_boilerplate_are_dropped():
    bodies = ["Alpha text.", "Beta text.", "Gamma text.", "Delta text.", "Epsilon text."]
    pages = [f"OWASP Top 10 v2.0\n{n}genai.owasp.org\n{b}" for n, b in enumerate(bodies, 1)]
    boilerplate = _repeated_lines(pages)
    assert clean_pdf_text(pages[2] + "\n7", boilerplate) == "Gamma text."


def test_table_of_contents_lines_are_dropped():
    raw = "Contents\nPrevention and Mitigation Strategies .  .  .  .  .  .   36\nReal text."
    assert clean_pdf_text(raw) == "Contents Real text."


def test_arxiv_stamp_is_removed():
    raw = "Applications in traditional OSes interact arXiv:2310.08560v2  [cs.AI]  12 Feb 2024 with memory."
    assert "arXiv" not in clean_pdf_text(raw)


def test_reference_lists_are_detected_but_prose_is_not():
    refs = (
        "Iz Beltagy, Kyle Lo, and Arman Cohan. 2019. SciBERT: A pretrained language model. In EMNLP. "
        "Alec Radford et al. Language models. arXiv preprint arXiv:1234.5678, 2019."
    )
    prose = "LoRA freezes the pretrained weights (Hu et al., 2021) and trains low-rank matrices."
    assert looks_like_references(refs)
    assert not looks_like_references(prose)


def test_markdown_is_split_by_heading(tmp_path):
    md = tmp_path / "note.md"
    md.write_text(
        "# My note\n\nIntro text.\n\n## Part A\n\nAlpha.\n\n![figure](x.svg)\n\nCaption.\n\n"
        "## Part B\n\n```\n# not a heading\n```\n",
        encoding="utf-8",
    )
    sections = load_markdown(Source(id="note", file=md))
    assert [s.section for s in sections] == ["My note", "Part A", "Part B"]
    assert all(s.title == "My note" for s in sections)
    assert "figure" not in sections[1].text and "Caption." in sections[1].text
    assert "# not a heading" in sections[2].text
