from pathlib import Path

import httpx
import pytest

from groundwork.config import get_settings
from groundwork.sources import DownloadError, Source, download, load_sources, sha256_of

PDF = b"%PDF-1.5 pinned content"


def serving(body: bytes) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=body)))


def source(sha: str | None) -> Source:
    return Source(id="paper", file=Path("paper.pdf"), url="https://example.org/paper.pdf", sha256=sha)


def test_download_accepts_the_pinned_file():
    assert download(source(sha256_of(PDF)), serving(PDF)) == PDF


def test_download_refuses_a_file_with_a_different_hash():
    with pytest.raises(DownloadError, match="sha256"):
        download(source(sha256_of(PDF)), serving(b"%PDF-1.5 a newer version"))


def test_download_refuses_something_that_is_not_a_pdf():
    with pytest.raises(DownloadError, match="not a PDF"):
        download(source(None), serving(b"<html>captcha</html>"))


def test_every_downloaded_source_is_pinned():
    for src in load_sources(get_settings().sources_file):
        if src.url:
            assert src.sha256 and len(src.sha256) == 64, src.id
            assert "/main/" not in src.url, src.id
