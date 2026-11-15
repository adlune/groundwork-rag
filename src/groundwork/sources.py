"""Reads the corpus list in data/sources.yaml and downloads its pinned files."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml

from groundwork.config import PROJECT_ROOT, get_settings


@dataclass(frozen=True)
class Source:
    id: str
    file: Path  # absolute path on disk
    title: str | None = None  # None means "read it from the document"
    url: str | None = None  # only for documents that are downloaded
    license: str | None = None
    sha256: str | None = None  # hash of the exact file the results were produced with


class DownloadError(RuntimeError):
    pass


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(src: Source, client: httpx.Client) -> bytes:
    """Fetch a source's file and check it is the pinned one, or raise DownloadError."""
    try:
        resp = client.get(src.url, follow_redirects=True, timeout=60)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise DownloadError(str(exc)) from exc
    # A PDF always starts with "%PDF". Anything else (an HTML error page, a
    # captcha) should not end up in the corpus.
    if not resp.content.startswith(b"%PDF"):
        raise DownloadError("response is not a PDF")
    if src.sha256 and sha256_of(resp.content) != src.sha256:
        raise DownloadError(
            f"sha256 is {sha256_of(resp.content)}, expected {src.sha256}. "
            "The file changed at the source, so results would no longer be comparable."
        )
    return resp.content


def load_sources(path: Path | None = None) -> list[Source]:
    path = path or get_settings().sources_file
    entries = yaml.safe_load(path.read_text(encoding="utf-8"))["documents"]
    return [
        Source(
            id=e["id"],
            file=PROJECT_ROOT / e["file"],
            title=e.get("title"),
            url=e.get("url"),
            license=e.get("license"),
            sha256=e.get("sha256"),
        )
        for e in entries
    ]
