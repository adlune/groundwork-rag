"""Reads the corpus list in data/sources.yaml."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from groundwork.config import PROJECT_ROOT, get_settings


@dataclass(frozen=True)
class Source:
    id: str
    file: Path  # absolute path on disk
    title: str | None = None  # None means "read it from the document"
    url: str | None = None  # only for documents that are downloaded
    license: str | None = None


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
        )
        for e in entries
    ]
