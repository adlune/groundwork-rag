"""Download the PDFs listed in data/sources.yaml into data/raw/.

Every file is checked against the sha256 pinned in sources.yaml. Files that already
exist are checked too and then skipped, so it is safe to run more than once.

    uv run python scripts/download_corpus.py
"""

import sys

import httpx

from groundwork.config import PROJECT_ROOT
from groundwork.sources import DownloadError, download, load_sources, sha256_of

HEADERS = {"User-Agent": "groundwork-rag corpus downloader"}


def main() -> int:
    failed = 0
    client = httpx.Client(headers=HEADERS)
    for src in load_sources():
        if src.url is None:
            continue  # a local file, nothing to download
        rel = src.file.relative_to(PROJECT_ROOT)
        if src.file.exists():
            if src.sha256 and sha256_of(src.file.read_bytes()) != src.sha256:
                print(f"FAIL  {rel} is on disk but differs from the pinned version. Delete it and run again.")
                failed += 1
            else:
                print(f"skip  {rel} (already downloaded)")
            continue
        try:
            content = download(src, client)
        except DownloadError as exc:
            print(f"FAIL  {src.id}: {exc}")
            failed += 1
            continue
        src.file.parent.mkdir(parents=True, exist_ok=True)
        src.file.write_bytes(content)
        print(f"ok    {rel} ({len(content) / 1e6:.1f} MB)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
