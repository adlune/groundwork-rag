"""Download the PDFs listed in data/sources.yaml into data/raw/.

Files that already exist are skipped, so it is safe to run more than once.

    uv run python scripts/download_corpus.py
"""

import sys

import httpx

from groundwork.config import PROJECT_ROOT
from groundwork.sources import load_sources

HEADERS = {"User-Agent": "groundwork-rag corpus downloader"}


def main() -> int:
    failed = 0
    for src in load_sources():
        if src.url is None:
            continue  # a local file, nothing to download
        rel = src.file.relative_to(PROJECT_ROOT)
        if src.file.exists():
            print(f"skip  {rel} (already downloaded)")
            continue
        try:
            resp = httpx.get(src.url, headers=HEADERS, follow_redirects=True, timeout=60)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            print(f"FAIL  {src.id}: {exc}")
            failed += 1
            continue
        # A PDF always starts with "%PDF". Anything else (an HTML error page, a
        # captcha) should not end up in the corpus.
        if not resp.content.startswith(b"%PDF"):
            print(f"FAIL  {src.id}: response is not a PDF")
            failed += 1
            continue
        src.file.parent.mkdir(parents=True, exist_ok=True)
        src.file.write_bytes(resp.content)
        print(f"ok    {rel} ({len(resp.content) / 1e6:.1f} MB)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
