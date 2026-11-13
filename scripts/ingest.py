"""Build or update the vector index from the documents in data/sources.yaml.

Only new or changed chunks are embedded; chunks that no longer exist are removed.

    uv run python scripts/ingest.py             # sync the index with the documents
    uv run python scripts/ingest.py --rebuild   # delete the index and embed everything again
"""

import argparse
import sys
import time

from groundwork.config import get_settings
from groundwork.embeddings import get_embedder
from groundwork.ingest.pipeline import build_chunks, ingest
from groundwork.retriever import open_store
from groundwork.store import StoreMismatchError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rebuild", action="store_true", help="start from an empty index")
    args = parser.parse_args()
    settings = get_settings()

    t0 = time.perf_counter()
    chunks = build_chunks(settings)
    print(f"Chunked the corpus: {len(chunks)} chunks ({time.perf_counter() - t0:.1f}s)")

    try:
        store = open_store(settings)
    except StoreMismatchError as exc:
        if not args.rebuild:
            print(f"Error: {exc}")
            return 1
        store = None
    if args.rebuild:
        if store is None:
            import chromadb

            chromadb.PersistentClient(path=str(settings.chroma_dir)).delete_collection(
                settings.collection
            )
            store = open_store(settings)
        else:
            store.reset()
        print("Index cleared.")

    t1 = time.perf_counter()
    report = ingest(chunks, get_embedder(settings), store)
    print(
        f"Index synced in {time.perf_counter() - t1:.1f}s: "
        f"{report.added} added, {report.removed} removed, {report.unchanged} unchanged, "
        f"{store.count()} chunks stored in {settings.chroma_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
