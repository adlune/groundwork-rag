"""Load and chunk the corpus, then print statistics and sample chunks.

Nothing is embedded or stored. This is for looking at what the chunker produces.

    uv run python scripts/inspect_chunks.py                  # statistics per document
    uv run python scripts/inspect_chunks.py --show 3         # plus 3 random chunks
    uv run python scripts/inspect_chunks.py --source hu2021-lora --show 2
"""

import argparse
import random
import sys
from collections import defaultdict

from groundwork.config import get_settings
from groundwork.ingest.pipeline import build_chunks

sys.stdout.reconfigure(encoding="utf-8")  # the Windows console defaults to cp1252


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", help="only show chunks from this source id")
    parser.add_argument("--show", type=int, default=0, help="number of random chunks to print")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    settings = get_settings()
    chunks = build_chunks(settings)
    if args.source:
        chunks = [c for c in chunks if c.source_id == args.source]

    by_source = defaultdict(list)
    for c in chunks:
        by_source[c.source_id].append(c)

    print(f"chunk size {settings.chunk_size} tokens, overlap {settings.chunk_overlap}\n")
    print(f"{'source':26} {'chunks':>6} {'min':>5} {'avg':>5} {'max':>5}")
    for source_id, items in by_source.items():
        sizes = [c.n_tokens for c in items]
        print(
            f"{source_id:26} {len(items):6} {min(sizes):5} "
            f"{sum(sizes) // len(sizes):5} {max(sizes):5}"
        )
    sizes = [c.n_tokens for c in chunks]
    print(f"{'TOTAL':26} {len(chunks):6} {min(sizes):5} {sum(sizes) // len(sizes):5} {max(sizes):5}")

    for c in random.Random(args.seed).sample(chunks, min(args.show, len(chunks))):
        print(f"\n--- {c.source_id} | {c.location} | chunk {c.index} | {c.n_tokens} tokens")
        print(c.text)


if __name__ == "__main__":
    main()
