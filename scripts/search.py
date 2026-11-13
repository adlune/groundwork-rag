"""Search from the command line: no LLM, just the retrieval step (hybrid by default).

    uv run python scripts/search.py "How does LoRA reduce trainable parameters?"
    uv run python scripts/search.py "Can attackers poison a chatbot's documents?" -k 3
"""

import argparse
import sys

from groundwork.config import get_settings
from groundwork.retriever import get_retriever

sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=get_settings().top_k, help="number of results")
    args = parser.parse_args()

    hits = get_retriever().search(args.query, args.k)
    print(f'\nTop {len(hits)} chunks for: "{args.query}"\n')
    for rank, hit in enumerate(hits, 1):
        c = hit.chunk
        snippet = " ".join(c.text.split())[:220]
        kw = f" | keywords {hit.keyword:.2f}" if hit.keyword is not None else ""
        print(f"{rank}. meaning {hit.score:.3f}{kw} | {c.source_id} | {c.location}")
        print(f"   {snippet}...\n")


if __name__ == "__main__":
    main()
