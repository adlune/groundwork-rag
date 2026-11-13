"""Ask a question from the command line: the full RAG flow with sources and scores.

    uv run python scripts/ask.py "How does LoRA reduce the number of trainable parameters?"
    uv run python scripts/ask.py "What is the capital of Australia?"
"""

import argparse
import sys

from groundwork.llm import LLMError
from groundwork.rag import get_pipeline

sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    args = parser.parse_args()

    print("Loading models...")
    pipeline = get_pipeline()
    pipeline.warmup()  # one-time cost, kept out of the timings below
    try:
        result = pipeline.ask(args.question)
    except LLMError as exc:
        print(f"Error: {exc}")
        return 1

    print(f"\nQ: {result.question}\n")
    print(result.answer)
    if result.refused:
        reason = {
            "low_similarity": "no chunk reached the similarity threshold, the LLM was not called",
            "no_answer_in_sources": "the LLM found no answer in the retrieved chunks",
        }[result.refusal_reason]
        print(f"\n(refused: {reason})")

    if result.sources:
        print("\nSources:")
        for s in result.sources:
            c = s.hit.chunk
            print(f"  [{s.number}] {c.title}, {c.location}  (score {s.hit.score:.3f})")

    print(
        f"\nRetrieved (used if meaning >= {result.min_similarity} or keywords >= {result.min_keyword}):"
    )
    for hit in result.retrieved:
        mark = "used" if hit in result.context else "below thresholds"
        kw = f"{hit.keyword:.2f}" if hit.keyword is not None else "  - "
        print(f"  meaning {hit.score:.3f}  keywords {kw}  {hit.chunk.source_id:24} {hit.chunk.location:22} {mark}")

    t = result.timings
    gen = f", generation {t['generation_ms'] / 1000:.1f}s" if "generation_ms" in t else ""
    print(f"\nModel: {result.model or 'not called'} | retrieval {t['retrieval_ms']:.0f}ms{gen}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
