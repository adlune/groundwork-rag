"""Evaluate retrieval and the refusal gate on eval/questions.yaml.

    uv run python scripts/evaluate.py              # retrieval + gate (no LLM, ~1 min)
    uv run python scripts/evaluate.py --generate   # also run the full pipeline with Ollama

Compares three retrieval configurations, calibrates the gate thresholds, and writes
everything to eval/results.json.
"""

import argparse
import json
import sys
import time

from groundwork.config import PROJECT_ROOT, get_settings
from groundwork.embeddings import SentenceTransformerEmbedder, get_embedder
from groundwork.evaluation import (
    Run,
    admits_not_found,
    best_interval_midpoint,
    first_relevant_rank,
    gate_errors,
    is_relevant,
    load_questions,
    retrieval_metrics,
)
from groundwork.keyword import KeywordIndex
from groundwork.llm import get_llm
from groundwork.rag import RagPipeline
from groundwork.retriever import Retriever, open_store

sys.stdout.reconfigure(encoding="utf-8")

QUESTIONS = PROJECT_ROOT / "eval" / "questions.yaml"
RESULTS = PROJECT_ROOT / "eval" / "results.json"
OFF = 9.0  # a keyword threshold no score can reach: keyword rescue disabled


def frange(lo: float, hi: float, step: float) -> list[float]:
    return [round(lo + i * step, 3) for i in range(round((hi - lo) / step) + 1)]


def calibrate_similarity(runs: list[Run]) -> tuple[float, int]:
    taus = frange(0.40, 0.80, 0.005)
    errors = [sum(map(len, gate_errors(runs, t, OFF))) for t in taus]
    return best_interval_midpoint(taus, errors)


def calibrate_keyword(runs: list[Run], tau: float) -> tuple[float, int]:
    betas = frange(0.30, 1.00, 0.01)
    errors = [sum(map(len, gate_errors(runs, tau, b))) for b in betas]
    return best_interval_midpoint(betas, errors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--generate", action="store_true", help="also run the LLM on every question")
    args = parser.parse_args()

    settings = get_settings()
    k = settings.top_k
    store = open_store(settings)
    questions = load_questions(QUESTIONS)

    # 0. Every answerable question must have at least one relevant chunk in the corpus.
    chunks = store.chunks()
    missing = [q.id for q in questions if q.type == "answerable" and not any(is_relevant(c, q) for c in chunks)]
    if missing:
        print(f"Error: no chunk in the index matches the evidence for: {missing}")
        return 1
    counts = {t: sum(q.type == t for q in questions) for t in ("answerable", "out_of_scope", "unanswerable")}
    print(f"{len(questions)} questions: {counts}\n")

    # 1. Retrieval under three configurations.
    embedder = get_embedder(settings)
    plain = SentenceTransformerEmbedder(settings.embedding_model, query_instruction="")
    plain._model = embedder.model  # same weights, only the query prefix differs
    index = KeywordIndex(store.texts())
    configs = {
        "dense, no query instruction": Retriever(plain, store),
        "dense": Retriever(embedder, store),
        "hybrid (dense + BM25)": Retriever(embedder, store, index, settings.candidates, settings.rrf_k),
    }
    runs = {name: [Run(q, r.search(q.question, k)) for q in questions] for name, r in configs.items()}

    print(f"## Retrieval quality (answerable questions, top {k})\n")
    print("| configuration | questions | n | hit@1 | hit@5 | MRR |")
    print("|---|---|---|---|---|---|")
    metrics = {}
    for name, rs in runs.items():
        metrics[name] = retrieval_metrics(rs)
        for style in ("direct", "paraphrase", "short", "all"):
            m = metrics[name][style]
            print(f"| {name} | {style} | {m['n']} | {m['hit@1']:.2f} | {m['hit@k']:.2f} | {m['mrr']:.2f} |")

    # 2. Gate calibration.
    print("\n## Refusal gate\n")
    print("| configuration | threshold(s) | false refusals | false accepts |")
    print("|---|---|---|---|")
    gate = {}
    for name in ("dense, no query instruction", "dense"):
        tau, _ = calibrate_similarity(runs[name])
        fr, fa = gate_errors(runs[name], tau, OFF)
        gate[name] = {"min_similarity": tau, "false_refusals": fr, "false_accepts": fa}
        print(f"| {name} | similarity ≥ {tau} | {len(fr)} {fr} | {len(fa)} {fa} |")
    hybrid = runs["hybrid (dense + BM25)"]
    tau, _ = calibrate_similarity(hybrid)
    beta, _ = calibrate_keyword(hybrid, tau)
    fr, fa = gate_errors(hybrid, tau, beta)
    gate["hybrid (dense + BM25)"] = {"min_similarity": tau, "min_keyword": beta, "false_refusals": fr, "false_accepts": fa}
    print(f"| hybrid (dense + BM25) | similarity ≥ {tau} or keyword ≥ {beta} | {len(fr)} {fr} | {len(fa)} {fa} |")

    print("\n## Per question (hybrid)\n")
    print("| id | type | best similarity | best keyword | first relevant rank | answered? |")
    print("|---|---|---|---|---|---|")
    per_question = []
    for run in hybrid:
        q = run.question
        rank = first_relevant_rank(run.hits, q) if q.type == "answerable" else None
        accepted = run.accepted(tau, beta)
        per_question.append({
            "id": q.id, "type": q.type, "style": q.style, "question": q.question,
            "best_similarity": round(run.best_dense, 3), "best_keyword": round(run.best_keyword, 3),
            "first_relevant_rank": rank, "passes_gate": accepted,
        })
        print(f"| {q.id} | {q.type} | {run.best_dense:.3f} | {run.best_keyword:.2f} | {rank or '-'} | {'yes' if accepted else 'no'} |")

    results = {"top_k": k, "retrieval": metrics, "gate": gate, "per_question": per_question}

    # 3. Optional: the full pipeline with the real model.
    if args.generate:
        print("\n## End-to-end with the LLM\n")
        pipeline = RagPipeline(configs["hybrid (dense + BM25)"], get_llm(settings), k, tau, beta)
        pipeline.warmup()
        generated = []
        t0 = time.perf_counter()
        for q in questions:
            r = pipeline.ask(q.question)
            cited_relevant = any(is_relevant(s.hit.chunk, q) for s in r.sources) if q.type == "answerable" else None
            generated.append({
                "id": q.id, "type": q.type, "answer": r.answer, "refused": r.refused,
                "refusal_reason": r.refusal_reason, "citations": [s.number for s in r.sources],
                "cited_relevant": cited_relevant,
                "admits_not_found": r.refused or admits_not_found(r.answer),
            })
            print(f"- {q.id}: {'refused' if r.refused else 'answered'}, {len(r.sources)} citation(s)")
        ans = [g for g in generated if g["type"] == "answerable"]
        una = [g for g in generated if g["type"] == "unanswerable"]
        oos = [g for g in generated if g["type"] == "out_of_scope"]
        summary = {
            "answerable_answered": sum(not g["refused"] for g in ans) / len(ans),
            "answerable_with_citation": sum(bool(g["citations"]) for g in ans) / len(ans),
            "answerable_cites_relevant_chunk": sum(bool(g["cited_relevant"]) for g in ans) / len(ans),
            "unanswerable_admits_not_found": sum(g["admits_not_found"] for g in una) / len(una),
            "out_of_scope_refused": sum(g["refused"] for g in oos) / len(oos),
            "minutes": round((time.perf_counter() - t0) / 60, 1),
        }
        print("\n" + "\n".join(f"- {k_}: {v:.2f}" if isinstance(v, float) and v <= 1 else f"- {k_}: {v}" for k_, v in summary.items()))
        results["generation"] = {"summary": summary, "answers": generated}

    RESULTS.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {RESULTS.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
