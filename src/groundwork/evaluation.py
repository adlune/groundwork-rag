"""Measuring the RAG system against eval/questions.yaml.

Two separate questions get two separate measurements:

1. Retrieval quality (answerable questions): did search put a relevant chunk in the
   top results? hit@1, hit@k and MRR (mean reciprocal rank: 1 for first place, 1/2 for
   second, ... 0 if missing).

2. The gate (answerable vs out-of-scope questions): does the app answer the questions
   it should and refuse the ones it should? Every threshold value gives a number of
   false refusals (answerable but refused) and false accepts (off-topic but answered).
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from groundwork.documents import Chunk
from groundwork.retriever import Hit, passes_gate


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    type: str  # answerable | out_of_scope | unanswerable
    question: str
    style: str | None = None  # direct | paraphrase | short
    evidence: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()


def load_questions(path: Path) -> list[EvalQuestion]:
    entries = yaml.safe_load(path.read_text(encoding="utf-8"))["questions"]
    return [
        EvalQuestion(
            id=e["id"],
            type=e["type"],
            question=e["question"],
            style=e.get("style"),
            evidence=tuple(e.get("evidence", ())),
            sources=tuple(e.get("sources", ())),
        )
        for e in entries
    ]


def _phrase_pattern(phrase: str) -> re.Pattern:
    """Whole-word match: "DAPT" must not match inside "adapter".

    A space in the phrase matches any whitespace, because chunk text keeps the
    document's line breaks ("Direct\\nPreference Optimization").
    """
    left = r"\b" if phrase[:1].isalnum() else ""
    right = r"\b" if phrase[-1:].isalnum() else ""
    body = r"\s+".join(re.escape(word) for word in phrase.split())
    return re.compile(left + body + right, re.IGNORECASE)


def is_relevant(chunk: Chunk, q: EvalQuestion) -> bool:
    if q.sources and chunk.source_id not in q.sources:
        return False
    return any(_phrase_pattern(p).search(chunk.text) for p in q.evidence)


def first_relevant_rank(hits: list[Hit], q: EvalQuestion) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if is_relevant(hit.chunk, q):
            return rank
    return None


@dataclass
class Run:
    """One question's retrieval results under one configuration."""

    question: EvalQuestion
    hits: list[Hit] = field(default_factory=list)

    @property
    def best_dense(self) -> float:
        return max((h.score for h in self.hits), default=0.0)

    @property
    def best_keyword(self) -> float:
        return max((h.keyword or 0.0 for h in self.hits), default=0.0)

    def accepted(self, min_similarity: float, min_keyword: float) -> bool:
        return any(passes_gate(h, min_similarity, min_keyword) for h in self.hits)


def retrieval_metrics(runs: list[Run]) -> dict[str, dict[str, float]]:
    """hit@1, hit@k and MRR per style (and overall) for answerable questions."""
    groups: dict[str, list[Run]] = {}
    for run in runs:
        if run.question.type == "answerable":
            groups.setdefault(run.question.style or "other", []).append(run)
            groups.setdefault("all", []).append(run)
    out = {}
    for name, items in groups.items():
        ranks = [first_relevant_rank(r.hits, r.question) for r in items]
        out[name] = {
            "n": len(items),
            "hit@1": sum(r == 1 for r in ranks) / len(items),
            "hit@k": sum(r is not None for r in ranks) / len(items),
            "mrr": sum(1 / r for r in ranks if r) / len(items),
        }
    return out


def gate_errors(runs: list[Run], min_similarity: float, min_keyword: float) -> tuple[list[str], list[str]]:
    """(false refusals, false accepts) as lists of question ids.

    Unanswerable questions are on topic, so passing the gate is fine for them; their
    test is whether the model admits it doesn't know (see the generation evaluation).
    """
    false_refusals = [
        r.question.id
        for r in runs
        if r.question.type == "answerable" and not r.accepted(min_similarity, min_keyword)
    ]
    false_accepts = [
        r.question.id
        for r in runs
        if r.question.type == "out_of_scope" and r.accepted(min_similarity, min_keyword)
    ]
    return false_refusals, false_accepts


def best_interval_midpoint(values: list[float], errors: list[int]) -> tuple[float, int]:
    """Middle of the widest run of values with the fewest errors.

    Taking the middle instead of the edge leaves a safety margin on both sides, so a
    slightly different score on a new question doesn't flip the decision.
    """
    fewest = min(errors)
    best: tuple[int, int] = (0, 0)
    start = None
    for i, e in enumerate([*errors, fewest + 1]):  # sentinel closes the last run
        if e == fewest and start is None:
            start = i
        elif e != fewest and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    lo, hi = best[0], best[1] - 1
    return round((values[lo] + values[hi]) / 2, 3), fewest


NOT_FOUND = re.compile(
    r"\b(do|does|did)(?: not|n't) (?:specify|mention|contain|provide|include|say|state|describe)"
    r"|\bnot (?:specified|mentioned|stated|provided|included)\b"
    r"|\bno (?:information|mention|details?)\b"
    r"|enough information",
    re.IGNORECASE,
)


def admits_not_found(answer: str) -> bool:
    """Did the model say the documents don't contain the answer (in any wording)?"""
    return bool(NOT_FOUND.search(answer))
