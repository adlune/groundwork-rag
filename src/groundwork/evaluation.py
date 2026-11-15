"""Measuring the RAG system against eval/questions.yaml.

Two separate questions get two separate measurements:

1. Retrieval quality (answerable questions): did search put a relevant chunk in the
   top results? hit@1, hit@k and MRR (mean reciprocal rank: 1 for first place, 1/2 for
   second, ... 0 if missing).

2. The gate (answerable vs out-of-scope questions): does the app answer the questions
   it should and refuse the ones it should? Every threshold value gives a number of
   false refusals (answerable but refused) and false accepts (off-topic but answered).

Thresholds chosen on a set of questions look better on that set than they will on new
questions. split_half_cv estimates the difference: calibrate on a random half, measure on
the other half, repeat.
"""

import math
import random
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


def frange(lo: float, hi: float, step: float) -> list[float]:
    return [round(lo + i * step, 3) for i in range(round((hi - lo) / step) + 1)]


SIMILARITY_GRID = frange(0.40, 0.80, 0.005)
KEYWORD_GRID = frange(0.30, 1.00, 0.01)

# One point per question for the gate: (best similarity, best keyword score, type).
# A question passes the gate if any hit passes, which is the same as its best
# similarity or its best keyword score passing.
GatePoint = tuple[float, float, str]


def gate_points(runs: list[Run]) -> list[GatePoint]:
    return [(r.best_dense, r.best_keyword, r.question.type) for r in runs if r.question.type != "unanswerable"]


def point_errors(points: list[GatePoint], tau: float, beta: float) -> tuple[int, int]:
    """(false refusals, false accepts) for a similarity threshold tau and keyword threshold beta."""
    fr = sum(1 for d, k, t in points if t == "answerable" and not (d >= tau or k >= beta))
    fa = sum(1 for d, k, t in points if t == "out_of_scope" and (d >= tau or k >= beta))
    return fr, fa


def calibrate_gate(points: list[GatePoint]) -> tuple[float, float, int]:
    """Both thresholds together: the centre of the region of the grid with the fewest errors.

    Tuning the similarity threshold first and the keyword one second (the earlier method)
    can lock in a similarity value that only made sense without the keyword check.
    Searching both at once and taking the centre of the best region leaves a margin in
    both directions.
    """
    grid = [(sum(point_errors(points, t, b)), t, b) for t in SIMILARITY_GRID for b in KEYWORD_GRID]
    fewest = min(e for e, _, _ in grid)
    region = [(t, b) for e, t, b in grid if e == fewest]
    tau = round(sum(t for t, _ in region) / len(region), 3)
    beta = round(sum(b for _, b in region) / len(region), 2)
    return tau, beta, fewest


def split_half_cv(points: list[GatePoint], splits: int = 300, seed: int = 0) -> dict:
    """Calibrate on a random half, count errors on the other half, many times over.

    Returns the average held-out false refusal and false accept rates and the spread of
    the chosen thresholds, which shows how much they depend on the exact questions.
    """
    rng = random.Random(seed)
    points = list(points)
    fr_rates, fa_rates, taus, betas = [], [], [], []
    for _ in range(splits):
        rng.shuffle(points)
        train, test = points[: len(points) // 2], points[len(points) // 2 :]
        n_ans = sum(t == "answerable" for _, _, t in test)
        n_oos = sum(t == "out_of_scope" for _, _, t in test)
        if not n_ans or not n_oos or len({t for _, _, t in train}) < 2:
            continue
        tau, beta, _ = calibrate_gate(train)
        fr, fa = point_errors(test, tau, beta)
        fr_rates.append(fr / n_ans)
        fa_rates.append(fa / n_oos)
        taus.append(tau)
        betas.append(beta)

    def spread(values: list[float]) -> list[float]:
        v = sorted(values)
        return [v[int(0.05 * (len(v) - 1))], v[len(v) // 2], v[int(0.95 * (len(v) - 1))]]

    return {
        "splits": len(fr_rates),
        "heldout_false_refusal_rate": round(sum(fr_rates) / len(fr_rates), 3),
        "heldout_false_accept_rate": round(sum(fa_rates) / len(fa_rates), 3),
        "min_similarity_5_50_95": spread(taus),
        "min_keyword_5_50_95": spread(betas),
    }


def random_hit_at_k(chunks: list[Chunk], questions: list[EvalQuestion], k: int) -> float:
    """Expected hit@k of a retriever that picks k chunks at random.

    For a question with r relevant chunks out of n, the chance that none of k random
    picks is relevant is C(n-r, k) / C(n, k). If many chunks count as relevant, even a
    random retriever scores well, so this is the floor a real retriever must beat.
    """
    n = len(chunks)
    answerable = [q for q in questions if q.type == "answerable"]
    total = 0.0
    for q in answerable:
        r = sum(is_relevant(c, q) for c in chunks)
        total += 1 - math.comb(n - r, k) / math.comb(n, k)
    return total / len(answerable)


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
