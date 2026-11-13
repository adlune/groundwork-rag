"""The full RAG flow: retrieve -> relevance gate -> prompt -> generate -> cite.

The app can say "I don't know" at two points:

  Layer 1, the retrieval gate. If no chunk reaches MIN_SIMILARITY, the documents don't
  cover the question. The app answers right away and the LLM is never called: cheaper,
  faster, and the model gets no chance to make something up.

  Layer 2, the model. Chunks can be similar enough to pass the gate (same topic) without
  containing the answer. The prompt tells the model to reply with a fixed sentence in
  that case, and the app detects it.
"""

import re
import time
from dataclasses import dataclass, field

from groundwork.config import Settings, get_settings
from groundwork.llm import LLMProvider, get_llm
from groundwork.prompts import NO_ANSWER, build_prompt
from groundwork.retriever import Hit, Retriever, get_retriever, passes_gate

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")  # [1], [2][3], [1, 2]


@dataclass(frozen=True)
class Source:
    number: int  # the [n] used in the answer
    hit: Hit


@dataclass
class RagAnswer:
    question: str
    answer: str
    refused: bool
    refusal_reason: str | None  # "low_similarity" (layer 1) or "no_answer_in_sources" (layer 2)
    sources: list[Source]  # chunks the answer actually cites
    context: list[Hit]  # chunks sent to the LLM (passed the gate)
    retrieved: list[Hit]  # everything semantic search returned, with scores
    min_similarity: float
    min_keyword: float
    model: str | None
    timings: dict[str, float] = field(default_factory=dict)  # milliseconds


def parse_citations(text: str, n_sources: int) -> list[int]:
    """Citation numbers in order of first appearance, ignoring numbers with no source."""
    seen: list[int] = []
    for group in _CITATION.findall(text):
        for part in group.split(","):
            n = int(part)
            if 1 <= n <= n_sources and n not in seen:
                seen.append(n)
    return seen


def is_no_answer(text: str) -> bool:
    """Did the model use the 'not in my documents' sentence? Tolerates small rewording."""
    return "enough information" in text.lower() or NO_ANSWER.lower() in text.lower()


class RagPipeline:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMProvider,
        top_k: int,
        min_similarity: float,
        min_keyword: float = 0.8,
    ):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.min_keyword = min_keyword

    def warmup(self) -> None:
        """Load the embedding model now, so the first question's timings are real.

        Loading PyTorch and the model takes ~15-20 s once per process. Without this, that
        one-time cost shows up as "retrieval time" on the first question.
        """
        self.retriever.embedder.embed_query("warmup")

    def ask(self, question: str) -> RagAnswer:
        question = question.strip()
        if not question:
            raise ValueError("The question is empty.")

        t0 = time.perf_counter()
        retrieved = self.retriever.search(question, self.top_k)
        t1 = time.perf_counter()
        timings = {"retrieval_ms": (t1 - t0) * 1000}

        # Only chunks that pass the gate go to the model: low-scoring neighbours are
        # noise that can distract it.
        context = [h for h in retrieved if passes_gate(h, self.min_similarity, self.min_keyword)]
        common = {
            "question": question,
            "retrieved": retrieved,
            "min_similarity": self.min_similarity,
            "min_keyword": self.min_keyword,
            "timings": timings,
        }

        # Layer 1: nothing relevant enough. Don't call the LLM at all.
        if not context:
            return RagAnswer(
                answer=NO_ANSWER,
                refused=True,
                refusal_reason="low_similarity",
                sources=[],
                context=[],
                model=None,
                **common,
            )

        system, prompt = build_prompt(question, context)
        text = self.llm.generate(system, prompt)
        timings["generation_ms"] = (time.perf_counter() - t1) * 1000

        cited = parse_citations(text, len(context))
        # Layer 2: the model says the sources don't answer it (and cites nothing).
        refused = not cited and is_no_answer(text)
        return RagAnswer(
            answer=NO_ANSWER if refused else text,
            refused=refused,
            refusal_reason="no_answer_in_sources" if refused else None,
            sources=[Source(n, context[n - 1]) for n in cited],
            context=context,
            model=self.llm.name,
            **common,
        )


def get_pipeline(settings: Settings | None = None) -> RagPipeline:
    settings = settings or get_settings()
    return RagPipeline(
        get_retriever(settings),
        get_llm(settings),
        settings.top_k,
        settings.min_similarity,
        settings.min_keyword,
    )
