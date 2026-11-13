"""Keyword search with BM25, the classic ranking function behind most search engines.

BM25 scores a chunk for a query by adding up, for each query word found in the chunk:

    idf(word) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * chunk_length / average_length))

- idf (inverse document frequency): rare words count more. "QLoRA" appears in 4 of 402
  chunks, "model" in most of them, so matching "QLoRA" is much stronger evidence.
- tf (term frequency): more occurrences count more, but with diminishing returns (k1).
- length normalization (b): a word found in a short chunk counts more than in a long one.

Embeddings capture meaning but blur rare exact terms; BM25 is the opposite. Combining
both is called hybrid search.
"""

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")

# Words that carry no topic: question words, articles, auxiliaries, pronouns.
# (Written as one string and split: easier to read than 150 quoted words.)
STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been before
    being below between both but by can could did do does doing done down during each few for
    from further had has have having he her here hers him his how i if in into is it its itself
    just me more most my no nor not now of off on once only or other our ours out over own same
    she should so some such than that the their theirs them then there these they this those
    through to too under until up very was we were what when where which while who whom why will
    with would you your yours explain tell give show use used using get""".split()  # noqa: SIM905
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


class KeywordIndex:
    def __init__(self, docs: dict[str, str], k1: float = 1.2, b: float = 0.75):
        """docs: chunk id -> chunk text."""
        self.k1, self.b = k1, b
        self.ids = list(docs)
        self.tfs = [Counter(tokenize(docs[i])) for i in self.ids]
        self.lengths = [sum(tf.values()) for tf in self.tfs]
        self.avg_length = sum(self.lengths) / max(len(self.lengths), 1)
        self.n = len(self.ids)
        df: Counter[str] = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        self.df = df

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log((self.n - df + 0.5) / (df + 0.5) + 1)

    def _score(self, terms: list[str], doc: int) -> float:
        tf, length = self.tfs[doc], self.lengths[doc]
        score = 0.0
        for t in terms:
            f = tf.get(t, 0)
            if f:
                norm = self.k1 * (1 - self.b + self.b * length / self.avg_length)
                score += self.idf(t) * f * (self.k1 + 1) / (f + norm)
        return score

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Top-k (chunk id, match strength), strength between 0 and 1.

        Raw BM25 scores have no fixed scale, so they're divided by the sum of the query
        words' idf: roughly "how much of the query's rare-word weight this chunk
        contains". 1.0 means every informative query word appears in the chunk (once,
        at average length); words that appear nowhere in the corpus count as missing.
        """
        terms = list(dict.fromkeys(tokenize(query)))  # unique, in order
        if not terms or not self.n:
            return []
        ideal = sum(self.idf(t) for t in terms)
        scored = [(self._score(terms, i), i) for i in range(self.n)]
        scored = [(s, i) for s, i in scored if s > 0]
        scored.sort(reverse=True)
        return [(self.ids[i], min(s / ideal, 1.0)) for s, i in scored[:k]]
