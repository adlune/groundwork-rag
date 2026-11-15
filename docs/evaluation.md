# Evaluation

This page measures how well the app finds the right passages, how well it decides when to answer or refuse, and what the final answers look like. Every number here comes from `scripts/evaluate.py` and is saved in [eval/results.json](../eval/results.json). Running the script again gives the same file (see [Reproducibility](#reproducibility)).

## The question set

[eval/questions.yaml](../eval/questions.yaml) has 45 questions in three groups:

| Group | Count | What the app should do |
|---|---|---|
| Answerable | 28 | Find a passage with the answer and answer from it |
| Off-topic | 12 | Refuse. Three are close to the domain on purpose ("How do I install Python on Windows?") |
| On topic, not answerable | 5 | Admit the documents don't say, instead of guessing ("What GPUs were used to train BloombergGPT?") |

The answerable questions come in three styles: 14 **direct** questions, 8 **paraphrased** ones that share few words with the text, and 6 **short** ones that are mostly a single rare term ("What is QLoRA?").

A retrieved passage counts as relevant if it contains one of the question's evidence phrases, for example "RAG-Token Model" or "dark knowledge". Matching is by whole word, so "DAPT" doesn't match inside "adapter". The phrases are specific on purpose: between 1 and 11 of the 402 chunks count as relevant for each question. With looser phrases a retriever that picks chunks at random would already look decent, so the table below shows that random baseline too.

## Retrieval

Three configurations were compared, each returning the top 5 passages:

| Configuration | hit@1 | hit@5 | MRR | Short questions, hit@5 |
|---|---|---|---|---|
| Random chunks (expected value) | | 0.06 | | |
| Embeddings, plain question | 0.39 | 0.71 | 0.53 | 0.50 |
| Embeddings with BGE's query instruction | 0.54 | 0.71 | 0.60 | 0.50 |
| Hybrid: embeddings + BM25 keyword search | 0.54 | **0.79** | **0.65** | **1.00** |

*hit@k: share of the 28 answerable questions with a relevant passage in the top k. MRR: average of 1/rank of the first relevant passage.*

The query instruction BGE recommends for search queries helps on its own. Keyword search fixes the short questions: an embedding of "What is QLoRA?" has almost nothing to work with, but BM25 matches the exact rare word. The trade-off is that keyword matches sometimes push a good passage out of the top 5 for direct questions (hit@5 for direct questions drops from 0.86 to 0.79).

## The refusal gate

Before calling the model, the app checks whether any retrieved passage is relevant enough. The thresholds are chosen from a grid of values. For the hybrid gate both thresholds are searched together, and the chosen pair is the centre of the region with the fewest mistakes, which leaves a margin on every side.

| Configuration | Gate | Wrongly refused (of 28) | Wrongly answered (of 12) |
|---|---|---|---|
| Embeddings, plain question | similarity ≥ 0.545 | 0 | 4 |
| Embeddings with query instruction | similarity ≥ 0.653 | 6 (all the short questions) | 0 |
| Hybrid | similarity ≥ 0.645 **or** keywords ≥ 0.70 | 0 | 0 |

With embeddings alone, no single threshold works: short questions score as low as off-topic ones (0.52 for "What is QLoRA?" and for "Who won the 2018 World Cup?"). Adding the keyword score as a second way in solves it.

![Every evaluation question plotted by its best similarity score and best keyword score, with the accepted area shaded](figures/gate-scores.svg)

Thresholds tuned on a set of questions look better on that set than on new ones. To estimate by how much, the script also runs a split-half cross-validation: calibrate on a random half of the 40 answerable and off-topic questions, count the mistakes on the other half, and repeat 300 times. On the held-out halves the gate wrongly refused 0.2% of answerable questions and wrongly answered 0.7% of off-topic ones. The chosen similarity threshold ranged from 0.615 to 0.663 across splits (5th to 95th percentile) and the keyword threshold from 0.64 to 0.76.

That is still an optimistic estimate, because all 45 questions were written by the same people who built the app. A question that shares rare words with the documents can pass on keywords alone even when it is off-topic. An earlier version of the RAG research note used "What is the capital of Australia?" as an example of a refusal, and that sentence was enough to let the question through the gate (keyword score 1.00). After the note was rewritten its keyword score is 0.45 and it is refused. A keyword match isn't the same as relevance, which is also how keyword-stuffed documents can be used to attack a RAG system.

## End to end with the model

The full pipeline ran on all 45 questions with qwen3.5:4b through Ollama, at temperature 0 with a fixed seed (1.6 minutes in total).

| Measure | Result |
|---|---|
| Answerable questions answered | 27 / 28 |
| Answers with at least one citation | 27 / 28 |
| Answers citing a passage that contains the evidence | 22 / 28 |
| Unanswerable questions where the model said the documents don't say | 5 / 5 |
| Off-topic questions refused | 12 / 12 |

The six answerable questions that missed the evidence fall into three groups:

- **Retrieval missed, the model said the documents don't cover it (3).** For "What is LLM08?" search returned OWASP's introduction pages, and the model replied that it didn't have the information instead of guessing.
- **Retrieval missed, the model answered from neighbouring passages (2).** Asked for the two RAG formulations, it described the RAG paper's retriever and generator instead.
- **Right document, wrong section (1).** The answer about OWASP's prompt injection advice quotes the mitigations for a different entry (improper output handling).

Every answer was also reviewed once with AI assistance. Most stay close to their sources, but not all of them:

- One answer turns a source around. Asked why a company would search its files instead of retraining, it says searching "does not require extra retrieval infrastructure", which the source says about continued pre-training.
- One answer to an uncovered question begins with "No," which reads as an answer to the question rather than as "the documents don't say".
- A few restate their sources loosely. The HNSW answer blends the index with the embedding model, and one answer says using adversarial examples cost under US$0.20, where the source says fine-tuning on them did.

Small models paraphrase imprecisely, and the citations are what make these slips easy to check.

## Limitations

- 45 questions is a small set, and the thresholds were chosen on the same set they are measured on. The cross-validation above estimates the effect, but a separate set of new questions is the real test.
- The similarity margin is narrow: the closest off-topic question scores 0.618 ("What is Kubernetes used for?") and the lowest answerable one that relies on similarity scores 0.670.
- The keyword path of the gate can be passed by off-topic questions that happen to share rare words with the documents.
- The relevance test checks for a phrase, not for an answer. A passage can contain the phrase and still not answer the question.
- Tables in the PDFs are flattened into rows of numbers, so questions about specific results in tables are answered poorly.

## Reproducibility

Two runs of `scripts/evaluate.py --generate` on the same machine produce identical result files, including every generated answer. Three things make that true:

- Search is exact. The question vector is compared with every chunk vector, instead of going through ChromaDB's approximate HNSW index. That index is rebuilt each time the program starts and sometimes missed some of the true nearest chunks, so earlier results moved slightly between runs.
- The model runs at temperature 0 with a fixed seed (`ollama_temperature` and `ollama_seed` in `src/groundwork/config.py`). A different GPU or Ollama version may still change the wording of some answers.
- The corpus and the embedding model are pinned. Every PDF in [data/sources.yaml](../data/sources.yaml) has a versioned URL and a sha256 hash, and `scripts/download_corpus.py` refuses a file that doesn't match.

| Component | Version |
|---|---|
| Embedding model | BAAI/bge-small-en-v1.5, revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` |
| Language model | qwen3.5:4b in Ollama, model ID `d8b0f5e9760c` |
| Papers | arXiv 2005.11401v4, 2106.09685v2, 1503.02531v1, 2310.08560v2, 2004.10964v3 |
| OWASP Top 10 for LLMs 2025 | OWASP repository at commit `17bfc16` |

```bash
uv run python scripts/download_corpus.py       # fetch the pinned PDFs
uv run python scripts/ingest.py                # build the index
uv run python scripts/evaluate.py              # retrieval and gate, no model (about 1 minute)
uv run python scripts/evaluate.py --generate   # also run every question through Ollama
uv run python scripts/plot_gate.py             # redraw the figure from eval/results.json
```
