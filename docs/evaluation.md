# Evaluation

This page measures how well the app finds the right passages, how well it decides when to answer or refuse, and what the final answers look like. Every number here comes from `scripts/evaluate.py` and is saved in [eval/results.json](../eval/results.json).

## The question set

[eval/questions.yaml](../eval/questions.yaml) has 45 questions in three groups:

| Group | Count | What the app should do |
|---|---|---|
| Answerable | 28 | Find a passage with the answer and answer from it |
| Off-topic | 12 | Refuse. Three are close to the domain on purpose ("How do I install Python on Windows?") |
| On topic, not answerable | 5 | Admit the documents don't say, instead of guessing ("What GPUs were used to train BloombergGPT?") |

The answerable questions come in three styles: 14 **direct** questions, 8 **paraphrased** ones that share few words with the text, and 6 **short** ones that are mostly a single rare term ("What is QLoRA?").

A retrieved passage counts as relevant if it contains one of the question's evidence phrases, for example "RAG-Token" or "dark knowledge". Matching is by whole word, so "DAPT" doesn't match inside "adapter".

## Retrieval

Three configurations were compared, each returning the top 5 passages:

| Configuration | hit@1 | hit@5 | MRR | Short questions, hit@5 |
|---|---|---|---|---|
| Embeddings, plain question | 0.43 | 0.71 | 0.55 | 0.50 |
| Embeddings with BGE's query instruction | 0.54 | 0.71 | 0.60 | 0.50 |
| Hybrid: embeddings + BM25 keyword search | 0.54 | **0.79** | **0.65** | **1.00** |

*hit@k: share of questions with a relevant passage in the top k. MRR: average of 1/rank of the first relevant passage.*

The query instruction BGE recommends for search queries helps on its own. Keyword search fixes the short questions: an embedding of "What is QLoRA?" has almost nothing to work with, but BM25 matches the exact rare word. The trade-off is that keyword matches sometimes push a good passage out of the top 5 for direct questions (hit@5 for direct questions drops from 0.86 to 0.79).

## The refusal gate

Before calling the model, the app checks whether any retrieved passage is relevant enough. Each threshold was chosen as the middle of the range with the fewest mistakes, to leave a margin on both sides.

| Configuration | Gate | Wrongly refused | Wrongly answered |
|---|---|---|---|
| Embeddings, plain question | similarity ≥ 0.545 | 0 | 4 off-topic |
| Embeddings with query instruction | similarity ≥ 0.65 | 6 (all the short questions) | 0 |
| Hybrid | similarity ≥ 0.645 **or** keywords ≥ 0.70 | 0 | 1 |

With embeddings alone, no single threshold works: short questions score as low as off-topic ones (0.52 for "What is QLoRA?" and for "Who won the 2018 World Cup?"). Adding the keyword score as a second way in solves it.

![Every evaluation question plotted by its best similarity score and best keyword score, with the accepted area shaded](figures/gate-scores.svg)

The one off-topic question that gets through is "What is the capital of Australia?". Its words appear in the documents: the RAG research note uses exactly that sentence as an example of a refusal. A keyword match isn't the same as relevance, which is also how keyword-stuffed documents can be used to attack a RAG system. In this case the model, which is the second layer of protection, answered that the documents don't contain the capital of Australia.

## End to end with the model

The full pipeline ran on all 45 questions with qwen3.5:4b through Ollama (1.6 minutes in total).

| Measure | Result |
|---|---|
| Answerable questions answered | 27 / 28 |
| Answers with at least one citation | 27 / 28 |
| Answers citing a passage that contains the evidence | 22 / 28 |
| Unanswerable questions where the model said the documents don't say | 5 / 5 |
| Off-topic questions refused | 11 / 12 |

Every answer was also read by hand. None made up facts that weren't in the retrieved passages. The six answerable questions that missed the evidence fall into three groups:

- **Retrieval missed, the model said so (3).** For "What is LLM08?" search returned OWASP's introduction pages, and the model replied that it didn't have the information instead of guessing.
- **True but incomplete or from a neighbouring section (2).** The answer about OWASP's prompt injection advice quotes the mitigations for a different entry (improper output handling).
- **Correct, but the evidence phrases were too strict to credit it (1).**

Three answers restate their sources a little loosely without inventing anything new. The HNSW answer blends the index with the embedding model, one answer presents chain-of-thought prompting as a fix for confident wrong answers when the source only mentions it, and one says adversarial examples cost under US$0.20 to *generate*, where the source says fine-tuning on them cost that. Small models paraphrase imprecisely; the citations make these slips easy to check.

## Limitations

- 45 questions is a small set, and the thresholds were chosen on the same set they are measured on, so the gate numbers are optimistic. New questions will be closer to the edges.
- The similarity margin is narrow: the closest off-topic question scores 0.617 ("What is Kubernetes used for?") and the lowest answerable one that relies on similarity scores 0.670.
- Results were only stable with nothing else using the vector database. A run made while the web app had the same database open gave different retrieval numbers, so evaluations should run on their own.
- Tables in the PDFs are flattened into rows of numbers, so questions about specific results in tables are answered poorly.

## Reproduce

```bash
uv run python scripts/evaluate.py              # retrieval and gate, no model (about 1 minute)
uv run python scripts/evaluate.py --generate   # also run every question through Ollama
uv run python scripts/plot_gate.py             # redraw the figure from eval/results.json
```
