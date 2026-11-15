# groundwork-rag

Retrieval Augmented Generation (RAG) app built step by step, plus research on how to give a language model new knowledge.

## Main idea

A language model only knows what it saw during training. RAG gives it access to new information without retraining it. See [RAG](docs/research/04-rag.md) for more detailed info.

![RAG pipeline: indexing turns documents into chunks, vectors and a vector database; querying embeds the question, searches, builds a prompt and generates an answer](docs/figures/rag-pipeline.svg)

If nothing in the documents is relevant enough, the app says it doesn't have the information instead of guessing.

Search is hybrid: passages are ranked both by meaning (embeddings) and by exact words (BM25 keyword search), which keeps short questions like "What is QLoRA?" from being missed.

The pipeline is written without RAG frameworks, so every step (chunking, embedding, search, prompt building) is visible in the code.

## Research

The **docs/research/** folder covers six ways of giving a model new knowledge. Each one explains how it works, whether it changes the model's parameters, its pros and cons, tools and a practical example, plus a short note on its security risks.

1. [Prompting and in-context learning](docs/research/01-prompting.md)
2. [Pre-training and continued pre-training](docs/research/02-pretraining.md)
3. [Fine-tuning](docs/research/03-fine-tuning.md)
4. [Retrieval-Augmented Generation](docs/research/04-rag.md)
5. [Conversational memory](docs/research/05-memory.md)
6. [Knowledge distillation](docs/research/06-distillation.md)

The [comparison table](docs/research/comparison.md) puts them side by side and explains when to use each, and the [conclusions](docs/conclusions.md) answer when to choose RAG over fine-tuning, how embeddings and semantic search work, and which technique fits information that changes daily.

[docs/architecture.md](docs/architecture.md) explains how the app is built and what happens to a question, step by step.

## Usage

```bash
git clone https://github.com/adlune/groundwork-rag.git
cd groundwork-rag
uv sync
```

Answers are generated locally with [Ollama](https://ollama.com) by default. Install it and pull the model used for this research:

```bash
ollama pull qwen3.5:4b
```

Settings like the model, the chunk size or the similarity threshold can be changed by copying **.env.example** to **.env**.

The documents the app answers questions about are listed in **data/sources.yaml**: six papers and reports (PDF) plus the research notes in this repo (Markdown). The PDFs aren't stored in the repo, so download them first:

```bash
uv run python scripts/download_corpus.py           # PDFs into data/raw/
uv run python scripts/ingest.py                    # chunk, embed and store in data/chroma/
uv run python scripts/search.py "How does LoRA work?"   # semantic search, no LLM
uv run python scripts/ask.py "How does LoRA work?"      # full answer with cited sources
uv run pytest                                      # tests
```

Running **ingest.py** again only embeds what changed. `scripts/inspect_chunks.py --show 3` prints chunk statistics and samples without embedding anything.

To start the web app:

```bash
uv run uvicorn groundwork.api:app --port 8000
```

Then open http://localhost:8000. Next to each answer, a retrieval inspector shows the chunks search found, their meaning and keyword scores, the thresholds, and which chunks were sent to the model. The API is documented at http://localhost:8000/docs.

The app refuses in two ways. If no chunk reaches either threshold (`MIN_SIMILARITY` or `MIN_KEYWORD` in **.env**), it answers right away that it doesn't have the information, without calling the model. If the chunks are on topic but don't contain the answer, the model is instructed to say so instead of guessing.

## Evaluation

The app was tested on 45 questions: answerable ones, off-topic ones, and on-topic ones the documents can't answer. Hybrid search finds a relevant passage in the top 5 for 79% of the 28 answerable questions (all 6 of the short ones), the gate wrongly refuses none of them, and every one of the 12 off-topic questions is refused. Results are reproducible: the same run gives the same file, answers included. Details, the failures and the limitations are in [docs/evaluation.md](docs/evaluation.md).

```bash
uv run python scripts/evaluate.py              # retrieval and refusal gate
uv run python scripts/evaluate.py --generate   # also run every question through the model
```

## About

Made by Adrian Lara (contact@adrianlara.dev).

This project was developed as a course assignment, under the [MIT License](LICENSE).

Development was assisted by Claude Code (Opus 5.5, Anthropic) as a collaborative learning tool; all code and documentation were reviewed by the author.
