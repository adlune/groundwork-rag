# Architecture

How groundwork-rag is put together, and what happens between typing a question and reading the answer.

![Architecture: the browser sends a question to the Python server, which searches ChromaDB and a keyword index, decides whether to answer, builds a prompt and calls the local model through Ollama](figures/architecture.svg)

Three programs run on the same computer. The **browser** shows the page, the **Python server** does the searching and decision making, and **Ollama** runs the language model on the GPU. Everything works offline once the models and documents are downloaded.

## Components

| Part | File | Job |
|---|---|---|
| Web page | `web/` | Question box, answer with clickable citations, retrieval inspector |
| API | `src/groundwork/api.py` | Loads the models once at startup and answers `POST /api/ask` |
| Search | `src/groundwork/retriever.py` | Runs meaning search and keyword search and merges the two rankings |
| Embedding model | `src/groundwork/embeddings.py` | bge-small, turns text into a list of 384 numbers |
| Vector database | `src/groundwork/store.py` | ChromaDB on disk: every chunk's text, vector and source. Search compares the question with every vector |
| Keyword index | `src/groundwork/keyword.py` | BM25, built in memory from the same chunks at startup |
| Gate and prompt | `src/groundwork/rag.py`, `prompts.py` | Decides whether to answer, builds the prompt, maps citations |
| Language model | `src/groundwork/llm/ollama.py` | Sends the prompt to qwen3.5:4b through Ollama |

## Building the index (once, ahead of time)

`scripts/ingest.py` prepares the documents listed in [data/sources.yaml](../data/sources.yaml): six PDFs (papers and the OWASP Top 10 for LLMs) and this repo's research notes.

**1. Clean the text.** PDFs store letters at positions on a page, not paragraphs, so the extracted text needs repair: broken words are joined, page headers and footers are removed, and reference lists, tables of contents and garbled tables are dropped. Without this, junk text ends up matching unrelated questions.

**2. Split into chunks.** Each page or section is cut into pieces of about 350 tokens (roughly a long paragraph), preferring to cut between paragraphs, then between sentences. Neighbouring chunks share about 60 tokens, so an idea sitting on a cut still appears whole in one of them. Small chunks matter because each one becomes a single vector: a whole paper would blur into one vague "average meaning". Chunks never cross a page or heading, so every chunk can be cited precisely.

**3. Turn each chunk into a vector.** The embedding model bge-small reads each chunk (plus its document title) and outputs 384 numbers. It was trained so that texts with similar meaning get similar numbers, which lets a question find a passage even when they share no words.

**4. Store.** ChromaDB keeps every chunk's vector, text and source on disk. Running ingestion again only processes chunks that changed.

The result: 402 chunks from 13 documents.

## Answering a question

**1. Search by meaning.** The question is turned into a vector with the same model and compared with every chunk's vector. The chunks whose vectors point in the most similar direction (cosine similarity) come first. With 402 chunks this exact comparison is a single small matrix product, so the app doesn't use ChromaDB's approximate index, which can miss some of the true nearest chunks and gives slightly different results each time it is rebuilt.

**2. Search by keywords.** BM25 scores chunks by the question's words, giving rare words like "QLoRA" much more weight than common ones like "model". This catches short questions that give the embedding too little to work with.

**3. Merge the two rankings.** Reciprocal Rank Fusion gives each chunk points for its position in each list. The top 5 move on.

**4. Gate.** If no chunk reaches a meaning score of 0.645 or a keyword score of 0.70, the documents don't cover the question. The app answers "I don't have enough information" right away and the model is never called. Both thresholds were calibrated on a set of test questions ([evaluation](evaluation.md)).

**5. Build the prompt.** The chunks that passed are numbered `[1]` to `[n]` and wrapped in tags that mark them as untrusted reference material. The instructions tell the model to answer only from them, cite every statement, and say so if the answer isn't there.

**6. Generate.** Ollama runs qwen3.5:4b on the GPU, at temperature 0 and with a fixed seed so the same question gives the same answer. With the models loaded, search takes tens of milliseconds and the answer about 5 to 6 seconds.

**7. Map citations and reply.** The `[n]` numbers in the answer are matched back to documents and pages. The browser receives the answer, the cited sources, and every retrieved chunk with its scores, which the retrieval inspector displays.

## Design decisions

- **No RAG framework.** Chunking, search, the gate and the prompt are written by hand so every step can be read and explained.
- **Two ways to refuse.** The gate refuses off-topic questions without calling the model. For on-topic questions the documents can't answer, the prompt tells the model to say so.
- **Hybrid search.** Meaning search alone couldn't tell short questions apart from off-topic ones; adding keyword search fixed that ([evaluation](evaluation.md)).
- **Documents are untrusted.** Retrieved text is marked as data in the prompt, which makes it harder (not impossible) for a malicious document to give the model instructions. The page never inserts document text as HTML, so a document can't run code in the browser.
- **Reproducible.** Exact search, a fixed model seed and pinned versions of every document and model mean an evaluation run can be repeated and gives the same results ([evaluation](evaluation.md#reproducibility)).
- **Local and offline.** The model, embeddings and database all run on one machine. After the first run downloads the models, questions and documents never leave it.
