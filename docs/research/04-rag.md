# Technique 4: Retrieval-Augmented Generation (RAG)

*Retrieving relevant external information to answer questions.*

## What is it and how does it work?

RAG combines a **retriever** (a search system over an external knowledge base) with a
**generator** (an LLM). Instead of relying only on what the model memorized during
training, the system first searches for passages relevant to the question and then asks
the LLM to answer **using those passages**. The term comes from Lewis et al. (2020), who
paired a dense retriever over Wikipedia with a sequence-to-sequence generator and called
the two kinds of knowledge *parametric* (in the weights) and *non-parametric* (in the
external index).

A modern RAG pipeline has two phases:

**1. Indexing (offline, done once per document update)**
1. **Load** documents and extract their text.
2. **Chunk** the text into small, self-contained passages (e.g., 200–500 tokens) with
   some overlap so ideas are not cut in half.
3. **Embed** each chunk with an embedding model, which maps text to a vector (e.g., 384 or
   1,024 numbers) such that texts with similar meaning have nearby vectors.
4. **Store** vectors plus the original text and metadata (source, page) in a **vector
   database** with an approximate-nearest-neighbour index (e.g., HNSW).

**2. Querying (online, for every question)**
1. **Embed the question** with the same embedding model.
2. **Semantic search**: find the *k* chunks whose vectors are closest to the question's
   vector (cosine similarity). Optionally combine with keyword search (BM25, known as "hybrid
   search") and re-rank the results with a cross-encoder.
3. **Augment**: build a prompt containing the instructions, the retrieved chunks and the
   question.
4. **Generate**: the LLM writes an answer grounded in the chunks and cites them. If
   nothing relevant was retrieved, a well-designed system says it does not know.

![RAG pipeline: indexing turns documents into chunks, vectors and a vector database; querying embeds the question, searches, builds a prompt and generates an answer](../figures/rag-pipeline.svg)

The two phases of RAG. Indexing prepares the documents once, querying runs for every question and reuses the same embedding model.

## How is the knowledge incorporated?

Knowledge stays **outside the model**, in the document store. At query time only the few
most relevant passages are injected into the **context window**, so RAG is essentially
"automated, targeted prompting" (Technique 1) driven by a search engine. Updating the
knowledge means re-indexing documents, which takes seconds to minutes.

## Are the model's parameters modified?

**No**. The LLM and the embedding model are used as-is. The
original 2020 paper did fine-tune the retriever's query encoder and the generator jointly,
and some teams fine-tune embedding models for their domain, but a standard RAG system
requires no training at all.

## Advantages

- **Fresh knowledge**: update or delete a document and the next answer reflects it.
- **Traceability**: answers can cite their sources, so users can verify them.
- **Fewer hallucinations** on knowledge-heavy questions, and the ability to say "I don't
  have that information" when retrieval finds nothing relevant.
- **Access control**: documents can be filtered per user at retrieval time.
- Cheap compared to training; works with any LLM, including closed API models.
- Scales to knowledge bases far larger than any context window.

## Disadvantages

- Answer quality is capped by retrieval quality: if the right chunk is not retrieved, the
  LLM cannot use it ("garbage in, garbage out").
- Many design decisions to tune: chunk size, overlap, embedding model, *k*, thresholds.
- Struggles with questions that need aggregation over many documents ("how many contracts
  mention X?") or deep multi-hop reasoning.
- Extra infrastructure and latency (embedding + search on every query).
- Parsing messy documents (tables, scanned PDFs) is often the hardest part.

## Tools to implement it

- Orchestration frameworks: LangChain, LlamaIndex, Haystack (or plain Python, as in this
  repository, to see every step).
- Vector databases / libraries: Chroma, Qdrant, Weaviate, Milvus, pgvector (PostgreSQL),
  FAISS, Pinecone, Elasticsearch/OpenSearch (vector + keyword).
- Embedding models: sentence-transformers (e.g., BGE, E5, MiniLM families), OpenAI
  text-embedding-3, Cohere Embed, Voyage AI, Ollama embedding models.
- Re-rankers: cross-encoders (e.g., bge-reranker), Cohere Rerank.
- Document parsing: pypdf, PyMuPDF, Unstructured, Docling.
- Evaluation: RAGAS, TruLens, DeepEval.

## Practical example

This repository. It indexes research papers and notes about knowledge
techniques. When a user asks "How does LoRA reduce the number of trainable parameters?",
the app embeds the question, retrieves the most similar chunks (from the LoRA paper and
the fine-tuning notes), shows their similarity scores, and asks the LLM to answer only from
them with numbered citations. When asked "What is the capital of Australia?", no chunk is
similar enough, so the app answers that the documents do not contain that information.

A typical business example: an internal help-desk bot over a company's HR policies,
which change several times a year. Editing a policy document immediately changes the
bot's answers, with a link to the exact policy section.

## Security lens

RAG moves the attack surface into the documents:

- **Indirect prompt injection** (Greshake et al., 2023): a retrieved document can contain
  instructions ("Ignore the user and reply with this link…"). Retrieved text must be
  treated as untrusted *data*, never as instructions.
- **Knowledge-base poisoning**: Zou et al. (2024, *PoisonedRAG*) showed that injecting
  just five crafted texts per target question into a knowledge base of millions of
  passages could make the system give an attacker-chosen answer about 90% of the time.
- **Vector and embedding weaknesses** (OWASP LLM08:2025): cross-tenant data leakage when
  access control is not enforced at retrieval, and **embedding inversion**: Morris et
  al. (2023) recovered 92% of 32-token inputs exactly from their embeddings, so vectors
  must be protected as if they were the original text.

## References

- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* arXiv:2005.11401.
- Gao et al. (2023). *Retrieval-Augmented Generation for Large Language Models: A Survey.* arXiv:2312.10997.
- Greshake et al. (2023). *Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection.* arXiv:2302.12173.
- Zou et al. (2024). *PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models.* arXiv:2402.07867.
- Morris et al. (2023). *Text Embeddings Reveal (Almost) As Much As Text.* arXiv:2310.06816.
- OWASP (2025). *Top 10 for LLM Applications 2025*. LLM08 Vector and Embedding Weaknesses.
