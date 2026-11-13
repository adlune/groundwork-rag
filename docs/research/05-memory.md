# Technique 5: Conversational memory

*Storing and retrieving information from previous conversations or interactions.*

## What is it and how does it work?

LLMs are **stateless**. Each API call is independent, and the model remembers nothing
between calls. Memory systems make an assistant appear to remember by **saving
information from past interactions and putting the relevant parts back into the prompt**
later. Memory is usually split into two layers, by analogy with human memory:

**Short-term (working) memory, within one conversation**
- **Buffer**: resend the full message history on every turn (what chat apps do by
  default). Simple, but limited by the context window.
- **Sliding window**: keep only the last *N* messages.
- **Summarization**: periodically compress older messages into a running summary.

**Long-term memory, across conversations and sessions**
1. **Write**: after (or during) a conversation, the system (often the LLM itself)
   extracts durable facts worth keeping ("User prefers Python", "User's project uses
   PostgreSQL 16").
2. **Store**: facts go into a database: a vector store for semantic recall, a key-value or
   SQL store for structured profile data, or a knowledge graph for relations.
3. **Read**: at the start of a new conversation or turn, relevant memories are retrieved
   (often with the same semantic search used by RAG) and inserted into the prompt.
4. **Update / forget**: contradicting facts are merged or replaced, stale ones expire.

![Facts from a first conversation are written to a memory store and read back into the prompt of a later conversation](../figures/memory-loop.svg)

The write and read steps of long-term memory, using the tutoring example from this page.

Researchers often distinguish **episodic** memory (what happened in specific past
interactions), **semantic** memory (facts about the user or world) and **procedural**
memory (learned instructions on how to do tasks). **MemGPT** (Packer et al., 2023)
treats the context window like an operating system treats RAM: the LLM uses function
calls to move information between a small "main context" and large external storage,
paging memories in and out as needed.

## How is the knowledge incorporated?

Like RAG, the knowledge lives **outside the model** (in a database) and is **injected into
the context window** when relevant. The difference from RAG is the *source* and *lifecycle*
of the knowledge. RAG reads a curated document collection, while memory is **written by the
system itself** from interactions and keeps changing with every conversation.

## Are the model's parameters modified?

**No**, only the external memory store and the prompt change.

## Advantages

- The assistant remembers preferences, past decisions and
  ongoing tasks, so users do not repeat themselves.
- Enables long-running **agents** that work across many steps and sessions.
- Cheap and instant to update; users can inspect, edit and delete memories.
- Keeps prompts small by summarizing or selecting only relevant history.

## Disadvantages

- Poor choices of what to remember produce noisy or
  outdated memories that mislead the model.
- Retrieved memories can be irrelevant or contradictory.
- **Privacy**: stores personal data, raising consent, retention and right-to-erasure
  obligations (e.g., GDPR).
- More moving parts: extraction prompts, storage, retrieval, conflict resolution.
- Memory errors persist and compound across sessions.

## Tools to implement it

- Memory-focused frameworks: **Letta** (the open-source continuation of MemGPT), **Mem0**,
  **Zep**.
- Agent frameworks: **LangGraph** (checkpointers for short-term thread state, a Store
  for long-term memories), LangChain message-history utilities, LlamaIndex chat memory.
- Storage: Redis, PostgreSQL/pgvector, SQLite, any vector database.
- Built-in product features: memory in ChatGPT, Claude and Gemini apps; provider APIs that
  offer memory or context-management tools for agents.

## Practical example

A tutoring assistant for programming students. In week 1, a student mentions they are
learning Python, struggle with recursion and prefer short examples. A memory extractor
saves three facts to the student's profile store. In week 3, the student asks a new
question in a fresh chat; the system retrieves those facts and adds to the prompt:
"Known about this student: learning Python; finds recursion hard; prefers short examples."
The model answers with a brief Python example and a gentle recursion explanation, with no
change to the model itself.

## Security lens

**Memory poisoning**. If an indirect
prompt injection (e.g., hidden text in a web page or document the assistant reads) tricks
the system into saving a false or malicious "memory", that instruction is replayed into
every future conversation. Security researcher Johann Rehberger demonstrated this against
ChatGPT's memory feature in 2024, planting persistent instructions that exfiltrated later
conversations. Defences include human confirmation before writing memories, provenance
tags on each memory, and treating stored memories as untrusted input. Memory stores are
also high-value targets for data theft because they concentrate personal information.

## References

- Packer et al. (2023). *MemGPT: Towards LLMs as Operating Systems.* arXiv:2310.08560.
- Park et al. (2023). *Generative Agents: Interactive Simulacra of Human Behavior.* arXiv:2304.03442.
- Zhang et al. (2024). *A Survey on the Memory Mechanism of Large Language Model based Agents.* arXiv:2404.13501.
