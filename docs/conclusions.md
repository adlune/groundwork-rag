# Conclusions

Answers to the four questions of the assignment, based on the [research](research/) and on what building and [evaluating](evaluation.md) this app showed.

## When is it better to use RAG instead of fine-tuning?

When the goal is for the model to **know information**, especially information that changes, is private, or has to be cited. Fine-tuning is the better tool when the goal is for the model to **behave** a certain way: follow a strict output format, write in a specific tone, or get good at one narrow task.

The two answer different problems. Fine-tuning changes what the model *is*; RAG changes what the model *sees* when it answers. Research backs this split: fine-tuning was found to be weaker than retrieval for adding new facts (Ovadia et al., 2023), and training a model on facts it didn't know can make it hallucinate more (Gekhman et al., 2024). A common rule of thumb is to fine-tune for form and retrieve for facts. They also combine well: a fine-tuned model can still read retrieved documents.

Use RAG when:
- the information is large, private or changes often (manuals, policies, internal documents)
- answers must point to their sources so people can check them
- the model should say "I don't know" instead of guessing
- there is no budget, data or time to train models

## What are embeddings, and how does semantic search work?

An **embedding** is a list of numbers that represents the meaning of a piece of text. An embedding model (this project uses bge-small, which outputs 384 numbers) is trained on millions of pairs of related texts, such as a question and the passage that answers it, until texts with similar meaning get similar numbers. Each text can be pictured as a point in space, where nearby points mean similar things.

**Semantic search** uses this to search by meaning instead of by exact words:

1. Every document chunk is turned into an embedding ahead of time and stored in a vector database.
2. The question is turned into an embedding with the same model.
3. The database finds the chunks whose embeddings point in the most similar direction (cosine similarity, from about 0.4 for unrelated text to about 0.8 for a close match in this project).

That is why "How can an assistant remember what I told it last week?" finds the notes about long-term memory even though they never use those words.

Semantic search has a blind spot: rare exact terms. "What is QLoRA?" scored 0.52, the same as "Who won the 2018 World Cup?". Combining it with keyword search (BM25) fixed every short question in the evaluation, which is why most production systems use both ("hybrid search").

## What advantages does RAG have over retraining a model?

- **Updates in seconds.** Changing a document and re-indexing it takes seconds; retraining takes hours to months and must be repeated for every change. Removing information is just as easy, which matters for privacy requests.
- **Answers can be checked.** Every statement cites the passage it came from. A retrained model can't say where a fact came from.
- **It can say "I don't know".** When the documents don't cover a question, the app refuses. In the evaluation, it declined all 5 questions whose answers weren't in the documents, and no answer made up facts.
- **Much cheaper.** No GPUs for training, no labelled dataset. This project runs entirely on one consumer GPU.
- **Works with any model.** The model can be swapped (local or cloud) without touching the knowledge.
- **Access control.** Different users can be limited to different documents at search time, which a single trained model can't do.

The costs are worth stating too: answer quality depends on search quality (if the right passage isn't found, the model can't use it), every question pays for an extra search step, and documents become an attack surface (planted instructions or poisoned content).

## Which technique would you choose for a company whose information changes daily, and why?

**RAG**, with the documents re-indexed as they change, plus a system prompt for the company's rules and tone.

The deciding factor is where the knowledge lives. Techniques that store knowledge in the model's weights (pre-training, fine-tuning, distillation) are frozen at training time; with daily changes, the model would be outdated the day after every training run, and retraining every day is too slow and too expensive. RAG keeps the knowledge in a database that is copied into the prompt when needed, so editing a document changes the next answer. Every answer cites the current document it came from, and outdated documents can be deleted immediately.

If the company also needs a fixed answer format or a very specific style, fine-tuning can be added for that behaviour while RAG keeps providing the facts. Conversational memory can be added on top if the assistant should remember each user's preferences.

For a security-minded deployment, the documents should come from trusted sources only, access should be checked at search time, and retrieved text should always be treated as data, never as instructions.

## References

- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* arXiv:2005.11401.
- Ovadia et al. (2023). *Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs.* arXiv:2312.05934.
- Gekhman et al. (2024). *Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?* arXiv:2405.05904.
