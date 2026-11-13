# Comparison of knowledge techniques

This document compares the six techniques researched in this folder and explains when to
use each one.

## Where does the knowledge live?

The simplest way to tell the techniques apart is to ask where the new knowledge ends up.

| Where knowledge lives | Techniques | How to update it |
|---|---|---|
| In the weights (parametric) | Pre-training, continued pre-training, fine-tuning, distillation | Train again |
| In the context window (temporary) | Prompting | Edit the prompt |
| In an external store, copied into the context when needed (non-parametric) | RAG, memory | Edit the database |

![Three places knowledge can live: in the weights, in the context window, and in an external store that is copied into the context](../figures/knowledge-locations.svg)

Techniques that change the weights change what the model *is*. Techniques that use the
context change what the model *sees*.

## Comparison table

| | Prompting | Pre-training / continued pre-training | Fine-tuning | RAG | Memory | Distillation |
|---|---|---|---|---|---|---|
| **What it does** | Puts instructions, facts and examples in the prompt | Trains on huge (domain) corpora with next-token prediction | Further trains a model on a curated task dataset | Searches documents and injects relevant chunks into the prompt | Saves facts from interactions, recalls them later | Trains a small student to imitate a large teacher |
| **Where knowledge lives** | Context window | Weights | Weights / adapters | External document store | External memory store | Student's weights |
| **Modifies parameters?** | No | Yes, all | Yes (all, or added LoRA weights) | No | No | Yes, the student's |
| **Cost to add knowledge** | Very low | Very high (GPU clusters, weeks) | Medium (GPU hours, labelled data) | Low (embedding + storage) | Low | High (teacher queries + training) |
| **Time to update** | Seconds | Weeks to months | Hours to days | Seconds to minutes (re-index) | Real time | Days |
| **Data needed** | A few examples or a short document | Billions to trillions of tokens | Hundreds to thousands of labelled examples | Your documents, unlabelled | Conversation history | Many inputs + teacher outputs |
| **Knowledge capacity** | Limited by context window | Very large | Moderate, facts unreliable | Practically unlimited | Grows over time | Bounded by teacher |
| **Cites sources?** | Only if facts are in the prompt | No | No | Yes | Partially (memory items) | No |
| **Hallucination control** | Medium | Low | Low (can increase it) | High (grounded, can refuse) | Medium | Low |
| **Expertise required** | Low | Very high | Medium to high | Medium | Medium | High |
| **Best at teaching** | Behaviour and small facts | Language, domain fluency | Behaviour, format, skills | Facts that change | Personal context | Efficiency (smaller, faster model) |
| **Main security risk** | Prompt injection, prompt leakage | Data poisoning, backdoors | Erosion of safety alignment, malicious adapters | Indirect injection, knowledge base poisoning, embedding inversion | Persistent memory poisoning, privacy | Model extraction, inherited flaws |

## When to use each technique

**Prompting.** Always, both on its own and as part of every other technique. On its own
it fits when the knowledge is small, changes rarely and has to work right away: a chatbot
that follows a one-page policy, a few-shot classifier prototype, or setting the output
format and tone.

**Pre-training from scratch.** Almost never. It makes sense for large organizations that
need full control over data and licensing, for languages that existing models barely
cover, or for new modalities.

**Continued pre-training.** When an existing model is weak in a whole domain or language
(medicine, law, a low-resource language) and there is a large unlabelled corpus from that
domain.

**Fine-tuning.** When the goal is consistent behaviour that prompting can't reliably
produce, like a strict output schema, a brand voice, a specialized classification task,
or a small model that is good at one narrow job. It is not the right tool for facts that
change.

**RAG.** When the model has to answer from documents that are large, private, change often
or need to be cited: internal knowledge bases, support over product documentation, legal
or policy search, research assistants. This is the default answer to "make the model know
our information".

**Memory.** When the system talks to the same users or works on the same tasks over time
and should remember preferences, history and decisions. Personal assistants, tutors and
long-running agents are the typical cases.

**Distillation.** When a large model already does the task well but is too expensive, too
slow, or can't run where it's needed (edge devices, on-premise, high volume), and a
one-time training effort to get a small, fast model is worth it.

## Combining them

These techniques work together more often than they compete. A production assistant could
use a fine-tuned or distilled model for consistent style and low cost, a system prompt for
rules, RAG for up-to-date company knowledge with citations, and memory for personalization.
A common rule of thumb is to fine-tune for form, retrieve for facts, and prompt for
everything else.
