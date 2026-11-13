# Technique 1: Prompting and in-context learning

*Providing information and instructions directly to the model.*

## What is it and how does it work?

Prompting means putting everything the model needs directly into its input text, the **prompt**. A Large
Language Model (LLM) generates each output token by attending to every token currently in
its **context window**, so anything placed there can shape the answer.

Common forms of prompting:

- **System prompt**: a persistent message that defines the role, rules and
  tone (e. g. "You are a support agent for ACME. Only answer questions about billing.").
- **Zero-shot**: just the instruction and the question.
- **In-context learning**: a handful of examples are
  included so the model imitates the pattern. Brown et al. (2020) showed with GPT-3 that
  large models can learn new tasks this way without any training.
- **Context stuffing**: pasting a whole document, policy, or table into the prompt so the
  model can answer about it.
- **Reasoning prompts**: asking the model to think step by step (chain-of-thought) before
  answering, which improves multi-step reasoning.

## How is the knowledge incorporated?

The knowledge exists **only inside the context window for the duration of one request**.
The model reads it like a student reads an open book during an exam. When the
conversation ends (or the text falls out of the window), the knowledge is gone. The
amount of knowledge is limited by the context window size.

![A context window holding a system prompt, examples, a document and a question, sent to a frozen model](../figures/context-window.svg)

Everything the model knows for this request has to be inside the window. The weights stay frozen.

## Are the model's parameters modified?

**No.** The weights are frozen. In-context learning is sometimes described as the model
"learning" from examples, but it is really conditioning: the same frozen network produces
a different output because its input is different.

## Advantages

- Changing a prompt takes seconds and needs no training, GPU, or dataset.
- Cheap to start and works with any model.
- Fully transparent and easy to version, review and A/B test.
- Excellent for controlling **behaviour**.

## Disadvantages

- Limited by context window size.
- Cost and latency grow with every token included in every request.
- Models attend unevenly to long contexts. Information in the middle of a long prompt is
  used less reliably than information at the start or end ("lost in the middle",
  Liu et al., 2023).
- Not persistent. Knowledge must be re-sent on every call.
- Instructions can be overridden or leaked (see Security lens).

## Tools to implement it

- Any LLM API or runtime: Anthropic Claude API, OpenAI API, Google Gemini API, Ollama,
  llama.cpp, vLLM.
- Prompt templating: Jinja2, LangChain PromptTemplate, LlamaIndex prompt templates.
- Prompt optimization and testing: DSPy (programmatic prompt optimization), promptfoo
  (regression tests for prompts).
- Prompt caching (offered by major providers) to reduce the cost of re-sending large,
  repeated prefixes such as long documents.

## Practical example

A small online store wants a chatbot that knows its return policy. The policy is two
paragraphs long, so the team puts it in the system prompt:

```
System: You are the assistant for "Lumen Lamps". Answer only using the policy below.
If the answer is not in the policy, say you don't know and suggest emailing support.

<policy>
Returns are accepted within 30 days of delivery. Items must be unused...
</policy>

User: Can I return a lamp I bought 45 days ago?
```

The model answers "No, returns are accepted within 30 days of delivery", having learned
the policy only for this request. This is the right tool here: the knowledge is small,
rarely changes, and fits easily in the context.

## Security lens

Prompting creates the **prompt injection** attack surface (OWASP LLM01:2025). Because
instructions and data share the same channel (text), a user can write "Ignore previous
instructions and..." (**direct injection**), and attackers can hide instructions inside
documents, web pages or emails that the model later reads (**indirect injection**).
System prompts can also be extracted by users (OWASP LLM07:2025 System Prompt Leakage),
so they must never contain secrets such as API keys or passwords.

## References

- Brown et al. (2020). *Language Models are Few-Shot Learners.* arXiv:2005.14165.
- Wei et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* arXiv:2201.11903.
- Liu et al. (2023). *Lost in the Middle: How Language Models Use Long Contexts.* arXiv:2307.03172.
- OWASP (2025). *Top 10 for LLM Applications 2025*. LLM01 Prompt Injection, LLM07 System Prompt Leakage.
