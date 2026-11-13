# Technique 2: Pre-training from scratch and continued pre-training

*Training a model from scratch or continuing its training with new data.*

## What is it and how does it work?

**Pre-training** is how an LLM is born. A neural network starts with random weights and is trained on an enormous text corpus with a
**self-supervised objective**: predict the next token. It works as a probabilistic model. Given "The capital of France is",
the model outputs a probability for every token in its vocabulary; the loss (cross-entropy)
measures how much probability it gave to the true next token ("Paris"), and
backpropagation nudges billions of weights to make the right answer more likely. Repeated
over trillions of tokens, this forces the network to encode grammar, facts, reasoning
patterns and world knowledge into its weights. For scale: GPT-3 was trained on roughly
300 billion tokens (Brown et al., 2020) and Llama 3 on over 15 trillion tokens
(Meta, 2024).

![Bar chart of next-token probabilities for "The capital of France is", with Paris at 46.5%](../figures/next-token.svg)

Real probabilities measured from a small open model (qwen3.5:4b). Even for a simple fact, the model spreads probability over many possible continuations, and training pushes the right one up.

**Continued pre-training** (also called domain-adaptive pre-training) takes an
already pre-trained model and keeps running the same next-token objective on a new,
usually domain-specific corpus: medical papers, legal contracts, source code, or a
company's internal documents. Gururangan et al. (2020) showed that a second phase of
pre-training on in-domain text consistently improves performance on tasks from that
domain. Code Llama, for example, was created by continuing to pre-train Llama 2 on a large
code corpus.

## How is the knowledge incorporated?

The knowledge is **compressed into the model's weights** (parametric knowledge). There is
no lookup table: facts are stored implicitly and in a distributed way across millions of
parameters. The model can then "recall" them when prompted, but it cannot point to where
a fact came from.

## Are the model's parameters modified?

**Yes.** Pre-training creates every parameter; continued pre-training
updates all of them.

## Advantages

- Deepest form of knowledge integration. The model truly "speaks" the domain, not only quotes it.
- Full control over data, architecture, tokenizer, licensing and
  provenance, which matters for regulated or sovereign deployments.
- Continued pre-training reuses an existing model, so it is far cheaper than starting from
  zero while still adapting to a new domain or language.
- No extra retrieval infrastructure at inference time.

## Disadvantages

- It's very expensive. Training a competitive model from scratch requires thousands of
  GPUs for weeks or months and very large, cleaned datasets.
- Knowledge is frozen at the training cut-off, which means updating a fact means training again.
- **Catastrophic forgetting**: continued pre-training on narrow data can degrade general
  abilities unless general data is mixed in ("replay").
- No source attribution and hallucinations remain possible.
- Requires deep ML engineering expertise (distributed training, stability, evaluation).

## Tools to implement it

- Frameworks: PyTorch, JAX; Hugging Face Transformers and Datasets.
- Large-scale distributed training: NVIDIA Megatron-LM and NeMo, Microsoft DeepSpeed,
  PyTorch FSDP, torchtitan.
- Data pipelines: Hugging Face datatrove, deduplication and quality-filtering tools.
- Educational: Andrej Karpathy's nanoGPT / nanochat.
- Cloud compute: GPU clusters on AWS, GCP, Azure, or specialized GPU clouds.

## Practical example

**From scratch:** BloombergGPT (Wu et al., 2023) is a 50-billion-parameter model trained
from scratch on a mix of roughly 363 billion tokens of financial data and 345 billion
tokens of general data, giving Bloomberg a model that excels at financial NLP tasks.

**Continued pre-training:** a hospital takes an open model such as Llama and continues
pre-training it on millions of de-identified clinical notes and medical papers, mixing in
general text to avoid forgetting. The result understands clinical abbreviations and
terminology much better than the original model, and can then be fine-tuned (Technique 3)
for specific tasks.

## Security lens

Training data is an attack surface: **data poisoning** (OWASP LLM04:2025 Data and Model
Poisoning). An attacker who can place content in a web-scraped corpus can implant biases
or **backdoors**, which are hidden behaviours triggered by a specific phrase. Carlini et al. (2023)
showed that poisoning web-scale datasets is practical (e.g., by buying expired domains
referenced in datasets). Hubinger et al. (2024, "Sleeper Agents") showed that deliberately
trained backdoors can persist through standard safety training. Models can also memorize
and later leak sensitive training data, so data provenance and deduplication matter.

## References

- Brown et al. (2020). *Language Models are Few-Shot Learners.* arXiv:2005.14165.
- Gururangan et al. (2020). *Don't Stop Pretraining: Adapt Language Models to Domains and Tasks.* arXiv:2004.10964.
- Rozière et al. (2023). *Code Llama: Open Foundation Models for Code.* arXiv:2308.12950.
- Wu et al. (2023). *BloombergGPT: A Large Language Model for Finance.* arXiv:2303.17564.
- Carlini et al. (2023). *Poisoning Web-Scale Training Datasets is Practical.* arXiv:2302.10149.
- Hubinger et al. (2024). *Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training.* arXiv:2401.05566.
