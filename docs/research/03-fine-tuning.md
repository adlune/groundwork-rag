# Technique 3: Fine-Tuning

*Adapting an existing model through additional training.*

## What is it and how does it work?

Fine-tuning takes a pre-trained model and trains it further on a **small, curated,
task-specific dataset**, usually made of labelled examples (prompt → desired response).
Where pre-training teaches general language, fine-tuning teaches a specific **behaviour**:
following instructions, answering in a fixed JSON format, classifying tickets, writing in
a brand's voice, or using a domain's conventions.

Main variants:

- **Supervised Fine-Tuning (SFT)**: train on input/output pairs with the usual
  cross-entropy loss, typically computed only on the response tokens. This is how base
  models become chat/instruction models.
- **Full fine-tuning**: update every weight. Most powerful, but needs a lot of GPU memory
  and produces a full-size copy of the model per task.
- **Parameter-Efficient Fine-Tuning (PEFT)**: freeze the original weights and train a tiny
  number of new ones.
  - **LoRA** (Hu et al., 2021): for a frozen weight matrix *W*, learn a low-rank update
    *ΔW = B·A*, where *B* and *A* are thin matrices of rank *r* (e.g., 8 or 16). Only *A*
    and *B* are trained (often well under 1% of the parameters), and the adapter is a
    small file that can be swapped or merged into *W*.
  - **QLoRA** (Dettmers et al., 2023): load the frozen base model in 4-bit precision and
    train LoRA adapters on top, making it possible to fine-tune a 65B model on a single
    48 GB GPU.
- **Preference tuning**: align outputs with human preferences using pairs of better/worse
  answers, using RLHF (reinforcement learning from human feedback) or the simpler DPO (Direct
  Preference Optimization, Rafailov et al., 2023).

![LoRA: the adapted weights W' equal the frozen W plus the product of two thin trainable matrices B and A](../figures/lora.svg)

LoRA on one 4096 × 4096 layer with rank 8. The big matrix stays frozen and only the two thin matrices are trained.

## How is the knowledge incorporated?

Knowledge and behaviour are **written into the weights** (or into adapter weights that sit
next to them). An important research finding: fine-tuning is very good at teaching
**form, style and skills**, but comparatively weak and risky for teaching **new facts**.
Ovadia et al. (2023) found that RAG consistently outperformed unsupervised fine-tuning for
injecting new knowledge, and Gekhman et al. (2024) found that fine-tuning on facts the
model did not already know is learned slowly and increases the tendency to hallucinate.

## Are the model's parameters modified?

**Yes.** Full fine-tuning modifies all parameters; LoRA/QLoRA leave the original weights
frozen but add new trainable parameters whose effect is added to the original weights.
Either way, the model's behaviour changes permanently (until the adapter is removed).

## Advantages

- Teaches consistent behaviour, formats and tone that are hard to enforce with prompting.
- Shorter prompts at inference (instructions are "baked in"), reducing cost and latency.
- Can make a small model match a large one on a narrow task.
- PEFT makes it affordable: a 7B model can be LoRA-tuned on a single consumer GPU.
- Works offline / on-premise with open-weight models.

## Disadvantages

- Needs a good labelled dataset (hundreds to thousands of high-quality examples).
- Every update requires another training run.
- Unreliable for facts, it may hallucinate and cannot cite sources.
- Risk of **catastrophic forgetting** and of degrading safety behaviour (see Security lens).
- Requires ML expertise, evaluation pipelines and GPU access; many closed models cannot
  be fine-tuned, or only through a provider's limited API.

## Tools to implement it

- Hugging Face **Transformers**, **PEFT** (LoRA/QLoRA) and **TRL** (SFTTrainer,
  DPOTrainer); bitsandbytes for 4-bit quantization.
- Higher-level fine-tuning tools: Unsloth (fast, low-memory), Axolotl, LLaMA-Factory,
  torchtune.
- Managed services: OpenAI fine-tuning API, Google Vertex AI, Amazon Bedrock custom
  models, Azure AI Foundry.
- Experiment tracking: Weights & Biases, MLflow.

## Practical example

A security operations team wants an assistant that turns raw alert logs into a fixed
incident-report JSON schema (`severity`, `mitre_technique`, `summary`, `next_steps`).
Prompting alone produces inconsistent fields. The team collects 2,000 historical alerts
with the reports analysts wrote, and trains a LoRA adapter on an open 8B model using
Unsloth on one GPU. The fine-tuned model now outputs valid, consistently styled reports
with a short prompt. The facts about *current* incidents still come from the logs given in
the prompt. Fine-tuning taught the **skill**, not the data.

## Security lens

Fine-tuning can **erase safety alignment**. Qi et al. (2023) showed that fine-tuning
GPT-3.5 Turbo on just 10 adversarially designed examples (costing under US$0.20 through
the API) removed most of its safety guardrails, and that even benign fine-tuning
datasets degrade safety somewhat. Fine-tuning datasets and third-party adapters are also a
**supply-chain** risk (OWASP LLM03:2025 Supply Chain, LLM04:2025 Data and Model
Poisoning): a malicious LoRA adapter downloaded from a public hub can contain a backdoor.

## References

- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685.
- Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* arXiv:2305.14314.
- Rafailov et al. (2023). *Direct Preference Optimization.* arXiv:2305.18290.
- Ovadia et al. (2023). *Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs.* arXiv:2312.05934.
- Gekhman et al. (2024). *Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?* arXiv:2405.05904.
- Qi et al. (2023). *Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!* arXiv:2310.03693.
