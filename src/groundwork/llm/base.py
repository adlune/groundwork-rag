from typing import Protocol


class LLMError(RuntimeError):
    """The model could not produce an answer (not running, bad key, refused...)."""


class LLMProvider(Protocol):
    name: str  # shown in the UI, e.g. "ollama/qwen3.5:4b"

    def generate(self, system: str, prompt: str) -> str:
        """Return the model's text answer to one system prompt + one user message."""
        ...
