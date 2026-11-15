"""Language model access. The RAG pipeline only depends on `LLMProvider`."""

from groundwork.config import Settings, get_settings
from groundwork.llm.base import LLMError, LLMProvider
from groundwork.llm.ollama import OllamaProvider


def get_llm(settings: Settings | None = None) -> LLMProvider:
    settings = settings or get_settings()
    return OllamaProvider(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        num_ctx=settings.ollama_num_ctx,
        temperature=settings.ollama_temperature,
        seed=settings.ollama_seed,
        timeout=settings.llm_timeout,
    )


__all__ = ["LLMError", "LLMProvider", "OllamaProvider", "get_llm"]
