import json

import httpx
import pytest

from groundwork.llm import LLMError
from groundwork.llm.ollama import OllamaProvider


def ollama_with(handler) -> OllamaProvider:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OllamaProvider("qwen3.5:4b", num_ctx=8192, client=client)


def test_ollama_sends_system_prompt_context_size_and_no_thinking():
    sent = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": " Answer [1]. "}})

    assert ollama_with(handler).generate("SYS", "PROMPT") == "Answer [1]."
    assert sent["messages"][0] == {"role": "system", "content": "SYS"}
    assert sent["think"] is False and sent["stream"] is False
    assert sent["options"]["num_ctx"] == 8192


def test_ollama_missing_model_gives_a_helpful_error():
    provider = ollama_with(lambda r: httpx.Response(404, json={"error": "model not found"}))
    with pytest.raises(LLMError, match="ollama pull"):
        provider.generate("s", "p")


def test_ollama_not_running_gives_a_helpful_error():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMError, match="ollama serve"):
        ollama_with(handler).generate("s", "p")
