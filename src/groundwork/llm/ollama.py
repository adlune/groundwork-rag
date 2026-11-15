"""Local generation with Ollama (https://ollama.com), through its HTTP chat API."""

import httpx

from groundwork.llm.base import LLMError


class OllamaProvider:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        num_ctx: int = 8192,
        temperature: float = 0.0,
        seed: int | None = 42,
        timeout: float = 180.0,
        client: httpx.Client | None = None,
    ):
        self.model = model
        self.name = f"ollama/{model}"
        self.base_url = base_url.rstrip("/")
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.seed = seed
        self.client = client or httpx.Client(timeout=timeout)

    def check(self) -> str | None:
        """None if Ollama is running and has the model, otherwise what's wrong."""
        try:
            resp = self.client.get(f"{self.base_url}/api/tags", timeout=3)
            names = {m["name"] for m in resp.json().get("models", [])}
        except (httpx.HTTPError, ValueError):
            return f"Ollama is not reachable at {self.base_url}. Start it with: ollama serve"
        if self.model not in names:
            return f"Model {self.model} is not installed. Run: ollama pull {self.model}"
        return None

    def generate(self, system: str, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            # Reasoning models (like qwen3.5) would otherwise "think out loud" first,
            # which is slow on a small GPU and not needed to answer from given sources.
            "think": False,
            "options": {
                "num_ctx": self.num_ctx,
                # Low temperature: stick to the most likely wording, i.e. to the sources.
                "temperature": self.temperature,
            },
        }
        if self.seed is not None:
            payload["options"]["seed"] = self.seed
        try:
            resp = self.client.post(f"{self.base_url}/api/chat", json=payload)
        except httpx.ConnectError as exc:
            raise LLMError(
                f"Can't reach Ollama at {self.base_url}. Is it running? Start it with: ollama serve"
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMError(f"Ollama took too long to answer ({self.model}).") from exc
        if resp.status_code == 404:
            raise LLMError(f"Model {self.model} not found in Ollama. Run: ollama pull {self.model}")
        if resp.status_code != 200:
            raise LLMError(f"Ollama returned HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.json()["message"]["content"].strip()
