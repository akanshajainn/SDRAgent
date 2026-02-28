from __future__ import annotations

import httpx

from .base import BaseLLM


class OllamaAdapter(BaseLLM):
    """Call local/remote Ollama `/api/generate` with deterministic settings."""

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://127.0.0.1:11434",
        max_tokens: int = 512,
        temp: float = 0.2,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temp = temp
        self.timeout_seconds = timeout_seconds

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate text using Ollama and return stripped response content."""
        payload = {
            "model": self.model_name,
            "system": system_prompt.strip(),
            "prompt": user_prompt.strip(),
            "stream": False,
            "options": {
                "num_predict": self.max_tokens,
                "temperature": self.temp,
            },
        }
        endpoint = f"{self.base_url}/api/generate"
        timeout = httpx.Timeout(self.timeout_seconds)

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(endpoint, json=payload)

        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama generate failed with status={response.status_code}: {response.text}"
            )

        body = response.json()
        output = body.get("response")
        if not isinstance(output, str):
            raise RuntimeError("Ollama response missing string field 'response'")
        return output.strip()
