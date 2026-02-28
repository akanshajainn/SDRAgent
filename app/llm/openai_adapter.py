from __future__ import annotations

import httpx

from .base import BaseLLM


class OpenAIAdapter(BaseLLM):
    """Call OpenAI-compatible chat endpoint and normalize text output."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        max_tokens: int = 512,
        temp: float = 0.2,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        self.api_key = api_key.strip()
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temp = temp
        self.timeout_seconds = timeout_seconds

    async def generate(self, system_prompt: str, user_prompt: str) -> str:  # noqa: C901
        """Generate text via chat completions and return extracted content."""
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            "temperature": self.temp,
            "max_tokens": self.max_tokens,
        }
        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        timeout = httpx.Timeout(self.timeout_seconds)

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(endpoint, json=payload, headers=headers)

        if response.status_code != 200:
            raise RuntimeError(
                f"OpenAI chat/completions failed with status={response.status_code}: {response.text}"
            )

        body = response.json()
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("OpenAI response missing 'choices'")

        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise RuntimeError("OpenAI response missing first choice 'message'")

        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_chunks: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_value = part.get("text")
                    if isinstance(text_value, str):
                        text_chunks.append(text_value)
            joined = "\n".join(text_chunks).strip()
            if joined:
                return joined

        raise RuntimeError("OpenAI response missing string 'message.content'")
