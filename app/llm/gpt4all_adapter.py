from __future__ import annotations

import asyncio
from pathlib import Path

from .base import BaseLLM

try:
    from gpt4all import GPT4All
except ImportError:  # pragma: no cover
    GPT4All = None


class GPT4AllAdapter(BaseLLM):
    """Wrap GPT4All sync generation behind an async adapter interface."""

    def __init__(
        self,
        model_name: str,
        model_path: str,
        max_tokens: int = 512,
        temp: float = 0.2,
    ) -> None:
        if GPT4All is None:
            raise RuntimeError("gpt4all package is not installed")
        self.model_name = model_name
        model_dir = Path(model_path).expanduser()
        model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = str(model_dir)
        self.model = None
        self.max_tokens = max_tokens
        self.temp = temp
        self._lock = asyncio.Lock()

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Run model generation in a thread executor to avoid blocking the event loop."""
        loop = asyncio.get_running_loop()
        async with self._lock:
            return await loop.run_in_executor(
                None,
                self._generate_sync,
                system_prompt,
                user_prompt,
            )

    def _generate_sync(self, system_prompt: str, user_prompt: str) -> str:
        """Load model lazily and perform synchronous text generation."""
        if self.model is None:
            self.model = GPT4All(
                model_name=self.model_name,
                model_path=self.model_path,
                allow_download=True,
            )

        # Keep prompt format model-agnostic for local GGUF models.
        prompt = (
            f"System:\n{system_prompt.strip()}\n\n"
            f"User:\n{user_prompt.strip()}\n\n"
            "Assistant:\n"
        )
        output = self.model.generate(
            prompt,
            max_tokens=self.max_tokens,
            temp=self.temp,
        )
        return str(output).strip()
