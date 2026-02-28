from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Common async interface for all language-model adapters."""

    @abstractmethod
    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a text response for the provided system/user prompts."""
        raise NotImplementedError
