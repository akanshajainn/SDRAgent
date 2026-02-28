from __future__ import annotations

from app.config import settings

from .base import BaseLLM
from .gpt4all_adapter import GPT4AllAdapter
from .mock_adapter import MockAdapter
from .ollama_adapter import OllamaAdapter
from .openai_adapter import OpenAIAdapter


def build_llm() -> BaseLLM:  # noqa: C901
    """Select and instantiate the LLM adapter for current environment config.

    Keeping provider selection centralized avoids provider-specific conditionals
    in agent logic.
    """
    if settings.llm_provider == "gpt4all":
        return GPT4AllAdapter(
            model_name=settings.gpt4all_model_name,
            model_path=settings.gpt4all_model_path,
            max_tokens=settings.gpt4all_max_tokens,
            temp=settings.gpt4all_temperature,
        )
    if settings.llm_provider == "ollama":
        return OllamaAdapter(
            model_name=settings.ollama_model_name,
            base_url=settings.ollama_base_url,
            max_tokens=settings.ollama_max_tokens,
            temp=settings.ollama_temperature,
            timeout_seconds=settings.ollama_timeout_seconds,
        )
    if settings.llm_provider == "mock":
        if not settings.allow_mock_llm:
            raise ValueError(
                "LLM_PROVIDER=mock is disabled. Set ALLOW_MOCK_LLM=true for tests/dev only."
            )
        return MockAdapter()
    if settings.llm_provider == "openai":
        return OpenAIAdapter(
            api_key=settings.openai_api_key,
            model_name=settings.openai_model_name,
            base_url=settings.openai_base_url,
            max_tokens=settings.openai_max_tokens,
            temp=settings.openai_temperature,
            timeout_seconds=settings.openai_timeout_seconds,
        )
    raise ValueError(
        "Unsupported LLM_PROVIDER "
        f"'{settings.llm_provider}'. Use 'gpt4all', 'ollama', 'openai', or 'mock'."
    )
