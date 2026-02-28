from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv() -> None:
        return None


load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Centralized runtime configuration for API, storage, and model adapters."""

    app_name: str = os.getenv("APP_NAME", "sdr-agent")
    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", "8000"))
    db_path: str = os.getenv("SDR_AGENT_DB_PATH", "./data/sdr_agent.db")
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    gpt4all_model_name: str = os.getenv(
        "GPT4ALL_MODEL_NAME",
        "Meta-Llama-3-8B-Instruct.Q4_0.gguf",
    )
    gpt4all_model_path: str = os.getenv(
        "GPT4ALL_MODEL_PATH",
        "~/.cache/gpt4all",
    )
    gpt4all_max_tokens: int = int(os.getenv("GPT4ALL_MAX_TOKENS", "512"))
    gpt4all_temperature: float = float(os.getenv("GPT4ALL_TEMPERATURE", "0.2"))
    ollama_model_name: str = os.getenv("OLLAMA_MODEL_NAME", "llama3.2:3b")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    ollama_max_tokens: int = int(os.getenv("OLLAMA_MAX_TOKENS", "512"))
    ollama_temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
    ollama_timeout_seconds: float = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60"))
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model_name: str = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_max_tokens: int = int(os.getenv("OPENAI_MAX_TOKENS", "512"))
    openai_temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    openai_timeout_seconds: float = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "60"))
    allow_mock_llm: bool = os.getenv("ALLOW_MOCK_LLM", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    max_reflection_rounds: int = int(os.getenv("MAX_REFLECTION_ROUNDS", "2"))


settings = Settings()
