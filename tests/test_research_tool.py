from __future__ import annotations

import asyncio
import json

import pytest

from app.tools.research_tool import ResearchTool, normalize_domain


def test_normalize_domain_valid_inputs() -> None:
    """Normalize common valid URL/domain forms."""
    assert normalize_domain("https://www.loom.com/pricing") == "loom.com"
    assert normalize_domain("apple.com") == "apple.com"


def test_normalize_domain_invalid() -> None:
    """Reject malformed domain strings."""
    with pytest.raises(ValueError):
        normalize_domain("not a domain")


def test_research_tool_returns_llm_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Research flow should return parsed fields from valid LLM JSON."""
    html = (
        "<html><head><title>Acme Corp</title></head>"
        "<body>"
        "<a href='/about'>About</a>"
        "<p>Acme provides operations software.</p>"
        "</body></html>"
    )

    class DummyLLM:
        async def generate(self, system_prompt: str, user_prompt: str) -> str:
            assert "keys: company_name, summary, pain_points, value_props" in user_prompt.lower()
            return json.dumps(
                {
                    "company_name": "Acme Corp",
                    "summary": "Acme summary.",
                    "pain_points": ["pain one", "pain two"],
                    "value_props": ["offer one", "offer two"],
                }
            )

    tool = ResearchTool(llm=DummyLLM())

    async def fake_get(client, url: str) -> str:  # type: ignore[no-untyped-def]
        return html

    monkeypatch.setattr(tool, "_safe_get", fake_get)
    result = asyncio.run(tool.run("acme.com"))

    assert result.domain == "acme.com"
    assert result.company_name == "Acme Corp"
    assert result.summary == "Acme summary."
    assert result.pain_points == ["pain one", "pain two"]
    assert result.value_props == ["offer one", "offer two"]
    assert "https://acme.com" in result.sources


def test_research_tool_parse_failure_returns_safe_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid LLM output should fall back to safe empty-list defaults."""
    html = "<html><head><title>Bad Corp</title></head><body><p>Bad content.</p></body></html>"

    class BadLLM:
        async def generate(self, system_prompt: str, user_prompt: str) -> str:
            return "not json"

    tool = ResearchTool(llm=BadLLM())

    async def fake_get(client, url: str) -> str:  # type: ignore[no-untyped-def]
        return html

    monkeypatch.setattr(tool, "_safe_get", fake_get)
    result = asyncio.run(tool.run("badcorp.com"))

    assert result.domain == "badcorp.com"
    assert result.summary
    assert result.pain_points == []
    assert result.value_props == []
