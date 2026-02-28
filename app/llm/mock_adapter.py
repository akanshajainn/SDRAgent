from __future__ import annotations

import json

from .base import BaseLLM


class MockAdapter(BaseLLM):
    """Return predictable JSON payloads for each agent stage."""

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        prompt = f"{system_prompt}\n{user_prompt}".lower()
        if "keys: subject, body, call_to_action" in prompt:
            return json.dumps(
                {
                    "subject": "Quick idea for your outbound process",
                    "body": "Noticed strong product momentum. We can help personalize outbound while keeping volume efficient.",
                    "call_to_action": "Open to a 15-minute discussion next week?",
                }
            )
        if "keys: score, critique" in prompt:
            return json.dumps({"score": 8, "critique": "Good relevance and clarity."})
        if "keys: relevance, personalization, tone, clarity, rationale" in prompt:
            return json.dumps(
                {
                    "relevance": 8,
                    "personalization": 7,
                    "tone": 8,
                    "clarity": 8,
                    "rationale": "Balanced message with clear CTA.",
                }
            )
        return "{}"
