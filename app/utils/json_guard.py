from __future__ import annotations

import json
from typing import Any

from app.llm.base import BaseLLM


class JSONValidationError(ValueError):
    """Raised when model output cannot be validated as required JSON."""

    pass


async def parse_json_with_repair(  # noqa: C901
    llm: BaseLLM,
    raw_text: str,
    required_keys: list[str],
    max_repair_retries: int = 2,
) -> dict[str, Any]:
    """Parse model output into validated JSON with bounded self-repair.

    Typical failure modes handled here:
    - markdown code fences around JSON
    - extra text before/after JSON object
    - missing required keys
    """

    def _extract_first_json_object(text: str) -> str:  # noqa: C901
        """Extract first balanced JSON object while respecting string escapes."""
        start = text.find("{")
        if start == -1:
            return text

        depth = 0
        in_string = False
        escape = False
        for idx, char in enumerate(text[start:], start=start):
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
        return text

    def _unwrap_json_text(candidate: str) -> str:
        """Remove markdown wrappers and isolate JSON candidate text."""
        text = candidate.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return _extract_first_json_object(text).strip()

    def _validate(candidate: str) -> dict[str, Any]:
        """Parse candidate JSON and enforce required key presence."""
        data = json.loads(_unwrap_json_text(candidate))
        if not isinstance(data, dict):
            raise JSONValidationError("Response is not a JSON object")
        missing = [k for k in required_keys if k not in data]
        if missing:
            raise JSONValidationError(f"Missing keys: {missing}")
        return data

    candidate = raw_text
    try:
        return _validate(candidate)
    except (json.JSONDecodeError, JSONValidationError):
        pass

    for _ in range(max_repair_retries):
        repair_system_prompt = "You fix invalid JSON. Return JSON only."
        repair_user_prompt = (
            "Repair this output into valid JSON with exactly these keys: "
            f"{required_keys}.\n"
            "Do not add markdown.\n\n"
            f"Input:\n{candidate}"
        )
        candidate = await llm.generate(repair_system_prompt, repair_user_prompt)
        try:
            return _validate(candidate)
        except (json.JSONDecodeError, JSONValidationError):
            continue

    raise JSONValidationError("Could not parse valid JSON after repair attempts")
