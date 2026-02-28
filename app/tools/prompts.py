from __future__ import annotations

from app.tools.research_tool import ResearchOutput


def generation_prompt(research: ResearchOutput) -> str:
    """Build prompt for first-pass email drafting."""
    return (
        "You are a senior SDR writing one cold outbound email.\n"
        "Return ONLY strict JSON with keys: subject, body, call_to_action.\n"
        "Constraints:\n"
        "- Body <= 140 words\n"
        "- No markdown\n"
        "- Concrete and specific, avoid generic buzzwords\n"
        "- Include one clear CTA\n\n"
        f"Company: {research.company_name}\n"
        f"Domain: {research.domain}\n"
        f"Summary: {research.summary}\n"
        f"Pain points: {research.pain_points}\n"
        f"Value props: {research.value_props}\n"
    )


def reflection_prompt(research: ResearchOutput, subject: str, body: str, cta: str) -> str:
    """Build prompt for quality critique and one scalar score."""
    return (
        "Critique this outbound email as a strict reviewer.\n"
        "Return ONLY strict JSON with keys: score, critique.\n"
        "Scoring guide (integer 1-10):\n"
        "- 1-3: poor fit or unclear\n"
        "- 4-6: acceptable but generic or weak\n"
        "- 7-8: strong and relevant\n"
        "- 9-10: exceptional specificity and clarity\n"
        "Use the full range when appropriate; do not default to 7.\n\n"
        f"Company: {research.company_name}\n"
        f"Context: {research.summary}\n"
        f"Subject: {subject}\n"
        f"Body: {body}\n"
        f"CTA: {cta}\n"
    )


def rewrite_prompt(research: ResearchOutput, subject: str, body: str, cta: str, critique: str) -> str:
    """Build prompt for rewriting the draft based on critique."""
    return (
        "Rewrite the outbound email based on the critique below.\n"
        "Return ONLY strict JSON with keys: subject, body, call_to_action.\n"
        "Constraints:\n"
        "- Body <= 140 words\n"
        "- No markdown\n"
        "- Preserve a clear CTA\n"
        "- Improve relevance and specificity, remove fluff\n\n"
        f"Company: {research.company_name}\n"
        f"Summary: {research.summary}\n"
        f"Current subject: {subject}\n"
        f"Current body: {body}\n"
        f"Current CTA: {cta}\n"
        f"Critique: {critique}\n"
    )


def evaluation_prompt(research: ResearchOutput, subject: str, body: str, cta: str) -> str:
    """Build prompt for final structured dimension scoring."""
    return (
        "Evaluate this outbound email.\n"
        "Return ONLY strict JSON with keys: relevance, personalization, tone, clarity, rationale.\n"
        "Each score must be an integer 1-10 and use this rubric:\n"
        "- relevance: fit to company context/problem\n"
        "- personalization: company-specific details vs generic copy\n"
        "- tone: professional, concise, credible\n"
        "- clarity: message structure and CTA clarity\n"
        "Important:\n"
        "- Use the full scale; do not cluster scores around 7.\n"
        "- If a dimension is weak, score it lower even if others are strong.\n"
        "- rationale should briefly justify each sub-score.\n\n"
        f"Company: {research.company_name}\n"
        f"Domain: {research.domain}\n"
        f"Context: {research.summary}\n"
        f"Subject: {subject}\n"
        f"Body: {body}\n"
        f"CTA: {cta}\n"
    )
