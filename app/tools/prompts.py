from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.tools.research_tool import ResearchOutput


def generation_system_prompt() -> str:
    """System instruction for first-pass email generation."""
    return "You are an expert SDR assistant. Return only JSON."


def generation_prompt(research: ResearchOutput) -> str:
    """Build user prompt for generating a grounded outbound email draft."""
    return (
        "You are a senior SDR writing one cold outbound email.\n"
        "Return ONLY strict JSON with keys: subject, body, call_to_action.\n"
        "Constraints:\n"
        "- Body <= 140 words\n"
        "- No markdown\n"
        "- Concrete and specific, avoid generic buzzwords\n"
        "- Mention exactly one company-specific observation from Summary or Value props\n"
        "- Do not invent metrics, percentages, customer names, case studies, or guarantees\n"
        "- If uncertain, use cautious language: may, can, often\n"
        "- Include one clear CTA\n\n"
        f"Company: {research.company_name}\n"
        f"Domain: {research.domain}\n"
        f"Summary: {research.summary}\n"
        f"Pain points: {research.pain_points}\n"
        f"Value props: {research.value_props}\n"
    )


def reflection_system_prompt() -> str:
    """System instruction for scoring and critiquing a draft."""
    return "You are a strict outbound email reviewer. Return only JSON."


def reflection_prompt(research: ResearchOutput, subject: str, body: str, cta: str) -> str:
    """Build user prompt for reflection-stage scoring."""
    return (
        "Critique this outbound email as a strict reviewer.\n"
        "Return ONLY strict JSON with keys: score, critique.\n"
        "Scoring guide (integer 1-10):\n"
        "- 1-3: poor fit or unclear\n"
        "- 4-6: acceptable but generic or weak\n"
        "- 7-8: strong and relevant\n"
        "- 9-10: exceptional specificity and clarity\n"
        "Use the full range when appropriate; do not default to 7.\n\n"
        "Critical checks:\n"
        "- Penalize fabricated metrics or unsupported factual claims.\n"
        "- Penalize generic language not tied to the provided context.\n\n"
        f"Company: {research.company_name}\n"
        f"Context: {research.summary}\n"
        f"Subject: {subject}\n"
        f"Body: {body}\n"
        f"CTA: {cta}\n"
    )


def rewrite_system_prompt() -> str:
    """System instruction for rewrite stage."""
    return "You are an SDR copywriter rewriting emails. Return only JSON."


def rewrite_prompt(research: ResearchOutput, subject: str, body: str, cta: str, critique: str) -> str:
    """Build user prompt for critique-driven rewrite generation."""
    return (
        "Rewrite the outbound email based on the critique below.\n"
        "Return ONLY strict JSON with keys: subject, body, call_to_action.\n"
        "Constraints:\n"
        "- Body <= 140 words\n"
        "- No markdown\n"
        "- Preserve a clear CTA\n"
        "- Improve relevance and specificity, remove fluff\n\n"
        "- Do not include made-up stats, percentages, or named proof points\n"
        "- Keep claims grounded in the supplied company context\n\n"
        f"Company: {research.company_name}\n"
        f"Summary: {research.summary}\n"
        f"Pain points: {research.pain_points}\n"
        f"Value props: {research.value_props}\n"
        f"Current subject: {subject}\n"
        f"Current body: {body}\n"
        f"Current CTA: {cta}\n"
        f"Critique: {critique}\n"
    )


def evaluation_system_prompt() -> str:
    """System instruction for final quality evaluation."""
    return "You score outbound email quality. Return only JSON."


def evaluation_prompt(research: ResearchOutput, subject: str, body: str, cta: str) -> str:
    """Build user prompt for final multi-dimension scoring."""
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


def unsupported_claims_rewrite_critique() -> str:
    """Reusable critique when copy contains likely hallucinated proof."""
    return "Remove unsupported metrics/proof claims and keep statements grounded in provided company context."


def constrained_json_repair_system_prompt() -> str:
    """System instruction for strict JSON conversion fallback."""
    return "You convert text into valid JSON. Return only minified JSON."


def constrained_json_repair_user_prompt(required_keys: list[str], raw_text: str) -> str:
    """Build conversion prompt that constrains output to required keys."""
    return (
        "Return one valid JSON object with these required keys: "
        f"{required_keys}. Use simple scalar values only.\n"
        f"Input to convert:\n{raw_text}"
    )


def json_repair_system_prompt() -> str:
    """System instruction for general JSON repair."""
    return "You fix invalid JSON. Return JSON only."


def json_repair_user_prompt(required_keys: list[str], candidate: str) -> str:
    """Build repair prompt for malformed model output."""
    return (
        "Repair this output into valid JSON with exactly these keys: "
        f"{required_keys}.\n"
        "Do not add markdown.\n\n"
        f"Input:\n{candidate}"
    )


def research_extraction_system_prompt() -> str:
    """System instruction for web-corpus research extraction."""
    return "You are a company research analyst. Return JSON only."


def research_extraction_user_prompt(
    domain: str,
    company_name: str,
    sources: list[str],
    corpus: str,
) -> str:
    """Build extraction prompt from crawled company webpage text."""
    return (
        "Extract company research from the provided website corpus.\n"
        "Return ONLY strict JSON with keys: company_name, summary, pain_points, value_props.\n"
        "Rules:\n"
        "- value_props = what the company offers.\n"
        "- pain_points = customer/business problems this company helps solve.\n"
        "- summary = 1-2 lines, plain text, no code/HTML.\n"
        "- pain_points/value_props should be concise arrays with 2-4 items each.\n"
        "- Use only evidence from the corpus; no invented facts.\n\n"
        f"Domain: {domain}\n"
        f"Default company_name: {company_name}\n"
        f"Sources: {sources}\n\n"
        f"Corpus:\n{corpus}"
    )
