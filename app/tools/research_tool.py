from __future__ import annotations

import html
import re
from dataclasses import dataclass

import httpx


@dataclass
class ResearchOutput:
    """Structured company context returned by `ResearchTool.run`.

    These fields become prompt inputs for generation and evaluation and are
    also persisted for CRM traceability.
    """

    domain: str
    company_name: str
    summary: str
    pain_points: list[str]
    value_props: list[str]
    sources: list[str]
    raw_excerpt: str


def normalize_domain(domain: str) -> str:
    """Convert raw user input to canonical domain form.

    Examples:
    - `https://www.stripe.com/pricing` -> `stripe.com`
    - `apple.com` -> `apple.com`
    """
    cleaned = domain.strip().lower()
    cleaned = re.sub(r"^https?://", "", cleaned)
    cleaned = cleaned.split("/")[0].replace("www.", "")
    if not cleaned or "." not in cleaned:
        raise ValueError(f"Invalid domain: {domain!r}")
    return cleaned


class ResearchTool:
    """Lightweight deterministic research tool based on public homepage HTML.

    This is intentionally heuristic (not deep crawling) to keep latency low and
    behavior predictable for an interview-sized project.
    """

    def __init__(self, timeout_seconds: float = 10.0) -> None:
        self.timeout_seconds = timeout_seconds

    async def run(self, domain: str) -> ResearchOutput:
        """Fetch + parse homepage content and derive outreach signals."""
        normalized = normalize_domain(domain)
        homepage_url = f"https://{normalized}"
        try:
            text = await self._fetch_text(homepage_url)
        except httpx.HTTPError:
            text = ""
        title = self._extract_title(text)
        visible = self._extract_visible_text(text)

        company_name = self._infer_company_name(normalized, title)
        summary = self._build_summary(company_name, visible)
        pain_points = self._pain_points(visible)
        value_props = self._value_props(visible)

        return ResearchOutput(
            domain=normalized,
            company_name=company_name,
            summary=summary,
            pain_points=pain_points,
            value_props=value_props,
            sources=[homepage_url],
            raw_excerpt=visible[:2000],
        )

    async def _fetch_text(self, url: str) -> str:
        """Download homepage HTML with redirect support."""
        async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=True) as client:
            response = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            return response.text

    def _extract_title(self, html_text: str) -> str:
        """Extract a cleaned `<title>` string when present."""
        match = re.search(r"<title[^>]*>(.*?)</title>", html_text, flags=re.IGNORECASE | re.DOTALL)
        return re.sub(r"\s+", " ", html.unescape(match.group(1))).strip() if match else ""

    def _extract_visible_text(self, html_text: str) -> str:
        """Remove scripts/styles/tags and collapse whitespace into plain text."""
        no_script = re.sub(
            r"<(script|style|noscript)[^>]*>.*?</\1>",
            " ",
            html_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        no_tags = re.sub(r"<[^>]+>", " ", no_script)
        return re.sub(r"\s+", " ", html.unescape(no_tags)).strip()[:10000]

    def _infer_company_name(self, domain: str, title: str) -> str:
        """Infer company name from title when possible, else from domain root."""
        if title:
            guess = title.split("|")[0].split("-")[0].strip()
            if len(guess) > 2:
                return guess
        return domain.split(".")[0].capitalize()

    def _build_summary(self, company_name: str, visible_text: str) -> str:
        """Build a short summary snippet from visible page content."""
        signal = visible_text[:350]
        if not signal:
            return f"Limited public data found for {company_name}."
        return f"{company_name} appears focused on: {signal}"

    def _pain_points(self, text: str) -> list[str]:
        """Map simple keyword signals to possible outbound pain points."""
        combined = text.lower()
        mapped: list[str] = []
        options = {
            "manual": "Likely manual workflows can be automated.",
            "scale": "Growth may strain existing prospecting process.",
            "data": "Data fragmentation may reduce targeting quality.",
            "customer": "Maintaining message relevance across segments may be hard.",
        }
        for key, value in options.items():
            if key in combined:
                mapped.append(value)
        return mapped[:3] or ["Could benefit from more personalized outbound at scale."]

    def _value_props(self, text: str) -> list[str]:
        """Map site keywords to concise value propositions."""
        combined = text.lower()
        value_props: list[str] = []
        if "ai" in combined or "automation" in combined:
            value_props.append("Automate repetitive outbound tasks while keeping personalization.")
        if "sales" in combined or "revenue" in combined:
            value_props.append("Lift conversion with account-specific outreach.")
        return value_props[:3] or ["Generate tailored outbound copy from lightweight research."]
