from __future__ import annotations

import asyncio
import html
import logging
import re
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx

from app.llm.base import BaseLLM
from app.tools.prompts import research_extraction_system_prompt, research_extraction_user_prompt
from app.utils.json_guard import JSONValidationError, parse_json_with_repair

logger = logging.getLogger(__name__)


@dataclass
class ResearchOutput:
    """Normalized research artifact consumed by downstream agent stages."""

    domain: str
    company_name: str
    summary: str
    pain_points: list[str]
    value_props: list[str]
    sources: list[str]
    raw_excerpt: str


def normalize_domain(domain: str) -> str:
    """Normalize a user-supplied domain-like string to `example.com` form."""
    cleaned = domain.strip().lower()
    cleaned = re.sub(r"^https?://", "", cleaned)
    cleaned = cleaned.split("/")[0].replace("www.", "")
    if not cleaned or "." not in cleaned:
        raise ValueError(f"Invalid domain: {domain!r}")
    return cleaned


class ResearchTool:
    """Fetch public website context and extract research with an LLM."""

    PAGE_HINTS: tuple[str, ...] = (
        "/about",
        "/about-us",
        "/company",
        "/products",
        "/product",
        "/solutions",
        "/services",
        "/use-cases",
        "/customers",
    )
    BLOCKED_EXTENSIONS: tuple[str, ...] = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".zip")

    def __init__(self, llm: BaseLLM, timeout_seconds: float = 10.0, max_pages: int = 6) -> None:
        """Configure network crawling limits and LLM dependency."""
        self.llm = llm
        self.timeout_seconds = timeout_seconds
        self.max_pages = max_pages

    async def run(self, domain: str) -> ResearchOutput:
        """Execute full research flow for a single domain."""
        normalized = normalize_domain(domain)
        homepage_url = f"https://{normalized}"
        pages = await self._fetch_pages(normalized, homepage_url)
        sources = [url for url, text in pages.items() if text] or [homepage_url]

        homepage_html = pages.get(homepage_url, "")
        title = self._extract_title(homepage_html)
        company_name = self._infer_company_name(normalized, title)

        corpus = self._build_corpus(pages, max_chars=6000, per_page_chars=1200)
        research = await self._extract_research_with_llm(
            domain=normalized,
            company_name=company_name,
            sources=sources,
            corpus=corpus,
        )

        return ResearchOutput(
            domain=normalized,
            company_name=str(research.get("company_name") or company_name),
            summary=str(research.get("summary") or f"Limited public data found for {company_name}."),
            pain_points=self._as_string_list(research.get("pain_points")),
            value_props=self._as_string_list(research.get("value_props")),
            sources=sources,
            raw_excerpt=corpus[:2000],
        )

    async def _fetch_pages(self, domain: str, homepage_url: str) -> dict[str, str]:
        """Fetch homepage plus high-signal internal pages concurrently."""
        async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=True) as client:
            homepage = await self._safe_get(client, homepage_url)
            if not homepage:
                return {homepage_url: ""}

            candidate_urls = self._candidate_urls(homepage_url, domain, homepage)
            urls = [homepage_url, *candidate_urls[: max(self.max_pages - 1, 0)]]
            unique_urls = list(dict.fromkeys(urls))
            pages: dict[str, str] = {homepage_url: homepage}

            async def fetch(url: str) -> None:
                pages[url] = await self._safe_get(client, url)

            await asyncio.gather(*(fetch(url) for url in unique_urls if url != homepage_url))
            return pages

    async def _safe_get(self, client: httpx.AsyncClient, url: str) -> str:
        """Fetch one URL and return empty text on transport/HTTP failures."""
        try:
            response = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            return response.text
        except httpx.HTTPError:
            return ""

    def _candidate_urls(self, homepage_url: str, domain: str, homepage_html: str) -> list[str]:
        """Rank and dedupe crawl candidates from links and known page hints."""
        hrefs = re.findall(r"""<a[^>]+href=["']([^"']+)["']""", homepage_html, flags=re.IGNORECASE)
        scored: list[tuple[int, str]] = []
        for href in hrefs:
            if href.startswith(("#", "mailto:", "tel:", "javascript:")):
                continue
            absolute = urljoin(homepage_url, href)
            parsed = urlparse(absolute)
            host = parsed.netloc.lower().replace("www.", "")
            if host != domain and not host.endswith(f".{domain}"):
                continue
            cleaned = parsed._replace(fragment="", query="").geturl()
            path = parsed.path.lower()
            if any(path.endswith(ext) for ext in self.BLOCKED_EXTENSIONS):
                continue
            score = sum(1 for hint in self.PAGE_HINTS if hint in path)
            if score > 0:
                scored.append((score, cleaned))

        for hint in self.PAGE_HINTS:
            scored.append((1, urljoin(homepage_url, hint)))
        scored.sort(key=lambda item: item[0], reverse=True)
        return list(dict.fromkeys(url for _, url in scored))

    def _extract_title(self, html_text: str) -> str:
        """Extract and sanitize `<title>` text from HTML."""
        match = re.search(r"<title[^>]*>(.*?)</title>", html_text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return ""
        title = re.sub(r"<[^>]+>", " ", match.group(1))
        return re.sub(r"\s+", " ", html.unescape(title)).strip()[:140]

    def _infer_company_name(self, domain: str, title: str) -> str:
        """Infer company name from page title with domain fallback."""
        if title:
            guess = title.split("|")[0].split("-")[0].strip()
            if len(guess) > 2:
                return guess
        return domain.split(".")[0].capitalize()

    def _build_corpus(self, pages: dict[str, str], max_chars: int, per_page_chars: int) -> str:
        """Build bounded plain-text corpus labeled by source URL."""
        chunks: list[str] = []
        for url, html_text in pages.items():
            if not html_text:
                continue
            text = self._extract_visible_text(html_text)
            if not text:
                continue
            chunks.append(f"SOURCE: {url}\n{text[:per_page_chars]}")
        return "\n\n".join(chunks)[:max_chars]

    def _extract_visible_text(self, html_text: str) -> str:
        """Strip scripts/tags/URLs and return normalized visible text."""
        no_comments = re.sub(r"<!--.*?-->", " ", html_text, flags=re.DOTALL)
        no_blocks = re.sub(
            r"<(script|style|noscript)[^>]*>.*?</\1>",
            " ",
            no_comments,
            flags=re.IGNORECASE | re.DOTALL,
        )
        no_tags = re.sub(r"<[^>]+>", " ", no_blocks)
        cleaned = html.unescape(no_tags)
        cleaned = re.sub(r"https?://\S+", " ", cleaned)
        cleaned = re.sub(r"\b(?:src|href)=\S+", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    async def _extract_research_with_llm(
        self,
        domain: str,
        company_name: str,
        sources: list[str],
        corpus: str,
    ) -> dict[str, object]:
        """Extract structured research fields from the crawled corpus."""
        if not corpus:
            return {
                "company_name": company_name,
                "summary": f"Limited public data found for {company_name}.",
                "pain_points": [],
                "value_props": [],
            }

        system_prompt = research_extraction_system_prompt()
        user_prompt = research_extraction_user_prompt(
            domain=domain,
            company_name=company_name,
            sources=sources,
            corpus=corpus,
        )

        required_keys = ["company_name", "summary", "pain_points", "value_props"]
        try:
            raw_full = await self.llm.generate(system_prompt, user_prompt)
            return await parse_json_with_repair(llm=self.llm, raw_text=raw_full, required_keys=required_keys)
        except JSONValidationError as exc:
            logger.warning("Research extraction failed on full corpus for domain=%s: %s", domain, exc)
        except Exception as exc:
            logger.warning(
                "Research generation failed on full corpus for domain=%s; retrying compact prompt: %s",
                domain,
                exc,
            )

        compact_corpus = corpus[:2500]
        compact_prompt = research_extraction_user_prompt(
            domain=domain,
            company_name=company_name,
            sources=sources,
            corpus=compact_corpus,
        )
        try:
            raw_compact = await self.llm.generate(system_prompt, compact_prompt)
            return await parse_json_with_repair(
                llm=self.llm,
                raw_text=raw_compact,
                required_keys=required_keys,
                max_repair_retries=1,
            )
        except JSONValidationError as exc:
            logger.error("Research extraction parse failed on compact corpus for domain=%s: %s", domain, exc)
            return {
                "company_name": company_name,
                "summary": f"Research extracted from {domain} public pages.",
                "pain_points": [],
                "value_props": [],
            }
        except Exception:
            # Bubble transport/runtime failures so agent-level retry_async can retry the whole research step.
            raise

    def _as_string_list(self, value: object) -> list[str]:
        """Coerce model field into short list of non-empty strings."""
        if isinstance(value, list):
            result = [str(item).strip() for item in value if str(item).strip()]
            return result[:4]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []
