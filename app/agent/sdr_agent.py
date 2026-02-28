from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from time import perf_counter

from app.config import settings
from app.llm.base import BaseLLM
from app.tools.crm_tool import CRMTool
from app.tools.prompts import (
    evaluation_prompt,
    generation_prompt,
    reflection_prompt,
    rewrite_prompt,
)
from app.tools.research_tool import ResearchOutput, ResearchTool
from app.utils.json_guard import parse_json_with_repair
from app.utils.retry import retry_async

logger = logging.getLogger(__name__)


@dataclass
class AgentRunResult:
    """Normalized output from one completed agent run.

    This object is returned by the orchestrator and then serialized by the API.
    It contains both user-visible artifacts (research + email) and identifiers
    for persisted CRM records.
    """

    lead_id: int
    research_snapshot_id: int
    email_id: int
    domain: str
    company_name: str
    summary: str
    pain_points: list[str]
    value_props: list[str]
    sources: list[str]
    subject: str
    body: str
    call_to_action: str
    reflection_rounds: int
    final_critique_score: int
    evaluation: dict[str, int | float | str]


class SDRAgent:
    """Deterministic agent orchestrator.

    The class owns flow control, not business data. Each step is delegated to a
    tool or adapter, and this class only coordinates sequencing, validation, and
    persistence boundaries.
    """

    def __init__(self, llm: BaseLLM, crm: CRMTool, researcher: ResearchTool) -> None:
        self.llm = llm
        self.crm = crm
        self.researcher = researcher

    async def run(self, domain: str) -> AgentRunResult:
        """Run the full agent lifecycle for one domain.

        Steps:
        1. Research domain context.
        2. Generate initial email draft.
        3. Reflect and optionally rewrite for up to `MAX_REFLECTION_ROUNDS`.
        4. Evaluate final draft on explicit dimensions.
        5. Persist all artifacts as one transactional CRM write.
        """
        started_at = perf_counter()
        logger.info("Agent run start domain=%s", domain)

        logger.info("Step: research start domain=%s", domain)
        research = await retry_async(lambda: self.researcher.run(domain), attempts=3, base_delay_seconds=0.5)
        logger.info("Step: research done domain=%s company=%s", domain, research.company_name)

        logger.info("Step: generate start domain=%s", domain)
        generated = await self._generate_email(research)
        subject = str(generated["subject"])
        body = str(generated["body"])
        cta = str(generated["call_to_action"])
        logger.info("Step: generate done domain=%s", domain)

        critique_score = 0
        reflection_rounds = 0
        logger.info("Step: reflect start domain=%s max_rounds=%s", domain, settings.max_reflection_rounds)
        for _ in range(settings.max_reflection_rounds):
            critique = await self._reflect(research, subject, body, cta)
            critique_score = self._normalize_score(critique["score"])
            if critique_score >= 7:
                break
            rewritten = await self._rewrite(research, subject, body, cta, str(critique["critique"]))
            subject = str(rewritten["subject"])
            body = str(rewritten["body"])
            cta = str(rewritten["call_to_action"])
            reflection_rounds += 1
        logger.info(
            "Step: reflect done domain=%s rounds=%s final_score=%s",
            domain,
            reflection_rounds,
            critique_score,
        )

        logger.info("Step: evaluate start domain=%s", domain)
        evaluation = await self._evaluate(research, subject, body, cta)
        evaluation["relevance"] = self._normalize_score(evaluation["relevance"])
        evaluation["personalization"] = self._normalize_score(evaluation["personalization"])
        evaluation["tone"] = self._normalize_score(evaluation["tone"])
        evaluation["clarity"] = self._normalize_score(evaluation["clarity"])
        evaluation["rationale"] = self._normalize_rationale(evaluation["rationale"])
        overall_score = (
            int(evaluation["relevance"])
            + int(evaluation["personalization"])
            + int(evaluation["tone"])
            + int(evaluation["clarity"])
        ) / 4.0
        evaluation["overall_score"] = overall_score
        logger.info("Step: evaluate done domain=%s overall_score=%.2f", domain, overall_score)

        logger.info("Step: persist start domain=%s", domain)
        ids = await self.crm.persist_agent_run(
            domain=research.domain,
            company_name=research.company_name,
            summary=research.summary,
            pain_points=research.pain_points,
            value_props=research.value_props,
            sources=research.sources,
            raw_excerpt=research.raw_excerpt,
            subject=subject,
            body=body,
            call_to_action=cta,
            reflection_rounds=reflection_rounds,
            final_critique_score=critique_score,
            evaluation=evaluation,
        )
        logger.info(
            "Step: persist done domain=%s lead_id=%s research_id=%s email_id=%s",
            domain,
            ids["lead_id"],
            ids["research_snapshot_id"],
            ids["email_id"],
        )

        elapsed = perf_counter() - started_at
        logger.info("Agent run done domain=%s elapsed_sec=%.2f", domain, elapsed)

        return AgentRunResult(
            lead_id=ids["lead_id"],
            research_snapshot_id=ids["research_snapshot_id"],
            email_id=ids["email_id"],
            domain=research.domain,
            company_name=research.company_name,
            summary=research.summary,
            pain_points=research.pain_points,
            value_props=research.value_props,
            sources=research.sources,
            subject=subject,
            body=body,
            call_to_action=cta,
            reflection_rounds=reflection_rounds,
            final_critique_score=critique_score,
            evaluation=evaluation,
        )

    async def _generate_email(self, research: ResearchOutput) -> dict[str, object]:
        """Generate initial outbound email draft."""
        return await self._generate_structured_json(
            system_prompt="You are an expert SDR assistant. Return only JSON.",
            user_prompt=generation_prompt(research),
            required_keys=["subject", "body", "call_to_action"],
        )

    async def _reflect(self, research: ResearchOutput, subject: str, body: str, cta: str) -> dict[str, object]:
        """Score and critique the current draft."""
        return await self._generate_structured_json(
            system_prompt="You are a strict outbound email reviewer. Return only JSON.",
            user_prompt=reflection_prompt(research, subject, body, cta),
            required_keys=["score", "critique"],
        )

    async def _rewrite(
        self,
        research: ResearchOutput,
        subject: str,
        body: str,
        cta: str,
        critique: str,
    ) -> dict[str, object]:
        """Rewrite draft using critique feedback."""
        return await self._generate_structured_json(
            system_prompt="You are an SDR copywriter rewriting emails. Return only JSON.",
            user_prompt=rewrite_prompt(research, subject, body, cta, critique),
            required_keys=["subject", "body", "call_to_action"],
        )

    async def _evaluate(self, research: ResearchOutput, subject: str, body: str, cta: str) -> dict[str, object]:
        """Produce final quality scores and rationale."""
        return await self._generate_structured_json(
            system_prompt="You score outbound email quality. Return only JSON.",
            user_prompt=evaluation_prompt(research, subject, body, cta),
            required_keys=["relevance", "personalization", "tone", "clarity", "rationale"],
        )

    async def _generate_structured_json(
        self,
        system_prompt: str,
        user_prompt: str,
        required_keys: list[str],
    ) -> dict[str, object]:
        """Generate structured JSON from the model with schema repair fallback."""
        raw = await self.llm.generate(system_prompt, user_prompt)
        return await parse_json_with_repair(
            llm=self.llm,
            raw_text=raw,
            required_keys=required_keys,
        )

    def _normalize_score(self, value: object) -> int:
        """Coerce model-provided score to the allowed integer range [1, 10]."""
        try:
            score = int(value)
        except (TypeError, ValueError):
            return 1
        return max(1, min(10, score))

    def _normalize_rationale(self, value: object) -> str:
        """Ensure rationale is always stored as a plain string."""
        if isinstance(value, str):
            return value
        if isinstance(value, (list, dict)):
            try:
                return json.dumps(value)
            except (TypeError, ValueError):
                return str(value)
        return str(value)
