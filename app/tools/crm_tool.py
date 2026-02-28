from __future__ import annotations

from app.db.store import LeadStore


class CRMTool:
    """Expose persistence actions as an explicit agent tool."""

    def __init__(self, store: LeadStore) -> None:
        self.store = store

    async def persist_agent_run(
        self,
        domain: str,
        company_name: str,
        summary: str,
        pain_points: list[str],
        value_props: list[str],
        sources: list[str],
        raw_excerpt: str,
        subject: str,
        body: str,
        call_to_action: str,
        reflection_rounds: int,
        final_critique_score: int,
        evaluation: dict[str, int | str | float],
    ) -> dict[str, int]:
        """Persist agent artifacts and return created entity identifiers."""
        return await self.store.persist_agent_run(
            domain=domain,
            company_name=company_name,
            summary=summary,
            pain_points=pain_points,
            value_props=value_props,
            sources=sources,
            raw_excerpt=raw_excerpt,
            subject=subject,
            body=body,
            call_to_action=call_to_action,
            reflection_rounds=reflection_rounds,
            final_critique_score=final_critique_score,
            evaluation=evaluation,
        )
