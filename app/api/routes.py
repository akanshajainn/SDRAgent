from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query, status

from app.agent.sdr_agent import SDRAgent
from app.api.schemas import (
    CRMFullRecord,
    CRMRecord,
    DimensionTrendResponse,
    EvalRegressionResponse,
    MetricsResponse,
    RunAgentRequest,
    RunAgentResponse,
)
from app.db.store import LeadStore

logger = logging.getLogger(__name__)


def build_router(agent: SDRAgent, store: LeadStore) -> APIRouter:  # noqa: C901
    """Create route handlers bound to runtime dependencies.

    We inject the agent and store from `main.py` so handlers stay easy to test.
    """
    router = APIRouter()

    @router.post("/run-agent", response_model=RunAgentResponse)
    async def run_agent(payload: RunAgentRequest) -> RunAgentResponse:
        """Run one agent execution for a single domain input."""
        logger.info("Received /run-agent request for domain=%s", payload.domain)
        try:
            result = await agent.run(payload.domain)
            logger.info(
                "Completed /run-agent for domain=%s lead_id=%s research_id=%s email_id=%s",
                payload.domain,
                result.lead_id,
                result.research_snapshot_id,
                result.email_id,
            )
            return RunAgentResponse(
                lead_id=result.lead_id,
                research_snapshot_id=result.research_snapshot_id,
                email_id=result.email_id,
                domain=result.domain,
                company_name=result.company_name,
                research={
                    "summary": result.summary,
                    "pain_points": result.pain_points,
                    "value_props": result.value_props,
                    "sources": result.sources,
                },
                email={
                    "subject": result.subject,
                    "body": result.body,
                    "call_to_action": result.call_to_action,
                    "reflection_rounds": result.reflection_rounds,
                    "final_critique_score": result.final_critique_score,
                },
                evaluation=result.evaluation,
            )
        except ValueError as exc:
            detail = str(exc)
            if detail.startswith("Invalid domain:"):
                logger.info("Invalid request for /run-agent domain=%s error=%s", payload.domain, detail)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=detail,
                ) from exc
            logger.exception("Agent execution failed for domain=%s", payload.domain)
            raise HTTPException(status_code=500, detail="agent failed") from exc
        except Exception as exc:
            logger.exception("Agent execution failed for domain=%s", payload.domain)
            raise HTTPException(status_code=500, detail="agent failed") from exc

    @router.get("/metrics", response_model=MetricsResponse)
    async def metrics() -> MetricsResponse:
        """Return 7-day aggregate evaluation metrics."""
        data = await store.metrics_7d()
        return MetricsResponse(**data)

    @router.get("/metrics/dimensions-trend", response_model=DimensionTrendResponse)
    async def metrics_dimensions_trend(days: int = Query(default=14, ge=3, le=90)) -> DimensionTrendResponse:
        """Return dimension-level quality trends for dashboards."""
        data = await store.dimension_trends(days=days)
        return DimensionTrendResponse(**data)

    @router.get("/crm/recent", response_model=list[CRMRecord])
    async def crm_recent(limit: int = Query(default=10, ge=1, le=100)) -> list[CRMRecord]:
        """Return a compact recent-run list for quick UI summaries."""
        rows = await store.recent_crm_records(limit=limit)
        return [CRMRecord(**row) for row in rows]

    @router.get("/crm/full", response_model=list[CRMFullRecord])
    async def crm_full(limit: int = Query(default=500, ge=1, le=5000)) -> list[CRMFullRecord]:
        """Return rich run records with full research, email, and eval fields."""
        rows = await store.full_crm_records(limit=limit)
        return [CRMFullRecord(**row) for row in rows]

    @router.get("/eval-regression", response_model=EvalRegressionResponse)
    async def eval_regression(
        threshold_drop: float = Query(default=0.5, ge=0.1, le=3.0),
    ) -> EvalRegressionResponse:
        """Compare recent quality window against baseline window."""
        data = await store.eval_regression_status(threshold_drop=threshold_drop)
        return EvalRegressionResponse(**data)

    return router
