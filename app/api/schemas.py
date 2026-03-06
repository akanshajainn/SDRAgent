from __future__ import annotations

from pydantic import BaseModel, Field


class RunAgentRequest(BaseModel):
    """Input payload for triggering one SDR run."""

    domain: str = Field(min_length=3, description="Company domain")


class EmailPayload(BaseModel):
    """Serialized email artifact returned by the agent."""

    subject: str
    body: str
    call_to_action: str
    reflection_rounds: int
    final_critique_score: int


class EvaluationPayload(BaseModel):
    """Dimension-level quality scores with explanation."""

    relevance: int
    personalization: int
    tone: int
    clarity: int
    rationale: str
    overall_score: float


class ResearchPayload(BaseModel):
    """Research summary extracted from public company pages."""

    summary: str
    pain_points: list[str]
    value_props: list[str]
    sources: list[str]


class RunAgentResponse(BaseModel):
    """Full API response for `/run-agent`."""

    lead_id: int
    research_snapshot_id: int
    email_id: int
    domain: str
    company_name: str
    research: ResearchPayload
    email: EmailPayload
    evaluation: EvaluationPayload


class MetricsResponse(BaseModel):
    """Top-line evaluation metrics for the last 7 days."""

    evaluations_last_7d: int
    avg_overall_score_last_7d: float


class DimensionAverages(BaseModel):
    """Averages by scoring dimension."""

    relevance: float
    personalization: float
    tone: float
    clarity: float
    overall: float


class DailyDimensionPoint(BaseModel):
    """One day of aggregate quality metrics."""

    day: str
    samples: int
    relevance: float
    personalization: float
    tone: float
    clarity: float
    overall: float


class DimensionTrendResponse(BaseModel):
    """Trend payload combining rolling and daily aggregates."""

    last_7d: DimensionAverages
    daily: list[DailyDimensionPoint]


class CRMRecord(BaseModel):
    """Compact CRM row used in recent-runs tables."""

    lead_id: int
    domain: str
    company_name: str
    research_snapshot_id: int
    email_id: int
    summary: str
    subject: str
    overall_score: float
    created_at: str


class CRMFullRecord(BaseModel):
    """Full-fidelity CRM row with research, email, and evaluation data."""

    lead_id: int
    domain: str
    company_name: str
    research_snapshot_id: int
    email_id: int
    summary: str
    pain_points: list[str]
    value_props: list[str]
    sources: list[str]
    subject: str
    body: str
    call_to_action: str
    reflection_rounds: int
    final_critique_score: int
    relevance: int
    personalization: int
    tone: int
    clarity: int
    rationale: str
    overall_score: float
    created_at: str


class EvalRegressionResponse(BaseModel):
    """Regression detector response comparing recent vs baseline windows."""

    status: str
    baseline_avg_overall_score: float
    recent_avg_overall_score: float
    delta: float
    baseline_count: int
    recent_count: int
    threshold_drop: float
