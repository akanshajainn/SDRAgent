from __future__ import annotations

from pydantic import BaseModel, Field


class RunAgentRequest(BaseModel):
    domain: str = Field(min_length=3, description="Company domain")


class EmailPayload(BaseModel):
    subject: str
    body: str
    call_to_action: str
    reflection_rounds: int
    final_critique_score: int


class EvaluationPayload(BaseModel):
    relevance: int
    personalization: int
    tone: int
    clarity: int
    rationale: str
    overall_score: float


class ResearchPayload(BaseModel):
    summary: str
    pain_points: list[str]
    value_props: list[str]
    sources: list[str]


class RunAgentResponse(BaseModel):
    lead_id: int
    research_snapshot_id: int
    email_id: int
    domain: str
    company_name: str
    research: ResearchPayload
    email: EmailPayload
    evaluation: EvaluationPayload


class MetricsResponse(BaseModel):
    evaluations_last_7d: int
    avg_overall_score_last_7d: float


class DimensionAverages(BaseModel):
    relevance: float
    personalization: float
    tone: float
    clarity: float
    overall: float


class DailyDimensionPoint(BaseModel):
    day: str
    samples: int
    relevance: float
    personalization: float
    tone: float
    clarity: float
    overall: float


class DimensionTrendResponse(BaseModel):
    last_7d: DimensionAverages
    daily: list[DailyDimensionPoint]


class CRMRecord(BaseModel):
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
    status: str
    baseline_avg_overall_score: float
    recent_avg_overall_score: float
    delta: float
    baseline_count: int
    recent_count: int
    threshold_drop: float
