from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidencePolicy(str, Enum):
    strict = "strict"
    light = "light"
    opinion = "opinion"
    none = "none"


class PageTypeSpec(StrictModel):
    name: str
    directory: str
    title_prefix: str
    template: str
    required_sections: list[str] = Field(default_factory=list)
    evidence_policy: EvidencePolicy = EvidencePolicy.light


class ProfileSpec(StrictModel):
    name: str
    version: str = "1"
    description: str = ""
    default_page_type: str = "concept"
    source_page_type: str = "source"
    page_types: dict[str, PageTypeSpec]
    template_root: Path | None = Field(default=None, exclude=True)

    @field_validator("page_types")
    @classmethod
    def page_type_keys_match_names(cls, value: dict[str, PageTypeSpec]) -> dict[str, PageTypeSpec]:
        for key, spec in value.items():
            if spec.name != key:
                raise ValueError(f"page_types key {key!r} does not match spec name {spec.name!r}")
        return value


class ProviderConfig(StrictModel):
    task: str
    provider: str
    model: str | None = None
    fixture_dir: Path | None = None
    endpoint: str | None = None
    api_key_env: str | None = None


class ProviderResult(StrictModel):
    task: str
    provider: str
    model: str | None = None
    raw_output: str
    parsed_output: dict[str, Any] | None = None
    parse_success: bool = False
    schema_valid: bool = False
    repair_attempted: bool = False
    latency_ms: int = 0
    cost_usd: float | None = None
    errors: list[str] = Field(default_factory=list)


class RawSpan(StrictModel):
    span_id: str
    raw_path: str
    raw_sha256: str
    start_char: int
    end_char: int
    text_sha256: str
    heading_path: str = "ROOT"
    text: str


class RawIndexArtifact(StrictModel):
    schema_version: Literal["raw_index.v1"] = "raw_index.v1"
    raw_path: str
    raw_sha256: str
    spans: list[RawSpan]


class SemanticAggregation(StrictModel):
    aggregation_id: str
    source_span_ids: list[str]
    summary: str
    extract_policy: Literal["extract", "context_only", "skip"] = "extract"


class SemanticAggregationArtifact(StrictModel):
    schema_version: Literal["semantic_aggregation.v1"] = "semantic_aggregation.v1"
    aggregations: list[SemanticAggregation]


class Claim(StrictModel):
    claim_id: str
    aggregation_id: str
    text: str
    evidence_span_ids: list[str]
    evidence_quote: str
    target_type: str
    target_title: str
    confidence: Literal["high", "medium", "low"] = "medium"


class ClaimsArtifact(StrictModel):
    schema_version: Literal["claims.v1"] = "claims.v1"
    claims: list[Claim]


class PagePlanItem(StrictModel):
    page_type: str
    title: str
    claim_ids: list[str] = Field(default_factory=list)
    source_raw_path: str | None = None
    summary: str = ""


class PagePlanArtifact(StrictModel):
    schema_version: Literal["page_plan.v1"] = "page_plan.v1"
    pages: list[PagePlanItem]


class StepRecord(StrictModel):
    name: str
    status: Literal["pending", "running", "completed", "failed", "skipped"] = "pending"
    started_at: str | None = None
    completed_at: str | None = None
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    error: str | None = None


class OperationManifest(StrictModel):
    schema_version: Literal["operation_manifest.v1"] = "operation_manifest.v1"
    operation_id: str
    operation_type: str
    profile: str
    status: str = "created"
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    steps: list[StepRecord] = Field(default_factory=list)


class EventRecord(StrictModel):
    time: str = Field(default_factory=utc_now)
    operation_id: str
    step: str
    event: str
    status: str | None = None
    duration_ms: int | None = None
    message: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class ReviewIssue(StrictModel):
    severity: Literal["blocking", "major", "minor", "note"]
    artifact: str
    pointer: str | None = None
    problem: str
    suggested_fix: str | None = None


class ReviewResult(StrictModel):
    schema_version: Literal["review_result.v1"] = "review_result.v1"
    reviewer: str
    task: str
    verdict: Literal["pass", "fail", "needs_human"]
    issues: list[ReviewIssue] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"


class EvalCaseResult(StrictModel):
    case_id: str
    schema_valid: bool
    parse_success: bool
    evidence_quote_validity: float | None = None
    page_type_accuracy: float | None = None
    latency_ms: int = 0
    cost_usd: float | None = None
    errors: list[str] = Field(default_factory=list)


class EvalRun(StrictModel):
    schema_version: Literal["eval_run.v1"] = "eval_run.v1"
    run_id: str
    module: str
    dataset: str
    created_at: str = Field(default_factory=utc_now)
    results: list[EvalCaseResult] = Field(default_factory=list)

    @property
    def schema_valid_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for item in self.results if item.schema_valid) / len(self.results)

    @property
    def parse_success_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for item in self.results if item.parse_success) / len(self.results)
