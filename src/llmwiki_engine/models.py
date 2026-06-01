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


class RunMode(str, Enum):
    dev = "dev"
    standard = "standard"


class ArtifactVisibility(str, Enum):
    run_cache = "run_cache"
    committed_receipt = "committed_receipt"
    wiki_output = "wiki_output"


class StepStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    skipped = "skipped"


class OperationStatus(str, Enum):
    created = "created"
    running = "running"
    failed = "failed"
    drafted = "drafted"
    applied = "applied"


class VerificationStatus(str, Enum):
    ok = "ok"
    drift = "drift"
    missing = "missing"
    raw_changed = "raw_changed"


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


class RawPreparationUncertainItem(StrictModel):
    item: str
    reason: str = ""
    severity: Literal["low", "medium", "high"] = "medium"


class RawPreparationArtifact(StrictModel):
    schema_version: Literal["raw_preparation.v0"] = "raw_preparation.v0"
    source_raw_path: str
    document_kind: Literal["transcript", "article", "notes", "mixed", "unknown"] = "unknown"
    prepared_markdown: str
    operations_applied: list[str] = Field(default_factory=list)
    omission_policy: str = "non_content_noise_only"
    uncertain_items: list[RawPreparationUncertainItem] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high"] = "medium"
    requires_human_review: bool = True
    review_notes: str = ""


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
    input_kind: Literal["original_raw", "prepared_raw"] = "original_raw"
    original_raw_path: str | None = None


class ExtractionWindow(StrictModel):
    window_id: str
    source_span_ids: list[str]
    strategy: str = "deterministic"
    reason: str = ""
    text: str
    extract_policy: Literal["extract", "context_only", "skip"] = "extract"


class ExtractionWindowsArtifact(StrictModel):
    schema_version: Literal["extraction_windows.v0"] = "extraction_windows.v0"
    raw_path: str
    raw_sha256: str
    strategy: str
    max_chars: int
    overlap_spans: int = 0
    windows: list[ExtractionWindow]


class Claim(StrictModel):
    claim_id: str
    source_window_id: str
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


class ArtifactRef(StrictModel):
    relative_path: str
    sha256: str
    size_bytes: int
    kind: str
    schema_version: str | None = None
    producer_step: str
    required_for_resume: bool = True
    visibility: ArtifactVisibility = ArtifactVisibility.run_cache


class RawBinding(StrictModel):
    relative_path: str
    sha256: str
    size_bytes: int


class ProviderRuntimeSpec(StrictModel):
    spec: str
    endpoint: str | None = None
    fixture_dir: str | None = None


class ProviderContextRecord(StrictModel):
    record_id: str
    source: Literal["initial_run", "resume_current_config"]
    from_step: str | None = None
    providers: dict[str, ProviderRuntimeSpec] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now)


class StepAttempt(StrictModel):
    attempt: int
    started_at: str
    completed_at: str | None = None
    inputs: list[ArtifactRef] = Field(default_factory=list)
    outputs: list[ArtifactRef] = Field(default_factory=list)
    provider_record_id: str | None = None
    provider_spec: str | None = None
    provider_context_source: Literal["initial_run", "resume_current_config"] | None = None
    error: str | None = None


class StepRecord(StrictModel):
    name: str
    status: StepStatus = StepStatus.pending
    started_at: str | None = None
    completed_at: str | None = None
    inputs: list[ArtifactRef] = Field(default_factory=list)
    outputs: list[ArtifactRef] = Field(default_factory=list)
    attempts: list[StepAttempt] = Field(default_factory=list)
    error: str | None = None


class OperationManifest(StrictModel):
    schema_version: Literal["operation_manifest.v1"] = "operation_manifest.v1"
    operation_id: str
    operation_type: str
    run_mode: RunMode = RunMode.dev
    engine_version: str
    profile: str
    profile_version: str = "1"
    workspace: str
    raw_bindings: list[RawBinding] = Field(default_factory=list)
    provider_contexts: list[ProviderContextRecord] = Field(default_factory=list)
    status: OperationStatus = OperationStatus.created
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


class VerifyIssue(StrictModel):
    code: VerificationStatus
    path: str
    message: str


class VerifyResult(StrictModel):
    ok: bool
    issues: list[VerifyIssue] = Field(default_factory=list)


class ApplyTarget(StrictModel):
    draft_path: str
    target_path: str
    preimage_sha256: str | None = None
    preimage_missing: bool = False


class ApplyPreview(StrictModel):
    schema_version: Literal["apply_preview.v1"] = "apply_preview.v1"
    operation_id: str
    targets: list[ApplyTarget]


class AppliedReceipt(StrictModel):
    schema_version: Literal["applied_receipt.v1"] = "applied_receipt.v1"
    operation_id: str
    applied_at: str = Field(default_factory=utc_now)
    raw_bindings: list[RawBinding]
    prepared_raw: ArtifactRef | None = None
    raw_preparation: ArtifactRef | None = None
    written_pages: list[ArtifactRef]
    profile: str
    profile_version: str
    engine_version: str
