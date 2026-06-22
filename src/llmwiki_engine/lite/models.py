from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


OperationStatus = Literal["created", "running", "failed", "written", "source_recorded"]
StepStatus = Literal["running", "completed", "failed"]
MergeAction = Literal["create", "update", "noop"]
RelatedSource = Literal["source_digest", "wiki_context"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ArtifactRef(StrictModel):
    path: str
    sha256: str
    size_bytes: int


class StepRecord(StrictModel):
    name: str
    status: StepStatus
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    counts: dict[str, int | float | str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    model_calls: int = 0
    repair_count: int = 0


class OperationManifest(StrictModel):
    operation_id: str
    status: OperationStatus = "created"
    created_at: str
    updated_at: str
    vault: str
    raw_path: str
    profile_name: str
    lite_mode: Literal["full_auto"] = "full_auto"
    engine_version: str
    steps: list[StepRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    receipt_path: str | None = None


class SourceRef(StrictModel):
    raw_path: str
    raw_sha256: str
    locator: str


class RawBinding(StrictModel):
    raw_path: str
    raw_sha256: str
    size_bytes: int
    mtime_ns: int
    bound_at: str


class SourcePageUnit(StrictModel):
    page_unit_id: str
    title: str
    page_type: str
    path_hint: str
    summary: str
    content_scope: str
    must_cover_points: list[str]
    source_refs: list[SourceRef]

    @field_validator("page_unit_id", "title", "page_type", "path_hint", "summary", "content_scope")
    @classmethod
    def non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("不能为空。")
        return value


class WeakOrNoiseItem(StrictModel):
    text: str
    reason: str


class SourceDigest(StrictModel):
    source_raw_path: str
    raw_sha256: str
    summary: str
    key_takeaways: list[str]
    page_units: list[SourcePageUnit] = Field(default_factory=list)
    weak_or_noise_items: list[WeakOrNoiseItem] = Field(default_factory=list)


class RepairItem(StrictModel):
    path: str
    reason: str
    local_fix: bool
    model_called: bool


class StructuredRepairReport(StrictModel):
    repairs: list[RepairItem] = Field(default_factory=list)
    model_calls: int = 0


class WikiKnowledgeEntry(StrictModel):
    path: str
    title: str
    page_type: str
    sha256: str
    summary: str
    aliases: list[str] = Field(default_factory=list)
    created: str = ""
    source_raw_paths: list[str] = Field(default_factory=list)
    source_raw_hashes: list[str] = Field(default_factory=list)
    source_prepared_hashes: list[str] = Field(default_factory=list)
    source_operation_ids: list[str] = Field(default_factory=list)
    updated: str = ""
    text_excerpt: str


class CandidateContextHit(StrictModel):
    path: str
    title: str
    rank: int = 0
    page_type: str = "unknown"
    page_sha256: str = ""
    score: float
    match_basis: str = "embedding"
    reason: str
    excerpt: str = ""
    truncated: bool = False
    embedding_cache_hit: bool = False


class CandidateContext(StrictModel):
    candidate_page_id: str
    query: str
    hits: list[CandidateContextHit] = Field(default_factory=list)


class CandidateContexts(StrictModel):
    retrieval_backend: str
    model: str
    input_version: str
    top_k: int
    knowledge_pool_size: int
    candidate_page_count: int
    candidate_pool_hash: str
    embedding_metrics: dict[str, Any] = Field(default_factory=dict)
    items: list[CandidateContext] = Field(default_factory=list)


class WikiSnapshot(StrictModel):
    wiki_root: str
    pool_hash: str
    generated_at: str
    entries: list[WikiKnowledgeEntry] = Field(default_factory=list)
    retrieval_backend: str = "sentence_transformers"
    embedding_metrics: dict[str, Any] = Field(default_factory=dict)


class CandidatePage(StrictModel):
    candidate_page_id: str
    page_unit_id: str
    title: str
    proposed_page_type: str
    proposed_path_hint: str
    summary: str
    body_markdown: str
    open_questions: list[str] = Field(default_factory=list)
    source_refs: list[SourceRef]
    evidence_notes: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class CandidatePages(StrictModel):
    pages: list[CandidatePage]
    skipped_page_unit_ids: list[str] = Field(default_factory=list)


class CandidatePagesWarmup(StrictModel):
    status: Literal["OK"]


class RelatedPageRef(StrictModel):
    target_path: str
    display_title: str
    source: RelatedSource
    reason: str

    @field_validator("target_path", "display_title", "reason")
    @classmethod
    def non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("不能为空。")
        return value


class RelatedCandidateReport(StrictModel):
    owner_id: str
    current_path: str
    target_path: str
    display_title: str = ""
    source: str
    decision: Literal["kept", "filtered", "cutoff"]
    reject_reason: str = ""
    reason: str = ""


class RelatedMergeReport(StrictModel):
    candidates: list[RelatedCandidateReport] = Field(default_factory=list)


class MergeDecision(StrictModel):
    decision_id: str
    candidate_page_id: str
    action: MergeAction
    target_path: str | None = None
    title: str
    page_type: str
    content_scope: str
    candidate_content_locators: list[str] = Field(default_factory=list)
    matched_existing_paths: list[str] = Field(default_factory=list)
    inspected_context_paths: list[str] = Field(default_factory=list)
    strongest_overlap: float = 0.0
    reason: str
    source_refs: list[SourceRef]
    warnings: list[str] = Field(default_factory=list)


class MergePlan(StrictModel):
    decisions: list[MergeDecision]
    action_counts: dict[str, int]
    warnings: list[str] = Field(default_factory=list)


class CompositionItem(StrictModel):
    final_page_id: str
    target_path: str
    action: MergeAction
    merge_decision_ids: list[str]
    candidate_page_ids: list[str]
    existing_page_refs: list[str] = Field(default_factory=list)
    section_order: list[str]
    preserve_rules: list[str] = Field(default_factory=list)
    insert_rules: list[str] = Field(default_factory=list)
    delete_rules: list[str] = Field(default_factory=list)
    source_ref_rules: list[str]
    readability_goal: str
    warnings: list[str] = Field(default_factory=list)


class CompositionPlan(StrictModel):
    items: list[CompositionItem]


class FinalPage(StrictModel):
    final_page_id: str
    target_path: str
    action: MergeAction
    title: str
    page_type: str
    content_sha256: str = ""
    markdown: str
    source_refs: list[SourceRef]
    preimage_sha256: str | None = None
    warnings: list[str] = Field(default_factory=list)


class FinalPages(StrictModel):
    pages: list[FinalPage]
    warnings: list[str] = Field(default_factory=list)


class ValidationIssue(StrictModel):
    severity: Literal["error", "warning"]
    code: str
    message: str
    path: str | None = None


class ValidationReport(StrictModel):
    ok: bool
    issues: list[ValidationIssue] = Field(default_factory=list)


class WriteSetItem(StrictModel):
    kind: Literal["knowledge", "source", "system"]
    target_path: str
    content_sha256: str
    preimage_sha256: str | None = None


class WriteSet(StrictModel):
    items: list[WriteSetItem]
    write_set_sha256: str


class WriteResult(StrictModel):
    written_targets: list[str]


class Receipt(StrictModel):
    operation_id: str
    raw_path: str
    raw_sha256: str
    artifact_hashes: dict[str, str]
    written_targets: list[str]
    action_counts: dict[str, int]
    warnings: list[str]
    provider_contexts: dict[str, Any]
    model_call_count: int
    repair_count: int
    embedding_metrics: dict[str, Any]
    engine_version: str
    profile_name: str
    created_at: str
