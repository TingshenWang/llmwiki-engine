from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


class RawPreparePolicy(str, Enum):
    auto = "auto"
    force_model = "force-model"
    skip_model = "skip-model"


class ArtifactVisibility(str, Enum):
    run_cache = "run_cache"
    committed_receipt = "committed_receipt"
    wiki_output = "wiki_output"


class StepStatus(str, Enum):
    pending = "pending"
    running = "running"
    awaiting_review = "awaiting_review"
    approved = "approved"
    rejected = "rejected"
    completed = "completed"
    failed = "failed"
    skipped = "skipped"


class OperationStatus(str, Enum):
    created = "created"
    running = "running"
    awaiting_review = "awaiting_review"
    failed = "failed"
    drafted = "drafted"
    apply_failed = "apply_failed"
    applied = "applied"
    source_recorded = "source_recorded"


class VerificationStatus(str, Enum):
    ok = "ok"
    drift = "drift"
    missing = "missing"
    raw_changed = "raw_changed"


class VaultConfig(StrictModel):
    wiki_language: Literal["zh-CN"] = "zh-CN"
    max_context_chars: int = 800_000
    max_ingest_candidates: int = 12
    raw_prepare_policy: RawPreparePolicy = RawPreparePolicy.auto
    embedding_retrieval: "EmbeddingRetrievalConfig" = Field(default_factory=lambda: EmbeddingRetrievalConfig())


class EmbeddingRetrievalConfig(StrictModel):
    enabled: bool = True
    backend: Literal["sentence_transformers", "exact"] = "sentence_transformers"
    model: str = "Qwen/Qwen3-Embedding-0.6B"
    device: str = "cpu"
    local_files_only: bool = True
    page_vector_cache: bool = True
    max_embedding_page_chars: int = 360
    max_embedding_query_chars: int = 700
    top_k: int = 5
    cache_dir: str = "~/.llmwiki/cache/embeddings"
    strong_score: float = 0.78
    medium_score: float = 0.62
    max_excerpt_chars: int = 1200


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
    json_repair_applied: bool = False
    schema_valid: bool = False
    repair_attempted: bool = False
    latency_ms: int = 0
    payload_char_count: int = 0
    http_attempt_count: int = 1
    cost_usd: float | None = None
    errors: list[str] = Field(default_factory=list)


class StructuredIssue(StrictModel):
    issue_code: str
    field_path: str = ""
    validator_id: str = ""
    message: str
    repairability: Literal["repairable", "non_repairable"] = "non_repairable"


class StructuredAttemptRef(StrictModel):
    attempt: int
    provider_result_ref: str
    issues: list[StructuredIssue] = Field(default_factory=list)
    repair_prompt_ref: str | None = None
    parse_success: bool = False
    schema_valid: bool = False
    duration_ms: int = 0


class StructuredRepairReport(StrictModel):
    schema_version: Literal["structured_repair_report.v1"] = "structured_repair_report.v1"
    task: str
    provider: str
    final_outcome: Literal["success", "failed"] = "success"
    repair_attempted: bool = False
    max_repair_attempts: int = 0
    attempt_count: int = 0
    repair_count: int = 0
    duration_ms: int = 0
    attempts: list[StructuredAttemptRef] = Field(default_factory=list)
    final_provider_result_ref: str = ""
    non_repairable_issues: list[StructuredIssue] = Field(default_factory=list)


class RawPreparationUncertainItem(StrictModel):
    item: str
    reason: str = ""
    severity: Literal["low", "medium", "high"] = "medium"


class RawPreparationArtifact(StrictModel):
    schema_version: Literal["raw_preparation.v1"] = "raw_preparation.v1"
    source_raw_path: str
    input_raw_sha256: str = ""
    raw_link_cleanup_ref: str = ""
    document_kind: Literal["transcript", "article", "notes", "mixed", "unknown"] = "unknown"
    prepared_markdown: str
    operations_applied: list[str] = Field(default_factory=list)
    omission_policy: str = "non_content_noise_only"
    uncertain_items: list[RawPreparationUncertainItem] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high"] = "medium"
    requires_human_review: bool = True
    review_notes: str = ""


class RawLinkCleanupLink(StrictModel):
    link_id: str
    link_kind: Literal["wikilink"]
    label: str
    target: str
    cleanup_action: Literal["unwrap_text"]
    cleanup_context: Literal["frontmatter", "body"]
    line_number: int
    original_line_hash: str
    line_excerpt: str


class RawLinkCleanupWarning(StrictModel):
    warning_type: Literal["preserved_media_embed"]
    message: str
    line_number: int
    line_excerpt: str


class RawLinkCleanupArtifact(StrictModel):
    schema_version: Literal["raw_link_cleanup.v1"] = "raw_link_cleanup.v1"
    cleanup_rule_version: str = "obsidian_text_wikilink.v1"
    raw_path: str
    changed: bool
    pre_cleanup_sha256: str
    post_cleanup_sha256: str
    cleaned_link_count: int = 0
    preserved_media_embed_count: int = 0
    links: list[RawLinkCleanupLink] = Field(default_factory=list)
    warnings: list[RawLinkCleanupWarning] = Field(default_factory=list)


class ReviewDecision(StrictModel):
    schema_version: Literal["review_decision.v1"] = "review_decision.v1"
    review_step: str
    decision: Literal["approved", "revised", "pending", "rejected"] = "approved"
    review_mode: Literal["auto_stub", "manual"] = "manual"
    auto_approved: bool = False
    revision: int = 1
    feedback_count: int = 0
    notes: str = ""


class SourceDigestCandidate(StrictModel):
    candidate_id: str
    name: str
    type: str
    one_sentence_summary: str
    why_matters: str
    wiki_value: str
    source_locator: str = ""
    suggested_page_title: str = ""
    related_candidates: list[str] = Field(default_factory=list)
    resolution_hint: str = ""
    duplicate_risk: Literal["low", "medium", "high"] = "low"
    open_question_or_tension: str = ""


class WeakOrNoiseItem(StrictModel):
    candidate_id: str
    name: str
    type: str = "noise"
    one_sentence_summary: str
    why_matters: str = ""
    wiki_value: str = ""
    source_locator: str = ""
    suggested_page_title: str = ""
    suggested_action: Literal["ignore"] = "ignore"
    related_candidates: list[str] = Field(default_factory=list)
    resolution_hint: str = ""
    duplicate_risk: Literal["low", "medium", "high"] = "low"
    open_question_or_tension: str = ""


class SourceDigestArtifact(StrictModel):
    schema_version: Literal["source_digest.v2"] = "source_digest.v2"
    source_raw_path: str
    summary: str
    key_takeaways: list[str] = Field(default_factory=list)
    entities: list[SourceDigestCandidate] = Field(default_factory=list)
    concepts: list[SourceDigestCandidate] = Field(default_factory=list)
    designs: list[SourceDigestCandidate] = Field(default_factory=list)
    comparisons: list[SourceDigestCandidate] = Field(default_factory=list)
    open_questions: list[SourceDigestCandidate] = Field(default_factory=list)
    budget_deferred_candidates: list[SourceDigestCandidate] = Field(default_factory=list)
    weak_or_noise_items: list[WeakOrNoiseItem] = Field(default_factory=list)

    def ingest_candidates(self) -> list[SourceDigestCandidate]:
        return [*self.entities, *self.concepts, *self.designs, *self.comparisons, *self.open_questions]


class SourceBasis(StrictModel):
    source_candidate_ids: list[str] = Field(default_factory=list)
    prepared_discovered_candidates: list[str] = Field(default_factory=list)
    source_locator: str = ""

    @field_validator("source_candidate_ids", "prepared_discovered_candidates")
    @classmethod
    def normalize_candidate_refs(cls, value: list[str]) -> list[str]:
        refs: list[str] = []
        for item in value:
            ref = str(item).strip()
            if ref and ref not in refs:
                refs.append(ref)
        return refs


class CandidateResolutionItem(StrictModel):
    page_plan_id: str = ""
    source_basis: SourceBasis
    page_type: str
    display_title: str
    path_stem: str = ""
    candidate_target_path: str = ""
    topic_summary: str
    why_this_page: str
    initial_section_intent: str = ""
    coverage_notes: str = ""
    reason: str


class RelatedPageRef(StrictModel):
    target_path: str
    display_title: str
    source: Literal["source_digest", "wiki_context"]
    reason: str


class ContextOverlapSignal(StrictModel):
    strength: Literal["none", "weak", "medium", "strong"] = "none"
    match_basis: str = ""
    path: str = ""
    score: float = 0.0
    reason: str = ""


class WikiMergePlanItem(StrictModel):
    page_plan_id: str
    source_basis: SourceBasis
    action: Literal["create", "update", "noop", "needs_human_decision"]
    model_action: Literal["create", "update", "noop", "needs_human_decision"] | None = None
    finalization_reason: str = ""
    canonical_target_path: str
    display_title: str
    page_type: str
    matched_page: str | None = None
    inspected_context_paths: list[str] = Field(default_factory=list)
    strongest_overlap: ContextOverlapSignal = Field(default_factory=ContextOverlapSignal)
    why_not_update: str = ""
    why_create_or_update: str = ""
    related_absence_reason: Literal[
        "no_candidate",
        "only_source_or_system",
        "self_link_only",
        "low_confidence",
        "cap_cutoff",
    ] | None = None
    prior_knowledge_state: str = ""
    new_understanding: str
    changed_view: str = ""
    knowledge_delta: str = ""
    why_this_matters: str = ""
    reuse_scenarios: list[str] = Field(default_factory=list)
    value_points: list[str] = Field(default_factory=list)
    section_plans: dict[str, str]
    related_pages: list[RelatedPageRef] = Field(default_factory=list)
    related_unresolved: list[str] = Field(default_factory=list)
    unresolved_related: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    quality_risks: list[str] = Field(default_factory=list)
    apply_eligibility: Literal["applyable", "blocked", "source_only"] = "applyable"
    blocked_reason: str = ""
    reason: str
    merged_page_plan_ids: list[str] = Field(default_factory=list)
    noop_covered_by_update: bool = False
    merge_reason: str = ""


class CandidateResolutionArtifact(StrictModel):
    schema_version: Literal["candidate_resolution.v3"] = "candidate_resolution.v3"
    items: list[CandidateResolutionItem]
    missed_candidate_risks: list[str] = Field(default_factory=list)


class WikiPageMetadata(StrictModel):
    path: str
    llmwiki_type: str
    title: str
    summary: str
    created: str = ""
    updated: str
    aliases: list[str] = Field(default_factory=list)
    source_raw_paths: list[str] = Field(default_factory=list)
    source_raw_hashes: list[str] = Field(default_factory=list)
    source_prepared_hashes: list[str] = Field(default_factory=list)
    source_operation_ids: list[str] = Field(default_factory=list)


class WikiKnowledgePoolEntry(StrictModel):
    path: str
    rel_path: str
    preimage_sha256: str
    metadata: WikiPageMetadata | None = None
    display_title: str
    summary: str = ""
    aliases: list[str] = Field(default_factory=list)
    llmwiki_type: str = "unknown"
    indexable: bool = True
    unindexable_reason: str = ""


class WikiContextEntry(StrictModel):
    path: str
    expected_state: Literal["present", "missing"]
    preimage_sha256: str | None = None
    content: str = ""
    metadata: WikiPageMetadata | None = None

    @property
    def missing(self) -> bool:
        return self.expected_state == "missing"

    @property
    def sha256(self) -> str | None:
        return self.preimage_sha256


class WikiContextSnapshot(StrictModel):
    schema_version: Literal["wiki_context_snapshot.v2"] = "wiki_context_snapshot.v2"
    log_date: str
    source_target_path: str
    candidate_contexts_ref: str = ""
    candidate_pool_sha256: str = ""
    knowledge_metadata_pool: list[WikiKnowledgePoolEntry] = Field(default_factory=list)
    candidate_contexts: "CandidateContextsArtifact" = Field(default_factory=lambda: CandidateContextsArtifact(retrieval_backend="exact"))
    entries: list[WikiContextEntry] = Field(default_factory=list)


class CandidateContextHit(StrictModel):
    page_plan_id: str
    rank: int
    path: str
    display_title: str
    score: float = 0.0
    score_bucket: int = 0
    strength: Literal["weak", "medium", "strong"] = "weak"
    match_basis: str = ""
    sort_explanation: str = ""
    forced: bool = False
    page_sha256: str
    excerpt: str = ""
    truncated: bool = False


class CandidateContextItem(StrictModel):
    page_plan_id: str
    query: str
    hits: list[CandidateContextHit] = Field(default_factory=list)
    unindexable_pages: list[str] = Field(default_factory=list)


class CandidateContextsArtifact(StrictModel):
    schema_version: Literal["candidate_contexts.v1"] = "candidate_contexts.v1"
    retrieval_backend: str
    model: str = ""
    model_revision: str = ""
    local_files_only: bool = False
    cache_dir: str = ""
    embedding_load_duration_ms: int = 0
    embedding_encode_duration_ms: int = 0
    embedding_total_duration_ms: int = 0
    embedding_page_vector_cache_hit: bool = False
    embedding_page_count: int = 0
    embedding_query_count: int = 0
    embedding_text_char_count: int = 0
    top_k: int = 5
    candidate_pool_size: int = 0
    candidate_pool_sha256: str = ""
    skipped_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    items: list[CandidateContextItem] = Field(default_factory=list)


class WikiMergePlanArtifact(StrictModel):
    schema_version: Literal["wiki_merge_plan.v5"] = "wiki_merge_plan.v5"
    log_date: str
    items: list[WikiMergePlanItem]
    context_snapshot_ref: str = ""


class SourceDuplicateGuardArtifact(StrictModel):
    schema_version: Literal["source_duplicate_guard.v1"] = "source_duplicate_guard.v1"
    source_raw_path: str
    source_raw_hash: str
    source_prepared_hash: str
    source_target_path: str
    status: Literal["clear", "source_duplicate", "source_revision_detected"]
    matched_source_page: str | None = None
    matched_raw_path: str | None = None
    matched_raw_hash: str | None = None
    matched_prepared_hash: str | None = None
    reason: str


class RawIngestCandidate(StrictModel):
    raw_path: str
    status: Literal["unprocessed", "changed", "duplicate_hash", "duplicate_url", "processed"]
    raw_sha256: str
    size_bytes: int
    mtime: str
    matched_by: Literal["none", "path", "hash", "url", "path_and_hash"] = "none"
    source_pages: list[str] = Field(default_factory=list)
    operation_ids: list[str] = Field(default_factory=list)
    reason: str


class RawIngestCandidateReport(StrictModel):
    schema_version: Literal["raw_ingest_candidates.v1"] = "raw_ingest_candidates.v1"
    vault: str
    raw_root: str
    include_processed: bool = False
    limit: int | None = None
    total_raw_files: int = 0
    candidate_count: int = 0
    processed_count: int = 0
    changed_count: int = 0
    duplicate_hash_count: int = 0
    duplicate_url_count: int = 0
    unprocessed_count: int = 0
    items: list[RawIngestCandidate] = Field(default_factory=list)


class DraftPageItem(StrictModel):
    page_plan_id: str
    action: Literal["create", "update"]
    canonical_target_path: str
    preimage_sha256: str | None = None
    summary: str = ""
    body_markdown: str = ""
    open_questions: str = ""
    change_summary: str
    source_coverage_notes: str = ""
    quality_risks: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def coerce_page_values(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if "quality_risks" in data:
            data["quality_risks"] = _coerce_string_list(data["quality_risks"])
        for field_name in ["summary", "body_markdown", "open_questions"]:
            if field_name in data:
                data[field_name] = _coerce_section_body(data[field_name])
        return data


def _coerce_section_body(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        lines: list[str] = []
        for item in value:
            item_text = _coerce_section_body_scalar(item).strip()
            if item_text:
                lines.append(f"- {item_text}")
        return "\n".join(lines)
    return _coerce_section_body_scalar(value)


def _coerce_section_body_scalar(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            stripped = _coerce_section_body_scalar(item).strip()
            if stripped:
                items.append(stripped)
        return items
    stripped = _coerce_section_body_scalar(value).strip()
    return [stripped] if stripped else []


class DraftRenderingArtifact(StrictModel):
    schema_version: Literal["draft_rendering.v3"] = "draft_rendering.v3"
    pages: list[DraftPageItem] = Field(default_factory=list)


class SectionMergeChange(StrictModel):
    section_key: str
    retained: list[str] = Field(default_factory=list)
    added: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    preserved_old: list[str] = Field(default_factory=list)
    needs_manual_resolution: bool = False
    removal_reason: str = ""


class UpdatePageMergeReport(StrictModel):
    page_plan_id: str
    target_path: str
    old_title: str = ""
    final_title: str = ""
    model_title: str = ""
    retained_title: bool = True
    merged_page_plan_ids: list[str] = Field(default_factory=list)
    noop_covered_by_update: bool = False
    sections: list[SectionMergeChange] = Field(default_factory=list)


class UpdateMergeReport(StrictModel):
    schema_version: Literal["update_merge_report.v1"] = "update_merge_report.v1"
    pages: list[UpdatePageMergeReport] = Field(default_factory=list)


class GroundingClaim(StrictModel):
    page_plan_id: str
    target_path: str
    section_key: str = ""
    claim_type: Literal["new_fact", "retained_fact", "inference", "needs_source"]
    text: str
    support: Literal["raw", "wiki_context", "existing_wiki", "inference", "unsupported"] = "unsupported"
    action: Literal["kept", "moved_to_open_questions", "removed", "warn", "needs_review"] = "kept"
    reason: str = ""


class DraftGroundingReview(StrictModel):
    schema_version: Literal["draft_grounding_review.v1"] = "draft_grounding_review.v1"
    unsupported_new_facts: list[GroundingClaim] = Field(default_factory=list)
    warnings: list[GroundingClaim] = Field(default_factory=list)
    claims: list[GroundingClaim] = Field(default_factory=list)
    requires_review: bool = False


class RelatedCandidateReport(StrictModel):
    page_plan_id: str
    target_path: str
    display_title: str
    reason: str = ""
    source: str = ""
    decision: Literal["kept", "filtered", "cutoff"] = "kept"
    reject_reason: str = ""


class RelatedMergeReport(StrictModel):
    schema_version: Literal["related_merge_report.v1"] = "related_merge_report.v1"
    candidates: list[RelatedCandidateReport] = Field(default_factory=list)


class DraftWriteTarget(StrictModel):
    action: Literal["create", "update", "source", "index", "global_log", "daily_log"]
    target_path: str
    draft_path: str
    expected_state: Literal["present", "missing"]
    preimage_sha256: str | None = None
    page_plan_id: str | None = None


class DraftWriteManifest(StrictModel):
    schema_version: Literal["draft_write_manifest.v1"] = "draft_write_manifest.v1"
    targets: list[DraftWriteTarget]
    has_updates: bool = False
    has_noops: bool = False
    source_only_noop: bool = False
    requires_grounding_review: bool = False


class DraftApproval(StrictModel):
    schema_version: Literal["draft_review.v1"] = "draft_review.v1"
    review_step: str = "draft_review"
    decision: Literal["approved", "pending", "rejected"] = "approved"
    review_mode: Literal["auto_stub", "manual", "not_required"] = "manual"
    auto_approved: bool = False
    approved_draft_json_sha256: str | None = None
    approved_markdown_sha256: dict[str, str] = Field(default_factory=dict)
    notes: str = ""


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
    max_retries: int | None = None
    retry_backoff_seconds: float | None = None


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
    duration_ms: int | None = None
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
    review_reason: str | None = None
    review_state: Literal["none", "awaiting", "approved", "rejected"] = "none"
    awaiting_since: str | None = None
    resolved_at: str | None = None
    review_decision_ref: str | None = None


class OperationManifest(StrictModel):
    schema_version: Literal["operation_manifest.v8"] = "operation_manifest.v8"
    operation_id: str
    operation_type: str
    run_mode: RunMode = RunMode.dev
    engine_version: str
    profile: str
    profile_version: str = "1"
    vault_config_snapshot: VaultConfig = Field(default_factory=VaultConfig)
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
    action: Literal["create", "update", "source", "index", "global_log", "daily_log"] = "create"
    expected_state: Literal["present", "missing"] = "missing"
    preimage_sha256: str | None = None
    current_sha256: str | None = None
    will_write: bool = True
    blocked_reason: str = ""
    approved_draft_ref: str | None = None
    page_plan_id: str | None = None


class ApplyPreview(StrictModel):
    schema_version: Literal["apply_preview.v2"] = "apply_preview.v2"
    operation_id: str
    operation_applyable: bool = True
    requires_draft_review: bool = False
    has_updates: bool = False
    has_noops: bool = False
    blocked_reasons: list[str] = Field(default_factory=list)
    write_set_sha256: str
    targets: list[ApplyTarget]
    source_targets: list[str] = Field(default_factory=list)
    log_targets: list[str] = Field(default_factory=list)
    index_targets: list[str] = Field(default_factory=list)


class AppliedReceipt(StrictModel):
    schema_version: Literal["applied_receipt.v1"] = "applied_receipt.v1"
    operation_id: str
    applied_at: str = Field(default_factory=utc_now)
    raw_bindings: list[RawBinding]
    raw_cleanup_pre_sha256: str = ""
    raw_cleanup_post_sha256: str = ""
    raw_cleanup_rule_version: str = ""
    raw_cleanup_artifact_ref: str = ""
    raw_cleanup_changed: bool = False
    raw_cleanup_cleaned_link_count: int = 0
    raw_cleanup_diff_ref: str = ""
    prepared_raw: ArtifactRef | None = None
    raw_preparation: ArtifactRef | None = None
    written_pages: list[ArtifactRef]
    written_targets: list[str] = Field(default_factory=list)
    profile: str
    profile_version: str
    engine_version: str
