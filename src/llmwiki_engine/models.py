from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


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
    why_matters: str = Field(default="", validation_alias=AliasChoices("why_matters", "why_matches"))
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
    weak_or_noise_items: list[WeakOrNoiseItem] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def strip_legacy_suggested_action(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        if value.get("schema_version") == "source_digest.v1":
            raise ValueError("source_digest.v1 is incompatible with current MVP pipeline; rerun ingest")
        data = dict(value)
        for group_name in ["entities", "concepts", "designs", "comparisons", "open_questions"]:
            group = data.get(group_name)
            if not isinstance(group, list):
                continue
            cleaned_group = []
            for item in group:
                if isinstance(item, dict) and "suggested_action" in item:
                    item = dict(item)
                    item.pop("suggested_action", None)
                cleaned_group.append(item)
            data[group_name] = cleaned_group
        return data

    def ingest_candidates(self) -> list[SourceDigestCandidate]:
        return [*self.entities, *self.concepts, *self.designs, *self.comparisons, *self.open_questions]


class SourceBasis(StrictModel):
    source_candidate_ids: list[str] = Field(default_factory=list)
    prepared_discovered_candidates: list[str] = Field(default_factory=list)
    source_locator: str = ""


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


class WikiMergePlanItem(StrictModel):
    page_plan_id: str
    source_basis: SourceBasis
    action: Literal["create", "update", "noop", "needs_human_decision"]
    canonical_target_path: str
    display_title: str
    page_type: str
    matched_page: str | None = None
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
    schema_version: Literal["wiki_context_snapshot.v1"] = "wiki_context_snapshot.v1"
    log_date: str
    source_target_path: str
    entries: list[WikiContextEntry] = Field(default_factory=list)


class WikiMergePlanArtifact(StrictModel):
    schema_version: Literal["wiki_merge_plan.v4"] = "wiki_merge_plan.v4"
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


class DraftPageItem(StrictModel):
    page_plan_id: str
    action: Literal["create", "update"]
    canonical_target_path: str
    preimage_sha256: str | None = None
    section_bodies: dict[str, str]
    change_summary: str
    source_coverage_notes: str = ""
    quality_risks: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def coerce_section_body_values(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        section_bodies = data.get("section_bodies")
        if not isinstance(section_bodies, dict):
            return data
        normalized = {str(key): _coerce_section_body(value) for key, value in section_bodies.items()}
        return {**data, "section_bodies": normalized}


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


class DraftRenderingArtifact(StrictModel):
    schema_version: Literal["draft_rendering.v2"] = "draft_rendering.v2"
    pages: list[DraftPageItem] = Field(default_factory=list)


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


class OperationManifest(StrictModel):
    schema_version: Literal["operation_manifest.v6"] = "operation_manifest.v6"
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
