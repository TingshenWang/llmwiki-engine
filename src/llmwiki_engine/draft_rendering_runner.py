from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, TypeVar

from pydantic import BaseModel

from . import draft_grounding as _draft_grounding
from . import draft_rendering_payloads as _draft_rendering_payloads
from . import draft_validation as _draft_validation
from . import run_metrics as _run_metrics
from . import update_preservation as _update_preservation
from .events import format_duration
from .io import read_json, read_model, write_json
from .models import (
    DraftPageItem,
    DraftRenderingArtifact,
    ProviderResult,
    SourceDigestArtifact,
    StructuredAttemptRef,
    StructuredIssue,
    StructuredRepairReport,
    WikiContextSnapshot,
    WikiMergePlanArtifact,
)
from .provider_config import ProviderExecutionContext
from .providers import Provider
from .structured import StructuredModelCall, parse_structured_json_object, render_structured_repair_report_markdown
from .system_pages import format_markdown_table
from .validators import ValidationError as ContractValidationError
from .wiki_context import snapshot_entry


DRAFT_RENDERING_BATCH_PAGE_LIMIT = 4
DRAFT_RENDERING_MAX_PARALLEL_BATCHES = 3


@dataclass(frozen=True)
class DraftRenderingRunContext:
    execution_context: ProviderExecutionContext
    profile_payload: dict[str, Any]
    language_contract: dict[str, Any]
    wiki_language: str


TModel = TypeVar("TModel", bound=BaseModel)


def redacted_model(ctx: DraftRenderingRunContext, model: TModel, model_type: type[TModel]) -> TModel:
    data = ctx.execution_context.redactor.redact(model.model_dump(mode="json"))
    return model_type.model_validate(data)


def write_draft_aux_report_if_active(
    *,
    output_dir: Path,
    stem: str,
    report: dict[str, Any],
    renderer: Any,
    count_keys: list[str],
) -> tuple[Path, Path] | None:
    has_activity = bool(report.get("changed")) or bool(report.get("pages"))
    for key in count_keys:
        try:
            has_activity = has_activity or int(report.get(key, 0)) > 0
        except (TypeError, ValueError):
            continue
    if not has_activity:
        return None
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    write_json(json_path, report)
    md_path.write_text(renderer(report), encoding="utf-8")
    return json_path, md_path


def run_draft_rendering_model(
    *,
    ctx: DraftRenderingRunContext,
    step_root: Path,
    digest: SourceDigestArtifact,
    merge_plan: WikiMergePlanArtifact,
    snapshot: WikiContextSnapshot,
    source_excerpt_pack: dict[str, Any],
    update_preservation_pack: dict[str, Any],
    approved_prepared_text: str,
) -> DraftRenderingArtifact:
    draftable_items = [item for item in merge_plan.items if item.action in {"create", "update"}]
    provider = ctx.execution_context.provider_for_task("draft_rendering")
    if len(draftable_items) <= DRAFT_RENDERING_BATCH_PAGE_LIMIT:
        return run_single_draft_rendering_model_call(
            ctx=ctx,
            provider=provider,
            output_dir=step_root,
            digest=digest,
            merge_plan=merge_plan,
            snapshot=snapshot,
            source_excerpt_pack=source_excerpt_pack,
            update_preservation_pack=update_preservation_pack,
            approved_prepared_text=approved_prepared_text,
        )

    batch_items_list = [
        draftable_items[index : index + DRAFT_RENDERING_BATCH_PAGE_LIMIT]
        for index in range(0, len(draftable_items), DRAFT_RENDERING_BATCH_PAGE_LIMIT)
    ]
    provider_spec = ctx.execution_context.runtime_for_task("draft_rendering").spec
    max_parallel_batches = (
        min(DRAFT_RENDERING_MAX_PARALLEL_BATCHES, len(batch_items_list))
        if provider_spec and provider_spec.startswith("openai_compatible:")
        else 1
    )
    parallel = max_parallel_batches > 1
    batch_jobs: list[dict[str, Any]] = []
    batch_root = step_root / "model_batches"
    for index, batch_items in enumerate(batch_items_list, start=1):
        batch_id = f"batch-{index:03d}"
        batch_dir = batch_root / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        batch_plan = merge_plan.model_copy(update={"items": list(batch_items)})
        batch_source_excerpt_pack = _draft_rendering_payloads.build_draft_source_excerpt_pack(
            approved_prepared_text,
            digest,
            batch_plan,
            force_excerpt=True,
        )
        batch_source_pack_path = batch_dir / "draft_source_excerpt_pack.json"
        batch_source_pack_md = batch_dir / "draft_source_excerpt_pack.md"
        write_json(batch_source_pack_path, batch_source_excerpt_pack)
        batch_source_pack_md.write_text(_draft_rendering_payloads.render_draft_source_excerpt_pack_markdown(batch_source_excerpt_pack), encoding="utf-8")
        batch_update_preservation_pack = _update_preservation.build_update_preservation_pack(batch_plan, snapshot)
        batch_update_pack_path = batch_dir / "update_preservation_pack.json"
        batch_update_pack_md = batch_dir / "update_preservation_pack.md"
        write_json(batch_update_pack_path, batch_update_preservation_pack)
        batch_update_pack_md.write_text(_update_preservation.render_update_preservation_pack_markdown(batch_update_preservation_pack), encoding="utf-8")
        batch_jobs.append(
            {
                "index": index,
                "batch_id": batch_id,
                "batch_items": batch_items,
                "batch_dir": batch_dir,
                "batch_plan": batch_plan,
                "source_excerpt_pack": batch_source_excerpt_pack,
                "update_preservation_pack": batch_update_preservation_pack,
            }
        )

    def run_batch(job: dict[str, Any]) -> dict[str, Any]:
        batch_provider = ctx.execution_context.provider_for_task("draft_rendering") if parallel else provider
        batch_artifact = run_single_draft_rendering_model_call(
            ctx=ctx,
            provider=batch_provider,
            output_dir=job["batch_dir"],
            digest=digest,
            merge_plan=job["batch_plan"],
            snapshot=snapshot,
            source_excerpt_pack=job["source_excerpt_pack"],
            update_preservation_pack=job["update_preservation_pack"],
            approved_prepared_text=approved_prepared_text,
        )
        report = read_model(job["batch_dir"] / "structured_repair_report.json", StructuredRepairReport)
        result = read_model(job["batch_dir"] / "provider_result.json", ProviderResult)
        reinforcement_path = job["batch_dir"] / "update_preservation_reinforcement_report.json"
        reinforcement_report = read_json(reinforcement_path) if reinforcement_path.exists() else {}
        grounding_rewrite_path = job["batch_dir"] / "grounding_paraphrase_rewrite_report.json"
        grounding_rewrite_report = read_json(grounding_rewrite_path) if grounding_rewrite_path.exists() else {}
        batch_payload_char_count = _run_metrics.provider_results_payload_char_count(
            [job["batch_dir"] / attempt.provider_result_ref for attempt in report.attempts]
        )
        batch_http_attempt_count = _run_metrics.provider_results_http_attempt_count(
            [job["batch_dir"] / attempt.provider_result_ref for attempt in report.attempts]
        )
        batch_items = job["batch_items"]
        batch_id = job["batch_id"]
        return {
            "index": job["index"],
            "artifact": batch_artifact,
            "summary": {
                "batch_id": batch_id,
                "page_plan_ids": [item.page_plan_id for item in batch_items],
                "target_paths": [item.canonical_target_path for item in batch_items],
                "source_excerpt_chars": job["source_excerpt_pack"].get("included_char_count", 0),
                "payload_char_count": batch_payload_char_count,
                "http_attempt_count": batch_http_attempt_count,
                "attempt_count": report.attempt_count,
                "repair_count": report.repair_count,
                "duration_ms": report.duration_ms,
                "provider": report.provider,
                "provider_result_ref": f"model_batches/{batch_id}/provider_result.json",
                "structured_repair_report_ref": f"model_batches/{batch_id}/structured_repair_report.json",
                "update_preservation_reinforcement_report_ref": (
                    f"model_batches/{batch_id}/update_preservation_reinforcement_report.json"
                    if reinforcement_path.exists()
                    else ""
                ),
                "reinforced_section_count": int(reinforcement_report.get("reinforced_section_count", 0)),
                "grounding_paraphrase_rewrite_report_ref": (
                    f"model_batches/{batch_id}/grounding_paraphrase_rewrite_report.json"
                    if grounding_rewrite_path.exists()
                    else ""
                ),
                "grounding_rewrite_count": int(grounding_rewrite_report.get("rewrite_count", 0)),
                "schema_valid": result.schema_valid,
            },
        }

    started = perf_counter()
    if parallel:
        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_parallel_batches) as executor:
            futures = [executor.submit(run_batch, job) for job in batch_jobs]
            for future in as_completed(futures):
                results.append(future.result())
    else:
        results = [run_batch(job) for job in batch_jobs]
    wall_duration_ms = round((perf_counter() - started) * 1000)

    pages: list[DraftPageItem] = []
    batch_summaries: list[dict[str, Any]] = []
    for result in sorted(results, key=lambda item: int(item["index"])):
        batch_artifact = result["artifact"]
        pages.extend(batch_artifact.pages)
        batch_summaries.append(result["summary"])
    draft_artifact = finalize_draft_rendering(DraftRenderingArtifact(pages=pages), merge_plan, snapshot)
    write_draft_rendering_batch_reports(
        step_root,
        draft_artifact,
        batch_summaries,
        max_parallel_batches=max_parallel_batches,
        wall_duration_ms=wall_duration_ms,
    )
    return draft_artifact

PAGE_SCOPED_DRAFT_REPAIR_ISSUE_CODES = {
    "unsupported_new_fact",
    "model_self_talk_leak",
    "stray_related_links_in_content",
    "forbidden_system_section_in_core",
    "thin_digestive_content",
    "old_knowledge_not_absorbed",
    "missing_repair_page",
}


def build_draft_rendering_missing_page_repair_payload(
    *,
    task: str,
    raw: str,
    issues: list[StructuredIssue],
    output_model: type[BaseModel],
    ctx: DraftRenderingRunContext,
    digest: SourceDigestArtifact,
    merge_plan: WikiMergePlanArtifact,
    snapshot: WikiContextSnapshot,
    source_excerpt_pack: dict[str, Any],
    update_preservation_pack: dict[str, Any],
    approved_prepared_text: str,
) -> dict[str, Any] | None:
    if not issues or any(issue.issue_code != "missing_page_plan_coverage" for issue in issues):
        return None
    partial = extract_valid_partial_draft_rendering(
        raw,
        merge_plan,
        snapshot,
        update_preservation_pack=update_preservation_pack,
        approved_prepared_text=approved_prepared_text,
        language=ctx.wiki_language,
    )
    if partial is None or not partial.pages:
        return None
    present_ids = {page.page_plan_id for page in partial.pages}
    required_items = [item for item in merge_plan.items if item.action in {"create", "update"}]
    missing_items = [item for item in required_items if item.page_plan_id not in present_ids]
    if not missing_items:
        return None
    missing_plan = merge_plan.model_copy(update={"items": missing_items})
    missing_source_excerpt_pack = _draft_rendering_payloads.build_draft_source_excerpt_pack(
        approved_prepared_text,
        digest,
        missing_plan,
        force_excerpt=True,
    )
    missing_update_preservation_pack = _update_preservation.build_update_preservation_pack(missing_plan, snapshot)
    missing_payload = _draft_rendering_payloads.build_draft_rendering_payload(
        digest=digest,
        merge_plan=missing_plan,
        snapshot=snapshot,
        source_excerpt_pack=missing_source_excerpt_pack,
        update_preservation_pack=missing_update_preservation_pack,
        approved_prepared_text=approved_prepared_text,
        profile_payload=ctx.profile_payload,
        language_contract=ctx.language_contract,
        grounding_risk_rules=list(_draft_grounding.DRAFT_RENDERING_GROUNDING_RISK_RULES),
    )
    return {
        "repair_contract": {
            "goal": "Complete a partial draft_rendering output without regenerating pages that already passed local validation.",
            "mode": "missing_page_completion",
            "task": task,
            "rules": [
                "Return only one complete JSON object matching the schema.",
                "The pages array must contain every accepted_partial_pages item unchanged plus exactly one generated page for each missing_page_plan_id.",
                "Do not regenerate, rewrite, remove, or reorder accepted_partial_pages; copy them into pages exactly as provided.",
                "Generate only the missing pages from missing_page_payload; do not create pages outside missing_page_plan_ids.",
                "All user-visible generated content must follow the language and grounding rules inside missing_page_payload.",
            ],
            "issues": [issue.model_dump(mode="json") for issue in issues],
            "accepted_page_plan_ids": [page.page_plan_id for page in partial.pages],
            "missing_page_plan_ids": [item.page_plan_id for item in missing_items],
            "required_page_plan_ids": [item.page_plan_id for item in required_items],
            "schema": output_model.model_json_schema(),
        },
        "accepted_partial_pages": [page.model_dump(mode="json") for page in partial.pages],
        "missing_page_payload": missing_payload,
        "source_excerpt_pack_omitted_reason": (
            "The original batch payload is intentionally replaced by a compact missing_page_payload; "
            f"previous full/pack source refs remain {source_excerpt_pack.get('approved_prepared_ref', '')}."
        ),
    }


def build_draft_rendering_page_repair_payload(
    *,
    task: str,
    raw: str,
    issues: list[StructuredIssue],
    output_model: type[BaseModel],
    ctx: DraftRenderingRunContext,
    digest: SourceDigestArtifact,
    merge_plan: WikiMergePlanArtifact,
    snapshot: WikiContextSnapshot,
    source_excerpt_pack: dict[str, Any],
    update_preservation_pack: dict[str, Any],
    approved_prepared_text: str,
    accepted_partial_pages_override: list[DraftPageItem] | None = None,
    include_local_accepted_pages: bool = False,
) -> dict[str, Any] | None:
    failing_ids = draft_repair_page_plan_ids_from_issues(issues, merge_plan)
    if not failing_ids:
        return None
    required_items = [item for item in merge_plan.items if item.action in {"create", "update"}]
    required_ids = {item.page_plan_id for item in required_items}
    accepted_ids = required_ids - failing_ids
    if not accepted_ids or not failing_ids < required_ids:
        return None
    if accepted_partial_pages_override is not None:
        partial = DraftRenderingArtifact(
            pages=[page for page in accepted_partial_pages_override if page.page_plan_id in accepted_ids]
        )
    else:
        partial = extract_valid_partial_draft_rendering(
            raw,
            merge_plan,
            snapshot,
            accepted_page_plan_ids=accepted_ids,
            update_preservation_pack=update_preservation_pack,
            approved_prepared_text=approved_prepared_text,
            language=ctx.wiki_language,
        )
    if partial is None or not partial.pages:
        return None
    if {page.page_plan_id for page in partial.pages} != accepted_ids:
        return None
    repair_items = [item for item in required_items if item.page_plan_id in failing_ids]
    if not repair_items:
        return None
    repair_plan = merge_plan.model_copy(update={"items": repair_items})
    repair_source_excerpt_pack = _draft_rendering_payloads.build_draft_source_excerpt_pack(
        approved_prepared_text,
        digest,
        repair_plan,
        force_excerpt=True,
    )
    repair_update_preservation_pack = _update_preservation.build_update_preservation_pack(repair_plan, snapshot)
    repair_payload = _draft_rendering_payloads.build_draft_rendering_payload(
        digest=digest,
        merge_plan=repair_plan,
        snapshot=snapshot,
        source_excerpt_pack=repair_source_excerpt_pack,
        update_preservation_pack=repair_update_preservation_pack,
        approved_prepared_text=approved_prepared_text,
        profile_payload=ctx.profile_payload,
        language_contract=ctx.language_contract,
        grounding_risk_rules=list(_draft_grounding.DRAFT_RENDERING_GROUNDING_RISK_RULES),
    )
    accepted_page_refs = draft_repair_accepted_page_refs(partial.pages, merge_plan)
    prompt = {
        "repair_contract": {
            "goal": "Repair page-scoped draft_rendering issues without regenerating pages that already passed local validation.",
            "mode": "page_scoped_repair",
            "task": task,
            "rules": [
                "Return only one complete JSON object matching the schema.",
                "The pages array must contain only repaired pages for repair_page_plan_ids.",
                "Do not include accepted_page_refs pages in the output; they are retained locally and will be merged after repair.",
                "accepted_page_refs may be used only for lightweight cross-page boundary awareness; do not regenerate them.",
                "Generate exactly one repaired page for each repair_page_plan_id from repair_page_payload; do not create pages outside repair_page_plan_ids.",
                "All user-visible repaired content must follow the language and grounding rules inside repair_page_payload.",
            ],
            "issues": [issue.model_dump(mode="json") for issue in issues],
            "accepted_page_plan_ids": [page.page_plan_id for page in partial.pages],
            "repair_page_plan_ids": [item.page_plan_id for item in repair_items],
            "required_page_plan_ids": [item.page_plan_id for item in required_items],
            "schema": output_model.model_json_schema(),
        },
        "accepted_page_refs": accepted_page_refs,
        "repair_page_payload": repair_payload,
        "source_excerpt_pack_omitted_reason": (
            "The original batch payload is intentionally replaced by a compact repair_page_payload; "
            f"previous full/pack source refs remain {source_excerpt_pack.get('approved_prepared_ref', '')}."
        ),
    }
    if include_local_accepted_pages:
        prompt["_local_accepted_partial_pages"] = [page.model_dump(mode="json") for page in partial.pages]
    return prompt


def draft_repair_accepted_page_refs(
    accepted_pages: list[DraftPageItem],
    merge_plan: WikiMergePlanArtifact,
) -> list[dict[str, str]]:
    items_by_id = {item.page_plan_id: item for item in merge_plan.items}
    refs: list[dict[str, str]] = []
    for page in accepted_pages:
        item = items_by_id.get(page.page_plan_id)
        refs.append(
            {
                "page_plan_id": page.page_plan_id,
                "action": page.action,
                "target_path": page.canonical_target_path,
                "display_title": item.display_title if item else page.canonical_target_path,
                "page_type": item.page_type if item else "",
            }
        )
    return refs


def merge_repaired_draft_with_accepted_pages(
    repair_only: DraftRenderingArtifact,
    *,
    accepted_pages_by_id: dict[str, dict[str, Any]],
    repair_page_plan_ids: set[str],
    merge_plan: WikiMergePlanArtifact,
) -> DraftRenderingArtifact:
    repair_pages_by_id = {
        page.page_plan_id: page
        for page in repair_only.pages
        if page.page_plan_id in repair_page_plan_ids
    }
    pages_by_id: dict[str, DraftPageItem] = {
        page_plan_id: DraftPageItem.model_validate(page)
        for page_plan_id, page in accepted_pages_by_id.items()
    }
    pages_by_id.update(repair_pages_by_id)
    ordered_ids = [
        item.page_plan_id
        for item in merge_plan.items
        if item.action in {"create", "update"} and item.page_plan_id in pages_by_id
    ]
    return DraftRenderingArtifact(pages=[pages_by_id[page_plan_id] for page_plan_id in ordered_ids])


def preserve_active_repair_page_issues(
    issues: list[StructuredIssue],
    active_repair_page_plan_ids: set[str],
) -> list[StructuredIssue]:
    if not active_repair_page_plan_ids:
        return issues
    issue_page_ids: set[str] = set()
    for issue in issues:
        if issue.repairability != "repairable" or issue.issue_code not in PAGE_SCOPED_DRAFT_REPAIR_ISSUE_CODES:
            return issues
        match = re.match(r"^pages\.([^.]+)(?:\.|$)", issue.field_path or "")
        if not match:
            return issues
        issue_page_ids.add(match.group(1))
    expanded = list(issues)
    for page_plan_id in sorted(active_repair_page_plan_ids):
        if page_plan_id in issue_page_ids:
            continue
        expanded.append(
            StructuredIssue(
                issue_code="missing_repair_page",
                field_path=f"pages.{page_plan_id}",
                validator_id="draft_page_scoped_repair",
                message=(
                    f"Page-scoped draft repair must return repaired page `{page_plan_id}` together with "
                    "the other active repair pages; prior partial repair output is not accepted until the full repair set validates."
                ),
                repairability="repairable",
            )
        )
    return expanded


def draft_repair_page_plan_ids_from_issues(
    issues: list[StructuredIssue],
    merge_plan: WikiMergePlanArtifact,
) -> set[str] | None:
    if not issues:
        return None
    draftable_ids = {item.page_plan_id for item in merge_plan.items if item.action in {"create", "update"}}
    page_plan_ids: set[str] = set()
    for issue in issues:
        if issue.issue_code not in PAGE_SCOPED_DRAFT_REPAIR_ISSUE_CODES:
            return None
        match = re.match(r"^pages\.([^.]+)(?:\.|$)", issue.field_path or "")
        if not match:
            return None
        page_plan_id = match.group(1)
        if page_plan_id not in draftable_ids:
            return None
        page_plan_ids.add(page_plan_id)
    return page_plan_ids or None


def accepted_partial_page_copy_issues(
    draft: DraftRenderingArtifact,
    accepted_pages_by_id: dict[str, dict[str, Any]],
) -> list[StructuredIssue]:
    if not accepted_pages_by_id:
        return []
    pages_by_id = {page.page_plan_id: page for page in draft.pages}
    issues: list[StructuredIssue] = []
    for page_plan_id, expected in accepted_pages_by_id.items():
        page = pages_by_id.get(page_plan_id)
        if page is None or page.model_dump(mode="json") != expected:
            issues.append(
                StructuredIssue(
                    issue_code="accepted_partial_page_changed",
                    field_path=f"pages.{page_plan_id}",
                    validator_id="draft_page_scoped_repair",
                    message=(
                        f"Page-scoped draft repair changed accepted partial page `{page_plan_id}`; "
                        "copy accepted_partial_pages exactly and only repair failing page ids."
                    ),
                    repairability="repairable",
                )
            )
    return issues


def extract_valid_partial_draft_rendering(
    raw: str,
    merge_plan: WikiMergePlanArtifact,
    snapshot: WikiContextSnapshot,
    *,
    accepted_page_plan_ids: set[str] | None = None,
    update_preservation_pack: dict[str, Any],
    approved_prepared_text: str,
    language: str | None,
) -> DraftRenderingArtifact | None:
    try:
        parsed, _json_repaired = parse_structured_json_object(raw)
        artifact = DraftRenderingArtifact.model_validate(parsed)
        candidate = finalize_draft_rendering(artifact, merge_plan, snapshot)
    except Exception:
        return None
    if accepted_page_plan_ids is not None:
        existing_ids = {page.page_plan_id for page in candidate.pages}
        if not accepted_page_plan_ids <= existing_ids:
            return None
        candidate = DraftRenderingArtifact(
            pages=[page for page in candidate.pages if page.page_plan_id in accepted_page_plan_ids]
        )
    present_ids = {page.page_plan_id for page in candidate.pages}
    required_ids = {item.page_plan_id for item in merge_plan.items if item.action in {"create", "update"}}
    if not present_ids or not present_ids < required_ids:
        return None
    partial_plan = merge_plan.model_copy(
        update={
            "items": [
                item
                for item in merge_plan.items
                if item.action not in {"create", "update"} or item.page_plan_id in present_ids
            ]
        }
    )
    try:
        _draft_validation.validate_draft_rendering(candidate, partial_plan, language=language)
    except Exception:
        return None
    if _draft_validation.draft_self_talk_issues(candidate):
        return None
    if _update_preservation.update_preservation_issues(candidate, update_preservation_pack):
        return None
    candidate, _grounding_rewrite_report = _draft_grounding.rewrite_grounding_sensitive_paraphrases(candidate, approved_prepared_text)
    grounding_review = _draft_grounding.build_draft_grounding_review(candidate, partial_plan, snapshot, approved_prepared_text)
    if grounding_review.requires_review:
        return None
    return candidate


def run_single_draft_rendering_model_call(
    *,
    ctx: DraftRenderingRunContext,
    provider: Provider,
    output_dir: Path,
    digest: SourceDigestArtifact,
    merge_plan: WikiMergePlanArtifact,
    snapshot: WikiContextSnapshot,
    source_excerpt_pack: dict[str, Any],
    update_preservation_pack: dict[str, Any],
    approved_prepared_text: str,
) -> DraftRenderingArtifact:
    payload = _draft_rendering_payloads.build_draft_rendering_payload(
        digest=digest,
        merge_plan=merge_plan,
        snapshot=snapshot,
        source_excerpt_pack=source_excerpt_pack,
        update_preservation_pack=update_preservation_pack,
        approved_prepared_text=approved_prepared_text,
        profile_payload=ctx.profile_payload,
        language_contract=ctx.language_contract,
        grounding_risk_rules=list(_draft_grounding.DRAFT_RENDERING_GROUNDING_RISK_RULES),
    )
    accepted_repair_pages_by_id: dict[str, dict[str, Any]] = {}
    active_repair_page_plan_ids: set[str] = set()
    last_merged_repair_artifact: DraftRenderingArtifact | None = None

    def validate_draft_rendering_model(model: DraftRenderingArtifact) -> None:
        nonlocal last_merged_repair_artifact
        validation_model = model
        if accepted_repair_pages_by_id and active_repair_page_plan_ids:
            returned_repair_ids = {
                page.page_plan_id
                for page in model.pages
                if page.page_plan_id in active_repair_page_plan_ids
            }
            missing_repair_ids = active_repair_page_plan_ids - returned_repair_ids
            if missing_repair_ids:
                raise ContractValidationError(
                    "page-scoped draft repair omitted required repaired pages.",
                    issues=[
                        StructuredIssue(
                            issue_code="missing_repair_page",
                            field_path=f"pages.{page_plan_id}",
                            validator_id="draft_page_scoped_repair",
                            message=(
                                f"Page-scoped draft repair must return repaired page `{page_plan_id}`; "
                                "accepted pages are retained locally and should not be returned instead."
                            ),
                            repairability="repairable",
                        )
                        for page_plan_id in sorted(missing_repair_ids)
                    ],
                )
            validation_model = merge_repaired_draft_with_accepted_pages(
                validation_model,
                accepted_pages_by_id=accepted_repair_pages_by_id,
                repair_page_plan_ids=active_repair_page_plan_ids,
                merge_plan=merge_plan,
            )
            last_merged_repair_artifact = validation_model
        else:
            last_merged_repair_artifact = None
        candidate = finalize_draft_rendering(validation_model, merge_plan, snapshot)
        _draft_validation.validate_draft_rendering(candidate, merge_plan, language=ctx.wiki_language)
        repair_issues = _draft_validation.draft_self_talk_issues(candidate)
        repair_issues.extend(_update_preservation.update_preservation_issues(candidate, update_preservation_pack))
        if not active_repair_page_plan_ids:
            repair_issues.extend(accepted_partial_page_copy_issues(candidate, accepted_repair_pages_by_id))
        rewritten_candidate, _grounding_rewrite_report = _draft_grounding.rewrite_grounding_sensitive_paraphrases(
            candidate,
            approved_prepared_text,
        )
        grounding_review = _draft_grounding.build_draft_grounding_review(rewritten_candidate, merge_plan, snapshot, approved_prepared_text)
        if grounding_review.requires_review:
            repair_issues.extend(
                [
                    StructuredIssue(
                        issue_code="unsupported_new_fact",
                        field_path=f"pages.{claim.page_plan_id}.{claim.section_key}",
                        validator_id="draft_grounding_review",
                        message=_draft_grounding.grounding_issue_message(claim),
                        repairability="repairable",
                    )
                    for claim in grounding_review.unsupported_new_facts
                ]
            )
        if repair_issues:
            raise ContractValidationError(
                "draft_rendering contains repairable quality issues; remove model self-talk, preserve update obligations, and fix unsupported facts.",
                issues=repair_issues,
            )

    def build_repair_payload(
        task: str,
        _payload: dict[str, Any],
        raw: str,
        issues: list[StructuredIssue],
        output_model: type[BaseModel],
    ) -> dict[str, Any] | None:
        nonlocal accepted_repair_pages_by_id, active_repair_page_plan_ids
        accepted_override = [
            DraftPageItem.model_validate(page)
            for page in accepted_repair_pages_by_id.values()
        ] if accepted_repair_pages_by_id else None
        page_repair_issues = preserve_active_repair_page_issues(issues, active_repair_page_plan_ids)
        page_repair_payload = build_draft_rendering_page_repair_payload(
            task=task,
            raw=raw,
            issues=page_repair_issues,
            output_model=output_model,
            ctx=ctx,
            digest=digest,
            merge_plan=merge_plan,
            snapshot=snapshot,
            source_excerpt_pack=source_excerpt_pack,
            update_preservation_pack=update_preservation_pack,
            approved_prepared_text=approved_prepared_text,
            accepted_partial_pages_override=accepted_override,
            include_local_accepted_pages=True,
        )
        if page_repair_payload is not None:
            local_accepted_pages = page_repair_payload.pop("_local_accepted_partial_pages", [])
            accepted_repair_pages_by_id = {
                str(page.get("page_plan_id", "")): page
                for page in local_accepted_pages
                if isinstance(page, dict) and page.get("page_plan_id")
            }
            active_repair_page_plan_ids = set(page_repair_payload["repair_contract"].get("repair_page_plan_ids", []))
            return page_repair_payload
        accepted_repair_pages_by_id = {}
        active_repair_page_plan_ids = set()
        return build_draft_rendering_missing_page_repair_payload(
            task=task,
            raw=raw,
            issues=issues,
            output_model=output_model,
            ctx=ctx,
            digest=digest,
            merge_plan=merge_plan,
            snapshot=snapshot,
            source_excerpt_pack=source_excerpt_pack,
            update_preservation_pack=update_preservation_pack,
            approved_prepared_text=approved_prepared_text,
        )

    draft_artifact, _ = StructuredModelCall(
        provider,
        output_dir=output_dir,
        result_filename="provider_result.json",
        redactor=ctx.execution_context.redactor,
    ).run(
        "draft_rendering",
        payload,
        DraftRenderingArtifact,
        validator=validate_draft_rendering_model,
        accept_after_repair_issue_codes={"unsupported_new_fact", "old_knowledge_not_absorbed"},
        repair_payload_builder=build_repair_payload,
    )
    if last_merged_repair_artifact is not None and accepted_repair_pages_by_id and active_repair_page_plan_ids:
        draft_artifact = last_merged_repair_artifact
    draft_artifact = redacted_model(ctx, draft_artifact, DraftRenderingArtifact)
    draft_artifact = finalize_draft_rendering(draft_artifact, merge_plan, snapshot)
    draft_artifact, reinforcement_report = _update_preservation.reinforce_update_preservation(draft_artifact, update_preservation_pack)
    draft_artifact, grounding_rewrite_report = _draft_grounding.rewrite_grounding_sensitive_paraphrases(draft_artifact, approved_prepared_text)
    write_draft_aux_report_if_active(
        output_dir=output_dir,
        stem="update_preservation_reinforcement_report",
        report=reinforcement_report,
        renderer=_update_preservation.render_update_preservation_reinforcement_report,
        count_keys=["reinforced_page_count", "reinforced_section_count"],
    )
    write_draft_aux_report_if_active(
        output_dir=output_dir,
        stem="grounding_paraphrase_rewrite_report",
        report=grounding_rewrite_report,
        renderer=_draft_grounding.render_grounding_paraphrase_rewrite_report,
        count_keys=["rewrite_count"],
    )
    return draft_artifact


def write_draft_rendering_batch_reports(
    step_root: Path,
    draft_artifact: DraftRenderingArtifact,
    batch_summaries: list[dict[str, Any]],
    *,
    max_parallel_batches: int = 1,
    wall_duration_ms: int | None = None,
) -> None:
    model_duration_ms = sum(int(batch["duration_ms"]) for batch in batch_summaries)
    payload_counts = [int(batch.get("payload_char_count", 0)) for batch in batch_summaries]
    batch_report = {
        "schema_version": "draft_rendering_batch_report.v1",
        "batch_page_limit": DRAFT_RENDERING_BATCH_PAGE_LIMIT,
        "parallel": max_parallel_batches > 1,
        "max_parallel_batches": max_parallel_batches,
        "batch_count": len(batch_summaries),
        "page_count": len(draft_artifact.pages),
        "attempt_count": sum(int(batch["attempt_count"]) for batch in batch_summaries),
        "repair_count": sum(int(batch["repair_count"]) for batch in batch_summaries),
        "http_attempt_count": sum(int(batch.get("http_attempt_count", 0)) for batch in batch_summaries),
        "duration_ms": model_duration_ms,
        "model_duration_ms": model_duration_ms,
        "wall_duration_ms": wall_duration_ms if wall_duration_ms is not None else model_duration_ms,
        "payload_char_count": sum(payload_counts),
        "max_batch_payload_char_count": max(payload_counts) if payload_counts else 0,
        "avg_batch_payload_char_count": round(sum(payload_counts) / len(payload_counts)) if payload_counts else 0,
        "batches": batch_summaries,
    }
    write_json(step_root / "draft_rendering_batch_report.json", batch_report)
    (step_root / "draft_rendering_batch_report.md").write_text(render_draft_rendering_batch_report(batch_report), encoding="utf-8")
    providers = ",".join(sorted({str(batch["provider"]) for batch in batch_summaries}))
    write_json(
        step_root / "provider_result.json",
        ProviderResult(
            task="draft_rendering",
            provider=f"batched:{providers}",
            raw_output="",
            parsed_output=draft_artifact.model_dump(mode="json"),
            parse_success=True,
            schema_valid=True,
            repair_attempted=batch_report["repair_count"] > 0,
            latency_ms=batch_report["duration_ms"],
            payload_char_count=batch_report["payload_char_count"],
            http_attempt_count=batch_report["http_attempt_count"],
        ),
    )
    attempts: list[StructuredAttemptRef] = []
    next_attempt = 1
    non_repairable: list[StructuredIssue] = []
    max_repair_attempts = 0
    for batch in batch_summaries:
        report = read_model(step_root / str(batch["structured_repair_report_ref"]), StructuredRepairReport)
        max_repair_attempts = max(max_repair_attempts, report.max_repair_attempts)
        non_repairable.extend(report.non_repairable_issues)
        for attempt in report.attempts:
            attempts.append(
                attempt.model_copy(
                    update={
                        "attempt": next_attempt,
                        "provider_result_ref": f"model_batches/{batch['batch_id']}/{attempt.provider_result_ref}",
                        "repair_prompt_ref": (
                            f"model_batches/{batch['batch_id']}/{attempt.repair_prompt_ref}"
                            if attempt.repair_prompt_ref
                            else None
                        ),
                    }
                )
            )
            next_attempt += 1
    aggregate_report = StructuredRepairReport(
        task="draft_rendering",
        provider=f"batched:{providers}",
        final_outcome="success",
        repair_attempted=batch_report["repair_count"] > 0,
        max_repair_attempts=max_repair_attempts,
        attempt_count=batch_report["attempt_count"],
        repair_count=batch_report["repair_count"],
        duration_ms=batch_report["duration_ms"],
        attempts=attempts,
        final_provider_result_ref="provider_result.json",
        non_repairable_issues=non_repairable,
    )
    write_json(step_root / "structured_repair_report.json", aggregate_report)
    (step_root / "structured_repair_report.md").write_text(render_structured_repair_report_markdown(aggregate_report), encoding="utf-8")


def render_draft_rendering_batch_report(report: dict[str, Any]) -> str:
    rows = [
        [
            batch["batch_id"],
            ", ".join(f"`{page_id}`" for page_id in batch["page_plan_ids"]),
            str(batch["attempt_count"]),
            str(batch.get("http_attempt_count", 0)),
            str(batch["repair_count"]),
            format_duration(batch["duration_ms"]),
            str(batch["source_excerpt_chars"]),
            f"{int(batch.get('payload_char_count', 0)):,}",
            str(batch.get("reinforced_section_count", 0)),
            str(batch.get("grounding_rewrite_count", 0)),
        ]
        for batch in report["batches"]
    ]
    return (
        "# Draft Rendering 分批报告\n\n"
        f"- 并行执行：`{str(bool(report.get('parallel', False))).lower()}`\n"
        f"- 最大并行批数：{report.get('max_parallel_batches', 1)}\n"
        f"- Batch 页面上限：{report.get('batch_page_limit', DRAFT_RENDERING_BATCH_PAGE_LIMIT)}\n"
        f"- 模型累计耗时：{format_duration(report.get('model_duration_ms', report.get('duration_ms')))}\n"
        f"- 墙钟耗时：{format_duration(report.get('wall_duration_ms'))}\n"
        f"- 最大单批 payload：{int(report.get('max_batch_payload_char_count', 0)):,} chars\n"
        f"- 平均单批 payload：{int(report.get('avg_batch_payload_char_count', 0)):,} chars\n\n"
        + format_markdown_table(
            [
                "Batch",
                "页面计划",
                "Attempts",
                "HTTP Attempts",
                "Repairs",
                "Duration",
                "Source Excerpt Chars",
                "Payload Chars",
                "Reinforced Sections",
                "Grounding Rewrites",
            ],
            rows,
        )
        + "\n"
    )

def finalize_draft_rendering(
    artifact: DraftRenderingArtifact,
    plan: WikiMergePlanArtifact,
    snapshot: WikiContextSnapshot,
) -> DraftRenderingArtifact:
    plan_by_id = {item.page_plan_id: item for item in plan.items if item.action in {"create", "update"}}
    pages: list[DraftPageItem] = []
    used_ids: set[str] = set()
    for page in artifact.pages:
        if not page.page_plan_id.strip():
            _draft_validation.raise_draft_issue("missing_field", "draft_rendering page_plan_id must not be empty", field_path="pages.page_plan_id")
        item = plan_by_id.get(page.page_plan_id)
        if item is None:
            item = next((candidate for candidate in plan_by_id.values() if candidate.canonical_target_path == page.canonical_target_path), None)
        if item is None:
            _draft_validation.raise_draft_issue(
                "unknown_page_plan_reference",
                f"draft_rendering references non-draftable page_plan_id: {page.page_plan_id}",
                field_path="pages.page_plan_id",
            )
        if item.page_plan_id in used_ids:
            _draft_validation.raise_draft_issue(
                "duplicate_page_plan_id",
                f"draft_rendering duplicates page_plan_id: {item.page_plan_id}",
                field_path="pages.page_plan_id",
            )
        used_ids.add(item.page_plan_id)
        entry = snapshot_entry(snapshot, f"wiki/{item.canonical_target_path}")
        pages.append(
            page.model_copy(
                update={
                    "page_plan_id": item.page_plan_id,
                    "action": item.action,
                    "canonical_target_path": item.canonical_target_path,
                    "preimage_sha256": entry.preimage_sha256,
                    **_draft_validation.canonical_draft_page_content(page, item=item),
                    "change_summary": _draft_validation.finalize_draft_change_summary(page.change_summary, item),
                    "source_coverage_notes": _draft_validation.finalize_draft_source_coverage_notes(page.source_coverage_notes, item),
                }
            )
        )
    return DraftRenderingArtifact(pages=pages)
