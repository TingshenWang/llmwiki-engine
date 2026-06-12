from __future__ import annotations

import difflib
import json
import re
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal

import yaml
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from llmwiki_engine import __version__

from . import prompts
from . import related as related_logic
from . import system_pages
from .embeddings import EmbeddingConfig, build_candidate_contexts, load_embedding_config, sync_page_embedding_cache
from .io import (
    append_jsonl,
    artifact_hash,
    atomic_write_text,
    ensure_under,
    now_utc,
    read_json,
    read_text,
    relative_posix,
    safe_filename,
    safe_id,
    sha256_file,
    sha256_text,
    stable_json_hash,
    write_json,
    write_text,
)
from .labels import count_label, step_label
from .models import (
    ArtifactRef,
    CandidateContext,
    CandidateContexts,
    CandidateMergePlan,
    CandidateMergeUnit,
    CandidatePage,
    CandidatePages,
    CandidatePagesWarmup,
    CompositionItem,
    CompositionPlan,
    FinalPage,
    FinalPages,
    MergeDecision,
    MergePlan,
    OperationManifest,
    RawBinding,
    Receipt,
    RepairItem,
    RelatedPageRef,
    SourceDigest,
    SourceDigestCandidate,
    SourceRef,
    StepRecord,
    StructuredRepairReport,
    ValidationIssue,
    ValidationReport,
    WikiKnowledgeEntry,
    WikiSnapshot,
    WriteResult,
    WriteSet,
    WriteSetItem,
)
from .profile import Profile, load_profile, write_profile
from .providers import MODEL_BACKED_STEPS, ProviderCallError, ProviderConfigError, ProviderRegistry, load_provider_registry
from .text import (
    contains_cjk,
    strip_frontmatter,
    summarize,
    title_from_markdown,
)
from .token_usage import (
    format_duration_ms,
    format_percent,
    format_price_cny,
    summarize_api_calls,
    tag_api_calls,
)


class PipelineError(RuntimeError):
    pass


@dataclass
class StepOutput:
    artifacts: list[Path]
    counts: dict[str, int | float | str]
    warnings: list[str] | None = None
    model_calls: int = 0
    repair_count: int = 0
    api_calls: list[dict[str, object]] | None = None


def init_vault(vault: Path, profile_name: str = "project_basic") -> Path:
    vault = vault.expanduser().resolve()
    vault.mkdir(parents=True, exist_ok=True)
    for directory in ["raw", "wiki", ".llmwiki/runs/ingest", ".llmwiki/applied", ".llmwiki/profiles"]:
        (vault / directory).mkdir(parents=True, exist_ok=True)
    (vault / "wiki" / "logs").mkdir(parents=True, exist_ok=True)
    profile = load_profile(vault, profile_name)
    write_profile(vault, profile)
    config = {
        "version": "lite-1",
        "profile": profile.name,
        "mode": "full_auto",
        "embedding": {
            "enabled": True,
            "backend": "sentence_transformers",
            "cache_dir": ".llmwiki/cache/embeddings",
            "top_k_pages": 5,
            "dimensions": 1024,
            "input_version": "page_card_v1",
            "max_page_chars": 6000,
            "max_query_chars": 4000,
            "batch_size": 8,
            "normalize_embeddings": True,
            "query_prompt_name": "query",
        },
        "page_generation": {},
    }
    write_json(vault / ".llmwiki" / "config.json", config)
    applied = vault / ".llmwiki" / "applied" / "operations.jsonl"
    applied.touch(exist_ok=True)
    _ensure_gitignore(vault)
    if not (vault / "wiki" / "index.md").exists():
        write_text(vault / "wiki" / "index.md", system_pages.initial_index_text())
    if not (vault / "wiki" / "log.md").exists():
        write_text(vault / "wiki" / "log.md", system_pages.initial_log_text())
    return vault


def run_ingest(
    vault: Path,
    raw_file: Path,
    *,
    profile_name: str | None = None,
    slug: str | None = None,
    console: Console | None = None,
    emit_progress: bool = True,
) -> OperationManifest:
    vault = vault.expanduser().resolve()
    if not (vault / ".llmwiki").exists():
        init_vault(vault, profile_name or "project_basic")
    config = _load_config(vault)
    provider_registry = load_provider_registry(vault)
    try:
        provider_registry.require_real_model_providers(MODEL_BACKED_STEPS)
    except ProviderConfigError as exc:
        raise PipelineError(str(exc)) from exc
    profile = load_profile(vault, profile_name or str(config.get("profile", "project_basic")))
    raw_abs = _resolve_raw(vault, raw_file)
    raw_rel = relative_posix(raw_abs, vault)
    _assert_llmwiki_not_tracked(vault)

    op_slug = safe_id(slug or raw_abs.stem, fallback="ingest")
    operation_id = f"ING-{now_utc().replace(':', '').replace('-', '')}-{op_slug}"
    run_dir = vault / ".llmwiki" / "runs" / "ingest" / operation_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = OperationManifest(
        operation_id=operation_id,
        status="created",
        created_at=now_utc(),
        updated_at=now_utc(),
        vault=vault.as_posix(),
        raw_path=raw_rel,
        profile_name=profile.name,
        engine_version=__version__,
    )
    _write_manifest(run_dir, manifest)
    _write_event(run_dir, "operation_started", {"operation_id": operation_id, "raw_path": raw_rel})
    progress_console = console or Console()

    state: dict[str, object] = {
        "raw_abs": raw_abs,
        "raw_rel": raw_rel,
        "profile": profile,
        "config": config,
        "embedding_config": load_embedding_config(config),
        "provider_registry": provider_registry,
        "provider_contexts": provider_registry.sanitized_contexts(MODEL_BACKED_STEPS),
        "operation_id": operation_id,
    }
    steps: list[tuple[str, Callable[[], StepOutput]]] = [
        ("raw_binding", lambda: _step_raw_binding(run_dir, state)),
        ("source_digest", lambda: _step_source_digest(run_dir, state)),
        ("candidate_merge", lambda: _step_candidate_merge(run_dir, state)),
        ("candidate_pages_warmup", lambda: _step_candidate_pages_warmup(run_dir, state)),
        ("candidate_pages", lambda: _step_candidate_pages(run_dir, state)),
        ("wiki_snapshot", lambda: _step_wiki_snapshot(vault, run_dir, state)),
        ("candidate_contexts", lambda: _step_candidate_contexts(run_dir, state)),
        ("merge_plan", lambda: _step_merge_plan(run_dir, state)),
        ("composition_plan", lambda: _step_composition_plan(vault, run_dir, state)),
        ("final_pages", lambda: _step_final_pages(vault, run_dir, state)),
        ("validation", lambda: _step_validation(vault, run_dir, state)),
        ("knowledge_write", lambda: _step_knowledge_write(vault, run_dir, state)),
        ("source_record_write", lambda: _step_source_record_write(vault, run_dir, state, manifest)),
        ("index_log_write", lambda: _step_index_log_write(vault, run_dir, state, manifest)),
        ("embedding_cache_refresh", lambda: _step_embedding_cache_refresh(vault, run_dir, state)),
        ("receipt", lambda: _step_receipt(vault, run_dir, state, manifest)),
    ]

    manifest.status = "running"
    _write_manifest(run_dir, manifest)
    try:
        for name, fn in steps:
            _run_step(run_dir, manifest, name, fn, progress_console, emit_progress=emit_progress)
        manifest.status = "source_recorded"
        if any(item.kind == "knowledge" for item in state.get("write_set_items", [])):  # type: ignore[arg-type]
            manifest.status = "written"
        manifest.updated_at = now_utc()
        _write_manifest(run_dir, manifest)
        _write_event(run_dir, "operation_completed", {"status": manifest.status})
    except Exception as exc:
        manifest.status = "failed"
        manifest.updated_at = now_utc()
        manifest.warnings.append(str(exc))
        _write_manifest(run_dir, manifest)
        _write_event(run_dir, "operation_failed", {"error": str(exc)})
        if isinstance(exc, PipelineError):
            raise
        raise PipelineError(str(exc)) from exc
    return manifest


def status(vault: Path, operation_id: str | None = None) -> OperationManifest:
    vault = vault.expanduser().resolve()
    operation_id = operation_id or latest_operation_id(vault)
    if operation_id is None:
        raise ValueError("没有找到 ingest operation。")
    return OperationManifest.model_validate(read_json(_run_dir(vault, operation_id) / "manifest.json"))


def inspect_operation(vault: Path, operation_id: str | None = None) -> dict[str, object]:
    manifest = status(vault, operation_id)
    run_dir = _run_dir(Path(manifest.vault), manifest.operation_id)
    return {
        "operation_id": manifest.operation_id,
        "status": manifest.status,
        "raw_path": manifest.raw_path,
        "profile_name": manifest.profile_name,
        "receipt_path": manifest.receipt_path,
        "steps": [step.model_dump(mode="json") for step in manifest.steps],
        "run_dir": run_dir.as_posix(),
    }


def latest_operation_id(vault: Path) -> str | None:
    runs = vault.expanduser().resolve() / ".llmwiki" / "runs" / "ingest"
    if not runs.exists():
        return None
    candidates = sorted([path.name for path in runs.iterdir() if path.is_dir()])
    return candidates[-1] if candidates else None


def verify_operation(vault: Path, operation_id: str | None = None) -> ValidationReport:
    manifest = status(vault, operation_id)
    run_dir = _run_dir(Path(manifest.vault), manifest.operation_id)
    issues: list[ValidationIssue] = []
    raw = Path(manifest.vault) / manifest.raw_path
    binding_path = run_dir / "raw_binding" / "raw_binding.json"
    if binding_path.exists() and raw.exists():
        binding = RawBinding.model_validate(read_json(binding_path))
        current_hash = sha256_file(raw)
        if current_hash != binding.raw_sha256:
            issues.append(ValidationIssue(severity="error", code="raw_hash_drift", message="raw 文件 hash 已变化。", path=manifest.raw_path))
    for step in manifest.steps:
        for artifact in step.artifacts:
            path = run_dir / artifact.path
            if not path.exists():
                issues.append(ValidationIssue(severity="error", code="artifact_missing", message="artifact 缺失。", path=artifact.path))
                continue
            current_hash, current_size = artifact_hash(path)
            if current_hash != artifact.sha256 or current_size != artifact.size_bytes:
                issues.append(ValidationIssue(severity="error", code="artifact_hash_drift", message="artifact hash 已变化。", path=artifact.path))
    return ValidationReport(ok=not any(issue.severity == "error" for issue in issues), issues=issues)


def scan_raw_candidates(vault: Path, *, include_processed: bool = False, limit: int | None = None) -> dict[str, object]:
    vault = vault.expanduser().resolve()
    processed = _processed_raw_paths(vault)
    items = []
    for raw in sorted((vault / "raw").rglob("*")):
        if not raw.is_file():
            continue
        rel = relative_posix(raw, vault)
        is_processed = rel in processed
        if is_processed and not include_processed:
            continue
        items.append({"raw_path": rel, "processed": is_processed, "sha256": sha256_file(raw), "size_bytes": raw.stat().st_size})
        if limit is not None and len(items) >= limit:
            break
    return {"vault": vault.as_posix(), "count": len(items), "items": items}


def normalize_source_digest(digest: SourceDigest) -> tuple[SourceDigest, StructuredRepairReport]:
    repairs: list[RepairItem] = []

    def fix(candidate: SourceDigestCandidate, path: str) -> SourceDigestCandidate:
        if candidate.suggested_page_title.strip():
            return candidate
        replacement = candidate.name.strip() or safe_filename(candidate.candidate_id)
        repairs.append(RepairItem(path=path, reason="suggested_page_title 为空，已使用候选名称补齐。", local_fix=True, model_called=False))
        return candidate.model_copy(update={"suggested_page_title": replacement})

    sections: dict[str, list[SourceDigestCandidate]] = {}
    for section in ["entities", "concepts", "designs", "comparisons", "open_questions", "budget_deferred_candidates"]:
        items = []
        for index, candidate in enumerate(getattr(digest, section)):
            items.append(fix(candidate, f"{section}[{index}].suggested_page_title"))
        sections[section] = items
    updated = digest.model_copy(update=sections)
    return updated, StructuredRepairReport(repairs=repairs, model_calls=0)


def _call_provider_artifact(
    state: dict[str, object],
    out_dir: Path,
    step: str,
    request: BaseModel,
    output_model: type[BaseModel],
) -> tuple[BaseModel, list[Path], int, list[dict[str, object]]]:
    registry: ProviderRegistry = state["provider_registry"]  # type: ignore[assignment]
    spec = registry.provider_for(step)
    provider_contexts: dict[str, dict[str, object]] = state["provider_contexts"]  # type: ignore[assignment]
    provider_contexts[step] = spec.sanitized_context()
    try:
        result = registry.call_structured(step, request, output_model)
    except (ProviderConfigError, ProviderCallError, ValueError) as exc:
        raise PipelineError(str(exc)) from exc
    model_dir = out_dir / "model_calls"
    prompt_path = model_dir / f"{step}.prompt.json"
    result_path = model_dir / f"{step}.provider_result.json"
    api_calls_path = model_dir / f"{step}.token_usage_calls.json"
    api_calls = tag_api_calls(result.api_calls, step)
    write_json(prompt_path, result.prompt_artifact)
    write_json(result_path, result.provider_result)
    write_json(api_calls_path, api_calls)
    provider_contexts[step] = result.sanitized_context
    return result.output, [prompt_path, result_path, api_calls_path], result.model_calls, api_calls


def _call_provider_artifacts_parallel(
    state: dict[str, object],
    out_dir: Path,
    step: str,
    requests: list[tuple[str, BaseModel]],
    output_model: type[BaseModel],
) -> tuple[list[BaseModel], list[Path], int, list[dict[str, object]]]:
    registry: ProviderRegistry = state["provider_registry"]  # type: ignore[assignment]
    spec = registry.provider_for(step)
    provider_contexts: dict[str, dict[str, object]] = state["provider_contexts"]  # type: ignore[assignment]
    provider_contexts[step] = {**spec.sanitized_context(), "parallel_request_count": len(requests)}
    if not requests:
        return [], [], 0, []

    max_workers = _page_generation_parallelism(state, len(requests))
    results_by_key: dict[str, BaseModel] = {}
    artifacts_by_key: dict[str, list[Path]] = {}
    model_calls_by_key: dict[str, int] = {}
    api_calls_by_key: dict[str, list[dict[str, object]]] = {}
    contexts = []
    model_dir = out_dir / "model_calls"

    def call_one(key: str, request: BaseModel) -> tuple[str, BaseModel, list[Path], dict[str, object], int, list[dict[str, object]]]:
        try:
            result = registry.call_structured(step, request, output_model)
        except (ProviderConfigError, ProviderCallError, ValueError) as exc:
            raise PipelineError(str(exc)) from exc
        artifact_stem = f"{step}_{safe_filename(key)}"
        prompt_path = model_dir / f"{artifact_stem}.prompt.json"
        result_path = model_dir / f"{artifact_stem}.provider_result.json"
        write_json(prompt_path, result.prompt_artifact)
        write_json(result_path, result.provider_result)
        return key, result.output, [prompt_path, result_path], result.sanitized_context, result.model_calls, tag_api_calls(result.api_calls, key)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(call_one, key, request) for key, request in requests]
        for future in as_completed(futures):
            key, output, artifacts, context, model_calls, api_calls = future.result()
            results_by_key[key] = output
            artifacts_by_key[key] = artifacts
            model_calls_by_key[key] = model_calls
            api_calls_by_key[key] = api_calls
            contexts.append(context)

    provider_contexts[step] = {
        **spec.sanitized_context(),
        "parallel_request_count": len(requests),
        "parallel_max_workers": max_workers,
        "per_request_contexts": contexts,
    }
    outputs = [results_by_key[key] for key, _ in requests]
    artifacts = [path for key, _ in requests for path in artifacts_by_key.get(key, [])]
    model_calls = sum(model_calls_by_key.get(key, 0) for key, _ in requests)
    api_calls = [call for key, _ in requests for call in api_calls_by_key.get(key, [])]
    api_calls_path = out_dir / "token_usage_calls.json"
    write_json(api_calls_path, api_calls)
    return outputs, [*artifacts, api_calls_path], model_calls, api_calls


def _token_usage_counts(api_calls: list[dict[str, object]]) -> dict[str, int | float]:
    summary = summarize_api_calls(api_calls)
    return {
        "api_call_count": int(summary["api_call_count"]),
        "api_success_count": int(summary["api_success_count"]),
        "api_paused_count": int(summary["api_paused_count"]),
        "prompt_tokens": int(summary["prompt_tokens"]),
        "prompt_cache_hit_tokens": int(summary["prompt_cache_hit_tokens"]),
        "prompt_cache_miss_tokens": int(summary["prompt_cache_miss_tokens"]),
        "completion_tokens": int(summary["completion_tokens"]),
        "reasoning_tokens": int(summary["reasoning_tokens"]),
        "total_tokens": int(summary["total_tokens"]),
        "cache_hit_rate_percent": float(summary["cache_hit_rate_percent"]),
        "price_cny": float(summary["price_cny"]),
    }


def _page_generation_parallelism(state: dict[str, object], request_count: int) -> int:
    if request_count <= 0:
        return 0
    config = state.get("config")
    if not isinstance(config, dict):
        return request_count
    generation = config.get("page_generation")
    if not isinstance(generation, dict):
        return request_count
    raw_limit = generation.get("max_parallel_requests")
    if raw_limit is None:
        return request_count
    try:
        return min(request_count, max(1, int(raw_limit)))
    except (TypeError, ValueError):
        return request_count


def _assert_source_digest_binding(digest: SourceDigest, raw_rel: str, raw_sha256: str) -> None:
    if digest.source_raw_path != raw_rel:
        raise PipelineError(f"source_digest source_raw_path 不一致：expected={raw_rel} actual={digest.source_raw_path}")
    if digest.raw_sha256 != raw_sha256:
        raise PipelineError("source_digest raw_sha256 不一致。")


def _assert_unique(values: list[str], label: str) -> None:
    duplicates = [value for value, count in Counter(values).items() if count > 1]
    if duplicates:
        raise PipelineError(f"{label} 存在重复值：{', '.join(sorted(duplicates))}")


def _assert_candidate_pages_have_sources(artifact: CandidatePages) -> None:
    for page in artifact.pages:
        if not page.candidate_unit_id.strip():
            raise PipelineError(f"候选页 {page.candidate_page_id} 缺少 candidate_unit_id。")
        if not page.source_refs:
            raise PipelineError(f"候选页 {page.candidate_page_id} 缺少 source_refs。")
        if not page.body_markdown.strip():
            raise PipelineError(f"候选页 {page.candidate_page_id} 正文为空。")


def _assert_source_digest_chinese(digest: SourceDigest) -> None:
    _require_chinese_text("source_digest.summary", digest.summary)
    for index, item in enumerate(digest.key_takeaways, start=1):
        _require_chinese_text(f"source_digest.key_takeaways[{index}]", item)
    for candidate in digest.candidates():
        _require_chinese_title(f"{candidate.candidate_id}.name", candidate.name, candidate.suggested_page_title, candidate.summary, candidate.source_basis)
        _require_chinese_title(f"{candidate.candidate_id}.suggested_page_title", candidate.suggested_page_title, candidate.summary, candidate.source_basis)
        _require_chinese_text(f"{candidate.candidate_id}.summary", candidate.summary)
        _require_chinese_text(f"{candidate.candidate_id}.source_basis", candidate.source_basis)
    for index, item in enumerate(digest.weak_or_noise_items, start=1):
        _require_chinese_text(f"weak_or_noise_items[{index}].reason", item.reason)


def _assert_candidate_pages_chinese(artifact: CandidatePages) -> None:
    for page in artifact.pages:
        _require_chinese_title(f"{page.candidate_page_id}.title", page.title, page.summary, page.body_markdown)
        _require_chinese_text(f"{page.candidate_page_id}.summary", page.summary)
        _require_chinese_text(f"{page.candidate_page_id}.body_markdown", page.body_markdown)
        for index, item in enumerate(page.open_questions, start=1):
            _require_chinese_text(f"{page.candidate_page_id}.open_questions[{index}]", item)
        for index, item in enumerate(page.evidence_notes, start=1):
            _require_chinese_text(f"{page.candidate_page_id}.evidence_notes[{index}]", item)


def _assert_candidate_merge_chinese(plan: CandidateMergePlan) -> None:
    for unit in plan.units:
        _require_chinese_title(f"{unit.candidate_unit_id}.title", unit.title, unit.summary, unit.merge_reason)
        _require_chinese_text(f"{unit.candidate_unit_id}.summary", unit.summary)
        _require_chinese_text(f"{unit.candidate_unit_id}.merge_reason", unit.merge_reason)
        for index, item in enumerate(unit.must_cover_points, start=1):
            _require_chinese_text(f"{unit.candidate_unit_id}.must_cover_points[{index}]", item)
    for index, item in enumerate(plan.warnings, start=1):
        _require_chinese_text(f"candidate_merge.warnings[{index}]", item)


def _assert_merge_plan_chinese(plan: MergePlan) -> None:
    for decision in plan.decisions:
        _require_chinese_title(f"{decision.decision_id}.title", decision.title, decision.reason, decision.content_scope)
        _require_chinese_text(f"{decision.decision_id}.content_scope", decision.content_scope)
        _require_chinese_text(f"{decision.candidate_page_id}.reason", decision.reason)
        for index, item in enumerate(decision.candidate_path_index, start=1):
            _require_chinese_text(f"{decision.decision_id}.candidate_path_index[{index}]", item)
        for ref in decision.related_pages:
            _require_chinese_text(f"{decision.candidate_page_id}.related_pages.reason", ref.reason)
        for index, item in enumerate(decision.warnings, start=1):
            _require_chinese_text(f"{decision.candidate_page_id}.warnings[{index}]", item)


def _assert_composition_plan_chinese(plan: CompositionPlan) -> None:
    for item in plan.items:
        _require_chinese_text(f"{item.final_page_id}.readability_goal", item.readability_goal)
        for field_name in ["preserve_rules", "insert_rules", "delete_rules", "source_ref_rules", "warnings"]:
            for index, value in enumerate(getattr(item, field_name), start=1):
                _require_chinese_text(f"{item.final_page_id}.{field_name}[{index}]", value)
        for ref in item.related_pages:
            _require_chinese_text(f"{item.final_page_id}.related_pages.reason", ref.reason)


def _assert_final_pages_chinese(artifact: FinalPages) -> None:
    for page in artifact.pages:
        body = strip_frontmatter(page.markdown)
        _require_chinese_title(f"{page.final_page_id}.title", page.title, body)
        _require_chinese_text(f"{page.final_page_id}.markdown", body)
        for index, item in enumerate(page.warnings, start=1):
            _require_chinese_text(f"{page.final_page_id}.warnings[{index}]", item)


def _require_chinese_title(field_path: str, value: str, *context: str) -> None:
    if contains_cjk(value or "") or any(contains_cjk(item or "") for item in context):
        return
    raise PipelineError(f"{field_path} 必须使用中文用户可读文本，或由中文上下文支撑的专有名词")


def _require_chinese_text(field_path: str, value: str) -> None:
    text = re.sub(r"`[^`]*`", " ", value or "")
    text = re.sub(r"\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[\[[^\]]*\]\]", " ", text)
    if text.strip() and not contains_cjk(text):
        raise PipelineError(f"{field_path} 必须使用中文用户可读文本")


def _normalize_candidate_pages(artifact: CandidatePages) -> CandidatePages:
    pages = []
    for page in artifact.pages:
        evidence_notes = [_chinese_scaffold(item, "来源定位") for item in page.evidence_notes]
        pages.append(page.model_copy(update={"evidence_notes": evidence_notes}))
    return artifact.model_copy(update={"pages": pages})


def _normalize_candidate_merge_plan(plan: CandidateMergePlan, digest: SourceDigest, profile: Profile) -> CandidateMergePlan:
    known_candidates = {candidate.candidate_id: candidate for candidate in digest.candidates()}
    used_source_ids: list[str] = []
    units: list[CandidateMergeUnit] = []
    for index, unit in enumerate(plan.units, start=1):
        source_ids = _dedupe_list(unit.source_candidate_ids)
        if not source_ids:
            raise PipelineError(f"候选合并单元 {unit.candidate_unit_id} 缺少 source_candidate_ids。")
        unknown = [candidate_id for candidate_id in source_ids if candidate_id not in known_candidates]
        if unknown:
            raise PipelineError(f"候选合并单元 {unit.candidate_unit_id} 引用了未知 source candidate：{', '.join(unknown)}")
        page_type = unit.page_type if unit.page_type in profile.page_types and unit.page_type != profile.source_page_type else profile.default_page_type
        path_hint = _normalize_path_hint(unit.path_hint, page_type, unit.title, profile)
        refs = _merge_source_refs([ref for candidate_id in source_ids for ref in known_candidates[candidate_id].source_refs] or unit.source_refs)
        units.append(
            unit.model_copy(
                update={
                    "candidate_unit_id": f"CM-{index:03d}",
                    "source_candidate_ids": source_ids,
                    "page_type": page_type,
                    "path_hint": path_hint,
                    "source_refs": refs,
                    "must_cover_points": unit.must_cover_points or [unit.summary],
                }
            )
        )
        used_source_ids.extend(source_ids)
    skipped = _dedupe_list([*plan.skipped_candidate_ids, *(candidate_id for candidate_id in known_candidates if candidate_id not in used_source_ids)])
    return plan.model_copy(update={"units": units, "skipped_candidate_ids": skipped})


def _normalize_path_hint(path_hint: str, page_type: str, title: str, profile: Profile) -> str:
    spec = profile.page_type(page_type)
    cleaned = path_hint.strip().replace("\\", "/").lstrip("/")
    if cleaned and not cleaned.endswith(".md"):
        cleaned += ".md"
    if cleaned.startswith(f"{spec.directory}/") and ".." not in Path(cleaned).parts:
        return cleaned
    filename = safe_filename(title) or "Untitled"
    if not filename.startswith(spec.title_prefix):
        filename = f"{spec.title_prefix}{filename}"
    return f"{spec.directory}/{filename}.md"


def _merge_parallel_candidate_pages(outputs: list[BaseModel], units: list[CandidateMergeUnit]) -> CandidatePages:
    if len(outputs) != len(units):
        raise PipelineError(f"候选页并发请求组返回 {len(outputs)} 个结果，但 candidate unit 数量是 {len(units)}。")
    pages: list[CandidatePage] = []
    skipped: list[str] = []
    for index, (output, unit) in enumerate(zip(outputs, units, strict=True), start=1):
        if not isinstance(output, CandidatePages):
            raise PipelineError("候选页并发请求返回了无效 artifact。")
        skipped.extend(output.skipped_candidate_ids)
        if len(output.pages) != 1:
            raise PipelineError(f"{unit.candidate_unit_id} 的候选页请求必须且只能返回 1 页。")
        page = output.pages[0]
        source_ids = _dedupe_list([*unit.source_candidate_ids, *page.source_candidate_ids])
        pages.append(
            page.model_copy(
                update={
                    "candidate_page_id": f"CP-{index:03d}",
                    "candidate_unit_id": unit.candidate_unit_id,
                    "source_candidate_ids": source_ids,
                    "proposed_page_type": unit.page_type,
                    "proposed_path_hint": unit.path_hint,
                    "source_refs": _merge_source_refs([*unit.source_refs, *page.source_refs]),
                }
            )
        )
    return CandidatePages(pages=pages, skipped_candidate_ids=_dedupe_list(skipped))


def _normalize_merge_plan(plan: MergePlan) -> MergePlan:
    action_counts = {action: 0 for action in ["create", "update", "noop"]}
    decisions: list[MergeDecision] = []
    for index, decision in enumerate(plan.decisions, start=1):
        action_counts[decision.action] += 1
        decision_id = decision.decision_id.strip() or f"MD-{index:03d}"
        decisions.append(decision.model_copy(update={"decision_id": decision_id}))
    return plan.model_copy(update={"decisions": decisions, "action_counts": action_counts})


def _assert_merge_plan_consumes_candidates(plan: MergePlan, candidate_pages: CandidatePages, contexts: CandidateContexts) -> None:
    expected = {page.candidate_page_id for page in candidate_pages.pages}
    actual = {decision.candidate_page_id for decision in plan.decisions}
    unknown = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unknown:
        raise PipelineError(f"merge_plan 引用了未知候选页：{unknown}")
    if missing:
        raise PipelineError(f"merge_plan 必须覆盖每个候选页：{missing}")
    _assert_unique([decision.decision_id for decision in plan.decisions], "merge decision_id")
    context_paths = {item.candidate_page_id: {hit.path for hit in item.hits} for item in contexts.items}
    for decision in plan.decisions:
        if decision.action in {"create", "update"} and not decision.target_path:
            raise PipelineError(f"合并决策 {decision.candidate_page_id} 缺少 target_path。")
        if not decision.source_refs:
            raise PipelineError(f"合并决策 {decision.decision_id} 缺少 source_refs。")
        if not decision.content_scope.strip():
            raise PipelineError(f"合并决策 {decision.decision_id} 缺少 content_scope。")
        if not decision.candidate_path_index:
            raise PipelineError(f"合并决策 {decision.decision_id} 缺少 candidate_path_index。")
        if decision.action == "update":
            allowed = context_paths.get(decision.candidate_page_id, set())
            if decision.target_path not in allowed:
                raise PipelineError(f"更新决策 {decision.decision_id} 的 target_path 不在该候选页 TopK 召回结果中。")
            if not decision.matched_existing_paths:
                raise PipelineError(f"更新决策 {decision.decision_id} 缺少 matched_existing_paths。")


def _assert_composition_covers_writes(artifact: CompositionPlan, plan: MergePlan) -> None:
    expected = sorted({decision.target_path for decision in plan.decisions if decision.action != "noop" and decision.target_path})
    actual = sorted(item.target_path for item in artifact.items)
    if expected != actual:
        raise PipelineError(f"composition_plan 必须覆盖所有可写合并目标：expected={expected} actual={actual}")
    consumed_decisions = sorted(decision_id for item in artifact.items for decision_id in item.merge_decision_ids)
    expected_decisions = sorted(decision.decision_id for decision in plan.decisions if decision.action != "noop")
    if consumed_decisions != expected_decisions:
        raise PipelineError(f"composition_plan 必须覆盖所有可写 merge decision：expected={expected_decisions} actual={consumed_decisions}")
    for item in artifact.items:
        if not item.merge_decision_ids:
            raise PipelineError(f"写作编排项 {item.final_page_id} 缺少 merge_decision_ids。")
        if not item.source_ref_rules:
            raise PipelineError(f"写作编排项 {item.final_page_id} 缺少 source_ref_rules。")


def _normalize_composition_plan(plan: CompositionPlan) -> CompositionPlan:
    grouped: dict[str, CompositionItem] = {}
    for item in plan.items:
        if item.target_path not in grouped:
            grouped[item.target_path] = item
            continue
        existing = grouped[item.target_path]
        grouped[item.target_path] = existing.model_copy(
            update={
                "candidate_page_ids": _dedupe_list([*existing.candidate_page_ids, *item.candidate_page_ids]),
                "merge_decision_ids": _dedupe_list([*existing.merge_decision_ids, *item.merge_decision_ids]),
                "existing_page_refs": _dedupe_list([*existing.existing_page_refs, *item.existing_page_refs]),
                "preserve_rules": _dedupe_list([*existing.preserve_rules, *item.preserve_rules]),
                "insert_rules": _dedupe_list([*existing.insert_rules, *item.insert_rules]),
                "delete_rules": _dedupe_list([*existing.delete_rules, *item.delete_rules]),
                "source_ref_rules": _dedupe_list([*existing.source_ref_rules, *item.source_ref_rules]),
                "related_pages": _merge_related_refs([*existing.related_pages, *item.related_pages], current_path=item.target_path),
                "related_absence_reason": existing.related_absence_reason or item.related_absence_reason,
                "related_unresolved": _dedupe_list([*existing.related_unresolved, *item.related_unresolved]),
                "warnings": _dedupe_list([*existing.warnings, *item.warnings, "多个合并决策指向同一个目标页面，已合并写作规则。"]),
            }
        )
    normalized = []
    for index, item in enumerate(grouped.values(), start=1):
        normalized.append(item.model_copy(update={"final_page_id": f"FP-{index:03d}"}))
    return CompositionPlan(items=normalized)


def _hydrate_provider_final_pages(pages: FinalPages, composition: CompositionPlan, snapshot: WikiSnapshot) -> FinalPages:
    items_by_id = {item.final_page_id: item for item in composition.items}
    items_by_target = {item.target_path: item for item in composition.items}
    entries_by_path = {entry.path: entry for entry in snapshot.entries}
    hydrated = []
    for page in pages.pages:
        item = items_by_id.get(page.final_page_id) or items_by_target.get(page.target_path)
        if item is None:
            hydrated.append(page.model_copy(update={"content_sha256": sha256_text(page.markdown)}))
            continue
        existing = entries_by_path.get(item.target_path)
        hydrated.append(
            page.model_copy(
                update={
                    "final_page_id": item.final_page_id,
                    "target_path": item.target_path,
                    "action": item.action,
                    "content_sha256": sha256_text(page.markdown),
                    "preimage_sha256": existing.sha256 if existing else None,
                }
            )
        )
    return pages.model_copy(update={"pages": hydrated})


def _merge_parallel_final_pages(outputs: list[BaseModel], composition: CompositionPlan) -> FinalPages:
    if len(outputs) != len(composition.items):
        raise PipelineError(f"最终页并发请求组返回 {len(outputs)} 个结果，但写作编排项数量是 {len(composition.items)}。")
    pages: list[FinalPage] = []
    warnings: list[str] = []
    for output, item in zip(outputs, composition.items, strict=True):
        if not isinstance(output, FinalPages):
            raise PipelineError("最终页并发请求返回了无效 artifact。")
        warnings.extend(output.warnings)
        if len(output.pages) != 1:
            raise PipelineError(f"{item.final_page_id} 的最终页请求必须且只能返回 1 页。")
        page = output.pages[0]
        pages.append(page.model_copy(update={"final_page_id": item.final_page_id, "target_path": item.target_path, "action": item.action}))
    return FinalPages(pages=pages, warnings=_dedupe_list(warnings))


def _normalize_final_pages(
    pages: FinalPages,
    composition: CompositionPlan,
    *,
    snapshot: WikiSnapshot | None = None,
    operation_id: str = "",
) -> FinalPages:
    items_by_id = {item.final_page_id: item for item in composition.items}
    items_by_target = {item.target_path: item for item in composition.items}
    entries_by_path = {entry.path: entry for entry in snapshot.entries} if snapshot else {}
    known_paths = set(entries_by_path) | {page.target_path for page in pages.pages}
    path_titles = {entry.path: entry.title for entry in entries_by_path.values()}
    normalized = []
    for page in pages.pages:
        item = items_by_id.get(page.final_page_id) or items_by_target.get(page.target_path)
        existing_entry = entries_by_path.get(page.target_path)
        model_title = page.title
        final_title = existing_entry.title if existing_entry is not None and item is not None and item.action == "update" else page.title
        path_titles[page.target_path] = final_title
        source_refs = _merge_source_refs(page.source_refs)
        updated_page = page.model_copy(update={"title": final_title, "source_refs": source_refs})
        markdown = _canonical_final_markdown(
            updated_page,
            item,
            operation_id=operation_id,
            existing_entry=existing_entry,
            known_paths=known_paths,
            path_titles=path_titles,
            model_title=model_title,
        )
        normalized.append(updated_page.model_copy(update={"markdown": markdown, "content_sha256": sha256_text(markdown)}))
    return pages.model_copy(update={"pages": normalized})


def _canonical_final_markdown(
    page: FinalPage,
    item: CompositionItem | None,
    *,
    operation_id: str = "",
    existing_entry: WikiKnowledgeEntry | None = None,
    known_paths: set[str] | None = None,
    path_titles: dict[str, str] | None = None,
    model_title: str = "",
) -> str:
    body = strip_frontmatter(page.markdown).strip()
    body = _drop_sections(body, {"Related", "相关页面"}).strip()
    link_errors = related_logic.precanonical_link_errors(markdown=body, target_path=page.target_path, title=page.title)
    if link_errors:
        raise PipelineError(f"最终页 {page.target_path} 不符合链接契约：{'; '.join(link_errors)}")
    if not body.startswith("# "):
        body = f"# {page.title}\n\n{body}" if body else f"# {page.title}\n"
    if item is not None:
        related = related_logic.render_related_section(
            current_path=page.target_path,
            related_pages=item.related_pages,
            known_paths=known_paths or set(),
            path_titles=path_titles,
        )
        body = _replace_section(body, {"Related", "相关页面"}, "相关页面", related)
    log_date = _operation_date(operation_id)
    source_refs = _merge_source_refs(page.source_refs)
    aliases = list(existing_entry.aliases if existing_entry else [])
    if existing_entry is not None and model_title and model_title != page.title and model_title not in aliases:
        aliases.append(model_title)
    created = existing_entry.created if existing_entry is not None and existing_entry.created else log_date
    source_raw_paths = _dedupe_list([*(existing_entry.source_raw_paths if existing_entry else []), *(ref.raw_path for ref in source_refs)])
    source_raw_hashes = _dedupe_list([*(existing_entry.source_raw_hashes if existing_entry else []), *(ref.raw_sha256 for ref in source_refs)])
    source_prepared_hashes = _dedupe_list([*(existing_entry.source_prepared_hashes if existing_entry else []), *(ref.raw_sha256 for ref in source_refs)])
    source_operation_ids = _dedupe_list([*(existing_entry.source_operation_ids if existing_entry else []), *([operation_id] if operation_id else [])])
    summary = summarize(body, max_sentences=2, max_chars=260)
    frontmatter = (
        "---\n"
        f"llmwiki_type: {page.page_type}\n"
        f"title: {_yaml_scalar(page.title)}\n"
        f"{_yaml_list('aliases', aliases)}"
        f"summary: {_yaml_scalar(summary)}\n"
        f"created: {created}\n"
        f"updated: {log_date}\n"
        f"{_yaml_list('source_raw_paths', source_raw_paths)}"
        f"{_yaml_list('source_raw_hashes', source_raw_hashes)}"
        f"{_yaml_list('source_prepared_hashes', source_prepared_hashes)}"
        f"{_yaml_list('source_operation_ids', source_operation_ids)}"
        f"last_ingest_operation: {_yaml_scalar(operation_id)}\n"
        "---"
    )
    return f"{frontmatter}\n\n{body.rstrip()}\n"


def _replace_section(markdown: str, headings: set[str], new_heading: str, new_body: str) -> str:
    wanted = {heading.lower() for heading in headings}
    lines = markdown.rstrip().splitlines()
    start: int | None = None
    end: int | None = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("## ") and stripped[3:].strip().lower() in wanted:
            start = index
            end = len(lines)
            for next_index in range(index + 1, len(lines)):
                if lines[next_index].strip().startswith("## "):
                    end = next_index
                    break
            break
    replacement = [f"## {new_heading}", "", new_body.strip()]
    if start is None or end is None:
        return markdown.rstrip() + "\n\n" + "\n".join(replacement) + "\n"
    return "\n".join([*lines[:start], *replacement, *lines[end:]]).rstrip() + "\n"


def _operation_date(operation_id: str) -> str:
    match = re.search(r"ING-(\d{8})T", operation_id)
    if match:
        raw = match.group(1)
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    return now_utc()[:10]


def _assert_final_pages_cover_composition(artifact: FinalPages, composition: CompositionPlan) -> None:
    expected = sorted(item.target_path for item in composition.items)
    actual = sorted(page.target_path for page in artifact.pages)
    if expected != actual:
        raise PipelineError(f"final_pages 必须覆盖写作编排目标：expected={expected} actual={actual}")
    for page in artifact.pages:
        if not page.source_refs:
            raise PipelineError(f"最终页 {page.target_path} 缺少 source_refs。")
        if not page.markdown.strip():
            raise PipelineError(f"最终页 {page.target_path} markdown 为空。")


def _dedupe_list(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _step_raw_binding(run_dir: Path, state: dict[str, object]) -> StepOutput:
    raw_abs = state["raw_abs"]  # type: ignore[assignment]
    raw_rel = state["raw_rel"]  # type: ignore[assignment]
    stat = raw_abs.stat()  # type: ignore[union-attr]
    if stat.st_size == 0:
        raise PipelineError("raw 文件为空。")
    binding = RawBinding(
        raw_path=str(raw_rel),
        raw_sha256=sha256_file(raw_abs),  # type: ignore[arg-type]
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        bound_at=now_utc(),
    )
    state["raw_binding"] = binding
    path = run_dir / "raw_binding" / "raw_binding.json"
    write_json(path, binding)
    return StepOutput([path], {"raw_size_bytes": stat.st_size})


def _step_source_digest(run_dir: Path, state: dict[str, object]) -> StepOutput:
    raw_abs: Path = state["raw_abs"]  # type: ignore[assignment]
    raw_rel: str = state["raw_rel"]  # type: ignore[assignment]
    binding: RawBinding = state["raw_binding"]  # type: ignore[assignment]
    raw_text = read_text(raw_abs)
    profile: Profile = state["profile"]  # type: ignore[assignment]
    out_dir = run_dir / "source_digest"
    digest_result, provider_artifacts, model_calls, api_calls = _call_provider_artifact(
        state,
        out_dir,
        "source_digest",
        prompts.source_digest_prompt(raw_path=raw_rel, raw_sha256=binding.raw_sha256, raw_text=raw_text, profile=profile),
        SourceDigest,
    )
    if not isinstance(digest_result, SourceDigest):
        raise PipelineError("source_digest provider 返回了无效 artifact。")
    digest = digest_result
    _assert_source_digest_binding(digest, raw_rel, binding.raw_sha256)
    digest, repair_report = normalize_source_digest(digest)
    _assert_source_digest_chinese(digest)
    _assert_unique([candidate.candidate_id for candidate in digest.candidates()], "source digest candidate_id")
    state["source_digest"] = digest
    json_path = out_dir / "source_digest.json"
    md_path = out_dir / "source_digest.md"
    source_map_json = out_dir / "source_map.json"
    source_map_md = out_dir / "source_map.md"
    repair_path = out_dir / "structured_repair_report.json"
    write_json(json_path, digest)
    write_text(md_path, _render_source_digest_md(digest))
    source_map = {"source_raw_path": raw_rel, "candidate_ids": [item.candidate_id for item in digest.candidates()]}
    write_json(source_map_json, source_map)
    write_text(source_map_md, "\n".join(f"- {item.candidate_id}: {item.name}" for item in digest.candidates()) + "\n")
    write_json(repair_path, repair_report)
    counts = {
        "candidate_count": len(digest.candidates()),
        "weak_noise_count": len(digest.weak_or_noise_items),
        "deferred_count": len(digest.budget_deferred_candidates),
        **_token_usage_counts(api_calls),
    }
    return StepOutput(
        [json_path, md_path, source_map_json, source_map_md, repair_path, *provider_artifacts],
        counts,
        model_calls=model_calls,
        repair_count=len(repair_report.repairs),
        api_calls=api_calls,
    )


def _step_candidate_merge(run_dir: Path, state: dict[str, object]) -> StepOutput:
    digest: SourceDigest = state["source_digest"]  # type: ignore[assignment]
    profile: Profile = state["profile"]  # type: ignore[assignment]
    out_dir = run_dir / "candidate_merge"
    plan_result, provider_artifacts, model_calls, api_calls = _call_provider_artifact(
        state,
        out_dir,
        "candidate_merge",
        prompts.candidate_merge_prompt(digest=digest, profile=profile),
        CandidateMergePlan,
    )
    if not isinstance(plan_result, CandidateMergePlan):
        raise PipelineError("candidate_merge provider 返回了无效 artifact。")
    plan = _normalize_candidate_merge_plan(plan_result, digest, profile)
    _assert_unique([unit.candidate_unit_id for unit in plan.units], "candidate_unit_id")
    _assert_candidate_merge_chinese(plan)
    state["candidate_merge"] = plan
    json_path = out_dir / "candidate_merge.json"
    md_path = out_dir / "candidate_merge.md"
    write_json(json_path, plan)
    write_text(md_path, _render_candidate_merge_md(plan))
    return StepOutput(
        [json_path, md_path, *provider_artifacts],
        {
            "candidate_unit_count": len(plan.units),
            "skipped_candidate_count": len(plan.skipped_candidate_ids),
            **_token_usage_counts(api_calls),
        },
        model_calls=model_calls,
        api_calls=api_calls,
    )


def _step_candidate_pages_warmup(run_dir: Path, state: dict[str, object]) -> StepOutput:
    digest: SourceDigest = state["source_digest"]  # type: ignore[assignment]
    candidate_merge: CandidateMergePlan = state["candidate_merge"]  # type: ignore[assignment]
    raw_abs: Path = state["raw_abs"]  # type: ignore[assignment]
    binding: RawBinding = state["raw_binding"]  # type: ignore[assignment]
    raw_rel: str = state["raw_rel"]  # type: ignore[assignment]
    profile: Profile = state["profile"]  # type: ignore[assignment]
    out_dir = run_dir / "candidate_pages_warmup"
    if not candidate_merge.units:
        return StepOutput([], {"warmup_count": 0, "candidate_unit_count": 0})
    raw_text = read_text(raw_abs)
    warmup_result, provider_artifacts, model_calls, api_calls = _call_provider_artifact(
        state,
        out_dir,
        "candidate_pages_warmup",
        prompts.candidate_pages_warmup_prompt(digest=digest, raw_path=raw_rel, raw_sha256=binding.raw_sha256, raw_text=raw_text, profile=profile),
        CandidatePagesWarmup,
    )
    if not isinstance(warmup_result, CandidatePagesWarmup) or warmup_result.status != "OK":
        raise PipelineError("candidate_pages_warmup provider 返回了无效 artifact。")
    state["candidate_pages_warmup"] = warmup_result
    json_path = out_dir / "candidate_pages_warmup.json"
    write_json(json_path, warmup_result)
    return StepOutput(
        [json_path, *provider_artifacts],
        {
            "warmup_count": 1,
            "candidate_unit_count": len(candidate_merge.units),
            **_token_usage_counts(api_calls),
        },
        model_calls=model_calls,
        api_calls=api_calls,
    )


def _step_wiki_snapshot(vault: Path, run_dir: Path, state: dict[str, object]) -> StepOutput:
    profile: Profile = state["profile"]  # type: ignore[assignment]
    embedding_config: EmbeddingConfig = state["embedding_config"]  # type: ignore[assignment]
    entries = _scan_wiki_entries(vault, profile)
    page_records, embedding_metrics = sync_page_embedding_cache(vault, entries, embedding_config)
    snapshot = WikiSnapshot(
        wiki_root="wiki",
        pool_hash=stable_json_hash([entry.model_dump(mode="json") for entry in entries]),
        generated_at=now_utc(),
        entries=entries,
        retrieval_backend=embedding_config.backend,
        embedding_metrics=embedding_metrics,
    )
    state["wiki_snapshot"] = snapshot
    state["embedding_page_records"] = page_records
    out_dir = run_dir / "wiki_snapshot"
    snapshot_path = out_dir / "wiki_snapshot.json"
    pool_path = out_dir / "knowledge_pool.json"
    cache_report_path = out_dir / "embedding_cache_report.json"
    write_json(snapshot_path, snapshot)
    write_json(pool_path, [entry.model_dump(mode="json") for entry in entries])
    write_json(cache_report_path, embedding_metrics)
    return StepOutput(
        [snapshot_path, pool_path, cache_report_path],
        {
            "knowledge_pool_size": len(entries),
            "retrieval_backend": embedding_config.backend,
            "cache_hit": int(embedding_metrics.get("cache_hit", 0)),
            "cache_refreshed": int(embedding_metrics.get("cache_refreshed", 0)),
            "cache_pruned": int(embedding_metrics.get("cache_pruned", 0)),
        },
    )


def _step_candidate_pages(run_dir: Path, state: dict[str, object]) -> StepOutput:
    digest: SourceDigest = state["source_digest"]  # type: ignore[assignment]
    candidate_merge: CandidateMergePlan = state["candidate_merge"]  # type: ignore[assignment]
    raw_abs: Path = state["raw_abs"]  # type: ignore[assignment]
    binding: RawBinding = state["raw_binding"]  # type: ignore[assignment]
    raw_rel: str = state["raw_rel"]  # type: ignore[assignment]
    profile: Profile = state["profile"]  # type: ignore[assignment]
    raw_text = read_text(raw_abs)
    out_dir = run_dir / "candidate_pages"
    units = candidate_merge.units
    outputs, provider_artifacts, model_calls, api_calls = _call_provider_artifacts_parallel(
        state,
        out_dir,
        "candidate_pages",
        [
            (
                unit.candidate_unit_id,
                prompts.candidate_page_prompt(
                    digest=digest,
                    candidate_unit=unit,
                    raw_path=raw_rel,
                    raw_sha256=binding.raw_sha256,
                    raw_text=raw_text,
                    profile=profile,
                ),
            )
            for unit in units
        ],
        CandidatePages,
    )
    artifact = _merge_parallel_candidate_pages(outputs, units)
    parallel_request_count = len(units)
    artifact = _normalize_candidate_pages(artifact)
    _assert_unique([page.candidate_page_id for page in artifact.pages], "candidate_page_id")
    _assert_candidate_pages_have_sources(artifact)
    _assert_candidate_pages_chinese(artifact)
    state["candidate_pages"] = artifact
    json_path = out_dir / "candidate_pages.json"
    md_path = out_dir / "candidate_pages.md"
    write_json(json_path, artifact)
    write_text(md_path, _render_candidate_pages_md(artifact))
    artifacts = [json_path, md_path]
    for page in artifact.pages:
        page_path = out_dir / f"page_{safe_filename(page.candidate_page_id)}.md"
        write_text(page_path, page.body_markdown)
        artifacts.append(page_path)
    counts = {
        "candidate_page_count": len(artifact.pages),
        "covered_digest_candidate_count": len({cid for page in artifact.pages for cid in page.source_candidate_ids}),
        "parallel_request_count": parallel_request_count if model_calls else 0,
        "parallel_max_workers": _page_generation_parallelism(state, parallel_request_count) if model_calls else 0,
        **_token_usage_counts(api_calls),
    }
    return StepOutput([*artifacts, *provider_artifacts], counts, model_calls=model_calls, api_calls=api_calls)


def _step_candidate_contexts(run_dir: Path, state: dict[str, object]) -> StepOutput:
    candidate_pages: CandidatePages = state["candidate_pages"]  # type: ignore[assignment]
    snapshot: WikiSnapshot = state["wiki_snapshot"]  # type: ignore[assignment]
    embedding_config: EmbeddingConfig = state["embedding_config"]  # type: ignore[assignment]
    page_records: dict[str, dict[str, object]] = state.get("embedding_page_records", {})  # type: ignore[assignment]
    artifact = build_candidate_contexts(candidate_pages, snapshot.entries, page_records, embedding_config)
    state["candidate_contexts"] = artifact
    out_dir = run_dir / "candidate_contexts"
    json_path = out_dir / "candidate_contexts.json"
    md_path = out_dir / "candidate_contexts.md"
    write_json(json_path, artifact)
    write_text(md_path, _render_contexts_md(artifact.items))
    return StepOutput(
        [json_path, md_path],
        {
            "candidate_page_count": len(candidate_pages.pages),
            "query_count": len(artifact.items),
            "top_k": artifact.top_k,
            "retrieval_backend": artifact.retrieval_backend,
        },
    )


def _step_merge_plan(run_dir: Path, state: dict[str, object]) -> StepOutput:
    candidate_pages: CandidatePages = state["candidate_pages"]  # type: ignore[assignment]
    digest: SourceDigest = state["source_digest"]  # type: ignore[assignment]
    snapshot: WikiSnapshot = state["wiki_snapshot"]  # type: ignore[assignment]
    contexts: CandidateContexts = state["candidate_contexts"]  # type: ignore[assignment]
    profile: Profile = state["profile"]  # type: ignore[assignment]
    out_dir = run_dir / "merge_plan"
    plan_result, provider_artifacts, model_calls, api_calls = _call_provider_artifact(
        state,
        out_dir,
        "merge_plan",
        prompts.merge_plan_prompt(candidate_pages=candidate_pages, candidate_contexts=contexts, profile=profile),
        MergePlan,
    )
    if not isinstance(plan_result, MergePlan):
        raise PipelineError("merge_plan provider 返回了无效 artifact。")
    plan = plan_result
    plan = _normalize_merge_plan(plan)
    _assert_merge_plan_chinese(plan)
    plan, related_report = related_logic.finalize_merge_plan_related(
        plan,
        candidate_pages=candidate_pages,
        digest=digest,
        snapshot=snapshot,
        contexts=contexts,
    )
    _assert_merge_plan_consumes_candidates(plan, candidate_pages, contexts)
    state["merge_plan"] = plan
    state["related_merge_report"] = related_report
    json_path = out_dir / "merge_plan.json"
    md_path = out_dir / "merge_plan.md"
    related_report_json = out_dir / "related_merge_report.json"
    related_report_md = out_dir / "related_merge_report.md"
    write_json(json_path, plan)
    rendered = _render_merge_plan_md(plan)
    write_text(md_path, rendered)
    write_json(related_report_json, related_report)
    write_text(related_report_md, related_logic.render_related_report(related_report))
    return StepOutput(
        [json_path, md_path, related_report_json, related_report_md, *provider_artifacts],
        {
            **{f"{key}_count": value for key, value in plan.action_counts.items()},
            "related_kept_count": sum(1 for item in related_report.candidates if item.decision == "kept"),
            "related_filtered_count": sum(1 for item in related_report.candidates if item.decision != "kept"),
            **_token_usage_counts(api_calls),
        },
        model_calls=model_calls,
        api_calls=api_calls,
    )


def _step_composition_plan(vault: Path, run_dir: Path, state: dict[str, object]) -> StepOutput:
    plan: MergePlan = state["merge_plan"]  # type: ignore[assignment]
    candidate_pages: CandidatePages = state["candidate_pages"]  # type: ignore[assignment]
    snapshot: WikiSnapshot = state["wiki_snapshot"]  # type: ignore[assignment]
    profile: Profile = state["profile"]  # type: ignore[assignment]
    out_dir = run_dir / "composition_plan"
    plan_result, provider_artifacts, model_calls, api_calls = _call_provider_artifact(
        state,
        out_dir,
        "composition_plan",
        prompts.composition_plan_prompt(merge_plan=plan, candidate_pages=candidate_pages, profile=profile),
        CompositionPlan,
    )
    if not isinstance(plan_result, CompositionPlan):
        raise PipelineError("composition_plan provider 返回了无效 artifact。")
    artifact = plan_result
    artifact = _normalize_composition_plan(artifact)
    _assert_composition_plan_chinese(artifact)
    artifact, composition_related_report = related_logic.finalize_composition_related(artifact, snapshot=snapshot, vault=vault)
    related_report = related_logic.merge_reports(
        state.get("related_merge_report", related_logic.RelatedMergeReport()),  # type: ignore[arg-type]
        composition_related_report,
    )
    _assert_composition_covers_writes(artifact, plan)
    state["composition_plan"] = artifact
    state["related_merge_report"] = related_report
    json_path = out_dir / "composition_plan.json"
    md_path = out_dir / "composition_plan.md"
    related_report_json = out_dir / "related_merge_report.json"
    related_report_md = out_dir / "related_merge_report.md"
    write_json(json_path, artifact)
    write_text(md_path, _render_composition_plan_md(artifact))
    write_json(related_report_json, related_report)
    write_text(related_report_md, related_logic.render_related_report(related_report))
    return StepOutput(
        [json_path, md_path, related_report_json, related_report_md, *provider_artifacts],
        {
            "final_target_count": len(artifact.items),
            "update_target_count": sum(1 for item in artifact.items if item.action == "update"),
            "related_link_count": sum(len(item.related_pages) for item in artifact.items),
            **_token_usage_counts(api_calls),
        },
        model_calls=model_calls,
        api_calls=api_calls,
    )


def _step_final_pages(vault: Path, run_dir: Path, state: dict[str, object]) -> StepOutput:
    composition: CompositionPlan = state["composition_plan"]  # type: ignore[assignment]
    candidate_pages: CandidatePages = state["candidate_pages"]  # type: ignore[assignment]
    snapshot: WikiSnapshot = state["wiki_snapshot"]  # type: ignore[assignment]
    profile: Profile = state["profile"]  # type: ignore[assignment]
    artifacts: list[Path] = []
    out_dir = run_dir / "final_pages"
    outputs, provider_artifacts, model_calls, api_calls = _call_provider_artifacts_parallel(
        state,
        out_dir,
        "final_pages",
        [
            (
                item.final_page_id,
                prompts.final_page_prompt(composition_item=item, candidate_pages=candidate_pages, snapshot=snapshot, profile=profile),
            )
            for item in composition.items
        ],
        FinalPages,
    )
    artifact = _hydrate_provider_final_pages(_merge_parallel_final_pages(outputs, composition), composition, snapshot)
    parallel_request_count = len(composition.items)
    artifact = _normalize_final_pages(
        artifact,
        composition,
        snapshot=snapshot,
        operation_id=str(state.get("operation_id", "")),
    )
    _assert_final_pages_cover_composition(artifact, composition)
    _assert_final_pages_chinese(artifact)
    for final in artifact.pages:
        page_path = out_dir / "pages" / final.target_path
        write_text(page_path, final.markdown)
        artifacts.append(page_path)
        old_text = read_text(vault / "wiki" / final.target_path) if (vault / "wiki" / final.target_path).exists() else ""
        diff_text = "".join(
            difflib.unified_diff(
                old_text.splitlines(keepends=True),
                final.markdown.splitlines(keepends=True),
                fromfile=f"a/wiki/{final.target_path}",
                tofile=f"b/wiki/{final.target_path}",
            )
        )
        diff_path = out_dir / "diffs" / f"{safe_filename(final.target_path)}.diff"
        write_text(diff_path, diff_text)
        artifacts.append(diff_path)
    state["final_pages"] = artifact
    json_path = out_dir / "final_pages.json"
    manifest_path = out_dir / "final_page_manifest.json"
    write_json(json_path, artifact)
    write_json(manifest_path, [{"target_path": page.target_path, "sha256": page.content_sha256, "action": page.action} for page in artifact.pages])
    artifacts.extend([json_path, manifest_path, *provider_artifacts])
    return StepOutput(
        artifacts,
        {
            "final_page_count": len(artifact.pages),
            "diff_count": len(artifact.pages),
            "parallel_request_count": parallel_request_count if model_calls else 0,
            "parallel_max_workers": _page_generation_parallelism(state, parallel_request_count) if model_calls else 0,
            **_token_usage_counts(api_calls),
        },
        model_calls=model_calls,
        api_calls=api_calls,
    )


def _step_validation(vault: Path, run_dir: Path, state: dict[str, object]) -> StepOutput:
    report = _validate_before_write(vault, state)
    out_dir = run_dir / "validation"
    json_path = out_dir / "validation_report.json"
    md_path = out_dir / "validation_report.md"
    write_json(json_path, report)
    write_text(md_path, _render_validation_md(report))
    if not report.ok:
        errors = "; ".join(issue.message for issue in report.issues if issue.severity == "error")
        raise PipelineError(f"校验失败：{errors}")
    return StepOutput([json_path, md_path], {"error_count": 0, "warning_count": sum(1 for issue in report.issues if issue.severity == "warning")})


def _step_knowledge_write(vault: Path, run_dir: Path, state: dict[str, object]) -> StepOutput:
    final_pages: FinalPages = state["final_pages"]  # type: ignore[assignment]
    merge_plan: MergePlan = state["merge_plan"]  # type: ignore[assignment]
    _preflight_knowledge_write(vault, final_pages)
    writes: dict[str, str] = {page.target_path: page.markdown for page in final_pages.pages}
    items: list[WriteSetItem] = []
    preimages = []
    for target, content in writes.items():
        target_abs = ensure_under(vault / "wiki" / target, vault / "wiki", label="knowledge write target")
        existed = target_abs.exists()
        pre_sha = sha256_file(target_abs) if existed else None
        preimages.append({"target_path": target, "existed": existed, "sha256": pre_sha})
        items.append(WriteSetItem(kind="knowledge", target_path=target, content_sha256=sha256_text(content), preimage_sha256=pre_sha))
    write_set = WriteSet(items=items, write_set_sha256=stable_json_hash([item.model_dump(mode="json") for item in items]))
    out_dir = run_dir / "knowledge_write"
    write_set_path = out_dir / "write_set.json"
    preimages_path = out_dir / "preimages.json"
    write_json(write_set_path, write_set)
    write_json(preimages_path, preimages)
    for target, content in writes.items():
        atomic_write_text(vault / "wiki" / target, content)
    result = WriteResult(written_targets=sorted(writes))
    result_path = out_dir / "write_result.json"
    write_json(result_path, result)
    state["write_set_items"] = items
    state["written_targets"] = sorted(writes)
    return StepOutput(
        [write_set_path, preimages_path, result_path],
        {"knowledge_written_count": len(writes), **{f"{key}_count": value for key, value in merge_plan.action_counts.items()}},
    )


def _step_source_record_write(vault: Path, run_dir: Path, state: dict[str, object], manifest: OperationManifest) -> StepOutput:
    final_pages: FinalPages = state["final_pages"]  # type: ignore[assignment]
    digest: SourceDigest = state["source_digest"]  # type: ignore[assignment]
    binding: RawBinding = state["raw_binding"]  # type: ignore[assignment]
    source_target = _source_page_target(binding.raw_path)
    content = _render_source_page(manifest.operation_id, digest, final_pages)
    item = _write_single_target(vault, source_target, "source", content, label="source record target")
    state["write_set_items"] = [*state.get("write_set_items", []), item]  # type: ignore[list-item]
    state["written_targets"] = sorted([*state.get("written_targets", []), source_target])  # type: ignore[list-item]
    out_dir = run_dir / "source_record_write"
    result = WriteResult(written_targets=[source_target])
    result_path = out_dir / "write_result.json"
    source_path = out_dir / "source_page.md"
    item_path = out_dir / "write_item.json"
    write_json(result_path, result)
    write_text(source_path, content)
    write_json(item_path, item)
    return StepOutput([result_path, source_path, item_path], {"source_record_count": 1, "written_target_count": 1})


def _step_index_log_write(vault: Path, run_dir: Path, state: dict[str, object], manifest: OperationManifest) -> StepOutput:
    binding: RawBinding = state["raw_binding"]  # type: ignore[assignment]
    merge_plan: MergePlan = state["merge_plan"]  # type: ignore[assignment]
    profile: Profile = state["profile"]  # type: ignore[assignment]
    log_date = _operation_date(manifest.operation_id)
    daily_target = f"logs/{log_date}.md"
    writes = {
        "index.md": _updated_index(vault, profile),
        "log.md": _updated_log_index(vault, manifest.operation_id, binding),
        daily_target: _updated_daily_log(vault, log_date, manifest.operation_id, binding, merge_plan),
    }
    out_dir = run_dir / "index_log_write"
    items = []
    rendered_artifacts: list[Path] = []
    for target, content in writes.items():
        item = _write_single_target(vault, target, "system", content, label="system write target")
        items.append(item)
        artifact_path = out_dir / "pages" / target
        write_text(artifact_path, content)
        rendered_artifacts.append(artifact_path)
    state["write_set_items"] = [*state.get("write_set_items", []), *items]  # type: ignore[list-item]
    state["written_targets"] = sorted([*state.get("written_targets", []), *writes.keys()])  # type: ignore[list-item]
    result = WriteResult(written_targets=sorted(writes))
    result_path = out_dir / "write_result.json"
    items_path = out_dir / "write_items.json"
    write_json(result_path, result)
    write_json(items_path, items)
    return StepOutput([result_path, items_path, *rendered_artifacts], {"system_written_count": len(writes), "written_target_count": len(writes)})


def _write_single_target(vault: Path, target: str, kind: Literal["source", "system"], content: str, *, label: str) -> WriteSetItem:
    target_abs = ensure_under(vault / "wiki" / target, vault / "wiki", label=label)
    existed = target_abs.exists()
    pre_sha = sha256_file(target_abs) if existed else None
    item = WriteSetItem(kind=kind, target_path=target, content_sha256=sha256_text(content), preimage_sha256=pre_sha)
    atomic_write_text(target_abs, content)
    return item


def _step_embedding_cache_refresh(vault: Path, run_dir: Path, state: dict[str, object]) -> StepOutput:
    profile: Profile = state["profile"]  # type: ignore[assignment]
    embedding_config: EmbeddingConfig = state["embedding_config"]  # type: ignore[assignment]
    entries = _scan_wiki_entries(vault, profile)
    _, metrics = sync_page_embedding_cache(vault, entries, embedding_config)
    state["embedding_refresh_metrics"] = metrics
    out_dir = run_dir / "embedding_cache_refresh"
    report_path = out_dir / "embedding_cache_refresh.json"
    write_json(report_path, metrics)
    return StepOutput(
        [report_path],
        {
            "knowledge_pool_size": len(entries),
            "cache_hit": int(metrics.get("cache_hit", 0)),
            "cache_refreshed": int(metrics.get("cache_refreshed", 0)),
            "cache_pruned": int(metrics.get("cache_pruned", 0)),
            "retrieval_backend": str(metrics.get("backend", embedding_config.backend)),
        },
    )


def _step_receipt(vault: Path, run_dir: Path, state: dict[str, object], manifest: OperationManifest) -> StepOutput:
    binding: RawBinding = state["raw_binding"]  # type: ignore[assignment]
    merge_plan: MergePlan = state["merge_plan"]  # type: ignore[assignment]
    snapshot: WikiSnapshot = state["wiki_snapshot"]  # type: ignore[assignment]
    embedding_metrics = state.get("embedding_refresh_metrics", snapshot.embedding_metrics)
    written_targets: list[str] = sorted(state.get("written_targets", []))  # type: ignore[arg-type]
    artifact_hashes = _important_artifact_hashes(run_dir)
    receipt = Receipt(
        operation_id=manifest.operation_id,
        raw_path=binding.raw_path,
        raw_sha256=binding.raw_sha256,
        artifact_hashes=artifact_hashes,
        written_targets=written_targets,
        action_counts=merge_plan.action_counts,
        warnings=manifest.warnings,
        provider_contexts=state.get("provider_contexts", {}),  # type: ignore[arg-type]
        model_call_count=sum(step.model_calls for step in manifest.steps),
        repair_count=sum(step.repair_count for step in manifest.steps),
        embedding_metrics=embedding_metrics,  # type: ignore[arg-type]
        engine_version=__version__,
        profile_name=manifest.profile_name,
        created_at=now_utc(),
    )
    out_dir = run_dir / "receipt"
    receipt_path = out_dir / "receipt.json"
    if receipt_path.exists():
        raise PipelineError(f"回执已存在：{relative_posix(receipt_path, vault)}")
    write_json(receipt_path, receipt)
    append_jsonl(vault / ".llmwiki" / "applied" / "operations.jsonl", receipt)
    manifest.receipt_path = relative_posix(receipt_path, vault)
    return StepOutput([receipt_path], {"receipt_count": 1, "written_target_count": len(written_targets)})


def _run_step(
    run_dir: Path,
    manifest: OperationManifest,
    name: str,
    fn: Callable[[], StepOutput],
    console: Console,
    *,
    emit_progress: bool,
) -> None:
    use_live_status = emit_progress and console.is_terminal and name in MODEL_BACKED_STEPS
    if emit_progress and not use_live_status:
        console.print(f"[cyan]开始[/] {step_label(name)}")
    started = now_utc()
    start_time = time.perf_counter()
    record = StepRecord(name=name, status="running", started_at=started)
    manifest.steps.append(record)
    manifest.updated_at = now_utc()
    _write_manifest(run_dir, manifest)
    _write_event(run_dir, "step_started", {"step": name})
    try:
        if use_live_status:
            with console.status(f"[cyan]运行[/] {step_label(name)}", spinner="dots"):
                output = fn()
        else:
            output = fn()
    except Exception:
        record.status = "failed"
        record.finished_at = now_utc()
        record.duration_seconds = round(time.perf_counter() - start_time, 4)
        manifest.updated_at = now_utc()
        _write_manifest(run_dir, manifest)
        _write_event(run_dir, "step_failed", {"step": name})
        if emit_progress:
            console.print(f"[red]失败[/] {step_label(name)} {record.duration_seconds:.2f}s")
        raise
    record.status = "completed"
    record.finished_at = now_utc()
    record.duration_seconds = round(time.perf_counter() - start_time, 4)
    record.artifacts = [_artifact_ref(run_dir, path) for path in output.artifacts]
    record.counts = output.counts
    record.warnings = output.warnings or []
    record.model_calls = output.model_calls
    record.repair_count = output.repair_count
    manifest.updated_at = now_utc()
    _write_manifest(run_dir, manifest)
    _write_event(run_dir, "step_completed", {"step": name, "counts": output.counts})
    if emit_progress:
        if output.api_calls:
            _print_api_call_table(console, name, output.api_calls)
        visible_counts: dict[str, int | float | str] = {
            "artifact_count": len(output.artifacts),
            "model_calls": output.model_calls,
            "repair_count": output.repair_count,
            **output.counts,
        }
        count_text = " ".join(f"{count_label(key)}={_format_count_value(key, value)}" for key, value in visible_counts.items())
        console.print(f"[green]完成[/] {step_label(name)} {record.duration_seconds:.2f}s {count_text}".rstrip())


def _print_api_call_table(console: Console, step_name: str, api_calls: list[dict[str, object]]) -> None:
    models = {str(call.get("model") or "") for call in api_calls if call.get("model")}
    model_suffix = f" · {next(iter(models))}" if len(models) == 1 else ""
    table = Table(title=f"{step_label(step_name)} API 调用{model_suffix}")
    table.add_column("状态", justify="center", no_wrap=True)
    table.add_column("请求")
    table.add_column("输入", justify="right")
    table.add_column("缓存", justify="right")
    table.add_column("输出", justify="right")
    table.add_column("思考", justify="right")
    table.add_column("命中率", justify="right")
    table.add_column("耗时", justify="right")
    table.add_column("价格", justify="right")
    for call in api_calls:
        status = "[green]✓[/]" if call.get("status") == "success" else "[red]⏸[/]"
        request_key = str(call.get("request_key") or call.get("step") or "")
        call_index = int(call.get("call_index") or 0)
        if call_index > 1:
            request_key = f"{request_key}#{call_index}"
        table.add_row(
            status,
            request_key,
            str(call.get("prompt_tokens") or 0),
            str(call.get("prompt_cache_hit_tokens") or 0),
            str(call.get("completion_tokens") or 0),
            str(call.get("reasoning_tokens") or 0),
            format_percent(call.get("cache_hit_rate_percent")),
            format_duration_ms(call.get("duration_ms")),
            format_price_cny(call.get("price_cny")),
        )
    console.print(table)


def _format_count_value(key: str, value: object) -> object:
    if key == "price_cny":
        return format_price_cny(value)
    if key == "cache_hit_rate_percent":
        return format_percent(value)
    return value


def _chinese_scaffold(text: str, label: str) -> str:
    stripped = text.strip()
    if contains_cjk(stripped):
        return stripped
    if stripped:
        return f"{label}：{stripped}"
    return f"{label}：暂无可用内容。"


def _scan_wiki_entries(vault: Path, profile: Profile) -> list[WikiKnowledgeEntry]:
    wiki_root = vault / "wiki"
    entries = []
    source_dir = profile.page_type(profile.source_page_type).directory
    for path in sorted(wiki_root.rglob("*.md")):
        rel = relative_posix(path, wiki_root)
        if rel in {"index.md", "log.md"} or rel.startswith("logs/") or rel.startswith(f"{source_dir}/"):
            continue
        text = read_text(path)
        frontmatter = _frontmatter_data(text) or {}
        title = _title_from_existing_page(text, path.stem)
        page_type = str(frontmatter.get("type") or frontmatter.get("llmwiki_type") or _page_type_from_path(profile, rel))
        if page_type == profile.source_page_type:
            continue
        summary = str(frontmatter.get("summary") or summarize(strip_frontmatter(text), max_sentences=2))
        source_raw_paths = _frontmatter_list(frontmatter, "source_raw_paths")
        source_raw_hashes = _frontmatter_list(frontmatter, "source_raw_hashes")
        entries.append(
            WikiKnowledgeEntry(
                path=rel,
                title=title,
                page_type=page_type,
                sha256=sha256_file(path),
                summary=summary,
                aliases=_frontmatter_list(frontmatter, "aliases"),
                created=str(frontmatter.get("created") or ""),
                source_raw_paths=source_raw_paths,
                source_raw_hashes=source_raw_hashes,
                source_prepared_hashes=_frontmatter_list(frontmatter, "source_prepared_hashes"),
                source_operation_ids=_frontmatter_list(frontmatter, "source_operation_ids"),
                updated=str(frontmatter.get("updated") or ""),
                text_excerpt=strip_frontmatter(text)[:1200],
            )
        )
    return entries


def _validate_before_write(vault: Path, state: dict[str, object]) -> ValidationReport:
    issues: list[ValidationIssue] = []
    binding: RawBinding = state["raw_binding"]  # type: ignore[assignment]
    raw_abs: Path = state["raw_abs"]  # type: ignore[assignment]
    final_pages: FinalPages = state["final_pages"]  # type: ignore[assignment]
    profile: Profile = state["profile"]  # type: ignore[assignment]
    merge_plan: MergePlan | None = state.get("merge_plan") if isinstance(state.get("merge_plan"), MergePlan) else None  # type: ignore[assignment]
    composition_plan: CompositionPlan | None = state.get("composition_plan") if isinstance(state.get("composition_plan"), CompositionPlan) else None  # type: ignore[assignment]
    snapshot: WikiSnapshot | None = state.get("wiki_snapshot") if isinstance(state.get("wiki_snapshot"), WikiSnapshot) else None  # type: ignore[assignment]
    issues.extend(related_logic.validate_related_plan(merge_plan=merge_plan, composition_plan=composition_plan, snapshot=snapshot))
    if sha256_file(raw_abs) != binding.raw_sha256:
        issues.append(ValidationIssue(severity="error", code="raw_hash_drift", message="raw 在 ingest 过程中发生变化。", path=binding.raw_path))
    targets = [page.target_path for page in final_pages.pages]
    for target, count in Counter(targets).items():
        if count > 1:
            issues.append(ValidationIssue(severity="error", code="duplicate_target", message="最终写入目标重复。", path=target))
    for page in final_pages.pages:
        try:
            ensure_under(vault / "wiki" / page.target_path, vault / "wiki", label="final target")
        except ValueError as exc:
            issues.append(ValidationIssue(severity="error", code="target_escape", message=f"最终页面目标路径越界：{exc}", path=page.target_path))
        if not _is_knowledge_target(profile, page):
            issues.append(ValidationIssue(severity="error", code="invalid_knowledge_target", message="最终页面目标不在允许的知识页目录中。", path=page.target_path))
        if sha256_text(page.markdown) != page.content_sha256:
            issues.append(ValidationIssue(severity="error", code="content_hash_mismatch", message="最终页面内容 hash 与 Markdown 不一致。", path=page.target_path))
        if not page.markdown.strip():
            issues.append(ValidationIssue(severity="error", code="empty_final_page", message="最终页面为空。", path=page.target_path))
        if not page.source_refs:
            issues.append(ValidationIssue(severity="error", code="missing_source_refs", message="最终页面缺少 source refs。", path=page.target_path))
        elif not any(ref.raw_path == binding.raw_path and ref.raw_sha256 == binding.raw_sha256 for ref in page.source_refs):
            issues.append(ValidationIssue(severity="error", code="source_ref_mismatch", message="最终页面没有包含本次绑定的 raw source ref。", path=page.target_path))
        frontmatter_issues = _validate_final_markdown_frontmatter(page)
        issues.extend(frontmatter_issues)
        issues.extend(related_logic.final_markdown_link_issues(markdown=page.markdown, target_path=page.target_path, title=page.title))
        existing = vault / "wiki" / page.target_path
        if existing.exists():
            current_sha = sha256_file(existing)
            if page.preimage_sha256 is None:
                issues.append(ValidationIssue(severity="error", code="unsafe_overwrite", message="目标已存在，但最终页面没有 preimage。", path=page.target_path))
            elif current_sha != page.preimage_sha256:
                issues.append(ValidationIssue(severity="error", code="preimage_drift", message="目标在快照后发生变化。", path=page.target_path))
        elif page.action == "update":
            issues.append(ValidationIssue(severity="error", code="update_missing_preimage", message="update 目标不存在。", path=page.target_path))
        elif page.preimage_sha256 is not None:
            issues.append(ValidationIssue(severity="error", code="preimage_target_missing", message="最终页面有 preimage，但目标不存在。", path=page.target_path))
    return ValidationReport(ok=not any(issue.severity == "error" for issue in issues), issues=issues)


def _preflight_knowledge_write(vault: Path, final_pages: FinalPages) -> None:
    for page in final_pages.pages:
        target_abs = ensure_under(vault / "wiki" / page.target_path, vault / "wiki", label="write target")
        exists = target_abs.exists()
        current_sha = sha256_file(target_abs) if exists else None
        if current_sha != page.preimage_sha256:
            raise PipelineError(f"{page.target_path} 的写入 preimage 不一致。")


def _is_knowledge_target(profile: Profile, page: FinalPage) -> bool:
    if page.target_path in {"index.md", "log.md"} or page.target_path.startswith("logs/"):
        return False
    source_dir = profile.page_type(profile.source_page_type).directory
    if page.target_path.startswith(f"{source_dir}/"):
        return False
    spec = profile.page_type(page.page_type)
    return page.target_path.startswith(f"{spec.directory}/") and page.target_path.endswith(".md")


def _validate_final_markdown_frontmatter(page: FinalPage) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    data = _frontmatter_data(page.markdown)
    if data is None:
        return [ValidationIssue(severity="error", code="missing_frontmatter", message="最终 Markdown 缺少 YAML frontmatter。", path=page.target_path)]
    if data.get("title") != page.title:
        issues.append(ValidationIssue(severity="error", code="frontmatter_title_mismatch", message="frontmatter title 与最终页面标题不一致。", path=page.target_path))
    if data.get("llmwiki_type") != page.page_type:
        issues.append(ValidationIssue(severity="error", code="frontmatter_type_mismatch", message="frontmatter llmwiki_type 与最终页面类型不一致。", path=page.target_path))
    for forbidden in ["type", "source_refs", "llmwiki"]:
        if forbidden in data:
            issues.append(ValidationIssue(severity="error", code="frontmatter_legacy_lite_field", message=f"frontmatter 不能包含 Lite-only 字段 {forbidden}。", path=page.target_path))
    aliases = data.get("aliases")
    if not isinstance(aliases, list):
        issues.append(ValidationIssue(severity="error", code="frontmatter_missing_aliases", message="frontmatter aliases 必须是列表。", path=page.target_path))
    for required in ["source_raw_paths", "source_raw_hashes", "source_prepared_hashes", "source_operation_ids"]:
        values = data.get(required)
        if not isinstance(values, list) or not values:
            issues.append(ValidationIssue(severity="error", code="frontmatter_missing_provenance", message=f"frontmatter {required} 缺失或为空。", path=page.target_path))
    if not data.get("created") or not data.get("updated") or not data.get("last_ingest_operation"):
        issues.append(ValidationIssue(severity="error", code="frontmatter_missing_dates", message="frontmatter 必须包含 created、updated 和 last_ingest_operation。", path=page.target_path))
    return issues


def _frontmatter_data(markdown: str) -> dict[str, object] | None:
    if not markdown.startswith("---\n"):
        return None
    end = markdown.find("\n---", 4)
    if end == -1:
        return None
    try:
        data = yaml.safe_load(markdown[4:end]) or {}
    except yaml.YAMLError:
        return None
    return data if isinstance(data, dict) else None


def _drop_sections(markdown: str, headings: set[str]) -> str:
    wanted = {heading.lower() for heading in headings}
    lines = markdown.splitlines()
    result: list[str] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped.startswith("## ") and stripped[3:].strip().lower() in wanted:
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("## "):
                index += 1
            continue
        result.append(lines[index])
        index += 1
    return "\n".join(result)


def _merge_source_refs(refs: list[SourceRef]) -> list[SourceRef]:
    seen = set()
    merged = []
    for ref in refs:
        key = (ref.raw_path, ref.raw_sha256, ref.locator)
        if key in seen:
            continue
        seen.add(key)
        merged.append(ref)
    return merged


def _merge_related_refs(refs: list[RelatedPageRef], *, current_path: str) -> list[RelatedPageRef]:
    current = system_pages.normalize_related_path(current_path)
    seen: set[str] = set()
    merged: list[RelatedPageRef] = []
    for ref in refs:
        normalized = system_pages.normalize_related_path(ref.target_path)
        if normalized is None or normalized == current or normalized in seen:
            continue
        seen.add(normalized)
        merged.append(ref.model_copy(update={"target_path": normalized}))
        if len(merged) >= system_pages.RELATED_LINK_LIMIT:
            break
    return merged


def _yaml_scalar(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _yaml_list(key: str, values: list[str]) -> str:
    if not values:
        return f"{key}: []\n"
    return f"{key}:\n" + "".join(f"  - {_yaml_scalar(value)}\n" for value in values)


def _render_source_page(operation_id: str, digest: SourceDigest, final_pages: FinalPages) -> str:
    derived = "\n".join(f"- `{page.target_path}`" for page in final_pages.pages) or "- 暂无派生知识页。"
    takeaways = "\n".join(f"- {item}" for item in digest.key_takeaways) or "- 暂无关键收获记录。"
    title = f"来源 {Path(digest.source_raw_path).name}"
    log_date = _operation_date(operation_id)
    frontmatter = (
        "---\n"
        "llmwiki_type: source\n"
        f"title: {_yaml_scalar(title)}\n"
        "aliases: []\n"
        f"summary: {_yaml_scalar(digest.summary)}\n"
        f"created: {log_date}\n"
        f"updated: {log_date}\n"
        "source_raw_paths:\n"
        f"  - {_yaml_scalar(digest.source_raw_path)}\n"
        "source_raw_hashes:\n"
        f"  - {_yaml_scalar(digest.raw_sha256)}\n"
        "source_prepared_hashes:\n"
        f"  - {_yaml_scalar(digest.raw_sha256)}\n"
        "source_operation_ids:\n"
        f"  - {_yaml_scalar(operation_id)}\n"
        f"last_ingest_operation: {_yaml_scalar(operation_id)}\n"
        "---"
    )
    return (
        f"{frontmatter}\n\n"
        f"# {title}\n\n"
        f"## 摘要\n\n{digest.summary}\n\n"
        f"## 原始材料\n\n- `{digest.source_raw_path}`\n\n"
        f"## 关键收获\n\n{takeaways}\n\n"
        f"## 派生知识页\n\n{derived}\n"
    )


def _updated_index(vault: Path, profile: Profile) -> str:
    entries = _scan_wiki_entries(vault, profile)
    return system_pages.render_index(
        entries=entries,
        tension_rows=_index_tension_rows(vault, entries),
        page_type_order=profile.page_types.keys(),
    )


def _updated_log_index(vault: Path, operation_id: str, binding: RawBinding) -> str:
    log_path = vault / "wiki" / "log.md"
    existing = read_text(log_path) if log_path.exists() else None
    return system_pages.render_log_index(date=_operation_date(operation_id), operation_id=operation_id, raw_path=binding.raw_path, existing_text=existing)


def _updated_daily_log(vault: Path, log_date: str, operation_id: str, binding: RawBinding, merge_plan: MergePlan) -> str:
    daily_path = vault / "wiki" / "logs" / f"{log_date}.md"
    existing = read_text(daily_path) if daily_path.exists() else None
    counts = system_pages.DailyLogCounts(
        created=merge_plan.action_counts.get("create", 0),
        updated=merge_plan.action_counts.get("update", 0),
        noop=merge_plan.action_counts.get("noop", 0),
    )
    return system_pages.render_daily_log(date=log_date, operation_id=operation_id, raw_path=binding.raw_path, counts=counts, existing_text=existing)


def _index_tension_rows(vault: Path, entries: list[WikiKnowledgeEntry]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for entry in entries:
        page_path = vault / "wiki" / entry.path
        if not page_path.exists():
            continue
        body = _section_text(read_text(page_path), {"Tensions / Open Questions", "矛盾与未决问题"})
        for question in _question_lines(body):
            rows.append({"question": question, "page": system_pages.obsidian_link(entry.path, entry.title), "updated": entry.updated})
    return rows


def _section_text(markdown: str, headings: set[str]) -> str:
    wanted = {heading.lower() for heading in headings}
    lines = strip_frontmatter(markdown).splitlines()
    capture = False
    body: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            title = stripped[3:].strip().lower()
            if capture:
                break
            capture = title in wanted
            continue
        if capture:
            body.append(line)
    return "\n".join(body).strip()


def _question_lines(text: str) -> list[str]:
    rows: list[str] = []
    ignored = {"none captured yet.", "(no tensions or open questions raised by this source.)", "暂无矛盾与未决问题记录。"}
    for line in text.splitlines():
        stripped = line.strip().lstrip("-*+ ").strip()
        if not stripped or stripped.lower() in ignored:
            continue
        rows.append(stripped)
    return rows


def _source_page_target(raw_path: str) -> str:
    return f"sources/Source_{safe_filename(Path(raw_path).stem)}.md"


def _important_artifact_hashes(run_dir: Path) -> dict[str, str]:
    paths = {
        "source_digest": run_dir / "source_digest" / "source_digest.json",
        "candidate_merge": run_dir / "candidate_merge" / "candidate_merge.json",
        "candidate_pages_warmup": run_dir / "candidate_pages_warmup" / "candidate_pages_warmup.json",
        "wiki_snapshot": run_dir / "wiki_snapshot" / "wiki_snapshot.json",
        "candidate_pages": run_dir / "candidate_pages" / "candidate_pages.json",
        "candidate_contexts": run_dir / "candidate_contexts" / "candidate_contexts.json",
        "merge_plan": run_dir / "merge_plan" / "merge_plan.json",
        "composition_plan": run_dir / "composition_plan" / "composition_plan.json",
        "related_merge_report": run_dir / "composition_plan" / "related_merge_report.json",
        "final_pages": run_dir / "final_pages" / "final_pages.json",
        "knowledge_write_set": run_dir / "knowledge_write" / "write_set.json",
        "source_record_write": run_dir / "source_record_write" / "write_result.json",
        "index_log_write": run_dir / "index_log_write" / "write_result.json",
        "embedding_cache_refresh": run_dir / "embedding_cache_refresh" / "embedding_cache_refresh.json",
    }
    return {name: sha256_file(path) for name, path in paths.items() if path.exists()}


def _render_source_digest_md(digest: SourceDigest) -> str:
    lines = [f"# 来源消化：{Path(digest.source_raw_path).name}", "", digest.summary, "", "## 候选知识项"]
    for candidate in digest.candidates():
        lines.append(f"- `{candidate.candidate_id}` {candidate.kind}: {candidate.name} - {candidate.summary}")
        if candidate.related_candidates:
            lines.append(f"  - 相关候选：{', '.join(candidate.related_candidates)}")
    return "\n".join(lines) + "\n"


def _render_candidate_merge_md(plan: CandidateMergePlan) -> str:
    lines = ["# 候选合并计划", ""]
    for unit in plan.units:
        lines.append(f"## {unit.candidate_unit_id}: {unit.title}")
        lines.append(f"- 类型：{unit.page_type}")
        lines.append(f"- 来源候选：{', '.join(unit.source_candidate_ids)}")
        lines.append(f"- 路径提示：`{unit.path_hint}`")
        lines.append(f"- 合并理由：{unit.merge_reason}")
        if unit.must_cover_points:
            lines.append("- 必须覆盖：")
            lines.extend(f"  - {item}" for item in unit.must_cover_points)
        lines.append("")
    if plan.skipped_candidate_ids:
        lines.append("## 跳过候选")
        lines.extend(f"- `{item}`" for item in plan.skipped_candidate_ids)
        lines.append("")
    return "\n".join(lines)


def _render_contexts_md(contexts: Iterable[CandidateContext]) -> str:
    lines = ["# 候选页召回上下文", ""]
    for context in contexts:
        lines.append(f"## {context.candidate_page_id}")
        if context.hits:
            for hit in context.hits:
                lines.append(f"- 第 {hit.rank} 名 `{hit.path}`；分数={hit.score}；依据={hit.match_basis}：{hit.reason}")
        else:
            lines.append("- 暂无召回命中。")
        lines.append("")
    return "\n".join(lines)


def _render_candidate_pages_md(artifact: CandidatePages) -> str:
    lines = ["# 候选知识页", ""]
    for page in artifact.pages:
        lines.append(f"## {page.candidate_page_id}: {page.title}")
        lines.append(f"- 候选单元：`{page.candidate_unit_id}`")
        lines.append(f"- 类型：{page.proposed_page_type}")
        lines.append(f"- 路径提示：`{page.proposed_path_hint}`")
        lines.append("")
        lines.append(page.summary)
        lines.append("")
    return "\n".join(lines)


def _render_merge_plan_md(plan: MergePlan) -> str:
    lines = ["# 合并计划", "", "## 数量统计"]
    for action, count in plan.action_counts.items():
        lines.append(f"- {_action_label(action)}：{count}")
    lines.append("")
    for decision in plan.decisions:
        lines.append(f"## {decision.decision_id}: {decision.candidate_page_id}")
        lines.append(f"- 动作：{_action_label(decision.action)}")
        lines.append(f"- 目标：`{decision.target_path}`")
        lines.append(f"- 标题：{decision.title}")
        lines.append(f"- 类型：{decision.page_type}")
        lines.append(f"- 内容范围：{decision.content_scope}")
        if decision.candidate_path_index:
            lines.append(f"- 候选路径索引：{', '.join(decision.candidate_path_index)}")
        lines.append(f"- 理由：{decision.reason}")
        lines.append(f"- 最强重合度：{decision.strongest_overlap}")
        if decision.related_pages:
            related = ", ".join(f"`{ref.target_path}`" for ref in decision.related_pages)
            lines.append(f"- 相关页面：{related}")
        elif decision.related_absence_reason:
            lines.append(f"- 无相关页面原因：{decision.related_absence_reason}")
        lines.append("")
    return "\n".join(lines)


def _render_composition_plan_md(plan: CompositionPlan) -> str:
    lines = ["# 写作编排计划", ""]
    for item in plan.items:
        lines.append(f"## {item.final_page_id}: {item.target_path}")
        lines.append(f"- 动作：{_action_label(item.action)}")
        lines.append(f"- 合并决策：{', '.join(item.merge_decision_ids)}")
        lines.append(f"- 章节顺序：{', '.join(item.section_order)}")
        if item.related_pages:
            related = ", ".join(f"`{ref.target_path}`" for ref in item.related_pages)
            lines.append(f"- 相关页面：{related}")
        elif item.related_absence_reason:
            lines.append(f"- 无相关页面原因：{item.related_absence_reason}")
        lines.append("")
    return "\n".join(lines)


def _render_validation_md(report: ValidationReport) -> str:
    lines = [f"# 校验结果：{'通过' if report.ok else '失败'}", ""]
    for issue in report.issues:
        lines.append(f"- {_severity_label(issue.severity)} `{issue.code}`：{issue.message}")
    if not report.issues:
        lines.append("- 暂无问题。")
    return "\n".join(lines) + "\n"


def _action_label(action: str) -> str:
    return {
        "create": "新建",
        "update": "更新",
        "noop": "不改动",
    }.get(action, action)


def _severity_label(severity: str) -> str:
    return {"error": "错误", "warning": "警告"}.get(severity, severity)


def _title_from_existing_page(text: str, fallback: str) -> str:
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            try:
                data = yaml.safe_load(text[4:end]) or {}
                if isinstance(data, dict) and data.get("title"):
                    return str(data["title"])
            except yaml.YAMLError:
                pass
    return title_from_markdown(strip_frontmatter(text), fallback)


def _frontmatter_list(data: dict[str, object], key: str) -> list[str]:
    value = data.get(key)
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _page_type_from_path(profile: Profile, rel: str) -> str:
    for page_type, spec in profile.page_types.items():
        if rel.startswith(f"{spec.directory}/"):
            return page_type
    return profile.default_page_type


def _artifact_ref(run_dir: Path, path: Path) -> ArtifactRef:
    sha, size = artifact_hash(path)
    return ArtifactRef(path=relative_posix(path, run_dir), sha256=sha, size_bytes=size)


def _write_manifest(run_dir: Path, manifest: OperationManifest) -> None:
    write_json(run_dir / "manifest.json", manifest)


def _write_event(run_dir: Path, event: str, payload: dict[str, object]) -> None:
    append_jsonl(run_dir / "events.jsonl", {"ts": now_utc(), "event": event, **payload})


def _run_dir(vault: Path, operation_id: str) -> Path:
    return vault.expanduser().resolve() / ".llmwiki" / "runs" / "ingest" / operation_id


def _resolve_raw(vault: Path, raw_file: Path) -> Path:
    raw = raw_file.expanduser()
    raw_abs = raw.resolve() if raw.is_absolute() else (vault / raw).resolve()
    ensure_under(raw_abs, vault / "raw", label="raw_file")
    if not raw_abs.exists():
        raise ValueError(f"raw 文件不存在：{raw_abs}")
    if not raw_abs.is_file():
        raise ValueError(f"raw 路径不是文件：{raw_abs}")
    return raw_abs


def _load_config(vault: Path) -> dict[str, object]:
    config_path = vault / ".llmwiki" / "config.json"
    if not config_path.exists():
        raise PipelineError("Lite vault 缺少 .llmwiki/config.json，请先运行 llmwiki init。")
    return read_json(config_path)


def _ensure_gitignore(vault: Path) -> None:
    path = vault / ".gitignore"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    needed = [".llmwiki/", ".DS_Store"]
    changed = False
    for item in needed:
        if item not in lines:
            lines.append(item)
            changed = True
    if changed:
        write_text(path, "\n".join(lines).rstrip() + "\n")


def _assert_llmwiki_not_tracked(vault: Path) -> None:
    if not (vault / ".git").exists():
        return
    try:
        result = subprocess.run(
            ["git", "-C", vault.as_posix(), "ls-files", ".llmwiki"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return
    if result.stdout.strip():
        raise PipelineError(".llmwiki 不能被 git 跟踪。")


def _processed_raw_paths(vault: Path) -> set[str]:
    path = vault / ".llmwiki" / "applied" / "operations.jsonl"
    if not path.exists():
        return set()
    processed: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        raw_path = payload.get("raw_path")
        if isinstance(raw_path, str):
            processed.add(raw_path)
    return processed
