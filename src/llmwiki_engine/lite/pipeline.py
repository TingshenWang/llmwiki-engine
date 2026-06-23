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
from .embeddings import EmbeddingConfig, build_candidate_contexts, cosine, embed_texts, load_embedding_config, sync_page_embedding_cache
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
    RelatedCandidateReport,
    RelatedPageRef,
    SourceDigest,
    SourceGranularityStats,
    SourcePageUnit,
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


RELATED_MIN_SIMILARITY = 0.72
RELATED_REPLACEMENT_MARGIN = 0.04


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
            "input_version": "page_card_v2",
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
        ("candidate_pages_warmup", lambda: _step_candidate_pages_warmup(run_dir, state)),
        ("candidate_pages", lambda: _step_candidate_pages(run_dir, state)),
        ("wiki_snapshot", lambda: _step_wiki_snapshot(vault, run_dir, state)),
        ("candidate_contexts", lambda: _step_candidate_contexts(run_dir, state)),
        ("merge_plan", lambda: _step_merge_plan(run_dir, state)),
        ("composition_plan", lambda: _step_composition_plan(run_dir, state)),
        ("final_pages", lambda: _step_final_pages(vault, run_dir, state)),
        ("related_refresh", lambda: _step_related_refresh(run_dir, state)),
        ("validation", lambda: _step_validation(vault, run_dir, state)),
        ("knowledge_write", lambda: _step_knowledge_write(vault, run_dir, state)),
        ("source_record_write", lambda: _step_source_record_write(vault, run_dir, state, manifest)),
        ("embedding_cache_refresh", lambda: _step_embedding_cache_refresh(vault, run_dir, state)),
        ("related_maintenance", lambda: _step_related_maintenance(vault, run_dir, state)),
        ("index_log_write", lambda: _step_index_log_write(vault, run_dir, state, manifest)),
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


def normalize_source_digest(digest: SourceDigest, profile: Profile) -> tuple[SourceDigest, StructuredRepairReport]:
    repairs: list[RepairItem] = []

    page_units: list[SourcePageUnit] = []
    for index, unit in enumerate(digest.page_units, start=1):
        title = unit.title.strip()
        if not title:
            title = f"未命名页面单元 {index}"
            repairs.append(RepairItem(path=f"page_units[{index - 1}].title", reason="title 为空，已使用默认标题补齐。", local_fix=True, model_called=False))
        page_type = unit.page_type if unit.page_type in profile.page_types and unit.page_type != profile.source_page_type else profile.default_page_type
        if page_type != unit.page_type:
            repairs.append(RepairItem(path=f"page_units[{index - 1}].page_type", reason="page_type 不在 profile 中或指向 source 类型，已改为默认知识页类型。", local_fix=True, model_called=False))
        path_hint = _normalize_path_hint(unit.path_hint, page_type, title, profile)
        if path_hint != unit.path_hint:
            repairs.append(RepairItem(path=f"page_units[{index - 1}].path_hint", reason="path_hint 越界或格式不正确，已按页面类型和标题重建。", local_fix=True, model_called=False))
        must_cover_points = _dedupe_list([item.strip() for item in unit.must_cover_points if item.strip()]) or [unit.summary]
        page_units.append(
            unit.model_copy(
                update={
                    "page_unit_id": f"PU-{index:03d}",
                    "title": title,
                    "page_type": page_type,
                    "path_hint": path_hint,
                    "must_cover_points": must_cover_points,
                    "source_refs": _merge_source_refs(unit.source_refs),
                }
            )
        )
    updated = digest.model_copy(update={"page_units": page_units})
    return updated, StructuredRepairReport(repairs=repairs, model_calls=0)


def _source_granularity_stats(raw_text: str, *, raw_size_bytes: int) -> SourceGranularityStats:
    text_without_frontmatter = strip_frontmatter(raw_text)
    code_block_count = len(re.findall(r"```.*?```", text_without_frontmatter, flags=re.S))
    text_without_code = re.sub(r"```.*?```", "\n", text_without_frontmatter, flags=re.S)
    heading_count = len(re.findall(r"(?m)^\s{0,3}#{1,6}\s+\S", text_without_code))
    markdown_link_count = len(re.findall(r"\[[^\]]+\]\([^)]+\)", text_without_code))
    navigation_noise_line_count = sum(1 for line in text_without_code.splitlines() if _is_navigation_noise_line(line))
    content_lines = [line for line in text_without_code.splitlines() if not _is_navigation_noise_line(line)]
    content_text = "\n".join(content_lines)
    paragraph_count = len([part for part in re.split(r"\n\s*\n", content_text) if part.strip()])
    char_count = len(re.sub(r"\s+", "", text_without_frontmatter))
    effective_char_count = len(re.sub(r"\s+", "", content_text))
    target = _suggested_page_unit_target(effective_char_count)
    suggested_min = max(1, int(target * 0.6))
    suggested_max = max(suggested_min, int(target * 1.45 + 0.999))
    if effective_char_count <= 1200:
        suggested_max = 1
    return SourceGranularityStats(
        raw_size_bytes=raw_size_bytes,
        char_count=char_count,
        effective_char_count=effective_char_count,
        paragraph_count=paragraph_count,
        heading_count=heading_count,
        code_block_count=code_block_count,
        markdown_link_count=markdown_link_count,
        navigation_noise_line_count=navigation_noise_line_count,
        suggested_min_page_units=suggested_min,
        suggested_target_page_units=round(target, 2),
        suggested_max_page_units=suggested_max,
        range_basis="初始经验公式：target = 1 + max(0, effective_char_count - 2500) / 5000；后续用真实回归数据拟合。",
    )


def _suggested_page_unit_target(effective_char_count: int) -> float:
    return max(1.0, 1.0 + max(0, effective_char_count - 2500) / 5000.0)


def _is_navigation_noise_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    if len(stripped) <= 80 and re.search(r"\b(skip to|search|sign in|log in|navigation|menu|footer|header|breadcrumb)\b", lowered):
        return True
    if len(stripped) <= 120 and lowered.count("](") >= 2:
        return True
    if len(stripped) <= 120 and re.fullmatch(r"[-*+]\s*(\[.*?\]\(.*?\)\s*)+", stripped):
        return True
    return False


def _assert_source_digest_granularity(digest: SourceDigest, stats: SourceGranularityStats) -> None:
    page_unit_count = len(digest.page_units)
    if page_unit_count == 0:
        if stats.effective_char_count < 600 or _has_strong_noise_reason(digest):
            return
        raise PipelineError(
            "source_digest page_unit_count=0，但 raw 看起来仍有可吸收内容；"
            f"effective_char_count={stats.effective_char_count}。如果原文确实是噪声，请在 weak_or_noise_items 中写清楚原因。"
        )
    if page_unit_count > stats.suggested_max_page_units:
        raise PipelineError(
            "source_digest page_unit_count 超出经验粒度区间："
            f"actual={page_unit_count}, suggested={stats.suggested_min_page_units}-{stats.suggested_max_page_units}, "
            f"effective_char_count={stats.effective_char_count}。请合并同主体、同读者任务、同页面类型的 page_units。"
        )
    if page_unit_count > 1:
        missing_rationale = [unit.page_unit_id for unit in digest.page_units if not unit.split_rationale.strip()]
        if missing_rationale:
            raise PipelineError(f"多个 page_units 时必须提供中文 split_rationale：{', '.join(missing_rationale)}")
        _assert_source_digest_no_near_duplicate_units(digest)


def _has_strong_noise_reason(digest: SourceDigest) -> bool:
    text = " ".join(item.reason for item in digest.weak_or_noise_items).lower()
    return bool(re.search(r"404|not found|无法访问|导航|菜单|噪声|重复|证据不足|空页面|错误页", text))


def _assert_source_digest_no_near_duplicate_units(digest: SourceDigest) -> None:
    units = digest.page_units
    for left_index, left in enumerate(units):
        for right in units[left_index + 1 :]:
            if left.page_type != right.page_type:
                continue
            title_score = difflib.SequenceMatcher(None, _granularity_text_key(left.title), _granularity_text_key(right.title)).ratio()
            scope_score = difflib.SequenceMatcher(None, _granularity_text_key(left.content_scope), _granularity_text_key(right.content_scope)).ratio()
            path_same_subject = _path_subject_key(left.path_hint) and _path_subject_key(left.path_hint) == _path_subject_key(right.path_hint)
            if title_score >= 0.82 or scope_score >= 0.76 or path_same_subject:
                raise PipelineError(
                    "source_digest 存在疑似过拆 page_units："
                    f"{left.page_unit_id}({left.title}) 与 {right.page_unit_id}({right.title}) 过于相近；"
                    "请合并为同一篇可读页面，除非主体对象、读者任务或页面类型明显不同。"
                )


def _granularity_text_key(text: str) -> str:
    return "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", text.lower()))


def _path_subject_key(path: str) -> str:
    stem = Path(path).stem.lower()
    stem = re.sub(r"^(concept|design|entity|comparison|open_question|source)_", "", stem)
    parts = [part for part in re.split(r"[_\-\s]+", stem) if part and part not in {"the", "and", "of", "for", "to", "a", "an"}]
    return "_".join(parts[:3])


def _call_provider_artifact(
    state: dict[str, object],
    out_dir: Path,
    step: str,
    request: BaseModel,
    output_model: type[BaseModel],
    artifact_stem: str | None = None,
) -> tuple[BaseModel, list[Path], int, list[dict[str, object]]]:
    registry: ProviderRegistry = state["provider_registry"]  # type: ignore[assignment]
    spec = registry.provider_for(step)
    provider_contexts: dict[str, dict[str, object]] = state["provider_contexts"]  # type: ignore[assignment]
    provider_contexts[step] = spec.sanitized_context()
    stem = artifact_stem or step
    try:
        result = registry.call_structured(step, request, output_model)
    except ProviderCallError as exc:
        _write_provider_failure_artifacts(out_dir, stem, exc, step)
        raise PipelineError(str(exc)) from exc
    except (ProviderConfigError, ValueError) as exc:
        raise PipelineError(str(exc)) from exc
    model_dir = out_dir / "model_calls"
    prompt_path = model_dir / f"{stem}.prompt.json"
    result_path = model_dir / f"{stem}.provider_result.json"
    api_calls_path = model_dir / f"{stem}.token_usage_calls.json"
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
    *,
    artifact_suffix: str = "",
    api_calls_filename: str = "token_usage_calls.json",
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
        artifact_stem = f"{step}_{safe_filename(key)}{artifact_suffix}"
        try:
            result = registry.call_structured(step, request, output_model)
        except ProviderCallError as exc:
            _write_provider_failure_artifacts(out_dir, artifact_stem, exc, key)
            raise PipelineError(str(exc)) from exc
        except (ProviderConfigError, ValueError) as exc:
            raise PipelineError(str(exc)) from exc
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
    api_calls_path = out_dir / api_calls_filename
    write_json(api_calls_path, api_calls)
    return outputs, [*artifacts, api_calls_path], model_calls, api_calls


def _call_provider_artifacts_parallel_soft(
    state: dict[str, object],
    out_dir: Path,
    step: str,
    requests: list[tuple[str, BaseModel]],
    output_model: type[BaseModel],
    *,
    artifact_suffix: str = "",
    api_calls_filename: str = "token_usage_calls.json",
) -> tuple[dict[str, BaseModel], list[Path], int, list[dict[str, object]], dict[str, str]]:
    registry: ProviderRegistry = state["provider_registry"]  # type: ignore[assignment]
    spec = registry.provider_for(step)
    provider_contexts: dict[str, dict[str, object]] = state["provider_contexts"]  # type: ignore[assignment]
    provider_contexts[step] = {**spec.sanitized_context(), "parallel_request_count": len(requests)}
    if not requests:
        return {}, [], 0, [], {}

    max_workers = _page_generation_parallelism(state, len(requests))
    results_by_key: dict[str, BaseModel] = {}
    artifacts_by_key: dict[str, list[Path]] = {}
    model_calls_by_key: dict[str, int] = {}
    api_calls_by_key: dict[str, list[dict[str, object]]] = {}
    errors_by_key: dict[str, str] = {}
    contexts = []
    model_dir = out_dir / "model_calls"

    def call_one(key: str, request: BaseModel) -> tuple[str, BaseModel | None, list[Path], dict[str, object], int, list[dict[str, object]], str | None]:
        artifact_stem = f"{step}_{safe_filename(key)}{artifact_suffix}"
        try:
            result = registry.call_structured(step, request, output_model)
        except ProviderCallError as exc:
            artifacts = _write_provider_failure_artifacts(out_dir, artifact_stem, exc, key)
            context = exc.sanitized_context or spec.sanitized_context()
            return key, None, artifacts, context, exc.model_calls, tag_api_calls(exc.api_calls, key), str(exc)
        except (ProviderConfigError, ValueError) as exc:
            raise PipelineError(str(exc)) from exc
        prompt_path = model_dir / f"{artifact_stem}.prompt.json"
        result_path = model_dir / f"{artifact_stem}.provider_result.json"
        write_json(prompt_path, result.prompt_artifact)
        write_json(result_path, result.provider_result)
        return key, result.output, [prompt_path, result_path], result.sanitized_context, result.model_calls, tag_api_calls(result.api_calls, key), None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(call_one, key, request) for key, request in requests]
        for future in as_completed(futures):
            key, output, artifacts, context, model_calls, api_calls, error = future.result()
            if output is not None:
                results_by_key[key] = output
            if error:
                errors_by_key[key] = error
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
    artifacts = [path for key, _ in requests for path in artifacts_by_key.get(key, [])]
    model_calls = sum(model_calls_by_key.get(key, 0) for key, _ in requests)
    api_calls = [call for key, _ in requests for call in api_calls_by_key.get(key, [])]
    api_calls_path = out_dir / api_calls_filename
    write_json(api_calls_path, api_calls)
    return results_by_key, [*artifacts, api_calls_path], model_calls, api_calls, errors_by_key


def _write_provider_failure_artifacts(out_dir: Path, artifact_stem: str, exc: ProviderCallError, request_key: str) -> list[Path]:
    model_dir = out_dir / "model_calls"
    prompt_path = model_dir / f"{artifact_stem}.prompt.json"
    result_path = model_dir / f"{artifact_stem}.provider_result.json"
    api_calls_path = model_dir / f"{artifact_stem}.token_usage_calls.json"
    api_calls = tag_api_calls(exc.api_calls, request_key)
    provider_result = dict(exc.provider_result or {})
    provider_result["status"] = "failed"
    provider_result["error"] = str(exc)
    provider_result["api_calls"] = api_calls
    write_json(prompt_path, exc.prompt_artifact or {"error": str(exc)})
    write_json(result_path, provider_result)
    write_json(api_calls_path, api_calls)
    return [prompt_path, result_path, api_calls_path]


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


def _mark_semantic_retry_failed(api_calls: list[dict[str, object]], error: str) -> list[dict[str, object]]:
    marked: list[dict[str, object]] = []
    for call in api_calls:
        item = dict(call)
        if item.get("status") == "success":
            item["status"] = "paused"
        item["error"] = f"系统语义校验失败：{error}"
        marked.append(item)
    return marked


def _mark_request_semantic_retry_failed(api_calls: list[dict[str, object]], errors_by_request: dict[str, str]) -> list[dict[str, object]]:
    marked: list[dict[str, object]] = []
    for call in api_calls:
        request_key = str(call.get("request_key") or "")
        error = errors_by_request.get(request_key)
        if error:
            marked.extend(_mark_semantic_retry_failed([call], error))
        else:
            marked.append(call)
    return marked


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
        if not page.page_unit_id.strip():
            raise PipelineError(f"候选页 {page.candidate_page_id} 缺少 page_unit_id。")
        if not page.source_refs:
            raise PipelineError(f"候选页 {page.candidate_page_id} 缺少 source_refs。")
        if not page.body_markdown.strip():
            raise PipelineError(f"候选页 {page.candidate_page_id} 正文为空。")


def _assert_source_digest_chinese(digest: SourceDigest) -> None:
    _require_chinese_text("source_digest.summary", digest.summary)
    for index, item in enumerate(digest.key_takeaways, start=1):
        _require_chinese_text(f"source_digest.key_takeaways[{index}]", item)
    for unit in digest.page_units:
        _require_chinese_title(f"{unit.page_unit_id}.title", unit.title, unit.summary, unit.content_scope)
        _require_chinese_text(f"{unit.page_unit_id}.summary", unit.summary)
        _require_chinese_text(f"{unit.page_unit_id}.content_scope", unit.content_scope)
        for index, item in enumerate(unit.must_cover_points, start=1):
            _require_chinese_text(f"{unit.page_unit_id}.must_cover_points[{index}]", item)
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


def _assert_merge_plan_chinese(plan: MergePlan) -> None:
    for decision in plan.decisions:
        _require_chinese_title(f"{decision.decision_id}.title", decision.title, decision.reason, decision.content_scope)
        _require_chinese_text(f"{decision.decision_id}.content_scope", decision.content_scope)
        _require_chinese_text(f"{decision.candidate_page_id}.reason", decision.reason)
        for index, item in enumerate(decision.candidate_content_locators, start=1):
            _require_chinese_text(f"{decision.decision_id}.candidate_content_locators[{index}]", item)
        for index, item in enumerate(decision.warnings, start=1):
            _require_chinese_text(f"{decision.candidate_page_id}.warnings[{index}]", item)


def _assert_composition_plan_chinese(plan: CompositionPlan) -> None:
    for item in plan.items:
        _require_chinese_text(f"{item.final_page_id}.readability_goal", item.readability_goal)
        for field_name in ["preserve_rules", "insert_rules", "delete_rules", "source_ref_rules", "warnings"]:
            for index, value in enumerate(getattr(item, field_name), start=1):
                _require_chinese_text(f"{item.final_page_id}.{field_name}[{index}]", value)


def _assert_final_pages_chinese(artifact: FinalPages) -> None:
    for page in artifact.pages:
        body = strip_frontmatter(page.markdown)
        _require_chinese_title(f"{page.final_page_id}.title", page.title, body)
        _require_chinese_text(f"{page.final_page_id}.markdown", body)
        for index, item in enumerate(page.preimage_coverage_report, start=1):
            _require_chinese_text(f"{page.final_page_id}.preimage_coverage_report[{index}].final_anchor", item.final_anchor)
            _require_chinese_text(f"{page.final_page_id}.preimage_coverage_report[{index}].evidence", item.evidence)
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
        open_questions = _normalize_candidate_open_questions(page)
        evidence_notes = [_chinese_scaffold(item, "来源定位") for item in page.evidence_notes]
        pages.append(page.model_copy(update={"open_questions": open_questions, "evidence_notes": evidence_notes}))
    return artifact.model_copy(update={"pages": pages})


def _normalize_candidate_open_questions(page: CandidatePage) -> list[str]:
    questions: list[str] = []
    for item in page.open_questions:
        stripped = item.strip()
        if not stripped:
            continue
        if contains_cjk(stripped):
            questions.append(_strip_locator_marker(stripped))
            continue
        resolved = _question_text_for_marker(page.body_markdown, stripped)
        if resolved:
            questions.append(resolved)
    return _dedupe_list([question for question in questions if question.strip()])


def _question_text_for_marker(markdown: str, marker: str) -> str:
    marker = marker.strip()
    if not marker:
        return ""
    for line in markdown.splitlines():
        if marker not in line:
            continue
        candidate = _strip_markdown_list_prefix(_strip_locator_marker(line, marker))
        if contains_cjk(candidate):
            return candidate
    for block in re.split(r"\n\s*\n", markdown):
        if marker not in block:
            continue
        candidate = _strip_locator_marker(re.sub(r"\s+", " ", block), marker).strip()
        if contains_cjk(candidate):
            return candidate
    return ""


def _strip_markdown_list_prefix(text: str) -> str:
    return re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", text).strip()


def _strip_locator_marker(text: str, marker: str | None = None) -> str:
    stripped = text.strip()
    if marker:
        escaped = re.escape(marker.strip())
        stripped = re.sub(rf"\s*(?:【{escaped}】|\[{escaped}\]|\({escaped}\)|`{escaped}`|{escaped})\s*$", "", stripped).strip()
    else:
        stripped = re.sub(r"\s*(?:【[-A-Za-z]+-\d+】|\[[-A-Za-z]+-\d+\]|\([-A-Za-z]+-\d+\)|`[-A-Za-z]+-\d+`)\s*$", "", stripped).strip()
    return stripped or text.strip()


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


def _merge_parallel_candidate_pages(outputs: list[BaseModel], page_units: list[SourcePageUnit]) -> CandidatePages:
    if len(outputs) != len(page_units):
        raise PipelineError(f"候选页并发请求组返回 {len(outputs)} 个结果，但 page_unit 数量是 {len(page_units)}。")
    pages: list[CandidatePage] = []
    skipped: list[str] = []
    for index, (output, unit) in enumerate(zip(outputs, page_units, strict=True), start=1):
        if not isinstance(output, CandidatePages):
            raise PipelineError("候选页并发请求返回了无效 artifact。")
        skipped.extend(output.skipped_page_unit_ids)
        if len(output.pages) != 1:
            raise PipelineError(f"{unit.page_unit_id} 的候选页请求必须且只能返回 1 页。")
        page = output.pages[0]
        pages.append(
            page.model_copy(
                update={
                    "candidate_page_id": f"CP-{index:03d}",
                    "page_unit_id": unit.page_unit_id,
                    "proposed_page_type": unit.page_type,
                    "proposed_path_hint": unit.path_hint,
                    "source_refs": _merge_source_refs([*unit.source_refs, *page.source_refs]),
                }
            )
        )
    return CandidatePages(pages=pages, skipped_page_unit_ids=_dedupe_list(skipped))


def _candidate_pages_from_outputs(outputs: list[BaseModel], page_units: list[SourcePageUnit]) -> CandidatePages:
    artifact = _merge_parallel_candidate_pages(outputs, page_units)
    artifact = _normalize_candidate_pages(artifact)
    _assert_unique([page.candidate_page_id for page in artifact.pages], "candidate_page_id")
    _assert_candidate_pages_have_sources(artifact)
    _assert_candidate_pages_chinese(artifact)
    return artifact


def _candidate_page_semantic_errors(
    outputs_by_page_unit_id: dict[str, BaseModel],
    page_units: list[SourcePageUnit],
    provider_errors: dict[str, str] | None = None,
) -> dict[str, str]:
    errors: dict[str, str] = dict(provider_errors or {})
    for unit in page_units:
        if unit.page_unit_id in errors:
            continue
        output = outputs_by_page_unit_id.get(unit.page_unit_id)
        if output is None:
            errors[unit.page_unit_id] = f"{unit.page_unit_id} 缺少 provider 输出。"
            continue
        try:
            _candidate_pages_from_outputs([output], [unit])
        except PipelineError as exc:
            errors[unit.page_unit_id] = str(exc)
    return errors


def _normalize_merge_plan(plan: MergePlan) -> MergePlan:
    action_counts = {action: 0 for action in ["create", "update", "noop"]}
    decisions: list[MergeDecision] = []
    for index, decision in enumerate(plan.decisions, start=1):
        action_counts[decision.action] += 1
        decision_id = decision.decision_id.strip() or f"MD-{index:03d}"
        candidate_content_locators = [
            _chinese_scaffold(item, "候选内容定位") for item in decision.candidate_content_locators
        ]
        decisions.append(
            decision.model_copy(
                update={
                    "decision_id": decision_id,
                    "candidate_content_locators": candidate_content_locators,
                }
            )
        )
    return plan.model_copy(update={"decisions": decisions, "action_counts": action_counts})


def _repair_merge_plan_candidate_content_locators(plan: MergePlan, candidate_pages: CandidatePages) -> MergePlan:
    pages_by_id = {page.candidate_page_id: page for page in candidate_pages.pages}
    repaired_decisions: list[MergeDecision] = []
    for decision in plan.decisions:
        if any(item.strip() for item in decision.candidate_content_locators):
            repaired_decisions.append(decision)
            continue
        page = pages_by_id.get(decision.candidate_page_id)
        if page is None:
            repaired_decisions.append(decision)
            continue
        locators = _dedupe_list(
            [
                decision.content_scope,
                f"候选页标题：{page.title}",
                f"候选页摘要：{page.summary}",
                f"页面单元：{page.page_unit_id}",
            ]
        )
        candidate_content_locators = [_chinese_scaffold(item, "候选内容定位") for item in locators if item.strip()]
        if not candidate_content_locators:
            candidate_content_locators = ["候选内容定位：候选页整体内容"]
        warnings = _dedupe_list([*decision.warnings, "候选内容定位由引擎根据候选页自动补齐。"])
        repaired_decisions.append(
            decision.model_copy(
                update={"candidate_content_locators": candidate_content_locators, "warnings": warnings}
            )
        )
    return plan.model_copy(update={"decisions": repaired_decisions})


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
        if not decision.candidate_content_locators:
            raise PipelineError(f"合并决策 {decision.decision_id} 缺少候选内容定位 candidate_content_locators。")
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
                "warnings": _dedupe_list([*existing.warnings, *item.warnings, "多个合并决策指向同一个目标页面，已合并写作规则。"]),
            }
        )
    normalized = []
    for index, item in enumerate(grouped.values(), start=1):
        normalized.append(
            item.model_copy(
                update={
                    "final_page_id": f"FP-{index:03d}",
                }
            )
        )
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


def _final_pages_from_outputs(
    outputs: list[BaseModel],
    composition: CompositionPlan,
    snapshot: WikiSnapshot,
    *,
    operation_id: str,
) -> FinalPages:
    artifact = _hydrate_provider_final_pages(_merge_parallel_final_pages(outputs, composition), composition, snapshot)
    artifact = _normalize_final_pages(
        artifact,
        composition,
        snapshot=snapshot,
        operation_id=operation_id,
    )
    _assert_final_pages_cover_composition(artifact, composition)
    _assert_final_pages_chinese(artifact)
    _assert_final_pages_preserve_preimage_coverage(artifact, composition, snapshot)
    return artifact


def _final_page_semantic_errors(
    outputs_by_id: dict[str, BaseModel],
    composition: CompositionPlan,
    snapshot: WikiSnapshot,
    *,
    operation_id: str,
) -> dict[str, str]:
    errors: dict[str, str] = {}
    for item in composition.items:
        output = outputs_by_id.get(item.final_page_id)
        if output is None:
            errors[item.final_page_id] = f"{item.final_page_id} 缺少 provider 输出。"
            continue
        single_composition = CompositionPlan(items=[item])
        try:
            _final_pages_from_outputs([output], single_composition, snapshot, operation_id=operation_id)
        except PipelineError as exc:
            errors[item.final_page_id] = str(exc)
    return errors


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
    for page in pages.pages:
        path_titles[page.target_path] = page.title
    normalized = []
    for page in pages.pages:
        item = items_by_id.get(page.final_page_id) or items_by_target.get(page.target_path)
        existing_entry = entries_by_path.get(page.target_path)
        model_title = page.title
        final_title = existing_entry.title if existing_entry is not None and item is not None and item.action == "update" else page.title
        source_refs = _merge_source_refs(page.source_refs)
        updated_page = page.model_copy(update={"title": final_title, "source_refs": source_refs})
        markdown = _canonical_final_markdown(
            updated_page,
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
    body = related_logic.canonicalize_body_wikilinks(body, known_paths=known_paths or set(), path_titles=path_titles or {})
    link_issues = related_logic.final_markdown_link_issues(
        markdown=body,
        target_path=page.target_path,
        title=model_title or page.title,
        known_paths=known_paths,
        path_titles=path_titles,
    )
    if link_issues:
        raise PipelineError(f"最终页 {page.target_path} 不符合链接契约：{'; '.join(issue.message for issue in link_issues)}")
    if not body.startswith("# "):
        body = f"# {page.title}\n\n{body}" if body else f"# {page.title}\n"
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


def _assert_final_pages_preserve_preimage_coverage(artifact: FinalPages, composition: CompositionPlan, snapshot: WikiSnapshot) -> None:
    items_by_target = {item.target_path: item for item in composition.items}
    entries_by_path = {entry.path: entry for entry in snapshot.entries}
    generic_anchors = {_granularity_text_key(item) for item in ["摘要", "核心内容", "矛盾与未决问题", "未决问题", "相关页面", "Related"]}
    for page in artifact.pages:
        item = items_by_target.get(page.target_path)
        if item is None or item.action != "update":
            continue
        existing = entries_by_path.get(page.target_path)
        if existing is None:
            continue
        requirements = prompts.preimage_coverage_requirements(existing)
        if not requirements:
            continue
        report_by_id = {report.requirement_id: report for report in page.preimage_coverage_report}
        required_ids = [str(requirement["requirement_id"]) for requirement in requirements]
        missing = [requirement_id for requirement_id in required_ids if requirement_id not in report_by_id]
        extra = sorted(set(report_by_id) - set(required_ids))
        if missing:
            raise PipelineError(f"最终页 {page.target_path} 缺少旧页覆盖报告：{', '.join(missing)}")
        if extra:
            raise PipelineError(f"最终页 {page.target_path} 覆盖报告包含未知 requirement_id：{', '.join(extra)}")
        body_key = _granularity_text_key(strip_frontmatter(page.markdown))
        for requirement in requirements:
            requirement_id = str(requirement["requirement_id"])
            report = report_by_id[requirement_id]
            final_anchor_key = _granularity_text_key(report.final_anchor)
            if not final_anchor_key or final_anchor_key in generic_anchors:
                raise PipelineError(f"最终页 {page.target_path} 的 {requirement_id} final_anchor 过于泛化，必须指向具体旧知识所在小节。")
            if final_anchor_key not in body_key:
                raise PipelineError(f"最终页 {page.target_path} 的 {requirement_id} final_anchor 未出现在最终正文：{report.final_anchor}")


def _final_preimage_coverage_report(artifact: FinalPages, composition: CompositionPlan, snapshot: WikiSnapshot) -> dict[str, object]:
    items_by_target = {item.target_path: item for item in composition.items}
    entries_by_path = {entry.path: entry for entry in snapshot.entries}
    pages: list[dict[str, object]] = []
    requirement_count = 0
    reported_count = 0
    for page in artifact.pages:
        item = items_by_target.get(page.target_path)
        if item is None or item.action != "update":
            continue
        existing = entries_by_path.get(page.target_path)
        requirements = prompts.preimage_coverage_requirements(existing)
        requirement_count += len(requirements)
        reported_count += len(page.preimage_coverage_report)
        report_by_id = {report.requirement_id: report for report in page.preimage_coverage_report}
        requirement_rows: list[dict[str, object]] = []
        for requirement in requirements:
            requirement_id = str(requirement["requirement_id"])
            report = report_by_id.get(requirement_id)
            requirement_rows.append(
                {
                    **requirement,
                    "reported": report is not None,
                    "report": report.model_dump(mode="json") if report is not None else None,
                }
            )
        pages.append(
            {
                "target_path": page.target_path,
                "title": page.title,
                "preimage_sha256": page.preimage_sha256,
                "requirements": requirement_rows,
            }
        )
    return {
        "requirement_count": requirement_count,
        "reported_count": reported_count,
        "pages": pages,
    }


def _render_final_preimage_coverage_report_md(report: dict[str, object]) -> str:
    lines = [
        "# 旧页覆盖报告",
        "",
        f"- 要求数：{report.get('requirement_count', 0)}",
        f"- 已报告数：{report.get('reported_count', 0)}",
        "",
    ]
    pages = report.get("pages")
    if not isinstance(pages, list) or not pages:
        lines.append("暂无 update 页面需要保留旧页覆盖。")
        return "\n".join(lines) + "\n"
    for page in pages:
        if not isinstance(page, dict):
            continue
        lines.append(f"## `{page.get('target_path')}`")
        lines.append("")
        requirements = page.get("requirements")
        if not isinstance(requirements, list) or not requirements:
            lines.append("- 无旧页覆盖要求。")
            lines.append("")
            continue
        for requirement in requirements:
            if not isinstance(requirement, dict):
                continue
            report_item = requirement.get("report")
            if isinstance(report_item, dict):
                lines.append(
                    "- "
                    f"{requirement.get('requirement_id')}：{report_item.get('status')}；"
                    f"锚点：{report_item.get('final_anchor')}；"
                    f"{report_item.get('evidence')}"
                )
            else:
                lines.append(f"- {requirement.get('requirement_id')}：未报告；{requirement.get('description')}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


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
    granularity_stats = _source_granularity_stats(raw_text, raw_size_bytes=binding.size_bytes)
    request = prompts.source_digest_prompt(raw_path=raw_rel, raw_sha256=binding.raw_sha256, raw_text=raw_text, profile=profile, granularity_stats=granularity_stats)
    provider_artifacts: list[Path] = []
    api_calls: list[dict[str, object]] = []
    model_calls = 0
    semantic_retry_count = 0
    digest: SourceDigest | None = None
    repair_report = StructuredRepairReport()
    last_semantic_error = ""
    for attempt in range(2):
        artifact_stem = "source_digest" if attempt == 0 else f"source_digest_retry_{attempt}"
        digest_result, attempt_artifacts, attempt_model_calls, attempt_api_calls = _call_provider_artifact(
            state,
            out_dir,
            "source_digest",
            request,
            SourceDigest,
            artifact_stem=artifact_stem,
        )
        provider_artifacts.extend(attempt_artifacts)
        model_calls += attempt_model_calls
        if not isinstance(digest_result, SourceDigest):
            raise PipelineError("source_digest provider 返回了无效 artifact。")
        try:
            _assert_source_digest_binding(digest_result, raw_rel, binding.raw_sha256)
            candidate_digest, candidate_repair_report = normalize_source_digest(digest_result, profile)
            _assert_source_digest_chinese(candidate_digest)
            _assert_unique([unit.page_unit_id for unit in candidate_digest.page_units], "source digest page_unit_id")
            _assert_source_digest_granularity(candidate_digest, granularity_stats)
        except PipelineError as exc:
            last_semantic_error = str(exc)
            api_calls.extend(_mark_semantic_retry_failed(attempt_api_calls, last_semantic_error))
            if attempt >= 1:
                break
            semantic_retry_count += 1
            request = prompts.source_digest_retry_prompt(
                raw_path=raw_rel,
                raw_sha256=binding.raw_sha256,
                raw_text=raw_text,
                profile=profile,
                previous_digest=digest_result,
                validation_error=last_semantic_error,
                granularity_stats=granularity_stats,
            )
            continue
        api_calls.extend(attempt_api_calls)
        digest = candidate_digest
        repair_report = candidate_repair_report
        break
    if digest is None:
        raise PipelineError(f"source_digest provider 重试后仍未通过系统语义校验：{last_semantic_error}")
    state["source_digest"] = digest
    json_path = out_dir / "source_digest.json"
    md_path = out_dir / "source_digest.md"
    page_units_json = out_dir / "page_units.json"
    page_units_md = out_dir / "page_units.md"
    granularity_path = out_dir / "granularity.json"
    repair_path = out_dir / "structured_repair_report.json"
    write_json(json_path, digest)
    write_text(md_path, _render_source_digest_md(digest))
    page_units = {"source_raw_path": raw_rel, "page_unit_ids": [item.page_unit_id for item in digest.page_units]}
    write_json(page_units_json, page_units)
    write_text(page_units_md, "\n".join(f"- {item.page_unit_id}: {item.title}" for item in digest.page_units) + "\n")
    write_json(granularity_path, granularity_stats)
    write_json(repair_path, repair_report)
    counts = {
        "page_unit_count": len(digest.page_units),
        "weak_noise_count": len(digest.weak_or_noise_items),
        "granularity_effective_chars": granularity_stats.effective_char_count,
        "granularity_suggested_min_page_units": granularity_stats.suggested_min_page_units,
        "granularity_suggested_target_page_units": granularity_stats.suggested_target_page_units,
        "granularity_suggested_max_page_units": granularity_stats.suggested_max_page_units,
        **({"semantic_retry_count": semantic_retry_count} if semantic_retry_count else {}),
        **_token_usage_counts(api_calls),
    }
    return StepOutput(
        [json_path, md_path, page_units_json, page_units_md, granularity_path, repair_path, *provider_artifacts],
        counts,
        model_calls=model_calls,
        repair_count=len(repair_report.repairs),
        api_calls=api_calls,
    )


def _step_candidate_pages_warmup(run_dir: Path, state: dict[str, object]) -> StepOutput:
    digest: SourceDigest = state["source_digest"]  # type: ignore[assignment]
    raw_abs: Path = state["raw_abs"]  # type: ignore[assignment]
    binding: RawBinding = state["raw_binding"]  # type: ignore[assignment]
    raw_rel: str = state["raw_rel"]  # type: ignore[assignment]
    profile: Profile = state["profile"]  # type: ignore[assignment]
    out_dir = run_dir / "candidate_pages_warmup"
    if not digest.page_units:
        return StepOutput([], {"warmup_count": 0, "page_unit_count": 0})
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
            "page_unit_count": len(digest.page_units),
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
    raw_abs: Path = state["raw_abs"]  # type: ignore[assignment]
    binding: RawBinding = state["raw_binding"]  # type: ignore[assignment]
    raw_rel: str = state["raw_rel"]  # type: ignore[assignment]
    profile: Profile = state["profile"]  # type: ignore[assignment]
    raw_text = read_text(raw_abs)
    out_dir = run_dir / "candidate_pages"
    page_units = digest.page_units
    initial_requests = [
        (
            unit.page_unit_id,
            prompts.candidate_page_prompt(
                digest=digest,
                page_unit=unit,
                raw_path=raw_rel,
                raw_sha256=binding.raw_sha256,
                raw_text=raw_text,
                profile=profile,
            ),
        )
        for unit in page_units
    ]
    outputs_by_page_unit_id, provider_artifacts, model_calls, api_calls, provider_errors = _call_provider_artifacts_parallel_soft(
        state,
        out_dir,
        "candidate_pages",
        initial_requests,
        CandidatePages,
    )
    parallel_request_count = len(page_units)
    semantic_retry_count = 0
    semantic_errors = _candidate_page_semantic_errors(outputs_by_page_unit_id, page_units, provider_errors)
    if semantic_errors:
        api_calls = _mark_request_semantic_retry_failed(api_calls, semantic_errors)
        retry_units = [unit for unit in page_units if unit.page_unit_id in semantic_errors]
        retry_requests = []
        for unit in retry_units:
            previous_output = outputs_by_page_unit_id.get(unit.page_unit_id)
            previous_pages = previous_output if isinstance(previous_output, CandidatePages) else None
            retry_requests.append(
                (
                    unit.page_unit_id,
                    prompts.candidate_page_retry_prompt(
                        digest=digest,
                        page_unit=unit,
                        raw_path=raw_rel,
                        raw_sha256=binding.raw_sha256,
                        raw_text=raw_text,
                        profile=profile,
                        previous_pages=previous_pages,
                        validation_error=semantic_errors[unit.page_unit_id],
                    ),
                )
            )
        semantic_retry_count = len(retry_requests)
        retry_outputs, retry_artifacts, retry_model_calls, retry_api_calls, retry_provider_errors = _call_provider_artifacts_parallel_soft(
            state,
            out_dir,
            "candidate_pages",
            retry_requests,
            CandidatePages,
            artifact_suffix="_retry_1",
            api_calls_filename="token_usage_calls_retry_1.json",
        )
        provider_artifacts.extend(retry_artifacts)
        model_calls += retry_model_calls
        retry_errors = _candidate_page_semantic_errors(retry_outputs, retry_units, retry_provider_errors)
        if retry_errors:
            api_calls.extend(_mark_request_semantic_retry_failed(retry_api_calls, retry_errors))
            raise PipelineError(f"candidate_pages 单候选页重试后仍未通过系统校验：{'; '.join(retry_errors.values())}")
        api_calls.extend(retry_api_calls)
        outputs_by_page_unit_id.update(retry_outputs)
        combined_api_calls_path = out_dir / "token_usage_calls.json"
        write_json(combined_api_calls_path, api_calls)
        if combined_api_calls_path not in provider_artifacts:
            provider_artifacts.append(combined_api_calls_path)
    artifact = _candidate_pages_from_outputs([outputs_by_page_unit_id[unit.page_unit_id] for unit in page_units], page_units)
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
        "covered_page_unit_count": len({page.page_unit_id for page in artifact.pages}),
        "parallel_request_count": parallel_request_count if model_calls else 0,
        "parallel_max_workers": _page_generation_parallelism(state, parallel_request_count) if model_calls else 0,
        **({"semantic_retry_count": semantic_retry_count} if semantic_retry_count else {}),
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
    contexts: CandidateContexts = state["candidate_contexts"]  # type: ignore[assignment]
    profile: Profile = state["profile"]  # type: ignore[assignment]
    out_dir = run_dir / "merge_plan"
    request = prompts.merge_plan_prompt(candidate_pages=candidate_pages, candidate_contexts=contexts, profile=profile)
    provider_artifacts: list[Path] = []
    api_calls: list[dict[str, object]] = []
    model_calls = 0
    semantic_retry_count = 0
    plan: MergePlan | None = None
    last_semantic_error = ""
    for attempt in range(2):
        artifact_stem = "merge_plan" if attempt == 0 else f"merge_plan_retry_{attempt}"
        plan_result, attempt_artifacts, attempt_model_calls, attempt_api_calls = _call_provider_artifact(
            state,
            out_dir,
            "merge_plan",
            request,
            MergePlan,
            artifact_stem=artifact_stem,
        )
        provider_artifacts.extend(attempt_artifacts)
        model_calls += attempt_model_calls
        if not isinstance(plan_result, MergePlan):
            raise PipelineError("merge_plan provider 返回了无效 artifact。")
        try:
            candidate_plan = _normalize_merge_plan(plan_result)
            candidate_plan = _repair_merge_plan_candidate_content_locators(candidate_plan, candidate_pages)
            _assert_merge_plan_chinese(candidate_plan)
            _assert_merge_plan_consumes_candidates(candidate_plan, candidate_pages, contexts)
        except PipelineError as exc:
            last_semantic_error = str(exc)
            api_calls.extend(_mark_semantic_retry_failed(attempt_api_calls, last_semantic_error))
            if attempt >= 1:
                break
            semantic_retry_count += 1
            request = prompts.merge_plan_retry_prompt(
                candidate_pages=candidate_pages,
                candidate_contexts=contexts,
                profile=profile,
                previous_plan=plan_result,
                validation_error=last_semantic_error,
            )
            continue
        api_calls.extend(attempt_api_calls)
        plan = candidate_plan
        break
    if plan is None:
        raise PipelineError(f"merge_plan provider 重试后仍未通过系统语义校验：{last_semantic_error}")
    state["merge_plan"] = plan
    json_path = out_dir / "merge_plan.json"
    md_path = out_dir / "merge_plan.md"
    write_json(json_path, plan)
    rendered = _render_merge_plan_md(plan)
    write_text(md_path, rendered)
    return StepOutput(
        [json_path, md_path, *provider_artifacts],
        {
            **{f"{key}_count": value for key, value in plan.action_counts.items()},
            **({"semantic_retry_count": semantic_retry_count} if semantic_retry_count else {}),
            **_token_usage_counts(api_calls),
        },
        model_calls=model_calls,
        api_calls=api_calls,
    )


def _step_composition_plan(run_dir: Path, state: dict[str, object]) -> StepOutput:
    plan: MergePlan = state["merge_plan"]  # type: ignore[assignment]
    candidate_pages: CandidatePages = state["candidate_pages"]  # type: ignore[assignment]
    profile: Profile = state["profile"]  # type: ignore[assignment]
    out_dir = run_dir / "composition_plan"
    request = prompts.composition_plan_prompt(merge_plan=plan, candidate_pages=candidate_pages, profile=profile)
    provider_artifacts: list[Path] = []
    api_calls: list[dict[str, object]] = []
    model_calls = 0
    semantic_retry_count = 0
    artifact: CompositionPlan | None = None
    last_semantic_error = ""
    for attempt in range(2):
        artifact_stem = "composition_plan" if attempt == 0 else f"composition_plan_retry_{attempt}"
        plan_result, attempt_artifacts, attempt_model_calls, attempt_api_calls = _call_provider_artifact(
            state,
            out_dir,
            "composition_plan",
            request,
            CompositionPlan,
            artifact_stem=artifact_stem,
        )
        provider_artifacts.extend(attempt_artifacts)
        model_calls += attempt_model_calls
        if not isinstance(plan_result, CompositionPlan):
            raise PipelineError("composition_plan provider 返回了无效 artifact。")
        try:
            candidate_plan = _normalize_composition_plan(plan_result)
            _assert_composition_plan_chinese(candidate_plan)
            _assert_composition_covers_writes(candidate_plan, plan)
        except PipelineError as exc:
            last_semantic_error = str(exc)
            api_calls.extend(_mark_semantic_retry_failed(attempt_api_calls, last_semantic_error))
            if attempt >= 1:
                break
            semantic_retry_count += 1
            request = prompts.composition_plan_retry_prompt(
                merge_plan=plan,
                candidate_pages=candidate_pages,
                profile=profile,
                previous_plan=plan_result,
                validation_error=last_semantic_error,
            )
            continue
        api_calls.extend(attempt_api_calls)
        artifact = candidate_plan
        break
    if artifact is None:
        raise PipelineError(f"composition_plan provider 重试后仍未通过系统语义校验：{last_semantic_error}")
    state["composition_plan"] = artifact
    json_path = out_dir / "composition_plan.json"
    md_path = out_dir / "composition_plan.md"
    write_json(json_path, artifact)
    write_text(md_path, _render_composition_plan_md(artifact))
    return StepOutput(
        [json_path, md_path, *provider_artifacts],
        {
            "final_target_count": len(artifact.items),
            "update_target_count": sum(1 for item in artifact.items if item.action == "update"),
            **({"semantic_retry_count": semantic_retry_count} if semantic_retry_count else {}),
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
    operation_id = str(state.get("operation_id", ""))
    initial_requests = [
        (
            item.final_page_id,
            prompts.final_page_prompt(composition_item=item, candidate_pages=candidate_pages, snapshot=snapshot, profile=profile),
        )
        for item in composition.items
    ]
    outputs, provider_artifacts, model_calls, api_calls = _call_provider_artifacts_parallel(
        state,
        out_dir,
        "final_pages",
        initial_requests,
        FinalPages,
    )
    parallel_request_count = len(composition.items)
    outputs_by_id = {item.final_page_id: output for item, output in zip(composition.items, outputs, strict=True)}
    semantic_retry_count = 0
    semantic_errors = _final_page_semantic_errors(outputs_by_id, composition, snapshot, operation_id=operation_id)
    if semantic_errors:
        api_calls = _mark_request_semantic_retry_failed(api_calls, semantic_errors)
        semantic_retry_count = len(semantic_errors)
        retry_items = [item for item in composition.items if item.final_page_id in semantic_errors]
        retry_requests: list[tuple[str, BaseModel]] = []
        for item in retry_items:
            previous_output = outputs_by_id[item.final_page_id]
            if not isinstance(previous_output, FinalPages):
                raise PipelineError(f"{item.final_page_id} 的最终页请求返回了无效 artifact。")
            retry_requests.append(
                (
                    item.final_page_id,
                    prompts.final_page_retry_prompt(
                        composition_item=item,
                        candidate_pages=candidate_pages,
                        snapshot=snapshot,
                        profile=profile,
                        previous_pages=previous_output,
                        validation_error=semantic_errors[item.final_page_id],
                    ),
                )
            )
        retry_outputs, retry_artifacts, retry_model_calls, retry_api_calls = _call_provider_artifacts_parallel(
            state,
            out_dir,
            "final_pages",
            retry_requests,
            FinalPages,
            artifact_suffix="_retry_1",
            api_calls_filename="token_usage_calls_retry_1.json",
        )
        provider_artifacts.extend(retry_artifacts)
        model_calls += retry_model_calls
        for item, output in zip(retry_items, retry_outputs, strict=True):
            outputs_by_id[item.final_page_id] = output
        retry_errors = _final_page_semantic_errors(
            {item.final_page_id: outputs_by_id[item.final_page_id] for item in retry_items},
            CompositionPlan(items=retry_items),
            snapshot,
            operation_id=operation_id,
        )
        if retry_errors:
            api_calls.extend(_mark_request_semantic_retry_failed(retry_api_calls, retry_errors))
            raise PipelineError(f"final_pages 单页重试后仍未通过系统语义校验：{'; '.join(retry_errors.values())}")
        api_calls.extend(retry_api_calls)
        combined_api_calls_path = out_dir / "token_usage_calls.json"
        write_json(combined_api_calls_path, api_calls)
        if combined_api_calls_path not in provider_artifacts:
            provider_artifacts.append(combined_api_calls_path)
    artifact = _final_pages_from_outputs(
        [outputs_by_id[item.final_page_id] for item in composition.items],
        composition,
        snapshot,
        operation_id=operation_id,
    )
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
    coverage_report = _final_preimage_coverage_report(artifact, composition, snapshot)
    coverage_json_path = out_dir / "preimage_coverage_report.json"
    coverage_md_path = out_dir / "preimage_coverage_report.md"
    write_json(coverage_json_path, coverage_report)
    write_text(coverage_md_path, _render_final_preimage_coverage_report_md(coverage_report))
    state["final_pages"] = artifact
    json_path = out_dir / "final_pages.json"
    manifest_path = out_dir / "final_page_manifest.json"
    write_json(json_path, artifact)
    write_json(manifest_path, [{"target_path": page.target_path, "sha256": page.content_sha256, "action": page.action} for page in artifact.pages])
    artifacts.extend([json_path, manifest_path, coverage_json_path, coverage_md_path, *provider_artifacts])
    return StepOutput(
        artifacts,
        {
            "final_page_count": len(artifact.pages),
            "diff_count": len(artifact.pages),
            "preimage_coverage_requirement_count": int(coverage_report["requirement_count"]),
            "preimage_coverage_report_count": int(coverage_report["reported_count"]),
            "parallel_request_count": parallel_request_count if model_calls else 0,
            "parallel_max_workers": _page_generation_parallelism(state, parallel_request_count) if model_calls else 0,
            **({"semantic_retry_count": semantic_retry_count} if semantic_retry_count else {}),
            **_token_usage_counts(api_calls),
        },
        model_calls=model_calls,
        api_calls=api_calls,
    )


def _step_related_refresh(run_dir: Path, state: dict[str, object]) -> StepOutput:
    final_pages: FinalPages = state["final_pages"]  # type: ignore[assignment]
    snapshot: WikiSnapshot = state["wiki_snapshot"]  # type: ignore[assignment]
    embedding_config: EmbeddingConfig = state["embedding_config"]  # type: ignore[assignment]
    page_records: dict[str, dict[str, object]] = state.get("embedding_page_records", {})  # type: ignore[assignment]
    threshold = _related_similarity_threshold(state)
    report = related_logic.RelatedMergeReport()
    if not final_pages.pages:
        return StepOutput([], {"related_link_count": 0, "related_threshold": threshold, "body_link_count": 0})

    final_cards = [_final_page_embedding_card(page, embedding_config.max_page_chars) for page in final_pages.pages]
    final_vectors = embed_texts(final_cards, embedding_config, is_query=False)
    final_vector_by_path = {page.target_path: vector for page, vector in zip(final_pages.pages, final_vectors, strict=True)}
    pool = _related_embedding_pool(
        snapshot=snapshot,
        page_records=page_records,
        final_pages=final_pages,
        final_vectors=final_vector_by_path,
    )
    known_paths = set(pool)
    path_titles = {path: str(item["title"]) for path, item in pool.items()}
    refreshed_pages: list[FinalPage] = []
    related_link_count = 0
    body_link_count = 0
    for page in final_pages.pages:
        body_links = set(related_logic.body_wikilink_targets(page.markdown))
        body_link_count += len(body_links)
        vector = final_vector_by_path[page.target_path]
        selected = _select_calculated_related(
            page=page,
            vector=vector,
            pool=pool,
            body_links=body_links,
            threshold=threshold,
            report=report,
        )
        if selected:
            related_link_count += 1
        markdown = _apply_calculated_related_section(
            page.markdown,
            current_path=page.target_path,
            related_page=selected,
            known_paths=known_paths,
            path_titles=path_titles,
        )
        refreshed_pages.append(page.model_copy(update={"markdown": markdown, "content_sha256": sha256_text(markdown)}))
    artifact = final_pages.model_copy(update={"pages": refreshed_pages})
    state["final_pages"] = artifact
    state["related_refresh_report"] = report
    out_dir = run_dir / "related_refresh"
    json_path = out_dir / "final_pages.json"
    report_json = out_dir / "related_refresh_report.json"
    report_md = out_dir / "related_refresh_report.md"
    write_json(json_path, artifact)
    write_json(report_json, report)
    write_text(report_md, related_logic.render_related_report(report))
    page_artifacts = []
    for page in artifact.pages:
        page_path = out_dir / "pages" / page.target_path
        write_text(page_path, page.markdown)
        page_artifacts.append(page_path)
    return StepOutput(
        [json_path, report_json, report_md, *page_artifacts],
        {
            "related_link_count": related_link_count,
            "related_kept_count": sum(1 for item in report.candidates if item.decision == "kept"),
            "related_filtered_count": sum(1 for item in report.candidates if item.decision != "kept"),
            "related_threshold": threshold,
            "body_link_count": body_link_count,
        },
    )


def _related_similarity_threshold(state: dict[str, object]) -> float:
    config = state.get("config")
    if isinstance(config, dict):
        related = config.get("related")
        if isinstance(related, dict):
            raw = related.get("min_similarity")
            if raw is not None:
                try:
                    return max(0.0, min(1.0, float(raw)))
                except (TypeError, ValueError):
                    pass
    return RELATED_MIN_SIMILARITY


def _related_replacement_margin(state: dict[str, object]) -> float:
    config = state.get("config")
    if isinstance(config, dict):
        related = config.get("related")
        if isinstance(related, dict):
            raw = related.get("replacement_margin")
            if raw is not None:
                try:
                    return max(0.0, min(1.0, float(raw)))
                except (TypeError, ValueError):
                    pass
    return RELATED_REPLACEMENT_MARGIN


def _step_related_maintenance(vault: Path, run_dir: Path, state: dict[str, object]) -> StepOutput:
    profile: Profile = state["profile"]  # type: ignore[assignment]
    final_pages: FinalPages = state["final_pages"]  # type: ignore[assignment]
    embedding_config: EmbeddingConfig = state["embedding_config"]  # type: ignore[assignment]
    page_records: dict[str, dict[str, object]] = state.get("embedding_page_records", {})  # type: ignore[assignment]
    final_targets = {page.target_path for page in final_pages.pages}
    out_dir = run_dir / "related_maintenance"
    report_path = out_dir / "related_maintenance_report.json"
    report_md_path = out_dir / "related_maintenance_report.md"
    result_path = out_dir / "write_result.json"
    items_path = out_dir / "write_items.json"
    if not final_targets:
        empty = {
            "threshold": _related_similarity_threshold(state),
            "replacement_margin": _related_replacement_margin(state),
            "checked_pages": [],
            "candidates": [],
            "written_targets": [],
        }
        write_json(report_path, empty)
        write_text(report_md_path, _render_related_maintenance_md(empty))
        result = WriteResult(written_targets=[])
        write_json(result_path, result)
        write_json(items_path, [])
        return StepOutput([report_path, report_md_path, result_path, items_path], {"related_maintenance_checked_count": 0, "related_maintenance_written_count": 0})

    threshold = _related_similarity_threshold(state)
    margin = _related_replacement_margin(state)
    entries = _scan_wiki_entries(vault, profile)
    pool = _related_pool_from_entries(entries=entries, page_records=page_records)
    path_titles = {path: str(item["title"]) for path, item in pool.items()}
    report = related_logic.RelatedMergeReport()
    checked_pages: list[dict[str, object]] = []
    writes: dict[str, str] = {}
    write_items: list[WriteSetItem] = []
    final_vectors = {path: pool[path]["vector"] for path in final_targets if path in pool}

    for entry in entries:
        if entry.path in final_targets:
            continue
        page_path = vault / "wiki" / entry.path
        markdown = read_text(page_path)
        current_targets = related_logic.related_section_targets(markdown)
        current_target = current_targets[0] if current_targets else ""
        vector = pool[entry.path]["vector"]
        affected_reasons = _related_maintenance_reasons(
            vector=vector,  # type: ignore[arg-type]
            current_target=current_target,
            final_targets=final_targets,
            final_vectors=final_vectors,  # type: ignore[arg-type]
            threshold=threshold,
        )
        if not affected_reasons:
            continue
        body_links = set(related_logic.body_wikilink_targets(markdown))
        best = _best_related_candidate(
            owner_id=f"MAINT-{len(checked_pages) + 1:03d}",
            current_path=entry.path,
            vector=vector,  # type: ignore[arg-type]
            pool=pool,
            body_links=body_links,
            threshold=threshold,
            report=report,
        )
        current_score = _related_score(vector, pool, current_target) if current_target else None  # type: ignore[arg-type]
        action = "keep_existing"
        reason = "现有 Related 足够稳定。"
        selected: RelatedPageRef | None = None
        candidate_path = ""
        candidate_score: float | None = None
        if best is None:
            if current_target and (current_target not in pool or current_target in body_links):
                action = "remove"
                reason = "现有 Related 目标无效或与正文链接重复，且没有可替代候选。"
            else:
                action = "keep_no_candidate"
                reason = "没有超过阈值的新候选。"
        else:
            candidate_score, candidate_path, candidate_item = best
            if not current_target:
                action = "add"
                reason = "旧页面没有 Related，新增超过阈值的最相关页面。"
                selected = _related_ref_from_candidate(candidate_path, candidate_item, candidate_score)
            elif current_target == candidate_path:
                action = "keep_existing"
                reason = "当前 Related 仍是最相关页面。"
            elif current_target not in pool or current_target in body_links or current_score is None:
                action = "replace"
                reason = "现有 Related 无效或与正文链接重复，替换为最相关页面。"
                selected = _related_ref_from_candidate(candidate_path, candidate_item, candidate_score)
            elif candidate_score >= current_score + margin:
                action = "replace"
                reason = f"新候选相似度比现有 Related 高至少 {margin:.2f}。"
                selected = _related_ref_from_candidate(candidate_path, candidate_item, candidate_score)
            else:
                action = "keep_existing"
                reason = f"新候选未比现有 Related 高出 {margin:.2f}，保持稳定。"

        if action in {"add", "replace", "remove"}:
            new_markdown = _apply_calculated_related_section(
                markdown,
                current_path=entry.path,
                related_page=selected,
                known_paths=set(pool),
                path_titles=path_titles,
            )
            if new_markdown != markdown:
                writes[entry.path] = new_markdown
                pre_sha = sha256_file(page_path)
                write_items.append(WriteSetItem(kind="knowledge", target_path=entry.path, content_sha256=sha256_text(new_markdown), preimage_sha256=pre_sha))
                atomic_write_text(page_path, new_markdown)
                page_artifact = out_dir / "pages" / entry.path
                write_text(page_artifact, new_markdown)
                diff_path = out_dir / "diffs" / f"{safe_filename(entry.path)}.diff"
                diff = "\n".join(
                    difflib.unified_diff(
                        markdown.splitlines(),
                        new_markdown.splitlines(),
                        fromfile=f"a/wiki/{entry.path}",
                        tofile=f"b/wiki/{entry.path}",
                        lineterm="",
                    )
                )
                write_text(diff_path, diff + ("\n" if diff else ""))
            else:
                action = "unchanged"
                reason = "计算结果与当前页面一致。"

        checked_pages.append(
            {
                "path": entry.path,
                "title": entry.title,
                "affected_reasons": affected_reasons,
                "current_target": current_target,
                "current_score": round(current_score, 4) if current_score is not None else None,
                "candidate_target": candidate_path,
                "candidate_score": round(candidate_score, 4) if candidate_score is not None else None,
                "action": action,
                "reason": reason,
            }
        )

    if write_items:
        state["write_set_items"] = [*state.get("write_set_items", []), *write_items]  # type: ignore[list-item]
        state["written_targets"] = sorted({*state.get("written_targets", []), *writes.keys()})  # type: ignore[arg-type]
        refreshed_entries = _scan_wiki_entries(vault, profile)
        refreshed_records, cache_metrics = sync_page_embedding_cache(vault, refreshed_entries, embedding_config)
        state["embedding_page_records"] = refreshed_records
        embedding_metrics = dict(state.get("embedding_refresh_metrics", {}))
        embedding_metrics["related_maintenance_cache_sync"] = cache_metrics
        state["embedding_refresh_metrics"] = embedding_metrics

    result = WriteResult(written_targets=sorted(writes))
    report_payload = {
        "threshold": threshold,
        "replacement_margin": margin,
        "checked_pages": checked_pages,
        "candidates": report.candidates,
        "written_targets": sorted(writes),
    }
    write_json(report_path, report_payload)
    write_text(report_md_path, _render_related_maintenance_md(report_payload))
    write_json(result_path, result)
    write_json(items_path, write_items)
    artifacts = [report_path, report_md_path, result_path, items_path]
    artifacts.extend(out_dir / "pages" / target for target in sorted(writes))
    artifacts.extend(out_dir / "diffs" / f"{safe_filename(target)}.diff" for target in sorted(writes))
    return StepOutput(
        artifacts,
        {
            "related_maintenance_checked_count": len(checked_pages),
            "related_maintenance_written_count": len(writes),
            "related_maintenance_added_count": sum(1 for item in checked_pages if item["action"] == "add"),
            "related_maintenance_replaced_count": sum(1 for item in checked_pages if item["action"] == "replace"),
            "related_maintenance_removed_count": sum(1 for item in checked_pages if item["action"] == "remove"),
            "related_threshold": threshold,
            "related_replacement_margin": margin,
        },
    )


def _related_maintenance_reasons(
    *,
    vector: list[float],
    current_target: str,
    final_targets: set[str],
    final_vectors: dict[str, list[float]],
    threshold: float,
) -> list[str]:
    reasons: list[str] = []
    if current_target in final_targets:
        reasons.append("current_related_points_to_written_page")
    best_written_score = 0.0
    for final_vector in final_vectors.values():
        best_written_score = max(best_written_score, cosine(vector, final_vector))
    if best_written_score >= threshold:
        reasons.append("similar_to_written_page")
    return reasons


def _related_pool_from_entries(
    *,
    entries: list[WikiKnowledgeEntry],
    page_records: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    pool: dict[str, dict[str, object]] = {}
    for entry in entries:
        record = page_records.get(entry.path)
        vector = record.get("vector", []) if isinstance(record, dict) else []
        if not isinstance(vector, list) or not vector:
            raise PipelineError(f"相关页面维护必须使用最新 embedding cache，但页面缺少有效向量：{entry.path}")
        pool[entry.path] = {
            "path": entry.path,
            "title": entry.title,
            "page_type": entry.page_type,
            "vector": vector,
        }
    return pool


def _best_related_candidate(
    *,
    owner_id: str,
    current_path: str,
    vector: list[float],
    pool: dict[str, dict[str, object]],
    body_links: set[str],
    threshold: float,
    report: related_logic.RelatedMergeReport,
) -> tuple[float, str, dict[str, object]] | None:
    ranked: list[tuple[float, str, dict[str, object]]] = []
    for path, item in pool.items():
        if path == current_path:
            report.candidates.append(
                RelatedCandidateReport(
                    owner_id=owner_id,
                    current_path=current_path,
                    target_path=path,
                    display_title=str(item["title"]),
                    source="embedding_similarity",
                    decision="filtered",
                    reject_reason="self_link",
                    reason="候选目标是当前页面自身。",
                )
            )
            continue
        score = round(cosine(vector, item["vector"]), 4)  # type: ignore[arg-type]
        reason = f"全文向量相似度={score:.4f}；阈值={threshold:.2f}。"
        if path in body_links:
            report.candidates.append(
                RelatedCandidateReport(
                    owner_id=owner_id,
                    current_path=current_path,
                    target_path=path,
                    display_title=str(item["title"]),
                    source="embedding_similarity",
                    decision="filtered",
                    reject_reason="already_body_link",
                    reason=reason,
                )
            )
            continue
        if score < threshold:
            report.candidates.append(
                RelatedCandidateReport(
                    owner_id=owner_id,
                    current_path=current_path,
                    target_path=path,
                    display_title=str(item["title"]),
                    source="embedding_similarity",
                    decision="filtered",
                    reject_reason="below_similarity_threshold",
                    reason=reason,
                )
            )
            continue
        ranked.append((score, path, item))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    if not ranked:
        return None
    score, path, item = ranked[0]
    report.candidates.append(
        RelatedCandidateReport(
            owner_id=owner_id,
            current_path=current_path,
            target_path=path,
            display_title=str(item["title"]),
            source="embedding_similarity",
            decision="kept",
            reason=f"全文向量相似度={score:.4f}，是超过阈值的最相关非正文链接页面。",
        )
    )
    for cutoff_score, cutoff_path, cutoff_item in ranked[1:]:
        report.candidates.append(
            RelatedCandidateReport(
                owner_id=owner_id,
                current_path=current_path,
                target_path=cutoff_path,
                display_title=str(cutoff_item["title"]),
                source="embedding_similarity",
                decision="cutoff",
                reject_reason="single_related_limit",
                reason=f"全文向量相似度={cutoff_score:.4f}，但 Related 只保留最相关 1 条。",
            )
        )
    return score, path, item


def _related_score(vector: list[float], pool: dict[str, dict[str, object]], target_path: str) -> float | None:
    item = pool.get(target_path)
    if not item:
        return None
    return cosine(vector, item["vector"])  # type: ignore[arg-type]


def _related_ref_from_candidate(path: str, item: dict[str, object], score: float) -> RelatedPageRef:
    return RelatedPageRef(
        target_path=path,
        display_title=str(item["title"]),
        source="wiki_context",
        reason=f"全文向量相似度={score:.4f}，是超过阈值的最相关非正文链接页面。",
    )


def _render_related_maintenance_md(report: dict[str, object]) -> str:
    lines = [
        "# 相关页面维护报告",
        "",
        f"- 阈值：{report.get('threshold')}",
        f"- 替换分差：{report.get('replacement_margin')}",
        f"- 写入页面数：{len(report.get('written_targets', [])) if isinstance(report.get('written_targets'), list) else 0}",
        "",
        "## 检查页面",
    ]
    checked = report.get("checked_pages")
    if not isinstance(checked, list) or not checked:
        lines.append("- 暂无需要维护的旧页面。")
        return "\n".join(lines) + "\n"
    for item in checked:
        if not isinstance(item, dict):
            continue
        lines.append(
            "- "
            f"`{item.get('path')}`：{item.get('action')}；"
            f"当前=`{item.get('current_target') or '<none>'}`({item.get('current_score')})；"
            f"候选=`{item.get('candidate_target') or '<none>'}`({item.get('candidate_score')})；"
            f"{item.get('reason')}"
        )
    return "\n".join(lines) + "\n"


def _related_embedding_pool(
    *,
    snapshot: WikiSnapshot,
    page_records: dict[str, dict[str, object]],
    final_pages: FinalPages,
    final_vectors: dict[str, list[float]],
) -> dict[str, dict[str, object]]:
    final_targets = {page.target_path for page in final_pages.pages}
    pool: dict[str, dict[str, object]] = {}
    for entry in snapshot.entries:
        if entry.path in final_targets:
            continue
        record = page_records.get(entry.path)
        vector = record.get("vector", []) if isinstance(record, dict) else []
        if not isinstance(vector, list) or not vector:
            raise PipelineError(f"相关页面计算必须使用 embedding cache，但页面缺少有效向量：{entry.path}")
        pool[entry.path] = {
            "path": entry.path,
            "title": entry.title,
            "page_type": entry.page_type,
            "vector": vector,
        }
    for page in final_pages.pages:
        pool[page.target_path] = {
            "path": page.target_path,
            "title": page.title,
            "page_type": page.page_type,
            "vector": final_vectors[page.target_path],
        }
    return pool


def _select_calculated_related(
    *,
    page: FinalPage,
    vector: list[float],
    pool: dict[str, dict[str, object]],
    body_links: set[str],
    threshold: float,
    report: related_logic.RelatedMergeReport,
) -> RelatedPageRef | None:
    best = _best_related_candidate(
        owner_id=page.final_page_id,
        current_path=page.target_path,
        vector=vector,
        pool=pool,
        body_links=body_links,
        threshold=threshold,
        report=report,
    )
    if best is None:
        return None
    score, path, item = best
    return _related_ref_from_candidate(path, item, score)


def _apply_calculated_related_section(
    markdown: str,
    *,
    current_path: str,
    related_page: RelatedPageRef | None,
    known_paths: set[str],
    path_titles: dict[str, str],
) -> str:
    if related_page is None:
        return _drop_sections(markdown, {"Related", "相关页面"}).rstrip() + "\n"
    body = related_logic.render_related_section(
        current_path=current_path,
        related_pages=[related_page],
        known_paths=known_paths,
        path_titles=path_titles,
    )
    if not body.strip():
        return _drop_sections(markdown, {"Related", "相关页面"}).rstrip() + "\n"
    return _replace_section(markdown, {"Related", "相关页面"}, "相关页面", body).rstrip() + "\n"


def _final_page_embedding_card(page: FinalPage, max_chars: int) -> str:
    body = _drop_sections(strip_frontmatter(page.markdown), {"Related", "相关页面"})
    values = [
        f"title: {page.title}",
        f"type: {page.page_type}",
        f"path: {page.target_path}",
        body,
    ]
    return _trim_text("\n\n".join(value for value in values if value.strip()), max_chars)


def _trim_text(text: str, max_chars: int) -> str:
    compact = re.sub(r"\n{3,}", "\n\n", text.strip())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rstrip()


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
    records, metrics = sync_page_embedding_cache(vault, entries, embedding_config)
    state["embedding_page_records"] = records
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
        visible_counts = {key: value for key, value in visible_counts.items() if key not in COMPLETION_LINE_HIDDEN_COUNT_KEYS}
        count_text = " ".join(f"{count_label(key)}={_format_count_value(key, value)}" for key, value in visible_counts.items())
        console.print(f"[green]完成[/] {step_label(name)} {record.duration_seconds:.2f}s {count_text}".rstrip())


COMPLETION_LINE_HIDDEN_COUNT_KEYS = {
    "api_call_count",
    "api_success_count",
    "api_paused_count",
    "prompt_tokens",
    "prompt_cache_hit_tokens",
    "prompt_cache_miss_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "total_tokens",
    "cache_hit_rate_percent",
    "price_cny",
}

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
    snapshot: WikiSnapshot | None = state.get("wiki_snapshot") if isinstance(state.get("wiki_snapshot"), WikiSnapshot) else None  # type: ignore[assignment]
    known_paths = {entry.path for entry in snapshot.entries} if snapshot else set()
    path_titles = {entry.path: entry.title for entry in snapshot.entries} if snapshot else {}
    known_paths.update(page.target_path for page in final_pages.pages)
    path_titles.update({page.target_path: page.title for page in final_pages.pages})
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
        issues.extend(
            related_logic.final_markdown_link_issues(
                markdown=page.markdown,
                target_path=page.target_path,
                title=page.title,
                known_paths=known_paths,
                path_titles=path_titles,
            )
        )
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
        "candidate_pages_warmup": run_dir / "candidate_pages_warmup" / "candidate_pages_warmup.json",
        "wiki_snapshot": run_dir / "wiki_snapshot" / "wiki_snapshot.json",
        "candidate_pages": run_dir / "candidate_pages" / "candidate_pages.json",
        "candidate_contexts": run_dir / "candidate_contexts" / "candidate_contexts.json",
        "merge_plan": run_dir / "merge_plan" / "merge_plan.json",
        "composition_plan": run_dir / "composition_plan" / "composition_plan.json",
        "final_pages": run_dir / "final_pages" / "final_pages.json",
        "related_refresh": run_dir / "related_refresh" / "related_refresh_report.json",
        "related_final_pages": run_dir / "related_refresh" / "final_pages.json",
        "knowledge_write_set": run_dir / "knowledge_write" / "write_set.json",
        "source_record_write": run_dir / "source_record_write" / "write_result.json",
        "embedding_cache_refresh": run_dir / "embedding_cache_refresh" / "embedding_cache_refresh.json",
        "related_maintenance": run_dir / "related_maintenance" / "related_maintenance_report.json",
        "related_maintenance_write": run_dir / "related_maintenance" / "write_result.json",
        "index_log_write": run_dir / "index_log_write" / "write_result.json",
    }
    return {name: sha256_file(path) for name, path in paths.items() if path.exists()}


def _render_source_digest_md(digest: SourceDigest) -> str:
    lines = [f"# 来源消化：{Path(digest.source_raw_path).name}", "", digest.summary, "", "## 页面单元计划"]
    for unit in digest.page_units:
        lines.append(f"## {unit.page_unit_id}: {unit.title}")
        lines.append(f"- 类型：{unit.page_type}")
        lines.append(f"- 路径提示：`{unit.path_hint}`")
        lines.append(f"- 内容范围：{unit.content_scope}")
        if unit.split_rationale:
            lines.append(f"- 拆分理由：{unit.split_rationale}")
        if unit.must_cover_points:
            lines.append("- 必须覆盖：")
            lines.extend(f"  - {item}" for item in unit.must_cover_points)
        lines.append("")
    if digest.weak_or_noise_items:
        lines.append("## 弱信息与噪声")
        for item in digest.weak_or_noise_items:
            lines.append(f"- {item.text}：{item.reason}")
        lines.append("")
    return "\n".join(lines) + "\n"


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
        lines.append(f"- 页面单元：`{page.page_unit_id}`")
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
        if decision.candidate_content_locators:
            lines.append(f"- 候选内容定位：{', '.join(decision.candidate_content_locators)}")
        lines.append(f"- 理由：{decision.reason}")
        lines.append(f"- 最强重合度：{decision.strongest_overlap}")
        lines.append("")
    return "\n".join(lines)


def _render_composition_plan_md(plan: CompositionPlan) -> str:
    lines = ["# 写作编排计划", ""]
    for item in plan.items:
        lines.append(f"## {item.final_page_id}: {item.target_path}")
        lines.append(f"- 动作：{_action_label(item.action)}")
        lines.append(f"- 合并决策：{', '.join(item.merge_decision_ids)}")
        lines.append(f"- 章节顺序：{', '.join(item.section_order)}")
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
