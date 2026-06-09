from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console

from . import __version__
from . import artifact_refs as _artifact_refs
from . import apply_guards as _apply_guards
from . import apply_preview as _apply_preview
from . import candidate_resolution as _candidate_resolution
from . import draft_grounding as _draft_grounding
from . import draft_rendering_runner as _draft_rendering_runner
from . import draft_rendering_payloads as _draft_rendering_payloads
from . import draft_reviewing as _draft_reviewing
from . import draft_validation as _draft_validation
from . import draft_write_assembly as _draft_write_assembly
from . import errors as _errors
from . import merge_plan_refinement as _merge_plan_refinement
from . import merge_planning as _merge_planning
from . import merge_reporting as _merge_reporting
from . import planning_payloads as _planning_payloads
from . import raw_steps as _raw_steps
from . import redaction as _redaction
from . import step_runtime as _step_runtime
from . import source_digest_budget as _source_digest_budget
from . import source_digest_payload as _source_digest_payload
from . import source_digest_rendering as _source_digest_rendering
from . import update_preservation as _update_preservation
from . import wiki_context as _wiki_context
from .events import EventLogger
from .hash_utils import sha256_file
from .io import read_model, read_yaml, write_json, write_yaml
from .manifest import (
    begin_model_step_attempt,
    begin_step_attempt,
    complete_step,
    fail_step,
    first_resumable_step,
    get_step,
    initial_steps,
    mark_step_awaiting_review,
    mark_step_approved,
    mark_from_pending,
    raw_ref,
    read_manifest,
    step_satisfied,
    write_manifest,
)
from .models import (
    CandidateResolutionArtifact,
    DraftRenderingArtifact,
    DraftWriteManifest,
    OperationConfigSnapshot,
    OperationManifest,
    OperationStatus,
    RawBinding,
    RawLinkCleanupArtifact,
    RawPreparePolicy,
    RawPreparationArtifact,
    ReviewDecision,
    SourceDigestArtifact,
    StepStatus,
    CandidateContextsArtifact,
    WikiContextSnapshot,
    WikiMergePlanArtifact,
    utc_now,
)
from .provider_config import ProviderExecutionContext, build_provider_execution_context
from .profiles import load_profile, profile_to_yaml_data, safe_filename
from .rendering import source_title_for_raw
from .retrieval import (
    resolve_cache_dir,
)
from . import run_metrics as _run_metrics
from .steps import (
    MODEL_BACKED_STEPS,
    STEP_NAMES,
    STEP_SPECS,
    StepSpec,
    downstream_steps,
    require_step_output_dir,
    step_index,
    step_output_dir,
)
from .system_pages import (
    ensure_system_pages,
    local_date,
)
from .validators import (
    nonempty_prepared_discovered_candidates,
    validate_candidate_resolution,
    validate_raw_preparation,
    validate_source_digest,
    validate_wiki_merge_plan,
)
from .verify import require_verified
from .vault_config import read_vault_config, write_default_vault_config
from .workspace import RunStore, apply_lock, ensure_workspace_layout, relative_to_vault, resolve_raw_path, run_lock


def init_vault(vault: Path, *, profile_name: str = "project_basic") -> None:
    profile = load_profile(profile_name)
    (vault / "raw").mkdir(parents=True, exist_ok=True)
    for spec in profile.page_types.values():
        (vault / "wiki" / spec.directory).mkdir(parents=True, exist_ok=True)
    ensure_system_pages(vault)
    ensure_workspace_layout(vault)
    write_default_vault_config(vault)
    profile_root = vault / ".llmwiki" / "profiles" / profile.name
    profile_root.mkdir(parents=True, exist_ok=True)
    write_yaml(profile_root / "profile.yaml", profile_to_yaml_data(profile))
    write_yaml(
        vault / ".llmwiki" / "config.yaml",
        {
            "profile": profile.name,
            "providers": {},
        },
    )


def run_simplified_ingest(
    *,
    vault: Path,
    raw_file: Path,
    mock_fixture_dir: Path | None = None,
    profile_name: str | None = None,
    slug: str | None = None,
    raw_prepare_policy: RawPreparePolicy | None = None,
    console: Console | None = None,
) -> OperationManifest:
    ensure_workspace_layout(vault)
    raw_path, raw_rel = resolve_raw_path(vault, raw_file)
    raw_hash, raw_size = raw_ref(raw_path)
    operation_id = f"ING-{_safe_timestamp()}-{slug or raw_path.stem}"
    store = RunStore(vault)
    if profile_name:
        resolved_profile_name = profile_name
    else:
        config = read_yaml(vault / ".llmwiki" / "config.yaml")
        resolved_profile_name = config.get("profile")
        if not isinstance(resolved_profile_name, str) or not resolved_profile_name:
            raise _errors.PipelineError(".llmwiki/config.yaml profile must be a non-empty string.")
    profile = load_profile(vault / ".llmwiki" / "profiles" / resolved_profile_name)
    vault_config = read_vault_config(vault)
    effective_raw_prepare_policy = raw_prepare_policy or RawPreparePolicy.auto
    vault_config_snapshot = OperationConfigSnapshot(
        **vault_config.model_dump(),
        raw_prepare_policy=effective_raw_prepare_policy,
    )
    model_steps = list(MODEL_BACKED_STEPS)
    if effective_raw_prepare_policy == RawPreparePolicy.skip:
        model_steps = [step for step in model_steps if step != "raw_prepare"]
    provider_execution_context = build_provider_execution_context(
        vault=vault,
        manifest_contexts=[],
        mock_fixture_dir=mock_fixture_dir,
        source="initial_run",
        from_step=None,
        tasks=model_steps,
    )
    with apply_lock(vault):
        run_dir = store.run_dir(operation_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = OperationManifest(
            operation_id=operation_id,
            operation_type="ingest",
            engine_version=__version__,
            profile=profile.name,
            profile_version=profile.version,
            vault_config_snapshot=vault_config_snapshot,
            workspace=relative_to_vault(vault, run_dir),
            raw_bindings=[RawBinding(relative_path=raw_rel, sha256=raw_hash, size_bytes=raw_size)],
            provider_contexts=[provider_execution_context.record] if provider_execution_context.record else [],
            steps=initial_steps(),
        )
        write_manifest(store.manifest_path(operation_id), manifest)
        with run_lock(vault, operation_id):
            return _execute_ingest(
                vault,
                operation_id,
                start_step=STEP_NAMES[0],
                execution_context=provider_execution_context,
                console=console,
            )


def resume_ingest(
    *,
    vault: Path,
    operation_id: str,
    from_step: str | None = None,
    mock_fixture_dir: Path | None = None,
    raw_prepare_policy: RawPreparePolicy | None = None,
    console: Console | None = None,
) -> OperationManifest:
    store = RunStore(vault)
    with apply_lock(vault), run_lock(vault, operation_id):
        manifest = read_manifest(store.manifest_path(operation_id))
        if manifest.status in {OperationStatus.applied, OperationStatus.source_recorded}:
            raise _errors.PipelineError("Applied operations are immutable. Start a new operation instead.")
        if manifest.status == OperationStatus.apply_failed:
            raise _errors.PipelineError("apply_failed operations cannot be resumed; inspect written targets and rerun ingest.")
        require_verified(vault, manifest)
        start, reset_from_step = _default_resume_start(vault, store.run_dir(operation_id), manifest, from_step)
        if start is None:
            return manifest
        if raw_prepare_policy is not None:
            if step_index(start) > step_index("raw_prepare"):
                raise _errors.PipelineError("raw prepare override only applies when raw_prepare will rerun; resume from raw_prepare or earlier.")
            manifest.vault_config_snapshot.raw_prepare_policy = raw_prepare_policy
        _validate_raw_link_cleanup_resume(run_dir=store.run_dir(operation_id), manifest=manifest, start=start)
        _validate_resume_start(manifest, start)
        _ensure_wiki_context_current_before_resume(vault, store.run_dir(operation_id), start)
        resumable_step_names = set(downstream_steps(start))
        model_steps = [step for step in MODEL_BACKED_STEPS if step in resumable_step_names]
        if "raw_prepare" in model_steps and manifest.vault_config_snapshot.raw_prepare_policy == RawPreparePolicy.skip:
            model_steps = [step for step in model_steps if step != "raw_prepare"]
        provider_execution_context = build_provider_execution_context(
            vault=vault,
            manifest_contexts=manifest.provider_contexts,
            mock_fixture_dir=mock_fixture_dir,
            source="resume_current_config",
            from_step=start,
            tasks=model_steps,
        )
        if provider_execution_context.record is not None:
            manifest.provider_contexts.append(provider_execution_context.record)
        if reset_from_step is not None:
            write_manifest(store.manifest_path(operation_id), manifest)
            _delete_downstream_step_dirs(vault, operation_id, reset_from_step)
            mark_from_pending(manifest, reset_from_step)
            write_manifest(store.manifest_path(operation_id), manifest)
        else:
            write_manifest(store.manifest_path(operation_id), manifest)
        return _execute_ingest(
            vault,
            operation_id,
            start_step=start,
            execution_context=provider_execution_context,
            console=console,
        )


def _default_resume_start(
    vault: Path,
    run_dir: Path,
    manifest: OperationManifest,
    from_step: str | None,
) -> tuple[str | None, str | None]:
    if from_step is not None:
        return from_step, from_step
    start = first_resumable_step(manifest)
    if start is not None:
        return start, None
    if manifest.status == OperationStatus.drafted and _drafted_wiki_context_drifted(vault, run_dir):
        return "wiki_context_snapshot", "wiki_context_snapshot"
    return None, None


def _drafted_wiki_context_drifted(vault: Path, run_dir: Path) -> bool:
    snapshot_path = require_step_output_dir(run_dir, "wiki_context_snapshot") / "wiki_context_snapshot.json"
    if not snapshot_path.exists():
        return False
    snapshot = read_model(snapshot_path, WikiContextSnapshot)
    return bool(_wiki_context.wiki_context_drift_messages(vault, snapshot))


def _ensure_wiki_context_current_before_resume(vault: Path, run_dir: Path, start_step: str) -> None:
    if step_index(start_step) <= step_index("wiki_context_snapshot"):
        return
    snapshot_path = require_step_output_dir(run_dir, "wiki_context_snapshot") / "wiki_context_snapshot.json"
    if not snapshot_path.exists():
        return
    snapshot = read_model(snapshot_path, WikiContextSnapshot)
    messages = _wiki_context.wiki_context_drift_messages(vault, snapshot)
    if messages:
        raise _errors.PipelineError("; ".join(messages))


def _execute_ingest(
    vault: Path,
    operation_id: str,
    *,
    start_step: str,
    execution_context: ProviderExecutionContext,
    console: Console | None = None,
) -> OperationManifest:
    store = RunStore(vault)
    run_dir = store.run_dir(operation_id)
    logger = EventLogger(operation_id, run_dir / "events.jsonl", console=console, redactor=execution_context.redactor)
    manifest = read_manifest(store.manifest_path(operation_id))
    profile = load_profile(vault / ".llmwiki" / "profiles" / manifest.profile)
    raw_path = vault / manifest.raw_bindings[0].relative_path
    _validate_resume_start(manifest, start_step)
    start_index = STEP_NAMES.index(start_step)
    for step_name in STEP_NAMES[start_index:]:
        if step_satisfied(get_step(manifest, step_name).status):
            continue
        try:
            _run_step(
                step_name,
                vault,
                store.manifest_path(operation_id),
                run_dir,
                raw_path,
                profile,
                manifest,
                logger,
                execution_context,
            )
        except Exception as exc:
            message = execution_context.redactor.redact_text(str(exc))
            fail_step(manifest, step_name, message)
            write_manifest(store.manifest_path(operation_id), manifest)
            _run_metrics.refresh_run_metrics(run_dir, manifest, warning_console=console)
            logger.emit(step_name, "failed", status="failed", message=message, duration_ms=_last_attempt_duration_ms(manifest, step_name))
            raise _errors.PipelineError(message) from exc
        write_manifest(store.manifest_path(operation_id), manifest)
        _run_metrics.refresh_run_metrics(run_dir, manifest, warning_console=console)
        if get_step(manifest, step_name).status == StepStatus.awaiting_review:
            manifest.status = OperationStatus.awaiting_review
            write_manifest(store.manifest_path(operation_id), manifest)
            _run_metrics.refresh_run_metrics(run_dir, manifest, warning_console=console)
            return manifest
    incomplete = [step for step in manifest.steps if not step_satisfied(step.status)]
    if incomplete:
        details = ", ".join(f"{step.name}={step.status.value}" for step in incomplete)
        raise _errors.PipelineError(f"Operation is not draft-ready; incomplete step(s): {details}")
    manifest.status = OperationStatus.drafted
    write_manifest(store.manifest_path(operation_id), manifest)
    _run_metrics.refresh_run_metrics(run_dir, manifest, warning_console=console)
    return manifest


def _validate_resume_start(manifest: OperationManifest, start_step: str) -> None:
    start_index = step_index(start_step)
    for step in manifest.steps[:start_index]:
        if not step_satisfied(step.status):
            raise _errors.PipelineError(
                f"Cannot resume from {start_step}: upstream step {step.name} is {step.status.value}; "
                f"resume from {step.name} or earlier."
            )


def _run_step(
    step_name: str,
    vault: Path,
    manifest_path: Path,
    run_dir: Path,
    raw_path: Path,
    profile,
    manifest: OperationManifest,
    logger: EventLogger,
    execution_context: ProviderExecutionContext,
) -> None:
    runner = STEP_RUNNERS.get(step_name)
    if runner is None:
        raise _errors.PipelineError(f"Unknown step: {step_name}")
    provider_record = execution_context.record if runner.spec.model_backed else None
    provider_runtime = provider_record.providers.get(step_name) if provider_record else None
    provider_spec_for_attempt = provider_runtime.spec if provider_runtime else None
    local_model_backed_step = (
        runner.spec.model_backed
        and provider_runtime is None
        and step_name == "raw_prepare"
        and manifest.vault_config_snapshot.raw_prepare_policy == RawPreparePolicy.skip
    )
    logger.emit(
        step_name,
        "started",
        status="running",
        model_backed=runner.spec.model_backed,
        provider_spec=provider_spec_for_attempt,
        may_use_local_shortcut=step_name in {"raw_prepare", "wiki_merge_planning"},
    )
    if runner.spec.model_backed:
        if local_model_backed_step:
            begin_step_attempt(manifest, step_name)
        elif provider_record is None or provider_runtime is None:
            raise _errors.PipelineError(f"No provider execution context found for model-backed step: {step_name}")
        else:
            begin_model_step_attempt(
                manifest,
                step_name,
                provider_record_id=provider_record.record_id,
                provider_spec=provider_spec_for_attempt,
                provider_context_source=provider_record.source,
            )
    else:
        begin_step_attempt(manifest, step_name)
    write_manifest(manifest_path, manifest)
    ctx = _step_runtime.StepRunContext(
        vault=vault,
        run_dir=run_dir,
        raw_path=raw_path,
        profile=profile,
        manifest=manifest,
        execution_context=execution_context,
    )
    output_dir = step_output_dir(run_dir, step_name)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    runner.run(ctx)
    status = get_step(manifest, step_name).status.value
    logger.emit(
        step_name,
        "completed",
        status=status,
        message=_step_completion_message(ctx, step_name),
        duration_ms=_last_attempt_duration_ms(manifest, step_name),
    )


def _run_source_digest(ctx: _step_runtime.StepRunContext) -> None:
    step_name = "source_digest"
    step_root = require_step_output_dir(ctx.run_dir, step_name)
    raw_rel = relative_to_vault(ctx.vault, ctx.raw_path)
    approved_prepared = require_step_output_dir(ctx.run_dir, "prepared_raw_review") / "approved_prepared.md"
    approved_prepared_text = approved_prepared.read_text(encoding="utf-8")
    approved_prepared_ref = approved_prepared.relative_to(ctx.run_dir).as_posix()
    source_map = _source_digest_payload.build_source_digest_source_map(approved_prepared_text, approved_prepared_ref=approved_prepared_ref)
    source_map_path = step_root / "source_digest_source_map.json"
    source_map_md = step_root / "source_digest_source_map.md"
    write_json(source_map_path, source_map)
    source_map_md.write_text(_source_digest_payload.render_source_digest_source_map_markdown(source_map), encoding="utf-8")
    source_map_payload = _source_digest_payload.project_source_digest_source_map_for_payload(
        source_map,
        full_source_map_ref=source_map_path.relative_to(ctx.run_dir).as_posix(),
    )
    source_map_payload_path = step_root / "source_digest_source_map_payload.json"
    write_json(source_map_payload_path, source_map_payload)
    source_kind_hints = _source_digest_payload.build_source_kind_hints(approved_prepared_text, raw_rel)
    source_kind_hints_path = step_root / "source_kind_hints.json"
    source_kind_hints_md = step_root / "source_kind_hints.md"
    write_json(source_kind_hints_path, source_kind_hints)
    source_kind_hints_md.write_text(_source_digest_payload.render_source_kind_hints_markdown(source_kind_hints), encoding="utf-8")
    payload = _source_digest_payload.build_source_digest_payload(
        raw_rel=raw_rel,
        approved_prepared_text=approved_prepared_text,
        approved_prepared_ref=approved_prepared_ref,
        source_map_payload=source_map_payload,
        source_kind_hints=source_kind_hints,
        profile=ctx.profile.model_dump(mode="json"),
        vault_config=ctx.manifest.vault_config_snapshot,
    )
    digest, _ = _step_runtime.structured_call(ctx.run_dir, ctx.execution_context, step_name).run(
        step_name,
        payload,
        SourceDigestArtifact,
        validator=lambda model: validate_source_digest(model, language=ctx.manifest.vault_config_snapshot.wiki_language),
    )
    digest = _redaction.redact_model(ctx.execution_context.redactor, digest, SourceDigestArtifact)
    if digest.source_raw_path != raw_rel:
        raise _errors.PipelineError(f"source_digest source path mismatch: {digest.source_raw_path} != {raw_rel}")
    digest = _source_digest_budget.augment_source_digest_anchor_entities(digest, approved_prepared_text)
    digest, budget_report = _source_digest_budget.cap_source_digest_candidates(digest, ctx.manifest.vault_config_snapshot.max_ingest_candidates)
    validate_source_digest(digest, language=ctx.manifest.vault_config_snapshot.wiki_language)
    out = step_root / "source_digest.json"
    write_json(out, digest)
    digest_md = step_root / "source_digest.md"
    digest_md.write_text(_source_digest_rendering.render_source_digest_markdown(digest), encoding="utf-8")
    budget_report_path = step_root / "source_digest_budget_report.json"
    budget_report_md = step_root / "source_digest_budget_report.md"
    write_json(budget_report_path, budget_report)
    budget_report_md.write_text(_source_digest_budget.render_source_digest_budget_report(budget_report), encoding="utf-8")
    outputs = [
        _artifact_refs.ref(ctx.run_dir, source_map_path, step_name, "json", "source_digest_source_map.v1"),
        _artifact_refs.ref(ctx.run_dir, source_map_md, step_name, "markdown"),
        _artifact_refs.ref(ctx.run_dir, source_map_payload_path, step_name, "json", "source_digest_source_map_payload.v1"),
        _artifact_refs.ref(ctx.run_dir, source_kind_hints_path, step_name, "json", "source_kind_hints.v1"),
        _artifact_refs.ref(ctx.run_dir, source_kind_hints_md, step_name, "markdown"),
        _artifact_refs.ref(ctx.run_dir, out, step_name, "json", "source_digest.v2"),
        _artifact_refs.ref(ctx.run_dir, digest_md, step_name, "markdown"),
        _artifact_refs.ref(ctx.run_dir, budget_report_path, step_name, "json", "source_digest_budget_report.v1"),
        _artifact_refs.ref(ctx.run_dir, budget_report_md, step_name, "markdown"),
    ]
    outputs.extend(_artifact_refs.structured_model_output_refs(ctx.run_dir, step_root, step_name))
    complete_step(ctx.manifest, step_name, outputs=outputs)


def _run_source_digest_review(ctx: _step_runtime.StepRunContext) -> None:
    step_name = "source_digest_review"
    step_root = require_step_output_dir(ctx.run_dir, step_name)
    digest_json = require_step_output_dir(ctx.run_dir, "source_digest") / "source_digest.json"
    digest_md = require_step_output_dir(ctx.run_dir, "source_digest") / "source_digest.md"
    approved_json = step_root / "approved_digest.json"
    approved_md = step_root / "approved_digest.md"
    approved_json.write_text(digest_json.read_text(encoding="utf-8"), encoding="utf-8")
    approved_md.write_text(digest_md.read_text(encoding="utf-8"), encoding="utf-8")
    prompt = step_root / "review_prompt.md"
    feedback = step_root / "review_feedback.jsonl"
    prompt.write_text(
        "# Source Digest 审核\n\n"
        "当前运行自动批准 source digest；这一步检查单篇材料提取是否完整、是否中文、是否把弱相关内容放进噪声区。\n\n"
        "## 核心判断\n\n"
        "- 是否漏掉了值得入库的实体、概念、设计、对比或未决问题？\n"
        "- 是否把只是口播过渡、广告、寒暄或弱相关提及错误变成页面候选？\n"
        "- 摘要、关键收获和候选说明是否为中文，英文术语是否有中文上下文？\n\n"
        "## 关键文件\n\n"
        f"- 待审摘要：`{digest_md.relative_to(ctx.run_dir).as_posix()}`\n"
        f"- 可编辑批准文件：`{approved_json.relative_to(ctx.run_dir).as_posix()}`\n"
        f"- 反馈记录：`{feedback.relative_to(ctx.run_dir).as_posix()}`\n\n"
        "后续可在这里接入 list/filter/show/diff/revise/approve 审核操作。\n",
        encoding="utf-8",
    )
    feedback.write_text("", encoding="utf-8")
    decision = ReviewDecision(
        review_step=step_name,
        decision="approved",
        auto_approved=True,
        notes="当前运行自动批准 source digest；交互式 digest 审核尚未接入。",
    )
    decision_path = step_root / "review_decision.json"
    write_json(decision_path, decision)
    _step_runtime.complete_review_step(
        ctx.manifest,
        step_name,
        outputs=[
            _artifact_refs.ref(ctx.run_dir, prompt, step_name, "markdown"),
            _artifact_refs.ref(ctx.run_dir, feedback, step_name, "jsonl"),
            _artifact_refs.ref(ctx.run_dir, decision_path, step_name, "json", "review_decision.v2"),
            _artifact_refs.ref(ctx.run_dir, approved_json, step_name, "json", "source_digest.v2"),
            _artifact_refs.ref(ctx.run_dir, approved_md, step_name, "markdown"),
        ],
        review_decision_ref=decision_path.relative_to(ctx.run_dir).as_posix(),
    )


def _run_source_duplicate_guard(ctx: _step_runtime.StepRunContext) -> None:
    step_name = "source_duplicate_guard"
    step_root = require_step_output_dir(ctx.run_dir, step_name)
    digest = read_model(require_step_output_dir(ctx.run_dir, "source_digest_review") / "approved_digest.json", SourceDigestArtifact)
    prepared = require_step_output_dir(ctx.run_dir, "prepared_raw_review") / "approved_prepared.md"
    artifact = _apply_guards.build_source_duplicate_guard_artifact(
        ctx.vault,
        source_raw_path=digest.source_raw_path,
        source_raw_hash=sha256_file(ctx.vault / digest.source_raw_path),
        source_prepared_hash=sha256_file(prepared),
        operation_id=ctx.manifest.operation_id,
    )
    out = step_root / "source_duplicate_guard.json"
    write_json(out, artifact)
    md = step_root / "source_duplicate_guard.md"
    md.write_text(_apply_guards.render_source_duplicate_guard_markdown(artifact), encoding="utf-8")
    if artifact.status == "source_duplicate":
        raise _errors.PipelineError(f"source_duplicate: {artifact.reason}")
    if artifact.status == "source_revision_detected":
        raise _errors.PipelineError(
            "同一路径内容已变化，source revision workflow 尚未实现；如确认为新材料，请另存为新 raw 文件名后重新 ingest。"
        )
    complete_step(
        ctx.manifest,
        step_name,
        outputs=[
            _artifact_refs.ref(ctx.run_dir, out, step_name, "json", "source_duplicate_guard.v1"),
            _artifact_refs.ref(ctx.run_dir, md, step_name, "markdown"),
        ],
    )


def _run_candidate_resolution(ctx: _step_runtime.StepRunContext) -> None:
    step_name = "candidate_resolution"
    step_root = require_step_output_dir(ctx.run_dir, step_name)
    digest = read_model(require_step_output_dir(ctx.run_dir, "source_digest_review") / "approved_digest.json", SourceDigestArtifact)
    approved_prepared = require_step_output_dir(ctx.run_dir, "prepared_raw_review") / "approved_prepared.md"
    validate_source_digest(digest, language=ctx.manifest.vault_config_snapshot.wiki_language)
    approved_prepared_text = approved_prepared.read_text(encoding="utf-8")
    source_pack = _planning_payloads.build_candidate_resolution_source_pack(approved_prepared_text, digest)
    source_pack_path = step_root / "candidate_resolution_source_excerpt_pack.json"
    source_pack_md = step_root / "candidate_resolution_source_excerpt_pack.md"
    write_json(source_pack_path, source_pack)
    source_pack_md.write_text(_planning_payloads.render_candidate_resolution_source_pack_markdown(source_pack), encoding="utf-8")
    payload = {
        "approved_prepared_markdown": approved_prepared_text if source_pack["full_source_in_payload"] else "",
        "approved_prepared_ref": "prepared_raw_review/approved_prepared.md",
        "source_excerpt_pack": source_pack,
        "approved_digest": digest.model_dump(mode="json"),
        "profile": ctx.profile.model_dump(mode="json"),
        "language_contract": ctx.manifest.vault_config_snapshot.model_dump(mode="json"),
        "candidate_coverage_required_ids": [candidate.candidate_id for candidate in digest.ingest_candidates()],
        "contract": {
            "goal": "Do coverage check and topic/page planning. Do not summarize the source again and do not decide create/update.",
            "required_minimum_fields": [
                "source_basis",
                "page_type",
                "display_title",
                "topic_summary",
                "why_this_page",
            ],
            "rules": [
                "Every approved_digest ingest candidate id must appear in source_basis.source_candidate_ids exactly once, including open_questions.",
                "Do not output weak_or_noise_items, noise-* ids, page_type=noise, or reason=ignore as formal page plans.",
                "If a weak/noise item is not wiki-worthy, omit it entirely from candidate_resolution instead of converting it to a concept.",
                "Use approved_digest candidates as the primary basis, but inspect approved_prepared for missed wiki-worthy topics.",
                "When approved_prepared_markdown is empty, use source_excerpt_pack as the only model-visible source support; the full approved source remains fixed by approved_prepared_ref for local audit.",
                "budget_deferred_candidates are not selected ingest candidates in this run; if you reuse one, put its id in prepared_discovered_candidates, not source_candidate_ids.",
                "Put any newly discovered topic in prepared_discovered_candidates.",
                "Do not read or infer existing wiki state.",
                "Write all user-visible fields in Chinese unless retaining a stable domain term.",
                "Leave page_plan_id/path_stem/candidate_target_path empty if unsure; the engine will deterministically set them.",
            ],
        },
    }
    def validate_candidate_resolution_model(model: CandidateResolutionArtifact) -> None:
        candidate = _candidate_resolution.finalize_candidate_resolution(ctx.vault, ctx.profile, model, digest)
        validate_candidate_resolution(digest, candidate)

    artifact, _ = _step_runtime.structured_call(ctx.run_dir, ctx.execution_context, step_name).run(
        step_name,
        payload,
        CandidateResolutionArtifact,
        validator=validate_candidate_resolution_model,
    )
    artifact = _redaction.redact_model(ctx.execution_context.redactor, artifact, CandidateResolutionArtifact)
    artifact = _candidate_resolution.backfill_missing_candidate_resolution_items(artifact, digest, ctx.profile)
    artifact = _candidate_resolution.finalize_candidate_resolution(ctx.vault, ctx.profile, artifact, digest)
    validate_candidate_resolution(digest, artifact)
    out = step_root / "candidate_resolution.json"
    write_json(out, artifact)
    table = step_root / "candidate_resolution.md"
    table.write_text(_candidate_resolution.render_candidate_resolution_markdown(artifact), encoding="utf-8")
    outputs = [
        _artifact_refs.ref(ctx.run_dir, source_pack_path, step_name, "json", "candidate_resolution_source_excerpt_pack.v1"),
        _artifact_refs.ref(ctx.run_dir, source_pack_md, step_name, "markdown"),
        _artifact_refs.ref(ctx.run_dir, out, step_name, "json", "candidate_resolution.v3"),
        _artifact_refs.ref(ctx.run_dir, table, step_name, "markdown"),
    ]
    outputs.extend(_artifact_refs.structured_model_output_refs(ctx.run_dir, step_root, step_name))
    complete_step(
        ctx.manifest,
        step_name,
        outputs=outputs,
    )


def _run_wiki_context_snapshot(ctx: _step_runtime.StepRunContext) -> None:
    step_name = "wiki_context_snapshot"
    step_root = require_step_output_dir(ctx.run_dir, step_name)
    resolution = read_model(
        require_step_output_dir(ctx.run_dir, "candidate_resolution") / "candidate_resolution.json",
        CandidateResolutionArtifact,
    )
    digest = read_model(require_step_output_dir(ctx.run_dir, "source_digest_review") / "approved_digest.json", SourceDigestArtifact)
    source_title = source_title_for_raw(digest.source_raw_path)
    log_date = local_date()
    retrieval_config = ctx.manifest.vault_config_snapshot.embedding_retrieval
    provider_runtimes = list(ctx.execution_context.record.providers.values()) if ctx.execution_context.record else []
    force_exact_backend = bool(provider_runtimes) and all(provider.spec.startswith("mock:") for provider in provider_runtimes)
    snapshot = _wiki_context.build_wiki_context_snapshot(
        ctx.vault,
        resolution,
        log_date=log_date,
        source_target_path=f"sources/{safe_filename(source_title)}.md",
        retrieval_config=retrieval_config,
        force_exact_backend=force_exact_backend,
    )
    snapshot_content_chars = sum(len(entry.content) for entry in snapshot.entries)
    max_context_chars = ctx.manifest.vault_config_snapshot.max_context_chars
    if snapshot_content_chars > max_context_chars:
        raise _errors.PipelineError(
            f"wiki_context_snapshot exceeds max_context_chars ({snapshot_content_chars} > {max_context_chars}); retry with a smaller vault or higher limit."
        )
    contexts_path = step_root / "candidate_contexts.json"
    write_json(contexts_path, snapshot.candidate_contexts)
    contexts_md = step_root / "candidate_contexts.md"
    contexts_md.write_text(
        _merge_reporting.render_candidate_contexts_markdown(
            snapshot.candidate_contexts,
            resolved_cache_path=resolve_cache_dir(ctx.vault, retrieval_config.cache_dir).as_posix(),
            query_count=len(snapshot.candidate_contexts.items),
            encoded_page_count=snapshot.candidate_contexts.candidate_pool_size
            if snapshot.candidate_contexts.retrieval_backend == "sentence_transformers"
            else 0,
        ),
        encoding="utf-8",
    )
    snapshot = snapshot.model_copy(update={"candidate_contexts_ref": contexts_path.relative_to(ctx.run_dir).as_posix()})
    snapshot_path = step_root / "wiki_context_snapshot.json"
    write_json(snapshot_path, snapshot)
    complete_step(
        ctx.manifest,
        step_name,
        outputs=[
            _artifact_refs.ref(ctx.run_dir, snapshot_path, step_name, "json", "wiki_context_snapshot.v3"),
            _artifact_refs.ref(ctx.run_dir, contexts_path, step_name, "json", "candidate_contexts.v2"),
            _artifact_refs.ref(ctx.run_dir, contexts_md, step_name, "markdown"),
        ],
    )


def _run_wiki_merge_planning(ctx: _step_runtime.StepRunContext) -> None:
    step_name = "wiki_merge_planning"
    step_root = require_step_output_dir(ctx.run_dir, step_name)
    resolution = read_model(
        require_step_output_dir(ctx.run_dir, "candidate_resolution") / "candidate_resolution.json",
        CandidateResolutionArtifact,
    )
    digest = read_model(require_step_output_dir(ctx.run_dir, "source_digest_review") / "approved_digest.json", SourceDigestArtifact)
    snapshot_path = require_step_output_dir(ctx.run_dir, "wiki_context_snapshot") / "wiki_context_snapshot.json"
    snapshot = read_model(snapshot_path, WikiContextSnapshot)
    candidate_contexts = read_model(require_step_output_dir(ctx.run_dir, "wiki_context_snapshot") / "candidate_contexts.json", CandidateContextsArtifact)
    approved_prepared_text = (require_step_output_dir(ctx.run_dir, "prepared_raw_review") / "approved_prepared.md").read_text(encoding="utf-8")
    snapshot_ref = snapshot_path.relative_to(ctx.run_dir).as_posix()
    context_pack = _planning_payloads.build_merge_planning_context_pack(
        approved_prepared_text=approved_prepared_text,
        digest=digest,
        resolution=resolution,
        snapshot=snapshot,
        candidate_contexts=candidate_contexts,
        snapshot_ref=snapshot_ref,
    )
    context_pack_path = step_root / "merge_planning_context_pack.json"
    context_pack_md = step_root / "merge_planning_context_pack.md"
    write_json(context_pack_path, context_pack)
    context_pack_md.write_text(_planning_payloads.render_merge_planning_context_pack_markdown(context_pack), encoding="utf-8")
    runtime = ctx.execution_context.runtime_for_task(step_name)
    if not runtime.spec.startswith("mock:"):
        shortcut_report = _merge_planning.empty_vault_create_merge_planning_shortcut_report(digest, resolution, snapshot, candidate_contexts)
    else:
        shortcut_report = {"used": False}
    if shortcut_report["used"]:
        shortcut_report_path = step_root / "merge_planning_shortcut_report.json"
        shortcut_report_md = step_root / "merge_planning_shortcut_report.md"
        write_json(shortcut_report_path, shortcut_report)
        shortcut_report_md.write_text(_merge_reporting.render_merge_planning_shortcut_report(shortcut_report), encoding="utf-8")
        plan = _merge_planning.build_wiki_merge_plan(resolution, digest, snapshot, log_date=snapshot.log_date)
        plan = _merge_planning.finalize_wiki_merge_plan(plan, resolution, snapshot, snapshot_ref, medium_missing_policy="preserve")
        plan = _merge_planning.block_unrepaired_medium_create_reason(plan)
        validate_wiki_merge_plan(digest, plan, resolution, snapshot, language=ctx.manifest.vault_config_snapshot.wiki_language)
        out = step_root / "wiki_merge_plan.json"
        write_json(out, plan)
        table = step_root / "wiki_merge_plan.md"
        table.write_text(_merge_reporting.render_merge_plan_markdown(plan), encoding="utf-8")
        report = step_root / "merge_decision_report.md"
        report.write_text(_merge_reporting.render_merge_decision_report(plan, snapshot), encoding="utf-8")
        complete_step(
            ctx.manifest,
            step_name,
            outputs=[
                _artifact_refs.ref(ctx.run_dir, context_pack_path, step_name, "json", "merge_planning_context_pack.v1"),
                _artifact_refs.ref(ctx.run_dir, context_pack_md, step_name, "markdown"),
                _artifact_refs.ref(ctx.run_dir, shortcut_report_path, step_name, "json", "merge_planning_shortcut_report.v1"),
                _artifact_refs.ref(ctx.run_dir, shortcut_report_md, step_name, "markdown"),
                _artifact_refs.ref(ctx.run_dir, out, step_name, "json", "wiki_merge_plan.v5"),
                _artifact_refs.ref(ctx.run_dir, table, step_name, "markdown"),
                _artifact_refs.ref(ctx.run_dir, report, step_name, "markdown"),
            ],
        )
        return
    payload = {
        "approved_prepared_markdown": approved_prepared_text if context_pack["full_source_in_payload"] else "",
        "approved_prepared_ref": context_pack["approved_prepared_ref"],
        "source_excerpt_pack": context_pack["source_excerpt_pack"],
        "approved_digest": digest.model_dump(mode="json"),
        "approved_digest_ref": context_pack["approved_digest_ref"],
        "candidate_resolution": resolution.model_dump(mode="json"),
        "candidate_resolution_ref": context_pack["candidate_resolution_ref"],
        "wiki_context_snapshot": context_pack["wiki_context_projection"],
        "wiki_context_snapshot_ref": snapshot_ref,
        "candidate_contexts": context_pack["candidate_contexts_projection"],
        "merge_planning_context_pack": _planning_payloads.merge_planning_payload_pack_summary(context_pack),
        "profile": ctx.profile.model_dump(mode="json"),
        "language_contract": ctx.manifest.vault_config_snapshot.model_dump(mode="json"),
        "contract": {
            "goal": "Read the compact frozen wiki context projection and decide create/update/noop/needs_human_decision for planned pages.",
            "actions": ["create", "update", "noop", "needs_human_decision"],
            "rules": [
                "approved_digest and candidate_resolution are the reviewed upstream artifacts; approved_digest_ref/candidate_resolution_ref identify their fixed local audit copies.",
                "wiki_context_snapshot is a compact projection; the full snapshot is fixed at wiki_context_snapshot_ref and will be used by local validators.",
                "Use source_excerpt_pack as source support when approved_prepared_markdown is empty; the full approved source remains fixed at approved_prepared_ref.",
                "For each page_plan_id, inspect candidate_contexts Top5 before choosing create/update/noop/needs_human_decision.",
                "Write inspected_context_paths using wiki-root-relative paths from candidate_contexts hits.",
                "For create with medium or strong inspected overlap, explain why_not_update against the strongest inspected old page.",
                "A valid why_not_update must compare scope_delta, source_delta, why_update_not_enough, and why_related_link_not_enough in concrete Chinese prose.",
                "Do not rely only on target page missing, different folder, or different page type as the reason to create.",
                "Prefer update when the new source adds examples, boundaries, counterexamples, use cases, value points, or clarifications to the same knowledge question/concept/methodology.",
                "Use create only when the topic is an independent reusable knowledge unit that cannot naturally live inside an inspected existing page.",
                "If new material partially answers an existing open_question, update that open_question and related the stable new concept/design/comparison page back to it.",
                "If the new page only answers one paragraph of an old open_question, prefer update instead of create.",
                "If an exact path/title/alias inspected context strongly overlaps but you still want create, use needs_human_decision.",
                "Suggest at most two related_pages; they must come from source sibling pages, inspected context pages, or exact title/alias matches.",
                "Every related_pages reason must be Chinese and explain a concrete relation from source relation, merge comparison, or sibling complement.",
                "Only same source is not enough for Related; prefer upstream/downstream, complement, counterexample, use scenario, or method dependency.",
                "If no Related is suitable, set related_absence_reason and explain whether the page is isolated, weakly related, self-only, or already navigable from index.",
                "Use canonical_target_path for final writes; for update use the matched existing page path.",
                "needs_human_decision is not writeable and must be resolved before drafting.",
                "noop only when an inspected existing wiki page already fully covers the source without new examples, expressions, links, or value points.",
                "For noop, bind matched_page to the inspected existing page and explain the no-content-delta reason.",
                "New but thin topics should be create, not noop.",
                "All user-visible fields must be Chinese unless keeping stable domain terms.",
                "Keep value_points and reuse_scenarios grounded in concrete source content.",
            ],
        },
    }
    def validate_merge_model(model: WikiMergePlanArtifact) -> None:
        candidate = _merge_planning.finalize_wiki_merge_plan(
            model,
            resolution,
            snapshot,
            snapshot_path.relative_to(ctx.run_dir).as_posix(),
            medium_missing_policy="preserve",
        )
        validate_wiki_merge_plan(digest, candidate, resolution, snapshot, language=ctx.manifest.vault_config_snapshot.wiki_language)

    plan, _ = _step_runtime.structured_call(ctx.run_dir, ctx.execution_context, step_name).run(
        step_name,
        payload,
        WikiMergePlanArtifact,
        validator=validate_merge_model,
    )
    plan = _redaction.redact_model(ctx.execution_context.redactor, plan, WikiMergePlanArtifact)
    plan = _merge_planning.finalize_wiki_merge_plan(plan, resolution, snapshot, snapshot_ref, medium_missing_policy="preserve")
    plan = _merge_planning.block_unrepaired_medium_create_reason(plan)
    validate_wiki_merge_plan(digest, plan, resolution, snapshot, language=ctx.manifest.vault_config_snapshot.wiki_language)
    out = step_root / "wiki_merge_plan.json"
    write_json(out, plan)
    table = step_root / "wiki_merge_plan.md"
    table.write_text(_merge_reporting.render_merge_plan_markdown(plan), encoding="utf-8")
    report = step_root / "merge_decision_report.md"
    report.write_text(_merge_reporting.render_merge_decision_report(plan, snapshot), encoding="utf-8")
    outputs = [
        _artifact_refs.ref(ctx.run_dir, context_pack_path, step_name, "json", "merge_planning_context_pack.v1"),
        _artifact_refs.ref(ctx.run_dir, context_pack_md, step_name, "markdown"),
        _artifact_refs.ref(ctx.run_dir, out, step_name, "json", "wiki_merge_plan.v5"),
        _artifact_refs.ref(ctx.run_dir, table, step_name, "markdown"),
        _artifact_refs.ref(ctx.run_dir, report, step_name, "markdown"),
    ]
    outputs.extend(_artifact_refs.structured_model_output_refs(ctx.run_dir, step_root, step_name))
    complete_step(
        ctx.manifest,
        step_name,
        outputs=outputs,
    )


def _run_merge_plan_review(ctx: _step_runtime.StepRunContext) -> None:
    step_name = "merge_plan_review"
    step_root = require_step_output_dir(ctx.run_dir, step_name)
    plan_path = require_step_output_dir(ctx.run_dir, "wiki_merge_planning") / "wiki_merge_plan.json"
    plan = read_model(plan_path, WikiMergePlanArtifact)
    approved_path = step_root / "approved_merge_plan.json"
    prompt_path = step_root / "review_prompt.md"
    feedback_path = step_root / "review_feedback.jsonl"
    feedback_path.write_text("", encoding="utf-8")
    prompt_path.write_text(_merge_reporting.render_merge_plan_review_prompt(plan), encoding="utf-8")
    has_needs_human = any(item.action == "needs_human_decision" or item.apply_eligibility == "blocked" for item in plan.items)
    max_auto_create_items = min(
        ctx.manifest.vault_config_snapshot.max_ingest_candidates,
        _merge_plan_refinement.MAX_AUTO_APPROVED_ALL_CREATE_ITEMS,
    )
    all_create_risk = _merge_plan_refinement.merge_plan_all_create_review_reason(plan, max_auto_create_items=max_auto_create_items)
    if has_needs_human or all_create_risk:
        pending_path = step_root / "pending_merge_plan.json"
        pending_path.write_text(plan_path.read_text(encoding="utf-8"), encoding="utf-8")
        decision = ReviewDecision(
            review_step=step_name,
            decision="pending",
            auto_approved=False,
            notes=all_create_risk or "合并计划包含 needs_human_decision；继续前需要先 revise 为 create/update/noop。",
        )
        decision_path = step_root / "review_decision.json"
        write_json(decision_path, decision)
        mark_step_awaiting_review(
            ctx.manifest,
            step_name,
            outputs=[
                _artifact_refs.ref(ctx.run_dir, prompt_path, step_name, "markdown"),
                _artifact_refs.ref(ctx.run_dir, feedback_path, step_name, "jsonl"),
                _artifact_refs.ref(ctx.run_dir, pending_path, step_name, "json", "wiki_merge_plan.v5"),
                _artifact_refs.ref(ctx.run_dir, decision_path, step_name, "json", "review_decision.v2"),
            ],
            reason=all_create_risk or "merge plan requires human decision; run merge-level revise.",
            review_decision_ref=decision_path.relative_to(ctx.run_dir).as_posix(),
        )
        return
    approved_path.write_text(plan_path.read_text(encoding="utf-8"), encoding="utf-8")
    decision = ReviewDecision(
        review_step=step_name,
        decision="approved",
        auto_approved=True,
        notes="合并计划不含 needs_human_decision，当前运行自动批准。",
    )
    decision_path = step_root / "review_decision.json"
    write_json(decision_path, decision)
    _step_runtime.complete_review_step(
        ctx.manifest,
        step_name,
        outputs=[
            _artifact_refs.ref(ctx.run_dir, prompt_path, step_name, "markdown"),
            _artifact_refs.ref(ctx.run_dir, feedback_path, step_name, "jsonl"),
            _artifact_refs.ref(ctx.run_dir, decision_path, step_name, "json", "review_decision.v2"),
            _artifact_refs.ref(ctx.run_dir, approved_path, step_name, "json", "wiki_merge_plan.v5"),
        ],
        review_decision_ref=decision_path.relative_to(ctx.run_dir).as_posix(),
    )


def _run_draft_rendering(ctx: _step_runtime.StepRunContext) -> None:
    step_name = "draft_rendering"
    step_root = require_step_output_dir(ctx.run_dir, step_name)
    digest = read_model(require_step_output_dir(ctx.run_dir, "source_digest_review") / "approved_digest.json", SourceDigestArtifact)
    resolution = read_model(
        require_step_output_dir(ctx.run_dir, "candidate_resolution") / "candidate_resolution.json",
        CandidateResolutionArtifact,
    )
    merge_plan = read_model(require_step_output_dir(ctx.run_dir, "merge_plan_review") / "approved_merge_plan.json", WikiMergePlanArtifact)
    snapshot = read_model(require_step_output_dir(ctx.run_dir, "wiki_context_snapshot") / "wiki_context_snapshot.json", WikiContextSnapshot)
    validate_source_digest(digest, language=ctx.manifest.vault_config_snapshot.wiki_language)
    validate_wiki_merge_plan(digest, merge_plan, resolution, snapshot, language=ctx.manifest.vault_config_snapshot.wiki_language)
    _step_runtime.ensure_wiki_context_current(ctx.vault, snapshot)
    approved_prepared_path = require_step_output_dir(ctx.run_dir, "prepared_raw_review") / "approved_prepared.md"
    approved_prepared_text = approved_prepared_path.read_text(encoding="utf-8")
    draftable_count = len([item for item in merge_plan.items if item.action in {"create", "update"}])
    uses_draft_batches = draftable_count > _draft_rendering_runner.DRAFT_RENDERING_BATCH_PAGE_LIMIT
    source_excerpt_pack = _draft_rendering_payloads.build_draft_source_excerpt_pack(
        approved_prepared_text,
        digest,
        merge_plan,
        force_excerpt=uses_draft_batches,
    )
    update_preservation_pack = _update_preservation.build_update_preservation_pack(merge_plan, snapshot)
    root_model_input_sidecars: list[Path] = []
    if not uses_draft_batches:
        source_excerpt_pack_path = step_root / "draft_source_excerpt_pack.json"
        source_excerpt_pack_md = step_root / "draft_source_excerpt_pack.md"
        write_json(source_excerpt_pack_path, source_excerpt_pack)
        source_excerpt_pack_md.write_text(_draft_rendering_payloads.render_draft_source_excerpt_pack_markdown(source_excerpt_pack), encoding="utf-8")
        update_preservation_pack_path = step_root / "update_preservation_pack.json"
        update_preservation_pack_md = step_root / "update_preservation_pack.md"
        write_json(update_preservation_pack_path, update_preservation_pack)
        update_preservation_pack_md.write_text(_update_preservation.render_update_preservation_pack_markdown(update_preservation_pack), encoding="utf-8")
        root_model_input_sidecars.extend(
            [source_excerpt_pack_path, source_excerpt_pack_md, update_preservation_pack_path, update_preservation_pack_md]
        )
    draft_run_context = _draft_rendering_runner.DraftRenderingRunContext(
        execution_context=ctx.execution_context,
        profile_payload=ctx.profile.model_dump(mode="json"),
        language_contract=ctx.manifest.vault_config_snapshot.model_dump(mode="json"),
        wiki_language=ctx.manifest.vault_config_snapshot.wiki_language,
    )
    draft_artifact = _draft_rendering_runner.run_draft_rendering_model(
        ctx=draft_run_context,
        step_root=step_root,
        digest=digest,
        merge_plan=merge_plan,
        snapshot=snapshot,
        source_excerpt_pack=source_excerpt_pack,
        update_preservation_pack=update_preservation_pack,
        approved_prepared_text=approved_prepared_text,
    )
    _draft_validation.validate_draft_rendering(draft_artifact, merge_plan, language=ctx.manifest.vault_config_snapshot.wiki_language)
    outputs = _draft_write_assembly.assemble_draft_write_outputs(
        vault=ctx.vault,
        run_dir=ctx.run_dir,
        step_root=step_root,
        profile=ctx.profile,
        operation_id=ctx.manifest.operation_id,
        digest=digest,
        merge_plan=merge_plan,
        snapshot=snapshot,
        draft_artifact=draft_artifact,
        approved_prepared_text=approved_prepared_text,
        raw_hash=sha256_file(ctx.vault / digest.source_raw_path),
        prepared_hash=sha256_file(approved_prepared_path),
        cleanup=read_model(require_step_output_dir(ctx.run_dir, "raw_link_cleanup") / "raw_link_cleanup.json", RawLinkCleanupArtifact),
        root_model_input_sidecars=root_model_input_sidecars,
    )
    refs = [_artifact_refs.draft_rendering_ref(ctx.run_dir, path, step_name) for path in outputs]
    refs.extend(_artifact_refs.structured_model_output_refs(ctx.run_dir, step_root, step_name))
    refs.extend(_artifact_refs.draft_rendering_model_batch_refs(ctx.run_dir, step_root, step_name))
    complete_step(ctx.manifest, step_name, outputs=refs)


def _run_validation(ctx: _step_runtime.StepRunContext) -> None:
    step_name = "validation"
    preparation = read_model(
        require_step_output_dir(ctx.run_dir, "raw_prepare") / "raw_preparation.json",
        RawPreparationArtifact,
    )
    digest = read_model(require_step_output_dir(ctx.run_dir, "source_digest_review") / "approved_digest.json", SourceDigestArtifact)
    resolution = read_model(
        require_step_output_dir(ctx.run_dir, "candidate_resolution") / "candidate_resolution.json",
        CandidateResolutionArtifact,
    )
    merge_plan = read_model(require_step_output_dir(ctx.run_dir, "merge_plan_review") / "approved_merge_plan.json", WikiMergePlanArtifact)
    snapshot = read_model(require_step_output_dir(ctx.run_dir, "wiki_context_snapshot") / "wiki_context_snapshot.json", WikiContextSnapshot)
    draft_write_manifest = read_model(require_step_output_dir(ctx.run_dir, "draft_review") / "approved_write_manifest.json", DraftWriteManifest)
    validate_raw_preparation(preparation)
    validate_source_digest(digest, language=ctx.manifest.vault_config_snapshot.wiki_language)
    validate_wiki_merge_plan(digest, merge_plan, resolution, snapshot, language=ctx.manifest.vault_config_snapshot.wiki_language)
    _step_runtime.ensure_wiki_context_current(ctx.vault, snapshot)
    if any(item.action == "needs_human_decision" for item in merge_plan.items):
        raise _errors.PipelineError("needs_human_decision must be revised to create/update/noop before validation.")
    if not digest.ingest_candidates() and not any(
        nonempty_prepared_discovered_candidates(item.source_basis) for item in merge_plan.items
    ):
        raise _errors.PipelineError("source_digest must include at least one wiki candidate")
    if not merge_plan.items:
        raise _errors.PipelineError("wiki_merge_plan must include at least one action")
    if not draft_write_manifest.targets:
        raise _errors.PipelineError("draft_write_manifest must include at least one target")
    complete_step(ctx.manifest, step_name)


def _run_apply_preview(ctx: _step_runtime.StepRunContext) -> None:
    step_name = "apply_preview"
    snapshot = read_model(require_step_output_dir(ctx.run_dir, "wiki_context_snapshot") / "wiki_context_snapshot.json", WikiContextSnapshot)
    _step_runtime.ensure_wiki_context_current(ctx.vault, snapshot)
    preview = _apply_preview.build_apply_preview(ctx.vault, ctx.run_dir)
    out = require_step_output_dir(ctx.run_dir, step_name) / "apply_preview.json"
    write_json(out, preview)
    complete_step(ctx.manifest, step_name, outputs=[_artifact_refs.ref(ctx.run_dir, out, step_name, "json", "apply_preview.v2")])


def _refresh_current_draft_grounding_artifacts(
    ctx: _step_runtime.StepRunContext,
    draft_manifest: DraftWriteManifest,
    draft_manifest_path: Path,
) -> DraftWriteManifest:
    draft_root = require_step_output_dir(ctx.run_dir, "draft_rendering")
    draft_artifact = read_model(draft_root / "draft_rendering.json", DraftRenderingArtifact)
    merge_plan = read_model(require_step_output_dir(ctx.run_dir, "merge_plan_review") / "approved_merge_plan.json", WikiMergePlanArtifact)
    snapshot = read_model(require_step_output_dir(ctx.run_dir, "wiki_context_snapshot") / "wiki_context_snapshot.json", WikiContextSnapshot)
    approved_prepared_text = (require_step_output_dir(ctx.run_dir, "prepared_raw_review") / "approved_prepared.md").read_text(encoding="utf-8")
    grounding_review = _draft_grounding.build_draft_grounding_review(draft_artifact, merge_plan, snapshot, approved_prepared_text)
    grounding_review_path, grounding_review_md = _draft_grounding.write_draft_grounding_review_outputs(draft_root, grounding_review)
    _refresh_draft_rendering_artifact_refs(ctx, grounding_review_path, grounding_review_md)
    if draft_manifest.requires_grounding_review == grounding_review.requires_review:
        return draft_manifest
    updated_manifest = draft_manifest.model_copy(update={"requires_grounding_review": grounding_review.requires_review})
    write_json(draft_manifest_path, updated_manifest)
    _refresh_draft_rendering_artifact_refs(ctx, draft_manifest_path)
    return updated_manifest


def _refresh_draft_rendering_artifact_refs(ctx: _step_runtime.StepRunContext, *paths: Path) -> None:
    draft_step = get_step(ctx.manifest, "draft_rendering")
    for path in paths:
        ref = _artifact_refs.draft_rendering_ref(ctx.run_dir, path, "draft_rendering")
        draft_step.outputs = _artifact_refs.replace_artifact_ref(draft_step.outputs, ref)
        for attempt in draft_step.attempts:
            attempt.outputs = _artifact_refs.replace_artifact_ref(attempt.outputs, ref)


def _run_draft_review(ctx: _step_runtime.StepRunContext) -> None:
    step_name = "draft_review"
    step_root = require_step_output_dir(ctx.run_dir, step_name)
    draft_manifest_path = require_step_output_dir(ctx.run_dir, "draft_rendering") / "draft_write_manifest.json"
    draft_manifest = read_model(draft_manifest_path, DraftWriteManifest)
    draft_manifest = _refresh_current_draft_grounding_artifacts(ctx, draft_manifest, draft_manifest_path)
    approved_manifest_path = step_root / "approved_write_manifest.json"
    approval_path = step_root / "draft_approval.json"
    prompt_path = step_root / "review_prompt.md"
    prompt_path.write_text(_draft_reviewing.render_draft_review_prompt(ctx.run_dir, draft_manifest), encoding="utf-8")
    auto_approval_notes = None
    if draft_manifest.source_only_noop:
        auto_approval_notes = "全 noop operation：来源会被记录，但没有知识页变化。"
    elif not _draft_reviewing.draft_review_requires_manual(ctx.run_dir, draft_manifest):
        auto_approval_notes = (
            "纯 create operation，当前运行自动批准。"
            if not draft_manifest.has_updates
            else "update operation 未发现 grounding 或旧页保留观察风险；本地旧知识补强已写入审计报告，当前运行自动批准。"
        )
    if auto_approval_notes is not None:
        write_json(approved_manifest_path, draft_manifest)
        approval = _draft_reviewing.build_draft_approval(
            ctx.run_dir,
            approved_manifest_path,
            decision="approved",
            auto_approved=True,
            notes=auto_approval_notes,
        )
        write_json(approval_path, approval)
        _step_runtime.complete_review_step(
            ctx.manifest,
            step_name,
            outputs=[
                _artifact_refs.ref(ctx.run_dir, prompt_path, step_name, "markdown"),
                _artifact_refs.ref(ctx.run_dir, approved_manifest_path, step_name, "json", "draft_write_manifest.v1"),
                _artifact_refs.ref(ctx.run_dir, approval_path, step_name, "json", "draft_review.v2"),
            ],
            review_decision_ref=approval_path.relative_to(ctx.run_dir).as_posix(),
        )
        return
    pending_manifest = step_root / "pending_write_manifest.json"
    write_json(pending_manifest, draft_manifest)
    review_reason = _draft_reviewing.draft_review_reason(ctx.run_dir, draft_manifest)
    approval = _draft_reviewing.pending_draft_approval(review_reason)
    write_json(approval_path, approval)
    mark_step_awaiting_review(
        ctx.manifest,
        step_name,
        outputs=[
            _artifact_refs.ref(ctx.run_dir, prompt_path, step_name, "markdown"),
            _artifact_refs.ref(ctx.run_dir, pending_manifest, step_name, "json", "draft_write_manifest.v1"),
            _artifact_refs.ref(ctx.run_dir, approval_path, step_name, "json", "draft_review.v2"),
        ],
        reason=review_reason,
        review_decision_ref=approval_path.relative_to(ctx.run_dir).as_posix(),
    )


@dataclass(frozen=True)
class StepRunner:
    spec: StepSpec
    run: Any


_STEP_RUN_FUNCTIONS = {
    "raw_link_cleanup": _raw_steps.run_raw_link_cleanup,
    "raw_prepare": _raw_steps.run_raw_prepare,
    "prepared_raw_review": _raw_steps.run_prepared_raw_review,
    "source_digest": _run_source_digest,
    "source_digest_review": _run_source_digest_review,
    "source_duplicate_guard": _run_source_duplicate_guard,
    "candidate_resolution": _run_candidate_resolution,
    "wiki_context_snapshot": _run_wiki_context_snapshot,
    "wiki_merge_planning": _run_wiki_merge_planning,
    "merge_plan_review": _run_merge_plan_review,
    "draft_rendering": _run_draft_rendering,
    "draft_review": _run_draft_review,
    "validation": _run_validation,
    "apply_preview": _run_apply_preview,
}

STEP_RUNNERS: dict[str, StepRunner] = {
    spec.name: StepRunner(spec, _STEP_RUN_FUNCTIONS[spec.name]) for spec in STEP_SPECS
}


def _delete_downstream_step_dirs(vault: Path, operation_id: str, start_step: str) -> None:
    store = RunStore(vault)
    run_dir = store.run_dir(operation_id)
    for step in downstream_steps(start_step):
        output_dir = step_output_dir(run_dir, step)
        if output_dir is not None:
            shutil.rmtree(output_dir, ignore_errors=True)


def _validate_raw_link_cleanup_resume(*, run_dir: Path, manifest: OperationManifest, start: str) -> None:
    if start != "raw_link_cleanup":
        return
    step = get_step(manifest, "raw_link_cleanup")
    if step.status != StepStatus.completed:
        return
    artifact_path = require_step_output_dir(run_dir, "raw_link_cleanup") / "raw_link_cleanup.json"
    if not artifact_path.exists():
        return
    cleanup = read_model(artifact_path, RawLinkCleanupArtifact)
    if cleanup.changed:
        raise _errors.PipelineError("Cannot resume from raw_link_cleanup after it changed raw; rerun ingest or resume from raw_prepare.")


def _last_attempt_duration_ms(manifest: OperationManifest, step_name: str) -> int | None:
    step = get_step(manifest, step_name)
    if not step.attempts:
        return None
    return step.attempts[-1].duration_ms


def _step_completion_message(ctx: _step_runtime.StepRunContext, step_name: str) -> str | None:
    if step_name == "raw_prepare":
        if ctx.manifest.vault_config_snapshot.raw_prepare_policy == RawPreparePolicy.skip:
            return "raw_prepare 使用 --prepare skip passthrough，跳过模型清洗"
        return None
    if step_name == "wiki_merge_planning":
        path = require_step_output_dir(ctx.run_dir, "wiki_merge_planning") / "merge_planning_shortcut_report.json"
        if not path.exists():
            return None
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if report.get("used") is True:
            return f"wiki_merge_planning 使用 local shortcut: {report.get('shortcut', '')}"
        return None
    if step_name != "raw_link_cleanup":
        return None
    path = require_step_output_dir(ctx.run_dir, "raw_link_cleanup") / "raw_link_cleanup.json"
    if not path.exists():
        return None
    cleanup = read_model(path, RawLinkCleanupArtifact)
    if not cleanup.changed:
        return None
    return f"raw 已规范化: {cleanup.raw_path}，清理 {cleanup.cleaned_link_count} 个 Obsidian 文本链接"


def status(vault: Path, operation_id: str) -> OperationManifest:
    return read_manifest(RunStore(vault).manifest_path(operation_id))


def approve_review(vault: Path, operation_id: str, review_step: str) -> OperationManifest:
    store = RunStore(vault)
    with run_lock(vault, operation_id):
        manifest = read_manifest(store.manifest_path(operation_id))
        _require_review_step_awaiting(manifest, review_step)
        run_dir = store.run_dir(operation_id)
        _require_upstream_artifacts_current(vault, run_dir, manifest, review_step)
        if review_step == "draft_review":
            step_root = require_step_output_dir(run_dir, "draft_review")
            pending = step_root / "pending_write_manifest.json"
            if not pending.exists():
                raise _errors.PipelineError("draft_review has no pending write manifest to approve.")
            approved = step_root / "approved_write_manifest.json"
            approved.write_text(pending.read_text(encoding="utf-8"), encoding="utf-8")
            approval = _draft_reviewing.build_draft_approval(
                run_dir,
                approved,
                decision="approved",
                auto_approved=False,
                notes="人工批准草稿；pending_write_manifest.json 已由 approved_write_manifest.json 取代，pending artifact 保留作审计。",
            )
            approval_path = step_root / "draft_approval.json"
            write_json(approval_path, approval)
            mark_step_approved(
                manifest,
                review_step,
                outputs=[
                    _artifact_refs.ref(run_dir, step_root / "review_prompt.md", review_step, "markdown"),
                    _artifact_refs.ref(run_dir, step_root / "pending_write_manifest.json", review_step, "json", "draft_write_manifest.v1"),
                    _artifact_refs.ref(run_dir, approved, review_step, "json", "draft_write_manifest.v1"),
                    _artifact_refs.ref(run_dir, approval_path, review_step, "json", "draft_review.v2"),
                ],
            )
            _delete_downstream_step_dirs(vault, operation_id, "validation")
            mark_from_pending(manifest, "validation")
            write_manifest(store.manifest_path(operation_id), manifest)
            _run_metrics.refresh_run_metrics(run_dir, manifest)
            return manifest
        if review_step == "merge_plan_review":
            step_root = require_step_output_dir(run_dir, "merge_plan_review")
            pending = step_root / "pending_merge_plan.json"
            if not pending.exists():
                raise _errors.PipelineError("merge_plan_review has no pending merge plan to approve.")
            digest = read_model(require_step_output_dir(run_dir, "source_digest_review") / "approved_digest.json", SourceDigestArtifact)
            resolution = read_model(
                require_step_output_dir(run_dir, "candidate_resolution") / "candidate_resolution.json",
                CandidateResolutionArtifact,
            )
            snapshot_path = require_step_output_dir(run_dir, "wiki_context_snapshot") / "wiki_context_snapshot.json"
            snapshot = read_model(snapshot_path, WikiContextSnapshot)
            plan = read_model(pending, WikiMergePlanArtifact)
            plan = _merge_planning.finalize_wiki_merge_plan(plan, resolution, snapshot, snapshot_path.relative_to(run_dir).as_posix())
            validate_wiki_merge_plan(digest, plan, resolution, snapshot, language=manifest.vault_config_snapshot.wiki_language)
            if any(item.action == "needs_human_decision" for item in plan.items):
                raise _errors.PipelineError("needs_human_decision must be revised to create/update/noop before approval.")
            approved = step_root / "approved_merge_plan.json"
            write_json(approved, plan)
            decision = ReviewDecision(
                review_step=review_step,
                decision="approved",
                auto_approved=False,
                notes="人工批准合并计划；pending_merge_plan.json 已由 approved_merge_plan.json 取代，pending artifact 保留作审计。",
            )
            decision_path = step_root / "review_decision.json"
            write_json(decision_path, decision)
            mark_step_approved(
                manifest,
                review_step,
                outputs=[
                    _artifact_refs.ref(run_dir, step_root / "review_prompt.md", review_step, "markdown"),
                    _artifact_refs.ref(run_dir, pending, review_step, "json", "wiki_merge_plan.v5"),
                    _artifact_refs.ref(run_dir, approved, review_step, "json", "wiki_merge_plan.v5"),
                    _artifact_refs.ref(run_dir, decision_path, review_step, "json", "review_decision.v2"),
                ],
            )
            _delete_downstream_step_dirs(vault, operation_id, "draft_rendering")
            mark_from_pending(manifest, "draft_rendering")
            write_manifest(store.manifest_path(operation_id), manifest)
            _run_metrics.refresh_run_metrics(run_dir, manifest)
            return manifest
        raise _errors.PipelineError(f"Unsupported review step: {review_step}")


def revise_review(vault: Path, operation_id: str, review_step: str) -> OperationManifest:
    store = RunStore(vault)
    with run_lock(vault, operation_id):
        manifest = read_manifest(store.manifest_path(operation_id))
        _require_review_step_awaiting(manifest, review_step)
        run_dir = store.run_dir(operation_id)
        _require_upstream_artifacts_current(vault, run_dir, manifest, review_step)
        if review_step == "merge_plan_review":
            _delete_downstream_step_dirs(vault, operation_id, "wiki_merge_planning")
            mark_from_pending(manifest, "wiki_merge_planning")
        elif review_step == "draft_review":
            _delete_downstream_step_dirs(vault, operation_id, "draft_rendering")
            mark_from_pending(manifest, "draft_rendering")
        else:
            raise _errors.PipelineError(f"Unsupported review step: {review_step}")
        manifest.status = OperationStatus.running
        write_manifest(store.manifest_path(operation_id), manifest)
        _run_metrics.refresh_run_metrics(run_dir, manifest)
        return manifest


def _require_upstream_artifacts_current(vault: Path, run_dir: Path, manifest: OperationManifest, review_step: str) -> None:
    issues: list[str] = []
    for raw in manifest.raw_bindings:
        raw_path = vault / raw.relative_path
        if not raw_path.exists():
            issues.append(f"{raw.relative_path} is missing")
        elif sha256_file(raw_path) != raw.sha256:
            issues.append(f"{raw.relative_path} changed")
    for step in manifest.steps:
        if step.name == review_step:
            break
        for ref in step.outputs:
            if not ref.required_for_resume:
                continue
            path = run_dir / ref.relative_path
            if not path.exists():
                issues.append(f"{ref.relative_path} is missing")
            elif not path.is_file():
                issues.append(f"{ref.relative_path} is not a file")
            elif sha256_file(path) != ref.sha256:
                issues.append(f"{ref.relative_path} changed")
    if issues:
        preview = "; ".join(issues[:8])
        suffix = "" if len(issues) <= 8 else f"; ... and {len(issues) - 8} more"
        raise _errors.PipelineError(f"upstream required artifacts changed before review: {preview}{suffix}")


def _require_review_step_awaiting(manifest: OperationManifest, review_step: str) -> None:
    if manifest.status in {OperationStatus.applied, OperationStatus.source_recorded}:
        raise _errors.PipelineError("Applied operations are immutable. Start a new operation instead.")
    if manifest.status == OperationStatus.apply_failed:
        raise _errors.PipelineError("apply_failed operations cannot be reviewed; inspect written targets and rerun ingest.")
    step = get_step(manifest, review_step)
    if step.status != StepStatus.awaiting_review:
        raise _errors.PipelineError(f"{review_step} is not awaiting_review; current status is {step.status.value}.")


def latest_operation(vault: Path) -> str | None:
    root = RunStore(vault).runs_root
    if not root.exists():
        return None
    candidates = sorted([path for path in root.iterdir() if path.is_dir() and (path / "manifest.json").exists()])
    return candidates[-1].name if candidates else None


def _safe_timestamp() -> str:
    return utc_now().replace("+00:00", "Z").replace(":", "")
