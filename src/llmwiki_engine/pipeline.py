from __future__ import annotations

import json
import shutil
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

from rich.console import Console
from pydantic import BaseModel

from . import __version__
from . import apply_guards as _apply_guards
from . import apply_preview as _apply_preview
from . import diff_utils as _diff_utils
from . import draft_grounding as _draft_grounding
from . import draft_rendering_runner as _draft_rendering_runner
from . import draft_rendering_payloads as _draft_rendering_payloads
from . import draft_reviewing as _draft_reviewing
from . import draft_validation as _draft_validation
from . import draft_write_assembly as _draft_write_assembly
from . import errors as _errors
from . import markdown_utils as _markdown_utils
from . import merge_plan_refinement as _merge_plan_refinement
from . import merge_reporting as _merge_reporting
from . import planning_payloads as _planning_payloads
from . import related_pages as _related_pages
from . import source_digest_budget as _source_digest_budget
from . import source_digest_payload as _source_digest_payload
from . import source_digest_rendering as _source_digest_rendering
from . import source_refs as _source_refs
from . import update_preservation as _update_preservation
from . import wiki_markup as _wiki_markup
from .events import EventLogger
from .hash_utils import artifact_ref, sha256_bytes, sha256_file
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
    ArtifactRef,
    CandidateResolutionArtifact,
    CandidateResolutionItem,
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
    RelatedPageRef,
    SourceBasis,
    SourceDigestArtifact,
    SourceDigestCandidate,
    StepStatus,
    StructuredIssue,
    CandidateContextsArtifact,
    ContextOverlapSignal,
    EmbeddingRetrievalConfig,
    WikiContextEntry,
    WikiContextSnapshot,
    WikiMergePlanArtifact,
    WikiMergePlanItem,
    WikiPageMetadata,
    utc_now,
)
from .provider_config import ProviderExecutionContext, build_provider_execution_context
from .profiles import load_profile, page_output_path, profile_to_yaml_data, safe_filename
from .raw_cleanup import cleanup_raw_wikilinks, render_raw_link_cleanup_markdown
from .rendering import source_title_for_raw
from .retrieval import (
    RetrievalError,
    build_candidate_contexts,
    build_knowledge_pool,
    candidate_pool_sha256,
    metadata_from_text,
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
from .structured import StructuredModelCall
from .system_pages import (
    ensure_system_pages,
    format_markdown_table,
    local_date,
)
from .validators import (
    ValidationError as ContractValidationError,
    create_reason_needs_repair,
    nonempty_prepared_discovered_candidates,
    validate_candidate_resolution,
    validate_raw_preparation,
    validate_source_digest,
    validate_wiki_merge_plan,
)
from .verify import require_verified
from .vault_config import read_vault_config, write_default_vault_config
from .wiki_context import snapshot_entry, wiki_context_drift_messages
from .workspace import RunStore, apply_lock, ensure_workspace_layout, relative_to_vault, resolve_raw_path, run_lock


RAW_PREPARE_CONTRACT = {
    "goal": "Create a higher-quality canonical prepared raw for downstream knowledge compilation.",
    "rules": [
        "Do not add facts that are not supported by the original raw.",
        "Remove or relocate non-content noise such as navigation fragments, boilerplate, self-promotion, and obvious formatting artifacts.",
        "Correct obvious wording or formatting errors only when the surrounding context makes the correction clear.",
        "Return prepared_markdown as clean Markdown suitable for source_digest and downstream knowledge digestion.",
    ],
}

MODEL_RELATED_SUGGESTION_LIMIT = 2


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
    operation_id = f"ING-{safe_timestamp()}-{slug or raw_path.stem}"
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
            return execute_ingest(
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
        start, reset_from_step = default_resume_start(vault, store.run_dir(operation_id), manifest, from_step)
        if start is None:
            return manifest
        if raw_prepare_policy is not None:
            if step_index(start) > step_index("raw_prepare"):
                raise _errors.PipelineError("raw prepare override only applies when raw_prepare will rerun; resume from raw_prepare or earlier.")
            manifest.vault_config_snapshot.raw_prepare_policy = raw_prepare_policy
        validate_raw_link_cleanup_resume(run_dir=store.run_dir(operation_id), manifest=manifest, start=start)
        validate_resume_start(manifest, start)
        ensure_wiki_context_current_before_resume(vault, store.run_dir(operation_id), start)
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
            delete_downstream_step_dirs(vault, operation_id, reset_from_step)
            mark_from_pending(manifest, reset_from_step)
            write_manifest(store.manifest_path(operation_id), manifest)
        else:
            write_manifest(store.manifest_path(operation_id), manifest)
        return execute_ingest(
            vault,
            operation_id,
            start_step=start,
            execution_context=provider_execution_context,
            console=console,
        )


def default_resume_start(
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
    if manifest.status == OperationStatus.drafted and drafted_wiki_context_drifted(vault, run_dir):
        return "wiki_context_snapshot", "wiki_context_snapshot"
    return None, None


def drafted_wiki_context_drifted(vault: Path, run_dir: Path) -> bool:
    snapshot_path = require_step_output_dir(run_dir, "wiki_context_snapshot") / "wiki_context_snapshot.json"
    if not snapshot_path.exists():
        return False
    snapshot = read_model(snapshot_path, WikiContextSnapshot)
    return bool(wiki_context_drift_messages(vault, snapshot))


def ensure_wiki_context_current_before_resume(vault: Path, run_dir: Path, start_step: str) -> None:
    if step_index(start_step) <= step_index("wiki_context_snapshot"):
        return
    snapshot_path = require_step_output_dir(run_dir, "wiki_context_snapshot") / "wiki_context_snapshot.json"
    if not snapshot_path.exists():
        return
    snapshot = read_model(snapshot_path, WikiContextSnapshot)
    messages = wiki_context_drift_messages(vault, snapshot)
    if messages:
        raise _errors.PipelineError("; ".join(messages))


def execute_ingest(
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
    validate_resume_start(manifest, start_step)
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
            logger.emit(step_name, "failed", status="failed", message=message, duration_ms=last_attempt_duration_ms(manifest, step_name))
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


def validate_resume_start(manifest: OperationManifest, start_step: str) -> None:
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
    ctx = StepRunContext(
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
        message=step_completion_message(ctx, step_name),
        duration_ms=last_attempt_duration_ms(manifest, step_name),
    )


@dataclass(frozen=True)
class StepRunContext:
    vault: Path
    run_dir: Path
    raw_path: Path
    profile: Any
    manifest: OperationManifest
    execution_context: ProviderExecutionContext


def _run_raw_link_cleanup(ctx: StepRunContext) -> None:
    step_name = "raw_link_cleanup"
    step_root = require_step_output_dir(ctx.run_dir, step_name)
    raw_rel = relative_to_vault(ctx.vault, ctx.raw_path)
    original_text = ctx.raw_path.read_text(encoding="utf-8")
    pre_hash = sha256_file(ctx.raw_path)
    cleaned_text, links, warnings, preserved_media_count = cleanup_raw_wikilinks(original_text)
    changed = cleaned_text != original_text
    if changed:
        if sha256_file(ctx.raw_path) != pre_hash:
            raise _errors.PipelineError("raw changed during raw_link_cleanup; rerun ingest")
        tmp = ctx.raw_path.with_name(f".{ctx.raw_path.name}.tmp")
        tmp.write_text(cleaned_text, encoding="utf-8")
        tmp.replace(ctx.raw_path)
    post_hash, post_size = raw_ref(ctx.raw_path)
    ctx.manifest.raw_bindings = [RawBinding(relative_path=raw_rel, sha256=post_hash, size_bytes=post_size)]
    artifact = RawLinkCleanupArtifact(
        raw_path=raw_rel,
        changed=changed,
        pre_cleanup_sha256=pre_hash,
        post_cleanup_sha256=post_hash,
        cleaned_link_count=len(links),
        preserved_media_embed_count=preserved_media_count,
        links=links,
        warnings=warnings,
    )
    out = step_root / "raw_link_cleanup.json"
    write_json(out, artifact)
    report = step_root / "raw_link_cleanup.md"
    report.write_text(render_raw_link_cleanup_markdown(artifact), encoding="utf-8")
    diff_path = step_root / "cleanup.diff"
    diff_path.write_text(_diff_utils.render_update_diff(original_text, cleaned_text, f"pre/{raw_rel}", f"post/{raw_rel}"), encoding="utf-8")
    complete_step(
        ctx.manifest,
        step_name,
        outputs=[
            _ref(ctx.run_dir, out, step_name, "json", "raw_link_cleanup.v1"),
            _ref(ctx.run_dir, report, step_name, "markdown"),
            _ref(ctx.run_dir, diff_path, step_name, "diff"),
        ],
    )


def build_raw_prepare_skip_passthrough(
    *,
    raw_path: Path,
    raw_rel: str,
    input_raw_sha256: str,
    cleanup_ref: str,
) -> RawPreparationArtifact:
    if raw_path.suffix.lower() not in {".md", ".markdown", ".mdown"}:
        raise _errors.PipelineError("--prepare skip requires Markdown raw; use --prepare auto or --prepare force for non-Markdown raw.")
    raw_text = raw_path.read_text(encoding="utf-8")
    if not raw_text.strip():
        raise _errors.PipelineError("--prepare skip requires non-empty raw Markdown.")
    return RawPreparationArtifact(
        source_raw_path=raw_rel,
        input_raw_sha256=input_raw_sha256,
        raw_link_cleanup_ref=cleanup_ref,
        prepared_markdown=raw_text.rstrip() + "\n",
        operations_applied=["user_skip_markdown_passthrough"],
        omission_policy="none",
    )


def _write_raw_prepare_outputs(ctx: StepRunContext, preparation: RawPreparationArtifact, *, include_model_outputs: bool) -> None:
    step_name = "raw_prepare"
    step_root = require_step_output_dir(ctx.run_dir, step_name)
    validate_raw_preparation(preparation)
    out = step_root / "raw_preparation.json"
    write_json(out, preparation)
    prepared = step_root / "prepared.md"
    prepared.parent.mkdir(parents=True, exist_ok=True)
    prepared.write_text(preparation.prepared_markdown.rstrip() + "\n", encoding="utf-8")
    outputs = [
        _ref(ctx.run_dir, out, step_name, "json", "raw_preparation.v1"),
        _ref(ctx.run_dir, prepared, step_name, "markdown"),
    ]
    if include_model_outputs:
        outputs.extend(structured_model_output_refs(ctx.run_dir, step_root, step_name))
    complete_step(ctx.manifest, step_name, outputs=outputs)


def _run_raw_prepare(ctx: StepRunContext) -> None:
    step_name = "raw_prepare"
    raw_rel = relative_to_vault(ctx.vault, ctx.raw_path)
    cleanup_path = require_step_output_dir(ctx.run_dir, "raw_link_cleanup") / "raw_link_cleanup.json"
    cleanup = read_model(cleanup_path, RawLinkCleanupArtifact)
    input_raw_sha256 = sha256_file(ctx.raw_path)
    cleanup_ref = cleanup_path.relative_to(ctx.run_dir).as_posix()
    raw_prepare_policy = ctx.manifest.vault_config_snapshot.raw_prepare_policy
    if raw_prepare_policy == RawPreparePolicy.skip:
        preparation = build_raw_prepare_skip_passthrough(
            raw_path=ctx.raw_path,
            raw_rel=raw_rel,
            input_raw_sha256=input_raw_sha256,
            cleanup_ref=cleanup_ref,
        )
        _write_raw_prepare_outputs(ctx, preparation, include_model_outputs=False)
        return

    payload = {
        "source_raw_path": raw_rel,
        "source_raw_sha256": input_raw_sha256,
        "raw_prepare_policy": raw_prepare_policy.value,
        "raw_markdown": ctx.raw_path.read_text(encoding="utf-8"),
        "raw_link_cleanup_ref": cleanup_ref,
        "raw_link_cleanup": {
            "changed": cleanup.changed,
            "cleaned_link_count": cleanup.cleaned_link_count,
            "preserved_media_embed_count": cleanup.preserved_media_embed_count,
            "cleanup_rule_version": cleanup.cleanup_rule_version,
        },
        "contract": RAW_PREPARE_CONTRACT,
    }

    def validate_raw_prepare_model(model: RawPreparationArtifact) -> None:
        candidate = model.model_copy(
            update={
                "input_raw_sha256": input_raw_sha256,
                "raw_link_cleanup_ref": cleanup_ref,
            }
        )
        validate_raw_preparation(candidate)
        if candidate.source_raw_path != raw_rel:
            raise ContractValidationError(
                f"raw_prepare source path mismatch: {candidate.source_raw_path} != {raw_rel}",
                issues=[
                    StructuredIssue(
                        issue_code="source_path_mismatch",
                        field_path="source_raw_path",
                        validator_id="validate_raw_prepare_model",
                        message=f"raw_prepare source path mismatch: {candidate.source_raw_path} != {raw_rel}",
                        repairability="repairable",
                    )
                ],
            )

    preparation, _ = _structured_call(ctx.run_dir, ctx.execution_context, step_name).run(
        step_name,
        payload,
        RawPreparationArtifact,
        validator=validate_raw_prepare_model,
    )
    preparation = _redacted_model(ctx, preparation, RawPreparationArtifact)
    preparation = preparation.model_copy(
        update={
            "input_raw_sha256": input_raw_sha256,
            "raw_link_cleanup_ref": cleanup_ref,
        }
    )
    if preparation.source_raw_path != raw_rel:
        raise _errors.PipelineError(f"raw_prepare source path mismatch: {preparation.source_raw_path} != {raw_rel}")
    _write_raw_prepare_outputs(ctx, preparation, include_model_outputs=True)


def _run_prepared_raw_review(ctx: StepRunContext) -> None:
    step_name = "prepared_raw_review"
    step_root = require_step_output_dir(ctx.run_dir, step_name)
    prepared = require_step_output_dir(ctx.run_dir, "raw_prepare") / "prepared.md"
    approved = step_root / "approved_prepared.md"
    approved.write_text(prepared.read_text(encoding="utf-8"), encoding="utf-8")
    prompt = step_root / "review_prompt.md"
    prompt.write_text(
        "# Prepared Raw 审核\n\n"
        "当前运行自动批准 prepared raw；下游步骤会继续基于 Approved Raw 校验。\n",
        encoding="utf-8",
    )
    feedback = step_root / "review_feedback.jsonl"
    feedback.write_text("", encoding="utf-8")
    decision = ReviewDecision(
        review_step=step_name,
        decision="approved",
        auto_approved=True,
        notes="当前运行自动批准；交互式审核尚未接入。",
    )
    decision_path = step_root / "review_decision.json"
    write_json(decision_path, decision)
    complete_review_step(
        ctx.manifest,
        step_name,
        outputs=[
            _ref(ctx.run_dir, prompt, step_name, "markdown"),
            _ref(ctx.run_dir, feedback, step_name, "jsonl"),
            _ref(ctx.run_dir, decision_path, step_name, "json", "review_decision.v2"),
            _ref(ctx.run_dir, approved, step_name, "markdown"),
        ],
        review_decision_ref=decision_path.relative_to(ctx.run_dir).as_posix(),
    )


def _run_source_digest(ctx: StepRunContext) -> None:
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
    digest, _ = _structured_call(ctx.run_dir, ctx.execution_context, step_name).run(
        step_name,
        payload,
        SourceDigestArtifact,
        validator=lambda model: validate_source_digest(model, language=ctx.manifest.vault_config_snapshot.wiki_language),
    )
    digest = _redacted_model(ctx, digest, SourceDigestArtifact)
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
        _ref(ctx.run_dir, source_map_path, step_name, "json", "source_digest_source_map.v1"),
        _ref(ctx.run_dir, source_map_md, step_name, "markdown"),
        _ref(ctx.run_dir, source_map_payload_path, step_name, "json", "source_digest_source_map_payload.v1"),
        _ref(ctx.run_dir, source_kind_hints_path, step_name, "json", "source_kind_hints.v1"),
        _ref(ctx.run_dir, source_kind_hints_md, step_name, "markdown"),
        _ref(ctx.run_dir, out, step_name, "json", "source_digest.v2"),
        _ref(ctx.run_dir, digest_md, step_name, "markdown"),
        _ref(ctx.run_dir, budget_report_path, step_name, "json", "source_digest_budget_report.v1"),
        _ref(ctx.run_dir, budget_report_md, step_name, "markdown"),
    ]
    outputs.extend(structured_model_output_refs(ctx.run_dir, step_root, step_name))
    complete_step(ctx.manifest, step_name, outputs=outputs)


def _run_source_digest_review(ctx: StepRunContext) -> None:
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
    complete_review_step(
        ctx.manifest,
        step_name,
        outputs=[
            _ref(ctx.run_dir, prompt, step_name, "markdown"),
            _ref(ctx.run_dir, feedback, step_name, "jsonl"),
            _ref(ctx.run_dir, decision_path, step_name, "json", "review_decision.v2"),
            _ref(ctx.run_dir, approved_json, step_name, "json", "source_digest.v2"),
            _ref(ctx.run_dir, approved_md, step_name, "markdown"),
        ],
        review_decision_ref=decision_path.relative_to(ctx.run_dir).as_posix(),
    )


def _run_source_duplicate_guard(ctx: StepRunContext) -> None:
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
            _ref(ctx.run_dir, out, step_name, "json", "source_duplicate_guard.v1"),
            _ref(ctx.run_dir, md, step_name, "markdown"),
        ],
    )


def _run_candidate_resolution(ctx: StepRunContext) -> None:
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
        candidate = finalize_candidate_resolution(ctx.vault, ctx.profile, model, digest)
        validate_candidate_resolution(digest, candidate)

    artifact, _ = _structured_call(ctx.run_dir, ctx.execution_context, step_name).run(
        step_name,
        payload,
        CandidateResolutionArtifact,
        validator=validate_candidate_resolution_model,
    )
    artifact = _redacted_model(ctx, artifact, CandidateResolutionArtifact)
    artifact = backfill_missing_candidate_resolution_items(artifact, digest, ctx.profile)
    artifact = finalize_candidate_resolution(ctx.vault, ctx.profile, artifact, digest)
    validate_candidate_resolution(digest, artifact)
    out = step_root / "candidate_resolution.json"
    write_json(out, artifact)
    table = step_root / "candidate_resolution.md"
    table.write_text(render_candidate_resolution_markdown(artifact), encoding="utf-8")
    outputs = [
        _ref(ctx.run_dir, source_pack_path, step_name, "json", "candidate_resolution_source_excerpt_pack.v1"),
        _ref(ctx.run_dir, source_pack_md, step_name, "markdown"),
        _ref(ctx.run_dir, out, step_name, "json", "candidate_resolution.v3"),
        _ref(ctx.run_dir, table, step_name, "markdown"),
    ]
    outputs.extend(structured_model_output_refs(ctx.run_dir, step_root, step_name))
    complete_step(
        ctx.manifest,
        step_name,
        outputs=outputs,
    )


def _run_wiki_context_snapshot(ctx: StepRunContext) -> None:
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
    snapshot = build_wiki_context_snapshot(
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
            _ref(ctx.run_dir, snapshot_path, step_name, "json", "wiki_context_snapshot.v3"),
            _ref(ctx.run_dir, contexts_path, step_name, "json", "candidate_contexts.v2"),
            _ref(ctx.run_dir, contexts_md, step_name, "markdown"),
        ],
    )


def block_unrepaired_medium_create_reason(plan: WikiMergePlanArtifact) -> WikiMergePlanArtifact:
    items: list[WikiMergePlanItem] = []
    for item in plan.items:
        if (
            item.action == "create"
            and item.strongest_overlap.strength == "medium"
            and item.strongest_overlap.path
            and create_reason_needs_repair(item.why_not_update)
        ):
            items.append(
                item.model_copy(
                    update={
                        "action": "needs_human_decision",
                        "apply_eligibility": "blocked",
                        "blocked_reason": item.blocked_reason
                        or f"召回到中等相关旧页 `{item.strongest_overlap.path}`，但模型选择 create 的理由不充分；需要人工确认。",
                        "finalization_reason": _markdown_utils.merge_markdown_blocks(
                            item.finalization_reason,
                            "模型 repair 后 why_not_update 仍不充分，已转为 needs_human_decision。",
                        ),
                    }
                )
            )
            continue
        items.append(item)
    return plan.model_copy(update={"items": items})


def empty_vault_create_merge_planning_shortcut_report(
    digest: SourceDigestArtifact,
    resolution: CandidateResolutionArtifact,
    snapshot: WikiContextSnapshot,
    candidate_contexts: CandidateContextsArtifact,
) -> dict[str, Any]:
    known_candidate_ids = {candidate.candidate_id for candidate in [*digest.ingest_candidates(), *digest.budget_deferred_candidates]}
    target_paths = [item.candidate_target_path for item in resolution.items if item.candidate_target_path]
    snapshot_by_path = {entry.path: entry for entry in snapshot.entries}
    present_targets = [
        target
        for target in target_paths
        if (entry := snapshot_by_path.get(f"wiki/{target}")) is not None and entry.expected_state == "present"
    ]
    missing_snapshot_targets = [target for target in target_paths if f"wiki/{target}" not in snapshot_by_path]
    total_hits = sum(len(item.hits) for item in candidate_contexts.items)
    source_page_plans = [item.page_plan_id for item in resolution.items if item.page_type.strip().lower() == "source"]
    empty_target_page_plans = [item.page_plan_id for item in resolution.items if not item.candidate_target_path]
    missing_source_candidate_page_plans = [
        item.page_plan_id
        for item in resolution.items
        if not _source_refs.source_basis_candidate_refs(item.source_basis)
    ]
    unknown_source_candidate_ids = sorted(
        {
            candidate_id
            for item in resolution.items
            for candidate_id in _source_refs.source_basis_candidate_refs(item.source_basis)
            if candidate_id not in known_candidate_ids
        }
    )
    duplicate_targets = sorted({target for target in target_paths if target_paths.count(target) > 1})
    blocking_conditions: list[str] = []
    if not resolution.items:
        blocking_conditions.append("candidate_resolution_empty")
    if snapshot.knowledge_metadata_pool:
        blocking_conditions.append("knowledge_metadata_pool_not_empty")
    if total_hits:
        blocking_conditions.append("candidate_context_hits_present")
    if present_targets:
        blocking_conditions.append("target_page_already_present")
    if missing_snapshot_targets:
        blocking_conditions.append("target_page_missing_from_snapshot")
    if source_page_plans:
        blocking_conditions.append("source_page_plan_present")
    if empty_target_page_plans:
        blocking_conditions.append("candidate_target_path_empty")
    if missing_source_candidate_page_plans:
        blocking_conditions.append("source_candidate_ids_empty")
    if unknown_source_candidate_ids:
        blocking_conditions.append("source_candidate_ids_unknown")
    if duplicate_targets:
        blocking_conditions.append("duplicate_candidate_target_path")
    return {
        "schema_version": "merge_planning_shortcut_report.v1",
        "shortcut": "empty_vault_all_create",
        "used": not blocking_conditions,
        "reason": (
            "空知识库、无候选召回命中、所有目标页均缺失；本地生成 create merge plan，跳过模型规划。"
            if not blocking_conditions
            else "未满足空库纯 create shortcut 条件，继续使用模型规划。"
        ),
        "blocking_conditions": blocking_conditions,
        "candidate_count": len(resolution.items),
        "knowledge_metadata_pool_count": len(snapshot.knowledge_metadata_pool),
        "candidate_context_item_count": len(candidate_contexts.items),
        "candidate_context_hit_count": total_hits,
        "present_target_count": len(present_targets),
        "missing_snapshot_target_count": len(missing_snapshot_targets),
        "source_page_plan_count": len(source_page_plans),
        "empty_target_page_plan_count": len(empty_target_page_plans),
        "missing_source_candidate_page_plan_count": len(missing_source_candidate_page_plans),
        "unknown_source_candidate_id_count": len(unknown_source_candidate_ids),
        "duplicate_candidate_target_count": len(duplicate_targets),
        "target_paths": target_paths,
        "present_targets": present_targets,
        "missing_snapshot_targets": missing_snapshot_targets,
        "source_page_plans": source_page_plans,
        "empty_target_page_plans": empty_target_page_plans,
        "missing_source_candidate_page_plans": missing_source_candidate_page_plans,
        "unknown_source_candidate_ids": unknown_source_candidate_ids,
        "duplicate_candidate_targets": duplicate_targets,
    }


def _run_wiki_merge_planning(ctx: StepRunContext) -> None:
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
        shortcut_report = empty_vault_create_merge_planning_shortcut_report(digest, resolution, snapshot, candidate_contexts)
    else:
        shortcut_report = {"used": False}
    if shortcut_report["used"]:
        shortcut_report_path = step_root / "merge_planning_shortcut_report.json"
        shortcut_report_md = step_root / "merge_planning_shortcut_report.md"
        write_json(shortcut_report_path, shortcut_report)
        shortcut_report_md.write_text(_merge_reporting.render_merge_planning_shortcut_report(shortcut_report), encoding="utf-8")
        plan = build_wiki_merge_plan(resolution, digest, snapshot, log_date=snapshot.log_date)
        plan = finalize_wiki_merge_plan(plan, resolution, snapshot, snapshot_ref, medium_missing_policy="preserve")
        plan = block_unrepaired_medium_create_reason(plan)
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
                _ref(ctx.run_dir, context_pack_path, step_name, "json", "merge_planning_context_pack.v1"),
                _ref(ctx.run_dir, context_pack_md, step_name, "markdown"),
                _ref(ctx.run_dir, shortcut_report_path, step_name, "json", "merge_planning_shortcut_report.v1"),
                _ref(ctx.run_dir, shortcut_report_md, step_name, "markdown"),
                _ref(ctx.run_dir, out, step_name, "json", "wiki_merge_plan.v5"),
                _ref(ctx.run_dir, table, step_name, "markdown"),
                _ref(ctx.run_dir, report, step_name, "markdown"),
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
        candidate = finalize_wiki_merge_plan(
            model,
            resolution,
            snapshot,
            snapshot_path.relative_to(ctx.run_dir).as_posix(),
            medium_missing_policy="preserve",
        )
        validate_wiki_merge_plan(digest, candidate, resolution, snapshot, language=ctx.manifest.vault_config_snapshot.wiki_language)

    plan, _ = _structured_call(ctx.run_dir, ctx.execution_context, step_name).run(
        step_name,
        payload,
        WikiMergePlanArtifact,
        validator=validate_merge_model,
    )
    plan = _redacted_model(ctx, plan, WikiMergePlanArtifact)
    plan = finalize_wiki_merge_plan(plan, resolution, snapshot, snapshot_ref, medium_missing_policy="preserve")
    plan = block_unrepaired_medium_create_reason(plan)
    validate_wiki_merge_plan(digest, plan, resolution, snapshot, language=ctx.manifest.vault_config_snapshot.wiki_language)
    out = step_root / "wiki_merge_plan.json"
    write_json(out, plan)
    table = step_root / "wiki_merge_plan.md"
    table.write_text(_merge_reporting.render_merge_plan_markdown(plan), encoding="utf-8")
    report = step_root / "merge_decision_report.md"
    report.write_text(_merge_reporting.render_merge_decision_report(plan, snapshot), encoding="utf-8")
    outputs = [
        _ref(ctx.run_dir, context_pack_path, step_name, "json", "merge_planning_context_pack.v1"),
        _ref(ctx.run_dir, context_pack_md, step_name, "markdown"),
        _ref(ctx.run_dir, out, step_name, "json", "wiki_merge_plan.v5"),
        _ref(ctx.run_dir, table, step_name, "markdown"),
        _ref(ctx.run_dir, report, step_name, "markdown"),
    ]
    outputs.extend(structured_model_output_refs(ctx.run_dir, step_root, step_name))
    complete_step(
        ctx.manifest,
        step_name,
        outputs=outputs,
    )


def _run_merge_plan_review(ctx: StepRunContext) -> None:
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
                _ref(ctx.run_dir, prompt_path, step_name, "markdown"),
                _ref(ctx.run_dir, feedback_path, step_name, "jsonl"),
                _ref(ctx.run_dir, pending_path, step_name, "json", "wiki_merge_plan.v5"),
                _ref(ctx.run_dir, decision_path, step_name, "json", "review_decision.v2"),
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
    complete_review_step(
        ctx.manifest,
        step_name,
        outputs=[
            _ref(ctx.run_dir, prompt_path, step_name, "markdown"),
            _ref(ctx.run_dir, feedback_path, step_name, "jsonl"),
            _ref(ctx.run_dir, decision_path, step_name, "json", "review_decision.v2"),
            _ref(ctx.run_dir, approved_path, step_name, "json", "wiki_merge_plan.v5"),
        ],
        review_decision_ref=decision_path.relative_to(ctx.run_dir).as_posix(),
    )


def _run_draft_rendering(ctx: StepRunContext) -> None:
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
    ensure_wiki_context_current(ctx.vault, snapshot)
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
    refs = [_draft_rendering_ref(ctx.run_dir, path, step_name) for path in outputs]
    refs.extend(structured_model_output_refs(ctx.run_dir, step_root, step_name))
    refs.extend(draft_rendering_model_batch_refs(ctx.run_dir, step_root, step_name))
    complete_step(ctx.manifest, step_name, outputs=refs)


def _run_validation(ctx: StepRunContext) -> None:
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
    ensure_wiki_context_current(ctx.vault, snapshot)
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


def _run_apply_preview(ctx: StepRunContext) -> None:
    step_name = "apply_preview"
    snapshot = read_model(require_step_output_dir(ctx.run_dir, "wiki_context_snapshot") / "wiki_context_snapshot.json", WikiContextSnapshot)
    ensure_wiki_context_current(ctx.vault, snapshot)
    preview = _apply_preview.build_apply_preview(ctx.vault, ctx.run_dir)
    out = require_step_output_dir(ctx.run_dir, step_name) / "apply_preview.json"
    write_json(out, preview)
    complete_step(ctx.manifest, step_name, outputs=[_ref(ctx.run_dir, out, step_name, "json", "apply_preview.v2")])


def refresh_current_draft_grounding_artifacts(
    ctx: StepRunContext,
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
    refresh_draft_rendering_artifact_refs(ctx, grounding_review_path, grounding_review_md)
    if draft_manifest.requires_grounding_review == grounding_review.requires_review:
        return draft_manifest
    updated_manifest = draft_manifest.model_copy(update={"requires_grounding_review": grounding_review.requires_review})
    write_json(draft_manifest_path, updated_manifest)
    refresh_draft_rendering_artifact_refs(ctx, draft_manifest_path)
    return updated_manifest


def refresh_draft_rendering_artifact_refs(ctx: StepRunContext, *paths: Path) -> None:
    draft_step = get_step(ctx.manifest, "draft_rendering")
    for path in paths:
        ref = _draft_rendering_ref(ctx.run_dir, path, "draft_rendering")
        draft_step.outputs = replace_artifact_ref(draft_step.outputs, ref)
        for attempt in draft_step.attempts:
            attempt.outputs = replace_artifact_ref(attempt.outputs, ref)


def replace_artifact_ref(refs: list[ArtifactRef], ref: ArtifactRef) -> list[ArtifactRef]:
    replaced = False
    next_refs: list[ArtifactRef] = []
    for existing in refs:
        if existing.relative_path == ref.relative_path:
            next_refs.append(ref)
            replaced = True
        else:
            next_refs.append(existing)
    if not replaced:
        next_refs.append(ref)
    return next_refs


def _run_draft_review(ctx: StepRunContext) -> None:
    step_name = "draft_review"
    step_root = require_step_output_dir(ctx.run_dir, step_name)
    draft_manifest_path = require_step_output_dir(ctx.run_dir, "draft_rendering") / "draft_write_manifest.json"
    draft_manifest = read_model(draft_manifest_path, DraftWriteManifest)
    draft_manifest = refresh_current_draft_grounding_artifacts(ctx, draft_manifest, draft_manifest_path)
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
        complete_review_step(
            ctx.manifest,
            step_name,
            outputs=[
                _ref(ctx.run_dir, prompt_path, step_name, "markdown"),
                _ref(ctx.run_dir, approved_manifest_path, step_name, "json", "draft_write_manifest.v1"),
                _ref(ctx.run_dir, approval_path, step_name, "json", "draft_review.v2"),
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
            _ref(ctx.run_dir, prompt_path, step_name, "markdown"),
            _ref(ctx.run_dir, pending_manifest, step_name, "json", "draft_write_manifest.v1"),
            _ref(ctx.run_dir, approval_path, step_name, "json", "draft_review.v2"),
        ],
        reason=review_reason,
        review_decision_ref=approval_path.relative_to(ctx.run_dir).as_posix(),
    )


@dataclass(frozen=True)
class StepRunner:
    spec: StepSpec
    run: Any


_STEP_RUN_FUNCTIONS = {
    "raw_link_cleanup": _run_raw_link_cleanup,
    "raw_prepare": _run_raw_prepare,
    "prepared_raw_review": _run_prepared_raw_review,
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


TModel = TypeVar("TModel", bound=BaseModel)


def _redacted_model(ctx: StepRunContext, model: TModel, model_type: type[TModel]) -> TModel:
    data = ctx.execution_context.redactor.redact(model.model_dump(mode="json"))
    return model_type.model_validate(data)


def build_wiki_merge_plan(
    resolution: CandidateResolutionArtifact,
    digest: SourceDigestArtifact,
    snapshot: WikiContextSnapshot,
    *,
    log_date: str,
) -> WikiMergePlanArtifact:
    snapshot_by_path = {entry.path: entry for entry in snapshot.entries}
    candidates = _source_refs.source_digest_candidate_lookup(digest)
    items: list[WikiMergePlanItem] = []
    for item in resolution.items:
        entry = snapshot_by_path.get(f"wiki/{item.candidate_target_path}")
        exists = entry is not None and entry.expected_state == "present"
        action = "update" if exists else "create"
        matched_page = None
        if action == "update":
            matched_page = item.candidate_target_path
        context_item = next((context for context in snapshot.candidate_contexts.items if context.page_plan_id == item.page_plan_id), None)
        inspected_paths = [hit.path for hit in context_item.hits] if context_item is not None else []
        strongest_hit = context_item.hits[0] if context_item is not None and context_item.hits else None
        strongest_overlap = (
            ContextOverlapSignal(
                strength=strongest_hit.strength,
                match_basis=strongest_hit.match_basis,
                path=strongest_hit.path,
                score=strongest_hit.score,
                reason=f"Top inspected context: {strongest_hit.display_title}",
            )
            if strongest_hit is not None
            else ContextOverlapSignal()
        )
        candidate = _source_refs.first_source_basis_candidate(item.source_basis, candidates)
        if candidate is None:
            refs = ", ".join(_source_refs.source_basis_candidate_refs(item.source_basis)) or "none"
            raise _errors.PipelineError(
                f"Cannot build deterministic merge plan for `{item.display_title or item.page_plan_id}`: "
                f"no source digest candidate found for refs: {refs}."
            )
        related_pages, related_unresolved = resolve_related_pages(item, candidate, resolution, snapshot)
        items.append(
            WikiMergePlanItem(
                page_plan_id=item.page_plan_id,
                source_basis=item.source_basis,
                action=action,
                model_action=action,
                finalization_reason="Deterministic local merge plan.",
                canonical_target_path=item.candidate_target_path,
                display_title=item.display_title,
                page_type=item.page_type,
                matched_page=matched_page,
                inspected_context_paths=inspected_paths,
                strongest_overlap=strongest_overlap,
                why_not_update="" if action == "update" else "未发现需要合并的已存在目标页。",
                why_create_or_update=item.reason,
                prior_knowledge_state="已有页面。" if exists else "当前 wiki 没有相关知识页。",
                new_understanding=item.topic_summary,
                changed_view="",
                knowledge_delta=item.topic_summary,
                why_this_matters=item.why_this_page,
                reuse_scenarios=[],
                value_points=[],
                section_plans={"summary": item.topic_summary, "detail": item.why_this_page},
                related_pages=related_pages,
                related_unresolved=related_unresolved,
                unresolved_related=related_unresolved,
                related_absence_reason=None if related_pages else ("no_candidate" if not inspected_paths else "low_confidence"),
                apply_eligibility="applyable",
                blocked_reason="",
                reason=item.reason,
            )
        )
    return WikiMergePlanArtifact(log_date=log_date, items=items, context_snapshot_ref="wiki_context_snapshot/wiki_context_snapshot.json")


def resolve_related_pages(
    item: CandidateResolutionItem,
    candidate: SourceDigestCandidate,
    resolution: CandidateResolutionArtifact,
    snapshot: WikiContextSnapshot,
) -> tuple[list[RelatedPageRef], list[str]]:
    if item.page_type.lower() == "source":
        return [], list(candidate.related_candidates)
    by_candidate_id = {
        candidate_id: other
        for other in resolution.items
        if other.page_type.lower() != "source"
        for candidate_id in _source_refs.source_basis_candidate_refs(other.source_basis)
    }
    by_current_title: dict[str, list[CandidateResolutionItem]] = {}
    for other in resolution.items:
        if other.page_type.lower() == "source":
            continue
        by_current_title.setdefault(normalize_related_key(other.display_title), []).append(other)
    metadata_lookup: dict[str, list[WikiPageMetadata]] = {}
    for pool_entry in snapshot.knowledge_metadata_pool:
        metadata = pool_entry.metadata
        if metadata is None:
            continue
        if metadata.llmwiki_type.lower() == "source":
            continue
        for key in [metadata.title, *metadata.aliases]:
            metadata_lookup.setdefault(normalize_related_key(key), []).append(metadata)
    related: list[RelatedPageRef] = []
    unresolved: list[str] = []
    seen_paths: set[str] = set()
    for raw in candidate.related_candidates:
        if len(related) >= _related_pages.FINAL_RELATED_LIMIT:
            unresolved.append(raw)
            continue
        resolved: RelatedPageRef | None = None
        other = by_candidate_id.get(raw)
        if other is None:
            matches = by_current_title.get(normalize_related_key(raw), [])
            if len(matches) == 1:
                other = matches[0]
            elif len(matches) > 1:
                unresolved.append(raw)
                continue
        if other is not None and other.candidate_target_path != item.candidate_target_path:
            resolved = RelatedPageRef(
                target_path=other.candidate_target_path,
                display_title=other.display_title,
                source="source_digest",
                reason=f"`{other.display_title}` 与本页同属本次材料中的互补主题，可帮助补足上下游理解。",
            )
        if resolved is None:
            metadata_matches = metadata_lookup.get(normalize_related_key(raw), [])
            if len(metadata_matches) > 1:
                unresolved.append(raw)
                continue
            metadata = metadata_matches[0] if metadata_matches else None
            if metadata is not None and metadata.path != item.candidate_target_path:
                resolved = RelatedPageRef(
                    target_path=metadata.path,
                    display_title=_wiki_markup.clean_display_title(metadata.title),
                    source="wiki_context",
                    reason=f"已有 wiki 页面标题或别名精确匹配 `{raw}`，可作为理解本页的相关背景。",
                )
        if resolved is None:
            unresolved.append(raw)
            continue
        if resolved.target_path in seen_paths:
            continue
        seen_paths.add(resolved.target_path)
        related.append(resolved)
    return related, unresolved


def normalize_related_key(value: str) -> str:
    text = value.strip().lower()
    for prefix in ["concept_", "entity_", "design_", "comparison_", "overview_", "event_", "memory_", "idea_", "open_question_"]:
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def _structured_call(
    run_dir: Path,
    execution_context: ProviderExecutionContext,
    task: str,
    *,
    result_filename: str = "provider_result.json",
) -> StructuredModelCall:
    return StructuredModelCall(
        execution_context.provider_for_task(task),
        output_dir=step_output_dir(run_dir, task),
        result_filename=result_filename,
        redactor=execution_context.redactor,
    )


def delete_downstream_step_dirs(vault: Path, operation_id: str, start_step: str) -> None:
    store = RunStore(vault)
    run_dir = store.run_dir(operation_id)
    for step in downstream_steps(start_step):
        output_dir = step_output_dir(run_dir, step)
        if output_dir is not None:
            shutil.rmtree(output_dir, ignore_errors=True)


def validate_raw_link_cleanup_resume(*, run_dir: Path, manifest: OperationManifest, start: str) -> None:
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


def last_attempt_duration_ms(manifest: OperationManifest, step_name: str) -> int | None:
    step = get_step(manifest, step_name)
    if not step.attempts:
        return None
    return step.attempts[-1].duration_ms


def step_completion_message(ctx: StepRunContext, step_name: str) -> str | None:
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
        require_upstream_artifacts_current(vault, run_dir, manifest, review_step)
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
                    _ref(run_dir, step_root / "review_prompt.md", review_step, "markdown"),
                    _ref(run_dir, step_root / "pending_write_manifest.json", review_step, "json", "draft_write_manifest.v1"),
                    _ref(run_dir, approved, review_step, "json", "draft_write_manifest.v1"),
                    _ref(run_dir, approval_path, review_step, "json", "draft_review.v2"),
                ],
            )
            delete_downstream_step_dirs(vault, operation_id, "validation")
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
            plan = finalize_wiki_merge_plan(plan, resolution, snapshot, snapshot_path.relative_to(run_dir).as_posix())
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
                    _ref(run_dir, step_root / "review_prompt.md", review_step, "markdown"),
                    _ref(run_dir, pending, review_step, "json", "wiki_merge_plan.v5"),
                    _ref(run_dir, approved, review_step, "json", "wiki_merge_plan.v5"),
                    _ref(run_dir, decision_path, review_step, "json", "review_decision.v2"),
                ],
            )
            delete_downstream_step_dirs(vault, operation_id, "draft_rendering")
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
        require_upstream_artifacts_current(vault, run_dir, manifest, review_step)
        if review_step == "merge_plan_review":
            delete_downstream_step_dirs(vault, operation_id, "wiki_merge_planning")
            mark_from_pending(manifest, "wiki_merge_planning")
        elif review_step == "draft_review":
            delete_downstream_step_dirs(vault, operation_id, "draft_rendering")
            mark_from_pending(manifest, "draft_rendering")
        else:
            raise _errors.PipelineError(f"Unsupported review step: {review_step}")
        manifest.status = OperationStatus.running
        write_manifest(store.manifest_path(operation_id), manifest)
        _run_metrics.refresh_run_metrics(run_dir, manifest)
        return manifest


def require_upstream_artifacts_current(vault: Path, run_dir: Path, manifest: OperationManifest, review_step: str) -> None:
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


def safe_timestamp() -> str:
    return utc_now().replace("+00:00", "Z").replace(":", "")


def _ref(
    run_dir: Path,
    path: Path,
    producer_step: str,
    kind: str,
    schema_version: str | None = None,
    required_for_resume: bool = True,
) -> ArtifactRef:
    return artifact_ref(
        base=run_dir,
        path=path,
        kind=kind,
        producer_step=producer_step,
        schema_version=schema_version,
        required_for_resume=required_for_resume,
    )


def structured_model_output_refs(run_dir: Path, step_root: Path, step_name: str) -> list[ArtifactRef]:
    refs: list[ArtifactRef] = []
    provider_result = step_root / "provider_result.json"
    if provider_result.exists():
        refs.append(_ref(run_dir, provider_result, step_name, "provider_result", "provider_result.v1"))
    report_json = step_root / "structured_repair_report.json"
    if report_json.exists():
        refs.append(_ref(run_dir, report_json, step_name, "json", "structured_repair_report.v1"))
    report_md = step_root / "structured_repair_report.md"
    if report_md.exists():
        refs.append(_ref(run_dir, report_md, step_name, "markdown"))
    provider_results = step_root / "provider_results"
    if provider_results.exists():
        for path in sorted(provider_results.glob("attempt-*.json")):
            refs.append(_ref(run_dir, path, step_name, "provider_result", "provider_result.v1", required_for_resume=False))
    repair_prompts = step_root / "repair_prompts"
    if repair_prompts.exists():
        for path in sorted(repair_prompts.glob("attempt-*.json")):
            refs.append(_ref(run_dir, path, step_name, "json", required_for_resume=False))
    return refs


def draft_rendering_model_batch_refs(run_dir: Path, step_root: Path, step_name: str) -> list[ArtifactRef]:
    batch_root = step_root / "model_batches"
    if not batch_root.exists():
        return []
    refs: list[ArtifactRef] = []
    for path in sorted(batch_root.rglob("*")):
        if not path.is_file():
            continue
        required = not any(part in {"provider_results", "repair_prompts"} for part in path.relative_to(step_root).parts)
        schema = None
        if path.name == "provider_result.json" or (path.parent.name == "provider_results" and path.name.startswith("attempt-")):
            schema = "provider_result.v1"
        elif path.name == "structured_repair_report.json":
            schema = "structured_repair_report.v1"
        elif path.name == "draft_source_excerpt_pack.json":
            schema = "draft_source_excerpt_pack.v1"
        elif path.name == "update_preservation_pack.json":
            schema = "update_preservation_pack.v1"
        elif path.name == "update_preservation_reinforcement_report.json":
            schema = "update_preservation_reinforcement_report.v1"
        elif path.name == "grounding_paraphrase_rewrite_report.json":
            schema = "grounding_paraphrase_rewrite_report.v1"
        refs.append(_ref(run_dir, path, step_name, artifact_kind_for_path(path), schema, required_for_resume=required))
    return refs


def artifact_kind_for_path(path: Path) -> str:
    return {
        ".md": "markdown",
        ".json": "json",
        ".jsonl": "jsonl",
        ".diff": "diff",
    }.get(path.suffix, "text")


def _draft_rendering_ref(run_dir: Path, path: Path, step_name: str) -> ArtifactRef:
    schemas = {
        "draft_rendering.json": "draft_rendering.v3",
        "draft_source_excerpt_pack.json": "draft_source_excerpt_pack.v1",
        "update_preservation_pack.json": "update_preservation_pack.v1",
        "update_preservation_reinforcement_report.json": "update_preservation_reinforcement_report.v1",
        "grounding_paraphrase_rewrite_report.json": "grounding_paraphrase_rewrite_report.v1",
        "draft_write_manifest.json": "draft_write_manifest.v1",
        "update_merge_report.json": "update_merge_report.v1",
        "related_merge_report.json": "related_merge_report.v1",
        "draft_grounding_review.json": "draft_grounding_review.v1",
        "index_open_questions_report.json": "index_open_questions_report.v1",
        "draft_rendering_batch_report.json": "draft_rendering_batch_report.v1",
    }
    return _ref(run_dir, path, step_name, artifact_kind_for_path(path), schemas.get(path.name))


def complete_review_step(
    manifest: OperationManifest,
    name: str,
    *,
    outputs: list[ArtifactRef],
    review_decision_ref: str,
) -> None:
    complete_step(manifest, name, outputs=outputs)
    step = get_step(manifest, name)
    step.review_state = "approved"
    step.review_reason = None
    step.awaiting_since = None
    step.resolved_at = step.completed_at
    step.review_decision_ref = review_decision_ref


def backfill_missing_candidate_resolution_items(
    artifact: CandidateResolutionArtifact,
    digest: SourceDigestArtifact,
    profile: Any,
) -> CandidateResolutionArtifact:
    covered_ids: set[str] = set()
    for item in artifact.items:
        covered_ids.update(item.source_basis.source_candidate_ids)
    additions: list[CandidateResolutionItem] = []
    notes = list(artifact.missed_candidate_risks)
    for group_name, candidates in [
        ("entities", digest.entities),
        ("concepts", digest.concepts),
        ("designs", digest.designs),
        ("comparisons", digest.comparisons),
        ("open_questions", digest.open_questions),
    ]:
        for candidate in candidates:
            if candidate.candidate_id in covered_ids:
                continue
            page_type = page_type_for_digest_candidate(group_name, candidate, profile)
            display_title = candidate.suggested_page_title.strip() or candidate.name.strip() or candidate.candidate_id
            additions.append(
                CandidateResolutionItem(
                    source_basis=SourceBasis(
                        source_candidate_ids=[candidate.candidate_id],
                        source_locator=candidate.source_locator,
                    ),
                    page_type=page_type,
                    display_title=display_title,
                    topic_summary=candidate.one_sentence_summary,
                    why_this_page=candidate.wiki_value or candidate.why_matters or "该候选来自 source_digest，模型在页面规划中遗漏，系统补齐为最小页面计划。",
                    initial_section_intent="系统补齐的最小页面计划；后续 merge planning / draft rendering 需要重新对照全文消化。",
                    coverage_notes=f"模型遗漏 approved_digest candidate `{candidate.candidate_id}`，系统已补齐。",
                    reason="candidate_resolution coverage backfill",
                )
            )
            notes.append(f"candidate_resolution model missed `{candidate.candidate_id}`; deterministic backfill added `{display_title}`.")
    if not additions:
        return artifact
    return CandidateResolutionArtifact(items=[*artifact.items, *additions], missed_candidate_risks=notes)


def page_type_for_digest_candidate(group_name: str, candidate: SourceDigestCandidate, profile: Any) -> str:
    normalized = candidate.type.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "question": "open_question",
        "open_questions": "open_question",
        "open_question": "open_question",
        "concept_overview": "overview",
        "design_overview": "overview",
        "entity_index": "overview",
        "open_question_overview": "open_question",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in profile.page_types and normalized != profile.source_page_type:
        return normalized
    preferred = {
        "entities": "entity",
        "concepts": "concept",
        "designs": "design",
        "comparisons": "comparison",
        "open_questions": "open_question",
    }.get(group_name)
    if preferred in profile.page_types:
        return preferred
    return profile.default_page_type


def finalize_candidate_resolution(
    vault: Path,
    profile: Any,
    artifact: CandidateResolutionArtifact,
    digest: SourceDigestArtifact | None = None,
) -> CandidateResolutionArtifact:
    items: list[CandidateResolutionItem] = []
    seen_paths: dict[str, int] = {}
    selected_candidate_ids = {candidate.candidate_id for candidate in digest.ingest_candidates()} if digest is not None else set()
    weak_or_noise_ids = {candidate.candidate_id for candidate in digest.weak_or_noise_items} if digest is not None else set()
    sanitized_items: list[CandidateResolutionItem] = []
    sanitation_notes = list(artifact.missed_candidate_risks)
    for index, item in enumerate(artifact.items):
        source_ids = list(item.source_basis.source_candidate_ids)
        leaked_ids = [
            candidate_id
            for candidate_id in source_ids
            if candidate_id in weak_or_noise_ids or candidate_id.strip().lower().startswith(("noise", "weak", "ignore"))
        ]
        if leaked_ids:
            cleaned_ids = [candidate_id for candidate_id in source_ids if candidate_id not in leaked_ids]
            if not cleaned_ids:
                sanitation_notes.append(
                    f"candidate_resolution item `{item.display_title or item.page_plan_id or index}` dropped because it only referenced weak/noise candidates: {sorted(leaked_ids)}."
                )
                continue
            item = item.model_copy(
                update={
                    "source_basis": item.source_basis.model_copy(update={"source_candidate_ids": cleaned_ids}),
                    "coverage_notes": _markdown_utils.merge_markdown_blocks(
                        item.coverage_notes,
                        f"系统清理 weak/noise candidate 引用：{', '.join(f'`{candidate_id}`' for candidate_id in leaked_ids)}。",
                    ),
                }
            )
            sanitation_notes.append(
                f"candidate_resolution item `{item.display_title or item.page_plan_id or index}` removed weak/noise candidate refs: {sorted(leaked_ids)}."
            )
        if digest is not None:
            unknown_source_ids = [
                candidate_id
                for candidate_id in item.source_basis.source_candidate_ids
                if candidate_id not in selected_candidate_ids
            ]
            if unknown_source_ids:
                prepared_discovered = list(item.source_basis.prepared_discovered_candidates)
                cleaned_ids = [
                    candidate_id
                    for candidate_id in item.source_basis.source_candidate_ids
                    if candidate_id in selected_candidate_ids
                ]
                if prepared_discovered:
                    for candidate_id in unknown_source_ids:
                        if candidate_id not in prepared_discovered:
                            prepared_discovered.append(candidate_id)
                    unknown_note = "系统将非 source_digest candidate id 移入 prepared_discovered_candidates："
                    sanitation_note = "moved unknown candidate refs to prepared_discovered_candidates"
                else:
                    prepared_discovered = list(unknown_source_ids)
                    unknown_note = "系统将非 source_digest candidate id 移入 prepared_discovered_candidates："
                    sanitation_note = "moved unknown-only candidate refs to prepared_discovered_candidates"
                item = item.model_copy(
                    update={
                        "source_basis": item.source_basis.model_copy(
                            update={
                                "source_candidate_ids": cleaned_ids,
                                "prepared_discovered_candidates": prepared_discovered,
                            }
                        ),
                        "coverage_notes": _markdown_utils.merge_markdown_blocks(
                            item.coverage_notes,
                            unknown_note + f"{', '.join(f'`{candidate_id}`' for candidate_id in unknown_source_ids)}。",
                        ),
                    }
                )
                sanitation_notes.append(
                    f"candidate_resolution item `{item.display_title or item.page_plan_id or index}` {sanitation_note}: {sorted(unknown_source_ids)}."
                )
        sanitized_items.append(item)
    artifact = artifact.model_copy(update={"items": sanitized_items, "missed_candidate_risks": sanitation_notes})
    issues: list[StructuredIssue] = []
    for index, item in enumerate(artifact.items):
        if item.page_type not in profile.page_types:
            issues.append(
                StructuredIssue(
                    issue_code="unknown_page_type",
                    field_path=f"items.{index}.page_type",
                    validator_id="finalize_candidate_resolution",
                    message=(
                        f"candidate_resolution uses unsupported page_type `{item.page_type}`; "
                        "remove weak/noise formal items or choose a valid profile page_type."
                    ),
                    repairability="repairable",
                )
            )
        if item.reason.strip().lower() in {"ignore", "ignored", "noise", "weak", "弱相关", "噪声"}:
            issues.append(
                StructuredIssue(
                    issue_code="ignore_as_formal_item",
                    field_path=f"items.{index}.reason",
                    validator_id="finalize_candidate_resolution",
                    message="formal page plans must not use ignore/noise as the reason; remove this item.",
                    repairability="repairable",
                )
            )
    if issues:
        raise ContractValidationError(
            "candidate_resolution contains weak/noise formal item(s); repair by deleting those formal items.",
            issues=issues,
        )
    for item in artifact.items:
        page_type = item.page_type
        display_title = _wiki_markup.clean_display_title(item.display_title) or item.display_title.strip()
        source_fingerprint = source_basis_fingerprint(item.source_basis)
        page_plan_id = stable_page_plan_id(page_type, display_title, source_fingerprint)
        stem = unicode_safe_stem(display_title)
        target = page_output_path(vault / "wiki", profile, page_type, stem)
        rel_target = target.relative_to(vault / "wiki").as_posix()
        if rel_target in seen_paths:
            seen_paths[rel_target] += 1
            path = Path(rel_target)
            suffix = sha256_bytes(f"{page_type}:{display_title}:{page_plan_id}".encode("utf-8"))[:8]
            rel_target = path.with_name(f"{path.stem}_{suffix}{path.suffix}").as_posix()
        else:
            seen_paths[rel_target] = 1
        items.append(
            item.model_copy(
                update={
                    "page_plan_id": page_plan_id,
                    "page_type": page_type,
                    "display_title": display_title,
                    "path_stem": stem,
                    "candidate_target_path": rel_target,
                }
            )
        )
    return CandidateResolutionArtifact(items=items, missed_candidate_risks=artifact.missed_candidate_risks)


def source_basis_fingerprint(source_basis: SourceBasis) -> str:
    payload = json.dumps(source_basis.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
    return sha256_bytes(payload.encode("utf-8"))[:12]


def stable_page_plan_id(page_type: str, display_title: str, source_fingerprint: str) -> str:
    base = f"{page_type}:{normalize_related_key(display_title)}:{source_fingerprint}"
    return f"PP-{sha256_bytes(base.encode('utf-8'))[:12]}"


def unicode_safe_stem(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value)
    normalized = re.sub(r"\s+", " ", normalized.strip())
    bad = '\\/:*?"<>|#^[]'
    cleaned = "".join("_" if char in bad else char for char in normalized).strip(" .")
    return cleaned or "untitled"


def render_candidate_resolution_markdown(artifact: CandidateResolutionArtifact) -> str:
    rows = [
        [
            item.page_plan_id,
            item.page_type,
            item.display_title,
            f"`{item.candidate_target_path}`",
            ", ".join(item.source_basis.source_candidate_ids),
            ", ".join(item.source_basis.prepared_discovered_candidates),
            item.why_this_page,
        ]
        for item in artifact.items
    ]
    return "# 候选页面规划\n\n" + format_markdown_table(
        ["页面计划", "类型", "标题", "目标", "来源候选", "Prepared 发现候选", "为什么写"],
        rows,
    ) + "\n"


def build_wiki_context_snapshot(
    vault: Path,
    resolution: CandidateResolutionArtifact,
    *,
    log_date: str,
    source_target_path: str,
    retrieval_config: EmbeddingRetrievalConfig | None = None,
    force_exact_backend: bool = False,
) -> WikiContextSnapshot:
    retrieval_config = retrieval_config or EmbeddingRetrievalConfig(backend="exact")
    knowledge_pool = build_knowledge_pool(vault)
    try:
        candidate_contexts = build_candidate_contexts(
            resolution=resolution,
            knowledge_pool=knowledge_pool,
            config=retrieval_config,
            vault=vault,
            force_exact_backend=force_exact_backend,
        )
    except RetrievalError as exc:
        raise _errors.PipelineError(str(exc)) from exc
    paths = {
        "wiki/index.md",
        "wiki/log.md",
        f"wiki/logs/{log_date}.md",
        f"wiki/{source_target_path}",
    }
    for item in resolution.items:
        paths.add(f"wiki/{item.candidate_target_path}")
    for context_item in candidate_contexts.items:
        for hit in context_item.hits:
            paths.add(f"wiki/{hit.path}")
    entries: list[WikiContextEntry] = []
    for rel in sorted(paths):
        path = vault / rel
        if path.exists():
            text = path.read_text(encoding="utf-8")
            entries.append(
                WikiContextEntry(
                    path=rel,
                    expected_state="present",
                    preimage_sha256=sha256_file(path),
                    content=text,
                    metadata=metadata_from_text(text, rel),
                )
            )
        else:
            entries.append(WikiContextEntry(path=rel, expected_state="missing", preimage_sha256=None, content=""))
    return WikiContextSnapshot(
        log_date=log_date,
        source_target_path=source_target_path,
        candidate_pool_sha256=candidate_pool_sha256(knowledge_pool),
        knowledge_metadata_pool=knowledge_pool,
        candidate_contexts=candidate_contexts,
        entries=entries,
    )


def resolve_model_related_pages(
    item: WikiMergePlanItem,
    resolution_item: CandidateResolutionItem,
    finalized_items: list[WikiMergePlanItem],
    resolution_by_id: dict[str, CandidateResolutionItem],
    snapshot: WikiContextSnapshot,
) -> tuple[list[RelatedPageRef], list[str]]:
    current_by_path: dict[str, WikiMergePlanItem] = {}
    current_by_title: dict[str, list[WikiMergePlanItem]] = {}
    for other in finalized_items:
        if other.page_type.lower() == "source" or other.action == "needs_human_decision":
            continue
        source_resolution = resolution_by_id.get(other.page_plan_id)
        paths = {other.canonical_target_path}
        if source_resolution is not None:
            paths.add(source_resolution.candidate_target_path)
        for path in paths:
            if path:
                current_by_path[path] = other
        current_by_title.setdefault(normalize_related_key(other.display_title), []).append(other)

    inspected_paths = set(item.inspected_context_paths)
    metadata_by_path: dict[str, WikiPageMetadata] = {}
    metadata_by_title: dict[str, list[WikiPageMetadata]] = {}
    for pool_entry in snapshot.knowledge_metadata_pool:
        metadata = pool_entry.metadata
        if metadata is None:
            continue
        if metadata.llmwiki_type.lower() == "source":
            continue
        metadata_by_path[metadata.path] = metadata
        for key in [metadata.title, *metadata.aliases]:
            metadata_by_title.setdefault(normalize_related_key(key), []).append(metadata)

    related: list[RelatedPageRef] = []
    unresolved: list[str] = []
    seen_paths: set[str] = set()
    for index, suggestion in enumerate(item.related_pages):
        if index >= MODEL_RELATED_SUGGESTION_LIMIT:
            unresolved.append(_related_debug_label(suggestion))
            continue
        if len(related) >= _related_pages.FINAL_RELATED_LIMIT:
            unresolved.append(_related_debug_label(suggestion))
            continue
        resolved = _resolve_single_model_related(
            suggestion,
            current_by_path=current_by_path,
            current_by_title=current_by_title,
            metadata_by_path=metadata_by_path,
            metadata_by_title=metadata_by_title,
            self_path=item.canonical_target_path,
            fallback_reason=suggestion.reason,
        )
        if resolved is None:
            unresolved.append(_related_debug_label(suggestion))
            continue
        if resolved.source == "wiki_context" and resolved.target_path not in inspected_paths:
            exact_key = normalize_related_key(resolved.display_title)
            if exact_key not in metadata_by_title:
                unresolved.append(_related_debug_label(suggestion))
                continue
        if resolved.target_path in seen_paths:
            continue
        seen_paths.add(resolved.target_path)
        related.append(resolved)
    return related, unresolved


def _resolve_single_model_related(
    suggestion: RelatedPageRef,
    *,
    current_by_path: dict[str, WikiMergePlanItem],
    current_by_title: dict[str, list[WikiMergePlanItem]],
    metadata_by_path: dict[str, WikiPageMetadata],
    metadata_by_title: dict[str, list[WikiPageMetadata]],
    self_path: str,
    fallback_reason: str,
) -> RelatedPageRef | None:
    for raw in _markdown_utils.dedupe_strings([suggestion.target_path, suggestion.display_title]):
        target_path = _related_pages.normalize_related_path(raw)
        if target_path:
            current = current_by_path.get(target_path)
            if current is not None and current.canonical_target_path != self_path:
                return RelatedPageRef(
                    target_path=current.canonical_target_path,
                    display_title=current.display_title,
                    source="source_digest",
                    reason=_related_pages.chinese_related_reason(fallback_reason, f"本次同源页面 `{current.display_title}` 与该主题互补。"),
                )
            metadata = metadata_by_path.get(target_path)
            if metadata is not None and metadata.path != self_path:
                return RelatedPageRef(
                    target_path=metadata.path,
                    display_title=_wiki_markup.clean_display_title(metadata.title),
                    source="wiki_context",
                    reason=_related_pages.chinese_related_reason(fallback_reason, f"召回旧页 `{_wiki_markup.clean_display_title(metadata.title)}` 与该主题存在可复用背景。"),
                )
        key = normalize_related_key(raw)
        current_matches = current_by_title.get(key, [])
        if len(current_matches) == 1 and current_matches[0].canonical_target_path != self_path:
            current = current_matches[0]
            return RelatedPageRef(
                target_path=current.canonical_target_path,
                display_title=current.display_title,
                source="source_digest",
                reason=_related_pages.chinese_related_reason(fallback_reason, f"本次同源页面 `{current.display_title}` 与该主题互补。"),
            )
        metadata_matches = metadata_by_title.get(key, [])
        if len(metadata_matches) == 1 and metadata_matches[0].path != self_path:
            metadata = metadata_matches[0]
            return RelatedPageRef(
                target_path=metadata.path,
                display_title=_wiki_markup.clean_display_title(metadata.title),
                source="wiki_context",
                reason=_related_pages.chinese_related_reason(fallback_reason, f"已有 wiki 页面标题或别名匹配 `{raw}`，可作为相关背景。"),
            )
    return None


def _related_debug_label(suggestion: RelatedPageRef) -> str:
    return f"{suggestion.display_title or '<untitled>'} -> {suggestion.target_path or '<no path>'}"


def finalize_wiki_merge_plan(
    plan: WikiMergePlanArtifact,
    resolution: CandidateResolutionArtifact,
    snapshot: WikiContextSnapshot,
    snapshot_ref: str,
    *,
    medium_missing_policy: Literal["preserve", "block"] = "block",
) -> WikiMergePlanArtifact:
    resolution_by_id = {item.page_plan_id: item for item in resolution.items}
    snapshot_paths = {entry.path for entry in snapshot.entries}
    context_by_id = {item.page_plan_id: item for item in snapshot.candidate_contexts.items}
    preliminary: list[WikiMergePlanItem] = []
    for item in plan.items:
        resolution_item = resolution_by_id.get(item.page_plan_id)
        if resolution_item is None:
            title_matches = [
                candidate
                for candidate in resolution.items
                if normalize_related_key(candidate.display_title) == normalize_related_key(item.display_title)
            ]
            typed_matches = [candidate for candidate in title_matches if candidate.page_type == item.page_type]
            resolution_item = (typed_matches or title_matches or [None])[0] if len(typed_matches or title_matches) == 1 else None
        if resolution_item is None:
            raise _errors.PipelineError(f"wiki_merge_plan references unknown page_plan_id: {item.page_plan_id}")
        context_item = context_by_id.get(resolution_item.page_plan_id)
        inspected_paths = [hit.path for hit in context_item.hits] if context_item is not None else []
        strongest_hit = context_item.hits[0] if context_item is not None and context_item.hits else None
        strongest_overlap = (
            ContextOverlapSignal(
                strength=strongest_hit.strength,
                match_basis=strongest_hit.match_basis,
                path=strongest_hit.path,
                score=strongest_hit.score,
                reason=f"Top inspected context: {strongest_hit.display_title}",
            )
            if strongest_hit is not None
            else ContextOverlapSignal()
        )
        canonical = _merge_plan_refinement.normalize_model_wiki_target_path(item.canonical_target_path or resolution_item.candidate_target_path)
        matched_page = _merge_plan_refinement.normalize_model_wiki_target_path(item.matched_page) if item.matched_page else None
        action = item.action
        model_action = item.model_action or item.action
        apply_eligibility = item.apply_eligibility
        blocked_reason = item.blocked_reason
        finalization_notes: list[str] = []
        if item.action == "update":
            matched_page = matched_page or canonical
            canonical = matched_page
        if item.action == "noop" and matched_page:
            canonical = matched_page
        if item.action == "create":
            canonical = _merge_plan_refinement.normalize_model_wiki_target_path(resolution_item.candidate_target_path)
            matched_page = None
        if f"wiki/{canonical}" not in snapshot_paths:
            raise _errors.PipelineError(f"wiki_merge_plan target is outside wiki_context_snapshot: wiki/{canonical}")
        entry = snapshot_entry(snapshot, f"wiki/{canonical}")
        if action == "needs_human_decision":
            pass
        elif entry.expected_state == "present":
            action = "update" if action != "noop" else "noop"
            matched_page = canonical if action in {"update", "noop"} else matched_page
            if action != model_action:
                finalization_notes.append("目标页已存在，最终动作改为 update/noop。")
        else:
            if action == "noop":
                action = "needs_human_decision"
                apply_eligibility = "blocked"
                blocked_reason = blocked_reason or "模型选择 noop，但没有绑定已召回的现有页面；需要人工确认。"
                finalization_notes.append("missing target noop 被转为 needs_human_decision。")
            else:
                action = "create"
                matched_page = None
                if action != model_action:
                    finalization_notes.append("目标页缺失，最终动作改为 create。")
        if (
            action == "create"
            and strongest_overlap.strength == "strong"
            and strongest_overlap.path
            and strongest_overlap.path != canonical
        ):
            action = "needs_human_decision"
            apply_eligibility = "blocked"
            blocked_reason = blocked_reason or (
                f"召回到强相关旧页 `{strongest_overlap.path}`，但模型仍选择 create；需要人工确认是否应 update。"
            )
            finalization_notes.append("strong overlap create 被转为 needs_human_decision。")
        if (
            action == "create"
            and strongest_overlap.strength == "medium"
            and strongest_overlap.path
            and create_reason_needs_repair(item.why_not_update)
        ):
            if medium_missing_policy == "preserve":
                synthesized_reason = _merge_plan_refinement.synthesize_medium_create_why_not_update(
                    item=item,
                    resolution_item=resolution_item,
                    strongest_hit=strongest_hit,
                    snapshot=snapshot,
                )
                item = item.model_copy(update={"why_not_update": synthesized_reason})
                finalization_notes.append(
                    f"medium overlap create 缺少 why_not_update，已{_merge_plan_refinement.LOCAL_MEDIUM_CREATE_REASON_MARKER}。"
                )
            elif medium_missing_policy == "block":
                action = "needs_human_decision"
                apply_eligibility = "blocked"
                blocked_reason = blocked_reason or (
                    f"召回到中等相关旧页 `{strongest_overlap.path}`，但模型选择 create 的理由不充分；需要人工确认。"
                )
                finalization_notes.append("medium overlap create 的 why_not_update 不充分，被转为 needs_human_decision。")
            else:
                finalization_notes.append("medium overlap create 的 why_not_update 不充分，已请求模型补充。")
        generic_old_title_reason = _merge_plan_refinement.medium_create_generic_old_title_review_reason(
            item.model_copy(
                update={
                    "action": action,
                    "canonical_target_path": canonical,
                    "strongest_overlap": strongest_overlap,
                }
            ),
            old_display_title=strongest_hit.display_title if strongest_hit is not None else "",
        )
        if action == "create" and generic_old_title_reason:
            action = "needs_human_decision"
            apply_eligibility = "blocked"
            blocked_reason = blocked_reason or generic_old_title_reason
            finalization_notes.append("medium overlap create 的旧页标题泛化风险，被转为 needs_human_decision。")
        if apply_eligibility == "blocked" and action != "needs_human_decision":
            action = "needs_human_decision"
            blocked_reason = blocked_reason or "模型将该项标记为 blocked，需要人工决策。"
            finalization_notes.append("blocked item 被转为 needs_human_decision。")
        related_absence_reason = item.related_absence_reason
        if not item.related_pages and related_absence_reason is None:
            related_absence_reason = "no_candidate" if not inspected_paths else "low_confidence"
        preliminary.append(
            item.model_copy(
                update={
                    "action": action,
                    "model_action": model_action,
                    "finalization_reason": "；".join(_markdown_utils.dedupe_strings([*finalization_notes, item.finalization_reason])) or "模型动作已按冻结 wiki context 校验。",
                    "canonical_target_path": canonical,
                    "matched_page": matched_page,
                    "inspected_context_paths": _markdown_utils.dedupe_strings([*item.inspected_context_paths, *inspected_paths]),
                    "strongest_overlap": strongest_overlap,
                    "why_not_update": item.why_not_update,
                    "why_create_or_update": item.why_create_or_update or item.reason,
                    "related_absence_reason": related_absence_reason,
                    "page_plan_id": resolution_item.page_plan_id,
                    "source_basis": resolution_item.source_basis,
                    "page_type": resolution_item.page_type,
                    "display_title": _wiki_markup.clean_display_title(item.display_title or resolution_item.display_title),
                    "apply_eligibility": apply_eligibility,
                    "blocked_reason": blocked_reason,
                }
            )
        )
    items: list[WikiMergePlanItem] = []
    for item in preliminary:
        resolution_item = resolution_by_id[item.page_plan_id]
        related_pages, related_unresolved = resolve_model_related_pages(item, resolution_item, preliminary, resolution_by_id, snapshot)
        unresolved = _markdown_utils.dedupe_strings([*item.related_unresolved, *item.unresolved_related, *related_unresolved])
        related_absence_reason = item.related_absence_reason
        if not related_pages and related_absence_reason is None:
            related_absence_reason = "cap_cutoff" if related_unresolved else ("no_candidate" if not item.inspected_context_paths else "low_confidence")
        items.append(
            item.model_copy(
                update={
                    "related_pages": related_pages,
                    "related_unresolved": unresolved,
                    "unresolved_related": unresolved,
                    "related_absence_reason": related_absence_reason,
                }
            )
        )
    items = _merge_plan_refinement.merge_same_source_duplicate_creates(items)
    items = _merge_plan_refinement.merge_update_noop_same_targets(items)
    return WikiMergePlanArtifact(log_date=snapshot.log_date, items=items, context_snapshot_ref=snapshot_ref)


def ensure_wiki_context_current(vault: Path, snapshot: WikiContextSnapshot) -> None:
    messages = wiki_context_drift_messages(vault, snapshot)
    if messages:
        raise _errors.PipelineError("; ".join(messages))
