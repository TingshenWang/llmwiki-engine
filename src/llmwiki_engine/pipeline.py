from __future__ import annotations

import json
import shutil
import re
import unicodedata
from difflib import unified_diff
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from rich.console import Console
from pydantic import BaseModel

from . import __version__
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
    ApplyPreview,
    ApplyTarget,
    ArtifactRef,
    CandidateResolutionArtifact,
    CandidateResolutionItem,
    DraftApproval,
    DraftGroundingReview,
    DraftPageItem,
    DraftRenderingArtifact,
    DraftWriteManifest,
    DraftWriteTarget,
    GroundingClaim,
    OperationManifest,
    OperationStatus,
    RawBinding,
    RawLinkCleanupArtifact,
    RawLinkCleanupLink,
    RawLinkCleanupWarning,
    RawPreparationArtifact,
    ReviewDecision,
    RunMode,
    RelatedPageRef,
    RelatedMergeReport,
    RelatedCandidateReport,
    SourceBasis,
    SourceDuplicateGuardArtifact,
    SourceDigestArtifact,
    SourceDigestCandidate,
    StepStatus,
    StructuredIssue,
    StructuredRepairReport,
    UpdateMergeReport,
    UpdatePageMergeReport,
    SectionMergeChange,
    VaultConfig,
    WeakOrNoiseItem,
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
from .profiles import load_profile, page_output_path, safe_filename
from .rendering import source_title_for_raw
from .retrieval import (
    RetrievalError,
    build_candidate_contexts,
    build_knowledge_pool,
    candidate_pool_sha256,
    metadata_from_text,
    resolve_cache_dir,
)
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
    assert_current_system_page,
    ensure_system_pages,
    format_markdown_table,
    local_date,
    render_daily_log,
    render_index,
    render_log_index,
)
from .validators import (
    ValidationError as ContractValidationError,
    create_reason_needs_repair,
    looks_like_untranslated_english,
    text_contains_source_graph_link,
    validate_candidate_resolution,
    validate_raw_preparation,
    validate_source_digest,
    validate_wiki_merge_plan,
)
from .verify import require_verified
from .vault_config import read_vault_config, write_default_vault_config
from .wiki_context import wiki_context_drift_messages
from .workspace import RunStore, apply_lock, ensure_workspace_layout, relative_to_vault, resolve_raw_path, run_lock


class PipelineError(RuntimeError):
    pass


RAW_PREPARE_CONTRACT = {
    "goal": "Create a higher-quality canonical prepared raw for downstream knowledge compilation.",
    "rules": [
        "Do not add facts that are not supported by the original raw.",
        "Remove or relocate non-content noise such as media timestamps, self-promotion, and obvious formatting artifacts.",
        "Correct obvious ASR/OCR/formatting errors only when the context makes the correction clear.",
        "Record uncertainty instead of guessing.",
        "Return prepared_markdown as clean Markdown suitable for source_digest and downstream knowledge digestion.",
    ],
}

MODEL_RELATED_SUGGESTION_LIMIT = 2
FINAL_RELATED_LIMIT = 3

def init_vault(vault: Path, *, profile_name: str = "project_basic") -> None:
    profile = load_profile(profile_name)
    (vault / "raw").mkdir(parents=True, exist_ok=True)
    for spec in profile.page_types.values():
        (vault / "wiki" / spec.directory).mkdir(parents=True, exist_ok=True)
    ensure_system_pages(vault)
    ensure_workspace_layout(vault)
    write_default_vault_config(vault)
    profile_root = vault / ".llmwiki" / "profiles" / profile.name
    (profile_root / "templates").mkdir(parents=True, exist_ok=True)
    write_yaml(profile_root / "profile.yaml", profile.model_dump(mode="json"))
    for page_type, spec in profile.page_types.items():
        source = profile.template_root / spec.template if profile.template_root else None
        if source and source.exists():
            target = profile_root / "templates" / spec.template
            if not target.exists():
                shutil.copyfile(source, target)
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
    fixture_dir: Path | None = None,
    profile_name: str | None = None,
    slug: str | None = None,
    run_mode: RunMode = RunMode.dev,
    console: Console | None = None,
) -> OperationManifest:
    ensure_workspace_layout(vault)
    raw_path, raw_rel = resolve_raw_path(vault, raw_file)
    raw_hash, raw_size = raw_ref(raw_path)
    operation_id = f"ING-{safe_timestamp()}-{slug or raw_path.stem}"
    store = RunStore(vault)
    resolved_profile_name = resolve_vault_profile_name(vault, profile_name)
    profile = load_profile(vault / ".llmwiki" / "profiles" / resolved_profile_name)
    provider_execution_context = build_provider_execution_context(
        vault=vault,
        manifest_contexts=[],
        fixture_dir=fixture_dir,
        source="initial_run",
        from_step=None,
        tasks=list(MODEL_BACKED_STEPS),
    )
    with apply_lock(vault):
        vault_config = read_vault_config(vault)
        run_dir = store.run_dir(operation_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = OperationManifest(
            operation_id=operation_id,
            operation_type="ingest",
            run_mode=run_mode,
            engine_version=__version__,
            profile=profile.name,
            profile_version=profile.version,
            vault_config_snapshot=vault_config,
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
    run_mode: RunMode | None = None,
    console: Console | None = None,
) -> OperationManifest:
    store = RunStore(vault)
    with apply_lock(vault), run_lock(vault, operation_id):
        manifest = read_manifest(store.manifest_path(operation_id))
        if run_mode is not None and run_mode != manifest.run_mode:
            raise PipelineError("run_mode is immutable for an operation; rerun ingest to use a different mode.")
        if manifest.status in {OperationStatus.applied, OperationStatus.source_recorded}:
            raise PipelineError("Applied operations are immutable. Start a new operation instead.")
        if manifest.status == OperationStatus.apply_failed:
            raise PipelineError("apply_failed operations cannot be resumed; inspect written targets and rerun ingest.")
        require_verified(vault, manifest)
        start, reset_from_step = default_resume_start(vault, store.run_dir(operation_id), manifest, from_step)
        if start is None:
            return manifest
        validate_raw_link_cleanup_resume(run_dir=store.run_dir(operation_id), manifest=manifest, start=start)
        validate_resume_start(manifest, start)
        ensure_wiki_context_current_before_resume(vault, store.run_dir(operation_id), start)
        model_steps = model_steps_from(start)
        provider_execution_context = build_provider_execution_context(
            vault=vault,
            manifest_contexts=manifest.provider_contexts,
            fixture_dir=None,
            source="resume_current_config",
            from_step=start,
            tasks=model_steps,
        )
        if provider_execution_context.record is not None:
            manifest.provider_contexts.append(provider_execution_context.record)
        if reset_from_step is not None:
            write_manifest(store.manifest_path(operation_id), manifest)
            delete_downstream_step_dirs(
                vault,
                operation_id,
                reset_from_step,
                archive=True,
                archive_reason=f"resume requested from {reset_from_step}; previous step artifacts archived before regeneration.",
            )
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
        raise PipelineError("; ".join(messages))


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
            refresh_run_metrics(vault, run_dir, manifest, warning_console=console)
            logger.emit(step_name, "failed", status="failed", message=message, duration_ms=last_attempt_duration_ms(manifest, step_name))
            raise PipelineError(message) from exc
        write_manifest(store.manifest_path(operation_id), manifest)
        refresh_run_metrics(vault, run_dir, manifest, warning_console=console)
        if get_step(manifest, step_name).status == StepStatus.awaiting_review:
            manifest.status = OperationStatus.awaiting_review
            write_manifest(store.manifest_path(operation_id), manifest)
            refresh_run_metrics(vault, run_dir, manifest, warning_console=console)
            return manifest
    ensure_pipeline_completed(manifest)
    manifest.status = OperationStatus.drafted
    write_manifest(store.manifest_path(operation_id), manifest)
    refresh_run_metrics(vault, run_dir, manifest, warning_console=console)
    return manifest


def validate_resume_start(manifest: OperationManifest, start_step: str) -> None:
    start_index = step_index(start_step)
    for step in manifest.steps[:start_index]:
        if not step_satisfied(step.status):
            raise PipelineError(
                f"Cannot resume from {start_step}: upstream step {step.name} is {step.status.value}; "
                f"resume from {step.name} or earlier."
            )


def ensure_pipeline_completed(manifest: OperationManifest) -> None:
    incomplete = [step for step in manifest.steps if not step_satisfied(step.status)]
    if incomplete:
        details = ", ".join(f"{step.name}={step.status.value}" for step in incomplete)
        raise PipelineError(f"Operation is not draft-ready; incomplete step(s): {details}")


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
        raise PipelineError(f"Unknown step: {step_name}")
    provider_record = execution_context.record if runner.spec.model_backed else None
    provider_runtime = provider_record.providers.get(step_name) if provider_record else None
    logger.emit(
        step_name,
        "started",
        status="running",
        model_backed=runner.spec.model_backed,
        provider_spec=provider_runtime.spec if provider_runtime else None,
    )
    if runner.spec.model_backed:
        if provider_record is None or provider_runtime is None:
            raise PipelineError(f"No provider execution context found for model-backed step: {step_name}")
        begin_model_step_attempt(
            manifest,
            step_name,
            provider_record_id=provider_record.record_id,
            provider_spec=provider_runtime.spec,
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


RAW_LINK_CLEANUP_RULE_VERSION = "obsidian_text_wikilink.v1"
WIKILINK_RE = re.compile(r"(?<!!)\[\[([^\]\n]+)\]\]")
MEDIA_EMBED_RE = re.compile(r"!\[\[([^\]\n]+)\]\]")


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
            raise PipelineError("raw changed during raw_link_cleanup; rerun ingest")
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
    diff_path.write_text(render_update_diff(original_text, cleaned_text, f"pre/{raw_rel}", f"post/{raw_rel}"), encoding="utf-8")
    complete_step(
        ctx.manifest,
        step_name,
        outputs=[
            _ref(ctx.run_dir, out, step_name, "json", "raw_link_cleanup.v1"),
            _ref(ctx.run_dir, report, step_name, "markdown"),
            _ref(ctx.run_dir, diff_path, step_name, "diff"),
        ],
    )


def cleanup_raw_wikilinks(text: str) -> tuple[str, list[RawLinkCleanupLink], list[RawLinkCleanupWarning], int]:
    lines = text.splitlines(keepends=True)
    cleaned_lines: list[str] = []
    links: list[RawLinkCleanupLink] = []
    warnings: list[RawLinkCleanupWarning] = []
    in_frontmatter = False
    frontmatter_seen = False
    in_fenced_code = False
    fence_marker = ""
    preserved_media_total = 0
    for index, line in enumerate(lines, start=1):
        newline = ""
        body = line
        if body.endswith("\r\n"):
            body = body[:-2]
            newline = "\r\n"
        elif body.endswith("\n"):
            body = body[:-1]
            newline = "\n"
        stripped = body.strip()
        if index == 1 and stripped == "---":
            in_frontmatter = True
            frontmatter_seen = True
            cleaned_lines.append(line)
            continue
        if in_frontmatter and index != 1 and stripped == "---":
            in_frontmatter = False
            cleaned_lines.append(line)
            continue
        fence_match = re.match(r"^(\s*)(```+|~~~+)", body)
        if not in_frontmatter and fence_match:
            marker = fence_match.group(2)[0]
            if not in_fenced_code:
                in_fenced_code = True
                fence_marker = marker
            elif fence_marker == marker:
                in_fenced_code = False
                fence_marker = ""
            cleaned_lines.append(line)
            continue
        context = "frontmatter" if in_frontmatter and frontmatter_seen else "body"
        if in_fenced_code:
            cleaned_lines.append(line)
            continue
        media_matches = list(MEDIA_EMBED_RE.finditer(body))
        for match in media_matches[: max(0, 20 - len(warnings))]:
            warnings.append(
                RawLinkCleanupWarning(
                    warning_type="preserved_media_embed",
                    message="保留 Obsidian 媒体链接；本轮只清理文本 wikilink。",
                    line_number=index,
                    line_excerpt=truncate_excerpt(body),
                )
            )
        preserved_media_count = len(media_matches)
        preserved_media_total += preserved_media_count
        cleaned_body, line_links = cleanup_wikilinks_in_line(body, line_number=index, context=context, start_index=len(links) + 1)
        links.extend(line_links)
        cleaned_lines.append(cleaned_body + newline)
    return "".join(cleaned_lines), links, warnings, preserved_media_total


def cleanup_wikilinks_in_line(
    line: str,
    *,
    line_number: int,
    context: Literal["frontmatter", "body"],
    start_index: int,
) -> tuple[str, list[RawLinkCleanupLink]]:
    pieces: list[str] = []
    links: list[RawLinkCleanupLink] = []
    cursor = 0
    in_code = False
    for match in re.finditer(r"`+", line):
        segment = line[cursor : match.start()]
        pieces.append(_cleanup_wikilink_segment(segment, line, line_number=line_number, context=context, start_index=start_index + len(links), links=links) if not in_code else segment)
        pieces.append(match.group(0))
        in_code = not in_code
        cursor = match.end()
    tail = line[cursor:]
    pieces.append(_cleanup_wikilink_segment(tail, line, line_number=line_number, context=context, start_index=start_index + len(links), links=links) if not in_code else tail)
    return "".join(pieces), links


def _cleanup_wikilink_segment(
    segment: str,
    original_line: str,
    *,
    line_number: int,
    context: Literal["frontmatter", "body"],
    start_index: int,
    links: list[RawLinkCleanupLink],
) -> str:
    def replace(match: re.Match[str]) -> str:
        raw = match.group(1).strip()
        target, label = split_wikilink(raw)
        link_id = f"L{start_index + len(links):03d}"
        links.append(
            RawLinkCleanupLink(
                link_id=link_id,
                link_kind="wikilink",
                label=label,
                target=target,
                cleanup_action="unwrap_text",
                cleanup_context=context,
                line_number=line_number,
                original_line_hash=sha256_bytes(original_line.encode("utf-8")),
                line_excerpt=truncate_excerpt(original_line),
            )
        )
        return label

    return WIKILINK_RE.sub(replace, segment)


def split_wikilink(raw: str) -> tuple[str, str]:
    if "|" in raw:
        target, label = raw.split("|", 1)
        return target.strip(), label.strip() or target.strip()
    return raw.strip(), raw.strip()


def truncate_excerpt(line: str, limit: int = 160) -> str:
    text = " ".join(line.split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1]}..."


def render_raw_link_cleanup_markdown(artifact: RawLinkCleanupArtifact) -> str:
    rows = [
        ["raw_path", f"`{artifact.raw_path}`"],
        ["changed", str(artifact.changed).lower()],
        ["cleanup_rule_version", artifact.cleanup_rule_version],
        ["pre_cleanup_sha256", f"`{artifact.pre_cleanup_sha256}`"],
        ["post_cleanup_sha256", f"`{artifact.post_cleanup_sha256}`"],
        ["cleaned_link_count", str(artifact.cleaned_link_count)],
        ["preserved_media_embed_count", str(artifact.preserved_media_embed_count)],
    ]
    link_rows = [
        [link.link_id, link.cleanup_context, str(link.line_number), link.target, link.label, link.line_excerpt]
        for link in artifact.links
    ]
    warning_rows = [
        [warning.warning_type, str(warning.line_number), warning.message, warning.line_excerpt]
        for warning in artifact.warnings[:20]
    ]
    return (
        "# Raw Obsidian Wikilink 规范化\n\n"
        f"{format_markdown_table(['字段', '值'], rows)}\n\n"
        "## 已清理文本 Wikilink\n\n"
        f"{format_markdown_table(['ID', '位置', '行号', 'Target', 'Label', '行摘录'], link_rows) if link_rows else '_暂无。_'}\n\n"
        "## 保留项 Warning\n\n"
        f"{format_markdown_table(['类型', '行号', '说明', '行摘录'], warning_rows) if warning_rows else '_暂无。_'}\n\n"
        "## Diff\n\n"
        "`cleanup.diff` 是 pre-clean -> post-clean 的 unified diff，仅供人工审计/恢复参考，不会自动 rollback。\n"
    )


def _run_raw_prepare(ctx: StepRunContext) -> None:
    step_name = "raw_prepare"
    step_root = require_step_output_dir(ctx.run_dir, step_name)
    raw_rel = relative_to_vault(ctx.vault, ctx.raw_path)
    cleanup_path = require_step_output_dir(ctx.run_dir, "raw_link_cleanup") / "raw_link_cleanup.json"
    cleanup = read_model(cleanup_path, RawLinkCleanupArtifact)
    input_raw_sha256 = sha256_file(ctx.raw_path)
    payload = {
        "source_raw_path": raw_rel,
        "source_raw_sha256": input_raw_sha256,
        "raw_markdown": ctx.raw_path.read_text(encoding="utf-8"),
        "raw_link_cleanup_ref": cleanup_path.relative_to(ctx.run_dir).as_posix(),
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
                "raw_link_cleanup_ref": cleanup_path.relative_to(ctx.run_dir).as_posix(),
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
            "raw_link_cleanup_ref": cleanup_path.relative_to(ctx.run_dir).as_posix(),
        }
    )
    validate_raw_preparation(preparation)
    if preparation.source_raw_path != raw_rel:
        raise PipelineError(f"raw_prepare source path mismatch: {preparation.source_raw_path} != {raw_rel}")
    out = step_root / "raw_preparation.json"
    write_json(out, preparation)
    prepared = step_root / "prepared.md"
    prepared.parent.mkdir(parents=True, exist_ok=True)
    prepared.write_text(preparation.prepared_markdown.rstrip() + "\n", encoding="utf-8")
    review = step_root / "preparation_review.md"
    review.write_text(render_preparation_review(preparation), encoding="utf-8")
    outputs = [
        _ref(ctx.run_dir, out, step_name, "json", "raw_preparation.v1"),
        _ref(ctx.run_dir, prepared, step_name, "markdown"),
        _ref(ctx.run_dir, review, step_name, "markdown"),
    ]
    outputs.extend(structured_model_output_refs(ctx.run_dir, step_root, step_name))
    complete_step(ctx.manifest, step_name, outputs=outputs)


def _run_prepared_raw_review(ctx: StepRunContext) -> None:
    step_name = "prepared_raw_review"
    step_root = require_step_output_dir(ctx.run_dir, step_name)
    prepared = require_step_output_dir(ctx.run_dir, "raw_prepare") / "prepared.md"
    approved = step_root / "approved_prepared.md"
    approved.write_text(prepared.read_text(encoding="utf-8"), encoding="utf-8")
    prompt = step_root / "review_prompt.md"
    prompt.write_text(
        "# Prepared Raw 审核\n\n"
        "当前 MVP 自动批准 prepared raw；后续会加入交互式人工审核。\n",
        encoding="utf-8",
    )
    feedback = step_root / "review_feedback.jsonl"
    feedback.write_text("", encoding="utf-8")
    decision = ReviewDecision(
        review_step=step_name,
        decision="approved",
        review_mode="auto_stub",
        auto_approved=True,
        notes="当前 MVP 自动批准；交互式审核是后续工作。",
    )
    decision_path = step_root / "review_decision.json"
    write_json(decision_path, decision)
    complete_review_step(
        ctx.manifest,
        step_name,
        outputs=[
            _ref(ctx.run_dir, prompt, step_name, "markdown"),
            _ref(ctx.run_dir, feedback, step_name, "jsonl"),
            _ref(ctx.run_dir, decision_path, step_name, "json", "review_decision.v1"),
            _ref(ctx.run_dir, approved, step_name, "markdown"),
        ],
        review_decision_ref=decision_path.relative_to(ctx.run_dir).as_posix(),
    )


def _run_source_digest(ctx: StepRunContext) -> None:
    step_name = "source_digest"
    step_root = require_step_output_dir(ctx.run_dir, step_name)
    raw_rel = relative_to_vault(ctx.vault, ctx.raw_path)
    approved_prepared = require_step_output_dir(ctx.run_dir, "prepared_raw_review") / "approved_prepared.md"
    payload = {
        "source_raw_path": raw_rel,
        "approved_prepared_markdown": approved_prepared.read_text(encoding="utf-8"),
        "profile": ctx.profile.model_dump(mode="json"),
        "contract": {
            "goal": "Create a complete source digest of wiki-worthy candidates from this one raw file.",
            "candidate_fields": list(SourceDigestCandidate.model_fields),
            "weak_or_noise_fields": list(WeakOrNoiseItem.model_fields),
            "rules": [
                "Return each candidate group as an array of candidate objects, never as bare fields.",
                "Every candidate object must include candidate_id, name, type, one_sentence_summary, why_matters, and wiki_value.",
                "For entities, concepts, designs, comparisons, and open_questions, suggested_page_title must be non-empty.",
                "Do not decide create, update, duplicate, or cross-reference actions in source_digest.",
                "Prefer wiki-worthy candidates over every minor mention.",
                "Use source_locator as a lightweight review locator, not a strict evidence chain.",
                "Put weak or noisy mentions in weak_or_noise_items instead of creating pages for them.",
                "weak_or_noise_items are review-only and are not ingested as wiki pages.",
                "weak_or_noise_items may leave suggested_page_title empty and may use suggested_action='ignore'.",
                "weak_or_noise_items may use why_matters or why_matches to explain why the mention was filtered.",
                "The vault language is zh-CN: write summary, key_takeaways, candidate summaries, why_matters, and wiki_value in Chinese.",
                "Stable domain terms such as Claude Code, RAG, PM, Workflow, Agent may stay in English, but explain them in Chinese when needed.",
                "Do not return whole English paragraphs for user-visible fields; zh-CN validation will fail instead of silently translating.",
            ],
        },
    }
    digest, _ = _structured_call(ctx.run_dir, ctx.execution_context, step_name).run(
        step_name,
        payload,
        SourceDigestArtifact,
        validator=lambda model: validate_source_digest(model, language=ctx.manifest.vault_config_snapshot.wiki_language),
    )
    digest = _redacted_model(ctx, digest, SourceDigestArtifact)
    if digest.source_raw_path != raw_rel:
        raise PipelineError(f"source_digest source path mismatch: {digest.source_raw_path} != {raw_rel}")
    out = step_root / "source_digest.json"
    write_json(out, digest)
    digest_md = step_root / "source_digest.md"
    digest_md.write_text(render_source_digest_markdown(digest), encoding="utf-8")
    outputs = [
        _ref(ctx.run_dir, out, step_name, "json", "source_digest.v2"),
        _ref(ctx.run_dir, digest_md, step_name, "markdown"),
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
        "当前 MVP 自动批准 source digest；这一步检查单篇材料提取是否完整、是否中文、是否把弱相关内容放进噪声区。\n\n"
        "## 核心判断\n\n"
        "- 是否漏掉了值得入库的实体、概念、设计、对比或未决问题？\n"
        "- 是否把只是口播过渡、广告、寒暄或弱相关提及错误变成页面候选？\n"
        "- 摘要、关键收获和候选说明是否为中文，英文术语是否有中文上下文？\n\n"
        "## 关键文件\n\n"
        f"- 待审摘要：`{digest_md.relative_to(ctx.run_dir).as_posix()}`\n"
        f"- 可编辑批准文件：`{approved_json.relative_to(ctx.run_dir).as_posix()}`\n"
        f"- 反馈记录：`{feedback.relative_to(ctx.run_dir).as_posix()}`\n\n"
        "后续会加入真正的 list/filter/show/diff/revise/approve；本轮仍为 auto-stub。\n",
        encoding="utf-8",
    )
    feedback.write_text("", encoding="utf-8")
    decision = ReviewDecision(
        review_step=step_name,
        decision="approved",
        review_mode="auto_stub",
        auto_approved=True,
        notes="当前 MVP 自动批准 source digest；交互式 digest 审核是后续工作。",
    )
    decision_path = step_root / "review_decision.json"
    write_json(decision_path, decision)
    complete_review_step(
        ctx.manifest,
        step_name,
        outputs=[
            _ref(ctx.run_dir, prompt, step_name, "markdown"),
            _ref(ctx.run_dir, feedback, step_name, "jsonl"),
            _ref(ctx.run_dir, decision_path, step_name, "json", "review_decision.v1"),
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
    artifact = build_source_duplicate_guard_artifact(
        ctx.vault,
        source_raw_path=digest.source_raw_path,
        source_raw_hash=sha256_file(ctx.vault / digest.source_raw_path),
        source_prepared_hash=sha256_file(prepared),
        operation_id=ctx.manifest.operation_id,
    )
    out = step_root / "source_duplicate_guard.json"
    write_json(out, artifact)
    md = step_root / "source_duplicate_guard.md"
    md.write_text(render_source_duplicate_guard_markdown(artifact), encoding="utf-8")
    if artifact.status == "source_duplicate":
        raise PipelineError(f"source_duplicate: {artifact.reason}")
    if artifact.status == "source_revision_detected":
        raise PipelineError(
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
    payload = {
        "approved_prepared_markdown": approved_prepared.read_text(encoding="utf-8"),
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
    retrieval_config = effective_retrieval_config(ctx)
    snapshot = build_wiki_context_snapshot(
        ctx.vault,
        resolution,
        log_date=log_date,
        source_target_path=f"sources/{safe_filename(source_title)}.md",
        retrieval_config=retrieval_config,
        force_exact_backend=uses_mock_provider_context(ctx.execution_context),
    )
    ensure_snapshot_within_limit(snapshot, ctx.manifest.vault_config_snapshot.max_context_chars)
    contexts_path = step_root / "candidate_contexts.json"
    write_json(contexts_path, snapshot.candidate_contexts)
    contexts_md = step_root / "candidate_contexts.md"
    contexts_md.write_text(
        render_candidate_contexts_markdown(
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
            _ref(ctx.run_dir, snapshot_path, step_name, "json", "wiki_context_snapshot.v2"),
            _ref(ctx.run_dir, contexts_path, step_name, "json", "candidate_contexts.v1"),
            _ref(ctx.run_dir, contexts_md, step_name, "markdown"),
        ],
    )


def effective_retrieval_config(ctx: StepRunContext) -> EmbeddingRetrievalConfig:
    return ctx.manifest.vault_config_snapshot.embedding_retrieval


def uses_mock_provider_context(execution_context: ProviderExecutionContext) -> bool:
    if execution_context.record is None:
        return False
    providers = list(execution_context.record.providers.values())
    return bool(providers) and all(provider.spec.startswith("mock:") for provider in providers)


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
                        "finalization_reason": merge_markdown_blocks(
                            item.finalization_reason,
                            "模型 repair 后 why_not_update 仍不充分，已转为 needs_human_decision。",
                        ),
                    }
                )
            )
            continue
        items.append(item)
    return plan.model_copy(update={"items": items})


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
    payload = {
        "approved_prepared_markdown": (require_step_output_dir(ctx.run_dir, "prepared_raw_review") / "approved_prepared.md").read_text(encoding="utf-8"),
        "approved_digest": digest.model_dump(mode="json"),
        "candidate_resolution": resolution.model_dump(mode="json"),
        "wiki_context_snapshot": snapshot.model_dump(mode="json"),
        "candidate_contexts": candidate_contexts.model_dump(mode="json"),
        "profile": ctx.profile.model_dump(mode="json"),
        "language_contract": ctx.manifest.vault_config_snapshot.model_dump(mode="json"),
        "contract": {
            "goal": "Read the frozen wiki context and decide create/update/noop/needs_human_decision for planned pages.",
            "actions": ["create", "update", "noop", "needs_human_decision"],
            "rules": [
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
    snapshot_ref = snapshot_path.relative_to(ctx.run_dir).as_posix()
    plan = finalize_wiki_merge_plan(plan, resolution, snapshot, snapshot_ref, medium_missing_policy="preserve")
    plan = block_unrepaired_medium_create_reason(plan)
    validate_wiki_merge_plan(digest, plan, resolution, snapshot, language=ctx.manifest.vault_config_snapshot.wiki_language)
    out = step_root / "wiki_merge_plan.json"
    write_json(out, plan)
    table = step_root / "wiki_merge_plan.md"
    table.write_text(render_merge_plan_markdown(plan), encoding="utf-8")
    report = step_root / "merge_decision_report.md"
    report.write_text(render_merge_decision_report(plan, snapshot), encoding="utf-8")
    outputs = [
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
    prompt_path.write_text(render_merge_plan_review_prompt(plan), encoding="utf-8")
    has_needs_human = any(item.action == "needs_human_decision" or item.apply_eligibility == "blocked" for item in plan.items)
    all_create_risk = merge_plan_all_create_review_reason(plan)
    if has_needs_human or all_create_risk:
        pending_path = step_root / "pending_merge_plan.json"
        pending_path.write_text(plan_path.read_text(encoding="utf-8"), encoding="utf-8")
        decision = ReviewDecision(
            review_step=step_name,
            decision="pending",
            review_mode="manual",
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
                _ref(ctx.run_dir, decision_path, step_name, "json", "review_decision.v1"),
            ],
            reason=all_create_risk or "merge plan requires human decision; run merge-level revise.",
            review_decision_ref=decision_path.relative_to(ctx.run_dir).as_posix(),
        )
        return
    approved_path.write_text(plan_path.read_text(encoding="utf-8"), encoding="utf-8")
    decision = ReviewDecision(
        review_step=step_name,
        decision="approved",
        review_mode="auto_stub",
        auto_approved=True,
        notes="合并计划不含 needs_human_decision，本轮按 auto-stub 自动批准。",
    )
    decision_path = step_root / "review_decision.json"
    write_json(decision_path, decision)
    complete_review_step(
        ctx.manifest,
        step_name,
        outputs=[
            _ref(ctx.run_dir, prompt_path, step_name, "markdown"),
            _ref(ctx.run_dir, feedback_path, step_name, "jsonl"),
            _ref(ctx.run_dir, decision_path, step_name, "json", "review_decision.v1"),
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
    approved_prepared_text = (require_step_output_dir(ctx.run_dir, "prepared_raw_review") / "approved_prepared.md").read_text(encoding="utf-8")
    payload = {
        "approved_prepared_markdown": approved_prepared_text,
        "approved_digest": digest.model_dump(mode="json"),
        "approved_merge_plan": merge_plan.model_dump(mode="json"),
        "wiki_context_snapshot": snapshot.model_dump(mode="json"),
        "profile": ctx.profile.model_dump(mode="json"),
        "language_contract": ctx.manifest.vault_config_snapshot.model_dump(mode="json"),
        "contract": {
            "goal": "Generate structured section bodies for each create/update page from the full approved source and frozen wiki context.",
            "section_body_keys": ["summary", "detail", "examples", "value_points", "additional_notes", "open_questions"],
            "rules": [
                "Return section body content only; do not include frontmatter, level-1 headings, source wikilinks, or full markdown pages.",
                "section_bodies must use only these exact keys: summary, detail, examples, value_points, additional_notes, open_questions.",
                "Each section_bodies value must be one Markdown string; for bullet lists, write bullets inside that string instead of returning JSON arrays.",
                "Use additional_notes for free-form observations or custom subtopics; do not invent custom top-level section keys.",
                "For updates, read the existing page content from snapshot and produce a complete replacement draft at the section-body level.",
                "Do not produce pages that are only source summaries; every page must include concrete digested understanding such as viewpoint, example, use scenario, boundary condition, or value point.",
                "For updates, change_summary must explain what the new source adds, changes, clarifies, retains, or removes from the old understanding.",
                "If the merge plan has merged_page_plan_ids, absorb the unique section intent/examples/value points from suppressed candidates into the canonical page.",
                "Write all user-visible content in Chinese except stable domain terms with Chinese explanation when needed.",
                "Ground examples, value points, and reuse scenarios in source content.",
                "Do not write implementation details, examples, or claims as facts unless they are supported by approved_prepared_markdown or inspected wiki context.",
                "If a useful detail is plausible but unsupported, put it under open_questions as 待补来源 instead of writing it as fact.",
                "source_coverage_notes must briefly say which source/wiki context supports the page and what was intentionally left uncertain.",
            ],
        },
    }
    def validate_draft_rendering_model(model: DraftRenderingArtifact) -> None:
        candidate = finalize_draft_rendering(model, merge_plan, snapshot)
        validate_draft_rendering(candidate, merge_plan, language=ctx.manifest.vault_config_snapshot.wiki_language)
        grounding_review = build_draft_grounding_review(candidate, merge_plan, snapshot, approved_prepared_text)
        if grounding_review.requires_review:
            raise ContractValidationError(
                "draft_rendering contains unsupported new_fact; repair by removing it, rewriting it as sourced text, or moving it to open_questions.",
                issues=[
                    StructuredIssue(
                        issue_code="unsupported_new_fact",
                        field_path=f"pages.{claim.page_plan_id}.{claim.section_key}",
                        validator_id="draft_grounding_review",
                        message=claim.reason or "unsupported new_fact",
                        repairability="repairable",
                    )
                    for claim in grounding_review.unsupported_new_facts
                ],
            )

    draft_artifact, _ = _structured_call(ctx.run_dir, ctx.execution_context, step_name).run(
        step_name,
        payload,
        DraftRenderingArtifact,
        validator=validate_draft_rendering_model,
        accept_after_repair_issue_codes={"unsupported_new_fact"},
    )
    draft_artifact = _redacted_model(ctx, draft_artifact, DraftRenderingArtifact)
    draft_artifact = finalize_draft_rendering(draft_artifact, merge_plan, snapshot)
    validate_draft_rendering(draft_artifact, merge_plan, language=ctx.manifest.vault_config_snapshot.wiki_language)
    draft_artifact_path = step_root / "draft_rendering.json"
    write_json(draft_artifact_path, draft_artifact)
    draft_root = step_root / "draft_pages"
    outputs: list[Path] = []
    target_manifest: list[DraftWriteTarget] = []
    source_title = source_title_for_raw(digest.source_raw_path)
    action_by_id = {item.page_plan_id: item for item in merge_plan.items}
    update_report_pages: list[UpdatePageMergeReport] = []
    related_report_candidates: list[RelatedCandidateReport] = []
    grounding_claims: list[GroundingClaim] = []
    known_related_paths = {
        entry.path
        for entry in snapshot.knowledge_metadata_pool
        if not entry.path.startswith("sources/") and not entry.path.startswith("logs/") and entry.path not in {"index.md", "log.md"}
    }
    known_related_paths.update(
        item.canonical_target_path
        for item in merge_plan.items
        if item.action in {"create", "update", "noop"} and not item.canonical_target_path.startswith(("sources/", "logs/"))
    )
    knowledge_changed_paths: list[str] = []
    no_change_pages = [item.canonical_target_path for item in merge_plan.items if item.action == "noop"]
    for page in draft_artifact.pages:
        plan_item = action_by_id[page.page_plan_id]
        entry = snapshot_entry(snapshot, f"wiki/{page.canonical_target_path}")
        target = draft_root / page.canonical_target_path
        target.parent.mkdir(parents=True, exist_ok=True)
        markdown = assemble_knowledge_page(
            item=plan_item,
            page=page,
            existing_entry=entry,
            raw_path=digest.source_raw_path,
            raw_hash=sha256_file(ctx.vault / digest.source_raw_path),
            prepared_hash=sha256_file(require_step_output_dir(ctx.run_dir, "prepared_raw_review") / "approved_prepared.md"),
            operation_id=ctx.manifest.operation_id,
            log_date=snapshot.log_date,
            update_reports=update_report_pages,
            related_reports=related_report_candidates,
            grounding_claims=grounding_claims,
            known_related_paths=known_related_paths,
            approved_raw_text=approved_prepared_text,
        )
        target.write_text(markdown, encoding="utf-8")
        outputs.append(target)
        knowledge_changed_paths.append(page.canonical_target_path)
        target_manifest.append(
            DraftWriteTarget(
                action=page.action,
                target_path=f"wiki/{page.canonical_target_path}",
                draft_path=target.relative_to(ctx.run_dir).as_posix(),
                expected_state=entry.expected_state,
                preimage_sha256=entry.preimage_sha256,
                page_plan_id=page.page_plan_id,
            )
        )
        if page.action == "update":
            diff_path = step_root / "diffs" / f"{page.page_plan_id}.diff"
            diff_path.parent.mkdir(parents=True, exist_ok=True)
            diff_path.write_text(
                render_update_diff(entry.content, markdown, f"old/{page.canonical_target_path}", f"new/{page.canonical_target_path}"),
                encoding="utf-8",
            )
            diff_json = step_root / "diffs" / f"{page.page_plan_id}.json"
            write_json(
                diff_json,
                {
                    "old_snapshot_ref": f"wiki_context_snapshot/wiki_context_snapshot.json#{entry.path}",
                    "old_rendered_markdown_path": entry.path,
                    "new_draft_ref": target.relative_to(ctx.run_dir).as_posix(),
                    "new_rendered_markdown_path": target.relative_to(ctx.run_dir).as_posix(),
                    "unified_diff": diff_path.relative_to(ctx.run_dir).as_posix(),
                    "change_summary": page.change_summary,
                    "preimage_sha256": page.preimage_sha256,
                    "draft_sha256": sha256_file(target),
                },
            )
            outputs.extend([diff_path, diff_json])
    source_page = draft_root / snapshot.source_target_path
    source_page.parent.mkdir(parents=True, exist_ok=True)
    source_page.write_text(
        render_source_page(
            title=source_title,
            digest=digest,
            operation_id=ctx.manifest.operation_id,
            linked_pages=knowledge_changed_paths,
            touched_pages=[item.canonical_target_path for item in merge_plan.items],
            no_change_pages=no_change_pages,
            log_date=snapshot.log_date,
            raw_hash=sha256_file(ctx.vault / digest.source_raw_path),
            prepared_hash=sha256_file(require_step_output_dir(ctx.run_dir, "prepared_raw_review") / "approved_prepared.md"),
            cleanup=read_model(require_step_output_dir(ctx.run_dir, "raw_link_cleanup") / "raw_link_cleanup.json", RawLinkCleanupArtifact),
        ),
        encoding="utf-8",
    )
    outputs.append(source_page)
    source_entry = snapshot_entry(snapshot, f"wiki/{snapshot.source_target_path}")
    target_manifest.append(
        DraftWriteTarget(
            action="source",
            target_path=f"wiki/{snapshot.source_target_path}",
            draft_path=source_page.relative_to(ctx.run_dir).as_posix(),
            expected_state=source_entry.expected_state,
            preimage_sha256=source_entry.preimage_sha256,
        )
    )
    log_date = snapshot.log_date
    if knowledge_changed_paths:
        assert_system_page_can_be_overwritten(ctx.vault, "wiki/index.md")
        index = draft_root / "index.md"
        open_question_rows, open_question_report = build_open_question_rows_with_report(merge_plan, draft_artifact, snapshot)
        index.write_text(
            render_index(
                knowledge_rows=build_index_rows(ctx.profile, merge_plan, draft_artifact, snapshot),
                tension_rows=open_question_rows,
                page_type_order=list(ctx.profile.page_types),
            ),
            encoding="utf-8",
        )
        open_question_report_path = step_root / "index_open_questions_report.json"
        open_question_report_md = step_root / "index_open_questions_report.md"
        write_json(open_question_report_path, open_question_report)
        open_question_report_md.write_text(render_index_open_questions_report(open_question_report), encoding="utf-8")
        outputs.append(index)
        outputs.extend([open_question_report_path, open_question_report_md])
        index_entry = snapshot_entry(snapshot, "wiki/index.md")
        target_manifest.append(
            DraftWriteTarget(
                action="index",
                target_path="wiki/index.md",
                draft_path=index.relative_to(ctx.run_dir).as_posix(),
                expected_state=index_entry.expected_state,
                preimage_sha256=index_entry.preimage_sha256,
            )
        )
    assert_system_page_can_be_overwritten(ctx.vault, "wiki/log.md")
    log_entry = snapshot_entry(snapshot, "wiki/log.md")
    log_index = draft_root / "log.md"
    log_index.write_text(
        render_log_index(
            date=log_date,
            operation_id=ctx.manifest.operation_id,
            source=digest.source_raw_path,
            existing_text=log_entry.content if log_entry.expected_state == "present" else None,
        ),
        encoding="utf-8",
    )
    outputs.append(log_index)
    target_manifest.append(
        DraftWriteTarget(
            action="global_log",
            target_path="wiki/log.md",
            draft_path=log_index.relative_to(ctx.run_dir).as_posix(),
            expected_state=log_entry.expected_state,
            preimage_sha256=log_entry.preimage_sha256,
        )
    )
    assert_system_page_can_be_overwritten(ctx.vault, f"wiki/logs/{log_date}.md")
    daily_entry = snapshot_entry(snapshot, f"wiki/logs/{log_date}.md")
    daily_log = draft_root / "logs" / f"{log_date}.md"
    daily_log.parent.mkdir(parents=True, exist_ok=True)
    daily_log.write_text(
        render_daily_log(
            date=log_date,
            operation_id=ctx.manifest.operation_id,
            raw_path=digest.source_raw_path,
            created=sum(1 for item in merge_plan.items if item.action == "create"),
            updated=sum(1 for item in merge_plan.items if item.action == "update"),
            noop=sum(1 for item in merge_plan.items if item.action == "noop"),
            needs_human=sum(1 for item in merge_plan.items if item.action == "needs_human_decision"),
            existing_text=daily_entry.content if daily_entry.expected_state == "present" else None,
        ),
        encoding="utf-8",
    )
    outputs.append(daily_log)
    target_manifest.append(
        DraftWriteTarget(
            action="daily_log",
            target_path=f"wiki/logs/{log_date}.md",
            draft_path=daily_log.relative_to(ctx.run_dir).as_posix(),
            expected_state=daily_entry.expected_state,
            preimage_sha256=daily_entry.preimage_sha256,
        )
    )
    update_report = UpdateMergeReport(pages=update_report_pages)
    update_report_path = step_root / "update_merge_report.json"
    update_report_md = step_root / "update_merge_report.md"
    write_json(update_report_path, update_report)
    update_report_md.write_text(render_update_merge_report(update_report), encoding="utf-8")
    related_report = RelatedMergeReport(candidates=related_report_candidates)
    related_report_path = step_root / "related_merge_report.json"
    related_report_md = step_root / "related_merge_report.md"
    write_json(related_report_path, related_report)
    related_report_md.write_text(render_related_merge_report(related_report), encoding="utf-8")
    grounding_review = draft_grounding_review_from_claims(grounding_claims)
    grounding_review_path = step_root / "draft_grounding_review.json"
    grounding_review_md = step_root / "draft_grounding_review.md"
    write_json(grounding_review_path, grounding_review)
    grounding_review_md.write_text(render_draft_grounding_review(grounding_review), encoding="utf-8")
    outputs.extend([update_report_path, update_report_md, related_report_path, related_report_md, grounding_review_path, grounding_review_md])
    write_manifest_artifact = DraftWriteManifest(
        targets=target_manifest,
        has_updates=any(item.action == "update" for item in merge_plan.items),
        has_noops=any(item.action == "noop" for item in merge_plan.items),
        source_only_noop=all(item.action == "noop" for item in merge_plan.items),
        requires_grounding_review=grounding_review.requires_review,
    )
    write_manifest_path = step_root / "draft_write_manifest.json"
    write_json(write_manifest_path, write_manifest_artifact)
    outputs.extend([draft_artifact_path, write_manifest_path])
    refs = [_draft_rendering_ref(ctx.run_dir, path, step_name) for path in outputs]
    refs.extend(structured_model_output_refs(ctx.run_dir, step_root, step_name))
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
        raise PipelineError("needs_human_decision must be revised to create/update/noop before validation.")
    if not digest.ingest_candidates() and not any(item.source_basis.prepared_discovered_candidates for item in merge_plan.items):
        raise PipelineError("source_digest must include at least one wiki candidate")
    if not merge_plan.items:
        raise PipelineError("wiki_merge_plan must include at least one action")
    if not draft_write_manifest.targets:
        raise PipelineError("draft_write_manifest must include at least one target")
    complete_step(ctx.manifest, step_name)


def _run_apply_preview(ctx: StepRunContext) -> None:
    step_name = "apply_preview"
    snapshot = read_model(require_step_output_dir(ctx.run_dir, "wiki_context_snapshot") / "wiki_context_snapshot.json", WikiContextSnapshot)
    ensure_wiki_context_current(ctx.vault, snapshot)
    preview = build_apply_preview(ctx.vault, ctx.run_dir)
    out = require_step_output_dir(ctx.run_dir, step_name) / "apply_preview.json"
    write_json(out, preview)
    complete_step(ctx.manifest, step_name, outputs=[_ref(ctx.run_dir, out, step_name, "json", "apply_preview.v2")])


def _run_draft_review(ctx: StepRunContext) -> None:
    step_name = "draft_review"
    step_root = require_step_output_dir(ctx.run_dir, step_name)
    draft_manifest_path = require_step_output_dir(ctx.run_dir, "draft_rendering") / "draft_write_manifest.json"
    draft_manifest = read_model(draft_manifest_path, DraftWriteManifest)
    approved_manifest_path = step_root / "approved_write_manifest.json"
    approval_path = step_root / "draft_approval.json"
    prompt_path = step_root / "review_prompt.md"
    prompt_path.write_text(render_draft_review_prompt(ctx.run_dir, draft_manifest), encoding="utf-8")
    if draft_manifest.source_only_noop:
        approved_manifest_path.write_text(draft_manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
        approval = build_draft_approval(
            ctx.run_dir,
            approved_manifest_path,
            decision="approved",
            review_mode="not_required",
            auto_approved=True,
            notes="全 noop operation：来源会被记录，但没有知识页变化。",
        )
        write_json(approval_path, approval)
        complete_review_step(
            ctx.manifest,
            step_name,
            outputs=[
                _ref(ctx.run_dir, prompt_path, step_name, "markdown"),
                _ref(ctx.run_dir, approved_manifest_path, step_name, "json", "draft_write_manifest.v1"),
                _ref(ctx.run_dir, approval_path, step_name, "json", "draft_review.v1"),
            ],
            review_decision_ref=approval_path.relative_to(ctx.run_dir).as_posix(),
        )
        return
    if not draft_manifest.has_updates and not draft_manifest.requires_grounding_review:
        approved_manifest_path.write_text(draft_manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
        approval = build_draft_approval(
            ctx.run_dir,
            approved_manifest_path,
            decision="approved",
            review_mode="auto_stub",
            auto_approved=True,
            notes="纯 create operation，本轮按 auto-stub 自动批准。",
        )
        write_json(approval_path, approval)
        complete_review_step(
            ctx.manifest,
            step_name,
            outputs=[
                _ref(ctx.run_dir, prompt_path, step_name, "markdown"),
                _ref(ctx.run_dir, approved_manifest_path, step_name, "json", "draft_write_manifest.v1"),
                _ref(ctx.run_dir, approval_path, step_name, "json", "draft_review.v1"),
            ],
            review_decision_ref=approval_path.relative_to(ctx.run_dir).as_posix(),
        )
        return
    pending_manifest = step_root / "pending_write_manifest.json"
    pending_manifest.write_text(draft_manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
    approval = DraftApproval(
        decision="pending",
        review_mode="manual",
        auto_approved=False,
        notes="Update 草稿或 grounding review 需要显式人工批准。",
    )
    write_json(approval_path, approval)
    review_reason = "Grounding review 发现 unsupported new_fact，需要人工确认。" if draft_manifest.requires_grounding_review else "Update 草稿需要显式人工批准。"
    mark_step_awaiting_review(
        ctx.manifest,
        step_name,
        outputs=[
            _ref(ctx.run_dir, prompt_path, step_name, "markdown"),
            _ref(ctx.run_dir, pending_manifest, step_name, "json", "draft_write_manifest.v1"),
            _ref(ctx.run_dir, approval_path, step_name, "json", "draft_review.v1"),
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


def render_source_digest_markdown(digest: SourceDigestArtifact) -> str:
    sections = [
        "# 来源消化",
        "",
        f"- 原始材料: `{digest.source_raw_path}`",
        f"- 摘要: {digest.summary}",
        "",
        "## 关键收获",
        "",
        "\n".join(f"- {item}" for item in digest.key_takeaways) or "- 暂无关键收获记录。",
    ]
    for title, candidates in [
        ("实体", digest.entities),
        ("概念", digest.concepts),
        ("设计", digest.designs),
        ("对比", digest.comparisons),
        ("未决问题", digest.open_questions),
        ("弱相关或噪声项", digest.weak_or_noise_items),
    ]:
        sections.extend(["", f"## {title}", "", render_candidate_table(candidates)])
    return "\n".join(sections).rstrip() + "\n"


def build_wiki_merge_plan(
    resolution: CandidateResolutionArtifact,
    digest: SourceDigestArtifact,
    snapshot: WikiContextSnapshot,
    *,
    log_date: str,
) -> WikiMergePlanArtifact:
    snapshot_by_path = {entry.path: entry for entry in snapshot.entries}
    candidates = {candidate.candidate_id: candidate for candidate in digest.ingest_candidates()}
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
        source_candidate_id = item.source_basis.source_candidate_ids[0] if item.source_basis.source_candidate_ids else ""
        candidate = candidates[source_candidate_id]
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


def existing_knowledge_page_paths(vault: Path) -> set[str]:
    return {entry.rel_path for entry in build_knowledge_pool(vault)}


def parse_frontmatter(text: str) -> dict[str, Any] | None:
    if not text.startswith("---\n"):
        return None
    parts = text.split("---\n", 2)
    if len(parts) < 3:
        return None
    data = yaml.safe_load(parts[1]) or {}
    return data if isinstance(data, dict) else None


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
        for candidate_id in other.source_basis.source_candidate_ids
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
        if len(related) >= FINAL_RELATED_LIMIT:
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
                reason=f"来源摘要把 `{raw}` 标记为相关候选，本页与该候选属于同一材料中的互补主题。",
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
                    display_title=clean_display_title(metadata.title),
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


def assert_system_page_can_be_overwritten(vault: Path, target_path: str) -> None:
    path = vault / target_path
    if not path.exists():
        return
    try:
        assert_current_system_page(path)
    except RuntimeError as exc:
        raise PipelineError(f"{exc}: {target_path}") from exc


def clean_display_title(title: str) -> str:
    stripped = title.strip()
    for prefix in ["Concept_", "Entity_", "Design_", "Comparison_", "Overview_", "Event_", "Memory_", "Idea_", "Open_Question_"]:
        if stripped.lower().startswith(prefix.lower()):
            return stripped[len(prefix) :].strip()
    return stripped


def yaml_scalar(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def render_preparation_review(preparation: RawPreparationArtifact) -> str:
    operations = "\n".join(f"- {operation}" for operation in preparation.operations_applied) or "- 暂无记录。"
    uncertain = "\n".join(
        f"- [{item.severity}] {item.item}: {item.reason}" for item in preparation.uncertain_items
    ) or "- 暂无记录。"
    return (
        "# Raw 清洗审核\n\n"
        f"- 原始材料: `{preparation.source_raw_path}`\n"
        f"- 文档类型: `{preparation.document_kind}`\n"
        f"- 风险等级: `{preparation.risk_level}`\n"
        f"- 是否需要人工审核: `{str(preparation.requires_human_review).lower()}`\n"
        f"- 省略策略: `{preparation.omission_policy}`\n\n"
        f"- Raw Wikilink 规范化: `{preparation.raw_link_cleanup_ref or 'raw_link_cleanup/raw_link_cleanup.json'}`\n"
        f"- 输入 raw hash: `{preparation.input_raw_sha256 or 'unknown'}`\n\n"
        "## 已执行操作\n\n"
        f"{operations}\n\n"
        "## 不确定项\n\n"
        f"{uncertain}\n\n"
        "## 审核备注\n\n"
        f"{preparation.review_notes or '暂无审核备注。'}\n"
    )


def resolve_vault_profile_name(vault: Path, profile_name: str | None) -> str:
    if profile_name:
        return profile_name
    config = read_yaml(vault / ".llmwiki" / "config.yaml")
    configured = config.get("profile")
    if not isinstance(configured, str) or not configured:
        raise PipelineError(".llmwiki/config.yaml profile must be a non-empty string.")
    return configured


def model_steps_from(start_step: str) -> list[str]:
    names = set(downstream_steps(start_step))
    return [step for step in MODEL_BACKED_STEPS if step in names]


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


def delete_downstream_step_dirs(
    vault: Path,
    operation_id: str,
    start_step: str,
    *,
    archive: bool = False,
    archive_reason: str = "",
) -> None:
    store = RunStore(vault)
    run_dir = store.run_dir(operation_id)
    if archive:
        archive_attempt_step_dirs(run_dir, start_step, archive_reason or "step artifacts archived before regeneration.")
    for step in downstream_steps(start_step):
        output_dir = step_output_dir(run_dir, step)
        if output_dir is not None:
            shutil.rmtree(output_dir, ignore_errors=True)


def archive_attempt_step_dirs(run_dir: Path, start_step: str, reason: str) -> Path | None:
    existing_steps = [
        step
        for step in downstream_steps(start_step)
        if (step_output_dir(run_dir, step) is not None and step_output_dir(run_dir, step).exists())
    ]
    if not existing_steps:
        return None
    archive_root = run_dir / "attempt_archive" / start_step / safe_timestamp()
    archive_root.mkdir(parents=True, exist_ok=True)
    for step in existing_steps:
        output_dir = step_output_dir(run_dir, step)
        if output_dir is None or not output_dir.exists():
            continue
        shutil.copytree(output_dir, archive_root / step)
    write_json(
        archive_root / "attempt_superseded.json",
        {
            "schema_version": "attempt_superseded.v1",
            "start_step": start_step,
            "superseded_at": utc_now(),
            "reason": reason,
            "archived_steps": existing_steps,
        },
    )
    return archive_root


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
        raise PipelineError("Cannot resume from raw_link_cleanup after it changed raw; rerun ingest or resume from raw_prepare.")


def last_attempt_duration_ms(manifest: OperationManifest, step_name: str) -> int | None:
    step = get_step(manifest, step_name)
    if not step.attempts:
        return None
    return step.attempts[-1].duration_ms


def step_completion_message(ctx: StepRunContext, step_name: str) -> str | None:
    if step_name != "raw_link_cleanup":
        return None
    path = require_step_output_dir(ctx.run_dir, "raw_link_cleanup") / "raw_link_cleanup.json"
    if not path.exists():
        return None
    cleanup = read_model(path, RawLinkCleanupArtifact)
    if not cleanup.changed:
        return None
    return f"raw 已规范化: {cleanup.raw_path}，清理 {cleanup.cleaned_link_count} 个 Obsidian 文本链接"


def refresh_run_metrics(vault: Path, run_dir: Path, manifest: OperationManifest, *, warning_console: Console | None = None) -> None:
    try:
        write_json(run_dir / "run_metrics.json", build_run_metrics(vault, run_dir, manifest))
    except Exception as exc:
        if warning_console is not None:
            warning_console.print(f"[yellow]warning:[/] run_metrics refresh failed: {exc}")


def build_run_metrics(vault: Path, run_dir: Path, manifest: OperationManifest) -> dict[str, Any]:
    steps = []
    model_durations: dict[str, int] = {}
    retry_count = 0
    internal_model_call_count = 0
    repair_count = 0
    repair_duration_ms = 0
    provider_result_count = 0
    for step in manifest.steps:
        durations = [attempt.duration_ms for attempt in step.attempts if attempt.duration_ms is not None]
        total = sum(durations)
        provider = step_provider_label(step.name, step.attempts[-1].provider_spec if step.attempts else None)
        retry_count += max(0, len(step.attempts) - 1)
        repair_metrics = step_repair_metrics(run_dir, step.name)
        internal_model_call_count += repair_metrics["attempt_count"]
        repair_count += repair_metrics["repair_count"]
        repair_duration_ms += repair_metrics["duration_ms"]
        provider_result_count += repair_metrics["provider_result_count"]
        row = {
            "name": step.name,
            "status": step.status.value,
            "attempts": len(step.attempts),
            "last_duration_ms": durations[-1] if durations else None,
            "total_duration_ms": total,
            "provider": provider,
        }
        if repair_metrics["attempt_count"]:
            row["internal_model_call_count"] = repair_metrics["attempt_count"]
            row["repair_count"] = repair_metrics["repair_count"]
            row["repair_duration_ms"] = repair_metrics["duration_ms"]
            row["provider_result_count"] = repair_metrics["provider_result_count"]
        waiting_ms = awaiting_review_duration_ms(step)
        if waiting_ms is not None:
            row["awaiting_review_duration_ms"] = waiting_ms
        steps.append(row)
        if step.attempts and step.attempts[-1].provider_spec:
            model_durations[step.name] = total
    cleanup_count = 0
    preserved_media_count = 0
    cleanup_path = run_dir / "raw_link_cleanup" / "raw_link_cleanup.json"
    if cleanup_path.exists():
        cleanup = read_model(cleanup_path, RawLinkCleanupArtifact)
        cleanup_count = cleanup.cleaned_link_count
        preserved_media_count = cleanup.preserved_media_embed_count
    created = updated = noop = 0
    plan_path = run_dir / "merge_plan_review" / "approved_merge_plan.json"
    if not plan_path.exists():
        plan_path = run_dir / "wiki_merge_planning" / "wiki_merge_plan.json"
    if plan_path.exists():
        plan = read_model(plan_path, WikiMergePlanArtifact)
        created = sum(1 for item in plan.items if item.action == "create")
        updated = sum(1 for item in plan.items if item.action == "update")
        noop = sum(1 for item in plan.items if item.action == "noop")
    written_target_count = 0
    preview_path = run_dir / "apply_preview" / "apply_preview.json"
    if preview_path.exists():
        preview = read_model(preview_path, ApplyPreview)
        written_target_count = len([target for target in preview.targets if target.will_write])
    return {
        "schema_version": "run_metrics.v1",
        "operation_id": manifest.operation_id,
        "status": manifest.status.value,
        "steps": steps,
        "model_durations_ms": model_durations,
        "retry_count": retry_count,
        "internal_model_call_count": internal_model_call_count,
        "repair_count": repair_count,
        "repair_duration_ms": repair_duration_ms,
        "provider_result_count": provider_result_count,
        "created_count": created,
        "updated_count": updated,
        "noop_count": noop,
        "cleaned_link_count": cleanup_count,
        "preserved_media_embed_count": preserved_media_count,
        "written_target_count": written_target_count,
    }


def step_repair_metrics(run_dir: Path, step_name: str) -> dict[str, int]:
    step_dirs = [run_dir / step_name]
    archive_root = run_dir / "attempt_archive"
    if archive_root.exists():
        for archived_step in sorted(archive_root.glob(f"*/**/{step_name}")):
            if archived_step.is_dir():
                step_dirs.append(archived_step)
    attempt_count = 0
    repair_count = 0
    duration_ms = 0
    provider_result_count = 0
    for step_dir in step_dirs:
        report_path = step_dir / "structured_repair_report.json"
        if report_path.exists():
            try:
                report = read_model(report_path, StructuredRepairReport)
                attempt_count += report.attempt_count
                repair_count += report.repair_count
                duration_ms += report.duration_ms
            except Exception:
                pass
        provider_results_dir = step_dir / "provider_results"
        if provider_results_dir.exists():
            provider_result_count += len(list(provider_results_dir.glob("attempt-*.json")))
        elif (step_dir / "provider_result.json").exists():
            provider_result_count += 1
    if attempt_count == 0:
        attempt_count = provider_result_count
    return {
        "attempt_count": attempt_count,
        "repair_count": repair_count,
        "duration_ms": duration_ms,
        "provider_result_count": provider_result_count,
    }


def step_provider_label(step_name: str, provider_spec: str | None) -> str:
    if provider_spec:
        return provider_spec
    if step_name.endswith("_review"):
        return "local:auto_review"
    return "local"


def awaiting_review_duration_ms(step: Any) -> int | None:
    if step.status != StepStatus.approved or not step.completed_at or not step.attempts:
        return None
    attempt = step.attempts[-1]
    if not attempt.completed_at:
        return None
    start = datetime.fromisoformat(attempt.completed_at)
    end = datetime.fromisoformat(step.completed_at)
    value = max(0, round((end - start).total_seconds() * 1000))
    return value if value > 0 else None


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
                raise PipelineError("draft_review has no pending write manifest to approve.")
            approved = step_root / "approved_write_manifest.json"
            approved.write_text(pending.read_text(encoding="utf-8"), encoding="utf-8")
            approval = build_draft_approval(
                run_dir,
                approved,
                decision="approved",
                review_mode="manual",
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
                    _ref(run_dir, approval_path, review_step, "json", "draft_review.v1"),
                ],
            )
            delete_downstream_step_dirs(vault, operation_id, "validation")
            mark_from_pending(manifest, "validation")
            write_manifest(store.manifest_path(operation_id), manifest)
            refresh_run_metrics(vault, run_dir, manifest)
            return manifest
        if review_step == "merge_plan_review":
            step_root = require_step_output_dir(run_dir, "merge_plan_review")
            pending = step_root / "pending_merge_plan.json"
            if not pending.exists():
                raise PipelineError("merge_plan_review has no pending merge plan to approve.")
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
                raise PipelineError("needs_human_decision must be revised to create/update/noop before approval.")
            approved = step_root / "approved_merge_plan.json"
            write_json(approved, plan)
            decision = ReviewDecision(
                review_step=review_step,
                decision="approved",
                review_mode="manual",
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
                    _ref(run_dir, decision_path, review_step, "json", "review_decision.v1"),
                ],
            )
            delete_downstream_step_dirs(vault, operation_id, "draft_rendering")
            mark_from_pending(manifest, "draft_rendering")
            write_manifest(store.manifest_path(operation_id), manifest)
            refresh_run_metrics(vault, run_dir, manifest)
            return manifest
        raise PipelineError(f"Unsupported review step: {review_step}")


def revise_review(vault: Path, operation_id: str, review_step: str) -> OperationManifest:
    store = RunStore(vault)
    with run_lock(vault, operation_id):
        manifest = read_manifest(store.manifest_path(operation_id))
        _require_review_step_awaiting(manifest, review_step)
        run_dir = store.run_dir(operation_id)
        require_upstream_artifacts_current(vault, run_dir, manifest, review_step)
        if review_step == "merge_plan_review":
            archive_pending_review_artifacts(run_dir, review_step)
            delete_downstream_step_dirs(vault, operation_id, "wiki_merge_planning")
            mark_from_pending(manifest, "wiki_merge_planning")
        elif review_step == "draft_review":
            archive_pending_review_artifacts(run_dir, review_step)
            delete_downstream_step_dirs(vault, operation_id, "draft_rendering")
            mark_from_pending(manifest, "draft_rendering")
        else:
            raise PipelineError(f"Unsupported review step: {review_step}")
        manifest.status = OperationStatus.running
        write_manifest(store.manifest_path(operation_id), manifest)
        refresh_run_metrics(vault, run_dir, manifest)
        return manifest


def archive_pending_review_artifacts(run_dir: Path, review_step: str) -> Path | None:
    step_root = run_dir / review_step
    if not step_root.exists():
        return None
    timestamp = safe_timestamp()
    archive_root = run_dir / "review_archive" / review_step / timestamp
    shutil.copytree(step_root, archive_root)
    write_json(
        archive_root / "superseded.json",
        {
            "schema_version": "review_superseded.v1",
            "review_step": review_step,
            "superseded_at": utc_now(),
            "reason": "revise requested; pending review artifacts were archived before downstream regeneration.",
        },
    )
    return archive_root


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
        raise PipelineError(f"upstream required artifacts changed before review: {preview}{suffix}")


def _require_review_step_awaiting(manifest: OperationManifest, review_step: str) -> None:
    if manifest.status in {OperationStatus.applied, OperationStatus.source_recorded}:
        raise PipelineError("Applied operations are immutable. Start a new operation instead.")
    if manifest.status == OperationStatus.apply_failed:
        raise PipelineError("apply_failed operations cannot be reviewed; inspect written targets and rerun ingest.")
    step = get_step(manifest, review_step)
    if step.status != StepStatus.awaiting_review:
        raise PipelineError(f"{review_step} is not awaiting_review; current status is {step.status.value}.")


def latest_operation(vault: Path) -> str | None:
    root = RunStore(vault).runs_root
    if not root.exists():
        return None
    candidates = sorted([path for path in root.iterdir() if path.is_dir() and (path / "manifest.json").exists()])
    return candidates[-1].name if candidates else None


def copy_fixture_raw(vault: Path, fixture_raw: Path) -> Path:
    target = vault / "raw" / fixture_raw.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(fixture_raw, target)
    return target


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
        "draft_write_manifest.json": "draft_write_manifest.v1",
        "update_merge_report.json": "update_merge_report.v1",
        "related_merge_report.json": "related_merge_report.v1",
        "draft_grounding_review.json": "draft_grounding_review.v1",
        "index_open_questions_report.json": "index_open_questions_report.v1",
    }
    return _ref(run_dir, path, step_name, artifact_kind_for_path(path), schemas.get(path.name))


M42_REQUIRED_DRAFT_SIDECARS = (
    "draft_rendering/draft_rendering.json",
    "draft_rendering/draft_write_manifest.json",
    "draft_rendering/provider_result.json",
    "draft_rendering/update_merge_report.json",
    "draft_rendering/update_merge_report.md",
    "draft_rendering/related_merge_report.json",
    "draft_rendering/related_merge_report.md",
    "draft_rendering/draft_grounding_review.json",
    "draft_rendering/draft_grounding_review.md",
)


def require_m42_draft_sidecars(run_dir: Path) -> None:
    missing = [rel_path for rel_path in M42_REQUIRED_DRAFT_SIDECARS if not (run_dir / rel_path).is_file()]
    if missing:
        preview = ", ".join(f"`{path}`" for path in missing[:6])
        suffix = "" if len(missing) <= 6 else f", ... and {len(missing) - 6} more"
        raise PipelineError(
            "M4.2 draft sidecar artifacts are missing; resume from draft_rendering or earlier before approve/apply: "
            f"{preview}{suffix}"
        )


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


# M3 helper implementations.


def render_candidate_table(candidates: list[SourceDigestCandidate] | list[WeakOrNoiseItem]) -> str:
    if not candidates:
        return "_暂无。_"
    return format_markdown_table(
        ["ID", "类型", "名称", "摘要", "重复风险"],
        [[f"`{item.candidate_id}`", item.type, item.name, item.one_sentence_summary, item.duplicate_risk] for item in candidates],
    )


def build_source_duplicate_guard_artifact(
    vault: Path,
    *,
    source_raw_path: str,
    source_raw_hash: str,
    source_prepared_hash: str,
    operation_id: str,
) -> SourceDuplicateGuardArtifact:
    normalized_raw_path = normalize_vault_path(source_raw_path)
    source_title = source_title_for_raw(normalized_raw_path)
    source_target_path = f"sources/{safe_filename(source_title)}.md"
    for source_page, frontmatter in scan_source_pages(vault):
        operation_ids = _frontmatter_list(frontmatter, "source_operation_ids")
        if operation_id in operation_ids:
            continue
        raw_paths = [normalize_vault_path(value) for value in _frontmatter_list(frontmatter, "source_raw_paths")]
        raw_hashes = _frontmatter_list(frontmatter, "source_raw_hashes")
        prepared_hashes = _frontmatter_list(frontmatter, "source_prepared_hashes")
        if normalized_raw_path in raw_paths and source_raw_hash in raw_hashes:
            return SourceDuplicateGuardArtifact(
                source_raw_path=normalized_raw_path,
                source_raw_hash=source_raw_hash,
                source_prepared_hash=source_prepared_hash,
                source_target_path=source_target_path,
                status="source_duplicate",
                matched_source_page=source_page,
                matched_raw_path=normalized_raw_path,
                matched_raw_hash=source_raw_hash,
                matched_prepared_hash=source_prepared_hash if source_prepared_hash in prepared_hashes else None,
                reason="same raw path and raw hash already recorded in source page frontmatter",
            )
        if source_raw_hash in raw_hashes:
            return SourceDuplicateGuardArtifact(
                source_raw_path=normalized_raw_path,
                source_raw_hash=source_raw_hash,
                source_prepared_hash=source_prepared_hash,
                source_target_path=source_target_path,
                status="source_duplicate",
                matched_source_page=source_page,
                matched_raw_hash=source_raw_hash,
                matched_prepared_hash=source_prepared_hash if source_prepared_hash in prepared_hashes else None,
                reason="same raw hash already recorded in source page frontmatter",
            )
        if normalized_raw_path in raw_paths and source_raw_hash not in raw_hashes:
            return SourceDuplicateGuardArtifact(
                source_raw_path=normalized_raw_path,
                source_raw_hash=source_raw_hash,
                source_prepared_hash=source_prepared_hash,
                source_target_path=source_target_path,
                status="source_revision_detected",
                matched_source_page=source_page,
                matched_raw_path=normalized_raw_path,
                reason="same raw path exists with a different content hash",
            )
    target = vault / "wiki" / source_target_path
    if target.exists():
        return SourceDuplicateGuardArtifact(
            source_raw_path=normalized_raw_path,
            source_raw_hash=source_raw_hash,
            source_prepared_hash=source_prepared_hash,
            source_target_path=source_target_path,
            status="source_duplicate",
            matched_source_page=f"wiki/{source_target_path}",
            reason="source target page already exists",
        )
    return SourceDuplicateGuardArtifact(
        source_raw_path=normalized_raw_path,
        source_raw_hash=source_raw_hash,
        source_prepared_hash=source_prepared_hash,
        source_target_path=source_target_path,
        status="clear",
        reason="no matching source path, hash, or source target page found",
    )


def render_source_duplicate_guard_markdown(artifact: SourceDuplicateGuardArtifact) -> str:
    return "\n".join(
        [
            "# 来源重复检查",
            "",
            format_markdown_table(
                ["字段", "值"],
                [
                    ["status", artifact.status],
                    ["source_raw_path", f"`{artifact.source_raw_path}`"],
                    ["source_raw_hash", f"`{artifact.source_raw_hash}`"],
                    ["source_prepared_hash", f"`{artifact.source_prepared_hash}`"],
                    ["source_target_path", f"`{artifact.source_target_path}`"],
                    ["matched_source_page", f"`{artifact.matched_source_page}`" if artifact.matched_source_page else ""],
                    ["reason", artifact.reason],
                ],
            ),
        ]
    ).rstrip() + "\n"


def scan_source_pages(vault: Path) -> list[tuple[str, dict[str, Any]]]:
    source_root = vault / "wiki" / "sources"
    if not source_root.exists():
        return []
    found: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(source_root.rglob("*.md")):
        frontmatter = parse_frontmatter(path.read_text(encoding="utf-8"))
        if frontmatter is not None:
            found.append((path.relative_to(vault).as_posix(), frontmatter))
    return found


def _frontmatter_list(frontmatter: dict[str, Any], key: str) -> list[str]:
    value = frontmatter.get(key, [])
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def normalize_vault_path(path: str) -> str:
    return unicodedata.normalize("NFC", path.strip()).replace("\\", "/")


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
    for group_name, candidate in digest_candidates_with_group(digest):
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


def digest_candidates_with_group(digest: SourceDigestArtifact) -> list[tuple[str, SourceDigestCandidate]]:
    return [
        *[("entities", item) for item in digest.entities],
        *[("concepts", item) for item in digest.concepts],
        *[("designs", item) for item in digest.designs],
        *[("comparisons", item) for item in digest.comparisons],
        *[("open_questions", item) for item in digest.open_questions],
    ]


def page_type_for_digest_candidate(group_name: str, candidate: SourceDigestCandidate, profile: Any) -> str:
    preferred = {
        "entities": "entity",
        "concepts": "concept",
        "designs": "design",
        "comparisons": "comparison",
        "open_questions": "open_question",
    }.get(group_name)
    if preferred in profile.page_types:
        return preferred
    normalized = candidate.type.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "question": "open_question",
        "open_questions": "open_question",
        "open_question": "open_question",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in profile.page_types and normalized != profile.source_page_type:
        return normalized
    return profile.default_page_type


def finalize_candidate_resolution(
    vault: Path,
    profile: Any,
    artifact: CandidateResolutionArtifact,
    digest: SourceDigestArtifact | None = None,
) -> CandidateResolutionArtifact:
    items: list[CandidateResolutionItem] = []
    seen_paths: dict[str, int] = {}
    weak_or_noise_ids = {candidate.candidate_id for candidate in digest.weak_or_noise_items} if digest is not None else set()
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
        leaked_ids = [
            candidate_id
            for candidate_id in item.source_basis.source_candidate_ids
            if candidate_id in weak_or_noise_ids or candidate_id.strip().lower().startswith(("noise", "weak", "ignore"))
        ]
        if leaked_ids:
            issues.append(
                StructuredIssue(
                    issue_code="weak_noise_candidate_reference",
                    field_path=f"items.{index}.source_basis.source_candidate_ids",
                    validator_id="finalize_candidate_resolution",
                    message=f"remove weak/noise candidate references from formal page plans: {sorted(leaked_ids)}",
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
        display_title = clean_display_title(item.display_title) or item.display_title.strip()
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
            item.why_this_page,
        ]
        for item in artifact.items
    ]
    return "# 候选页面规划\n\n" + format_markdown_table(
        ["页面计划", "类型", "标题", "目标", "来源候选", "为什么写"],
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
        raise PipelineError(str(exc)) from exc
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


def ensure_snapshot_within_limit(snapshot: WikiContextSnapshot, max_context_chars: int) -> None:
    total = sum(len(entry.content) for entry in snapshot.entries)
    if total > max_context_chars:
        raise PipelineError(f"wiki_context_snapshot exceeds max_context_chars ({total} > {max_context_chars}); retry with a smaller vault or higher limit.")


def read_wiki_page_metadata(vault: Path, rel_path: str) -> WikiPageMetadata | None:
    path = vault / rel_path
    if not path.exists() or path.suffix != ".md":
        return None
    frontmatter = parse_frontmatter(path.read_text(encoding="utf-8"))
    if frontmatter is None:
        return None
    llmwiki_type = frontmatter.get("llmwiki_type")
    title = frontmatter.get("title")
    summary = frontmatter.get("summary")
    created = frontmatter.get("created", "")
    updated = frontmatter.get("updated")
    if isinstance(created, (datetime, date)):
        created = created.isoformat()
    if isinstance(updated, (datetime, date)):
        updated = updated.isoformat()
    if not all(isinstance(value, str) and value.strip() for value in [llmwiki_type, title, summary, updated]):
        return None
    aliases_raw = frontmatter.get("aliases", [])
    aliases = [item for item in aliases_raw if isinstance(item, str)] if isinstance(aliases_raw, list) else []
    return WikiPageMetadata(
        path=rel_path.removeprefix("wiki/"),
        llmwiki_type=llmwiki_type,
        title=title,
        summary=summary,
        created=created if isinstance(created, str) else "",
        updated=updated,
        aliases=aliases,
        source_raw_paths=_frontmatter_list(frontmatter, "source_raw_paths"),
        source_raw_hashes=_frontmatter_list(frontmatter, "source_raw_hashes"),
        source_prepared_hashes=_frontmatter_list(frontmatter, "source_prepared_hashes"),
        source_operation_ids=_frontmatter_list(frontmatter, "source_operation_ids"),
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
        if len(related) >= FINAL_RELATED_LIMIT:
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
    for raw in _dedupe_strings([suggestion.target_path, suggestion.display_title]):
        target_path = _normalize_related_path(raw)
        if target_path:
            current = current_by_path.get(target_path)
            if current is not None and current.canonical_target_path != self_path:
                return RelatedPageRef(
                    target_path=current.canonical_target_path,
                    display_title=current.display_title,
                    source="source_digest",
                    reason=chinese_related_reason(fallback_reason, f"本次同源页面 `{current.display_title}` 与该主题互补。"),
                )
            metadata = metadata_by_path.get(target_path)
            if metadata is not None and metadata.path != self_path:
                return RelatedPageRef(
                    target_path=metadata.path,
                    display_title=clean_display_title(metadata.title),
                    source="wiki_context",
                    reason=chinese_related_reason(fallback_reason, f"召回旧页 `{clean_display_title(metadata.title)}` 与该主题存在可复用背景。"),
                )
        key = normalize_related_key(raw)
        current_matches = current_by_title.get(key, [])
        if len(current_matches) == 1 and current_matches[0].canonical_target_path != self_path:
            current = current_matches[0]
            return RelatedPageRef(
                target_path=current.canonical_target_path,
                display_title=current.display_title,
                source="source_digest",
                reason=chinese_related_reason(fallback_reason, f"本次同源页面 `{current.display_title}` 与该主题互补。"),
            )
        metadata_matches = metadata_by_title.get(key, [])
        if len(metadata_matches) == 1 and metadata_matches[0].path != self_path:
            metadata = metadata_matches[0]
            return RelatedPageRef(
                target_path=metadata.path,
                display_title=clean_display_title(metadata.title),
                source="wiki_context",
                reason=chinese_related_reason(fallback_reason, f"已有 wiki 页面标题或别名匹配 `{raw}`，可作为相关背景。"),
            )
    return None


def chinese_related_reason(model_reason: str, fallback: str) -> str:
    return fallback if not model_reason.strip() or looks_like_untranslated_english(model_reason) else model_reason.strip()


def _normalize_related_path(value: str) -> str | None:
    text = value.strip().strip("`").replace("\\", "/")
    if not text.endswith(".md"):
        return None
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        return None
    if path.parts and path.parts[0] == "wiki":
        path = Path(*path.parts[1:])
    if not path.parts or path.parts[0] in {"sources", "logs"} or path.as_posix() in {"index.md", "log.md"}:
        return None
    return path.as_posix()


def _related_debug_label(suggestion: RelatedPageRef) -> str:
    return f"{suggestion.display_title or '<untitled>'} -> {suggestion.target_path or '<no path>'}"


def _dedupe_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


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
            raise PipelineError(f"wiki_merge_plan references unknown page_plan_id: {item.page_plan_id}")
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
        canonical = normalize_model_wiki_target_path(item.canonical_target_path or resolution_item.candidate_target_path)
        matched_page = normalize_model_wiki_target_path(item.matched_page) if item.matched_page else None
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
            canonical = normalize_model_wiki_target_path(resolution_item.candidate_target_path)
            matched_page = None
        if f"wiki/{canonical}" not in snapshot_paths:
            raise PipelineError(f"wiki_merge_plan target is outside wiki_context_snapshot: wiki/{canonical}")
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
            if medium_missing_policy == "block":
                action = "needs_human_decision"
                apply_eligibility = "blocked"
                blocked_reason = blocked_reason or (
                    f"召回到中等相关旧页 `{strongest_overlap.path}`，但模型选择 create 的理由不充分；需要人工确认。"
                )
                finalization_notes.append("medium overlap create 的 why_not_update 不充分，被转为 needs_human_decision。")
            else:
                finalization_notes.append("medium overlap create 的 why_not_update 不充分，已请求模型补充。")
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
                    "finalization_reason": "；".join(_dedupe_strings([*finalization_notes, item.finalization_reason])) or "模型动作已按冻结 wiki context 校验。",
                    "canonical_target_path": canonical,
                    "matched_page": matched_page,
                    "inspected_context_paths": _dedupe_strings([*item.inspected_context_paths, *inspected_paths]),
                    "strongest_overlap": strongest_overlap,
                    "why_not_update": item.why_not_update,
                    "why_create_or_update": item.why_create_or_update or item.reason,
                    "related_absence_reason": related_absence_reason,
                    "page_plan_id": resolution_item.page_plan_id,
                    "source_basis": resolution_item.source_basis,
                    "page_type": resolution_item.page_type,
                    "display_title": clean_display_title(item.display_title or resolution_item.display_title),
                    "apply_eligibility": apply_eligibility,
                    "blocked_reason": blocked_reason,
                }
            )
        )
    items: list[WikiMergePlanItem] = []
    for item in preliminary:
        resolution_item = resolution_by_id[item.page_plan_id]
        related_pages, related_unresolved = resolve_model_related_pages(item, resolution_item, preliminary, resolution_by_id, snapshot)
        unresolved = _dedupe_strings([*item.related_unresolved, *item.unresolved_related, *related_unresolved])
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
    items = merge_same_source_duplicate_creates(items)
    items = merge_update_noop_same_targets(items)
    return WikiMergePlanArtifact(log_date=snapshot.log_date, items=items, context_snapshot_ref=snapshot_ref)


def merge_update_noop_same_targets(items: list[WikiMergePlanItem]) -> list[WikiMergePlanItem]:
    updates_by_target = {item.canonical_target_path: item for item in items if item.action == "update"}
    noop_by_target: dict[str, list[WikiMergePlanItem]] = {}
    for item in items:
        if item.action == "noop" and item.canonical_target_path in updates_by_target:
            noop_by_target.setdefault(item.canonical_target_path, []).append(item)
    if not noop_by_target:
        return items

    merged: list[WikiMergePlanItem] = []
    skipped_noops: set[str] = set()
    for item in items:
        if item.action == "noop" and item.canonical_target_path in updates_by_target:
            skipped_noops.add(item.page_plan_id)
            continue
        if item.action != "update" or item.canonical_target_path not in noop_by_target:
            merged.append(item)
            continue
        covered_noops = noop_by_target[item.canonical_target_path]
        source_basis = SourceBasis(
            source_candidate_ids=_dedupe_strings(
                [
                    *item.source_basis.source_candidate_ids,
                    *[
                        candidate_id
                        for noop in covered_noops
                        for candidate_id in noop.source_basis.source_candidate_ids
                    ],
                ]
            ),
            prepared_discovered_candidates=_dedupe_strings(
                [
                    *item.source_basis.prepared_discovered_candidates,
                    *[
                        candidate
                        for noop in covered_noops
                        for candidate in noop.source_basis.prepared_discovered_candidates
                    ],
                ]
            ),
            source_locator=item.source_basis.source_locator,
        )
        related_pages = [*item.related_pages]
        for noop in covered_noops:
            related_pages.extend(noop.related_pages)
        deduped_related: list[RelatedPageRef] = []
        seen_related: set[str] = set()
        for related in related_pages:
            path = normalize_related_candidate_path(related.target_path)
            if path is None or path in seen_related:
                continue
            seen_related.add(path)
            deduped_related.append(related.model_copy(update={"target_path": path}))
            if len(deduped_related) >= FINAL_RELATED_LIMIT:
                break
        noop_ids = [noop.page_plan_id for noop in covered_noops]
        merged_ids = _dedupe_strings([*item.merged_page_plan_ids, item.page_plan_id, *noop_ids])
        reason = f"同一 canonical target 出现 update + noop；{', '.join(noop_ids)} 已由 update `{item.page_plan_id}` 覆盖。"
        merged.append(
            item.model_copy(
                update={
                    "source_basis": source_basis,
                    "related_pages": deduped_related,
                    "merged_page_plan_ids": merged_ids,
                    "noop_covered_by_update": True,
                    "merge_reason": merge_markdown_blocks(item.merge_reason, reason),
                    "finalization_reason": merge_markdown_blocks(item.finalization_reason, reason),
                }
            )
        )
    # Preserve deterministic order while ensuring skipped noop ids are only represented in merged_page_plan_ids.
    return [item for item in merged if item.page_plan_id not in skipped_noops]


def merge_same_source_duplicate_creates(items: list[WikiMergePlanItem]) -> list[WikiMergePlanItem]:
    result = list(items)
    changed = True
    while changed:
        changed = False
        for left_index in range(len(result)):
            left = result[left_index]
            if left.action != "create":
                continue
            for right_index in range(left_index + 1, len(result)):
                right = result[right_index]
                if right.action != "create":
                    continue
                if not same_source_duplicate_create(left, right):
                    continue
                canonical, suppressed = choose_duplicate_canonical(left, right)
                merged = absorb_duplicate_create(canonical, suppressed)
                keep_index = left_index if canonical is left else right_index
                drop_index = right_index if canonical is left else left_index
                result[keep_index] = merged
                del result[drop_index]
                changed = True
                break
            if changed:
                break
    return result


def same_source_duplicate_create(left: WikiMergePlanItem, right: WikiMergePlanItem) -> bool:
    left_title_tokens = duplicate_tokens(left.display_title)
    right_title_tokens = duplicate_tokens(right.display_title)
    if len(left_title_tokens | right_title_tokens) < 2:
        return False
    title_overlap = jaccard(left_title_tokens, right_title_tokens)
    if title_overlap < 0.6 and not both_agent_workflow_compare(left.display_title, right.display_title):
        return False
    intent_overlap = jaccard(duplicate_tokens(duplicate_intent_text(left)), duplicate_tokens(duplicate_intent_text(right)))
    if intent_overlap < 0.48 and not both_agent_workflow_compare(left.display_title, right.display_title):
        return False
    if duplicate_shape_conflict(left, right) and title_overlap < 0.82:
        return False
    return True


def duplicate_tokens(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", text.lower())
    normalized = normalized.replace("workflow", "workflow").replace("workflows", "workflow")
    normalized = normalized.replace("agentic", "agent").replace("agents", "agent")
    tokens = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", normalized))
    stop = {"concept", "comparison", "design", "entity", "open", "question", "概念", "设计", "实体", "问题", "对比", "区别", "比较", "什么", "如何", "为什么", "页面"}
    return {token for token in tokens if token not in stop}


def duplicate_intent_text(item: WikiMergePlanItem) -> str:
    return "\n".join(
        [
            item.display_title,
            item.new_understanding,
            item.knowledge_delta,
            item.why_this_matters,
            item.reason,
            " ".join(item.value_points),
            "\n".join(item.section_plans.values()),
        ]
    )


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def both_agent_workflow_compare(left_title: str, right_title: str) -> bool:
    left = left_title.lower()
    right = right_title.lower()
    left_has_agent = "agent" in left or "智能体" in left
    left_has_workflow = "workflow" in left or "工作流" in left or "流程" in left
    right_has_agent = "agent" in right or "智能体" in right
    right_has_workflow = "workflow" in right or "工作流" in right or "流程" in right
    if not (left_has_agent and left_has_workflow and right_has_agent and right_has_workflow):
        return False
    combined = f"{left} {right}"
    has_compare = any(marker in combined for marker in ["vs", "对比", "比较", "区别"])
    return has_compare


def duplicate_shape_conflict(left: WikiMergePlanItem, right: WikiMergePlanItem) -> bool:
    pair = {left.page_type, right.page_type}
    if pair <= {"concept", "comparison"}:
        return False
    if pair <= {"concept", "open_question"}:
        return True
    if pair <= {"concept", "design"}:
        left_tokens = duplicate_tokens(left.display_title)
        right_tokens = duplicate_tokens(right.display_title)
        return jaccard(left_tokens, right_tokens) < 0.9
    return len(pair) > 1


def choose_duplicate_canonical(left: WikiMergePlanItem, right: WikiMergePlanItem) -> tuple[WikiMergePlanItem, WikiMergePlanItem]:
    ranked = sorted([left, right], key=duplicate_canonical_rank)
    return ranked[0], ranked[1]


def duplicate_canonical_rank(item: WikiMergePlanItem) -> tuple[int, int, str]:
    title = item.display_title.lower()
    type_rank = {
        "comparison": 0 if any(marker in title for marker in ["vs", "对比", "比较", "区别"]) else 2,
        "design": 1,
        "concept": 2,
        "open_question": 3,
        "entity": 4,
    }.get(item.page_type, 5)
    return (type_rank, -len(duplicate_intent_text(item)), item.canonical_target_path)


def absorb_duplicate_create(canonical: WikiMergePlanItem, suppressed: WikiMergePlanItem) -> WikiMergePlanItem:
    source_basis = SourceBasis(
        source_candidate_ids=_dedupe_strings([*canonical.source_basis.source_candidate_ids, *suppressed.source_basis.source_candidate_ids]),
        prepared_discovered_candidates=_dedupe_strings(
            [*canonical.source_basis.prepared_discovered_candidates, *suppressed.source_basis.prepared_discovered_candidates]
        ),
        source_locator=canonical.source_basis.source_locator or suppressed.source_basis.source_locator,
    )
    section_plans = dict(canonical.section_plans)
    for key, value in suppressed.section_plans.items():
        if key in section_plans:
            section_plans[key] = merge_markdown_blocks(section_plans[key], f"合并自 `{suppressed.page_plan_id}`：{value}")
        else:
            section_plans[key] = f"合并自 `{suppressed.page_plan_id}`：{value}"
    related_pages = [*canonical.related_pages, *suppressed.related_pages]
    merged_related: list[RelatedPageRef] = []
    seen_related: set[str] = set()
    for related in related_pages:
        path = normalize_related_candidate_path(related.target_path)
        if path is None or path in seen_related or path == canonical.canonical_target_path:
            continue
        seen_related.add(path)
        merged_related.append(related.model_copy(update={"target_path": path}))
        if len(merged_related) >= FINAL_RELATED_LIMIT:
            break
    absorbed_note = (
        f"同源近重复自动合并：`{suppressed.page_plan_id}`（{suppressed.display_title}）的信息已并入 "
        f"`{canonical.page_plan_id}`；其 section intent、examples/value points 通过 source_basis/section_plans 进入 canonical draft。"
    )
    return canonical.model_copy(
        update={
            "source_basis": source_basis,
            "section_plans": section_plans,
            "related_pages": merged_related,
            "value_points": _dedupe_strings([*canonical.value_points, *suppressed.value_points]),
            "reuse_scenarios": _dedupe_strings([*canonical.reuse_scenarios, *suppressed.reuse_scenarios]),
            "merged_page_plan_ids": _dedupe_strings(
                [*canonical.merged_page_plan_ids, canonical.page_plan_id, suppressed.page_plan_id, *suppressed.merged_page_plan_ids]
            ),
            "merge_reason": merge_markdown_blocks(canonical.merge_reason, absorbed_note),
            "finalization_reason": merge_markdown_blocks(canonical.finalization_reason, absorbed_note),
            "quality_risks": _dedupe_strings(
                [
                    *canonical.quality_risks,
                    *suppressed.quality_risks,
                    f"已自动合并 `{suppressed.page_plan_id}`；draft review 需确认被合并候选没有独特信息丢失。",
                ]
            ),
        }
    )


def normalize_model_wiki_target_path(value: str) -> str:
    path = value.strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    if path.startswith("/"):
        path = path[1:]
    if path.startswith("wiki/"):
        path = path.removeprefix("wiki/")
    return path


def ensure_wiki_context_current(vault: Path, snapshot: WikiContextSnapshot) -> None:
    messages = wiki_context_drift_messages(vault, snapshot)
    if messages:
        raise PipelineError("; ".join(messages))


def render_merge_plan_markdown(plan: WikiMergePlanArtifact) -> str:
    rows = []
    for item in plan.items:
        rows.append(
            [
                item.page_plan_id,
                item.model_action or item.action,
                item.action,
                item.display_title,
                f"`{item.canonical_target_path}`",
                f"{item.strongest_overlap.strength} `{item.strongest_overlap.path}`".strip(),
                item.why_not_update,
                item.new_understanding,
                item.apply_eligibility,
                item.blocked_reason,
            ]
        )
    return "# Wiki 合并计划\n\n" + format_markdown_table(
        ["页面计划", "模型动作", "最终动作", "标题", "目标", "最强召回", "为什么不更新旧页", "新增理解", "Apply", "阻断原因"],
        rows,
    ) + "\n"


def render_merge_plan_review_prompt(plan: WikiMergePlanArtifact) -> str:
    decision_rows = [
        [
            item.page_plan_id,
            item.model_action or item.action,
            item.action,
            item.display_title,
            f"`{item.canonical_target_path}`",
            item.strongest_overlap.strength,
            item.blocked_reason,
        ]
        for item in plan.items
    ]
    return (
        "# 合并计划审核\n\n"
        "审查这一步回答：写哪些页面、为什么写、哪些旧页已经被看过。\n\n"
        "## 核心判断\n\n"
        "- create 是否真的不能 update 到已召回的旧页？理由是否具体到范围、来源增量和边界？\n"
        "- update/noop 是否绑定了被召回并读过全文的旧页？\n"
        "- Related 是否少而准，是否只保留最相关的 0-2 条？\n"
        "- 是否存在 needs_human_decision 或 all-create 风险需要先 revise？\n\n"
        "## 关键文件\n\n"
        "- 合并计划：`wiki_merge_planning/wiki_merge_plan.json`\n"
        "- 合并报告：`wiki_merge_planning/merge_decision_report.md`\n"
        "- 召回上下文：`wiki_context_snapshot/candidate_contexts.md`\n"
        "- 可编辑文件：`merge_plan_review/pending_merge_plan.json`（仅 awaiting_review 时存在）\n\n"
        "## 下一步命令\n\n"
        "- 批准：`uv run llmwiki ingest approve \"$VAULT\" \"$OP\" merge_plan_review`\n"
        "- 重新生成/修订：`uv run llmwiki ingest revise \"$VAULT\" \"$OP\" merge_plan_review`\n"
        "- 批准后继续：`uv run llmwiki ingest resume \"$VAULT\" \"$OP\"`\n\n"
        "## 决策概览\n\n"
        + format_markdown_table(["页面计划", "模型动作", "最终动作", "标题", "目标", "最强召回", "阻断原因"], decision_rows)
        + "\n\n"
        "## 完整计划\n\n"
        f"{render_merge_plan_markdown(plan)}"
    )


def merge_plan_all_create_review_reason(plan: WikiMergePlanArtifact) -> str:
    if not plan.items or any(item.action != "create" for item in plan.items):
        return ""
    risky = [
        item
        for item in plan.items
        if item.strongest_overlap.strength in {"medium", "strong"}
    ]
    if not risky:
        return ""
    names = ", ".join(f"{item.page_plan_id}:{item.strongest_overlap.strength or 'none'}" for item in risky[:8])
    return f"合并计划全部为 create，但存在召回风险（{names}）；请审核这些页面为什么不应 update 到已有知识页。"


def render_candidate_contexts_markdown(
    artifact: CandidateContextsArtifact,
    *,
    resolved_cache_path: str = "",
    query_count: int | None = None,
    encoded_page_count: int | None = None,
) -> str:
    sections = [
        "# 候选页召回上下文",
        "",
        f"- 后端：`{artifact.retrieval_backend}`",
        f"- 模型：`{artifact.model}`",
        f"- 缓存路径（resolved cache path）：`{resolved_cache_path or artifact.cache_dir or '未使用'}`",
        f"- TopK：{artifact.top_k}",
        f"- 候选池页面数：{artifact.candidate_pool_size}",
        f"- 查询数（query count）：{artifact.candidate_pool_size if query_count is None else query_count}",
        f"- 编码页面数：{artifact.candidate_pool_size if encoded_page_count is None else encoded_page_count}",
        f"- 不完整 frontmatter 页面数：{artifact.skipped_count}",
        f"- 候选池 Hash：`{artifact.candidate_pool_sha256}`",
    ]
    if artifact.warnings:
        sections.extend(["", "## 警告", "", *[f"- {warning}" for warning in artifact.warnings]])
    for item in artifact.items:
        rows = [
            [
                str(hit.rank),
                hit.strength,
                hit.match_basis,
                f"{hit.score:.4f}",
                "`forced`" if hit.forced else "",
                f"`{hit.path}`",
                hit.display_title,
                "`truncated`" if hit.truncated else "",
                hit.excerpt[:180].replace("\n", " "),
            ]
            for hit in item.hits
        ]
        sections.extend(
            [
                "",
                f"## {item.page_plan_id}",
                "",
                f"查询文本: {item.query[:500]}",
                "",
                format_markdown_table(["排名", "强度", "依据", "分数", "强制命中", "路径", "标题", "截断", "片段"], rows)
                if rows
                else "未召回到候选旧页。",
            ]
        )
        if item.unindexable_pages:
            sections.extend(["", "Frontmatter 不完整但已进入低置信候选池：", "", *[f"- `{path}`" for path in item.unindexable_pages[:20]]])
    return "\n".join(sections).rstrip() + "\n"


def render_merge_decision_report(plan: WikiMergePlanArtifact, snapshot: WikiContextSnapshot) -> str:
    context_by_id = {item.page_plan_id: item for item in snapshot.candidate_contexts.items}
    sections = ["# 合并决策报告", ""]
    for item in plan.items:
        context = context_by_id.get(item.page_plan_id)
        inspected = item.inspected_context_paths or ([hit.path for hit in context.hits] if context else [])
        overlap_rows = []
        if context is not None:
            for hit in context.hits:
                if hit.strength in {"medium", "strong"}:
                    overlap_rows.append(
                        [
                            hit.rank,
                            hit.strength,
                            hit.match_basis,
                            f"{hit.score:.4f}",
                            f"`{hit.path}`",
                            hit.display_title,
                            hit.excerpt[:160].replace("\n", " "),
                        ]
                    )
        related_text = (
            ", ".join(f"`{related.target_path}`" for related in item.related_pages)
            if item.related_pages
            else f"无（{item.related_absence_reason or 'no_candidate'}）"
        )
        action_question = {
            "create": "为什么不 update",
            "update": "为什么 update",
            "noop": "为什么 noop",
            "needs_human_decision": "为什么需要人工决策",
        }.get(item.action, "为什么 create/update/noop")
        action_answer = {
            "create": item.why_not_update or "未提供",
            "update": item.why_create_or_update or item.reason,
            "noop": item.why_create_or_update or item.reason,
            "needs_human_decision": item.blocked_reason or item.why_create_or_update or item.reason,
        }.get(item.action, item.reason)
        sections.extend(
            [
                f"## {item.display_title}",
                "",
                f"- 页面计划：`{item.page_plan_id}`",
                f"- 模型动作：`{item.model_action or item.action}`",
                f"- 最终动作：`{item.action}`",
                f"- 目标：`{item.canonical_target_path}`",
                f"- 最像旧页：`{item.strongest_overlap.path or '无'}` ({item.strongest_overlap.strength}, {item.strongest_overlap.match_basis})",
                f"- 看过的旧页：{', '.join(f'`{path}`' for path in inspected) if inspected else '无'}",
                f"- {action_question}：{action_answer}",
                f"- 为什么 create/update/noop：{item.why_create_or_update or item.reason}",
                f"- Related：{related_text}",
                f"- Finalizer：{item.finalization_reason}",
            ]
        )
        if item.action == "create" and item.strongest_overlap.strength in {"medium", "strong"}:
            sections.extend(
                [
                    "",
                    "### Create 对比审计",
                    "",
                    "- scope_delta：见 why_not_update 中的新旧页面范围差异。",
                    "- source_delta：见 why_not_update 中的新材料增量。",
                    "- why_update_not_enough：见 why_not_update 中为什么整页更新不合适。",
                    "- why_related_link_not_enough：见 why_not_update 中为什么只做 Related 不够。",
                ]
            )
        if overlap_rows:
            sections.extend(
                [
                    "",
                    "### Medium/Strong 召回命中",
                    "",
                    format_markdown_table(["排名", "强度", "依据", "分数", "路径", "标题", "片段"], overlap_rows),
                ]
            )
        if item.blocked_reason:
            sections.append(f"- 阻断原因：{item.blocked_reason}")
        sections.append("")
    return "\n".join(sections).rstrip() + "\n"


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
            raise_draft_issue("missing_field", "draft_rendering page_plan_id must not be empty", field_path="pages.page_plan_id")
        item = plan_by_id.get(page.page_plan_id)
        if item is None:
            item = next((candidate for candidate in plan_by_id.values() if candidate.canonical_target_path == page.canonical_target_path), None)
        if item is None:
            raise_draft_issue(
                "unknown_page_plan_reference",
                f"draft_rendering references non-draftable page_plan_id: {page.page_plan_id}",
                field_path="pages.page_plan_id",
            )
        if item.page_plan_id in used_ids:
            raise_draft_issue(
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
                    "section_bodies": normalize_draft_section_bodies(page.section_bodies),
                }
            )
        )
    return DraftRenderingArtifact(pages=pages)


CANONICAL_DRAFT_SECTION_KEYS = ("summary", "detail", "examples", "value_points", "additional_notes", "open_questions")


def normalize_draft_section_bodies(section_bodies: dict[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for raw_key, raw_body in section_bodies.items():
        if not isinstance(raw_body, str):
            continue
        body = raw_body.strip()
        if not body:
            continue
        canonical_key = draft_section_key_alias(raw_key)
        if canonical_key is None:
            header = re.sub(r"\s+", " ", raw_key.strip()) or "Additional Notes"
            body = f"### {header}\n\n{body}"
            canonical_key = "detail"
        normalized[canonical_key] = merge_markdown_blocks(normalized.get(canonical_key, ""), body)
    return normalized


def draft_section_key_alias(raw_key: str) -> str | None:
    key = unicodedata.normalize("NFKC", raw_key)
    key = re.sub(r"^[#*\s`]+|[:：#*\s`]+$", "", key)
    key = re.sub(r"[-_]+", " ", key)
    key = re.sub(r"\s+", " ", key).strip().lower()
    aliases = {
        "summary": "summary",
        "摘要": "summary",
        "detail": "detail",
        "details": "detail",
        "understanding": "detail",
        "详情": "detail",
        "理解": "detail",
        "examples": "examples",
        "example": "examples",
        "cases": "examples",
        "case": "examples",
        "例子": "examples",
        "案例": "examples",
        "value points": "value_points",
        "value point": "value_points",
        "values": "value_points",
        "value": "value_points",
        "why this matters": "value_points",
        "advice": "value_points",
        "advice for pms": "value_points",
        "价值点": "value_points",
        "建议": "value_points",
        "additional notes": "additional_notes",
        "additional note": "additional_notes",
        "notes": "additional_notes",
        "observations": "additional_notes",
        "observation": "additional_notes",
        "freeform": "additional_notes",
        "free form": "additional_notes",
        "补充观察": "additional_notes",
        "补充": "additional_notes",
        "观察": "additional_notes",
        "open questions": "open_questions",
        "open question": "open_questions",
        "questions": "open_questions",
        "tensions": "open_questions",
        "conflicts": "open_questions",
        "uncertainties": "open_questions",
        "矛盾与未决问题": "open_questions",
        "未决问题": "open_questions",
    }
    if key in CANONICAL_DRAFT_SECTION_KEYS:
        return key
    return aliases.get(key)


def merge_markdown_blocks(existing: str, addition: str) -> str:
    existing = existing.strip()
    addition = addition.strip()
    if not existing:
        return addition
    if not addition:
        return existing
    return f"{existing}\n\n{addition}"


def validate_draft_rendering(artifact: DraftRenderingArtifact, plan: WikiMergePlanArtifact, *, language: str | None = None) -> None:
    allowed_sections = set(CANONICAL_DRAFT_SECTION_KEYS)
    required_sections = {"summary", "detail"}
    plan_by_id = {item.page_plan_id: item for item in plan.items}
    required_ids = {item.page_plan_id for item in plan.items if item.action in {"create", "update"}}
    actual_ids = {page.page_plan_id for page in artifact.pages}
    missing = required_ids - actual_ids
    if missing:
        raise_draft_issue("missing_page_plan_coverage", f"draft_rendering misses page_plan_id(s): {sorted(missing)}", field_path="pages")
    extra = actual_ids - required_ids
    if extra:
        raise_draft_issue("unknown_page_plan_reference", f"draft_rendering contains unexpected page_plan_id(s): {sorted(extra)}", field_path="pages")
    for page in artifact.pages:
        if not page.canonical_target_path.strip():
            raise_draft_issue("missing_field", f"{page.page_plan_id} canonical_target_path must not be empty", field_path="canonical_target_path")
        if not page.section_bodies:
            raise_draft_issue("missing_field", f"{page.page_plan_id} section_bodies must not be empty", field_path="section_bodies")
        section_keys = set(page.section_bodies)
        unknown_sections = section_keys - allowed_sections
        if unknown_sections:
            raise_draft_issue(
                "unsupported_section_key",
                f"{page.page_plan_id} section_bodies contains unsupported section key(s): {sorted(unknown_sections)}",
                field_path="section_bodies",
            )
        missing_sections = required_sections - section_keys
        if missing_sections:
            raise_draft_issue(
                "missing_field",
                f"{page.page_plan_id} section_bodies misses required section key(s): {sorted(missing_sections)}",
                field_path="section_bodies",
            )
        if not page.change_summary.strip():
            raise_draft_issue("missing_field", f"{page.page_plan_id} change_summary must not be empty", field_path="change_summary")
        if not page.source_coverage_notes.strip():
            raise_draft_issue("missing_field", f"{page.page_plan_id} source_coverage_notes must not be empty", field_path="source_coverage_notes")
        for section_key, body in page.section_bodies.items():
            if section_key in required_sections and not body.strip():
                raise_draft_issue(
                    "missing_field",
                    f"{page.page_plan_id} section {section_key} must not be empty",
                    field_path=f"section_bodies.{section_key}",
                )
            if "---\n" in body or body.lstrip().startswith("# ") or contains_source_graph_link(body):
                raise_draft_issue(
                    "forbidden_page_markdown",
                    f"{page.page_plan_id} section body contains forbidden page-level markdown",
                    field_path=f"section_bodies.{section_key}",
                )
            if language == "zh-CN" and looks_like_untranslated_english(body):
                raise_draft_issue(
                    "zh_cn_untranslated_user_text",
                    f"{page.page_plan_id} section {section_key} must be Chinese for zh-CN vault",
                    field_path=f"section_bodies.{section_key}",
                )
        if language == "zh-CN" and looks_like_untranslated_english(page.change_summary):
            raise_draft_issue(
                "zh_cn_untranslated_user_text",
                f"{page.page_plan_id} change_summary must be Chinese for zh-CN vault",
                field_path="change_summary",
            )
        if language == "zh-CN" and looks_like_untranslated_english(page.source_coverage_notes):
            raise_draft_issue(
                "zh_cn_untranslated_user_text",
                f"{page.page_plan_id} source_coverage_notes must be Chinese for zh-CN vault",
                field_path="source_coverage_notes",
            )
        plan_item = plan_by_id.get(page.page_plan_id)
        if plan_item is not None:
            validate_digestive_quality(page, plan_item)


def validate_digestive_quality(page: DraftPageItem, item: WikiMergePlanItem) -> None:
    summary = page.section_bodies.get("summary", "")
    detail = page.section_bodies.get("detail", "")
    examples = page.section_bodies.get("examples", "")
    values = page.section_bodies.get("value_points", "")
    notes = page.section_bodies.get("additional_notes", "")
    substantive_slots = [
        text
        for text in [detail, examples, values, notes]
        if is_substantive_digestive_text(text)
    ]
    if not substantive_slots or normalized_digest_text(summary) == normalized_digest_text(detail):
        raise_draft_issue(
            "thin_digestive_content",
            (
                f"{page.page_plan_id} must include concrete digested understanding: viewpoint, example, "
                "use scenario, boundary condition, or value point; it must not be only a source summary."
            ),
            field_path="section_bodies",
        )
    if item.action == "update" and not update_change_summary_is_specific(page.change_summary):
        raise_draft_issue(
            "thin_update_change_summary",
            f"{page.page_plan_id} update change_summary must explain what the new material补充/改变/澄清了旧理解。",
            field_path="change_summary",
        )


def is_substantive_digestive_text(text: str) -> bool:
    normalized = normalized_digest_text(text)
    if not normalized or any(marker in normalized for marker in ["暂无", "没有相关", "无相关", "n/a"]):
        return False
    return len(normalized) >= 24 or any(marker in text for marker in ["例如", "适用", "边界", "价值", "场景", "反例", "意味着", "可以用来"])


def normalized_digest_text(text: str) -> str:
    return re.sub(r"[\s\-*#`，。；;：:、,.!?！？（）()]+", "", text.strip().lower())


def update_change_summary_is_specific(text: str) -> bool:
    normalized = normalized_digest_text(text)
    if len(normalized) < 12:
        return False
    return any(marker in text for marker in ["补充", "改变", "澄清", "更新", "整合", "新增", "保留", "删除", "修正", "扩展"])


def raise_draft_issue(issue_code: str, message: str, *, field_path: str = "", repairable: bool = True) -> None:
    raise ContractValidationError(
        message,
        issues=[
            StructuredIssue(
                issue_code=issue_code,
                field_path=field_path,
                validator_id="validate_draft_rendering",
                message=message,
                repairability="repairable" if repairable else "non_repairable",
            )
        ],
    )


def contains_source_graph_link(text: str) -> bool:
    return text_contains_source_graph_link(text)


def snapshot_entry(snapshot: WikiContextSnapshot, path: str) -> WikiContextEntry:
    for entry in snapshot.entries:
        if entry.path == path:
            return entry
    raise PipelineError(f"snapshot missing path: {path}")


SECTION_TITLE_TO_KEY = {
    "摘要": "summary",
    "详情": "detail",
    "例子": "examples",
    "价值点": "value_points",
    "补充观察": "additional_notes",
    "相关页面": "related",
    "矛盾与未决问题": "open_questions",
    "Open Questions": "open_questions",
    "Tensions / Open Questions": "open_questions",
}


def parse_existing_sections(markdown: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current_key: str | None = None
    for line in markdown.splitlines():
        match = re.match(r"^##\s+(.+?)\s*$", line)
        if match:
            current_key = SECTION_TITLE_TO_KEY.get(match.group(1).strip())
            if current_key is not None:
                sections.setdefault(current_key, [])
            continue
        if current_key is not None:
            sections[current_key].append(line)
    return {key: "\n".join(value).strip() for key, value in sections.items()}


def is_empty_placeholder(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text)
    return not normalized or any(marker in normalized for marker in ["暂无", "没有相关", "无相关", "N/A"])


def merge_update_section(section_key: str, old: str, new: str) -> tuple[str, SectionMergeChange]:
    old = old.strip()
    new = new.strip()
    retained: list[str] = []
    added: list[str] = []
    removed: list[str] = []
    if old and new and (old == new or old in new):
        retained.append(old)
    if new and not is_empty_placeholder(new):
        added.append(new)
    if old and not retained and old != new and not is_empty_placeholder(old):
        removed.append(old)
    return (
        new,
        SectionMergeChange(
            section_key=section_key,
            retained=retained,
            added=added,
            removed=removed,
            removal_reason="模型完整重写后未逐字保留该旧内容；请在 draft review 中确认其知识价值已被吸收或可剪除。" if removed else "",
        ),
    )


def collect_grounding_claims(
    *,
    item: WikiMergePlanItem,
    page: DraftPageItem,
    existing_entry: WikiContextEntry,
    approved_raw_text: str,
    claims: list[GroundingClaim],
) -> None:
    for section_key, body in page.section_bodies.items():
        for quote in re.findall(r"[“\"]([^”\"]{6,})[”\"]", body):
            if section_key == "examples" and not explicit_direct_quote_context(body, quote):
                claims.append(
                    GroundingClaim(
                        page_plan_id=page.page_plan_id,
                        target_path=item.canonical_target_path,
                        section_key=section_key,
                        claim_type="inference",
                        text=quote,
                        support="inference",
                        action="kept",
                        reason="例子区的通用示例句按 illustrative example 处理，不要求 raw exact match。",
                    )
                )
                continue
            supported = quote in approved_raw_text or quote in existing_entry.content
            claims.append(
                GroundingClaim(
                    page_plan_id=page.page_plan_id,
                    target_path=item.canonical_target_path,
                    section_key=section_key,
                    claim_type="new_fact",
                    text=quote,
                    support="raw" if quote in approved_raw_text else ("existing_wiki" if quote in existing_entry.content else "unsupported"),
                    action="kept" if supported else "needs_review",
                    reason="直接引用必须在 raw 或已有 wiki 中 exact match。",
                )
            )
        unsupported_markers = ["被多个", "被广泛", "公认", "业界普遍", "多个社区"]
        for line in body.splitlines():
            text = line.strip(" -*")
            if not text or len(text) < 8:
                continue
            if any(marker in text for marker in unsupported_markers) and text not in approved_raw_text and text not in existing_entry.content:
                claims.append(
                    GroundingClaim(
                        page_plan_id=page.page_plan_id,
                        target_path=item.canonical_target_path,
                        section_key=section_key,
                        claim_type="new_fact",
                        text=text,
                        support="unsupported",
                        action="needs_review",
                        reason="新增外部背书/强事实未在 raw 或 inspected wiki 中出现。",
                    )
                )
    if item.action == "update" and existing_entry.content:
        claims.append(
            GroundingClaim(
                page_plan_id=page.page_plan_id,
                target_path=item.canonical_target_path,
                claim_type="retained_fact",
                text="旧页事实作为 existing wiki 背景参与更新。",
                support="existing_wiki",
                action="kept",
            )
        )


def explicit_direct_quote_context(body: str, quote: str) -> bool:
    index = body.find(quote)
    if index < 0:
        return False
    prefix = body[max(0, index - 24) : index]
    return any(marker in prefix for marker in ["原文", "直接引用", "引用", "他说", "她说", "对方说", "访谈中说"])


def build_draft_grounding_review(
    artifact: DraftRenderingArtifact,
    plan: WikiMergePlanArtifact,
    snapshot: WikiContextSnapshot,
    approved_raw_text: str,
) -> DraftGroundingReview:
    plan_by_id = {item.page_plan_id: item for item in plan.items}
    claims: list[GroundingClaim] = []
    for page in artifact.pages:
        item = plan_by_id.get(page.page_plan_id)
        if item is None:
            continue
        entry = snapshot_entry(snapshot, f"wiki/{page.canonical_target_path}")
        collect_grounding_claims(
            item=item,
            page=page,
            existing_entry=entry,
            approved_raw_text=approved_raw_text,
            claims=claims,
        )
    return draft_grounding_review_from_claims(claims)


def draft_grounding_review_from_claims(claims: list[GroundingClaim]) -> DraftGroundingReview:
    unsupported_new_facts = [
        claim
        for claim in claims
        if claim.claim_type == "new_fact" and claim.support == "unsupported" and claim.action == "needs_review"
    ]
    return DraftGroundingReview(
        unsupported_new_facts=unsupported_new_facts,
        claims=claims,
        requires_review=bool(unsupported_new_facts),
    )


def assemble_knowledge_page(
    *,
    item: WikiMergePlanItem,
    page: DraftPageItem,
    existing_entry: WikiContextEntry,
    raw_path: str,
    raw_hash: str,
    prepared_hash: str,
    operation_id: str,
    log_date: str,
    update_reports: list[UpdatePageMergeReport] | None = None,
    related_reports: list[RelatedCandidateReport] | None = None,
    grounding_claims: list[GroundingClaim] | None = None,
    known_related_paths: set[str] | None = None,
    approved_raw_text: str = "",
) -> str:
    existing_sections = parse_existing_sections(existing_entry.content)
    summary = page.section_bodies.get("summary") or item.new_understanding
    detail = page.section_bodies.get("detail") or item.knowledge_delta or item.new_understanding
    examples = page.section_bodies.get("examples") or "暂无相关例子记录。"
    values = page.section_bodies.get("value_points") or "\n".join(f"- {value}" for value in item.value_points) or "暂无明确价值点记录。"
    additional_notes = page.section_bodies.get("additional_notes", "").strip()
    questions = page.section_bodies.get("open_questions") or "暂无矛盾与未决问题记录。"
    metadata = existing_entry.metadata
    is_update = item.action == "update" and metadata is not None
    final_title = metadata.title if is_update and metadata.title else item.display_title
    aliases = list(metadata.aliases if metadata is not None else [])
    if is_update and item.display_title and item.display_title != final_title and item.display_title not in aliases:
        aliases.append(item.display_title)
    created = metadata.created if metadata is not None and metadata.created else log_date
    section_changes: list[SectionMergeChange] = []
    if is_update:
        summary, summary_change = merge_update_section("summary", existing_sections.get("summary", ""), summary)
        detail, detail_change = merge_update_section("detail", existing_sections.get("detail", ""), detail)
        examples, examples_change = merge_update_section("examples", existing_sections.get("examples", ""), examples)
        values, values_change = merge_update_section("value_points", existing_sections.get("value_points", ""), values)
        additional_notes, notes_change = merge_update_section("additional_notes", existing_sections.get("additional_notes", ""), additional_notes)
        questions, questions_change = merge_update_section("open_questions", existing_sections.get("open_questions", ""), questions)
        section_changes.extend([summary_change, detail_change, examples_change, values_change, notes_change, questions_change])
    additional_notes_section = f"## 补充观察\n\n{additional_notes}\n\n" if additional_notes else ""
    related = render_related_pages(
        item,
        existing_entry=existing_entry,
        report_list=related_reports,
        known_paths=known_related_paths,
    )
    if update_reports is not None and is_update:
        update_reports.append(
            UpdatePageMergeReport(
                page_plan_id=item.page_plan_id,
                target_path=item.canonical_target_path,
                old_title=metadata.title,
                final_title=final_title,
                model_title=item.display_title,
                retained_title=final_title == metadata.title,
                merged_page_plan_ids=item.merged_page_plan_ids,
                noop_covered_by_update=item.noop_covered_by_update,
                sections=section_changes,
            )
        )
    if grounding_claims is not None:
        collect_grounding_claims(
            item=item,
            page=page,
            existing_entry=existing_entry,
            approved_raw_text=approved_raw_text,
            claims=grounding_claims,
        )
    source_raw_paths = _append_unique(metadata.source_raw_paths if metadata is not None else [], raw_path)
    source_raw_hashes = _append_unique(metadata.source_raw_hashes if metadata is not None else [], raw_hash)
    source_prepared_hashes = _append_unique(metadata.source_prepared_hashes if metadata is not None else [], prepared_hash)
    source_operation_ids = _append_unique(metadata.source_operation_ids if metadata is not None else [], operation_id)
    return (
        "---\n"
        f"llmwiki_type: {item.page_type}\n"
        f"title: {yaml_scalar(final_title)}\n"
        f"{_yaml_list('aliases', aliases)}"
        f"summary: {yaml_scalar(summary)}\n"
        f"created: {created}\n"
        f"updated: {log_date}\n"
        f"{_yaml_list('source_raw_paths', source_raw_paths)}"
        f"{_yaml_list('source_raw_hashes', source_raw_hashes)}"
        f"{_yaml_list('source_prepared_hashes', source_prepared_hashes)}"
        f"{_yaml_list('source_operation_ids', source_operation_ids)}"
        f"last_ingest_operation: {yaml_scalar(operation_id)}\n"
        "---\n\n"
        f"# {final_title}\n\n"
        "## 摘要\n\n"
        f"{summary}\n\n"
        "## 详情\n\n"
        f"{detail}\n\n"
        "## 例子\n\n"
        f"{examples}\n\n"
        "## 价值点\n\n"
        f"{values}\n\n"
        f"{additional_notes_section}"
        "## 相关页面\n\n"
        f"{related}\n\n"
        "## 矛盾与未决问题\n\n"
        f"{questions}\n"
    )


def _append_unique(existing: list[str], value: str) -> list[str]:
    items: list[str] = []
    for item in [*existing, value]:
        if item and item not in items:
            items.append(item)
    return items


def _yaml_list(key: str, values: list[str]) -> str:
    if not values:
        return f"{key}: []\n"
    return f"{key}:\n" + "".join(f"  - {yaml_scalar(value)}\n" for value in values)


def render_source_page(
    *,
    title: str,
    digest: SourceDigestArtifact,
    operation_id: str,
    linked_pages: list[str],
    touched_pages: list[str],
    no_change_pages: list[str],
    log_date: str,
    raw_hash: str,
    prepared_hash: str,
    cleanup: RawLinkCleanupArtifact,
) -> str:
    links = "\n".join(f"- `{path}`" for path in linked_pages) or "- 暂无派生知识页。"
    no_change = (
        "未改动页面：\n" + "\n".join(f"- `{path}`" for path in no_change_pages)
        if no_change_pages
        else "暂无未写入页面。"
    )
    summary = neutralize_markdown_links(digest.summary)
    takeaways = "\n".join(f"- {neutralize_markdown_links(item)}" for item in digest.key_takeaways) or "- 暂无关键收获记录。"
    return (
        "---\n"
        "llmwiki_type: source\n"
        f"title: {yaml_scalar(title)}\n"
        "aliases: []\n"
        f"summary: {yaml_scalar(summary)}\n"
        f"created: {log_date}\n"
        f"updated: {log_date}\n"
        "source_raw_paths:\n"
        f"  - {yaml_scalar(digest.source_raw_path)}\n"
        "source_raw_hashes:\n"
        f"  - {yaml_scalar(raw_hash)}\n"
        "source_prepared_hashes:\n"
        f"  - {yaml_scalar(prepared_hash)}\n"
        "source_operation_ids:\n"
        f"  - {yaml_scalar(operation_id)}\n"
        f"raw_cleanup_pre_sha256: {yaml_scalar(cleanup.pre_cleanup_sha256)}\n"
        f"raw_cleanup_post_sha256: {yaml_scalar(cleanup.post_cleanup_sha256)}\n"
        f"raw_cleanup_rule_version: {yaml_scalar(cleanup.cleanup_rule_version)}\n"
        f"raw_cleanup_artifact_ref: {yaml_scalar('raw_link_cleanup/raw_link_cleanup.json')}\n"
        f"raw_cleanup_changed: {str(cleanup.changed).lower()}\n"
        f"raw_cleanup_cleaned_link_count: {cleanup.cleaned_link_count}\n"
        f"raw_cleanup_diff_ref: {yaml_scalar('raw_link_cleanup/cleanup.diff')}\n"
        f"last_ingest_operation: {yaml_scalar(operation_id)}\n"
        "---\n\n"
        f"# {title}\n\n"
        "## 摘要\n\n"
        f"{summary}\n\n"
        "## 原始材料\n\n"
        f"- `{digest.source_raw_path}`\n\n"
        "## 关键收获\n\n"
        f"{takeaways}\n\n"
        "## 派生知识页\n\n"
        f"{links}\n\n"
        "## 未写入说明\n\n"
        f"{no_change}\n"
    )


def neutralize_markdown_links(text: str) -> str:
    def wiki_repl(match: re.Match[str]) -> str:
        label = match.group(1).replace("|", " / ").strip()
        return f"`{label}`" if label else ""

    def markdown_repl(match: re.Match[str]) -> str:
        label = match.group(1).strip()
        target = match.group(2).strip()
        if not label:
            return f"`{target}`"
        return f"{label} (`{target}`)" if target else label

    text = re.sub(r"\[\[([^\]]+)\]\]", wiki_repl, text)
    return re.sub(r"\[([^\]]*)\]\(([^)]+)\)", markdown_repl, text)


def build_index_rows(profile: Any, plan: WikiMergePlanArtifact, draft: DraftRenderingArtifact, snapshot: WikiContextSnapshot) -> list[dict[str, str]]:
    rows_by_path: dict[str, dict[str, str]] = {}
    for pool_entry in snapshot.knowledge_metadata_pool:
        metadata = pool_entry.metadata
        if metadata is None or metadata.llmwiki_type.lower() == "source":
            continue
        rows_by_path[metadata.path] = {
            "title": clean_display_title(metadata.title),
            "page": obsidian_link(metadata.path),
            "type": metadata.llmwiki_type,
            "summary": metadata.summary,
            "updated": metadata.updated,
        }
    page_by_id = {page.page_plan_id: page for page in draft.pages}
    for item in plan.items:
        if item.action not in {"create", "update"}:
            continue
        if item.page_type.lower() == "source":
            continue
        page = page_by_id.get(item.page_plan_id)
        summary = page.section_bodies.get("summary") if page else item.new_understanding
        title = item.display_title
        if item.action == "update":
            entry = next((entry for entry in snapshot.entries if entry.path == f"wiki/{item.canonical_target_path}"), None)
            if entry is not None and entry.metadata is not None:
                title = clean_display_title(entry.metadata.title)
        rows_by_path[item.canonical_target_path] = {
            "title": title,
            "page": obsidian_link(item.canonical_target_path),
            "type": item.page_type,
            "summary": summary or item.new_understanding,
            "updated": plan.log_date,
        }
    type_order = list(profile.page_types)
    rows = list(rows_by_path.values())
    rows.sort(key=lambda row: row["page"])
    rows.sort(key=lambda row: row["updated"], reverse=True)
    rows.sort(key=lambda row: type_order.index(row["type"]) if row["type"] in type_order else len(type_order))
    return rows


def build_open_question_rows(plan: WikiMergePlanArtifact, draft: DraftRenderingArtifact, snapshot: WikiContextSnapshot) -> list[dict[str, str]]:
    rows, _ = build_open_question_rows_with_report(plan, draft, snapshot)
    return rows


def build_open_question_rows_with_report(
    plan: WikiMergePlanArtifact,
    draft: DraftRenderingArtifact,
    snapshot: WikiContextSnapshot,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    candidates: list[dict[str, str]] = []
    for entry in snapshot.entries:
        metadata = entry.metadata
        if entry.expected_state != "present" or metadata is None or metadata.llmwiki_type.lower() == "source":
            continue
        for question in extract_open_questions(entry.content):
            candidates.append({
                "question": question,
                "page": obsidian_link(metadata.path, clean_display_title(metadata.title)),
                "path": metadata.path,
                "updated": metadata.updated,
                "page_type": metadata.llmwiki_type,
                "source": "existing_wiki",
            })
    plan_by_id = {item.page_plan_id: item for item in plan.items}
    for page in draft.pages:
        item = plan_by_id.get(page.page_plan_id)
        if item is None:
            continue
        for question in meaningful_open_question_lines(page.section_bodies.get("open_questions", "")):
            candidates.append({
                "question": question,
                "page": obsidian_link(item.canonical_target_path, item.display_title),
                "path": item.canonical_target_path,
                "updated": plan.log_date,
                "page_type": item.page_type,
                "source": "draft",
            })
    by_key: dict[str, list[dict[str, str]]] = {}
    for candidate in candidates:
        by_key.setdefault(open_question_key(candidate["question"]), []).append(candidate)
    rows: list[dict[str, str]] = []
    report_items: list[dict[str, Any]] = []
    for key, grouped in sorted(by_key.items()):
        representative = max(grouped, key=lambda item: item["updated"])
        low_signal = is_low_signal_open_question(representative["question"])
        repeated_gap = len(grouped) >= 2 and low_signal
        keep = (
            any(item["page_type"] == "open_question" for item in grouped)
            or repeated_gap
            or not low_signal
        )
        pages = _dedupe_strings([item["page"] for item in sorted(grouped, key=lambda item: item["updated"], reverse=True)])[:3]
        decision = "kept" if keep else "filtered"
        reason = "open_question_page" if any(item["page_type"] == "open_question" for item in grouped) else ""
        if not reason:
            reason = "repeated_source_gap" if repeated_gap else ("low_signal_or_source_gap" if low_signal else "high_signal")
        report_items.append(
            {
                "normalized_key": key,
                "question": representative["question"],
                "decision": decision,
                "reason": reason,
                "pages": pages,
                "occurrences": len(grouped),
            }
        )
        if not keep:
            continue
        rows.append(
            {
                "question": representative["question"],
                "page": ", ".join(pages),
                "updated": max(item["updated"] for item in grouped),
            }
        )
    rows.sort(key=lambda row: (row["updated"], row["page"], row["question"]), reverse=True)
    return rows, {
        "schema_version": "index_open_questions_report.v1",
        "kept_count": sum(1 for item in report_items if item["decision"] == "kept"),
        "filtered_count": sum(1 for item in report_items if item["decision"] == "filtered"),
        "items": report_items,
    }


def open_question_key(question: str) -> str:
    text = re.sub(r"^\s*[-*]\s+", "", question.strip())
    text = re.sub(r"^(待补来源|待补充来源|需要来源|缺少来源)\s*[:：]\s*", "", text)
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[\s，。；;：:、,.!?！？（）()【】\[\]\"'“”‘’]+", "", text)


def is_low_signal_open_question(question: str) -> bool:
    normalized = open_question_key(question)
    if len(normalized) < 10:
        return True
    low_signal_markers = ["待补来源", "待补充来源", "需要来源", "缺少来源", "source needed", "citation needed"]
    if any(marker in question.lower() for marker in low_signal_markers):
        return True
    source_gap_markers = ["具体引用", "具体来源", "出处", "引用链接", "原始证据"]
    return any(marker in question for marker in source_gap_markers)


def render_index_open_questions_report(report: dict[str, Any]) -> str:
    rows = [
        [
            item["decision"],
            item["reason"],
            item["question"],
            ", ".join(item["pages"]),
            str(item["occurrences"]),
        ]
        for item in report.get("items", [])
    ]
    return (
        "# Index 未决问题筛选报告\n\n"
        f"- 保留：{report.get('kept_count', 0)}\n"
        f"- 过滤：{report.get('filtered_count', 0)}\n\n"
        + (format_markdown_table(["决策", "原因", "问题", "关联页面", "次数"], rows) if rows else "暂无未决问题。")
        + "\n"
    )


def extract_open_questions(markdown: str) -> list[str]:
    match = re.search(r"(?ms)^##\s+矛盾与未决问题\s*$\n(?P<body>.*?)(?=^##\s+|\Z)", markdown)
    if not match:
        return []
    return meaningful_open_question_lines(match.group("body"))


def meaningful_open_question_lines(text: str) -> list[str]:
    results: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"^\s*[-*]\s+", "", raw_line).strip()
        line = line.strip("。；; ")
        if not line:
            continue
        normalized = re.sub(r"\s+", "", line)
        if any(marker in normalized for marker in ["暂无", "没有", "无未决", "无矛盾", "不适用", "N/A", "na"]):
            continue
        if len(normalized) < 4:
            continue
        results.append(line)
    return _dedupe_strings(results)


def render_related_pages(
    item: WikiMergePlanItem,
    *,
    existing_entry: WikiContextEntry | None = None,
    report_list: list[RelatedCandidateReport] | None = None,
    known_paths: set[str] | None = None,
) -> str:
    candidates: list[dict[str, Any]] = []
    for order, related in enumerate(item.related_pages):
        candidates.append(
            {
                "target_path": _strip_wiki_prefix(related.target_path),
                "display_title": related.display_title,
                "reason": chinese_related_reason(related.reason, "该页面与当前主题存在明确内容互补关系。"),
                "source": related.source,
                "priority": related_candidate_priority(related.source),
                "order": order,
            }
        )
    if existing_entry is not None and existing_entry.expected_state == "present":
        candidates.extend(parse_existing_related_candidates(existing_entry.content))

    candidates.sort(key=lambda candidate: (int(candidate.get("priority", 50)), int(candidate.get("order", 0))))
    rows: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        path = normalize_related_candidate_path(candidate["target_path"])
        title = candidate["display_title"].strip() or clean_display_title(Path(candidate["target_path"]).stem)
        reason = chinese_related_reason(candidate["reason"], "该页面与当前主题存在明确内容互补关系。")
        reject_reason = ""
        if path is None:
            reject_reason = "unknown_path"
        elif path == item.canonical_target_path:
            reject_reason = "self_link"
        elif path in seen:
            reject_reason = "duplicate"
        elif known_paths is not None and path not in known_paths:
            reject_reason = "unknown_path"
        elif len(rows) >= FINAL_RELATED_LIMIT:
            reject_reason = "cap_cutoff"
        if report_list is not None:
            report_list.append(
                RelatedCandidateReport(
                    page_plan_id=item.page_plan_id,
                    target_path=path or candidate["target_path"],
                    display_title=title,
                    reason=reason,
                    source=candidate["source"],
                    decision="cutoff" if reject_reason == "cap_cutoff" else ("filtered" if reject_reason else "kept"),
                    reject_reason=reject_reason,
                )
            )
        if reject_reason or path is None:
            continue
        seen.add(path)
        rows.append(f"- {obsidian_alias_link(path, title)}：{reason}")
    if not rows:
        if report_list is not None:
            report_list.append(
                RelatedCandidateReport(
                    page_plan_id=item.page_plan_id,
                    target_path=item.canonical_target_path,
                    display_title=item.display_title,
                    reason=item.related_absence_reason or "low_confidence",
                    source="absence_reason",
                    decision="filtered",
                    reject_reason=item.related_absence_reason or "low_confidence",
                )
            )
        return "- 暂无相关页面记录。"
    return "\n".join(rows)


def parse_existing_related_candidates(markdown: str) -> list[dict[str, str]]:
    match = re.search(r"(?ms)^##\s+相关页面\s*$\n(?P<body>.*?)(?=^##\s+|\Z)", markdown)
    if not match:
        return []
    candidates: list[dict[str, Any]] = []
    for line in match.group("body").splitlines():
        for link in re.finditer(r"\[\[([^\]]+)\]\]", line):
            target, _, alias = link.group(1).partition("|")
            path = normalize_related_candidate_path(target)
            order = len(candidates)
            candidates.append(
                {
                    "target_path": path or target.strip(),
                    "display_title": alias.strip() or clean_display_title(Path(target).stem),
                    "reason": "旧 Related 作为候选重新参与排序。",
                    "source": "existing_related",
                    "priority": 0,
                    "order": order,
                }
            )
    return candidates


def related_candidate_priority(source: str) -> int:
    return {
        "existing_related": 0,
        "exact_or_alias": 1,
        "source_digest": 2,
        "wiki_context": 3,
    }.get(source, 9)


def normalize_related_candidate_path(value: str) -> str | None:
    text = value.strip().strip("`").replace("\\", "/")
    if not text:
        return None
    text = text.split("#", 1)[0]
    while text.startswith("./"):
        text = text[2:]
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        return None
    if path.parts and path.parts[0] == "wiki":
        path = Path(*path.parts[1:])
    if not path.parts or path.parts[0] in {"sources", "logs"} or path.as_posix() in {"index.md", "log.md"}:
        return None
    if path.suffix != ".md":
        path = path.with_suffix(".md")
    return path.as_posix()


def _strip_wiki_prefix(value: str) -> str:
    path = Path(value.strip().replace("\\", "/"))
    if path.parts and path.parts[0] == "wiki":
        return Path(*path.parts[1:]).as_posix()
    return path.as_posix()


def render_update_merge_report(report: UpdateMergeReport) -> str:
    if not report.pages:
        return "# Update 合并报告\n\n本次没有 update 页面。\n"
    sections = ["# Update 合并报告", ""]
    for page in report.pages:
        sections.extend(
            [
                f"## {page.final_title or page.target_path}",
                "",
                f"- 页面计划：`{page.page_plan_id}`",
                f"- 目标：`{page.target_path}`",
                f"- 旧标题：{page.old_title or '无'}",
                f"- 模型标题：{page.model_title or '无'}",
                f"- 最终标题：{page.final_title or '无'}",
                f"- 保留旧标题：{'是' if page.retained_title else '否'}",
                f"- 合并计划页：{', '.join(f'`{item}`' for item in page.merged_page_plan_ids) if page.merged_page_plan_ids else '无'}",
                f"- noop 被 update 覆盖：{'是' if page.noop_covered_by_update else '否'}",
                "",
            ]
        )
        rows = [
            [
                change.section_key,
                "\n".join(change.retained) or "无",
                "\n".join(change.added) or "无",
                "\n".join(change.removed) or "无",
                change.removal_reason,
            ]
            for change in page.sections
        ]
        sections.append(
            format_markdown_table(["段落", "保留", "新增", "删除", "删除原因"], rows)
            if rows
            else "没有记录 section 级变更。"
        )
        sections.append("")
    return "\n".join(sections).rstrip() + "\n"


def render_related_merge_report(report: RelatedMergeReport) -> str:
    rows = [
        [
            item.page_plan_id,
            f"`{item.target_path}`",
            item.display_title,
            item.source,
            item.decision,
            item.reject_reason,
            item.reason,
        ]
        for item in report.candidates
    ]
    body = format_markdown_table(["页面计划", "目标", "标题", "来源", "决策", "过滤原因", "理由"], rows) if rows else "没有 Related 候选。"
    return "# 相关页面合并报告\n\n" + body + "\n"


def render_draft_grounding_review(review: DraftGroundingReview) -> str:
    summary = "需要人工确认" if review.requires_review else "通过"
    rows = [
        [
            claim.page_plan_id,
            f"`{claim.target_path}`",
            claim.section_key,
            claim.claim_type,
            claim.support,
            claim.action,
            claim.reason,
            claim.text[:240],
        ]
        for claim in review.claims
    ]
    unsupported_rows = [
        [
            claim.page_plan_id,
            f"`{claim.target_path}`",
            claim.section_key,
            claim.reason,
            claim.text[:240],
        ]
        for claim in review.unsupported_new_facts
    ]
    sections = [
        "# 草稿来源支撑审查",
        "",
        f"- 结果：{summary}",
        f"- 未支撑新增事实数量：{len(review.unsupported_new_facts)}",
        "",
        "## 需要确认的新事实",
        "",
        format_markdown_table(["页面计划", "目标", "段落", "原因", "文本"], unsupported_rows) if unsupported_rows else "暂无。",
        "",
        "## 全部分类",
        "",
        format_markdown_table(["页面计划", "目标", "段落", "类型", "支持", "处理", "原因", "文本"], rows) if rows else "暂无分类记录。",
    ]
    return "\n".join(sections).rstrip() + "\n"


def obsidian_link(path: str, title: str | None = None) -> str:
    target = Path(path)
    if target.parts and target.parts[0] == "wiki":
        target = Path(*target.parts[1:])
    return f"[[{target.with_suffix('').as_posix()}]]"


def obsidian_alias_link(path: str, title: str) -> str:
    target = Path(path)
    if target.parts and target.parts[0] == "wiki":
        target = Path(*target.parts[1:])
    return f"[[{target.with_suffix('').as_posix()}|{obsidian_link_label(title)}]]"


def obsidian_link_label(value: str) -> str:
    label = " ".join(value.replace("|", "/").replace("]", "").split())
    return label or "Untitled"


def render_update_diff(old: str, new: str, old_name: str, new_name: str) -> str:
    return "".join(
        unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=old_name,
            tofile=new_name,
        )
    )


def build_draft_approval(
    run_dir: Path,
    approved_manifest_path: Path,
    *,
    decision: Literal["approved", "pending", "rejected"],
    review_mode: Literal["auto_stub", "manual", "not_required"],
    auto_approved: bool,
    notes: str,
) -> DraftApproval:
    require_m42_draft_sidecars(run_dir)
    approved_manifest = read_model(approved_manifest_path, DraftWriteManifest)
    markdown_hashes: dict[str, str] = {}
    for target in approved_manifest.targets:
        draft = run_dir / target.draft_path
        if draft.suffix == ".md" and draft.exists():
            markdown_hashes[target.draft_path] = sha256_file(draft)
    for rel_path in M42_REQUIRED_DRAFT_SIDECARS:
        path = run_dir / rel_path
        markdown_hashes[rel_path] = sha256_file(path)
    return DraftApproval(
        decision=decision,
        review_mode=review_mode,
        auto_approved=auto_approved,
        approved_draft_json_sha256=sha256_file(approved_manifest_path),
        approved_markdown_sha256=markdown_hashes,
        notes=notes,
    )


def render_draft_review_prompt(run_dir: Path, draft_manifest: DraftWriteManifest) -> str:
    rows = [
        [
            target.action,
            f"`{target.target_path}`",
            f"`{target.draft_path}`",
            draft_diff_ref(run_dir, target),
            draft_change_summary(run_dir, target),
            target.expected_state,
            target.preimage_sha256 or "",
        ]
        for target in draft_manifest.targets
    ]
    return (
        "# 草稿审核\n\n"
        "审查这一步回答：具体写什么、是否应批准写入。\n\n"
        f"- 需要 Grounding 人工确认：{'是' if draft_manifest.requires_grounding_review else '否'}\n\n"
        "## 核心判断\n\n"
        "- create/update 的正文是否忠实于 raw 和已召回旧页？\n"
        "- update diff 是否符合你的理解，没有覆盖掉旧页中仍然重要的内容？\n"
        "- 未被来源支持的细节是否放在“矛盾与未决问题/待补来源”，而不是写成事实？\n"
        "- Related 是否少而准，单页主动连接不超过 3 条？\n\n"
        "## 下一步命令\n\n"
        "- 批准：`uv run llmwiki ingest approve \"$VAULT\" \"$OP\" draft_review`\n"
        "- 重新生成/修订：`uv run llmwiki ingest revise \"$VAULT\" \"$OP\" draft_review`\n"
        "- 批准后继续：`uv run llmwiki ingest resume \"$VAULT\" \"$OP\"`\n"
        "- Apply：`uv run llmwiki ingest apply \"$VAULT\" \"$OP\"`\n\n"
        "## 关键文件\n\n"
        "- Update 合并报告：`draft_rendering/update_merge_report.md`\n"
        "- Grounding 审查：`draft_rendering/draft_grounding_review.md`\n"
        "- Related 合并报告：`draft_rendering/related_merge_report.md`\n"
        "- 草稿目录：`draft_rendering/draft_pages/`\n"
        "- Diff 目录：`draft_rendering/diffs/`\n\n"
        "## 草稿清单\n\n"
        + format_markdown_table(
            ["动作", "目标", "草稿", "Diff", "变更摘要", "预期状态", "Preimage"],
            rows,
        )
        + "\n"
    )


def draft_diff_ref(run_dir: Path, target: DraftWriteTarget) -> str:
    if not target.page_plan_id:
        return ""
    diff = run_dir / "draft_rendering" / "diffs" / f"{target.page_plan_id}.diff"
    return f"`{diff.relative_to(run_dir).as_posix()}`" if diff.exists() else ""


def draft_change_summary(run_dir: Path, target: DraftWriteTarget) -> str:
    if not target.page_plan_id:
        return ""
    draft_json = run_dir / "draft_rendering" / "draft_rendering.json"
    if not draft_json.exists():
        return ""
    try:
        draft = read_model(draft_json, DraftRenderingArtifact)
    except Exception:
        return ""
    for page in draft.pages:
        if page.page_plan_id == target.page_plan_id:
            return page.change_summary
    return ""


def build_apply_preview(vault: Path, run_dir: Path) -> ApplyPreview:
    operation_id = run_dir.name
    manifest_path = require_step_output_dir(run_dir, "draft_review") / "approved_write_manifest.json"
    draft_manifest = read_model(manifest_path, DraftWriteManifest)
    target_paths = [item.target_path for item in draft_manifest.targets]
    if len(target_paths) != len(set(target_paths)):
        raise PipelineError("draft_write_manifest contains duplicate target_path values")
    targets: list[ApplyTarget] = []
    source_targets: list[str] = []
    log_targets: list[str] = []
    index_targets: list[str] = []
    for item in draft_manifest.targets:
        target_path = item.target_path
        current = vault / target_path
        current_sha = sha256_file(current) if current.exists() else None
        if item.action == "source":
            source_targets.append(target_path)
        if item.action in {"global_log", "daily_log"}:
            log_targets.append(target_path)
        if item.action == "index":
            index_targets.append(target_path)
        targets.append(
            ApplyTarget(
                action=item.action,
                target_path=target_path,
                draft_path=item.draft_path,
                expected_state=item.expected_state,
                preimage_sha256=item.preimage_sha256,
                current_sha256=current_sha,
                will_write=True,
                approved_draft_ref=manifest_path.relative_to(run_dir).as_posix(),
                page_plan_id=item.page_plan_id,
            )
        )
    write_set_payload = {
        "approved_manifest_sha256": sha256_file(manifest_path),
        "targets": [
            {
                "target_path": target.target_path,
                "draft_path": target.draft_path,
                "preimage_sha256": target.preimage_sha256,
                "expected_state": target.expected_state,
                "draft_sha256": sha256_file(run_dir / target.draft_path),
            }
            for target in targets
        ],
    }
    write_set_sha = sha256_bytes(json.dumps(write_set_payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return ApplyPreview(
        operation_id=operation_id,
        operation_applyable=bool(targets),
        requires_draft_review=draft_manifest.has_updates or draft_manifest.requires_grounding_review,
        has_updates=draft_manifest.has_updates,
        has_noops=draft_manifest.has_noops,
        blocked_reasons=[],
        write_set_sha256=write_set_sha,
        targets=targets,
        source_targets=source_targets,
        log_targets=log_targets,
        index_targets=index_targets,
    )
