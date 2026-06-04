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
    DraftPageItem,
    DraftRenderingArtifact,
    DraftWriteManifest,
    DraftWriteTarget,
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
    SourceBasis,
    SourceDuplicateGuardArtifact,
    SourceDigestArtifact,
    SourceDigestCandidate,
    StepStatus,
    VaultConfig,
    WeakOrNoiseItem,
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
    preparation, _ = _structured_call(ctx.run_dir, ctx.execution_context, step_name).run(
        step_name,
        payload,
        RawPreparationArtifact,
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
    provider_result = step_root / "provider_result.json"
    if provider_result.exists():
        outputs.append(_ref(ctx.run_dir, provider_result, step_name, "provider_result", "provider_result.v1"))
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
    complete_step(
        ctx.manifest,
        step_name,
        outputs=[
            _ref(ctx.run_dir, prompt, step_name, "markdown"),
            _ref(ctx.run_dir, feedback, step_name, "jsonl"),
            _ref(ctx.run_dir, decision_path, step_name, "json", "review_decision.v1"),
            _ref(ctx.run_dir, approved, step_name, "markdown"),
        ],
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
            ],
        },
    }
    digest, _ = _structured_call(ctx.run_dir, ctx.execution_context, step_name).run(
        step_name,
        payload,
        SourceDigestArtifact,
    )
    digest = _redacted_model(ctx, digest, SourceDigestArtifact)
    if digest.source_raw_path != raw_rel:
        raise PipelineError(f"source_digest source path mismatch: {digest.source_raw_path} != {raw_rel}")
    validate_source_digest(digest)
    out = step_root / "source_digest.json"
    write_json(out, digest)
    digest_md = step_root / "source_digest.md"
    digest_md.write_text(render_source_digest_markdown(digest), encoding="utf-8")
    outputs = [
        _ref(ctx.run_dir, out, step_name, "json", "source_digest.v2"),
        _ref(ctx.run_dir, digest_md, step_name, "markdown"),
    ]
    provider_result = step_root / "provider_result.json"
    if provider_result.exists():
        outputs.append(_ref(ctx.run_dir, provider_result, step_name, "provider_result", "provider_result.v1"))
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
    prompt.write_text(
        "# Source Digest 审核\n\n"
        "当前 MVP 自动批准 source digest；后续会加入 list/filter/show/diff/revise/approve。\n",
        encoding="utf-8",
    )
    feedback = step_root / "review_feedback.jsonl"
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
    complete_step(
        ctx.manifest,
        step_name,
        outputs=[
            _ref(ctx.run_dir, prompt, step_name, "markdown"),
            _ref(ctx.run_dir, feedback, step_name, "jsonl"),
            _ref(ctx.run_dir, decision_path, step_name, "json", "review_decision.v1"),
            _ref(ctx.run_dir, approved_json, step_name, "json", "source_digest.v2"),
            _ref(ctx.run_dir, approved_md, step_name, "markdown"),
        ],
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
    validate_source_digest(digest)
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
                "Use approved_digest candidates as the primary basis, but inspect approved_prepared for missed wiki-worthy topics.",
                "Put any newly discovered topic in prepared_discovered_candidates.",
                "Do not read or infer existing wiki state.",
                "Write all user-visible fields in Chinese unless retaining a stable domain term.",
                "Leave page_plan_id/path_stem/candidate_target_path empty if unsure; the engine will deterministically set them.",
            ],
        },
    }
    artifact, _ = _structured_call(ctx.run_dir, ctx.execution_context, step_name).run(
        step_name,
        payload,
        CandidateResolutionArtifact,
    )
    artifact = _redacted_model(ctx, artifact, CandidateResolutionArtifact)
    artifact = backfill_missing_candidate_resolution_items(artifact, digest, ctx.profile)
    artifact = finalize_candidate_resolution(ctx.vault, ctx.profile, artifact)
    validate_candidate_resolution(digest, artifact)
    out = step_root / "candidate_resolution.json"
    write_json(out, artifact)
    table = step_root / "candidate_resolution.md"
    table.write_text(render_candidate_resolution_markdown(artifact), encoding="utf-8")
    outputs = [
        _ref(ctx.run_dir, out, step_name, "json", "candidate_resolution.v3"),
        _ref(ctx.run_dir, table, step_name, "markdown"),
    ]
    provider_result = step_root / "provider_result.json"
    if provider_result.exists():
        outputs.append(_ref(ctx.run_dir, provider_result, step_name, "provider_result", "provider_result.v1"))
    complete_step(
        ctx.manifest,
        step_name,
        outputs=outputs,
    )


def _run_wiki_context_snapshot(ctx: StepRunContext) -> None:
    step_name = "wiki_context_snapshot"
    resolution = read_model(
        require_step_output_dir(ctx.run_dir, "candidate_resolution") / "candidate_resolution.json",
        CandidateResolutionArtifact,
    )
    digest = read_model(require_step_output_dir(ctx.run_dir, "source_digest_review") / "approved_digest.json", SourceDigestArtifact)
    source_title = source_title_for_raw(digest.source_raw_path)
    log_date = local_date()
    snapshot = build_wiki_context_snapshot(
        ctx.vault,
        resolution,
        log_date=log_date,
        source_target_path=f"sources/{safe_filename(source_title)}.md",
    )
    ensure_snapshot_within_limit(snapshot, ctx.manifest.vault_config_snapshot.max_context_chars)
    snapshot_path = require_step_output_dir(ctx.run_dir, step_name) / "wiki_context_snapshot.json"
    write_json(snapshot_path, snapshot)
    complete_step(
        ctx.manifest,
        step_name,
        outputs=[_ref(ctx.run_dir, snapshot_path, step_name, "json", "wiki_context_snapshot.v1")],
    )


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
    payload = {
        "approved_prepared_markdown": (require_step_output_dir(ctx.run_dir, "prepared_raw_review") / "approved_prepared.md").read_text(encoding="utf-8"),
        "approved_digest": digest.model_dump(mode="json"),
        "candidate_resolution": resolution.model_dump(mode="json"),
        "wiki_context_snapshot": snapshot.model_dump(mode="json"),
        "profile": ctx.profile.model_dump(mode="json"),
        "language_contract": ctx.manifest.vault_config_snapshot.model_dump(mode="json"),
        "contract": {
            "goal": "Read the frozen wiki context and decide create/update/noop/needs_human_decision for planned pages.",
            "actions": ["create", "update", "noop", "needs_human_decision"],
            "rules": [
                "Use canonical_target_path for final writes; for update use the matched existing page path.",
                "needs_human_decision is not writeable and must be resolved before drafting.",
                "noop only when existing wiki already fully covers the source without new examples, expressions, links, or value points.",
                "New but thin topics should be create, not noop.",
                "All user-visible fields must be Chinese unless keeping stable domain terms.",
                "Keep value_points and reuse_scenarios grounded in concrete source content.",
            ],
        },
    }
    plan, _ = _structured_call(ctx.run_dir, ctx.execution_context, step_name).run(
        step_name,
        payload,
        WikiMergePlanArtifact,
    )
    plan = _redacted_model(ctx, plan, WikiMergePlanArtifact)
    plan = finalize_wiki_merge_plan(plan, resolution, snapshot, snapshot_path.relative_to(ctx.run_dir).as_posix())
    validate_wiki_merge_plan(digest, plan, resolution)
    out = step_root / "wiki_merge_plan.json"
    write_json(out, plan)
    table = step_root / "wiki_merge_plan.md"
    table.write_text(render_merge_plan_markdown(plan), encoding="utf-8")
    outputs = [
        _ref(ctx.run_dir, out, step_name, "json", "wiki_merge_plan.v4"),
        _ref(ctx.run_dir, table, step_name, "markdown"),
    ]
    provider_result = step_root / "provider_result.json"
    if provider_result.exists():
        outputs.append(_ref(ctx.run_dir, provider_result, step_name, "provider_result", "provider_result.v1"))
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
    if has_needs_human:
        pending_path = step_root / "pending_merge_plan.json"
        pending_path.write_text(plan_path.read_text(encoding="utf-8"), encoding="utf-8")
        decision = ReviewDecision(
            review_step=step_name,
            decision="pending",
            review_mode="manual",
            auto_approved=False,
            notes="merge plan contains needs_human_decision; revise to create/update/noop before continuing.",
        )
        decision_path = step_root / "review_decision.json"
        write_json(decision_path, decision)
        mark_step_awaiting_review(
            ctx.manifest,
            step_name,
            outputs=[
                _ref(ctx.run_dir, prompt_path, step_name, "markdown"),
                _ref(ctx.run_dir, feedback_path, step_name, "jsonl"),
                _ref(ctx.run_dir, pending_path, step_name, "json", "wiki_merge_plan.v4"),
                _ref(ctx.run_dir, decision_path, step_name, "json", "review_decision.v1"),
            ],
            error="merge plan requires human decision; run merge-level revise.",
        )
        return
    approved_path.write_text(plan_path.read_text(encoding="utf-8"), encoding="utf-8")
    decision = ReviewDecision(
        review_step=step_name,
        decision="approved",
        review_mode="auto_stub",
        auto_approved=True,
        notes="Auto-approved because merge plan contains no needs_human_decision.",
    )
    decision_path = step_root / "review_decision.json"
    write_json(decision_path, decision)
    complete_step(
        ctx.manifest,
        step_name,
        outputs=[
            _ref(ctx.run_dir, prompt_path, step_name, "markdown"),
            _ref(ctx.run_dir, feedback_path, step_name, "jsonl"),
            _ref(ctx.run_dir, decision_path, step_name, "json", "review_decision.v1"),
            _ref(ctx.run_dir, approved_path, step_name, "json", "wiki_merge_plan.v4"),
        ],
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
    validate_source_digest(digest)
    validate_wiki_merge_plan(digest, merge_plan, resolution)
    ensure_wiki_context_current(ctx.vault, snapshot)
    payload = {
        "approved_prepared_markdown": (require_step_output_dir(ctx.run_dir, "prepared_raw_review") / "approved_prepared.md").read_text(encoding="utf-8"),
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
                "Write all user-visible content in Chinese except stable domain terms with Chinese explanation when needed.",
                "Ground examples, value points, and reuse scenarios in source content.",
            ],
        },
    }
    draft_artifact, _ = _structured_call(ctx.run_dir, ctx.execution_context, step_name).run(
        step_name,
        payload,
        DraftRenderingArtifact,
    )
    draft_artifact = _redacted_model(ctx, draft_artifact, DraftRenderingArtifact)
    draft_artifact = finalize_draft_rendering(draft_artifact, merge_plan, snapshot)
    validate_draft_rendering(draft_artifact, merge_plan)
    draft_artifact_path = step_root / "draft_rendering.json"
    write_json(draft_artifact_path, draft_artifact)
    draft_root = step_root / "draft_pages"
    outputs: list[Path] = []
    target_manifest: list[DraftWriteTarget] = []
    source_title = source_title_for_raw(digest.source_raw_path)
    action_by_id = {item.page_plan_id: item for item in merge_plan.items}
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
    tension_rows: list[dict[str, str]] = []
    log_date = snapshot.log_date
    if knowledge_changed_paths:
        assert_system_page_can_be_overwritten(ctx.vault, "wiki/index.md")
        index = draft_root / "index.md"
        index.write_text(
            render_index(
                knowledge_rows=build_index_rows(ctx.profile, merge_plan, draft_artifact, snapshot),
                tension_rows=tension_rows,
                page_type_order=list(ctx.profile.page_types),
            ),
            encoding="utf-8",
        )
        outputs.append(index)
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
    write_manifest_artifact = DraftWriteManifest(
        targets=target_manifest,
        has_updates=any(item.action == "update" for item in merge_plan.items),
        has_noops=any(item.action == "noop" for item in merge_plan.items),
        source_only_noop=all(item.action == "noop" for item in merge_plan.items),
    )
    write_manifest_path = step_root / "draft_write_manifest.json"
    write_json(write_manifest_path, write_manifest_artifact)
    outputs.extend([draft_artifact_path, write_manifest_path])
    refs = [_ref(ctx.run_dir, path, step_name, "markdown" if path.suffix == ".md" else "json") for path in outputs]
    provider_result = step_root / "provider_result.json"
    if provider_result.exists():
        refs.append(_ref(ctx.run_dir, provider_result, step_name, "provider_result", "provider_result.v1"))
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
    validate_source_digest(digest)
    validate_wiki_merge_plan(digest, merge_plan, resolution)
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
        complete_step(
            ctx.manifest,
            step_name,
            outputs=[
                _ref(ctx.run_dir, prompt_path, step_name, "markdown"),
                _ref(ctx.run_dir, approved_manifest_path, step_name, "json", "draft_write_manifest.v1"),
                _ref(ctx.run_dir, approval_path, step_name, "json", "draft_review.v1"),
            ],
        )
        return
    if not draft_manifest.has_updates:
        approved_manifest_path.write_text(draft_manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
        approval = build_draft_approval(
            ctx.run_dir,
            approved_manifest_path,
            decision="approved",
            review_mode="auto_stub",
            auto_approved=True,
            notes="Create-only operation auto-approved.",
        )
        write_json(approval_path, approval)
        complete_step(
            ctx.manifest,
            step_name,
            outputs=[
                _ref(ctx.run_dir, prompt_path, step_name, "markdown"),
                _ref(ctx.run_dir, approved_manifest_path, step_name, "json", "draft_write_manifest.v1"),
                _ref(ctx.run_dir, approval_path, step_name, "json", "draft_review.v1"),
            ],
        )
        return
    pending_manifest = step_root / "pending_write_manifest.json"
    pending_manifest.write_text(draft_manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
    approval = DraftApproval(
        decision="pending",
        review_mode="manual",
        auto_approved=False,
        notes="Update draft requires explicit approval.",
    )
    write_json(approval_path, approval)
    mark_step_awaiting_review(
        ctx.manifest,
        step_name,
        outputs=[
            _ref(ctx.run_dir, prompt_path, step_name, "markdown"),
            _ref(ctx.run_dir, pending_manifest, step_name, "json", "draft_write_manifest.v1"),
            _ref(ctx.run_dir, approval_path, step_name, "json", "draft_review.v1"),
        ],
        error="update draft requires explicit approval.",
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
        source_candidate_id = item.source_basis.source_candidate_ids[0] if item.source_basis.source_candidate_ids else ""
        candidate = candidates[source_candidate_id]
        related_pages, related_unresolved = resolve_related_pages(item, candidate, resolution, snapshot)
        items.append(
            WikiMergePlanItem(
                page_plan_id=item.page_plan_id,
                source_basis=item.source_basis,
                action=action,
                canonical_target_path=item.candidate_target_path,
                display_title=item.display_title,
                page_type=item.page_type,
                matched_page=matched_page,
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
                apply_eligibility="applyable",
                blocked_reason="",
                reason=item.reason,
            )
        )
    return WikiMergePlanArtifact(log_date=log_date, items=items, context_snapshot_ref="wiki_context_snapshot/wiki_context_snapshot.json")


def existing_knowledge_page_paths(vault: Path) -> set[str]:
    wiki = vault / "wiki"
    if not wiki.exists():
        return set()
    paths: set[str] = set()
    for path in wiki.rglob("*.md"):
        rel = path.relative_to(vault).as_posix()
        if rel in {"wiki/index.md", "wiki/log.md"}:
            continue
        parts = Path(rel).parts
        if len(parts) > 1 and parts[1] in {"sources", "logs"}:
            continue
        metadata = read_wiki_page_metadata(vault, rel)
        if metadata is not None and metadata.llmwiki_type.lower() != "source":
            paths.add(rel)
    return paths


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
    for entry in snapshot.entries:
        if entry.metadata is None:
            continue
        if entry.metadata.llmwiki_type.lower() == "source":
            continue
        for key in [entry.metadata.title, *entry.metadata.aliases]:
            metadata_lookup.setdefault(normalize_related_key(key), []).append(entry.metadata)
    related: list[RelatedPageRef] = []
    unresolved: list[str] = []
    seen_paths: set[str] = set()
    for raw in candidate.related_candidates:
        if len(related) >= 8:
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
                reason=f"Resolved related candidate id {raw!r} from source digest.",
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
                    reason=f"Resolved exact title/alias match {raw!r} from wiki context.",
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


def _structured_call(run_dir: Path, execution_context: ProviderExecutionContext, task: str) -> StructuredModelCall:
    return StructuredModelCall(
        execution_context.provider_for_task(task),
        output_dir=step_output_dir(run_dir, task),
        result_filename="provider_result.json",
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
    for step in manifest.steps:
        durations = [attempt.duration_ms for attempt in step.attempts if attempt.duration_ms is not None]
        total = sum(durations)
        provider = step_provider_label(step.name, step.attempts[-1].provider_spec if step.attempts else None)
        retry_count += max(0, len(step.attempts) - 1)
        row = {
            "name": step.name,
            "status": step.status.value,
            "attempts": len(step.attempts),
            "last_duration_ms": durations[-1] if durations else None,
            "total_duration_ms": total,
            "provider": provider,
        }
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
        "created_count": created,
        "updated_count": updated,
        "noop_count": noop,
        "cleaned_link_count": cleanup_count,
        "preserved_media_embed_count": preserved_media_count,
        "written_target_count": written_target_count,
    }


def step_provider_label(step_name: str, provider_spec: str | None) -> str:
    if provider_spec:
        return provider_spec
    if step_name.endswith("_review"):
        return "local:auto_review"
    return "local"


def status(vault: Path, operation_id: str) -> OperationManifest:
    return read_manifest(RunStore(vault).manifest_path(operation_id))


def approve_review(vault: Path, operation_id: str, review_step: str) -> OperationManifest:
    store = RunStore(vault)
    with run_lock(vault, operation_id):
        manifest = read_manifest(store.manifest_path(operation_id))
        _require_review_step_awaiting(manifest, review_step)
        run_dir = store.run_dir(operation_id)
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
                notes="Manually approved draft.",
            )
            approval_path = step_root / "draft_approval.json"
            write_json(approval_path, approval)
            mark_step_approved(
                manifest,
                review_step,
                outputs=[
                    _ref(run_dir, approved, review_step, "json", "draft_write_manifest.v1"),
                    _ref(run_dir, approval_path, review_step, "json", "draft_review.v1"),
                ],
            )
            delete_downstream_step_dirs(vault, operation_id, "validation")
            mark_from_pending(manifest, "validation")
            write_manifest(store.manifest_path(operation_id), manifest)
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
            validate_wiki_merge_plan(digest, plan, resolution)
            if any(item.action == "needs_human_decision" for item in plan.items):
                raise PipelineError("needs_human_decision must be revised to create/update/noop before approval.")
            approved = step_root / "approved_merge_plan.json"
            write_json(approved, plan)
            decision = ReviewDecision(review_step=review_step, decision="approved", review_mode="manual", auto_approved=False)
            decision_path = step_root / "review_decision.json"
            write_json(decision_path, decision)
            mark_step_approved(
                manifest,
                review_step,
                outputs=[
                    _ref(run_dir, approved, review_step, "json", "wiki_merge_plan.v4"),
                    _ref(run_dir, decision_path, review_step, "json", "review_decision.v1"),
                ],
            )
            delete_downstream_step_dirs(vault, operation_id, "draft_rendering")
            mark_from_pending(manifest, "draft_rendering")
            write_manifest(store.manifest_path(operation_id), manifest)
            return manifest
        raise PipelineError(f"Unsupported review step: {review_step}")


def revise_review(vault: Path, operation_id: str, review_step: str) -> OperationManifest:
    store = RunStore(vault)
    with run_lock(vault, operation_id):
        manifest = read_manifest(store.manifest_path(operation_id))
        _require_review_step_awaiting(manifest, review_step)
        if review_step == "merge_plan_review":
            delete_downstream_step_dirs(vault, operation_id, "wiki_merge_planning")
            mark_from_pending(manifest, "wiki_merge_planning")
        elif review_step == "draft_review":
            delete_downstream_step_dirs(vault, operation_id, "draft_rendering")
            mark_from_pending(manifest, "draft_rendering")
        else:
            raise PipelineError(f"Unsupported review step: {review_step}")
        manifest.status = OperationStatus.running
        write_manifest(store.manifest_path(operation_id), manifest)
        return manifest


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


def finalize_candidate_resolution(vault: Path, profile: Any, artifact: CandidateResolutionArtifact) -> CandidateResolutionArtifact:
    items: list[CandidateResolutionItem] = []
    seen_paths: dict[str, int] = {}
    for item in artifact.items:
        if item.page_type not in profile.page_types:
            raise PipelineError(f"candidate_resolution uses unknown page_type: {item.page_type}")
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
) -> WikiContextSnapshot:
    paths = {
        "wiki/index.md",
        "wiki/log.md",
        f"wiki/logs/{log_date}.md",
        f"wiki/{source_target_path}",
    }
    for item in resolution.items:
        paths.add(f"wiki/{item.candidate_target_path}")
    paths.update(existing_knowledge_page_paths(vault))
    entries: list[WikiContextEntry] = []
    for rel in sorted(paths):
        path = vault / rel
        if path.exists():
            entries.append(
                WikiContextEntry(
                    path=rel,
                    expected_state="present",
                    preimage_sha256=sha256_file(path),
                    content=path.read_text(encoding="utf-8"),
                    metadata=read_wiki_page_metadata(vault, rel),
                )
            )
        else:
            entries.append(WikiContextEntry(path=rel, expected_state="missing", preimage_sha256=None, content=""))
    return WikiContextSnapshot(log_date=log_date, source_target_path=source_target_path, entries=entries)


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
        if other.page_type.lower() == "source":
            continue
        source_resolution = resolution_by_id.get(other.page_plan_id)
        paths = {other.canonical_target_path}
        if source_resolution is not None:
            paths.add(source_resolution.candidate_target_path)
        for path in paths:
            if path:
                current_by_path[path] = other
        current_by_title.setdefault(normalize_related_key(other.display_title), []).append(other)

    metadata_by_path: dict[str, WikiPageMetadata] = {}
    metadata_by_title: dict[str, list[WikiPageMetadata]] = {}
    for entry in snapshot.entries:
        if entry.metadata is None:
            continue
        if entry.metadata.llmwiki_type.lower() == "source":
            continue
        metadata_by_path[entry.metadata.path] = entry.metadata
        for key in [entry.metadata.title, *entry.metadata.aliases]:
            metadata_by_title.setdefault(normalize_related_key(key), []).append(entry.metadata)

    related: list[RelatedPageRef] = []
    unresolved: list[str] = []
    seen_paths: set[str] = set()
    for suggestion in item.related_pages:
        if len(related) >= 8:
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
                    reason=fallback_reason or f"Resolved exact path {raw!r} from current merge plan.",
                )
            metadata = metadata_by_path.get(target_path)
            if metadata is not None and metadata.path != self_path:
                return RelatedPageRef(
                    target_path=metadata.path,
                    display_title=clean_display_title(metadata.title),
                    source="wiki_context",
                    reason=fallback_reason or f"Resolved exact path {raw!r} from wiki context.",
                )
        key = normalize_related_key(raw)
        current_matches = current_by_title.get(key, [])
        if len(current_matches) == 1 and current_matches[0].canonical_target_path != self_path:
            current = current_matches[0]
            return RelatedPageRef(
                target_path=current.canonical_target_path,
                display_title=current.display_title,
                source="source_digest",
                reason=fallback_reason or f"Resolved exact title {raw!r} from current merge plan.",
            )
        metadata_matches = metadata_by_title.get(key, [])
        if len(metadata_matches) == 1 and metadata_matches[0].path != self_path:
            metadata = metadata_matches[0]
            return RelatedPageRef(
                target_path=metadata.path,
                display_title=clean_display_title(metadata.title),
                source="wiki_context",
                reason=fallback_reason or f"Resolved exact title/alias {raw!r} from wiki context.",
            )
    return None


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
) -> WikiMergePlanArtifact:
    resolution_by_id = {item.page_plan_id: item for item in resolution.items}
    snapshot_paths = {entry.path for entry in snapshot.entries}
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
        canonical = item.canonical_target_path or resolution_item.candidate_target_path
        matched_page = item.matched_page
        action = item.action
        if item.action == "update":
            matched_page = matched_page or canonical
            canonical = matched_page
        if item.action == "create":
            canonical = resolution_item.candidate_target_path
            matched_page = None
        if f"wiki/{canonical}" not in snapshot_paths:
            raise PipelineError(f"wiki_merge_plan target is outside wiki_context_snapshot: wiki/{canonical}")
        entry = snapshot_entry(snapshot, f"wiki/{canonical}")
        if action == "needs_human_decision":
            pass
        elif entry.expected_state == "present":
            action = "update" if action != "noop" else "noop"
            matched_page = canonical if action == "update" else matched_page
        else:
            action = "create"
            matched_page = None
        apply_eligibility = item.apply_eligibility
        blocked_reason = item.blocked_reason
        if apply_eligibility == "blocked" and action != "needs_human_decision":
            action = "needs_human_decision"
            blocked_reason = blocked_reason or "模型将该项标记为 blocked，需要人工决策。"
        preliminary.append(
            item.model_copy(
                update={
                    "action": action,
                    "canonical_target_path": canonical,
                    "matched_page": matched_page,
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
        items.append(
            item.model_copy(
                update={
                    "related_pages": related_pages,
                    "related_unresolved": unresolved,
                    "unresolved_related": unresolved,
                }
            )
        )
    return WikiMergePlanArtifact(log_date=snapshot.log_date, items=items, context_snapshot_ref=snapshot_ref)


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
                item.action,
                item.display_title,
                f"`{item.canonical_target_path}`",
                item.new_understanding,
                item.why_this_matters,
                item.apply_eligibility,
                item.blocked_reason,
            ]
        )
    return "# Wiki 合并计划\n\n" + format_markdown_table(
        ["页面计划", "动作", "标题", "目标", "新增理解", "价值点", "Apply", "阻断原因"],
        rows,
    ) + "\n"


def render_merge_plan_review_prompt(plan: WikiMergePlanArtifact) -> str:
    return (
        "# 合并计划审核\n\n"
        "审查这一步回答：写哪些页面、为什么写。\n\n"
        f"{render_merge_plan_markdown(plan)}"
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
            raise PipelineError("draft_rendering page_plan_id must not be empty")
        item = plan_by_id.get(page.page_plan_id)
        if item is None:
            item = next((candidate for candidate in plan_by_id.values() if candidate.canonical_target_path == page.canonical_target_path), None)
        if item is None:
            raise PipelineError(f"draft_rendering references non-draftable page_plan_id: {page.page_plan_id}")
        if item.page_plan_id in used_ids:
            raise PipelineError(f"draft_rendering duplicates page_plan_id: {item.page_plan_id}")
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


def validate_draft_rendering(artifact: DraftRenderingArtifact, plan: WikiMergePlanArtifact) -> None:
    allowed_sections = set(CANONICAL_DRAFT_SECTION_KEYS)
    required_sections = {"summary", "detail"}
    required_ids = {item.page_plan_id for item in plan.items if item.action in {"create", "update"}}
    actual_ids = {page.page_plan_id for page in artifact.pages}
    missing = required_ids - actual_ids
    if missing:
        raise PipelineError(f"draft_rendering misses page_plan_id(s): {sorted(missing)}")
    extra = actual_ids - required_ids
    if extra:
        raise PipelineError(f"draft_rendering contains unexpected page_plan_id(s): {sorted(extra)}")
    for page in artifact.pages:
        if not page.canonical_target_path.strip():
            raise PipelineError(f"{page.page_plan_id} canonical_target_path must not be empty")
        if not page.section_bodies:
            raise PipelineError(f"{page.page_plan_id} section_bodies must not be empty")
        section_keys = set(page.section_bodies)
        unknown_sections = section_keys - allowed_sections
        if unknown_sections:
            raise PipelineError(f"{page.page_plan_id} section_bodies contains unsupported section key(s): {sorted(unknown_sections)}")
        missing_sections = required_sections - section_keys
        if missing_sections:
            raise PipelineError(f"{page.page_plan_id} section_bodies misses required section key(s): {sorted(missing_sections)}")
        if not page.change_summary.strip():
            raise PipelineError(f"{page.page_plan_id} change_summary must not be empty")
        for body in page.section_bodies.values():
            if "---\n" in body or body.lstrip().startswith("# ") or contains_source_graph_link(body):
                raise PipelineError(f"{page.page_plan_id} section body contains forbidden page-level markdown")


def contains_source_graph_link(text: str) -> bool:
    return text_contains_source_graph_link(text)


def snapshot_entry(snapshot: WikiContextSnapshot, path: str) -> WikiContextEntry:
    for entry in snapshot.entries:
        if entry.path == path:
            return entry
    raise PipelineError(f"snapshot missing path: {path}")


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
) -> str:
    summary = page.section_bodies.get("summary") or item.new_understanding
    detail = page.section_bodies.get("detail") or item.knowledge_delta or item.new_understanding
    examples = page.section_bodies.get("examples") or "暂无相关例子记录。"
    values = page.section_bodies.get("value_points") or "\n".join(f"- {value}" for value in item.value_points) or "暂无明确价值点记录。"
    additional_notes = page.section_bodies.get("additional_notes", "").strip()
    additional_notes_section = f"## 补充观察\n\n{additional_notes}\n\n" if additional_notes else ""
    questions = page.section_bodies.get("open_questions") or "暂无矛盾与未决问题记录。"
    related = render_related_pages(item)
    metadata = existing_entry.metadata
    aliases = metadata.aliases if metadata is not None else []
    created = metadata.created if metadata is not None and metadata.created else log_date
    source_raw_paths = _append_unique(metadata.source_raw_paths if metadata is not None else [], raw_path)
    source_raw_hashes = _append_unique(metadata.source_raw_hashes if metadata is not None else [], raw_hash)
    source_prepared_hashes = _append_unique(metadata.source_prepared_hashes if metadata is not None else [], prepared_hash)
    source_operation_ids = _append_unique(metadata.source_operation_ids if metadata is not None else [], operation_id)
    return (
        "---\n"
        f"llmwiki_type: {item.page_type}\n"
        f"title: {yaml_scalar(item.display_title)}\n"
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
        f"# {item.display_title}\n\n"
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
    touched = "\n".join(f"- `{path}`" for path in touched_pages) or "- 暂无触达页面。"
    no_change = "\n".join(f"- `{path}`" for path in no_change_pages) or "- 暂无未写入页面。"
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
        f"触达页面：\n{touched}\n\n未改动页面：\n{no_change}\n"
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
    for entry in snapshot.entries:
        if entry.metadata is None or entry.metadata.llmwiki_type.lower() == "source":
            continue
        rows_by_path[entry.metadata.path] = {
            "title": clean_display_title(entry.metadata.title),
            "page": obsidian_link(entry.metadata.path),
            "type": entry.metadata.llmwiki_type,
            "summary": entry.metadata.summary,
            "updated": entry.metadata.updated,
        }
    page_by_id = {page.page_plan_id: page for page in draft.pages}
    for item in plan.items:
        if item.action not in {"create", "update"}:
            continue
        if item.page_type.lower() == "source":
            continue
        page = page_by_id.get(item.page_plan_id)
        summary = page.section_bodies.get("summary") if page else item.new_understanding
        rows_by_path[item.canonical_target_path] = {
            "title": item.display_title,
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


def render_related_pages(item: WikiMergePlanItem) -> str:
    if not item.related_pages:
        return "- 暂无相关页面记录。"
    rows = []
    for related in item.related_pages[:8]:
        rows.append(f"- {obsidian_alias_link(related.target_path, related.display_title)}：{related.reason}")
    return "\n".join(rows)


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
    approved_manifest = read_model(approved_manifest_path, DraftWriteManifest)
    markdown_hashes: dict[str, str] = {}
    for target in approved_manifest.targets:
        draft = run_dir / target.draft_path
        if draft.suffix == ".md" and draft.exists():
            markdown_hashes[target.draft_path] = sha256_file(draft)
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
            target.expected_state,
            target.preimage_sha256 or "",
        ]
        for target in draft_manifest.targets
    ]
    return "# 草稿审核\n\n审查这一步回答：具体写什么。\n\n" + format_markdown_table(
        ["动作", "目标", "草稿", "预期状态", "Preimage"],
        rows,
    ) + "\n"


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
        requires_draft_review=draft_manifest.has_updates,
        has_updates=draft_manifest.has_updates,
        has_noops=draft_manifest.has_noops,
        blocked_reasons=[],
        write_set_sha256=write_set_sha,
        targets=targets,
        source_targets=source_targets,
        log_targets=log_targets,
        index_targets=index_targets,
    )
