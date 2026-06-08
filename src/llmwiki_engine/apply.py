from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .hash_utils import artifact_ref, sha256_file
from .io import append_jsonl, read_model, write_json
from .manifest import read_manifest, step_satisfied, write_manifest
from .models import (
    AppliedReceipt,
    ApplyPreview,
    ArtifactRef,
    ArtifactVisibility,
    DraftApproval,
    DraftWriteManifest,
    OperationManifest,
    OperationStatus,
    RawLinkCleanupArtifact,
    SourceDigestArtifact,
    WikiContextSnapshot,
    utc_now,
)
from .steps import require_step_output_dir
from .verify import require_verified
from . import run_metrics as _run_metrics
from . import apply_guards as _apply_guards
from .wiki_context import wiki_context_drift_messages
from .workspace import RunStore, apply_lock, run_lock

WIKI_CONTEXT_DRIFT_MESSAGE = "当前 operation 的 apply plan 已过期，因为 wiki 在 plan 生成后发生变化。请 resume 后再 apply。"

class ApplyError(RuntimeError):
    def __init__(self, message: str, *, details: list[str] | None = None) -> None:
        super().__init__(message)
        self.details = details or []


def apply_operation(vault: Path, operation_id: str) -> list[Path]:
    store = RunStore(vault)
    run_dir = store.run_dir(operation_id)
    with apply_lock(vault), run_lock(vault, operation_id):
        manifest = read_manifest(store.manifest_path(operation_id))
        if manifest.status == OperationStatus.applied:
            raise ApplyError("Operation has already been applied.")
        if manifest.status != OperationStatus.drafted:
            raise ApplyError(f"Operation is not apply-ready: {manifest.status}")
        _verify_pipeline_completed(manifest)
        if _receipt_exists(store.applied_log, operation_id):
            raise ApplyError("Applied receipt already exists for this operation.")
        require_verified(vault, manifest)
        apply_preview_dir = require_step_output_dir(run_dir, "apply_preview")
        raw_prepare_dir = require_step_output_dir(run_dir, "raw_prepare")
        raw_cleanup_dir = require_step_output_dir(run_dir, "raw_link_cleanup")
        raw_cleanup = read_model(raw_cleanup_dir / "raw_link_cleanup.json", RawLinkCleanupArtifact)
        _verify_wiki_context(vault, run_dir)
        preview = read_model(apply_preview_dir / "apply_preview.json", ApplyPreview)
        if not preview.operation_applyable:
            raise ApplyError("; ".join(preview.blocked_reasons) or "Operation is not applyable.")
        if not preview.targets:
            raise ApplyError("Apply preview has no targets.")
        _verify_unique_preview_targets(preview)
        _verify_draft_approval(run_dir, preview)
        _verify_write_set(run_dir, preview)
        _verify_source_duplicate(vault, run_dir, manifest.operation_id)
        _verify_preimages(vault, preview)
        targets = _prepare_writes(vault, run_dir, preview)
        written: list[Path] = []
        try:
            for target in targets:
                _write_bytes_atomic(target.output, target.data)
                written.append(target.output)
            knowledge_written = [path for path in written if "/wiki/sources/" not in path.as_posix() and "/wiki/log" not in path.as_posix()]
            final_status = OperationStatus.applied if knowledge_written else OperationStatus.source_recorded
            receipt = AppliedReceipt(
                operation_id=operation_id,
                raw_bindings=manifest.raw_bindings,
                raw_cleanup_pre_sha256=raw_cleanup.pre_cleanup_sha256,
                raw_cleanup_post_sha256=raw_cleanup.post_cleanup_sha256,
                raw_cleanup_rule_version=raw_cleanup.cleanup_rule_version,
                raw_cleanup_artifact_ref="raw_link_cleanup/raw_link_cleanup.json",
                raw_cleanup_changed=raw_cleanup.changed,
                raw_cleanup_cleaned_link_count=raw_cleanup.cleaned_link_count,
                raw_cleanup_diff_ref="raw_link_cleanup/cleanup.diff",
                prepared_raw=_optional_receipt_ref(vault, raw_prepare_dir / "prepared.md", "markdown", "raw_prepare"),
                raw_preparation=_optional_receipt_ref(vault, raw_prepare_dir / "raw_preparation.json", "json", "raw_prepare"),
                written_pages=[
                    artifact_ref(
                        base=vault,
                        path=path,
                        kind="markdown",
                        producer_step="apply",
                        required_for_resume=False,
                        visibility=ArtifactVisibility.wiki_output,
                    )
                    for path in written
                ],
                written_targets=[path.relative_to(vault).as_posix() for path in written],
                profile=manifest.profile,
                profile_version=manifest.profile_version,
                engine_version=manifest.engine_version,
            )
            append_jsonl(store.applied_log, [receipt])
            manifest.status = final_status
            manifest.updated_at = utc_now()
            write_manifest(store.manifest_path(operation_id), manifest)
            _run_metrics.refresh_run_metrics(run_dir, manifest)
        except Exception as exc:
            _record_apply_failed(vault, store, run_dir, manifest, operation_id, written, exc)
            raise ApplyError(f"Apply write failed; inspect written targets before retry: {exc}") from exc
        return written


@dataclass(frozen=True)
class PendingWrite:
    output: Path
    data: bytes
    original: bytes | None


def _verify_pipeline_completed(manifest: OperationManifest) -> None:
    incomplete = [step for step in manifest.steps if not step_satisfied(step.status)]
    if incomplete:
        details = ", ".join(f"{step.name}={step.status.value}" for step in incomplete)
        raise ApplyError(f"Operation is not apply-ready; incomplete step(s): {details}")


def _verify_unique_preview_targets(preview: ApplyPreview) -> None:
    target_paths = [target.target_path for target in preview.targets]
    if len(target_paths) != len(set(target_paths)):
        raise ApplyError("Apply preview contains duplicate target_path values.")


def _prepare_writes(vault: Path, run_dir: Path, preview: ApplyPreview) -> list[PendingWrite]:
    writes: list[PendingWrite] = []
    for target in preview.targets:
        draft = _contained_path(run_dir, target.draft_path, "draft_path")
        output = _contained_path(vault, target.target_path, "target_path")
        _require_within(output, vault / "wiki", "target_path")
        original = output.read_bytes() if output.exists() else None
        writes.append(PendingWrite(output=output, data=draft.read_bytes(), original=original))
    return writes


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _record_apply_failed(
    vault: Path,
    store: RunStore,
    run_dir: Path,
    manifest: OperationManifest,
    operation_id: str,
    written: list[Path],
    exc: Exception,
) -> None:
    manifest.status = OperationStatus.apply_failed
    manifest.updated_at = utc_now()
    failure_path = run_dir / "apply_failed.json"
    write_json(
        failure_path,
        {
            "written_targets": [path.relative_to(vault).as_posix() for path in written],
            "error": str(exc),
        },
    )
    write_manifest(store.manifest_path(operation_id), manifest)
    _run_metrics.refresh_run_metrics(run_dir, manifest)


def _optional_receipt_ref(vault: Path, path: Path, kind: str, producer_step: str) -> ArtifactRef | None:
    if not path.exists():
        return None
    return artifact_ref(
        base=vault,
        path=path,
        kind=kind,
        producer_step=producer_step,
        required_for_resume=False,
        visibility=ArtifactVisibility.run_cache,
    )


def _verify_preimages(vault: Path, preview: ApplyPreview) -> None:
    for target in preview.targets:
        path = _contained_path(vault, target.target_path, "target_path")
        _require_within(path, vault / "wiki", "target_path")
        if target.expected_state == "missing":
            if path.exists():
                raise ApplyError(f"Target appeared after preview: {target.target_path}")
            continue
        if not path.exists():
            raise ApplyError(f"Target disappeared after preview: {target.target_path}")
        if sha256_file(path) != target.preimage_sha256:
            raise ApplyError(f"Target changed after preview: {target.target_path}")


def _verify_wiki_context(vault: Path, run_dir: Path) -> None:
    snapshot_path = require_step_output_dir(run_dir, "wiki_context_snapshot") / "wiki_context_snapshot.json"
    snapshot = read_model(snapshot_path, WikiContextSnapshot)
    messages = wiki_context_drift_messages(vault, snapshot)
    if messages:
        raise ApplyError(WIKI_CONTEXT_DRIFT_MESSAGE, details=messages)


def _verify_draft_approval(run_dir: Path, preview: ApplyPreview) -> None:
    try:
        _apply_guards.require_draft_rendering_sidecars(run_dir)
    except Exception as exc:
        raise ApplyError(str(exc)) from exc
    approval_path = require_step_output_dir(run_dir, "draft_review") / "draft_approval.json"
    approval = read_model(approval_path, DraftApproval)
    if approval.decision != "approved":
        raise ApplyError(f"Draft review is not approved: {approval.decision}")
    manifest_path = require_step_output_dir(run_dir, "draft_review") / "approved_write_manifest.json"
    if approval.approved_draft_json_sha256 != sha256_file(manifest_path):
        raise ApplyError("Approved draft manifest changed after review.")
    for rel_path, expected in approval.approved_markdown_sha256.items():
        path = run_dir / rel_path
        if not path.exists() or sha256_file(path) != expected:
            raise ApplyError(f"Approved draft markdown changed after review: {rel_path}")


def _verify_write_set(run_dir: Path, preview: ApplyPreview) -> None:
    manifest_path = require_step_output_dir(run_dir, "draft_review") / "approved_write_manifest.json"
    draft_manifest = read_model(manifest_path, DraftWriteManifest)
    approved_targets = [
        {
            "action": target.action,
            "target_path": target.target_path,
            "draft_path": target.draft_path,
            "preimage_sha256": target.preimage_sha256,
            "expected_state": target.expected_state,
            "page_plan_id": target.page_plan_id,
        }
        for target in draft_manifest.targets
    ]
    preview_targets = [
        {
            "action": target.action,
            "target_path": target.target_path,
            "draft_path": target.draft_path,
            "preimage_sha256": target.preimage_sha256,
            "expected_state": target.expected_state,
            "page_plan_id": target.page_plan_id,
        }
        for target in preview.targets
    ]
    if preview_targets != approved_targets:
        raise ApplyError("Apply preview targets do not match approved draft manifest.")
    payload = {
        "approved_manifest_sha256": sha256_file(manifest_path),
        "targets": [
            {
                "target_path": target.target_path,
                "draft_path": target.draft_path,
                "preimage_sha256": target.preimage_sha256,
                "expected_state": target.expected_state,
                "draft_sha256": sha256_file(run_dir / target.draft_path),
            }
            for target in preview.targets
        ],
    }
    digest = __import__("hashlib").sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    if digest != preview.write_set_sha256:
        raise ApplyError("Apply preview write_set_sha256 does not match approved draft write set.")


def _verify_source_duplicate(vault: Path, run_dir: Path, operation_id: str) -> None:
    digest = read_model(require_step_output_dir(run_dir, "source_digest_review") / "approved_digest.json", SourceDigestArtifact)
    prepared = require_step_output_dir(run_dir, "prepared_raw_review") / "approved_prepared.md"
    artifact = _apply_guards.build_source_duplicate_guard_artifact(
        vault,
        source_raw_path=digest.source_raw_path,
        source_raw_hash=sha256_file(vault / digest.source_raw_path),
        source_prepared_hash=sha256_file(prepared),
        operation_id=operation_id,
    )
    if artifact.status == "source_duplicate":
        raise ApplyError(f"source_duplicate: {artifact.reason}")
    if artifact.status == "source_revision_detected":
        raise ApplyError("同一路径内容已变化，source revision workflow 尚未实现；如确认为新材料，请另存为新 raw 文件名后重新 ingest。")


def _contained_path(base: Path, relative_path: str, label: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ApplyError(f"{label} must be a safe relative path: {relative_path}")
    candidate = base / path
    _require_within(candidate, base, label)
    return candidate


def _require_within(path: Path, base: Path, label: str) -> None:
    resolved_path = path.resolve(strict=False)
    resolved_base = base.resolve(strict=False)
    if resolved_path == resolved_base or resolved_base in resolved_path.parents:
        return
    raise ApplyError(f"{label} escapes expected directory: {path}")


def _receipt_exists(path: Path, operation_id: str) -> bool:
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if json.loads(line).get("operation_id") == operation_id:
            return True
    return False
