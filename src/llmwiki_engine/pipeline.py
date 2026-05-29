from __future__ import annotations

import json
import shutil
from pathlib import Path

from rich.console import Console

from . import __version__
from .events import EventLogger
from .hash_utils import artifact_ref, sha256_bytes, sha256_file
from .io import read_model, write_json, write_json_atomic, write_yaml
from .manifest import (
    begin_step,
    complete_step,
    fail_step,
    first_resumable_step,
    get_step,
    initial_steps,
    mark_from_pending,
    raw_ref,
    read_manifest,
    write_manifest,
)
from .models import (
    ApplyPreview,
    ApplyTarget,
    ArtifactRef,
    ArtifactVisibility,
    ClaimsArtifact,
    OperationManifest,
    OperationStatus,
    PagePlanArtifact,
    RawBinding,
    RawIndexArtifact,
    RawSpan,
    RunMode,
    SemanticAggregationArtifact,
    StepStatus,
    utc_now,
)
from .profiles import load_profile
from .providers import ProviderRegistry
from .rendering import normalized_page_plan, render_drafts
from .steps import STEP_NAMES, downstream_steps
from .structured import StructuredModelCall
from .validators import validate_aggregation, validate_claims, validate_page_plan, validate_raw_index
from .verify import require_verified
from .workspace import RunStore, archive_paths, ensure_v2_layout, relative_to_vault, resolve_raw_path, run_lock


class PipelineError(RuntimeError):
    pass


def init_vault(vault: Path, *, profile_name: str = "project_basic") -> None:
    profile = load_profile(profile_name)
    (vault / "raw").mkdir(parents=True, exist_ok=True)
    for spec in profile.page_types.values():
        (vault / "wiki" / spec.directory).mkdir(parents=True, exist_ok=True)
    ensure_v2_layout(vault)
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
            "providers": {
                "semantic_aggregation": "mock:fixture",
                "claim_extraction": "mock:fixture",
                "page_planning": "mock:fixture",
                "critic": "mock:fixture",
            },
        },
    )


def run_simplified_ingest(
    *,
    vault: Path,
    raw_file: Path,
    fixture_dir: Path,
    profile_name: str = "project_basic",
    slug: str | None = None,
    run_mode: RunMode = RunMode.dev,
    console: Console | None = None,
) -> OperationManifest:
    ensure_v2_layout(vault)
    raw_path, raw_rel = resolve_raw_path(vault, raw_file)
    raw_hash, raw_size = raw_ref(raw_path)
    operation_id = f"ING-{safe_timestamp()}-{slug or raw_path.stem}"
    store = RunStore(vault)
    run_dir = store.run_dir(operation_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    profile = load_profile(vault / ".llmwiki" / "profiles" / profile_name)
    snapshot_hashes = create_run_snapshots(run_dir, profile, fixture_dir)
    manifest = OperationManifest(
        operation_id=operation_id,
        operation_type="ingest",
        run_mode=run_mode,
        engine_version=__version__,
        profile=profile.name,
        profile_version=profile.version,
        workspace=relative_to_vault(vault, run_dir),
        raw_bindings=[RawBinding(relative_path=raw_rel, sha256=raw_hash, size_bytes=raw_size)],
        profile_snapshot_hash=snapshot_hashes["profile"],
        template_hashes=snapshot_hashes["templates"],
        provider_snapshot_hashes=snapshot_hashes["providers"],
        steps=initial_steps(),
    )
    write_manifest(store.manifest_path(operation_id), manifest)
    with run_lock(vault, operation_id):
        return execute_ingest(vault, operation_id, start_step=STEP_NAMES[0], console=console)


def resume_ingest(
    *,
    vault: Path,
    operation_id: str,
    from_step: str | None = None,
    run_mode: RunMode | None = None,
    console: Console | None = None,
) -> OperationManifest:
    store = RunStore(vault)
    manifest = read_manifest(store.manifest_path(operation_id))
    if manifest.status == OperationStatus.applied:
        raise PipelineError("Applied operations are immutable. Start a new operation instead.")
    require_verified(vault, manifest)
    start = from_step or first_resumable_step(manifest)
    if start is None:
        return manifest
    if from_step is not None:
        archive_downstream(vault, operation_id, from_step)
        mark_from_pending(manifest, from_step)
        if run_mode is not None:
            manifest.run_mode = run_mode
        write_manifest(store.manifest_path(operation_id), manifest)
    with run_lock(vault, operation_id):
        return execute_ingest(vault, operation_id, start_step=start, console=console)


def execute_ingest(vault: Path, operation_id: str, *, start_step: str, console: Console | None = None) -> OperationManifest:
    store = RunStore(vault)
    run_dir = store.run_dir(operation_id)
    logger = EventLogger(operation_id, run_dir / "events.jsonl", console=console)
    manifest = read_manifest(store.manifest_path(operation_id))
    profile = load_profile(run_dir / "snapshots" / "profile")
    raw_path = vault / manifest.raw_bindings[0].relative_path
    provider_dir = run_dir / "snapshots" / "mock_fixture"
    provider = ProviderRegistry().create("mock:fixture", fixture_dir=provider_dir)
    caller = StructuredModelCall(provider, output_dir=run_dir / "model_calls")
    start_index = STEP_NAMES.index(start_step)
    for step_name in STEP_NAMES[start_index:]:
        if get_step(manifest, step_name).status == StepStatus.completed:
            continue
        try:
            _run_step(step_name, vault, store.manifest_path(operation_id), run_dir, raw_path, profile, caller, manifest, logger)
        except Exception as exc:
            fail_step(manifest, step_name, str(exc))
            write_manifest(store.manifest_path(operation_id), manifest)
            logger.emit(step_name, "failed", status="failed", message=str(exc))
            raise
        write_manifest(store.manifest_path(operation_id), manifest)
    manifest.status = OperationStatus.drafted
    write_manifest(store.manifest_path(operation_id), manifest)
    return manifest


def _run_step(
    step_name: str,
    vault: Path,
    manifest_path: Path,
    run_dir: Path,
    raw_path: Path,
    profile,
    caller: StructuredModelCall,
    manifest: OperationManifest,
    logger: EventLogger,
) -> None:
    logger.emit(step_name, "started", status="running")
    begin_step(manifest, step_name)
    write_manifest(manifest_path, manifest)
    if step_name == "raw_index":
        raw_index = build_raw_index(vault, raw_path)
        out = run_dir / "raw_index.json"
        write_json(out, raw_index)
        complete_step(manifest, step_name, outputs=[_ref(run_dir, out, step_name, "json", "raw_index.v1")])
    elif step_name == "semantic_aggregation":
        raw_index = read_model(run_dir / "raw_index.json", RawIndexArtifact)
        aggregation, _ = caller.run("semantic_aggregation", raw_index.model_dump(mode="json"), SemanticAggregationArtifact)
        out = run_dir / "semantic_aggregation.json"
        write_json(out, aggregation)
        outputs = [_ref(run_dir, out, step_name, "json", "semantic_aggregation.v1")]
        provider_result = run_dir / "model_calls" / "semantic_aggregation.provider_result.json"
        if provider_result.exists():
            outputs.append(_ref(run_dir, provider_result, step_name, "provider_result", "provider_result.v1"))
        complete_step(manifest, step_name, outputs=outputs)
    elif step_name == "claim_extraction":
        raw_index = read_model(run_dir / "raw_index.json", RawIndexArtifact)
        aggregation = read_model(run_dir / "semantic_aggregation.json", SemanticAggregationArtifact)
        payload = {"raw_index": raw_index.model_dump(mode="json"), "semantic_aggregation": aggregation.model_dump(mode="json")}
        claims, _ = caller.run("claim_extraction", payload, ClaimsArtifact)
        out = run_dir / "claims.json"
        write_json(out, claims)
        outputs = [_ref(run_dir, out, step_name, "json", "claims.v1")]
        provider_result = run_dir / "model_calls" / "claim_extraction.provider_result.json"
        if provider_result.exists():
            outputs.append(_ref(run_dir, provider_result, step_name, "provider_result", "provider_result.v1"))
        complete_step(manifest, step_name, outputs=outputs)
    elif step_name == "page_planning":
        claims = read_model(run_dir / "claims.json", ClaimsArtifact)
        payload = {"profile": profile.model_dump(mode="json"), "claims": claims.model_dump(mode="json")}
        plan, _ = caller.run("page_planning", payload, PagePlanArtifact)
        raw_index = read_model(run_dir / "raw_index.json", RawIndexArtifact)
        plan = normalized_page_plan(raw_index, claims, plan)
        out = run_dir / "page_plan.json"
        write_json(out, plan)
        outputs = [_ref(run_dir, out, step_name, "json", "page_plan.v1")]
        provider_result = run_dir / "model_calls" / "page_planning.provider_result.json"
        if provider_result.exists():
            outputs.append(_ref(run_dir, provider_result, step_name, "provider_result", "provider_result.v1"))
        complete_step(manifest, step_name, outputs=outputs)
    elif step_name == "draft_rendering":
        raw_index = read_model(run_dir / "raw_index.json", RawIndexArtifact)
        claims = read_model(run_dir / "claims.json", ClaimsArtifact)
        plan = read_model(run_dir / "page_plan.json", PagePlanArtifact)
        outputs = render_drafts(draft_root=run_dir / "draft_pages", profile=profile, raw_index=raw_index, claims=claims, plan=plan)
        complete_step(manifest, step_name, outputs=[_ref(run_dir, path, step_name, "markdown") for path in outputs])
    elif step_name == "validation":
        raw_index = read_model(run_dir / "raw_index.json", RawIndexArtifact)
        aggregation = read_model(run_dir / "semantic_aggregation.json", SemanticAggregationArtifact)
        claims = read_model(run_dir / "claims.json", ClaimsArtifact)
        plan = read_model(run_dir / "page_plan.json", PagePlanArtifact)
        validate_raw_index(raw_index)
        validate_aggregation(raw_index, aggregation)
        validate_claims(raw_index, aggregation, claims)
        validate_page_plan(profile, claims, plan)
        complete_step(manifest, step_name)
    elif step_name == "apply_preview":
        preview = build_apply_preview(vault, run_dir)
        out = run_dir / "apply_preview.json"
        write_json(out, preview)
        complete_step(manifest, step_name, outputs=[_ref(run_dir, out, step_name, "json", "apply_preview.v1")])
    else:
        raise PipelineError(f"Unknown step: {step_name}")
    logger.emit(step_name, "completed", status="completed")


def build_raw_index(vault: Path, raw_path: Path) -> RawIndexArtifact:
    text = raw_path.read_text(encoding="utf-8")
    digest = sha256_bytes(text.encode("utf-8"))
    spans: list[RawSpan] = []
    cursor = 0
    raw_rel = relative_to_vault(vault, raw_path)
    for part in [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]:
        start = text.find(part, cursor)
        end = start + len(part)
        cursor = end
        spans.append(
            RawSpan(
                span_id=f"S{len(spans) + 1:03d}",
                raw_path=raw_rel,
                raw_sha256=digest,
                start_char=start,
                end_char=end,
                text_sha256=sha256_bytes(part.encode("utf-8")),
                text=part,
            )
        )
    return RawIndexArtifact(raw_path=raw_rel, raw_sha256=digest, spans=spans)


def create_run_snapshots(run_dir: Path, profile, fixture_dir: Path) -> dict[str, dict | str]:
    snapshot_root = run_dir / "snapshots"
    profile_root = snapshot_root / "profile"
    fixture_root = snapshot_root / "mock_fixture"
    profile_root.mkdir(parents=True, exist_ok=True)
    fixture_root.mkdir(parents=True, exist_ok=True)
    profile_file = profile_root / "profile.yaml"
    write_yaml(profile_file, profile.model_dump(mode="json"))
    template_hashes: dict[str, str] = {}
    for spec in profile.page_types.values():
        source = profile.template_root / spec.template if profile.template_root else None
        if source and source.exists():
            target = profile_root / "templates" / spec.template
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            template_hashes[target.relative_to(run_dir).as_posix()] = sha256_file(target)
    provider_hashes: dict[str, str] = {}
    for source in sorted(fixture_dir.glob("*.json")):
        target = fixture_root / source.name
        shutil.copyfile(source, target)
        provider_hashes[target.relative_to(run_dir).as_posix()] = sha256_file(target)
    provider_config = snapshot_root / "provider_config.json"
    write_json_atomic(provider_config, {"provider": "mock:fixture", "fixture_files": sorted(provider_hashes)})
    provider_hashes[provider_config.relative_to(run_dir).as_posix()] = sha256_file(provider_config)
    return {
        "profile": sha256_file(profile_file),
        "templates": template_hashes,
        "providers": provider_hashes,
    }


def build_apply_preview(vault: Path, run_dir: Path) -> ApplyPreview:
    operation_id = run_dir.name
    targets: list[ApplyTarget] = []
    for draft in sorted((run_dir / "draft_pages").rglob("*.md")):
        target = vault / "wiki" / draft.relative_to(run_dir / "draft_pages")
        if target.exists():
            preimage = sha256_file(target)
            missing = False
        else:
            preimage = None
            missing = True
        targets.append(
            ApplyTarget(
                draft_path=draft.relative_to(run_dir).as_posix(),
                target_path=target.relative_to(vault).as_posix(),
                preimage_sha256=preimage,
                preimage_missing=missing,
            )
        )
    return ApplyPreview(operation_id=operation_id, targets=targets)


def archive_downstream(vault: Path, operation_id: str, start_step: str) -> None:
    store = RunStore(vault)
    manifest = read_manifest(store.manifest_path(operation_id))
    run_dir = store.run_dir(operation_id)
    paths: list[Path] = []
    names = {step.name for step in downstream_steps(start_step)}
    for step in manifest.steps:
        if step.name not in names:
            continue
        for ref in step.outputs:
            paths.append(run_dir / ref.relative_path)
    archive_paths(vault, operation_id, paths)


def status(vault: Path, operation_id: str) -> OperationManifest:
    return read_manifest(RunStore(vault).manifest_path(operation_id))


def latest_operation(vault: Path) -> str | None:
    root = RunStore(vault).runs_root
    if not root.exists():
        return None
    candidates = sorted([path for path in root.iterdir() if path.is_dir()])
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
