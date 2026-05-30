from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from rich.console import Console

from . import __version__
from .events import EventLogger
from .hash_utils import artifact_ref, sha256_bytes, sha256_file
from .io import read_model, read_yaml, write_json, write_yaml
from .manifest import (
    begin_step_attempt,
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
    ClaimsArtifact,
    ExtractionWindow,
    ExtractionWindowsArtifact,
    OperationManifest,
    OperationStatus,
    PagePlanArtifact,
    ProviderContextRecord,
    ProviderRuntimeSpec,
    RawBinding,
    RawIndexArtifact,
    RawPreparationArtifact,
    RawSpan,
    RunMode,
    StepStatus,
    utc_now,
)
from .profiles import load_profile
from .providers import Provider, ProviderRegistry
from .rendering import normalized_page_plan, render_drafts
from .steps import STEP_NAMES, downstream_steps
from .structured import StructuredModelCall
from .validators import validate_claims, validate_extraction_windows, validate_page_plan, validate_raw_index, validate_raw_preparation
from .verify import require_verified
from .workspace import RunStore, ensure_v2_layout, relative_to_vault, resolve_raw_path, run_lock


class PipelineError(RuntimeError):
    pass


RAW_PREPARE_CONTRACT = {
    "goal": "Create a higher-quality canonical prepared raw for downstream knowledge compilation.",
    "rules": [
        "Do not add facts that are not supported by the original raw.",
        "Remove or relocate non-content noise such as media timestamps, self-promotion, and obvious formatting artifacts.",
        "Correct obvious ASR/OCR/formatting errors only when the context makes the correction clear.",
        "Record uncertainty instead of guessing.",
        "Return prepared_markdown as clean Markdown suitable for indexing and extraction windows.",
    ],
}

EXTRACTION_WINDOW_STRATEGY = "deterministic_span_window"
DEFAULT_EXTRACTION_WINDOW_MAX_CHARS = 2200
DEFAULT_EXTRACTION_WINDOW_OVERLAP_SPANS = 1
MODEL_STEPS = ("raw_prepare", "claim_extraction", "page_planning")


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
                "raw_prepare": "mock:fixture",
                "claim_extraction": "mock:fixture",
                "page_planning": "mock:fixture",
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
    provider_specs = load_provider_specs(vault)
    provider_context = build_provider_context_record(
        vault=vault,
        manifest_contexts=[],
        provider_specs=provider_specs,
        fixture_dir=fixture_dir,
        source="initial_run",
        from_step=None,
        affected_steps=list(MODEL_STEPS),
    )
    manifest = OperationManifest(
        operation_id=operation_id,
        operation_type="ingest",
        run_mode=run_mode,
        engine_version=__version__,
        profile=profile.name,
        profile_version=profile.version,
        workspace=relative_to_vault(vault, run_dir),
        raw_bindings=[RawBinding(relative_path=raw_rel, sha256=raw_hash, size_bytes=raw_size)],
        provider_contexts=[provider_context],
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
    refresh_providers: bool = False,
    run_mode: RunMode | None = None,
    console: Console | None = None,
) -> OperationManifest:
    store = RunStore(vault)
    with run_lock(vault, operation_id):
        manifest = read_manifest(store.manifest_path(operation_id))
        if manifest.status == OperationStatus.applied:
            raise PipelineError("Applied operations are immutable. Start a new operation instead.")
        if refresh_providers and from_step is None:
            raise PipelineError("--refresh-providers requires --from STEP.")
        require_verified(vault, manifest)
        start = from_step or first_resumable_step(manifest)
        if start is None:
            return manifest
        if refresh_providers:
            provider_specs = load_provider_specs(vault)
            affected_steps = model_steps_from(start)
            if not affected_steps:
                raise PipelineError(f"--refresh-providers has no provider-backed steps from {start}.")
            provider_context = build_provider_context_record(
                vault=vault,
                manifest_contexts=manifest.provider_contexts,
                provider_specs=provider_specs,
                fixture_dir=None,
                source="resume_refresh",
                from_step=start,
                affected_steps=affected_steps,
            )
            validate_provider_context(provider_context)
            manifest.provider_contexts.append(provider_context)
            if console is not None:
                console.print(
                    f"[yellow]Refreshing providers[/] from [bold]{start}[/] "
                    f"for model steps: {', '.join(affected_steps) or 'none'}"
                )
        if from_step is not None:
            delete_downstream_step_dirs(vault, operation_id, from_step)
            mark_from_pending(manifest, from_step)
            if run_mode is not None:
                manifest.run_mode = run_mode
            write_manifest(store.manifest_path(operation_id), manifest)
        return execute_ingest(vault, operation_id, start_step=start, console=console)


def execute_ingest(vault: Path, operation_id: str, *, start_step: str, console: Console | None = None) -> OperationManifest:
    store = RunStore(vault)
    run_dir = store.run_dir(operation_id)
    logger = EventLogger(operation_id, run_dir / "events.jsonl", console=console)
    manifest = read_manifest(store.manifest_path(operation_id))
    profile = load_profile(vault / ".llmwiki" / "profiles" / manifest.profile)
    raw_path = vault / manifest.raw_bindings[0].relative_path
    start_index = STEP_NAMES.index(start_step)
    for step_name in STEP_NAMES[start_index:]:
        if get_step(manifest, step_name).status == StepStatus.completed:
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
            )
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
    manifest: OperationManifest,
    logger: EventLogger,
) -> None:
    logger.emit(step_name, "started", status="running")
    provider_record = provider_context_for_step(manifest, step_name) if step_name in MODEL_STEPS else None
    provider_runtime = provider_record.providers.get(step_name) if provider_record else None
    begin_step_attempt(
        manifest,
        step_name,
        provider_record_id=provider_record.record_id if provider_record else None,
        provider_spec=provider_runtime.spec if provider_runtime else None,
        provider_context_source=provider_record.source if provider_record else None,
    )
    write_manifest(manifest_path, manifest)
    if step_name == "raw_prepare":
        step_root = step_dir(run_dir, "raw_prepare")
        raw_rel = relative_to_vault(vault, raw_path)
        payload = {
            "source_raw_path": raw_rel,
            "source_raw_sha256": sha256_file(raw_path),
            "original_markdown": raw_path.read_text(encoding="utf-8"),
            "contract": RAW_PREPARE_CONTRACT,
        }
        preparation, _ = _structured_call(run_dir, manifest, "raw_prepare").run(
            "raw_prepare",
            payload,
            RawPreparationArtifact,
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
            _ref(run_dir, out, step_name, "json", "raw_preparation.v0"),
            _ref(run_dir, prepared, step_name, "markdown"),
            _ref(run_dir, review, step_name, "markdown"),
        ]
        provider_result = step_root / "provider_result.json"
        if provider_result.exists():
            outputs.append(_ref(run_dir, provider_result, step_name, "provider_result", "provider_result.v1"))
        complete_step(manifest, step_name, outputs=outputs)
    elif step_name == "raw_index":
        prepared_path = step_dir(run_dir, "raw_prepare") / "prepared.md"
        raw_index = build_raw_index(vault, prepared_path, original_raw_path=raw_path)
        out = step_dir(run_dir, "raw_index") / "raw_index.json"
        write_json(out, raw_index)
        complete_step(manifest, step_name, outputs=[_ref(run_dir, out, step_name, "json", "raw_index.v1")])
    elif step_name == "extraction_windows":
        raw_index = read_model(step_dir(run_dir, "raw_index") / "raw_index.json", RawIndexArtifact)
        windows = build_extraction_windows(raw_index)
        out = step_dir(run_dir, "extraction_windows") / "extraction_windows.json"
        write_json(out, windows)
        complete_step(manifest, step_name, outputs=[_ref(run_dir, out, step_name, "json", "extraction_windows.v0")])
    elif step_name == "claim_extraction":
        step_root = step_dir(run_dir, "claim_extraction")
        raw_index = read_model(step_dir(run_dir, "raw_index") / "raw_index.json", RawIndexArtifact)
        windows = read_model(step_dir(run_dir, "extraction_windows") / "extraction_windows.json", ExtractionWindowsArtifact)
        payload = {"raw_index": raw_index.model_dump(mode="json"), "extraction_windows": windows.model_dump(mode="json")}
        claims, _ = _structured_call(run_dir, manifest, "claim_extraction").run(
            "claim_extraction",
            payload,
            ClaimsArtifact,
        )
        out = step_root / "claims.json"
        write_json(out, claims)
        outputs = [_ref(run_dir, out, step_name, "json", "claims.v1")]
        provider_result = step_root / "provider_result.json"
        if provider_result.exists():
            outputs.append(_ref(run_dir, provider_result, step_name, "provider_result", "provider_result.v1"))
        complete_step(manifest, step_name, outputs=outputs)
    elif step_name == "page_planning":
        step_root = step_dir(run_dir, "page_planning")
        claims = read_model(step_dir(run_dir, "claim_extraction") / "claims.json", ClaimsArtifact)
        payload = {"profile": profile.model_dump(mode="json"), "claims": claims.model_dump(mode="json")}
        plan, _ = _structured_call(run_dir, manifest, "page_planning").run(
            "page_planning",
            payload,
            PagePlanArtifact,
        )
        raw_index = read_model(step_dir(run_dir, "raw_index") / "raw_index.json", RawIndexArtifact)
        plan = normalized_page_plan(raw_index, claims, plan)
        out = step_root / "page_plan.json"
        write_json(out, plan)
        outputs = [_ref(run_dir, out, step_name, "json", "page_plan.v1")]
        provider_result = step_root / "provider_result.json"
        if provider_result.exists():
            outputs.append(_ref(run_dir, provider_result, step_name, "provider_result", "provider_result.v1"))
        complete_step(manifest, step_name, outputs=outputs)
    elif step_name == "draft_rendering":
        raw_index = read_model(step_dir(run_dir, "raw_index") / "raw_index.json", RawIndexArtifact)
        claims = read_model(step_dir(run_dir, "claim_extraction") / "claims.json", ClaimsArtifact)
        plan = read_model(step_dir(run_dir, "page_planning") / "page_plan.json", PagePlanArtifact)
        outputs = render_drafts(
            draft_root=step_dir(run_dir, "draft_rendering") / "draft_pages",
            profile=profile,
            raw_index=raw_index,
            claims=claims,
            plan=plan,
        )
        complete_step(manifest, step_name, outputs=[_ref(run_dir, path, step_name, "markdown") for path in outputs])
    elif step_name == "validation":
        preparation = read_model(step_dir(run_dir, "raw_prepare") / "raw_preparation.json", RawPreparationArtifact)
        raw_index = read_model(step_dir(run_dir, "raw_index") / "raw_index.json", RawIndexArtifact)
        windows = read_model(step_dir(run_dir, "extraction_windows") / "extraction_windows.json", ExtractionWindowsArtifact)
        claims = read_model(step_dir(run_dir, "claim_extraction") / "claims.json", ClaimsArtifact)
        plan = read_model(step_dir(run_dir, "page_planning") / "page_plan.json", PagePlanArtifact)
        validate_raw_preparation(preparation)
        validate_raw_index(raw_index)
        validate_extraction_windows(raw_index, windows)
        validate_claims(raw_index, windows, claims)
        validate_page_plan(profile, claims, plan)
        complete_step(manifest, step_name)
    elif step_name == "apply_preview":
        preview = build_apply_preview(vault, run_dir)
        out = step_dir(run_dir, "apply_preview") / "apply_preview.json"
        write_json(out, preview)
        complete_step(manifest, step_name, outputs=[_ref(run_dir, out, step_name, "json", "apply_preview.v1")])
    else:
        raise PipelineError(f"Unknown step: {step_name}")
    logger.emit(step_name, "completed", status="completed")


def build_raw_index(vault: Path, raw_path: Path, *, original_raw_path: Path | None = None) -> RawIndexArtifact:
    text = raw_path.read_text(encoding="utf-8")
    digest = sha256_bytes(text.encode("utf-8"))
    spans: list[RawSpan] = []
    cursor = 0
    raw_rel = relative_to_vault(vault, raw_path)
    original_raw_rel = relative_to_vault(vault, original_raw_path) if original_raw_path is not None else None
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
    return RawIndexArtifact(
        raw_path=raw_rel,
        raw_sha256=digest,
        spans=spans,
        input_kind="prepared_raw" if original_raw_rel is not None else "original_raw",
        original_raw_path=original_raw_rel,
    )


def build_extraction_windows(
    raw_index: RawIndexArtifact,
    *,
    max_chars: int = DEFAULT_EXTRACTION_WINDOW_MAX_CHARS,
    overlap_spans: int = DEFAULT_EXTRACTION_WINDOW_OVERLAP_SPANS,
) -> ExtractionWindowsArtifact:
    windows: list[ExtractionWindow] = []
    current: list[RawSpan] = []
    current_chars = 0
    for span in raw_index.spans:
        span_len = len(span.text)
        if current and current_chars + span_len > max_chars:
            windows.append(_window_from_spans(current, len(windows) + 1))
            current = current[-overlap_spans:] if overlap_spans else []
            current_chars = sum(len(item.text) for item in current)
        current.append(span)
        current_chars += span_len
    if current:
        windows.append(_window_from_spans(current, len(windows) + 1))
    return ExtractionWindowsArtifact(
        raw_path=raw_index.raw_path,
        raw_sha256=raw_index.raw_sha256,
        strategy=EXTRACTION_WINDOW_STRATEGY,
        max_chars=max_chars,
        overlap_spans=overlap_spans,
        windows=windows,
    )


def _window_from_spans(spans: list[RawSpan], index: int) -> ExtractionWindow:
    return ExtractionWindow(
        window_id=f"W{index:03d}",
        source_span_ids=[span.span_id for span in spans],
        strategy=EXTRACTION_WINDOW_STRATEGY,
        reason="Generated for extraction context; not a knowledge-unit judgment.",
        text="\n\n".join(span.text for span in spans),
        extract_policy="extract",
    )


def render_preparation_review(preparation: RawPreparationArtifact) -> str:
    operations = "\n".join(f"- {operation}" for operation in preparation.operations_applied) or "- none recorded"
    uncertain = "\n".join(
        f"- [{item.severity}] {item.item}: {item.reason}" for item in preparation.uncertain_items
    ) or "- none recorded"
    return (
        "# Raw Preparation Review\n\n"
        f"- Source raw: `{preparation.source_raw_path}`\n"
        f"- Document kind: `{preparation.document_kind}`\n"
        f"- Risk level: `{preparation.risk_level}`\n"
        f"- Requires human review: `{str(preparation.requires_human_review).lower()}`\n"
        f"- Omission policy: `{preparation.omission_policy}`\n\n"
        "## Operations Applied\n\n"
        f"{operations}\n\n"
        "## Uncertain Items\n\n"
        f"{uncertain}\n\n"
        "## Review Notes\n\n"
        f"{preparation.review_notes or 'No review notes.'}\n"
    )


def load_provider_specs(vault: Path) -> dict[str, str | dict]:
    config = read_yaml(vault / ".llmwiki" / "config.yaml")
    providers = config.get("providers", {})
    if not isinstance(providers, dict):
        raise PipelineError(".llmwiki/config.yaml providers must be a mapping.")
    return dict(providers)


def build_provider_context_record(
    *,
    vault: Path,
    manifest_contexts: list[ProviderContextRecord],
    provider_specs: dict[str, Any],
    fixture_dir: Path | None,
    source: str,
    from_step: str | None,
    affected_steps: list[str],
) -> ProviderContextRecord:
    providers: dict[str, ProviderRuntimeSpec] = {}
    for task in affected_steps:
        config = provider_config_for_task(provider_specs, task)
        providers[task] = provider_runtime_spec_for_task(
            vault,
            task,
            config,
            fixture_dir=fixture_dir,
            previous=previous_provider_runtime(manifest_contexts, task),
        )
    return ProviderContextRecord(
        record_id=f"provider-context-{len(manifest_contexts) + 1:03d}",
        source=source,
        from_step=from_step,
        affected_steps=affected_steps,
        providers=providers,
    )


def provider_config_for_task(provider_specs: dict[str, Any], task: str) -> Any:
    if task in provider_specs:
        return provider_specs[task]
    if "default" in provider_specs:
        return provider_specs["default"]
    return "mock:fixture"


PROVIDER_RUNTIME_KEYS = {"spec", "endpoint", "api_key_env", "fixture_dir"}


def provider_runtime_spec_for_task(
    vault: Path,
    task: str,
    config: Any,
    *,
    fixture_dir: Path | None,
    previous: ProviderRuntimeSpec | None,
) -> ProviderRuntimeSpec:
    endpoint = None
    api_key_env = None
    config_fixture_dir = None
    if isinstance(config, dict):
        if "api_key" in config:
            raise PipelineError(f"Provider config for {task} must use api_key_env, not api_key.")
        unknown = set(config) - PROVIDER_RUNTIME_KEYS
        if unknown:
            names = ", ".join(sorted(unknown))
            raise PipelineError(f"Unsupported provider config field(s) for {task}: {names}")
        spec = config.get("spec")
        endpoint = config.get("endpoint")
        api_key_env = config.get("api_key_env")
        config_fixture_dir = config.get("fixture_dir")
    else:
        spec = config
    if not isinstance(spec, str) or not spec:
        raise PipelineError(f"Invalid provider config for task: {task}")
    if endpoint is not None and not isinstance(endpoint, str):
        raise PipelineError(f"Invalid provider endpoint for task: {task}")
    if api_key_env is not None and not isinstance(api_key_env, str):
        raise PipelineError(f"Invalid provider api_key_env for task: {task}")
    if config_fixture_dir is not None and not isinstance(config_fixture_dir, str):
        raise PipelineError(f"Invalid provider fixture_dir for task: {task}")
    provider_name = spec.partition(":")[0]
    resolved_fixture_dir = None
    if provider_name == "mock":
        if fixture_dir is not None:
            resolved_fixture_dir = fixture_dir.resolve()
        elif config_fixture_dir:
            resolved_fixture_dir = resolve_config_path(vault, config_fixture_dir)
        elif previous and previous.fixture_dir:
            resolved_fixture_dir = Path(previous.fixture_dir)
        else:
            raise PipelineError(f"mock provider for {task} requires fixture_dir")
    return ProviderRuntimeSpec(
        spec=spec,
        endpoint=endpoint,
        api_key_env=api_key_env,
        fixture_dir=resolved_fixture_dir.as_posix() if resolved_fixture_dir is not None else None,
    )


def previous_provider_runtime(manifest_contexts: list[ProviderContextRecord], task: str) -> ProviderRuntimeSpec | None:
    for record in reversed(manifest_contexts):
        runtime = record.providers.get(task)
        if runtime is not None:
            return runtime
    return None


def resolve_config_path(vault: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (vault / path).resolve()


def validate_provider_context(record: ProviderContextRecord) -> None:
    for task in record.affected_steps:
        provider_for_task(record, task)


def provider_context_for_step(manifest: OperationManifest, step_name: str) -> ProviderContextRecord:
    for record in reversed(manifest.provider_contexts):
        if step_name in record.affected_steps:
            return record
    raise PipelineError(f"No provider context found for step: {step_name}")


def model_steps_from(start_step: str) -> list[str]:
    names = set(downstream_steps(start_step))
    return [step for step in MODEL_STEPS if step in names]


def _structured_call(run_dir: Path, manifest: OperationManifest, task: str) -> StructuredModelCall:
    return StructuredModelCall(
        provider_for_task(provider_context_for_step(manifest, task), task),
        output_dir=step_dir(run_dir, task),
        result_filename="provider_result.json",
    )


def provider_for_task(record: ProviderContextRecord, task: str) -> Provider:
    if task not in record.providers:
        raise PipelineError(f"Provider context {record.record_id} does not cover task: {task}")
    runtime = record.providers[task]
    fixture_dir = Path(runtime.fixture_dir) if runtime.fixture_dir else None
    return ProviderRegistry().create(
        runtime.spec,
        fixture_dir=fixture_dir,
        endpoint=runtime.endpoint,
        api_key_env=runtime.api_key_env,
    )


def step_dir(run_dir: Path, step_name: str) -> Path:
    return run_dir / step_name


def build_apply_preview(vault: Path, run_dir: Path) -> ApplyPreview:
    operation_id = run_dir.name
    targets: list[ApplyTarget] = []
    draft_root = step_dir(run_dir, "draft_rendering") / "draft_pages"
    for draft in sorted(draft_root.rglob("*.md")):
        target = vault / "wiki" / draft.relative_to(draft_root)
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


def delete_downstream_step_dirs(vault: Path, operation_id: str, start_step: str) -> None:
    store = RunStore(vault)
    run_dir = store.run_dir(operation_id)
    for step in downstream_steps(start_step):
        shutil.rmtree(step_dir(run_dir, step), ignore_errors=True)


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
