from __future__ import annotations

from pathlib import Path

from .hash_utils import sha256_file
from .io import read_json, write_json_atomic
from .models import ArtifactRef, OperationManifest, OperationStatus, StepAttempt, StepRecord, StepStatus, utc_now
from .steps import STEP_NAMES


def read_manifest(path: Path) -> OperationManifest:
    data = read_json(path)
    schema_version = data.get("schema_version")
    if schema_version and schema_version != "operation_manifest.v3":
        raise ValueError(f"Unsupported manifest schema_version: {schema_version}. Create a new operation.")
    return OperationManifest.model_validate(data)


def write_manifest(path: Path, manifest: OperationManifest) -> None:
    manifest.updated_at = utc_now()
    write_json_atomic(path, manifest)


def initial_steps() -> list[StepRecord]:
    return [StepRecord(name=name) for name in STEP_NAMES]


def get_step(manifest: OperationManifest, name: str) -> StepRecord:
    for step in manifest.steps:
        if step.name == name:
            return step
    raise KeyError(name)


def begin_step_attempt(
    manifest: OperationManifest,
    name: str,
    inputs: list[ArtifactRef] | None = None,
    *,
    provider_record_id: str | None = None,
    provider_spec: str | None = None,
    provider_context_source: str | None = None,
) -> StepAttempt:
    step = get_step(manifest, name)
    step.status = StepStatus.running
    step.started_at = utc_now()
    step.completed_at = None
    step.error = None
    attempt = StepAttempt(
        attempt=len(step.attempts) + 1,
        started_at=step.started_at,
        inputs=inputs or [],
        provider_record_id=provider_record_id,
        provider_spec=provider_spec,
        provider_context_source=provider_context_source,
    )
    step.inputs = attempt.inputs
    step.attempts.append(attempt)
    manifest.status = OperationStatus.running
    return attempt


def complete_step(
    manifest: OperationManifest,
    name: str,
    *,
    outputs: list[ArtifactRef] | None = None,
    status: StepStatus = StepStatus.completed,
) -> None:
    step = get_step(manifest, name)
    step.status = status
    step.completed_at = utc_now()
    step.error = None
    step.outputs = outputs or []
    if step.attempts:
        step.attempts[-1].completed_at = step.completed_at
        step.attempts[-1].outputs = step.outputs


def fail_step(manifest: OperationManifest, name: str, error: str) -> None:
    step = get_step(manifest, name)
    step.status = StepStatus.failed
    step.completed_at = utc_now()
    step.error = error
    if step.attempts:
        step.attempts[-1].completed_at = step.completed_at
        step.attempts[-1].error = error
    manifest.status = OperationStatus.failed


def mark_from_pending(manifest: OperationManifest, start: str) -> None:
    seen = False
    for step in manifest.steps:
        if step.name == start:
            seen = True
        if seen:
            step.status = StepStatus.pending
            step.started_at = None
            step.completed_at = None
            step.error = None
            step.inputs = []
            step.outputs = []
            for attempt in step.attempts:
                attempt.outputs = []


def first_resumable_step(manifest: OperationManifest) -> str | None:
    for step in manifest.steps:
        if step.status in {StepStatus.failed, StepStatus.pending}:
            return step.name
    return None


def raw_ref(path: Path) -> tuple[str, int]:
    return sha256_file(path), path.stat().st_size
