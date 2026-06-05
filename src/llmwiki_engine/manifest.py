from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import ValidationError as PydanticValidationError

from .hash_utils import sha256_file
from .io import read_json, write_json_atomic
from .models import ArtifactRef, OperationManifest, OperationStatus, StepAttempt, StepRecord, StepStatus, utc_now
from .steps import STEP_NAMES

MVP_PIPELINE_INCOMPATIBLE = "operation is incompatible with current MVP pipeline; rerun ingest"
REQUIRED_MANIFEST_KEYS = frozenset(OperationManifest.model_fields)


def read_manifest(path: Path) -> OperationManifest:
    try:
        data = read_json(path)
    except FileNotFoundError as exc:
        raise ValueError(f"Operation manifest not found: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(MVP_PIPELINE_INCOMPATIBLE)
    schema_version = data.get("schema_version")
    if schema_version != "operation_manifest.v8":
        raise ValueError(MVP_PIPELINE_INCOMPATIBLE)
    if REQUIRED_MANIFEST_KEYS - set(data):
        raise ValueError(MVP_PIPELINE_INCOMPATIBLE)
    try:
        manifest = OperationManifest.model_validate(data)
    except PydanticValidationError as exc:
        raise ValueError(MVP_PIPELINE_INCOMPATIBLE) from exc
    if tuple(step.name for step in manifest.steps) != STEP_NAMES:
        raise ValueError(MVP_PIPELINE_INCOMPATIBLE)
    return manifest


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
) -> StepAttempt:
    return _begin_step_attempt(manifest, name, inputs=inputs)


def begin_model_step_attempt(
    manifest: OperationManifest,
    name: str,
    *,
    provider_record_id: str,
    provider_spec: str,
    provider_context_source: Literal["initial_run", "resume_current_config"],
    inputs: list[ArtifactRef] | None = None,
) -> StepAttempt:
    return _begin_step_attempt(
        manifest,
        name,
        inputs=inputs,
        provider_record_id=provider_record_id,
        provider_spec=provider_spec,
        provider_context_source=provider_context_source,
    )


def _begin_step_attempt(
    manifest: OperationManifest,
    name: str,
    inputs: list[ArtifactRef] | None = None,
    *,
    provider_record_id: str | None = None,
    provider_spec: str | None = None,
    provider_context_source: Literal["initial_run", "resume_current_config"] | None = None,
) -> StepAttempt:
    step = get_step(manifest, name)
    step.status = StepStatus.running
    step.started_at = utc_now()
    step.completed_at = None
    step.error = None
    step.review_reason = None
    step.review_state = "none"
    step.awaiting_since = None
    step.resolved_at = None
    step.review_decision_ref = None
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
        step.attempts[-1].duration_ms = _duration_ms(step.attempts[-1].started_at, step.completed_at)
        step.attempts[-1].outputs = step.outputs


def mark_step_awaiting_review(
    manifest: OperationManifest,
    name: str,
    *,
    outputs: list[ArtifactRef] | None = None,
    reason: str | None = None,
    review_decision_ref: str | None = None,
) -> None:
    step = get_step(manifest, name)
    step.status = StepStatus.awaiting_review
    step.completed_at = utc_now()
    step.error = None
    step.review_reason = reason
    step.review_state = "awaiting"
    step.awaiting_since = step.completed_at
    step.resolved_at = None
    step.review_decision_ref = review_decision_ref
    step.outputs = outputs or []
    if step.attempts:
        step.attempts[-1].completed_at = step.completed_at
        step.attempts[-1].duration_ms = _duration_ms(step.attempts[-1].started_at, step.completed_at)
        step.attempts[-1].outputs = step.outputs
        step.attempts[-1].error = None
    manifest.status = OperationStatus.awaiting_review


def mark_step_approved(
    manifest: OperationManifest,
    name: str,
    *,
    outputs: list[ArtifactRef] | None = None,
) -> None:
    step = get_step(manifest, name)
    step.status = StepStatus.approved
    step.completed_at = utc_now()
    step.error = None
    step.review_reason = None
    step.review_state = "approved"
    step.resolved_at = utc_now()
    step.outputs = outputs or step.outputs
    if step.attempts:
        if step.attempts[-1].completed_at is None:
            step.attempts[-1].completed_at = step.completed_at
            step.attempts[-1].duration_ms = _duration_ms(step.attempts[-1].started_at, step.completed_at)
        step.attempts[-1].outputs = step.outputs
        step.attempts[-1].error = None
    manifest.status = OperationStatus.running


def fail_step(manifest: OperationManifest, name: str, error: str) -> None:
    step = get_step(manifest, name)
    step.status = StepStatus.failed
    step.completed_at = utc_now()
    step.error = error
    if step.attempts:
        step.attempts[-1].completed_at = step.completed_at
        step.attempts[-1].duration_ms = _duration_ms(step.attempts[-1].started_at, step.completed_at)
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
            step.review_reason = None
            step.review_state = "none"
            step.awaiting_since = None
            step.resolved_at = None
            step.review_decision_ref = None


def first_resumable_step(manifest: OperationManifest) -> str | None:
    for step in manifest.steps:
        if step.status in {StepStatus.failed, StepStatus.pending}:
            return step.name
    return None


def step_satisfied(status: StepStatus) -> bool:
    return status in {StepStatus.completed, StepStatus.approved, StepStatus.skipped}


def raw_ref(path: Path) -> tuple[str, int]:
    return sha256_file(path), path.stat().st_size


def _duration_ms(started_at: str, completed_at: str) -> int:
    start = datetime.fromisoformat(started_at)
    end = datetime.fromisoformat(completed_at)
    return max(0, round((end - start).total_seconds() * 1000))
