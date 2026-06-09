from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import errors as _errors
from . import wiki_context as _wiki_context
from .manifest import complete_step, get_step
from .models import ArtifactRef, OperationManifest, WikiContextSnapshot
from .provider_config import ProviderExecutionContext
from .steps import step_output_dir
from .structured import StructuredModelCall


@dataclass(frozen=True)
class StepRunContext:
    vault: Path
    run_dir: Path
    raw_path: Path
    profile: Any
    manifest: OperationManifest
    execution_context: ProviderExecutionContext


def structured_call(
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


def ensure_wiki_context_current(vault: Path, snapshot: WikiContextSnapshot) -> None:
    messages = _wiki_context.wiki_context_drift_messages(vault, snapshot)
    if messages:
        raise _errors.PipelineError("; ".join(messages))
