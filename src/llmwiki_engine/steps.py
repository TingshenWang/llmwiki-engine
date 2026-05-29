from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StepSpec:
    name: str
    outputs: tuple[str, ...]
    required_for_resume: bool = True


STEP_REGISTRY: tuple[StepSpec, ...] = (
    StepSpec("raw_index", ("raw_index.json",)),
    StepSpec("semantic_aggregation", ("semantic_aggregation.json", "model_calls/semantic_aggregation.provider_result.json")),
    StepSpec("claim_extraction", ("claims.json", "model_calls/claim_extraction.provider_result.json")),
    StepSpec("page_planning", ("page_plan.json", "model_calls/page_planning.provider_result.json")),
    StepSpec("draft_rendering", ("draft_pages",)),
    StepSpec("validation", ()),
    StepSpec("apply_preview", ("apply_preview.json",)),
)


STEP_NAMES = [step.name for step in STEP_REGISTRY]


def step_index(name: str) -> int:
    try:
        return STEP_NAMES.index(name)
    except ValueError as exc:
        raise ValueError(f"Unknown step: {name}") from exc


def downstream_steps(name: str) -> list[StepSpec]:
    return list(STEP_REGISTRY[step_index(name) :])

