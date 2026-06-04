from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StepSpec:
    name: str
    model_backed: bool
    output_dir: str | None
    eval_supported: bool = False


STEP_SPECS: tuple[StepSpec, ...] = (
    StepSpec("raw_link_cleanup", False, "raw_link_cleanup"),
    StepSpec("raw_prepare", True, "raw_prepare"),
    StepSpec("prepared_raw_review", False, "prepared_raw_review"),
    StepSpec("source_digest", True, "source_digest", eval_supported=True),
    StepSpec("source_digest_review", False, "source_digest_review"),
    StepSpec("source_duplicate_guard", False, "source_duplicate_guard"),
    StepSpec("candidate_resolution", True, "candidate_resolution"),
    StepSpec("wiki_context_snapshot", False, "wiki_context_snapshot"),
    StepSpec("wiki_merge_planning", True, "wiki_merge_planning"),
    StepSpec("merge_plan_review", False, "merge_plan_review"),
    StepSpec("draft_rendering", True, "draft_rendering"),
    StepSpec("draft_review", False, "draft_review"),
    StepSpec("validation", False, None),
    StepSpec("apply_preview", False, "apply_preview"),
)

STEP_NAMES: tuple[str, ...] = tuple(spec.name for spec in STEP_SPECS)
MODEL_BACKED_STEPS: tuple[str, ...] = tuple(spec.name for spec in STEP_SPECS if spec.model_backed)
EVAL_MODULES: tuple[str, ...] = tuple(spec.name for spec in STEP_SPECS if spec.eval_supported)
PROVIDER_CONFIG_KEYS: tuple[str, ...] = ("default", *MODEL_BACKED_STEPS)


def step_spec(name: str) -> StepSpec:
    for spec in STEP_SPECS:
        if spec.name == name:
            return spec
    raise ValueError(f"Unknown step: {name}")


def step_index(name: str) -> int:
    try:
        return STEP_NAMES.index(name)
    except ValueError as exc:
        raise ValueError(f"Unknown step: {name}") from exc


def downstream_steps(name: str) -> list[str]:
    return list(STEP_NAMES[step_index(name) :])


def step_output_dir(run_dir: Path, step_name: str) -> Path | None:
    output_dir = step_spec(step_name).output_dir
    if output_dir is None:
        return None
    return run_dir / output_dir


def require_step_output_dir(run_dir: Path, step_name: str) -> Path:
    output_dir = step_output_dir(run_dir, step_name)
    if output_dir is None:
        raise ValueError(f"Step has no output directory: {step_name}")
    return output_dir
