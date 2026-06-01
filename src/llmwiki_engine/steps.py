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
    StepSpec("raw_prepare", True, "raw_prepare"),
    StepSpec("raw_index", False, "raw_index"),
    StepSpec("extraction_windows", False, "extraction_windows", eval_supported=True),
    StepSpec("claim_extraction", True, "claim_extraction", eval_supported=True),
    StepSpec("page_planning", True, "page_planning", eval_supported=True),
    StepSpec("draft_rendering", False, "draft_rendering"),
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
