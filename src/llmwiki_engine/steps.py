from __future__ import annotations


STEP_NAMES: tuple[str, ...] = (
    "raw_prepare",
    "raw_index",
    "extraction_windows",
    "claim_extraction",
    "page_planning",
    "draft_rendering",
    "validation",
    "apply_preview",
)


def step_index(name: str) -> int:
    try:
        return STEP_NAMES.index(name)
    except ValueError as exc:
        raise ValueError(f"Unknown step: {name}") from exc


def downstream_steps(name: str) -> list[str]:
    return list(STEP_NAMES[step_index(name) :])
