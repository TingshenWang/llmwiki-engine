from __future__ import annotations

import json
from pathlib import Path

from .io import read_json, write_json
from .models import EvalCaseResult, EvalRun


SUPPORTED_MODULES = {
    "extraction_windows",
    "claim_extraction",
    "page_planning",
    "section_fill",
    "critic_review",
}


def run_eval(module: str, dataset: Path, output_root: Path) -> EvalRun:
    if module not in SUPPORTED_MODULES:
        raise ValueError(f"Unsupported eval module: {module}")
    results: list[EvalCaseResult] = []
    for case_dir in sorted(path for path in dataset.iterdir() if path.is_dir()):
        actual_path = case_dir / "actual.json"
        expected_path = case_dir / "expected.json"
        errors: list[str] = []
        schema_valid = actual_path.exists()
        parse_success = False
        page_type_accuracy = None
        if actual_path.exists():
            try:
                actual = read_json(actual_path)
                parse_success = True
                if expected_path.exists():
                    expected = read_json(expected_path)
                    page_type_accuracy = _page_type_accuracy(actual, expected)
            except (json.JSONDecodeError, OSError) as exc:
                errors.append(str(exc))
        else:
            errors.append("actual.json missing")
        results.append(
            EvalCaseResult(
                case_id=case_dir.name,
                schema_valid=schema_valid,
                parse_success=parse_success,
                page_type_accuracy=page_type_accuracy,
                errors=errors,
            )
        )
    run = EvalRun(run_id=f"EVAL-{module}-{len(results)}", module=module, dataset=str(dataset), results=results)
    out = output_root / f"{run.run_id}.json"
    write_json(out, run)
    return run


def load_eval_report(path: Path) -> EvalRun:
    return EvalRun.model_validate(read_json(path))


def _page_type_accuracy(actual: dict, expected: dict) -> float | None:
    actual_pages = actual.get("pages")
    expected_pages = expected.get("pages")
    if not isinstance(actual_pages, list) or not isinstance(expected_pages, list) or not expected_pages:
        return None
    actual_types = [item.get("page_type") for item in actual_pages if isinstance(item, dict)]
    expected_types = [item.get("page_type") for item in expected_pages if isinstance(item, dict)]
    matches = sum(1 for left, right in zip(actual_types, expected_types) if left == right)
    return matches / len(expected_types)
