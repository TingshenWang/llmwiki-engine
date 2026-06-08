from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console

from .events import format_duration
from .io import read_model, write_json
from .models import ApplyPreview, OperationManifest, ProviderResult, RawLinkCleanupArtifact, StepStatus
from .models import StructuredRepairReport, WikiMergePlanArtifact
from .system_pages import format_markdown_table


REPAIR_METRIC_KEYS = (
    "attempt_count", "repair_count", "json_repair_count", "duration_ms",
    "provider_result_count", "http_attempt_count", "payload_char_count",
)


def refresh_run_metrics(run_dir: Path, manifest: OperationManifest, *, warning_console: Console | None = None) -> None:
    try:
        metrics = build_run_metrics(run_dir, manifest)
        write_json(run_dir / "run_metrics.json", metrics)
        (run_dir / "run_metrics.md").write_text(render_run_metrics_markdown(metrics), encoding="utf-8")
    except Exception as exc:
        if warning_console is not None:
            warning_console.print(f"[yellow]warning:[/] run_metrics refresh failed: {exc}")


def build_run_metrics(run_dir: Path, manifest: OperationManifest) -> dict[str, Any]:
    steps = []
    model_durations: dict[str, int] = {}
    retry_count = 0
    current_totals = _empty_repair_metrics()
    for step in manifest.steps:
        durations = [attempt.duration_ms for attempt in step.attempts if attempt.duration_ms is not None]
        total = sum(durations)
        provider_spec = step.attempts[-1].provider_spec if step.attempts else None
        provider = provider_spec or ("local:auto_review" if step.name.endswith("_review") else "local")
        retry_count += max(0, len(step.attempts) - 1)
        repair_metrics = step_repair_metrics(run_dir, step.name)
        _add_repair_metrics(current_totals, repair_metrics)
        row = {
            "name": step.name,
            "status": step.status.value,
            "attempts": len(step.attempts),
            "last_duration_ms": durations[-1] if durations else None,
            "total_duration_ms": total,
            "provider": provider,
        }
        if repair_metrics["attempt_count"]:
            row["internal_model_call_count"] = repair_metrics["attempt_count"]
            row["repair_count"] = repair_metrics["repair_count"]
            row["local_json_repair_count"] = repair_metrics["json_repair_count"]
            row["repair_duration_ms"] = repair_metrics["duration_ms"]
            row["provider_result_count"] = repair_metrics["provider_result_count"]
            row["http_attempt_count"] = repair_metrics["http_attempt_count"]
            row["payload_char_count"] = repair_metrics["payload_char_count"]
        shortcut_report = run_dir / step.name / "merge_planning_shortcut_report.json"
        if shortcut_report.exists():
            try:
                shortcut_data = json.loads(shortcut_report.read_text(encoding="utf-8"))
                row["local_shortcut"] = bool(shortcut_data.get("used"))
                row["local_shortcut_rule"] = shortcut_data.get("shortcut", "")
            except Exception:
                row["local_shortcut"] = False
        waiting_ms = awaiting_review_duration_ms(step)
        if waiting_ms is not None:
            row["awaiting_review_duration_ms"] = waiting_ms
        steps.append(row)
        if step.attempts and step.attempts[-1].provider_spec:
            model_durations[step.name] = total
    payload_steps = [
        {
            "name": str(row["name"]),
            "payload_char_count": int(row.get("payload_char_count", 0) or 0),
            "provider_result_count": int(row.get("provider_result_count", 0) or 0),
            "http_attempt_count": int(row.get("http_attempt_count", 0) or 0),
            "repair_count": int(row.get("repair_count", 0) or 0),
            "local_json_repair_count": int(row.get("local_json_repair_count", 0) or 0),
            "duration_ms": int(row.get("repair_duration_ms", 0) or 0),
        }
        for row in steps
        if int(row.get("payload_char_count", 0) or 0) > 0
    ]
    payload_steps.sort(key=lambda row: (-int(row["payload_char_count"]), str(row["name"])))
    largest_payload = payload_steps[0] if payload_steps else None
    cleanup_count = 0
    preserved_media_count = 0
    cleanup_path = run_dir / "raw_link_cleanup" / "raw_link_cleanup.json"
    if cleanup_path.exists():
        cleanup = read_model(cleanup_path, RawLinkCleanupArtifact)
        cleanup_count = cleanup.cleaned_link_count
        preserved_media_count = cleanup.preserved_media_embed_count
    created = updated = noop = 0
    plan_path = run_dir / "merge_plan_review" / "approved_merge_plan.json"
    if not plan_path.exists():
        plan_path = run_dir / "wiki_merge_planning" / "wiki_merge_plan.json"
    if plan_path.exists():
        plan = read_model(plan_path, WikiMergePlanArtifact)
        created = sum(1 for item in plan.items if item.action == "create")
        updated = sum(1 for item in plan.items if item.action == "update")
        noop = sum(1 for item in plan.items if item.action == "noop")
    written_target_count = 0
    preview_path = run_dir / "apply_preview" / "apply_preview.json"
    if preview_path.exists():
        preview = read_model(preview_path, ApplyPreview)
        written_target_count = len([target for target in preview.targets if target.will_write])
    current_attempt_duration_ms = sum(int(row.get("total_duration_ms") or 0) for row in steps)
    current_model_duration_ms = current_totals["duration_ms"]
    budget_metrics = source_digest_budget_metrics(run_dir)
    return {
        "schema_version": "run_metrics.v1",
        "operation_id": manifest.operation_id,
        "status": manifest.status.value,
        "steps": steps,
        "model_durations_ms": model_durations,
        "current_attempt_duration_ms": current_attempt_duration_ms,
        "current_model_duration_ms": current_model_duration_ms,
        "retry_count": retry_count,
        "internal_model_call_count": current_totals["attempt_count"],
        "repair_count": current_totals["repair_count"],
        "local_json_repair_count": current_totals["json_repair_count"],
        "repair_duration_ms": current_totals["duration_ms"],
        "provider_result_count": current_totals["provider_result_count"],
        "http_attempt_count": current_totals["http_attempt_count"],
        "internal_model_payload_char_count": current_totals["payload_char_count"],
        "payload_by_step": payload_steps,
        "largest_payload_step": largest_payload["name"] if largest_payload else "",
        "largest_payload_char_count": largest_payload["payload_char_count"] if largest_payload else 0,
        "created_count": created,
        "updated_count": updated,
        "noop_count": noop,
        **budget_metrics,
        "cleaned_link_count": cleanup_count,
        "preserved_media_embed_count": preserved_media_count,
        "written_target_count": written_target_count,
    }


def render_run_metrics_markdown(metrics: dict[str, Any]) -> str:
    payload_rows = [
        [
            str(row.get("name", "")),
            f"{int(row.get('payload_char_count', 0) or 0):,}",
            str(row.get("provider_result_count", 0)),
            str(row.get("http_attempt_count", 0)),
            str(row.get("repair_count", 0)),
            str(row.get("local_json_repair_count", 0)),
            format_duration(row.get("duration_ms")),
        ]
        for row in metrics.get("payload_by_step", [])
    ]
    step_rows = [
        [
            str(row.get("name", "")),
            str(row.get("status", "")),
            str(row.get("attempts", 0)),
            format_duration(row.get("last_duration_ms")),
            format_duration(row.get("total_duration_ms")),
            f"{int(row.get('payload_char_count', 0) or 0):,}" if row.get("payload_char_count") else "",
        ]
        for row in metrics.get("steps", [])
    ]
    return (
        "# Run Metrics\n\n"
        f"- Operation: `{metrics.get('operation_id', '')}`\n"
        f"- Status: `{metrics.get('status', '')}`\n"
        f"- Current model calls: `{metrics.get('internal_model_call_count', 0)}`\n"
        f"- Current HTTP attempts: `{metrics.get('http_attempt_count', 0)}`\n"
        f"- Local JSON repairs: `{metrics.get('local_json_repair_count', 0)}`\n"
        f"- Current payload chars: `{int(metrics.get('internal_model_payload_char_count', 0) or 0):,}`\n"
        f"- Largest payload step: `{metrics.get('largest_payload_step', '') or 'none'}` "
        f"({int(metrics.get('largest_payload_char_count', 0) or 0):,} chars)\n\n"
        "## Payload By Step\n\n"
        f"{format_markdown_table(['Step', 'Payload Chars', 'Provider Results', 'HTTP Attempts', 'Model Repairs', 'Local JSON Repairs', 'Model Duration'], payload_rows) if payload_rows else '_No model payloads recorded._'}\n\n"
        "## Steps\n\n"
        f"{format_markdown_table(['Step', 'Status', 'Attempts', 'Last Duration', 'Attempt Total', 'Payload Chars'], step_rows)}\n"
    )


def source_digest_budget_metrics(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "source_digest" / "source_digest_budget_report.json"
    if not path.exists():
        return {}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {
        "candidate_page_budget": report.get("budget", 0),
        "candidate_count_before_dedupe": report.get("total_formal_candidates_before_dedupe", report.get("total_formal_candidates_before_budget", 0)),
        "candidate_count_before_budget": report.get("total_formal_candidates_before_budget", 0),
        "candidate_deduped_count": report.get("deduped_count", 0),
        "candidate_selected_count": report.get("selected_count", 0),
        "candidate_deferred_count": report.get("deferred_count", 0),
        "candidate_budget_applied": bool(report.get("applied")),
    }


def _empty_repair_metrics() -> dict[str, int]:
    return {key: 0 for key in REPAIR_METRIC_KEYS}


def _add_repair_metrics(totals: dict[str, int], metrics: dict[str, int]) -> None:
    for key in REPAIR_METRIC_KEYS:
        totals[key] += metrics[key]


def step_repair_metrics(run_dir: Path, step_name: str) -> dict[str, int]:
    step_dirs = [run_dir / step_name]
    attempt_count = 0
    repair_count = 0
    json_repair_count = 0
    duration_ms = 0
    provider_result_count = 0
    http_attempt_count = 0
    payload_char_count = 0
    for step_dir in step_dirs:
        step_attempt_count = 0
        attempt_result_paths: list[Path] = []
        report_path = step_dir / "structured_repair_report.json"
        if report_path.exists():
            try:
                report = read_model(report_path, StructuredRepairReport)
                step_attempt_count = report.attempt_count
                attempt_count += step_attempt_count
                repair_count += report.repair_count
                duration_ms += report.duration_ms
                attempt_result_paths = [step_dir / attempt.provider_result_ref for attempt in report.attempts]
            except Exception:
                pass
        existing_attempt_result_paths = [path for path in attempt_result_paths if path.exists()]
        if existing_attempt_result_paths:
            provider_result_count += len(existing_attempt_result_paths)
            http_attempt_count += provider_results_http_attempt_count(existing_attempt_result_paths)
            payload_char_count += provider_results_payload_char_count(existing_attempt_result_paths)
            json_repair_count += provider_results_json_repair_count(existing_attempt_result_paths)
            continue
        provider_results_dir = step_dir / "provider_results"
        if provider_results_dir.exists():
            provider_result_paths = list(provider_results_dir.glob("attempt-*.json"))
            provider_result_count += len(provider_result_paths)
            http_attempt_count += provider_results_http_attempt_count(provider_result_paths)
            payload_char_count += provider_results_payload_char_count(provider_result_paths)
            json_repair_count += provider_results_json_repair_count(provider_result_paths)
        elif step_attempt_count:
            provider_result_count += step_attempt_count
            result_paths = [step_dir / "provider_result.json"]
            http_attempt_count += provider_results_http_attempt_count(result_paths)
            payload_char_count += provider_results_payload_char_count(result_paths)
            json_repair_count += provider_results_json_repair_count(result_paths)
        elif (step_dir / "provider_result.json").exists():
            provider_result_count += 1
            result_paths = [step_dir / "provider_result.json"]
            http_attempt_count += provider_results_http_attempt_count(result_paths)
            payload_char_count += provider_results_payload_char_count(result_paths)
            json_repair_count += provider_results_json_repair_count(result_paths)
    if attempt_count == 0:
        attempt_count = provider_result_count
    return {
        "attempt_count": attempt_count,
        "repair_count": repair_count,
        "json_repair_count": json_repair_count,
        "duration_ms": duration_ms,
        "provider_result_count": provider_result_count,
        "http_attempt_count": http_attempt_count,
        "payload_char_count": payload_char_count,
    }


def provider_results_payload_char_count(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        if not path.exists():
            continue
        try:
            result = read_model(path, ProviderResult)
            total += int(result.payload_char_count or 0)
        except Exception:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                total += int(data.get("payload_char_count", 0) or 0)
            except Exception:
                continue
    return total


def provider_results_http_attempt_count(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        if not path.exists():
            continue
        try:
            result = read_model(path, ProviderResult)
            total += max(1, int(result.http_attempt_count or 1))
        except Exception:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                total += max(1, int(data.get("http_attempt_count", 1) or 1))
            except Exception:
                continue
    return total


def provider_results_json_repair_count(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("json_repair_applied"):
            total += 1
    return total


def awaiting_review_duration_ms(step: Any) -> int | None:
    if step.status != StepStatus.approved or not step.completed_at or not step.attempts:
        return None
    attempt = step.attempts[-1]
    if not attempt.completed_at:
        return None
    start = datetime.fromisoformat(attempt.completed_at)
    end = datetime.fromisoformat(step.completed_at)
    value = max(0, round((end - start).total_seconds() * 1000))
    return value if value > 0 else None
