from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .apply import ApplyError, apply_operation
from .eval import load_eval_report, run_eval
from .events import format_duration
from .io import read_json, read_jsonl, read_model
from .models import (
    OperationManifest,
    RawIngestCandidate,
    RawIngestCandidateReport,
    RawPreparePolicy,
    ReviewDecision,
    VerificationStatus,
)
from .pipeline import (
    PipelineError,
    approve_review,
    init_vault,
    latest_operation,
    resume_ingest,
    revise_review,
    run_simplified_ingest,
    scan_raw_ingest_candidates,
    status as ingest_status,
)
from .provider_checks import check_providers
from .provider_config import ProviderConfigError
from .profiles import builtin_profile_names, load_profile
from .providers import ProviderRegistry
from .raw_import import RawUrlImportError, RawUrlImportResult, import_raw_url
from .steps import STEP_NAMES
from .verify import VerifyError, verify_run
from .workspace import RunStore, WorkspaceError

app = typer.Typer(help="LLM-Wiki knowledge compilation engine.")
ingest_app = typer.Typer(help="Run and manage simplified ingest operations.")
providers_app = typer.Typer(help="Inspect and check providers.")
profile_app = typer.Typer(help="Inspect and validate profiles.")
eval_app = typer.Typer(help="Run module evals.")

app.add_typer(ingest_app, name="ingest")
app.add_typer(providers_app, name="providers")
app.add_typer(profile_app, name="profile")
app.add_typer(eval_app, name="eval")

console = Console()
VALID_RESUME_STEPS_HELP = ", ".join(STEP_NAMES)


def _raw_prepare_command_suffix(
    *,
    prepare: RawPreparePolicy | None = None,
) -> str:
    if prepare is not None:
        return f" --prepare {prepare.value}"
    return ""


@app.command()
def init(vault: Path, profile: str = "project_basic") -> None:
    """Initialize a vault with profile directories and config."""
    init_vault(vault, profile_name=profile)
    console.print(f"[green]Initialized[/] {vault} with profile [bold]{profile}[/]")


@ingest_app.command("run")
def ingest_run(
    vault: Path,
    raw: Path,
    mock_fixture_dir: Optional[Path] = typer.Option(
        None,
        "--mock-fixture-dir",
        help="Force all model-backed steps to use mock:fixture with this fixture directory.",
    ),
    profile: Optional[str] = typer.Option(None, "--profile", help="Override the vault config profile for this run."),
    slug: Optional[str] = None,
    prepare: Optional[RawPreparePolicy] = typer.Option(
        None,
        "--prepare",
        help="Raw prepare policy: auto, skip, or force.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON."),
) -> None:
    """Run simplified Ingest through draft generation."""
    try:
        run_console = Console(file=io.StringIO()) if json_output else console
        manifest = run_simplified_ingest(
            vault=vault,
            raw_file=raw,
            mock_fixture_dir=mock_fixture_dir,
            profile_name=profile,
            slug=slug,
            raw_prepare_policy=prepare,
            console=run_console,
        )
    except (PipelineError, ProviderConfigError, WorkspaceError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if json_output:
        typer.echo(json.dumps(_operation_inspect_payload(vault.expanduser().resolve(), manifest), ensure_ascii=False, indent=2))
        return
    _print_operation_outcome(manifest)


@ingest_app.command("status")
def ingest_status_cmd(
    vault: Path,
    operation_id: Optional[str] = typer.Argument(None, help="Operation id. Defaults to the latest ingest operation."),
    verify: bool = typer.Option(False, "--verify", help="Recompute raw and artifact hashes without writing files."),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Show operation status."""
    operation_id = operation_id or latest_operation(vault)
    if operation_id is None:
        raise typer.BadParameter("No ingest operation found.")
    try:
        manifest = ingest_status(vault, operation_id)
    except (WorkspaceError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if json_output:
        typer.echo(manifest.model_dump_json(indent=2))
        if verify:
            result = verify_run(vault, manifest)
            raise typer.Exit(0 if result.ok else _verify_exit_code(result))
        return
    _print_manifest_table(vault, manifest)
    if verify:
        result = verify_run(vault, manifest)
        if result.ok:
            console.print("[green]verify: ok[/]")
        else:
            console.print("[red]verify: failed[/]")
            for issue in result.issues:
                console.print(f"- {issue.code.value}: {issue.path} - {issue.message}")
            raise typer.Exit(_verify_exit_code(result))


@ingest_app.command("inspect")
def ingest_inspect(
    vault: Path,
    operation_id: Optional[str] = typer.Argument(None, help="Operation id. Defaults to the latest ingest operation."),
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON."),
) -> None:
    """Show operation summary, metrics, and artifact hints."""
    operation_id = operation_id or latest_operation(vault)
    if operation_id is None:
        raise typer.BadParameter("No ingest operation found.")
    try:
        manifest = ingest_status(vault, operation_id)
    except (WorkspaceError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    payload = _operation_inspect_payload(vault.expanduser().resolve(), manifest)
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    _print_inspect_report(payload)


@ingest_app.command("raw-candidates")
def ingest_raw_candidates(
    vault: Path,
    include_processed: bool = typer.Option(False, "--all", help="Include raw files already recorded by source pages."),
    limit: Optional[int] = typer.Option(None, "--limit", min=0, help="Maximum rows to show after status filtering."),
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON."),
) -> None:
    """List raw files that are likely ready for ingest."""
    try:
        report = scan_raw_ingest_candidates(vault, include_processed=include_processed, limit=limit)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
        return
    _print_raw_candidates_report(report)


@ingest_app.command("run-next")
def ingest_run_next(
    vault: Path,
    include_changed: bool = typer.Option(False, "--include-changed", help="Allow changed raw files when no unprocessed raw is available."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the selected raw file without running ingest."),
    mock_fixture_dir: Optional[Path] = typer.Option(
        None,
        "--mock-fixture-dir",
        help="Force all model-backed steps to use mock:fixture with this fixture directory.",
    ),
    profile: Optional[str] = typer.Option(None, "--profile", help="Override the vault config profile for this run."),
    slug: Optional[str] = None,
    prepare: Optional[RawPreparePolicy] = typer.Option(
        None,
        "--prepare",
        help="Raw prepare policy: auto, skip, or force.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON."),
) -> None:
    """Run ingest for the next safe raw candidate."""
    try:
        prepare_cli_suffix = _raw_prepare_command_suffix(
            prepare=prepare,
        )
        report = scan_raw_ingest_candidates(vault)
        candidate = _select_next_raw_candidate(report, include_changed=include_changed)
        if candidate is None:
            allowed = "unprocessed or changed" if include_changed else "unprocessed"
            raise ValueError(
                f"No {allowed} raw ingest candidate found. "
                f"summary: unprocessed={report.unprocessed_count}; changed={report.changed_count}; "
                f"duplicate_hash={report.duplicate_hash_count}; duplicate_url={report.duplicate_url_count}; "
                f"processed={report.processed_count}"
            )
        raw_abs = Path(report.vault) / candidate.raw_path
        if dry_run:
            if json_output:
                typer.echo(
                    json.dumps(
                        _run_next_payload(report, candidate, raw_abs, dry_run=True, prepare_cli_suffix=prepare_cli_suffix),
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return
            console.print(f"selected raw: `{raw_abs}`")
            console.print(f"status: `{candidate.status}`")
            console.print(f"next: `llmwiki ingest run {report.vault} {raw_abs}{prepare_cli_suffix}`")
            return
        if not json_output:
            console.print(f"selected raw: `{raw_abs}`")
        run_console = Console(file=io.StringIO()) if json_output else console
        manifest = run_simplified_ingest(
            vault=Path(report.vault),
            raw_file=raw_abs,
            mock_fixture_dir=mock_fixture_dir,
            profile_name=profile,
            slug=slug,
            raw_prepare_policy=prepare,
            console=run_console,
        )
    except (PipelineError, ProviderConfigError, WorkspaceError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if json_output:
        typer.echo(
            json.dumps(
                _run_next_payload(report, candidate, raw_abs, dry_run=False, manifest=manifest),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    _print_operation_outcome(manifest)


@ingest_app.command("raw-import-url")
def ingest_raw_import_url(
    vault: Path,
    url: str,
    title: Optional[str] = typer.Option(None, "--title", help="Override the imported raw title."),
    output_name: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Relative output path under raw/. Defaults to a safe title-derived Markdown filename.",
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite an existing raw file instead of choosing a suffix."),
    dedupe_url: bool = typer.Option(True, "--dedupe-url/--no-dedupe-url", help="Reuse an existing imported raw file with the same URL."),
    prefer_arxiv_html: bool = typer.Option(
        True,
        "--arxiv-html/--no-arxiv-html",
        help="Resolve arXiv abs/pdf URLs to arXiv HTML before fetching.",
    ),
    timeout: float = typer.Option(30.0, "--timeout", min=1.0, help="HTTP request timeout in seconds."),
    max_bytes: int = typer.Option(5_000_000, "--max-bytes", min=1024, help="Maximum fetched response size in bytes."),
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON."),
) -> None:
    """Fetch an HTML/Markdown/text URL into raw/ as Markdown."""
    try:
        result = import_raw_url(
            vault,
            url,
            title=title,
            output_name=output_name,
            overwrite=overwrite,
            dedupe_url=dedupe_url,
            prefer_arxiv_html=prefer_arxiv_html,
            timeout=timeout,
            max_bytes=max_bytes,
        )
    except (RawUrlImportError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if json_output:
        typer.echo(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return
    _print_raw_url_import_result(result)


@ingest_app.command("resume")
def ingest_resume(
    vault: Path,
    operation_id: str,
    from_step: Optional[str] = typer.Option(
        None,
        "--from",
        help=f"Resume from this step, deleting this step and downstream outputs. Valid steps: {VALID_RESUME_STEPS_HELP}.",
    ),
    mock_fixture_dir: Optional[Path] = typer.Option(
        None,
        "--mock-fixture-dir",
        help="Force resumed model-backed steps to use mock:fixture with this fixture directory.",
    ),
    prepare: Optional[RawPreparePolicy] = typer.Option(
        None,
        "--prepare",
        help="Raw prepare policy when resuming from raw_prepare or earlier: auto, skip, or force.",
    ),
) -> None:
    """Resume using the current provider config for steps that will execute."""
    try:
        manifest = resume_ingest(
            vault=vault,
            operation_id=operation_id,
            from_step=from_step,
            mock_fixture_dir=mock_fixture_dir,
            raw_prepare_policy=prepare,
            console=console,
        )
    except (PipelineError, ProviderConfigError, VerifyError, WorkspaceError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print_operation_outcome(manifest)


def _print_manifest_table(vault: Path, manifest: OperationManifest) -> None:
    table = Table(title=f"Ingest {manifest.operation_id}")
    table.add_column("Step", no_wrap=True, overflow="fold")
    table.add_column("Status", no_wrap=True)
    table.add_column("Review", overflow="fold")
    table.add_column("Attempts", justify="right")
    table.add_column("Last Duration", justify="right")
    table.add_column("Attempt Total", justify="right")
    table.add_column("Provider", overflow="fold")
    provider_labels: list[str] = []
    for step in manifest.steps:
        durations = [attempt.duration_ms for attempt in step.attempts if attempt.duration_ms is not None]
        last_duration = durations[-1] if durations else None
        total_duration = sum(durations) if durations else None
        provider_label = _provider_label(vault, manifest, step.name, step.attempts[-1].provider_spec if step.attempts else None)
        provider_labels.append(f"{step.name}={provider_label}")
        table.add_row(
            step.name,
            step.status.value,
            _review_label(vault, manifest, step.name),
            str(len(step.attempts)),
            format_duration(last_duration),
            format_duration(total_duration),
            provider_label,
        )
    console.print(table)
    console.print("columns: Step | Status | Review | Attempts | Last Duration | Attempt Total | Provider")
    console.print("duration note: Last/Attempt Total are manifest attempt durations; metrics current_* counts use current artifacts.")
    console.print("providers: " + "; ".join(provider_labels))
    reviews = [
        f"{step.name}={label}"
        for step in manifest.steps
        if (label := _review_label(vault, manifest, step.name))
    ]
    if reviews:
        console.print("reviews: " + "; ".join(reviews))
    review_reasons = [
        f"{step.name}: {step.review_reason}"
        for step in manifest.steps
        if step.status.value == "awaiting_review" and step.review_reason
    ]
    if review_reasons:
        console.print("[yellow]awaiting review:[/] " + "; ".join(review_reasons))
    console.print(f"status: [bold]{manifest.status.value}[/]")
    latest_error = next((step.error for step in reversed(manifest.steps) if step.error), None)
    if latest_error:
        console.print(f"[red]latest error:[/] {latest_error}")
    _print_metrics_summary(vault, manifest)
    _print_artifact_hints(vault, manifest)
    grounding_summary = _draft_grounding_summary(vault, manifest.operation_id)
    if grounding_summary.get("exists") and (grounding_summary.get("blocking_count") or grounding_summary.get("warning_count")):
        console.print(
            "[yellow]grounding review:[/] "
            f"blocking={grounding_summary.get('blocking_count', 0)}; "
            f"非阻塞提醒={grounding_summary.get('warning_count', 0)}; "
            f"see `{grounding_summary.get('path', '')}`"
        )
    console.print(f"next: {_next_action(manifest)}")


def _print_operation_outcome(manifest: OperationManifest) -> None:
    if manifest.status.value == "awaiting_review":
        step = next((step for step in manifest.steps if step.status.value == "awaiting_review"), None)
        step_name = step.name if step is not None else "unknown"
        console.print(f"[yellow]Operation awaiting review[/]: {step_name} ({manifest.operation_id})")
        if step is not None and step.review_reason:
            console.print(f"[yellow]reason[/]: {step.review_reason}")
        return
    if manifest.status.value == "source_recorded":
        console.print(f"[green]Source recorded[/]: {manifest.operation_id}")
        return
    if manifest.status.value == "failed":
        console.print(f"[red]Operation failed[/]: {manifest.operation_id}")
        return
    console.print(f"[green]Operation ready[/]: {manifest.operation_id}")


def _operation_inspect_payload(vault: Path, manifest: OperationManifest) -> dict[str, object]:
    metrics_path = RunStore(vault).run_dir(manifest.operation_id) / "run_metrics.json"
    metrics = None
    if metrics_path.exists():
        try:
            metrics = read_json(metrics_path)
        except Exception as exc:
            metrics = {"error": f"unreadable metrics: {exc}"}
    awaiting_review_step = _awaiting_review_step_name(manifest)
    awaiting_review_reasons = [
        {"step": step.name, "reason": step.review_reason}
        for step in manifest.steps
        if step.status.value == "awaiting_review" and step.review_reason
    ]
    failed_steps = [
        {"step": step.name, "error": step.error}
        for step in manifest.steps
        if step.status.value == "failed" or step.error
    ]
    applied_log = RunStore(vault).applied_log
    current_receipt_exists = _current_operation_receipt_exists(vault, manifest.operation_id)
    grounding_summary = _draft_grounding_summary(vault, manifest.operation_id)
    return {
        "vault": vault.as_posix(),
        "operation_id": manifest.operation_id,
        "operation_status": manifest.status.value,
        "awaiting_review_step": awaiting_review_step,
        "awaiting_review_reasons": awaiting_review_reasons,
        "failed_steps": failed_steps,
        "raw_bindings": [raw.model_dump(mode="json") for raw in manifest.raw_bindings],
        "metrics": metrics,
        "artifact_hints": _run_next_artifact_hints(vault, manifest.operation_id),
        "grounding_review": grounding_summary,
        "applied_receipt_log": applied_log.as_posix(),
        "current_operation_receipt_exists": current_receipt_exists,
        "next_action": _next_action(manifest),
    }


def _draft_grounding_summary(vault: Path, operation_id: str) -> dict[str, object]:
    path = RunStore(vault).run_dir(operation_id) / "draft_rendering" / "draft_grounding_review.json"
    if not path.exists():
        return {"path": path.as_posix(), "exists": False, "blocking_count": 0, "warning_count": 0}
    try:
        data = read_json(path)
    except Exception as exc:
        return {"path": path.as_posix(), "exists": True, "error": str(exc), "blocking_count": 0, "warning_count": 0}
    return {
        "path": path.as_posix(),
        "exists": True,
        "blocking_count": len(data.get("unsupported_new_facts", []) or []),
        "warning_count": len(data.get("warnings", []) or []),
        "requires_review": bool(data.get("requires_review", False)),
    }


def _current_operation_receipt_exists(vault: Path, operation_id: str) -> bool:
    try:
        return any(row.get("operation_id") == operation_id for row in read_jsonl(RunStore(vault).applied_log))
    except Exception:
        return False


def _print_inspect_report(payload: dict[str, object]) -> None:
    table = Table(title=f"Ingest inspect {payload['operation_id']}")
    table.add_column("Field", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("status", str(payload["operation_status"]))
    table.add_row("awaiting_review_step", str(payload.get("awaiting_review_step") or ""))
    grounding_review = payload.get("grounding_review")
    if isinstance(grounding_review, dict) and grounding_review.get("exists"):
        table.add_row("grounding_blocking_count", str(grounding_review.get("blocking_count", 0)))
        table.add_row("grounding_warning_count", str(grounding_review.get("warning_count", 0)))
        table.add_row("grounding_review", f"`{grounding_review.get('path', '')}`")
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        for key in [
            "internal_model_call_count",
            "repair_count",
            "internal_model_payload_char_count",
            "largest_payload_step",
            "largest_payload_char_count",
            "candidate_selected_count",
            "candidate_deferred_count",
        ]:
            if key in metrics:
                table.add_row(key, str(metrics.get(key)))
    table.add_row("next_action", str(payload["next_action"]))
    table.add_row("applied_receipt_log", f"`{payload.get('applied_receipt_log', '')}`")
    table.add_row("current_operation_receipt_exists", str(payload.get("current_operation_receipt_exists", False)).lower())
    console.print(table)
    hints = payload.get("artifact_hints")
    if isinstance(hints, list):
        existing = [hint for hint in hints if isinstance(hint, dict) and hint.get("exists")]
        console.print(f"artifact_hints: {len(existing)}/{len(hints)} existing")
        for hint in existing[:12]:
            console.print(f"{hint.get('label')}: `{hint.get('path')}`")


def _print_raw_candidates_report(report: RawIngestCandidateReport) -> None:
    table = Table(title=f"Raw ingest candidates ({report.candidate_count}/{report.total_raw_files})")
    table.add_column("Status", no_wrap=True)
    table.add_column("Raw", overflow="fold")
    table.add_column("Size", justify="right")
    table.add_column("Source Pages", overflow="fold")
    table.add_column("Operations", overflow="fold")
    table.add_column("Reason", overflow="fold")
    for item in report.items:
        table.add_row(
            _raw_candidate_status_label(item.status),
            item.raw_path,
            _format_bytes(item.size_bytes),
            ", ".join(item.source_pages),
            ", ".join(item.operation_ids),
            item.reason,
        )
    console.print(table)
    console.print(
        "summary: "
        f"unprocessed={report.unprocessed_count}; "
        f"changed={report.changed_count}; "
        f"duplicate_hash={report.duplicate_hash_count}; "
        f"duplicate_url={report.duplicate_url_count}; "
        f"processed={report.processed_count}"
    )
    if not report.include_processed:
        console.print("default: processed raw files are hidden; use `--all` to include them.")
    next_item = next((item for item in report.items if item.status in {"unprocessed", "changed"}), None)
    if next_item is not None:
        raw_abs = Path(report.vault) / next_item.raw_path
        console.print(f"next: `llmwiki ingest run {report.vault} {raw_abs}`")
    elif report.candidate_count == 0:
        console.print("[green]No raw ingest candidates found.[/]")


def _select_next_raw_candidate(
    report: RawIngestCandidateReport,
    *,
    include_changed: bool,
) -> RawIngestCandidate | None:
    for item in report.items:
        if item.status == "unprocessed":
            return item
    if include_changed:
        for item in report.items:
            if item.status == "changed":
                return item
    return None


def _run_next_payload(
    report: RawIngestCandidateReport,
    candidate,
    raw_abs: Path,
    *,
    dry_run: bool,
    manifest: OperationManifest | None = None,
    prepare_cli_suffix: str = "",
) -> dict[str, object]:
    operation_id = manifest.operation_id if manifest is not None else None
    operation_status = manifest.status.value if manifest is not None else None
    return {
        "vault": report.vault,
        "dry_run": dry_run,
        "selected_raw_path": candidate.raw_path,
        "selected_raw_absolute_path": raw_abs.as_posix(),
        "candidate_status": candidate.status,
        "candidate_sha256": candidate.raw_sha256,
        "operation_id": operation_id,
        "operation_status": operation_status,
        "awaiting_review_step": _awaiting_review_step_name(manifest) if manifest is not None else None,
        "artifact_hints": _run_next_artifact_hints(Path(report.vault), operation_id) if operation_id else [],
        "next_command": f"llmwiki ingest run {report.vault} {raw_abs}{prepare_cli_suffix}" if dry_run else (
            f"llmwiki ingest status {report.vault} {operation_id}" if operation_id else None
        ),
    }


def _awaiting_review_step_name(manifest: OperationManifest) -> str | None:
    for step in manifest.steps:
        if step.status.value == "awaiting_review":
            return step.name
    return None


def _run_next_artifact_hints(vault: Path, operation_id: str) -> list[dict[str, object]]:
    store = RunStore(vault)
    run_dir = store.run_dir(operation_id)
    paths = [
        ("run_dir", run_dir),
        ("manifest", store.manifest_path(operation_id)),
        ("run_metrics", run_dir / "run_metrics.json"),
        ("run_metrics_markdown", run_dir / "run_metrics.md"),
        ("prepared_markdown", run_dir / "raw_prepare" / "prepared.md"),
        ("source_digest", run_dir / "source_digest" / "source_digest.json"),
        ("candidate_budget", run_dir / "source_digest" / "source_digest_budget_report.md"),
        ("candidate_contexts", run_dir / "wiki_context_snapshot" / "candidate_contexts.md"),
        ("merge_decision_report", run_dir / "wiki_merge_planning" / "merge_decision_report.md"),
        ("draft_review", run_dir / "draft_review" / "review_prompt.md"),
        ("draft_rendering", run_dir / "draft_rendering" / "draft_rendering.json"),
        ("draft_batches", run_dir / "draft_rendering" / "draft_rendering_batch_report.md"),
        ("update_merge_report", run_dir / "draft_rendering" / "update_merge_report.md"),
        ("grounding_review", run_dir / "draft_rendering" / "draft_grounding_review.md"),
        ("global_applied_receipt_log", store.applied_log),
    ]
    return [{"label": label, "path": path.as_posix(), "exists": path.exists()} for label, path in paths]


def _print_raw_url_import_result(result: RawUrlImportResult) -> None:
    table = Table(title="Imported raw URL")
    table.add_column("Field", no_wrap=True)
    table.add_column("Value", overflow="fold")
    for key, value in [
        ("status", result.status),
        ("raw_path", result.raw_path),
        ("title", result.title),
        ("format", result.format),
        ("content_type", result.content_type or "unknown"),
        ("fetch_url", result.fetch_url),
        ("sha256", result.sha256),
        ("size_bytes", str(result.size_bytes)),
        ("overwritten", str(result.overwritten).lower()),
        ("source", result.url),
    ]:
        table.add_row(key, value)
    console.print(table)
    console.print(f"next: `llmwiki ingest run {result.vault} {Path(result.absolute_path)}`")


def _raw_candidate_status_label(status: str) -> str:
    labels = {
        "unprocessed": "[green]unprocessed[/]",
        "changed": "[yellow]changed[/]",
        "duplicate_hash": "[cyan]duplicate_hash[/]",
        "duplicate_url": "[cyan]duplicate_url[/]",
        "processed": "processed",
    }
    return labels.get(status, status)


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ["B", "KB", "MB", "GB"]:
        if value < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def _review_label(vault: Path, manifest: OperationManifest, step_name: str) -> str:
    if not step_name.endswith("_review"):
        return ""
    path = RunStore(vault).run_dir(manifest.operation_id) / step_name / "review_decision.json"
    if not path.exists():
        return ""
    try:
        decision = read_model(path, ReviewDecision)
    except Exception:
        return "decision unreadable"
    label = f"{decision.review_mode}/{decision.decision}"
    if decision.auto_approved:
        label += " (auto-approved)"
    return label


def _next_action(manifest: OperationManifest) -> str:
    if manifest.status.value == "applied":
        return "operation already applied"
    if manifest.status.value == "source_recorded":
        return "来源已记录，没有知识页变化"
    if manifest.status.value == "apply_failed":
        return "inspect apply_failed.json and written targets before retrying"
    for step in manifest.steps:
        if step.status.value == "awaiting_review":
            return f"review `{step.name}` then run `llmwiki ingest approve <vault> {manifest.operation_id} {step.name}`"
    for step in manifest.steps:
        if step.status.value in {"failed", "pending"}:
            return f"run `llmwiki ingest resume <vault> {manifest.operation_id}`"
    if manifest.status.value == "drafted":
        return f"run `llmwiki ingest apply <vault> {manifest.operation_id}`"
    return "inspect status"


def _print_artifact_hints(vault: Path, manifest: OperationManifest) -> None:
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    hints = [
        ("raw cleanup", run_dir / "raw_link_cleanup" / "raw_link_cleanup.md"),
        ("raw cleanup diff", run_dir / "raw_link_cleanup" / "cleanup.diff"),
        ("run metrics", run_dir / "run_metrics.md"),
        ("merge review", run_dir / "merge_plan_review" / "review_prompt.md"),
        ("merge decision report", run_dir / "wiki_merge_planning" / "merge_decision_report.md"),
        ("candidate contexts", run_dir / "wiki_context_snapshot" / "candidate_contexts.md"),
        ("draft review", run_dir / "draft_review" / "review_prompt.md"),
        ("structured repair", run_dir / "draft_rendering" / "structured_repair_report.md"),
        ("draft batches", run_dir / "draft_rendering" / "draft_rendering_batch_report.md"),
        ("update merge report", run_dir / "draft_rendering" / "update_merge_report.md"),
        ("grounding review", run_dir / "draft_rendering" / "draft_grounding_review.md"),
        ("related merge report", run_dir / "draft_rendering" / "related_merge_report.md"),
        ("draft root", run_dir / "draft_rendering" / "draft_pages"),
        ("diffs", run_dir / "draft_rendering" / "diffs"),
        ("apply preview", run_dir / "apply_preview" / "apply_preview.json"),
    ]
    for label, path in hints:
        if path.exists():
            console.print(f"{label}: `{path}`")
    store = RunStore(vault)
    if store.applied_log.exists():
        exists = _current_operation_receipt_exists(vault, manifest.operation_id)
        console.print("global applied receipt log path: `.llmwiki/applied/operations.jsonl`")
        console.print(f"current operation receipt: `{'present' if exists else 'not found yet'}`")


def _print_metrics_summary(vault: Path, manifest: OperationManifest) -> None:
    metrics_path = RunStore(vault).run_dir(manifest.operation_id) / "run_metrics.json"
    if not metrics_path.exists():
        return
    try:
        metrics = read_json(metrics_path)
    except Exception:
        return
    summary = (
        "metrics: "
        f"current_model_calls={metrics.get('internal_model_call_count', 0)}; "
        f"current_repairs={metrics.get('repair_count', 0)}; "
        f"current_provider_results={metrics.get('provider_result_count', 0)}; "
        f"current_attempt_duration={format_duration(metrics.get('current_attempt_duration_ms'))}; "
        f"current_model_duration={format_duration(metrics.get('current_model_duration_ms'))}"
    )
    payload_chars = int(metrics.get("internal_model_payload_char_count", 0) or 0)
    if payload_chars:
        summary += f"; current_payload_chars={payload_chars:,}"
        largest_step = str(metrics.get("largest_payload_step") or "")
        largest_chars = int(metrics.get("largest_payload_char_count", 0) or 0)
        if largest_step and largest_chars:
            summary += f"; largest_payload_step={largest_step} ({largest_chars:,})"
    archived_calls = metrics.get("archived_internal_model_call_count", 0)
    archived_duration = metrics.get("archived_model_duration_ms", 0)
    archived_payload_chars = int(metrics.get("archived_internal_model_payload_char_count", 0) or 0)
    if archived_calls or archived_duration:
        summary += (
            f"; archived_model_calls={archived_calls}; "
            f"archived_model_duration={format_duration(archived_duration)}; "
            f"total_model_calls={metrics.get('total_internal_model_call_count', archived_calls)}; "
            f"total_model_duration={format_duration(metrics.get('total_model_duration_ms'))}"
        )
        if archived_payload_chars or payload_chars:
            summary += f"; total_payload_chars={int(metrics.get('total_internal_model_payload_char_count', payload_chars + archived_payload_chars) or 0):,}"
    if "candidate_selected_count" in metrics:
        summary += (
            f"; candidates={metrics.get('candidate_selected_count', 0)}/"
            f"{metrics.get('candidate_count_before_budget', 0)}"
            f" (budget={metrics.get('candidate_page_budget', 0)}); "
            f"deduped={metrics.get('candidate_deduped_count', 0)}; "
            f"deferred={metrics.get('candidate_deferred_count', 0)}"
        )
    console.print(summary)
    bottlenecks = _metrics_bottleneck_summary(metrics)
    if bottlenecks:
        console.print("bottlenecks: " + bottlenecks)


def _metrics_bottleneck_summary(metrics: dict[str, object], *, limit: int = 3) -> str:
    payload_by_step = metrics.get("payload_by_step")
    if not isinstance(payload_by_step, list):
        return ""
    rows: list[dict[str, object]] = [row for row in payload_by_step if isinstance(row, dict)]
    rows = [
        row
        for row in rows
        if int(row.get("duration_ms", 0) or 0) > 0 or int(row.get("payload_char_count", 0) or 0) > 0
    ]
    rows.sort(
        key=lambda row: (
            int(row.get("duration_ms", 0) or 0),
            int(row.get("payload_char_count", 0) or 0),
        ),
        reverse=True,
    )
    parts: list[str] = []
    for row in rows[:limit]:
        name = str(row.get("name") or "unknown")
        duration = format_duration(int(row.get("duration_ms", 0) or 0))
        payload_chars = int(row.get("payload_char_count", 0) or 0)
        repairs = int(row.get("repair_count", 0) or 0)
        provider_results = int(row.get("provider_result_count", 0) or 0)
        details = [f"{duration}", f"{payload_chars:,} chars"]
        if provider_results:
            details.append(f"{provider_results} calls")
        if repairs:
            details.append(f"{repairs} repairs")
        parts.append(f"{name} ({', '.join(details)})")
    return "; ".join(parts)


def _provider_label(vault: Path, manifest: OperationManifest, step_name: str, provider_spec: str | None) -> str:
    if step_name == "raw_prepare" and provider_spec is None and manifest.vault_config_snapshot.raw_prepare_policy == RawPreparePolicy.skip:
        return "local:skip"
    if provider_spec:
        return provider_spec
    if step_name.endswith("_review"):
        return "local:auto_review"
    return "local"


def _verify_exit_code(result) -> int:
    codes = {issue.code for issue in result.issues}
    if VerificationStatus.raw_changed in codes:
        return 5
    if VerificationStatus.missing in codes:
        return 4
    if VerificationStatus.drift in codes:
        return 3
    return 3


@ingest_app.command("apply")
def ingest_apply(
    vault: Path,
    operation_id: str,
) -> None:
    """Apply draft pages into the vault wiki."""
    try:
        written = apply_operation(vault, operation_id)
    except (ApplyError, VerifyError, WorkspaceError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"[green]Applied[/] {len(written)} draft pages")
    console.print("applied receipt log: `.llmwiki/applied/operations.jsonl`")


@ingest_app.command("review")
def ingest_review(vault: Path, operation_id: str, review_step: str) -> None:
    """Show review artifact paths for a review step."""
    run_dir = RunStore(vault).run_dir(operation_id)
    root = run_dir / review_step
    if not root.exists():
        raise typer.BadParameter(f"Review step artifact not found: {review_step}")
    for path in sorted(root.iterdir()):
        console.print(path)


@ingest_app.command("approve")
def ingest_approve(vault: Path, operation_id: str, review_step: str) -> None:
    """Approve a pending review gate after manual inspection."""
    try:
        manifest = approve_review(vault, operation_id, review_step)
    except (PipelineError, WorkspaceError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"[green]Approved[/] {review_step} for {manifest.operation_id}")


@ingest_app.command("revise")
def ingest_revise(vault: Path, operation_id: str, review_step: str) -> None:
    """Invalidate a review step and downstream artifacts before revising."""
    try:
        manifest = revise_review(vault, operation_id, review_step)
    except (PipelineError, WorkspaceError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"[yellow]Revision reset[/] {review_step} for {manifest.operation_id}")


@providers_app.command("list")
def providers_list() -> None:
    """List available provider types."""
    table = Table(title="Providers")
    table.add_column("Name")
    for name in ProviderRegistry().names():
        table.add_row(name)
    console.print(table)


@providers_app.command("check")
def providers_check(vault: Path, live: bool = typer.Option(False, "--live", help="Run minimal live checks.")) -> None:
    """Check merged provider config without creating an ingest run."""
    result = check_providers(vault, live=live)
    table = Table(title="Provider Check")
    table.add_column("Step")
    table.add_column("Spec")
    table.add_column("Endpoint")
    table.add_column("Credential")
    table.add_column("Fixture")
    for row in result.rows:
        table.add_row(row.task, row.spec, row.endpoint or "", row.credential_label or "", row.fixture_dir or "")
    if result.rows:
        console.print(table)
    for warning in result.warnings:
        console.print(f"[yellow]warning:[/] {warning}")
    for error in result.errors:
        console.print(f"[red]error:[/] {error}")
    if not result.ok:
        raise typer.Exit(1)
    if live:
        console.print("[green]live check: ok[/]")
    console.print("[green]providers check: ok[/]")


@profile_app.command("list")
def profile_list() -> None:
    """List built-in profiles."""
    table = Table(title="Built-in profiles")
    table.add_column("Name")
    for name in builtin_profile_names():
        table.add_row(name)
    console.print(table)


@profile_app.command("validate")
def profile_validate(path_or_name: str) -> None:
    """Validate a built-in or custom profile."""
    profile = load_profile(path_or_name)
    console.print(f"[green]Profile OK[/]: {profile.name} ({len(profile.page_types)} page types)")


@eval_app.command("run")
def eval_run(module: str, dataset: Path, output_root: Path = typer.Option(Path("eval_runs"), "--output-root")) -> None:
    """Run a module eval over a fixture dataset."""
    result = run_eval(module, dataset, output_root)
    console.print(f"[green]Eval complete[/]: {result.run_id}")
    console.print(f"schema_valid_rate={result.schema_valid_rate:.2f}")
    console.print(f"parse_success_rate={result.parse_success_rate:.2f}")


@eval_app.command("report")
def eval_report(run: Path) -> None:
    """Print a saved eval report."""
    result = load_eval_report(run)
    console.print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
