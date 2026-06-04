from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .apply import ApplyError, apply_operation
from .eval import load_eval_report, run_eval
from .events import format_duration
from .io import read_model
from .models import OperationManifest, ReviewDecision, RunMode, VerificationStatus
from .pipeline import (
    PipelineError,
    approve_review,
    init_vault,
    latest_operation,
    resume_ingest,
    revise_review,
    run_simplified_ingest,
    status as ingest_status,
)
from .provider_checks import check_providers
from .provider_config import ProviderConfigError
from .profiles import builtin_profile_names, load_profile
from .providers import ProviderRegistry
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


@app.command()
def init(vault: Path, profile: str = "project_basic") -> None:
    """Initialize a vault with profile directories and config."""
    init_vault(vault, profile_name=profile)
    console.print(f"[green]Initialized[/] {vault} with profile [bold]{profile}[/]")


@ingest_app.command("run")
def ingest_run(
    vault: Path,
    raw: Path,
    fixture_dir: Optional[Path] = typer.Option(None, "--fixture-dir", help="MockProvider fixture directory."),
    profile: Optional[str] = typer.Option(None, "--profile", help="Override the vault config profile for this run."),
    slug: Optional[str] = None,
    mode: RunMode = RunMode.dev,
) -> None:
    """Run simplified Ingest through draft generation."""
    try:
        manifest = run_simplified_ingest(
            vault=vault,
            raw_file=raw,
            fixture_dir=fixture_dir,
            profile_name=profile,
            slug=slug,
            run_mode=mode,
            console=console,
        )
    except (PipelineError, ProviderConfigError, WorkspaceError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"[green]Operation ready[/]: {manifest.operation_id}")


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


@ingest_app.command("resume")
def ingest_resume(
    vault: Path,
    operation_id: str,
    from_step: Optional[str] = typer.Option(
        None,
        "--from",
        help=f"Resume from this step, deleting this step and downstream outputs. Valid steps: {VALID_RESUME_STEPS_HELP}.",
    ),
    mode: Optional[RunMode] = None,
) -> None:
    """Resume using the current provider config for steps that will execute."""
    try:
        manifest = resume_ingest(
            vault=vault,
            operation_id=operation_id,
            from_step=from_step,
            run_mode=mode,
            console=console,
        )
    except (PipelineError, ProviderConfigError, VerifyError, WorkspaceError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"[green]Operation ready[/]: {manifest.operation_id}")


def _print_manifest_table(vault: Path, manifest: OperationManifest) -> None:
    table = Table(title=f"Ingest {manifest.operation_id}")
    table.add_column("Step", no_wrap=True, overflow="fold")
    table.add_column("Status", no_wrap=True)
    table.add_column("Review", overflow="fold")
    table.add_column("Attempts", justify="right")
    table.add_column("Last Duration", justify="right")
    table.add_column("Total Duration", justify="right")
    table.add_column("Provider")
    for step in manifest.steps:
        durations = [attempt.duration_ms for attempt in step.attempts if attempt.duration_ms is not None]
        last_duration = durations[-1] if durations else None
        total_duration = sum(durations) if durations else None
        table.add_row(
            step.name,
            step.status.value,
            _review_label(vault, manifest, step.name),
            str(len(step.attempts)),
            format_duration(last_duration),
            format_duration(total_duration),
            _provider_label(step.name, step.attempts[-1].provider_spec if step.attempts else None),
        )
    console.print(table)
    console.print("columns: Step | Status | Review | Attempts | Last Duration | Total Duration | Provider")
    reviews = [
        f"{step.name}={label}"
        for step in manifest.steps
        if (label := _review_label(vault, manifest, step.name))
    ]
    if reviews:
        console.print("reviews: " + "; ".join(reviews))
    console.print(f"mode: [bold]{manifest.run_mode.value}[/]")
    console.print(f"status: [bold]{manifest.status.value}[/]")
    latest_error = next((step.error for step in reversed(manifest.steps) if step.error), None)
    if latest_error:
        console.print(f"[red]latest error:[/] {latest_error}")
    _print_artifact_hints(vault, manifest)
    console.print(f"next: {_next_action(manifest)}")


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
        if manifest.run_mode == RunMode.standard:
            return "standard mode does not allow manual apply in this MVP"
        return f"run `llmwiki ingest apply <vault> {manifest.operation_id}`"
    return "inspect status"


def _print_artifact_hints(vault: Path, manifest: OperationManifest) -> None:
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    hints = [
        ("raw cleanup", run_dir / "raw_link_cleanup" / "raw_link_cleanup.md"),
        ("raw cleanup diff", run_dir / "raw_link_cleanup" / "cleanup.diff"),
        ("merge review", run_dir / "merge_plan_review" / "review_prompt.md"),
        ("draft review", run_dir / "draft_review" / "review_prompt.md"),
        ("draft root", run_dir / "draft_rendering" / "draft_pages"),
        ("diffs", run_dir / "draft_rendering" / "diffs"),
        ("apply preview", run_dir / "apply_preview" / "apply_preview.json"),
    ]
    for label, path in hints:
        if path.exists():
            console.print(f"{label}: `{path}`")


def _provider_label(step_name: str, provider_spec: str | None) -> str:
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
    commit: bool = typer.Option(False, "--commit", help="Disabled in this MVP."),
) -> None:
    """Apply draft pages into the vault wiki."""
    try:
        written = apply_operation(vault, operation_id, commit=commit)
    except (ApplyError, VerifyError, WorkspaceError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"[green]Applied[/] {len(written)} draft pages")


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
