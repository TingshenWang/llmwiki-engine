from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .apply import ApplyError, apply_operation
from .eval import load_eval_report, run_eval
from .models import OperationManifest, RunMode, VerificationStatus
from .pipeline import PipelineError, init_vault, latest_operation, resume_ingest, run_simplified_ingest, status as ingest_status
from .provider_checks import check_providers
from .provider_config import ProviderConfigError
from .profiles import builtin_profile_names, load_profile
from .providers import ProviderRegistry
from .verify import VerifyError, verify_run
from .workspace import WorkspaceError

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
    _print_manifest_table(manifest)
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
        help="Resume from this step, deleting this step and downstream outputs. Valid steps: "
        "raw_prepare, raw_index, extraction_windows, claim_extraction, page_planning, draft_rendering, validation, apply_preview.",
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


def _print_manifest_table(manifest: OperationManifest) -> None:
    table = Table(title=f"Ingest {manifest.operation_id}")
    table.add_column("Step")
    table.add_column("Status")
    for step in manifest.steps:
        table.add_row(step.name, step.status.value)
    console.print(table)
    console.print(f"mode: [bold]{manifest.run_mode.value}[/]")
    console.print(f"status: [bold]{manifest.status.value}[/]")
    latest_error = next((step.error for step in reversed(manifest.steps) if step.error), None)
    if latest_error:
        console.print(f"[red]latest error:[/] {latest_error}")
    console.print(f"next: {_next_action(manifest)}")


def _next_action(manifest: OperationManifest) -> str:
    if manifest.status.value == "applied":
        return "operation already applied"
    for step in manifest.steps:
        if step.status.value in {"failed", "pending"}:
            return f"run `llmwiki ingest resume <vault> {manifest.operation_id}`"
    if manifest.status.value == "drafted":
        return f"run `llmwiki ingest apply <vault> {manifest.operation_id}`"
    return "inspect status"


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
def ingest_apply(vault: Path, operation_id: str, commit: bool = typer.Option(False, "--commit")) -> None:
    """Apply draft pages into the vault wiki."""
    try:
        written = apply_operation(vault, operation_id, commit=commit)
    except (ApplyError, VerifyError, WorkspaceError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"[green]Applied[/] {len(written)} draft pages")


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
