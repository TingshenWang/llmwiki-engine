from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .apply import apply_operation
from .eval import load_eval_report, run_eval
from .pipeline import init_vault, latest_operation, run_simplified_ingest, status as ingest_status
from .profiles import builtin_profile_names, load_profile
from .providers import ProviderRegistry

app = typer.Typer(help="LLM-Wiki knowledge compilation engine.")
ingest_app = typer.Typer(help="Run and manage simplified ingest operations.")
providers_app = typer.Typer(help="Inspect and test providers.")
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
    fixture_dir: Path = typer.Option(..., "--fixture-dir", help="MockProvider fixture directory."),
    profile: str = "project_basic",
    slug: Optional[str] = None,
) -> None:
    """Run simplified Ingest through draft generation."""
    manifest = run_simplified_ingest(
        vault=vault,
        raw_file=raw,
        fixture_dir=fixture_dir,
        profile_name=profile,
        slug=slug,
        console=console,
    )
    console.print(f"[green]Operation ready[/]: {manifest.operation_id}")


@ingest_app.command("status")
def ingest_status_cmd(
    vault: Path,
    operation_id: Optional[str] = typer.Argument(None, help="Operation id. Defaults to the latest ingest operation."),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Show operation status."""
    operation_id = operation_id or latest_operation(vault)
    if operation_id is None:
        raise typer.BadParameter("No ingest operation found.")
    manifest = ingest_status(vault, operation_id)
    if json_output:
        console.print(manifest.model_dump_json(indent=2))
        return
    table = Table(title=f"Ingest {manifest.operation_id}")
    table.add_column("Step")
    table.add_column("Status")
    for step in manifest.steps:
        table.add_row(step.name, step.status)
    console.print(table)
    console.print(f"status: [bold]{manifest.status}[/]")


@ingest_app.command("apply")
def ingest_apply(vault: Path, operation_id: str, commit: bool = typer.Option(False, "--commit")) -> None:
    """Apply draft pages into the vault wiki."""
    written = apply_operation(vault, operation_id, commit=commit)
    console.print(f"[green]Applied[/] {len(written)} draft pages")


@providers_app.command("list")
def providers_list() -> None:
    """List available provider types."""
    table = Table(title="Providers")
    table.add_column("Name")
    for name in ProviderRegistry().names():
        table.add_row(name)
    console.print(table)


@providers_app.command("test")
def providers_test(provider: str, fixture_dir: Optional[Path] = typer.Option(None, "--fixture-dir")) -> None:
    """Create a provider and report whether it is locally constructible."""
    created = ProviderRegistry().create(provider, fixture_dir=fixture_dir)
    console.print(f"[green]Provider OK[/]: {created.name}")


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
