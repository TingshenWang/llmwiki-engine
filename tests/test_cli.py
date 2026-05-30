from pathlib import Path

from typer.testing import CliRunner

from llmwiki_engine.cli import app
from llmwiki_engine.pipeline import copy_fixture_raw, init_vault, run_simplified_ingest
from llmwiki_engine.workspace import RunStore


ROOT = Path(__file__).parent
FIXTURE_ROOT = ROOT / "fixtures" / "simple_project"


def test_status_verify_exit_codes(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    raw = copy_fixture_raw(vault, FIXTURE_ROOT / "raw_project_note.md")
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="cli")
    runner = CliRunner()
    ok = runner.invoke(app, ["ingest", "status", str(vault), manifest.operation_id, "--verify"])
    assert ok.exit_code == 0
    claims = RunStore(vault).run_dir(manifest.operation_id) / "claim_extraction" / "claims.json"
    claims.write_text(claims.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    drift = runner.invoke(app, ["ingest", "status", str(vault), manifest.operation_id, "--verify"])
    assert drift.exit_code == 3


def test_resume_refresh_providers_requires_from(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    raw = copy_fixture_raw(vault, FIXTURE_ROOT / "raw_project_note.md")
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="cli-refresh")
    runner = CliRunner()
    result = runner.invoke(app, ["ingest", "resume", str(vault), manifest.operation_id, "--refresh-providers"])
    assert result.exit_code != 0
    assert "--refresh-providers requires --from" in result.output
    assert "Traceback" not in result.output


def test_resume_invalid_from_step_reports_single_line_error(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    raw = copy_fixture_raw(vault, FIXTURE_ROOT / "raw_project_note.md")
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="cli-from")
    runner = CliRunner()
    result = runner.invoke(app, ["ingest", "resume", str(vault), manifest.operation_id, "--from", "not_a_step"])
    assert result.exit_code != 0
    assert "Unknown step: not_a_step" in result.output
    assert "Traceback" not in result.output


def test_ingest_run_raw_outside_vault_reports_single_line_error(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    outside_raw = tmp_path / "outside.md"
    outside_raw.write_text("outside", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "ingest",
            "run",
            str(vault),
            str(outside_raw),
            "--fixture-dir",
            str(FIXTURE_ROOT / "mock"),
        ],
    )
    assert result.exit_code != 0
    assert "Raw input must be inside the vault raw/ directory." in result.output
    assert "Traceback" not in result.output
