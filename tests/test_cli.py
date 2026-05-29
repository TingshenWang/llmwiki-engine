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
    claims = RunStore(vault).run_dir(manifest.operation_id) / "claims.json"
    claims.write_text(claims.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    drift = runner.invoke(app, ["ingest", "status", str(vault), manifest.operation_id, "--verify"])
    assert drift.exit_code == 3

