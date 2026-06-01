import json
from pathlib import Path

import httpx
from typer.testing import CliRunner

import llmwiki_engine.cli as cli_module
from llmwiki_engine.cli import app
from llmwiki_engine.io import read_json, read_jsonl, read_yaml, write_json, write_yaml
from llmwiki_engine.pipeline import copy_fixture_raw, init_vault, latest_operation, run_simplified_ingest
from llmwiki_engine.provider_checks import check_providers as check_providers_impl
from llmwiki_engine.providers import OpenAICompatibleProvider
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
    raw_json = runner.invoke(app, ["ingest", "status", str(vault), manifest.operation_id, "--json"])
    assert json.loads(raw_json.output)["operation_id"] == manifest.operation_id
    claims = RunStore(vault).run_dir(manifest.operation_id) / "claim_extraction" / "claims.json"
    claims.write_text(claims.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    drift = runner.invoke(app, ["ingest", "status", str(vault), manifest.operation_id, "--verify"])
    assert drift.exit_code == 3


def test_resume_refresh_providers_option_is_removed(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    raw = copy_fixture_raw(vault, FIXTURE_ROOT / "raw_project_note.md")
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="cli-refresh")
    runner = CliRunner()
    result = runner.invoke(app, ["ingest", "resume", str(vault), manifest.operation_id, "--refresh-providers"])
    assert result.exit_code != 0
    assert "No such option" in result.output
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


def test_old_manifest_reports_single_line_error_for_user_commands(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    raw = copy_fixture_raw(vault, FIXTURE_ROOT / "raw_project_note.md")
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="old")
    manifest_path = RunStore(vault).manifest_path(manifest.operation_id)
    data = read_json(manifest_path)
    data["schema_version"] = "operation_manifest.v4"
    write_json(manifest_path, data)

    runner = CliRunner()
    for command in ["status", "resume", "apply"]:
        result = runner.invoke(app, ["ingest", command, str(vault), manifest.operation_id])
        assert result.exit_code != 0
        assert "Unsupported manifest schema_version: operation_manifest.v4" in result.output
        assert "Traceback" not in result.output


def test_verify_drift_reports_single_line_error_for_resume_and_apply(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    raw = copy_fixture_raw(vault, FIXTURE_ROOT / "raw_project_note.md")
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="drift-cli")
    raw.write_text(raw.read_text(encoding="utf-8") + "\nchanged", encoding="utf-8")

    runner = CliRunner()
    for command in ["resume", "apply"]:
        result = runner.invoke(app, ["ingest", command, str(vault), manifest.operation_id])
        assert result.exit_code != 0
        assert "raw file hash changed" in result.output
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


def test_ingest_run_missing_raw_reports_single_line_error(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    missing_raw = vault / "raw" / "missing.md"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "ingest",
            "run",
            str(vault),
            str(missing_raw),
            "--fixture-dir",
            str(FIXTURE_ROOT / "mock"),
        ],
    )
    assert result.exit_code != 0
    assert "Raw input file does not exist: raw/missing.md" in result.output
    assert "Traceback" not in result.output


def test_providers_check_warns_for_non_git_vault(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    config_path = vault / ".llmwiki" / "config.yaml"
    config = read_yaml(config_path)
    config["providers"] = {"default": {"spec": "mock:fixture", "fixture_dir": str(FIXTURE_ROOT / "mock")}}
    write_yaml(config_path, config)
    runner = CliRunner()
    result = runner.invoke(app, ["providers", "check", str(vault)])
    assert result.exit_code == 0
    assert "providers check: ok" in result.output
    assert "not a Git repository" in result.output
    assert "live check: ok" not in result.output


def test_providers_check_live_prints_success_marker(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    config_path = vault / ".llmwiki" / "config.yaml"
    config = read_yaml(config_path)
    config["providers"] = {"default": {"spec": "mock:fixture", "fixture_dir": str(FIXTURE_ROOT / "mock")}}
    write_yaml(config_path, config)

    runner = CliRunner()
    result = runner.invoke(app, ["providers", "check", str(vault), "--live"])

    assert result.exit_code == 0
    assert "live check: ok" in result.output
    assert "providers check: ok" in result.output


def test_providers_list_works() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["providers", "list"])
    assert result.exit_code == 0
    assert "openai_compatible" in result.output


def test_providers_check_config_error_does_not_print_empty_table(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    config_path = vault / ".llmwiki" / "config.yaml"
    config = read_yaml(config_path)
    config["providers"] = {"unknown": "human"}
    write_yaml(config_path, config)

    runner = CliRunner()
    result = runner.invoke(app, ["providers", "check", str(vault)])

    assert result.exit_code == 1
    assert "Unknown provider key" in result.output
    assert "Provider Check" not in result.output


def test_providers_check_cli_redacts_fallback_failure(monkeypatch, tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    config_path = vault / ".llmwiki" / "config.yaml"
    config = read_yaml(config_path)
    config["providers"] = {
        "default": {
            "spec": "openai_compatible:test-model",
            "endpoint": "https://example.test/v1/chat/completions",
            "api_key": "sk-cli-secret",
        }
    }
    write_yaml(config_path, config)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(422, json={"error": {"message": "json_object response_format unsupported"}})
        raise httpx.ConnectError("fallback boom sk-cli-secret", request=request)

    def fake_check_providers(vault_path: Path, *, live: bool = False):
        return check_providers_impl(
            vault_path,
            live=live,
            http_client_factory=lambda: httpx.Client(transport=httpx.MockTransport(handler)),
        )

    monkeypatch.setattr(cli_module, "check_providers", fake_check_providers)

    runner = CliRunner()
    result = runner.invoke(app, ["providers", "check", str(vault), "--live"])

    assert result.exit_code == 1
    assert len(seen) == 2
    assert "sk-cli-secret" not in result.output
    assert "[REDACTED]" in result.output


def test_api_key_does_not_spread_across_e2e_cli_boundaries(monkeypatch, tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    raw = copy_fixture_raw(vault, FIXTURE_ROOT / "raw_project_note.md")
    secret = "sk-e2e-never-leak"
    config_path = vault / ".llmwiki" / "config.yaml"
    config = read_yaml(config_path)
    config["providers"] = {
        "default": {
            "spec": "openai_compatible:test-model",
            "endpoint": "https://example.test/v1/chat/completions",
            "api_key": secret,
        }
    }
    write_yaml(config_path, config)

    def fake_generate_raw(self, task, payload, output_model):
        data = json.loads((FIXTURE_ROOT / "mock" / f"{task}.json").read_text(encoding="utf-8"))
        if task == "raw_prepare":
            data["prepared_markdown"] += f"\n{secret}\n"
            data["review_notes"] = f"review note {secret}"
        if task == "claim_extraction":
            data["claims"][0]["text"] += f" {secret}"
        if task == "page_planning":
            data["pages"][0]["summary"] += f" {secret}"
        return json.dumps(data, ensure_ascii=False)

    monkeypatch.setattr(OpenAICompatibleProvider, "generate_raw", fake_generate_raw)

    runner = CliRunner()
    run_result = runner.invoke(app, ["ingest", "run", str(vault), str(raw), "--slug", "secret-e2e"])
    assert run_result.exit_code == 0
    operation_id = latest_operation(vault)
    assert operation_id is not None
    status_result = runner.invoke(app, ["ingest", "status", str(vault), operation_id, "--json"])
    assert status_result.exit_code == 0
    apply_result = runner.invoke(app, ["ingest", "apply", str(vault), operation_id])
    assert apply_result.exit_code == 0

    run_dir = RunStore(vault).run_dir(operation_id)
    boundaries = [
        ("run CLI output", run_result.output),
        ("status CLI JSON", status_result.output),
        ("apply CLI output", apply_result.output),
        ("manifest", RunStore(vault).manifest_path(operation_id).read_text(encoding="utf-8")),
        ("events", (run_dir / "events.jsonl").read_text(encoding="utf-8")),
        ("applied receipt", json.dumps(read_jsonl(vault / ".llmwiki" / "applied" / "operations.jsonl"), ensure_ascii=False)),
    ]
    boundaries.extend(
        (f"provider result {path}", path.read_text(encoding="utf-8"))
        for path in sorted(run_dir.rglob("provider_result.json"))
    )
    boundaries.extend((f"wiki output {path}", path.read_text(encoding="utf-8")) for path in sorted((vault / "wiki").rglob("*.md")))

    assert boundaries
    for label, content in boundaries:
        assert secret not in content, label

    assert secret in config_path.read_text(encoding="utf-8")
    assert any(
        "[REDACTED]" in content
        for label, content in boundaries
        if label.startswith(("provider result", "wiki output"))
    )
