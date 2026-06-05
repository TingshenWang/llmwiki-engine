import json
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

import llmwiki_engine.cli as cli_module
from llmwiki_engine.cli import app
from llmwiki_engine.io import read_json, read_jsonl, read_yaml, write_json, write_yaml
from llmwiki_engine.models import RunMode
from llmwiki_engine.pipeline import copy_fixture_raw, init_vault, latest_operation, run_simplified_ingest
from llmwiki_engine.provider_checks import check_providers as check_providers_impl
from llmwiki_engine.providers import OpenAICompatibleProvider
from llmwiki_engine.steps import STEP_NAMES
from llmwiki_engine.workspace import RunStore, WorkspaceError


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
    assert "Review" in ok.output
    assert "Attempts" in ok.output
    assert "Last Duration" in ok.output
    assert "Total Duration" in ok.output
    assert "Provider" in ok.output
    assert "prepared_raw_review" in ok.output
    assert "source_digest_review" in ok.output
    assert "raw cleanup" in ok.output
    assert ok.output.count("auto_stub/approved (auto-approved)") >= 2
    raw_json = runner.invoke(app, ["ingest", "status", str(vault), manifest.operation_id, "--json"])
    assert json.loads(raw_json.output)["operation_id"] == manifest.operation_id
    digest = RunStore(vault).run_dir(manifest.operation_id) / "source_digest" / "source_digest.json"
    digest.write_text(digest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
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


def test_resume_help_lists_step_names_from_metadata() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["ingest", "resume", "--help"])
    assert result.exit_code == 0
    for step_name in STEP_NAMES:
        assert step_name in result.output


@pytest.mark.parametrize("schema_version", ["operation_manifest.v4", "operation_manifest.v7", "operation_manifest.v9"])
def test_unsupported_manifest_schema_reports_single_line_error_for_user_commands(
    tmp_path: Path,
    schema_version: str,
) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    raw = copy_fixture_raw(vault, FIXTURE_ROOT / "raw_project_note.md")
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="old")
    manifest_path = RunStore(vault).manifest_path(manifest.operation_id)
    data = read_json(manifest_path)
    data["schema_version"] = schema_version
    write_json(manifest_path, data)

    runner = CliRunner()
    for command in ["status", "resume", "apply"]:
        result = runner.invoke(app, ["ingest", command, str(vault), manifest.operation_id])
        assert result.exit_code != 0
        assert "operation is incompatible with current MVP pipeline; rerun ingest" in _compact_output(result.output)
        assert "Traceback" not in result.output


@pytest.mark.parametrize(
    "mutator",
    [
        lambda names: names[:-1],
        lambda names: [*names, "extra_step"],
        lambda names: [*names, names[-1]],
        lambda names: [names[1], names[0], *names[2:]],
    ],
)
def test_manifest_step_topology_reports_single_line_error_for_user_commands(tmp_path: Path, mutator) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    raw = copy_fixture_raw(vault, FIXTURE_ROOT / "raw_project_note.md")
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="topology-cli")
    manifest_path = RunStore(vault).manifest_path(manifest.operation_id)
    data = read_json(manifest_path)
    steps_by_name = {step["name"]: step for step in data["steps"]}
    mutated_names = mutator([step["name"] for step in data["steps"]])
    data["steps"] = [dict(steps_by_name.get(name, data["steps"][0]), name=name) for name in mutated_names]
    write_json(manifest_path, data)

    runner = CliRunner()
    for command in ["status", "resume", "apply"]:
        result = runner.invoke(app, ["ingest", command, str(vault), manifest.operation_id])
        assert result.exit_code != 0
        assert "operation is incompatible with current MVP pipeline; rerun ingest" in _compact_output(result.output)
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


def test_apply_wiki_context_drift_reports_fixed_chinese_message(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    raw = copy_fixture_raw(vault, FIXTURE_ROOT / "raw_project_note.md")
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="wiki-drift-cli")
    (vault / "wiki" / "index.md").write_text("changed after planning\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, ["ingest", "apply", str(vault), manifest.operation_id])
    assert result.exit_code != 0
    assert "当前 operation 的 apply plan 已过期，因为 wiki 在 plan 生成后发生变化。请 resume 后再 apply。" in _compact_output(result.output)
    assert "wiki context changed after planning" not in result.output
    assert "Traceback" not in result.output


def test_blank_operation_id_is_rejected_before_building_run_path(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")

    with pytest.raises(WorkspaceError, match="Operation id is empty"):
        RunStore(vault).run_dir(" ")


@pytest.mark.parametrize("command", ["status", "resume", "apply"])
def test_missing_operation_manifest_reports_single_line_error(tmp_path: Path, command: str) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")

    runner = CliRunner()
    result = runner.invoke(app, ["ingest", command, str(vault), "ING-missing"])

    assert result.exit_code != 0
    assert "Operation manifest not found:" in result.output
    assert "Traceback" not in result.output


def test_standard_status_does_not_prompt_manual_apply(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    raw = copy_fixture_raw(vault, FIXTURE_ROOT / "raw_project_note.md")
    manifest = run_simplified_ingest(
        vault=vault,
        raw_file=raw,
        fixture_dir=FIXTURE_ROOT / "mock",
        slug="standard-next",
        run_mode=RunMode.standard,
    )

    runner = CliRunner()
    result = runner.invoke(app, ["ingest", "status", str(vault), manifest.operation_id])
    assert result.exit_code == 0
    assert "standard mode does not allow manual apply in this MVP" in result.output
    assert "llmwiki ingest apply" not in result.output


def _compact_output(output: str) -> str:
    for char in "│╭╮╰╯─":
        output = output.replace(char, " ")
    return " ".join(output.split())


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


def test_providers_check_bad_yaml_reports_single_line_error(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    config_path = vault / ".llmwiki" / "config.yaml"
    config_path.write_text("providers:\n  default: [\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, ["providers", "check", str(vault)])

    assert result.exit_code == 1
    assert "Config YAML parse failed" in result.output
    assert "vault config" in result.output
    assert str(config_path) in result.output.replace("\n", "")
    assert "Traceback" not in result.output
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
        if task == "source_digest":
            data["summary"] += f" {secret}"
            data["concepts"][0]["why_matters"] += f" {secret}"
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
    boundaries.extend(
        (f"provider attempt {path}", path.read_text(encoding="utf-8"))
        for path in sorted(run_dir.rglob("provider_results/attempt-*.json"))
    )
    boundaries.extend(
        (f"report {path}", path.read_text(encoding="utf-8"))
        for pattern in [
            "structured_repair_report.*",
            "draft_grounding_review.*",
            "update_merge_report.*",
            "related_merge_report.*",
        ]
        for path in sorted(run_dir.rglob(pattern))
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
