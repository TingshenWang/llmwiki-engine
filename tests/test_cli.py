import json
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

import llmwiki_engine.cli as cli_module
from llmwiki_engine.cli import app
from llmwiki_engine.hash_utils import sha256_file
from llmwiki_engine.io import read_json, read_jsonl, read_yaml, write_json, write_yaml
from llmwiki_engine.models import RawPreparePolicy
from llmwiki_engine.pipeline import copy_fixture_raw, init_vault, latest_operation, run_simplified_ingest
from llmwiki_engine.provider_checks import check_providers as check_providers_impl
from llmwiki_engine.providers import OpenAICompatibleProvider
from llmwiki_engine.raw_import import ArxivRawImportItem, ArxivRawImportReport, RawUrlImportResult
from llmwiki_engine.steps import STEP_NAMES
from llmwiki_engine.workspace import RunStore, WorkspaceError


ROOT = Path(__file__).parent
FIXTURE_ROOT = ROOT / "fixtures" / "simple_project"


def configure_openai_provider(vault: Path) -> None:
    config_path = vault / ".llmwiki" / "config.yaml"
    config = read_yaml(config_path)
    config["providers"] = {
        "default": {
            "spec": "openai_compatible:test-model",
            "endpoint": "https://example.test/v1/chat/completions",
            "api_key": "sk-test",
        }
    }
    write_yaml(config_path, config)


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
    assert "Attempt Total" in ok.output
    assert "duration note" in ok.output
    assert "Provider" in ok.output
    assert "prepared_raw_review" in ok.output
    assert "source_digest_review" in ok.output
    assert "current_model_calls" in ok.output
    assert "current_attempt_duration" in ok.output
    assert "current_model_duration" in ok.output
    assert "current_payload_chars" in ok.output
    assert "largest_payload_step" in ok.output
    assert "bottlenecks:" in ok.output
    assert "draft_rendering" in ok.output
    assert "candidates=" in ok.output
    assert "deduped=" in ok.output
    assert "deferred=" in ok.output
    assert "raw cleanup" in ok.output
    assert "run metrics" in ok.output
    assert "global applied receipt log path" in ok.output
    assert "current operation receipt: `not found yet`" in ok.output
    assert ".llmwiki/applied/operations.jsonl" in _compact_output(ok.output)
    assert ok.output.count("auto_stub/approved (auto-approved)") >= 2
    batch_report = RunStore(vault).run_dir(manifest.operation_id) / "draft_rendering" / "draft_rendering_batch_report.md"
    batch_report.write_text("# Draft Rendering Batches\n", encoding="utf-8")
    batch_hint = runner.invoke(app, ["ingest", "status", str(vault), manifest.operation_id])
    assert batch_hint.exit_code == 0
    assert "draft batches" in batch_hint.output
    raw_json = runner.invoke(app, ["ingest", "status", str(vault), manifest.operation_id, "--json"])
    assert json.loads(raw_json.output)["operation_id"] == manifest.operation_id
    inspect_json = runner.invoke(app, ["ingest", "inspect", str(vault), manifest.operation_id, "--json"])
    assert inspect_json.exit_code == 0
    inspect_payload = json.loads(inspect_json.output)
    assert inspect_payload["operation_id"] == manifest.operation_id
    assert inspect_payload["operation_status"] == manifest.status.value
    assert inspect_payload["metrics"]["internal_model_call_count"] >= 0
    assert inspect_payload["next_action"]
    inspect_hints = {item["label"]: item for item in inspect_payload["artifact_hints"]}
    assert inspect_hints["manifest"]["exists"] is True
    assert inspect_hints["run_metrics"]["exists"] is True
    assert inspect_hints["candidate_budget"]["exists"] is True
    assert inspect_hints["candidate_budget"]["path"].endswith("/source_digest/source_digest_budget_report.md")
    assert inspect_hints["draft_review"]["path"].endswith("/draft_review/review_prompt.md")
    assert inspect_payload["current_operation_receipt_exists"] is False
    assert inspect_payload["applied_receipt_log"].endswith("/.llmwiki/applied/operations.jsonl")
    inspect_table = runner.invoke(app, ["ingest", "inspect", str(vault), manifest.operation_id])
    assert inspect_table.exit_code == 0
    assert "Ingest inspect" in inspect_table.output
    assert "artifact_hints" in inspect_table.output
    grounding_path = RunStore(vault).run_dir(manifest.operation_id) / "draft_rendering" / "draft_grounding_review.json"
    grounding = read_json(grounding_path)
    warning_claim = {
        "page_plan_id": "PP-WARN",
        "target_path": "concepts/Concept_Warn.md",
        "section_key": "detail",
        "claim_type": "new_fact",
        "text": "好的产品判断往往来自长期实践中形成的经验直觉",
        "support": "unsupported",
        "action": "warn",
        "reason": "低风险未支撑引号内容仅记录为 warning，不阻塞自动 ingest；如需严谨可人工回看来源。",
    }
    grounding["warnings"] = [warning_claim]
    grounding["claims"] = [*grounding.get("claims", []), warning_claim]
    write_json(grounding_path, grounding)
    warning_status = runner.invoke(app, ["ingest", "status", str(vault), manifest.operation_id])
    assert warning_status.exit_code == 0
    assert "grounding review" in warning_status.output
    assert "blocking=0" in warning_status.output
    assert "非阻塞提醒=1" in warning_status.output
    warning_inspect = runner.invoke(app, ["ingest", "inspect", str(vault), manifest.operation_id, "--json"])
    assert warning_inspect.exit_code == 0
    warning_payload = json.loads(warning_inspect.output)
    assert warning_payload["grounding_review"]["warning_count"] == 1
    digest = RunStore(vault).run_dir(manifest.operation_id) / "source_digest" / "source_digest.json"
    digest.write_text(digest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    drift = runner.invoke(app, ["ingest", "status", str(vault), manifest.operation_id, "--verify"])
    assert drift.exit_code == 3


def test_ingest_run_json_with_mock_fixture_is_pure_json(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    raw = copy_fixture_raw(vault, FIXTURE_ROOT / "raw_project_note.md")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "ingest",
            "run",
            str(vault),
            str(raw),
            "--mock-fixture-dir",
            str(FIXTURE_ROOT / "mock"),
            "--slug",
            "run-json-real",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["operation_id"]
    assert payload["operation_status"] in {"awaiting_review", "drafted"}
    assert payload["raw_bindings"][0]["relative_path"] == "raw/raw_project_note.md"
    assert payload["metrics"]["internal_model_call_count"] >= 0
    hints = {item["label"]: item for item in payload["artifact_hints"]}
    assert hints["manifest"]["exists"] is True
    assert hints["run_metrics"]["exists"] is True
    assert hints["draft_review"]["path"].endswith("/draft_review/review_prompt.md")


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


def test_ingest_run_reports_awaiting_review_instead_of_ready(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    raw = copy_fixture_raw(vault, FIXTURE_ROOT / "raw_project_note.md")
    target = vault / "wiki" / "concepts" / "Concept_知识编译工程骨架.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "---\n"
        "llmwiki_type: concept\n"
        "title: 知识编译工程骨架\n"
        "aliases: []\n"
        "summary: 已有摘要。\n"
        "created: 2026-01-01\n"
        "updated: 2026-01-02\n"
        "---\n\n"
        "# 知识编译工程骨架\n\n"
        "## 详情\n\n"
        "Managed Agents / harness 视角强调安全边界、隔离容器、工具权限和会话对象。\n",
        encoding="utf-8",
    )
    fixture_dir = tmp_path / "grounding-fixture"
    fixture_dir.mkdir()
    for name in ["raw_prepare.json", "source_digest.json", "candidate_resolution.json", "wiki_merge_planning.json", "draft_rendering.json"]:
        data = read_json(FIXTURE_ROOT / "mock" / name)
        if name == "raw_prepare.json":
            data["prepared_markdown"] += "\n\nAnthropic 收购了 OpenAI。"
        if name == "draft_rendering.json":
            data["pages"][0]["body_markdown"] += "\n\nOpenAI 收购了 Anthropic。"
        write_json(fixture_dir / name, data)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "ingest",
            "run",
            str(vault),
            str(raw),
            "--fixture-dir",
            str(fixture_dir),
            "--slug",
            "awaiting-review",
        ],
    )

    assert result.exit_code == 0
    assert "Operation awaiting review" in result.output
    assert "draft_review" in result.output
    assert "Operation ready" not in result.output


def test_run_mock_fixture_dir_forces_mock_provider_over_live_config(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    raw = copy_fixture_raw(vault, FIXTURE_ROOT / "raw_project_note.md")
    config_path = vault / ".llmwiki" / "config.yaml"
    config = read_yaml(config_path)
    config["providers"] = {
        "default": {
            "spec": "openai_compatible:should-not-be-used",
            "endpoint": "https://example.invalid/v1/chat/completions",
            "api_key": "sk-should-not-be-used",
        }
    }
    write_yaml(config_path, config)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "ingest",
            "run",
            str(vault),
            str(raw),
            "--mock-fixture-dir",
            str(FIXTURE_ROOT / "mock"),
            "--slug",
            "forced-mock",
        ],
    )

    assert result.exit_code == 0
    operation_id = latest_operation(vault)
    assert operation_id is not None
    manifest = read_json(RunStore(vault).manifest_path(operation_id))
    providers = manifest["provider_contexts"][0]["providers"]
    assert providers
    assert {runtime["spec"] for runtime in providers.values()} == {"mock:fixture"}
    assert {runtime["fixture_dir"] for runtime in providers.values()} == {(FIXTURE_ROOT / "mock").resolve().as_posix()}


def test_run_rejects_fixture_dir_and_mock_fixture_dir_together(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    raw = copy_fixture_raw(vault, FIXTURE_ROOT / "raw_project_note.md")
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "ingest",
            "run",
            str(vault),
            str(raw),
            "--fixture-dir",
            str(FIXTURE_ROOT / "mock"),
            "--mock-fixture-dir",
            str(FIXTURE_ROOT / "mock"),
        ],
    )

    assert result.exit_code != 0
    assert "Use either --fixture-dir or --mock-fixture-dir" in result.output


@pytest.mark.parametrize(
    ("prepare", "expected"),
    [
        ("auto", RawPreparePolicy.auto),
        ("skip", RawPreparePolicy.skip),
        ("force", RawPreparePolicy.force),
    ],
)
def test_run_passes_prepare_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    prepare: str,
    expected: RawPreparePolicy,
) -> None:
    vault = tmp_path / "vault"
    raw = tmp_path / "raw.md"
    raw.write_text("# Raw\n", encoding="utf-8")
    seen: dict[str, object] = {}

    def fake_run_simplified_ingest(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(cli_module, "run_simplified_ingest", fake_run_simplified_ingest)
    monkeypatch.setattr(cli_module, "_print_operation_outcome", lambda manifest: None)
    runner = CliRunner()
    result = runner.invoke(app, ["ingest", "run", str(vault), str(raw), "--prepare", prepare])

    assert result.exit_code == 0
    assert seen["raw_prepare_policy"] == expected


def test_status_labels_skip_policy_as_local_provider(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    raw = copy_fixture_raw(vault, FIXTURE_ROOT / "raw_project_note.md")
    manifest = run_simplified_ingest(
        vault=vault,
        raw_file=raw,
        fixture_dir=FIXTURE_ROOT / "mock",
        slug="skip-prepare-status",
        raw_prepare_policy=RawPreparePolicy.skip,
    )
    runner = CliRunner()

    result = runner.invoke(app, ["ingest", "status", str(vault), manifest.operation_id])

    assert result.exit_code == 0
    assert "local:skip" in result.output
    assert "openai_compatible" not in next(line for line in result.output.splitlines() if "raw_prepare" in line)


def test_resume_passes_prepare_policy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    seen: dict[str, object] = {}

    def fake_resume_ingest(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(cli_module, "resume_ingest", fake_resume_ingest)
    monkeypatch.setattr(cli_module, "_print_operation_outcome", lambda manifest: None)
    runner = CliRunner()
    result = runner.invoke(app, ["ingest", "resume", str(vault), "ING-demo", "--prepare", "force"])

    assert result.exit_code == 0
    assert seen["raw_prepare_policy"] == RawPreparePolicy.force


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


@pytest.mark.parametrize(
    "command",
    [
        ["ingest", "run", "--help"],
        ["ingest", "run-next", "--help"],
        ["ingest", "resume", "--help"],
    ],
)
def test_prepare_help_uses_single_policy_values(command: list[str]) -> None:
    runner = CliRunner()
    result = runner.invoke(app, command)

    assert result.exit_code == 0
    assert "[auto|skip|force]" in result.output


def test_unsupported_manifest_schema_reports_single_line_error_for_user_commands(
    tmp_path: Path,
) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    raw = copy_fixture_raw(vault, FIXTURE_ROOT / "raw_project_note.md")
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="invalid-schema")
    manifest_path = RunStore(vault).manifest_path(manifest.operation_id)
    data = read_json(manifest_path)
    data["schema_version"] = "operation_manifest.invalid"
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


def test_drafted_status_prompts_manual_apply(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    raw = copy_fixture_raw(vault, FIXTURE_ROOT / "raw_project_note.md")
    manifest = run_simplified_ingest(
        vault=vault,
        raw_file=raw,
        fixture_dir=FIXTURE_ROOT / "mock",
        slug="single-mode-next",
    )

    runner = CliRunner()
    result = runner.invoke(app, ["ingest", "status", str(vault), manifest.operation_id])
    assert result.exit_code == 0
    assert f"llmwiki ingest apply <vault> {manifest.operation_id}" in result.output


def test_raw_candidates_reports_unprocessed_changed_and_duplicate_hash(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    processed = vault / "raw" / "processed.md"
    changed = vault / "raw" / "changed.md"
    duplicate = vault / "raw" / "duplicate.md"
    unprocessed = vault / "raw" / "unprocessed.md"
    hidden = vault / "raw" / ".hidden.md"
    raw_log = vault / "raw" / "log" / "日志_2026-06-06.md"
    processed.write_text("# Processed\n\nsame content\n", encoding="utf-8")
    changed.write_text("# Changed\n\nnew content\n", encoding="utf-8")
    duplicate.write_text(processed.read_text(encoding="utf-8"), encoding="utf-8")
    unprocessed.write_text("# New\n\nnew candidate\n", encoding="utf-8")
    hidden.write_text("# Hidden\n\nignored\n", encoding="utf-8")
    raw_log.parent.mkdir(parents=True, exist_ok=True)
    raw_log.write_text("# Raw log\n\nignored\n", encoding="utf-8")
    source = vault / "wiki" / "sources" / "来源_existing.md"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "---\n"
        "llmwiki_type: source\n"
        "title: Existing\n"
        "aliases: []\n"
        "summary: Existing source.\n"
        "created: 2026-01-01\n"
        "updated: 2026-01-01\n"
        "source_raw_paths:\n"
        "  - raw/processed.md\n"
        "  - raw/changed.md\n"
        "source_raw_hashes:\n"
        f"  - {sha256_file(processed)}\n"
        "  - old-changed-hash\n"
        "source_operation_ids:\n"
        "  - ING-old\n"
        "---\n\n"
        "# Existing\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(app, ["ingest", "raw-candidates", str(vault), "--json"])

    assert result.exit_code == 0
    report = json.loads(result.output)
    statuses = {Path(item["raw_path"]).name: item["status"] for item in report["items"]}
    assert statuses == {
        "unprocessed.md": "unprocessed",
        "changed.md": "changed",
        "duplicate.md": "duplicate_hash",
    }
    assert report["processed_count"] == 1
    assert report["changed_count"] == 1
    assert report["duplicate_hash_count"] == 1
    assert report["unprocessed_count"] == 1
    assert report["total_raw_files"] == 4
    changed_item = next(item for item in report["items"] if item["raw_path"] == "raw/changed.md")
    assert changed_item["matched_by"] == "path"
    assert changed_item["operation_ids"] == ["ING-old"]
    duplicate_item = next(item for item in report["items"] if item["raw_path"] == "raw/duplicate.md")
    assert duplicate_item["matched_by"] == "hash"
    assert duplicate_item["source_pages"] == ["wiki/sources/来源_existing.md"]


def test_raw_candidates_reports_duplicate_imported_url(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    first = vault / "raw" / "a-paper.md"
    second = vault / "raw" / "b-paper.md"
    first.write_text(
        "# Paper\n\n"
        "Imported from: https://arxiv.org/abs/2507.21504\n"
        "Fetched URL: https://arxiv.org/html/2507.21504\n"
        "Final URL: https://arxiv.org/html/2507.21504\n\n"
        "---\n\n"
        "first version\n",
        encoding="utf-8",
    )
    second.write_text(
        "# Paper\n\n"
        "Imported from: https://arxiv.org/pdf/2507.21504.pdf\n"
        "Fetched URL: https://arxiv.org/html/2507.21504\n"
        "Final URL: https://arxiv.org/html/2507.21504\n\n"
        "---\n\n"
        "second version with a different content hash\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(app, ["ingest", "raw-candidates", str(vault), "--json"])
    table_result = runner.invoke(app, ["ingest", "raw-candidates", str(vault)])

    assert result.exit_code == 0
    report = json.loads(result.output)
    statuses = {Path(item["raw_path"]).name: item["status"] for item in report["items"]}
    assert statuses == {"a-paper.md": "unprocessed", "b-paper.md": "duplicate_url"}
    assert report["unprocessed_count"] == 1
    assert report["duplicate_url_count"] == 1
    duplicate_item = next(item for item in report["items"] if item["raw_path"] == "raw/b-paper.md")
    assert duplicate_item["matched_by"] == "url"
    assert "raw/a-paper.md" in duplicate_item["reason"]
    assert table_result.exit_code == 0
    assert "duplicate_url=1" in _compact_output(table_result.output)


def test_cli_reference_raw_import_overview_matches_current_flags() -> None:
    for doc_path in [
        Path("docs/cli-reference.en.md"),
        Path("docs/cli-reference.zh-CN.md"),
    ]:
        lines = doc_path.read_text(encoding="utf-8").splitlines()
        raw_url_line = next(line for line in lines if line.startswith("llmwiki ingest raw-import-url "))
        arxiv_line = next(line for line in lines if line.startswith("llmwiki ingest raw-import-arxiv "))

        assert "--output PATH" in raw_url_line
        assert "--dedupe-url|--no-dedupe-url" in raw_url_line
        assert "--arxiv-html|--no-arxiv-html" in raw_url_line
        assert "--slug" not in raw_url_line
        assert "--format" not in raw_url_line

        assert "--limit N" in arxiv_line
        assert "--sort-by VALUE" in arxiv_line
        assert "--min-relevance-score N" in arxiv_line
        assert "--max-results" not in arxiv_line


def test_raw_candidates_all_includes_processed_and_table_gives_next_command(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    processed = vault / "raw" / "processed.md"
    unprocessed = vault / "raw" / "unprocessed.md"
    processed.write_text("# Processed\n", encoding="utf-8")
    unprocessed.write_text("# Unprocessed\n", encoding="utf-8")
    source = vault / "wiki" / "sources" / "来源_existing.md"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "---\n"
        "llmwiki_type: source\n"
        "title: Existing\n"
        "aliases: []\n"
        "summary: Existing source.\n"
        "created: 2026-01-01\n"
        "updated: 2026-01-01\n"
        "source_raw_paths:\n"
        "  - raw/processed.md\n"
        "source_raw_hashes:\n"
        f"  - {sha256_file(processed)}\n"
        "source_operation_ids:\n"
        "  - ING-old\n"
        "---\n\n"
        "# Existing\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    all_result = runner.invoke(app, ["ingest", "raw-candidates", str(vault), "--all", "--json"])
    table_result = runner.invoke(app, ["ingest", "raw-candidates", str(vault)])

    assert all_result.exit_code == 0
    all_report = json.loads(all_result.output)
    assert {Path(item["raw_path"]).name: item["status"] for item in all_report["items"]} == {
        "unprocessed.md": "unprocessed",
        "processed.md": "processed",
    }
    assert table_result.exit_code == 0
    assert "Raw ingest candidates" in table_result.output
    assert "processed raw files are hidden" in table_result.output
    assert "llmwiki ingest run" in table_result.output
    assert "raw/unprocessed.md" in table_result.output


def test_run_next_dry_run_selects_unprocessed_raw(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    raw = vault / "raw" / "next.md"
    raw.write_text("# Next\n\ncandidate\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, ["ingest", "run-next", str(vault), "--dry-run"])

    assert result.exit_code == 0
    assert "selected raw" in result.output
    assert "raw/next.md" in result.output
    assert "llmwiki ingest run" in result.output


def test_run_next_dry_run_preserves_prepare_choice_in_next_command(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    raw = vault / "raw" / "next.md"
    raw.write_text("# Next\n\ncandidate\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, ["ingest", "run-next", str(vault), "--dry-run", "--prepare", "skip"])
    json_result = runner.invoke(app, ["ingest", "run-next", str(vault), "--dry-run", "--prepare", "auto", "--json"])

    assert result.exit_code == 0
    assert "llmwiki ingest run" in result.output
    assert "--prepare skip" in result.output
    assert json_result.exit_code == 0
    assert json.loads(json_result.output)["next_command"].endswith("--prepare auto")


def test_run_next_dry_run_outputs_json(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    raw = vault / "raw" / "next.md"
    raw.write_text("# Next\n\ncandidate\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, ["ingest", "run-next", str(vault), "--dry-run", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["dry_run"] is True
    assert payload["selected_raw_path"] == "raw/next.md"
    assert payload["selected_raw_absolute_path"] == raw.resolve().as_posix()
    assert payload["candidate_status"] == "unprocessed"
    assert payload["operation_id"] is None
    assert payload["artifact_hints"] == []
    assert "llmwiki ingest run" in payload["next_command"]


def test_run_next_requires_include_changed_for_changed_only_candidate(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    raw = vault / "raw" / "changed.md"
    raw.write_text("# Changed\n\nnew content\n", encoding="utf-8")
    source = vault / "wiki" / "sources" / "来源_existing.md"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "---\n"
        "llmwiki_type: source\n"
        "title: Existing\n"
        "aliases: []\n"
        "summary: Existing source.\n"
        "created: 2026-01-01\n"
        "updated: 2026-01-01\n"
        "source_raw_paths:\n"
        "  - raw/changed.md\n"
        "source_raw_hashes:\n"
        "  - old-hash\n"
        "---\n\n"
        "# Existing\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    blocked = runner.invoke(app, ["ingest", "run-next", str(vault), "--dry-run"])
    allowed = runner.invoke(app, ["ingest", "run-next", str(vault), "--dry-run", "--include-changed"])

    assert blocked.exit_code != 0
    assert "No unprocessed raw ingest candidate found" in blocked.output
    assert allowed.exit_code == 0
    assert "status: `changed`" in allowed.output
    assert "raw/changed.md" in allowed.output


def test_run_next_invokes_ingest_with_selected_raw(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    raw = vault / "raw" / "next.md"
    raw.write_text("# Next\n\ncandidate\n", encoding="utf-8")
    mock_fixture_dir = tmp_path / "mock"
    seen: dict[str, object] = {}

    def fake_run_simplified_ingest(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(cli_module, "run_simplified_ingest", fake_run_simplified_ingest)
    monkeypatch.setattr(cli_module, "_print_operation_outcome", lambda manifest: None)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "ingest",
            "run-next",
            str(vault),
            "--mock-fixture-dir",
            str(mock_fixture_dir),
            "--slug",
            "next-run",
            "--prepare",
            "force",
        ],
    )

    assert result.exit_code == 0
    assert seen["vault"] == vault.resolve()
    assert seen["raw_file"] == raw.resolve()
    assert seen["mock_fixture_dir"] == mock_fixture_dir
    assert seen["slug"] == "next-run"
    assert seen["raw_prepare_policy"] == RawPreparePolicy.force


def test_run_next_outputs_json_after_ingest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    raw = vault / "raw" / "next.md"
    raw.write_text("# Next\n\ncandidate\n", encoding="utf-8")

    class Status:
        value = "awaiting_review"

    class StepStatusValue:
        value = "awaiting_review"

    class Step:
        name = "draft_review"
        status = StepStatusValue()

    class Manifest:
        operation_id = "ING-next"
        status = Status()
        steps = [Step()]

    def fake_run_simplified_ingest(**kwargs):
        return Manifest()

    monkeypatch.setattr(cli_module, "run_simplified_ingest", fake_run_simplified_ingest)
    runner = CliRunner()
    result = runner.invoke(app, ["ingest", "run-next", str(vault), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["dry_run"] is False
    assert payload["selected_raw_path"] == "raw/next.md"
    assert payload["operation_id"] == "ING-next"
    assert payload["operation_status"] == "awaiting_review"
    assert payload["awaiting_review_step"] == "draft_review"
    assert payload["next_command"] == f"llmwiki ingest status {vault.resolve()} ING-next"
    hints = {item["label"]: item for item in payload["artifact_hints"]}
    assert hints["run_dir"]["path"] == RunStore(vault).run_dir("ING-next").as_posix()
    assert hints["manifest"]["path"] == RunStore(vault).manifest_path("ING-next").as_posix()
    assert hints["run_metrics"]["path"].endswith("/ING-next/run_metrics.json")
    assert hints["draft_review"]["path"].endswith("/ING-next/draft_review/review_prompt.md")


def test_run_next_json_with_mock_fixture_is_pure_json(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    raw = copy_fixture_raw(vault, FIXTURE_ROOT / "raw_project_note.md")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "ingest",
            "run-next",
            str(vault),
            "--mock-fixture-dir",
            str(FIXTURE_ROOT / "mock"),
            "--slug",
            "run-next-json-real",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["dry_run"] is False
    assert payload["selected_raw_path"] == "raw/raw_project_note.md"
    assert payload["selected_raw_absolute_path"] == raw.resolve().as_posix()
    assert payload["operation_id"]
    assert payload["operation_status"] in {"awaiting_review", "drafted"}
    assert payload["next_command"] == f"llmwiki ingest status {vault.resolve()} {payload['operation_id']}"
    hints = {item["label"]: item for item in payload["artifact_hints"]}
    assert hints["manifest"]["exists"] is True
    assert hints["run_metrics"]["exists"] is True
    assert hints["draft_review"]["path"].endswith("/draft_review/review_prompt.md")


def test_raw_import_url_cli_outputs_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    fake_result = RawUrlImportResult(
        vault=vault.as_posix(),
        url="https://example.com/source",
        fetch_url="https://example.com/source",
        final_url="https://example.com/source",
        title="Research Source",
        raw_path="raw/Research Source.md",
        absolute_path=(vault / "raw" / "Research Source.md").as_posix(),
        content_type="text/html",
        format="html",
        imported_at="2026-06-06T00:00:00+00:00",
        sha256="abc123",
        size_bytes=123,
        overwritten=False,
    )

    def fake_import_raw_url(
        received_vault: Path,
        received_url: str,
        *,
        title: str | None,
        output_name: str | None,
        overwrite: bool,
        dedupe_url: bool,
        prefer_arxiv_html: bool,
        timeout: float,
        max_bytes: int,
    ) -> RawUrlImportResult:
        assert received_vault == vault
        assert received_url == "https://example.com/source"
        assert title == "Research Source"
        assert output_name == "sources/source.md"
        assert not overwrite
        assert dedupe_url
        assert prefer_arxiv_html
        assert timeout == 9.0
        assert max_bytes == 2048
        return fake_result

    monkeypatch.setattr(cli_module, "import_raw_url", fake_import_raw_url)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "ingest",
            "raw-import-url",
            str(vault),
            "https://example.com/source",
            "--title",
            "Research Source",
            "--output",
            "sources/source.md",
            "--timeout",
            "9",
            "--max-bytes",
            "2048",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["raw_path"] == "raw/Research Source.md"
    assert payload["title"] == "Research Source"
    assert payload["sha256"] == "abc123"


def test_raw_import_arxiv_cli_outputs_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    fake_report = ArxivRawImportReport(
        vault=vault.as_posix(),
        query="LLM agents",
        search_query="all:LLM AND all:agents",
        sort_by="lastUpdatedDate",
        sort_order="ascending",
        candidate_window=2,
        min_relevance_score=3,
        skipped_count=0,
        limit=2,
        dry_run=True,
        fetched_count=1,
        imported_count=0,
        existing_count=0,
        failed_count=0,
        items=(
            ArxivRawImportItem(
                arxiv_id="2507.21504",
                title="Evaluation Survey",
                abs_url="https://arxiv.org/abs/2507.21504",
                html_url="https://arxiv.org/html/2507.21504",
                status="found",
                relevance_score=8,
            ),
        ),
    )

    def fake_import_arxiv_search(
        received_vault: Path,
        received_query: str,
        *,
        limit: int,
        dry_run: bool,
        overwrite: bool,
        dedupe_url: bool,
        sort_by: str,
        sort_order: str,
        min_relevance_score: int,
        timeout: float,
        max_bytes: int,
    ) -> ArxivRawImportReport:
        assert received_vault == vault
        assert received_query == "LLM agents"
        assert limit == 2
        assert dry_run is True
        assert not overwrite
        assert dedupe_url is False
        assert sort_by == "lastUpdatedDate"
        assert sort_order == "ascending"
        assert min_relevance_score == 3
        assert timeout == 8.0
        assert max_bytes == 4096
        return fake_report

    monkeypatch.setattr(cli_module, "import_arxiv_search", fake_import_arxiv_search)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "ingest",
            "raw-import-arxiv",
            str(vault),
            "LLM agents",
            "--limit",
            "2",
            "--dry-run",
            "--no-dedupe-url",
            "--sort-by",
            "lastUpdatedDate",
            "--sort-order",
            "ascending",
            "--min-relevance-score",
            "3",
            "--timeout",
            "8",
            "--max-bytes",
            "4096",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["dry_run"] is True
    assert payload["candidate_window"] == 2
    assert payload["min_relevance_score"] == 3
    assert payload["items"][0]["status"] == "found"
    assert payload["items"][0]["relevance_score"] == 8
    assert payload["items"][0]["html_url"] == "https://arxiv.org/html/2507.21504"


def test_raw_import_arxiv_cli_prints_direct_next_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    raw_path = "raw/Evaluation Survey.md"
    fake_report = ArxivRawImportReport(
        vault=vault.as_posix(),
        query="LLM agents",
        search_query="all:LLM AND all:agents",
        sort_by="relevance",
        sort_order="descending",
        candidate_window=20,
        min_relevance_score=1,
        skipped_count=0,
        limit=1,
        dry_run=False,
        fetched_count=1,
        imported_count=1,
        existing_count=0,
        failed_count=0,
        items=(
            ArxivRawImportItem(
                arxiv_id="2507.21504",
                title="Evaluation Survey",
                abs_url="https://arxiv.org/abs/2507.21504",
                html_url="https://arxiv.org/html/2507.21504",
                status="imported",
                raw_path=raw_path,
                relevance_score=8,
            ),
        ),
    )

    def fake_import_arxiv_search(
        received_vault: Path,
        received_query: str,
        *,
        limit: int,
        dry_run: bool,
        overwrite: bool,
        dedupe_url: bool,
        sort_by: str,
        sort_order: str,
        min_relevance_score: int,
        timeout: float,
        max_bytes: int,
    ) -> ArxivRawImportReport:
        assert received_vault == vault
        assert received_query == "LLM agents"
        assert limit == 1
        assert dry_run is False
        return fake_report

    monkeypatch.setattr(cli_module, "import_arxiv_search", fake_import_arxiv_search)
    runner = CliRunner()
    result = runner.invoke(app, ["ingest", "raw-import-arxiv", str(vault), "LLM agents"])

    assert result.exit_code == 0
    compact = _compact_output(result.output)
    assert "next: `llmwiki ingest run" in compact
    assert "Evaluation Survey.md`" in compact
    assert "inspect: `llmwiki ingest raw-candidates" in compact


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
    assert "applied receipt log" in apply_result.output
    assert ".llmwiki/applied/operations.jsonl" in _compact_output(apply_result.output)

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
