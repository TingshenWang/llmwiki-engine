import json
import subprocess
from pathlib import Path

import pytest

from llmwiki_engine.apply import ApplyError, apply_operation
from llmwiki_engine.io import read_json, read_jsonl, read_yaml, write_json, write_yaml
from llmwiki_engine.manifest import read_manifest
from llmwiki_engine.models import OperationStatus, StepStatus, VerificationStatus
from llmwiki_engine.pipeline import (
    STEP_RUNNERS,
    copy_fixture_raw,
    init_vault,
    latest_operation,
    resume_ingest,
    run_simplified_ingest,
    status,
)
from llmwiki_engine.providers import OpenAICompatibleProvider
from llmwiki_engine.steps import MODEL_BACKED_STEPS, STEP_NAMES, STEP_SPECS
from llmwiki_engine.verify import VerifyError, verify_run
from llmwiki_engine.workspace import RunStore, WorkspaceError, ensure_workspace_layout


ROOT = Path(__file__).parent
FIXTURE_ROOT = ROOT / "fixtures" / "simple_project"


def make_vault(tmp_path: Path) -> tuple[Path, Path]:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    raw = copy_fixture_raw(vault, FIXTURE_ROOT / "raw_project_note.md")
    return vault, raw


def test_init_creates_workspace_layout_and_gitignore(tmp_path: Path) -> None:
    vault, _ = make_vault(tmp_path)
    assert (vault / ".llmwiki" / "config.yaml").exists()
    assert (vault / ".llmwiki" / "profiles" / "project_basic" / "profile.yaml").exists()
    assert (vault / ".llmwiki" / "applied" / "operations.jsonl").exists()
    assert (vault / ".llmwiki" / "runs").exists()
    gitignore_lines = (vault / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".llmwiki/" in gitignore_lines
    assert ".llmwiki/runs/" not in gitignore_lines
    assert not (vault / ".git").exists()


def test_legacy_stage_layout_blocks_without_workspace_layout(tmp_path: Path) -> None:
    vault = tmp_path / "legacy"
    (vault / "stage" / "ingest").mkdir(parents=True)
    with pytest.raises(WorkspaceError):
        ensure_workspace_layout(vault)


def test_init_ingest_status_apply_closes_loop(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(
        vault=vault,
        raw_file=raw,
        fixture_dir=FIXTURE_ROOT / "mock",
        profile_name="project_basic",
        slug="test",
    )
    assert manifest.status == OperationStatus.drafted
    loaded = status(vault, manifest.operation_id)
    assert loaded.operation_id == manifest.operation_id
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    assert (run_dir / "raw_prepare" / "raw_preparation.json").exists()
    assert (run_dir / "raw_prepare" / "prepared.md").exists()
    assert (run_dir / "raw_prepare" / "preparation_review.md").exists()
    assert (run_dir / "raw_index" / "raw_index.json").exists()
    assert (run_dir / "extraction_windows" / "extraction_windows.json").exists()
    assert "input_kind" in (run_dir / "raw_index" / "raw_index.json").read_text(encoding="utf-8")
    assert (run_dir / "apply_preview" / "apply_preview.json").exists()
    assert (run_dir / "draft_rendering" / "draft_pages" / "sources" / "Source_raw_project_note.md").exists()
    assert (run_dir / "draft_rendering" / "draft_pages" / "concepts" / "Concept_知识编译工程骨架.md").exists()
    assert not (run_dir / "snapshots").exists()
    assert not (run_dir / "validation").exists()
    assert loaded.schema_version == "operation_manifest.v1"
    assert loaded.provider_contexts[0].providers["raw_prepare"].spec == "mock:fixture"
    assert loaded.provider_contexts[0].providers["raw_prepare"].fixture_dir == (FIXTURE_ROOT / "mock").resolve().as_posix()
    assert "affected_steps" not in loaded.provider_contexts[0].model_dump()
    assert verify_run(vault, loaded).ok
    written = apply_operation(vault, manifest.operation_id)
    assert (vault / "wiki" / "sources" / "Source_raw_project_note.md") in written
    receipts = read_jsonl(vault / ".llmwiki" / "applied" / "operations.jsonl")
    assert receipts
    assert receipts[-1]["profile"] == "project_basic"
    assert receipts[-1]["profile_version"] == "1"
    assert "profile_snapshot_hash" not in receipts[-1]
    assert "prepared_raw" in receipts[-1]
    assert status(vault, manifest.operation_id).status == OperationStatus.applied


def test_ingest_run_uses_vault_config_profile_by_default(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="research_basic")
    raw = copy_fixture_raw(vault, FIXTURE_ROOT / "raw_project_note.md")
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="profile")
    assert manifest.profile == "research_basic"


def test_resume_after_failed_step(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    broken_fixture = tmp_path / "broken"
    broken_fixture.mkdir()
    for name in ["raw_prepare.json", "claim_extraction.json"]:
        (broken_fixture / name).write_text((FIXTURE_ROOT / "mock" / name).read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(Exception):
        run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=broken_fixture, slug="broken")
    operation_id = next(RunStore(vault).runs_root.iterdir()).name
    manifest = status(vault, operation_id)
    assert manifest.status == OperationStatus.failed
    assert [step for step in manifest.steps if step.status == StepStatus.failed][0].name == "page_planning"
    config = read_yaml(vault / ".llmwiki" / "config.yaml")
    config["providers"]["page_planning"] = {
        "spec": "mock:fixture",
        "fixture_dir": str(broken_fixture),
    }
    write_yaml(vault / ".llmwiki" / "config.yaml", config)
    (broken_fixture / "page_planning.json").write_text(
        (FIXTURE_ROOT / "mock" / "page_planning.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    resumed = resume_ingest(vault=vault, operation_id=operation_id)
    assert resumed.status == OperationStatus.drafted
    page_step = [step for step in resumed.steps if step.name == "page_planning"][0]
    assert page_step.attempts[-1].provider_spec == "mock:fixture"


def test_pipeline_uses_task_provider_config(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    config_path = vault / ".llmwiki" / "config.yaml"
    config = read_yaml(config_path)
    config["providers"]["page_planning"] = "human"
    write_yaml(config_path, config)

    with pytest.raises(Exception, match="HumanProvider"):
        run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="provider")
    operation_id = next(RunStore(vault).runs_root.iterdir()).name
    manifest = status(vault, operation_id)
    assert manifest.status == OperationStatus.failed
    failed_step = [step for step in manifest.steps if step.status == StepStatus.failed][0]
    assert failed_step.name == "page_planning"
    assert failed_step.attempts[-1].provider_spec == "human"


def test_mock_ingest_requires_fixture_dir_when_config_has_none(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    with pytest.raises(Exception, match="requires fixture_dir"):
        run_simplified_ingest(vault=vault, raw_file=raw, slug="no-fixture")
    assert latest_operation(vault) is None
    runs_root = RunStore(vault).runs_root
    assert not any(path.is_dir() and not (path / "manifest.json").exists() for path in runs_root.iterdir())


def test_provider_construction_failure_records_attempt_provider(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    config_path = vault / ".llmwiki" / "config.yaml"
    config = read_yaml(config_path)
    config["providers"]["page_planning"] = {
        "spec": "openai_compatible:planner",
        "endpoint": "http://127.0.0.1:1/v1/chat/completions",
        "api_key": "secret-provider-key",
    }
    write_yaml(config_path, config)

    with pytest.raises(Exception):
        run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="provider-build")
    operation_id = next(RunStore(vault).runs_root.iterdir()).name
    manifest = status(vault, operation_id)
    failed_step = [step for step in manifest.steps if step.status == StepStatus.failed][0]
    assert failed_step.name == "page_planning"
    assert failed_step.attempts[-1].provider_record_id == "provider-context-001"
    assert failed_step.attempts[-1].provider_spec == "openai_compatible:planner"
    assert failed_step.attempts[-1].provider_context_source == "initial_run"
    assert "secret-provider-key" not in failed_step.error


def test_model_step_attempts_record_provider_context(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="attempts")
    for step in manifest.steps:
        attempt = step.attempts[-1]
        if step.name in MODEL_BACKED_STEPS:
            assert attempt.provider_record_id == "provider-context-001"
            assert attempt.provider_spec == "mock:fixture"
            assert attempt.provider_context_source == "initial_run"
        else:
            assert attempt.provider_record_id is None
            assert attempt.provider_spec is None
            assert attempt.provider_context_source is None


def test_step_metadata_and_runners_stay_in_sync() -> None:
    assert STEP_NAMES == tuple(spec.name for spec in STEP_SPECS)
    assert MODEL_BACKED_STEPS == tuple(spec.name for spec in STEP_SPECS if spec.model_backed)
    assert tuple(STEP_RUNNERS) == STEP_NAMES


def test_model_artifacts_are_redacted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    vault, raw = make_vault(tmp_path)
    secret = "sk-artifact-secret"
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
            data["prepared_markdown"] += f"\n{secret}"
            data["review_notes"] = secret
        if task == "claim_extraction":
            data["claims"][0]["text"] += f" {secret}"
        if task == "page_planning":
            data["pages"][0]["summary"] += f" {secret}"
        return json.dumps(data, ensure_ascii=False)

    monkeypatch.setattr(OpenAICompatibleProvider, "generate_raw", fake_generate_raw)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, slug="redacted-artifacts")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)

    for path in run_dir.rglob("*"):
        if path.is_file():
            assert secret not in path.read_text(encoding="utf-8")


def test_resume_from_deletes_downstream_step_dirs(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="rerun")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    assert (run_dir / "claim_extraction" / "claims.json").exists()
    stale = run_dir / "draft_rendering" / "draft_pages" / "stale.md"
    stale.write_text("stale", encoding="utf-8")
    config = read_yaml(vault / ".llmwiki" / "config.yaml")
    config["providers"]["page_planning"] = {
        "spec": "mock:fixture",
        "fixture_dir": str(FIXTURE_ROOT / "mock"),
    }
    write_yaml(vault / ".llmwiki" / "config.yaml", config)
    resumed = resume_ingest(vault=vault, operation_id=manifest.operation_id, from_step="page_planning")
    assert resumed.status == OperationStatus.drafted
    assert not (run_dir / "archives").exists()
    assert not stale.exists()
    assert (run_dir / "page_planning" / "page_plan.json").exists()
    assert (run_dir / "draft_rendering" / "draft_pages").exists()
    assert (run_dir / "claim_extraction" / "claims.json").exists()
    page_step = [step for step in resumed.steps if step.name == "page_planning"][0]
    assert page_step.attempts[0].outputs == []
    assert page_step.attempts[-1].outputs


def test_resume_invalid_provider_config_does_not_delete_outputs(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="invalid")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    page_plan = run_dir / "page_planning" / "page_plan.json"
    assert page_plan.exists()
    config = read_yaml(vault / ".llmwiki" / "config.yaml")
    config["providers"]["page_planning"] = "missing:model"
    write_yaml(vault / ".llmwiki" / "config.yaml", config)
    with pytest.raises(Exception, match="Unknown provider"):
        resume_ingest(vault=vault, operation_id=manifest.operation_id, from_step="page_planning")
    assert page_plan.exists()


def test_invalid_task_provider_config_does_not_fallback_to_default(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="bad-fallback")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    page_plan = run_dir / "page_planning" / "page_plan.json"
    config = read_yaml(vault / ".llmwiki" / "config.yaml")
    config["providers"]["default"] = "human"
    config["providers"]["page_planning"] = ""
    write_yaml(vault / ".llmwiki" / "config.yaml", config)
    with pytest.raises(Exception, match="Invalid provider config for task: page_planning"):
        resume_ingest(vault=vault, operation_id=manifest.operation_id, from_step="page_planning")
    assert page_plan.exists()
    assert len(status(vault, manifest.operation_id).provider_contexts) == 1


def test_resume_from_outputless_step_does_not_require_provider_context(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="no-model-resume")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    apply_preview = run_dir / "apply_preview" / "apply_preview.json"
    assert apply_preview.exists()
    resumed = resume_ingest(vault=vault, operation_id=manifest.operation_id, from_step="validation")
    assert resumed.status == OperationStatus.drafted
    assert apply_preview.exists()
    assert len(status(vault, manifest.operation_id).provider_contexts) == 1


def test_resume_mock_provider_requires_current_fixture_dir(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="mock-resume")
    with pytest.raises(Exception, match="requires fixture_dir"):
        resume_ingest(vault=vault, operation_id=manifest.operation_id, from_step="page_planning")


def test_provider_config_rejects_unknown_field_before_deleting_outputs(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="secret")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    page_plan = run_dir / "page_planning" / "page_plan.json"
    assert page_plan.exists()
    config = read_yaml(vault / ".llmwiki" / "config.yaml")
    config["providers"]["page_planning"] = {"spec": "openai_compatible:gpt-test", "unexpected": "SECRET"}
    write_yaml(vault / ".llmwiki" / "config.yaml", config)
    with pytest.raises(Exception, match="unexpected"):
        resume_ingest(vault=vault, operation_id=manifest.operation_id, from_step="page_planning")
    assert page_plan.exists()


def test_resume_current_config_records_provider_on_failed_attempt(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="human")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    config = read_yaml(vault / ".llmwiki" / "config.yaml")
    config["providers"]["page_planning"] = "human"
    write_yaml(vault / ".llmwiki" / "config.yaml", config)

    with pytest.raises(Exception, match="HumanProvider"):
        resume_ingest(vault=vault, operation_id=manifest.operation_id, from_step="page_planning")

    resumed = status(vault, manifest.operation_id)
    page_step = [step for step in resumed.steps if step.name == "page_planning"][0]
    assert page_step.status == StepStatus.failed
    assert page_step.attempts[-1].provider_context_source == "resume_current_config"
    assert page_step.attempts[-1].provider_spec == "human"
    assert page_step.attempts[-1].provider_record_id == "provider-context-002"
    assert (run_dir / "claim_extraction" / "claims.json").exists()
    assert not (run_dir / "draft_rendering").exists()
    assert not (run_dir / "apply_preview").exists()


def test_raw_and_artifact_drift_block_resume(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="drift")
    raw.write_text(raw.read_text(encoding="utf-8") + "\nchanged", encoding="utf-8")
    with pytest.raises(VerifyError) as raw_error:
        resume_ingest(vault=vault, operation_id=manifest.operation_id)
    assert raw_error.value.result.issues[0].code == VerificationStatus.raw_changed

    vault2, raw2 = make_vault(tmp_path / "second")
    manifest2 = run_simplified_ingest(vault=vault2, raw_file=raw2, fixture_dir=FIXTURE_ROOT / "mock", slug="artifact")
    claims = RunStore(vault2).run_dir(manifest2.operation_id) / "claim_extraction" / "claims.json"
    claims.write_text(claims.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(VerifyError):
        resume_ingest(vault=vault2, operation_id=manifest2.operation_id)


def test_apply_preimage_repeat_and_applied_resume_block(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="apply")
    target = vault / "wiki" / "concepts" / "Concept_知识编译工程骨架.md"
    target.write_text("user edit", encoding="utf-8")
    with pytest.raises(ApplyError):
        apply_operation(vault, manifest.operation_id)

    vault2, raw2 = make_vault(tmp_path / "clean")
    manifest2 = run_simplified_ingest(vault=vault2, raw_file=raw2, fixture_dir=FIXTURE_ROOT / "mock", slug="apply")
    apply_operation(vault2, manifest2.operation_id)
    with pytest.raises(ApplyError):
        apply_operation(vault2, manifest2.operation_id)
    with pytest.raises(Exception):
        resume_ingest(vault=vault2, operation_id=manifest2.operation_id)


def test_apply_commit_requires_git_before_writing(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="commit")
    target = vault / "wiki" / "concepts" / "Concept_知识编译工程骨架.md"
    assert not target.exists()
    with pytest.raises(ApplyError, match="not a Git repository"):
        apply_operation(vault, manifest.operation_id, commit=True)
    assert not target.exists()


def test_apply_commit_rejects_tracked_llmwiki_before_writing(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="tracked")
    subprocess.run(["git", "init"], cwd=vault, check=True, capture_output=True)
    subprocess.run(["git", "add", "-f", ".llmwiki/config.yaml"], cwd=vault, check=True, capture_output=True)
    target = vault / "wiki" / "concepts" / "Concept_知识编译工程骨架.md"
    with pytest.raises(ApplyError, match=".llmwiki/ must not be tracked"):
        apply_operation(vault, manifest.operation_id, commit=True)
    assert not target.exists()


def test_apply_commit_does_not_include_unrelated_staged_files(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="commit-only")
    subprocess.run(["git", "init"], cwd=vault, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=vault, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=vault, check=True, capture_output=True)
    (vault / "baseline.txt").write_text("baseline\n", encoding="utf-8")
    subprocess.run(["git", "add", ".gitignore", "baseline.txt"], cwd=vault, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=vault, check=True, capture_output=True)

    unrelated = vault / "unrelated.txt"
    unrelated.write_text("do not include me\n", encoding="utf-8")
    subprocess.run(["git", "add", "unrelated.txt"], cwd=vault, check=True, capture_output=True)

    apply_operation(vault, manifest.operation_id, commit=True)

    committed = subprocess.run(
        ["git", "show", "--name-only", "--format=", "HEAD"],
        cwd=vault,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=vault,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert any(path.startswith("wiki/") for path in committed)
    assert "unrelated.txt" not in committed
    assert "unrelated.txt" in staged


@pytest.mark.parametrize("schema_version", ["operation_manifest.v3", "operation_manifest.v4"])
def test_old_manifest_is_rejected_with_clear_error(tmp_path: Path, schema_version: str) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="v2")
    manifest_path = RunStore(vault).manifest_path(manifest.operation_id)
    data = read_json(manifest_path)
    data["schema_version"] = schema_version
    write_json(manifest_path, data)
    with pytest.raises(ValueError, match=f"Unsupported manifest schema_version: {schema_version}"):
        read_manifest(manifest_path)
