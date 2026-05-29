from pathlib import Path

import pytest

from llmwiki_engine.apply import ApplyError, apply_operation
from llmwiki_engine.models import OperationStatus, StepStatus, VerificationStatus
from llmwiki_engine.pipeline import copy_fixture_raw, init_vault, resume_ingest, run_simplified_ingest, status
from llmwiki_engine.verify import VerifyError, verify_run
from llmwiki_engine.workspace import RunStore, WorkspaceError, ensure_v2_layout


ROOT = Path(__file__).parent
FIXTURE_ROOT = ROOT / "fixtures" / "simple_project"


def make_vault(tmp_path: Path) -> tuple[Path, Path]:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    raw = copy_fixture_raw(vault, FIXTURE_ROOT / "raw_project_note.md")
    return vault, raw


def test_init_creates_v2_layout_and_gitignore(tmp_path: Path) -> None:
    vault, _ = make_vault(tmp_path)
    assert (vault / ".llmwiki" / "config.yaml").exists()
    assert (vault / ".llmwiki" / "profiles" / "project_basic" / "profile.yaml").exists()
    assert (vault / ".llmwiki" / "applied" / "operations.jsonl").exists()
    assert (vault / ".llmwiki" / "runs").exists()
    gitignore_lines = (vault / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".llmwiki/runs/" in gitignore_lines
    assert ".llmwiki/" not in gitignore_lines


def test_legacy_stage_layout_blocks_without_v2_layout(tmp_path: Path) -> None:
    vault = tmp_path / "legacy"
    (vault / "stage" / "ingest").mkdir(parents=True)
    with pytest.raises(WorkspaceError):
        ensure_v2_layout(vault)


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
    assert (run_dir / "raw_index.json").exists()
    assert (run_dir / "apply_preview.json").exists()
    assert (run_dir / "draft_pages" / "sources" / "Source_raw_project_note.md").exists()
    assert (run_dir / "draft_pages" / "concepts" / "Concept_知识编译工程骨架.md").exists()
    assert verify_run(vault, loaded).ok
    written = apply_operation(vault, manifest.operation_id)
    assert (vault / "wiki" / "sources" / "Source_raw_project_note.md") in written
    assert (vault / ".llmwiki" / "applied" / "operations.jsonl").read_text(encoding="utf-8")
    assert status(vault, manifest.operation_id).status == OperationStatus.applied


def test_resume_after_failed_step(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    broken_fixture = tmp_path / "broken"
    broken_fixture.mkdir()
    for name in ["semantic_aggregation.json", "claim_extraction.json"]:
        (broken_fixture / name).write_text((FIXTURE_ROOT / "mock" / name).read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(Exception):
        run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=broken_fixture, slug="broken")
    operation_id = next(RunStore(vault).runs_root.iterdir()).name
    manifest = status(vault, operation_id)
    assert manifest.status == OperationStatus.failed
    assert [step for step in manifest.steps if step.status == StepStatus.failed][0].name == "page_planning"
    snapshot_fixture = RunStore(vault).run_dir(operation_id) / "snapshots" / "mock_fixture" / "page_planning.json"
    snapshot_fixture.write_text((FIXTURE_ROOT / "mock" / "page_planning.json").read_text(encoding="utf-8"), encoding="utf-8")
    resumed = resume_ingest(vault=vault, operation_id=operation_id)
    assert resumed.status == OperationStatus.drafted


def test_resume_from_archives_downstream_outputs(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="rerun")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    assert (run_dir / "claims.json").exists()
    resumed = resume_ingest(vault=vault, operation_id=manifest.operation_id, from_step="semantic_aggregation")
    assert resumed.status == OperationStatus.drafted
    archives = list((run_dir / "archives").glob("*"))
    assert archives
    assert (run_dir / "claims.json").exists()


def test_raw_and_artifact_drift_block_resume(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="drift")
    raw.write_text(raw.read_text(encoding="utf-8") + "\nchanged", encoding="utf-8")
    with pytest.raises(VerifyError) as raw_error:
        resume_ingest(vault=vault, operation_id=manifest.operation_id)
    assert raw_error.value.result.issues[0].code == VerificationStatus.raw_changed

    vault2, raw2 = make_vault(tmp_path / "second")
    manifest2 = run_simplified_ingest(vault=vault2, raw_file=raw2, fixture_dir=FIXTURE_ROOT / "mock", slug="artifact")
    claims = RunStore(vault2).run_dir(manifest2.operation_id) / "claims.json"
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
