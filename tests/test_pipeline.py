from pathlib import Path

from llmwiki_engine.apply import apply_operation
from llmwiki_engine.pipeline import copy_fixture_raw, init_vault, run_simplified_ingest, status


ROOT = Path(__file__).parent
FIXTURE_ROOT = ROOT / "fixtures" / "simple_project"


def test_init_ingest_status_apply_closes_loop(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    raw = copy_fixture_raw(vault, FIXTURE_ROOT / "raw_project_note.md")
    manifest = run_simplified_ingest(
        vault=vault,
        raw_file=raw,
        fixture_dir=FIXTURE_ROOT / "mock",
        profile_name="project_basic",
        slug="test",
    )
    assert manifest.status == "drafted"
    loaded = status(vault, manifest.operation_id)
    assert loaded.operation_id == manifest.operation_id
    stage_dir = vault / "stage" / "ingest" / manifest.operation_id
    assert (stage_dir / "raw_index.json").exists()
    assert (stage_dir / "draft_pages" / "sources" / "Source_raw_project_note.md").exists()
    assert (stage_dir / "draft_pages" / "concepts" / "Concept_知识编译工程骨架.md").exists()
    written = apply_operation(vault, manifest.operation_id)
    assert (vault / "wiki" / "sources" / "Source_raw_project_note.md") in written
    assert (vault / "logs" / "audit.md").exists()

