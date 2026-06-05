import json
from pathlib import Path

import pytest
import yaml

import llmwiki_engine.apply as apply_module
import llmwiki_engine.pipeline as pipeline_module
import llmwiki_engine.steps as steps_module
from llmwiki_engine.apply import ApplyError, apply_operation
from llmwiki_engine.hash_utils import sha256_file
from llmwiki_engine.io import read_json, read_jsonl, read_yaml, write_json, write_yaml
from llmwiki_engine.manifest import read_manifest
from llmwiki_engine.models import (
    CandidateResolutionArtifact,
    CandidateResolutionItem,
    DraftRenderingArtifact,
    OperationStatus,
    RunMode,
    SourceBasis,
    SourceDigestArtifact,
    SourceDigestCandidate,
    StepStatus,
    VerificationStatus,
)
from llmwiki_engine.pipeline import (
    PipelineError,
    STEP_RUNNERS,
    _STEP_RUN_FUNCTIONS,
    build_index_rows,
    backfill_missing_candidate_resolution_items,
    build_wiki_context_snapshot,
    build_wiki_merge_plan,
    copy_fixture_raw,
    init_vault,
    latest_operation,
    approve_review,
    resume_ingest,
    run_simplified_ingest,
    revise_review,
    status,
)
from llmwiki_engine.providers import OpenAICompatibleProvider
from llmwiki_engine.steps import (
    EVAL_MODULES,
    MODEL_BACKED_STEPS,
    STEP_NAMES,
    STEP_SPECS,
    StepSpec,
    require_step_output_dir,
    step_output_dir,
)
from llmwiki_engine.system_pages import local_date
from llmwiki_engine.verify import VerifyError, verify_run
from llmwiki_engine.workspace import RunStore, WorkspaceError, ensure_workspace_layout


ROOT = Path(__file__).parent
FIXTURE_ROOT = ROOT / "fixtures" / "simple_project"


def make_vault(tmp_path: Path) -> tuple[Path, Path]:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    raw = copy_fixture_raw(vault, FIXTURE_ROOT / "raw_project_note.md")
    return vault, raw


def resolution_item(
    candidate_id: str,
    *,
    page_type: str,
    target_path: str,
    display_title: str,
    summary: str = "Topic summary.",
) -> CandidateResolutionItem:
    return CandidateResolutionItem(
        page_plan_id=f"PP-{candidate_id}",
        source_basis=SourceBasis(source_candidate_ids=[candidate_id], source_locator="test"),
        page_type=page_type,
        display_title=display_title,
        path_stem=display_title,
        candidate_target_path=target_path,
        topic_summary=summary,
        why_this_page="It is useful wiki knowledge.",
        initial_section_intent="Test section intent.",
        coverage_notes=f"Covered {candidate_id}.",
        reason="test",
    )


def make_variant_fixture(tmp_path: Path, raw_rel: str, suffix: str) -> Path:
    fixture_dir = tmp_path / f"fixture-{suffix}"
    fixture_dir.mkdir()
    raw_prepare = read_json(FIXTURE_ROOT / "mock" / "raw_prepare.json")
    raw_prepare["source_raw_path"] = raw_rel
    write_json(fixture_dir / "raw_prepare.json", raw_prepare)
    digest = read_json(FIXTURE_ROOT / "mock" / "source_digest.json")
    digest["source_raw_path"] = raw_rel
    digest["summary"] = f"{digest['summary']} ({suffix})"
    for section in ["concepts", "designs"]:
        for candidate in digest[section]:
            candidate["candidate_id"] = f"{candidate['candidate_id']}_{suffix}"
            candidate["name"] = f"{candidate['name']} {suffix}"
            candidate["suggested_page_title"] = f"{candidate['suggested_page_title']} {suffix}"
            candidate["one_sentence_summary"] = f"{candidate['one_sentence_summary']} ({suffix})"
    write_json(fixture_dir / "source_digest.json", digest)
    resolution = read_json(FIXTURE_ROOT / "mock" / "candidate_resolution.json")
    resolution["items"][0]["source_basis"]["source_candidate_ids"] = [f"CAND001_{suffix}"]
    resolution["items"][0]["display_title"] = f"知识编译工程骨架 {suffix}"
    resolution["items"][0]["topic_summary"] = f"{resolution['items'][0]['topic_summary']} ({suffix})"
    resolution["items"][1]["source_basis"]["source_candidate_ids"] = [f"CAND002_{suffix}"]
    resolution["items"][1]["display_title"] = f"简化 Ingest 草稿流程 {suffix}"
    resolution["items"][1]["topic_summary"] = f"{resolution['items'][1]['topic_summary']} ({suffix})"
    write_json(fixture_dir / "candidate_resolution.json", resolution)
    merge = read_json(FIXTURE_ROOT / "mock" / "wiki_merge_planning.json")
    merge["items"][0]["source_basis"]["source_candidate_ids"] = [f"CAND001_{suffix}"]
    merge["items"][0]["display_title"] = f"知识编译工程骨架 {suffix}"
    merge["items"][1]["source_basis"]["source_candidate_ids"] = [f"CAND002_{suffix}"]
    merge["items"][1]["display_title"] = f"简化 Ingest 草稿流程 {suffix}"
    merge["items"][1]["related_pages"][0]["target_path"] = f"concepts/Concept_知识编译工程骨架 {suffix}.md"
    merge["items"][1]["related_pages"][0]["display_title"] = f"知识编译工程骨架 {suffix}"
    write_json(fixture_dir / "wiki_merge_planning.json", merge)
    draft = read_json(FIXTURE_ROOT / "mock" / "draft_rendering.json")
    draft["pages"][0]["canonical_target_path"] = f"concepts/Concept_知识编译工程骨架 {suffix}.md"
    draft["pages"][1]["canonical_target_path"] = f"designs/Design_简化 Ingest 草稿流程 {suffix}.md"
    write_json(fixture_dir / "draft_rendering.json", draft)
    return fixture_dir


def test_init_creates_workspace_layout_and_gitignore(tmp_path: Path) -> None:
    vault, _ = make_vault(tmp_path)
    assert (vault / ".llmwiki" / "config.yaml").exists()
    assert read_yaml(vault / ".llmwiki" / "config.yaml")["providers"] == {}
    assert (vault / ".llmwiki" / "profiles" / "project_basic" / "profile.yaml").exists()
    assert (vault / ".llmwiki" / "applied" / "operations.jsonl").exists()
    assert (vault / ".llmwiki" / "runs").exists()
    assert (vault / "wiki" / "index.md").exists()
    assert (vault / "wiki" / "log.md").exists()
    assert (vault / "wiki" / "logs").is_dir()
    gitignore_lines = (vault / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".llmwiki/" in gitignore_lines
    assert ".llmwiki/runs/" not in gitignore_lines
    assert not (vault / ".git").exists()


def test_raw_link_cleanup_normalizes_only_obsidian_text_wikilinks(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    raw = vault / "raw" / "links.md"
    raw.write_text(
        "---\n"
        "title: [[Lenny's Podcast]]\n"
        "related: [[target-page|显示名]]\n"
        "homepage: https://example.com\n"
        "---\n\n"
        "正文里有 [[Cat Wu]] 和 [[Anthropic|Anthropic 产品团队]]。\n"
        "网页链接保持 [site](https://example.com)，裸 URL https://example.com 也保持。\n"
        "HTML <a href=\"https://example.com\">site</a> 保持，reference [ref][id] 保持。\n"
        "普通相对链接 [note](notes/local.md) 保持。\n"
        "媒体 ![[image.png]] 保持。\n"
        "inline `[[Inline Code]]` 保持。\n"
        "```md\n"
        "[[Code Block]]\n"
        "![[code-image.png]]\n"
        "```\n\n"
        "[id]: https://example.com/ref\n",
        encoding="utf-8",
    )

    fixture_dir = make_variant_fixture(tmp_path, "raw/links.md", "cleanup")
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=fixture_dir, slug="cleanup")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    cleaned = raw.read_text(encoding="utf-8")
    cleanup = read_json(run_dir / "raw_link_cleanup" / "raw_link_cleanup.json")

    assert "title: Lenny's Podcast" in cleaned
    assert "related: 显示名" in cleaned
    assert "正文里有 Cat Wu 和 Anthropic 产品团队。" in cleaned
    assert "[site](https://example.com)" in cleaned
    assert "裸 URL https://example.com" in cleaned
    assert "<a href=\"https://example.com\">site</a>" in cleaned
    assert "[ref][id]" in cleaned
    assert "[note](notes/local.md)" in cleaned
    assert "![[image.png]]" in cleaned
    assert "`[[Inline Code]]`" in cleaned
    assert "[[Code Block]]" in cleaned
    assert cleanup["changed"] is True
    assert cleanup["cleaned_link_count"] == 4
    assert cleanup["preserved_media_embed_count"] == 1
    assert {link["cleanup_context"] for link in cleanup["links"]} == {"frontmatter", "body"}
    assert len(cleanup["warnings"]) == 1
    diff_text = (run_dir / "raw_link_cleanup" / "cleanup.diff").read_text(encoding="utf-8")
    assert "--- pre/raw/links.md" in diff_text
    assert "+++ post/raw/links.md" in diff_text
    assert "+title: Lenny's Podcast" in diff_text
    assert "-title: [[Lenny's Podcast]]" in diff_text
    assert read_json(RunStore(vault).manifest_path(manifest.operation_id))["raw_bindings"][0]["sha256"] == sha256_file(raw)


def test_changed_raw_link_cleanup_cannot_resume_from_cleanup_but_keeps_artifact_for_raw_prepare(
    tmp_path: Path,
) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    raw = vault / "raw" / "resume-links.md"
    raw.write_text("需要清理 [[Link]].\n", encoding="utf-8")
    fixture_dir = make_variant_fixture(tmp_path, "raw/resume-links.md", "cleanup-resume")
    write_yaml(
        vault / ".llmwiki" / "config.yaml",
        {
            "profile": "project_basic",
            "providers": {
                "default": {
                    "spec": "mock:fixture",
                    "fixture_dir": fixture_dir.as_posix(),
                }
            },
        },
    )
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=fixture_dir, slug="cleanup-resume")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    cleanup_report = run_dir / "raw_link_cleanup" / "raw_link_cleanup.json"
    assert cleanup_report.exists()

    with pytest.raises(PipelineError, match="Cannot resume from raw_link_cleanup"):
        resume_ingest(vault=vault, operation_id=manifest.operation_id, from_step="raw_link_cleanup")

    resumed = resume_ingest(vault=vault, operation_id=manifest.operation_id, from_step="raw_prepare")
    assert resumed.status == OperationStatus.drafted
    assert cleanup_report.exists()


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
    assert (run_dir / "raw_link_cleanup" / "raw_link_cleanup.json").exists()
    assert (run_dir / "raw_link_cleanup" / "raw_link_cleanup.md").exists()
    assert (run_dir / "raw_link_cleanup" / "cleanup.diff").exists()
    assert (run_dir / "raw_prepare" / "raw_preparation.json").exists()
    assert (run_dir / "raw_prepare" / "prepared.md").exists()
    assert (run_dir / "raw_prepare" / "preparation_review.md").exists()
    assert (run_dir / "prepared_raw_review" / "approved_prepared.md").exists()
    prepared_decision = read_json(run_dir / "prepared_raw_review" / "review_decision.json")
    assert prepared_decision["decision"] == "approved"
    assert prepared_decision["review_mode"] == "auto_stub"
    assert prepared_decision["auto_approved"] is True
    assert (run_dir / "source_digest" / "source_digest.json").exists()
    assert (run_dir / "source_digest_review" / "approved_digest.json").exists()
    digest_decision = read_json(run_dir / "source_digest_review" / "review_decision.json")
    assert digest_decision["decision"] == "approved"
    assert digest_decision["review_mode"] == "auto_stub"
    assert digest_decision["auto_approved"] is True
    assert (run_dir / "candidate_resolution" / "candidate_resolution.json").exists()
    assert (run_dir / "source_duplicate_guard" / "source_duplicate_guard.json").exists()
    assert (run_dir / "wiki_context_snapshot" / "wiki_context_snapshot.json").exists()
    assert (run_dir / "merge_plan_review" / "approved_merge_plan.json").exists()
    assert (run_dir / "draft_review" / "approved_write_manifest.json").exists()
    assert (run_dir / "apply_preview" / "apply_preview.json").exists()
    assert (run_dir / "draft_rendering" / "draft_pages" / "sources" / "Source_raw_project_note.md").exists()
    assert (run_dir / "draft_rendering" / "draft_pages" / "concepts" / "Concept_知识编译工程骨架.md").exists()
    assert (run_dir / "draft_rendering" / "draft_pages" / "designs" / "Design_简化 Ingest 草稿流程.md").exists()
    assert (run_dir / "draft_rendering" / "draft_pages" / "index.md").exists()
    assert (run_dir / "draft_rendering" / "draft_pages" / "log.md").exists()
    assert list((run_dir / "draft_rendering" / "draft_pages" / "logs").glob("*.md"))
    assert not (run_dir / "snapshots").exists()
    assert not (run_dir / "validation").exists()
    draft_root = run_dir / "draft_rendering" / "draft_pages"
    knowledge_text = (draft_root / "concepts" / "Concept_知识编译工程骨架.md").read_text(encoding="utf-8")
    frontmatter = yaml.safe_load(knowledge_text.split("---\n", 2)[1])
    assert frontmatter["source_raw_paths"] == ["raw/raw_project_note.md"]
    assert frontmatter["source_operation_ids"] == [manifest.operation_id]
    assert frontmatter["last_ingest_operation"] == manifest.operation_id
    assert "sources" not in frontmatter
    assert "[[sources/" not in knowledge_text
    assert "## Sources" not in knowledge_text
    assert "## 详情" in knowledge_text
    assert "## 价值点" in knowledge_text
    source_text = (draft_root / "sources" / "Source_raw_project_note.md").read_text(encoding="utf-8")
    source_frontmatter = yaml.safe_load(source_text.split("---\n", 2)[1])
    assert source_frontmatter["raw_cleanup_pre_sha256"] == source_frontmatter["raw_cleanup_post_sha256"]
    assert source_frontmatter["raw_cleanup_rule_version"] == "obsidian_text_wikilink.v1"
    assert source_frontmatter["raw_cleanup_artifact_ref"] == "raw_link_cleanup/raw_link_cleanup.json"
    assert source_frontmatter["raw_cleanup_changed"] is False
    assert source_frontmatter["raw_cleanup_cleaned_link_count"] == 0
    assert source_frontmatter["raw_cleanup_diff_ref"] == "raw_link_cleanup/cleanup.diff"
    assert "## 派生知识页" in source_text
    assert "`concepts/Concept_知识编译工程骨架.md`" in source_text
    assert "[[concepts/" not in source_text
    assert loaded.schema_version == "operation_manifest.v6"
    assert [ref.schema_version for ref in loaded.steps[0].outputs if ref.kind == "json"] == ["raw_link_cleanup.v1"]
    assert [ref.schema_version for ref in loaded.steps[1].outputs if ref.kind == "json"] == ["raw_preparation.v1"]
    assert [ref.schema_version for ref in loaded.steps[3].outputs if ref.kind == "json"] == ["source_digest.v2"]
    assert [ref.schema_version for ref in loaded.steps[4].outputs if ref.relative_path.endswith("approved_digest.json")] == [
        "source_digest.v2"
    ]
    assert [ref.schema_version for ref in loaded.steps[6].outputs if ref.kind == "json"] == ["candidate_resolution.v3"]
    assert [ref.schema_version for ref in loaded.steps[8].outputs if ref.relative_path.endswith("wiki_merge_plan.json")] == [
        "wiki_merge_plan.v4"
    ]
    plan = read_json(run_dir / "wiki_merge_planning" / "wiki_merge_plan.json")
    assert plan["schema_version"] == "wiki_merge_plan.v4"
    snapshot = read_json(run_dir / "wiki_context_snapshot" / "wiki_context_snapshot.json")
    snapshot_paths = {entry["path"] for entry in snapshot["entries"]}
    assert {
        "wiki/index.md",
        "wiki/log.md",
        f"wiki/logs/{snapshot['log_date']}.md",
        "wiki/sources/Source_raw_project_note.md",
        "wiki/concepts/Concept_知识编译工程骨架.md",
        "wiki/designs/Design_简化 Ingest 草稿流程.md",
    } <= snapshot_paths
    assert loaded.provider_contexts[0].providers["raw_prepare"].spec == "mock:fixture"
    assert loaded.provider_contexts[0].providers["raw_prepare"].fixture_dir == (FIXTURE_ROOT / "mock").resolve().as_posix()
    assert loaded.provider_contexts[0].providers["source_digest"].spec == "mock:fixture"
    assert "affected_steps" not in loaded.provider_contexts[0].model_dump()
    metrics = read_json(run_dir / "run_metrics.json")
    assert metrics["schema_version"] == "run_metrics.v1"
    assert metrics["steps"][0]["name"] == "raw_link_cleanup"
    assert metrics["steps"][0]["attempts"] == 1
    assert metrics["steps"][0]["last_duration_ms"] is not None
    assert metrics["steps"][0]["provider"] == "local"
    assert metrics["cleaned_link_count"] == 0
    events = read_jsonl(run_dir / "events.jsonl")
    assert {event["event"] for event in events} <= {"started", "completed", "failed"}
    completed_events = [event for event in events if event["event"] == "completed"]
    assert completed_events
    assert all("duration_ms" in event and event["duration_ms"] is not None for event in completed_events)
    assert verify_run(vault, loaded).ok
    written = apply_operation(vault, manifest.operation_id)
    assert (vault / "wiki" / "sources" / "Source_raw_project_note.md") in written
    assert (vault / "wiki" / "index.md") in written
    assert (vault / "wiki" / "log.md") in written
    assert any(path.match("*/wiki/logs/*.md") for path in written)
    receipts = read_jsonl(vault / ".llmwiki" / "applied" / "operations.jsonl")
    assert receipts
    assert receipts[-1]["profile"] == "project_basic"
    assert receipts[-1]["profile_version"] == "1"
    assert "profile_snapshot_hash" not in receipts[-1]
    assert "prepared_raw" in receipts[-1]
    assert receipts[-1]["raw_cleanup_artifact_ref"] == "raw_link_cleanup/raw_link_cleanup.json"
    assert receipts[-1]["raw_cleanup_diff_ref"] == "raw_link_cleanup/cleanup.diff"
    assert receipts[-1]["raw_cleanup_changed"] is False
    assert receipts[-1]["raw_cleanup_cleaned_link_count"] == 0
    assert status(vault, manifest.operation_id).status == OperationStatus.applied


def test_ingest_run_uses_vault_config_profile_by_default(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="research_basic")
    raw = copy_fixture_raw(vault, FIXTURE_ROOT / "raw_project_note.md")
    fixture_dir = tmp_path / "research-fixture"
    fixture_dir.mkdir()
    for name in ["raw_prepare.json", "source_digest.json", "candidate_resolution.json", "wiki_merge_planning.json", "draft_rendering.json"]:
        data = read_json(FIXTURE_ROOT / "mock" / name)
        if name == "candidate_resolution.json":
            data["items"][1]["page_type"] = "concept"
        if name == "wiki_merge_planning.json":
            data["items"][1]["page_type"] = "concept"
            data["items"][1]["canonical_target_path"] = "concepts/Concept_简化 Ingest 草稿流程.md"
        if name == "draft_rendering.json":
            data["pages"][1]["canonical_target_path"] = "concepts/Concept_简化 Ingest 草稿流程.md"
        write_json(fixture_dir / name, data)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=fixture_dir, slug="profile")
    assert manifest.profile == "research_basic"


def test_resume_after_failed_step(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    broken_fixture = tmp_path / "broken"
    broken_fixture.mkdir()
    for name in ["raw_prepare.json"]:
        (broken_fixture / name).write_text((FIXTURE_ROOT / "mock" / name).read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(Exception):
        run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=broken_fixture, slug="broken")
    operation_id = next(RunStore(vault).runs_root.iterdir()).name
    manifest = status(vault, operation_id)
    assert manifest.status == OperationStatus.failed
    assert [step for step in manifest.steps if step.status == StepStatus.failed][0].name == "source_digest"
    config = read_yaml(vault / ".llmwiki" / "config.yaml")
    config["providers"]["default"] = {
        "spec": "mock:fixture",
        "fixture_dir": str(broken_fixture),
    }
    config["providers"]["source_digest"] = {
        "spec": "mock:fixture",
        "fixture_dir": str(broken_fixture),
    }
    write_yaml(vault / ".llmwiki" / "config.yaml", config)
    (broken_fixture / "source_digest.json").write_text(
        (FIXTURE_ROOT / "mock" / "source_digest.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    for name in ["candidate_resolution.json", "wiki_merge_planning.json", "draft_rendering.json"]:
        (broken_fixture / name).write_text((FIXTURE_ROOT / "mock" / name).read_text(encoding="utf-8"), encoding="utf-8")
    resumed = resume_ingest(vault=vault, operation_id=operation_id)
    assert resumed.status == OperationStatus.drafted
    digest_step = [step for step in resumed.steps if step.name == "source_digest"][0]
    assert digest_step.attempts[-1].provider_spec == "mock:fixture"


def test_pipeline_uses_task_provider_config(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    config_path = vault / ".llmwiki" / "config.yaml"
    config = read_yaml(config_path)
    config["providers"]["source_digest"] = "human"
    write_yaml(config_path, config)

    with pytest.raises(Exception, match="HumanProvider"):
        run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="provider")
    operation_id = next(RunStore(vault).runs_root.iterdir()).name
    manifest = status(vault, operation_id)
    assert manifest.status == OperationStatus.failed
    failed_step = [step for step in manifest.steps if step.status == StepStatus.failed][0]
    assert failed_step.name == "source_digest"
    assert failed_step.attempts[-1].provider_spec == "human"


def test_mock_ingest_requires_fixture_dir_when_config_has_none(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    with pytest.raises(Exception, match="requires fixture_dir"):
        run_simplified_ingest(vault=vault, raw_file=raw, slug="no-fixture")
    assert latest_operation(vault) is None
    runs_root = RunStore(vault).runs_root
    assert not any(path.is_dir() and not (path / "manifest.json").exists() for path in runs_root.iterdir())


def test_ingest_requires_vault_config_json(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    (vault / ".llmwiki" / "config.json").unlink()

    with pytest.raises(RuntimeError, match="config.json is missing"):
        run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="missing-config")


def test_provider_construction_failure_records_attempt_provider(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    config_path = vault / ".llmwiki" / "config.yaml"
    config = read_yaml(config_path)
    config["providers"]["source_digest"] = {
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
    assert failed_step.name == "source_digest"
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
    assert EVAL_MODULES == tuple(spec.name for spec in STEP_SPECS if spec.eval_supported)
    assert set(EVAL_MODULES) <= set(STEP_NAMES)
    assert set(_STEP_RUN_FUNCTIONS) == set(STEP_NAMES)
    assert tuple(STEP_RUNNERS) == STEP_NAMES


def test_step_output_dir_helpers_use_step_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "run"
    assert step_output_dir(run_dir, "raw_prepare") == run_dir / "raw_prepare"
    assert require_step_output_dir(run_dir, "apply_preview") == run_dir / "apply_preview"
    assert step_output_dir(run_dir, "validation") is None
    monkeypatch.setattr(
        steps_module,
        "STEP_SPECS",
        (*STEP_SPECS, StepSpec("renamed_step", False, "custom_output_dir")),
    )
    assert step_output_dir(run_dir, "renamed_step") == run_dir / "custom_output_dir"
    assert step_output_dir(run_dir, "renamed_step") != run_dir / "renamed_step"
    with pytest.raises(ValueError, match="Step has no output directory: validation"):
        require_step_output_dir(run_dir, "validation")
    with pytest.raises(ValueError, match="Unknown step: missing"):
        step_output_dir(run_dir, "missing")


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
        if task == "source_digest":
            data["summary"] += f" {secret}"
            data["concepts"][0]["why_matters"] += f" {secret}"
        return json.dumps(data, ensure_ascii=False)

    monkeypatch.setattr(OpenAICompatibleProvider, "generate_raw", fake_generate_raw)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, slug="redacted-artifacts")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)

    for path in run_dir.rglob("*"):
        if path.is_file():
            assert secret not in path.read_text(encoding="utf-8")


def test_source_digest_provider_payload_omits_formal_candidate_suggested_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vault, raw = make_vault(tmp_path)
    captured_payloads: dict[str, dict] = {}
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

    def fake_generate_raw(self, task, payload, output_model):
        captured_payloads[task] = payload
        data = read_json(FIXTURE_ROOT / "mock" / f"{task}.json")
        return json.dumps(data, ensure_ascii=False)

    monkeypatch.setattr(OpenAICompatibleProvider, "generate_raw", fake_generate_raw)

    run_simplified_ingest(vault=vault, raw_file=raw, slug="payload")

    source_payload = captured_payloads["source_digest"]
    assert "suggested_action" not in source_payload["contract"]["candidate_fields"]
    assert "suggested_action" in source_payload["contract"]["weak_or_noise_fields"]
    assert "candidate_coverage_required_ids" in captured_payloads["candidate_resolution"]


def test_candidate_resolution_backfills_missed_open_question_candidates(tmp_path: Path) -> None:
    vault, _ = make_vault(tmp_path)
    profile = pipeline_module.load_profile(vault / ".llmwiki" / "profiles" / "project_basic")
    digest = SourceDigestArtifact(
        source_raw_path="raw/sample.md",
        summary="Digest summary.",
        concepts=[
            SourceDigestCandidate(
                candidate_id="CAND001",
                name="Knowledge digestion",
                type="concept",
                one_sentence_summary="Knowledge digestion summary.",
                why_matters="It matters.",
                wiki_value="It belongs in the wiki.",
                suggested_page_title="Knowledge digestion",
            )
        ],
        open_questions=[
            SourceDigestCandidate(
                candidate_id="oq-1",
                name="How should review work?",
                type="open_question",
                one_sentence_summary="Review gate design remains unresolved.",
                why_matters="It affects update quality.",
                wiki_value="It should become a tracked open question.",
                suggested_page_title="Review Gate Granularity",
                source_locator="discussion / open question",
            )
        ],
    )
    artifact = CandidateResolutionArtifact(
        items=[
            CandidateResolutionItem(
                source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
                page_type="concept",
                display_title="Knowledge digestion",
                topic_summary="Knowledge digestion summary.",
                why_this_page="It belongs in the wiki.",
                reason="model covered concept",
            )
        ]
    )

    backfilled = backfill_missing_candidate_resolution_items(artifact, digest, profile)
    finalized = pipeline_module.finalize_candidate_resolution(vault, profile, backfilled)

    assert {item.source_basis.source_candidate_ids[0] for item in finalized.items} == {"CAND001", "oq-1"}
    open_question = [item for item in finalized.items if item.source_basis.source_candidate_ids == ["oq-1"]][0]
    assert open_question.page_type == "open_question"
    assert open_question.candidate_target_path == "open_questions/Open_Question_Review Gate Granularity.md"


def test_resume_from_deletes_downstream_step_dirs(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="rerun")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    assert (run_dir / "prepared_raw_review" / "approved_prepared.md").exists()
    assert (run_dir / "source_digest" / "source_digest.json").exists()
    stale = run_dir / "draft_rendering" / "draft_pages" / "stale.md"
    stale.write_text("stale", encoding="utf-8")
    config = read_yaml(vault / ".llmwiki" / "config.yaml")
    config["providers"]["default"] = {
        "spec": "mock:fixture",
        "fixture_dir": str(FIXTURE_ROOT / "mock"),
    }
    config["providers"]["source_digest"] = {
        "spec": "mock:fixture",
        "fixture_dir": str(FIXTURE_ROOT / "mock"),
    }
    write_yaml(vault / ".llmwiki" / "config.yaml", config)
    resumed = resume_ingest(vault=vault, operation_id=manifest.operation_id, from_step="source_digest")
    assert resumed.status == OperationStatus.drafted
    assert not (run_dir / "archives").exists()
    assert not stale.exists()
    assert (run_dir / "source_digest" / "source_digest.json").exists()
    assert (run_dir / "draft_rendering" / "draft_pages").exists()
    assert (run_dir / "prepared_raw_review" / "approved_prepared.md").exists()
    digest_step = [step for step in resumed.steps if step.name == "source_digest"][0]
    assert digest_step.attempts[0].outputs == []
    assert digest_step.attempts[-1].outputs


def test_resume_invalid_provider_config_does_not_delete_outputs(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="invalid")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    digest = run_dir / "source_digest" / "source_digest.json"
    assert digest.exists()
    config = read_yaml(vault / ".llmwiki" / "config.yaml")
    config["providers"]["source_digest"] = "missing:model"
    write_yaml(vault / ".llmwiki" / "config.yaml", config)
    with pytest.raises(Exception, match="Unknown provider"):
        resume_ingest(vault=vault, operation_id=manifest.operation_id, from_step="source_digest")
    assert digest.exists()


def test_invalid_task_provider_config_does_not_fallback_to_default(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="bad-fallback")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    digest = run_dir / "source_digest" / "source_digest.json"
    config = read_yaml(vault / ".llmwiki" / "config.yaml")
    config["providers"]["default"] = "human"
    config["providers"]["source_digest"] = ""
    write_yaml(vault / ".llmwiki" / "config.yaml", config)
    with pytest.raises(Exception, match="Invalid provider config for task: source_digest"):
        resume_ingest(vault=vault, operation_id=manifest.operation_id, from_step="source_digest")
    assert digest.exists()
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
        resume_ingest(vault=vault, operation_id=manifest.operation_id, from_step="source_digest")


def test_provider_config_rejects_unknown_field_before_deleting_outputs(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="secret")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    digest = run_dir / "source_digest" / "source_digest.json"
    assert digest.exists()
    config = read_yaml(vault / ".llmwiki" / "config.yaml")
    config["providers"]["source_digest"] = {"spec": "openai_compatible:gpt-test", "unexpected": "SECRET"}
    write_yaml(vault / ".llmwiki" / "config.yaml", config)
    with pytest.raises(Exception, match="unexpected"):
        resume_ingest(vault=vault, operation_id=manifest.operation_id, from_step="source_digest")
    assert digest.exists()


def test_resume_current_config_records_provider_on_failed_attempt(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="human")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    config = read_yaml(vault / ".llmwiki" / "config.yaml")
    config["providers"]["default"] = {
        "spec": "mock:fixture",
        "fixture_dir": str(FIXTURE_ROOT / "mock"),
    }
    config["providers"]["source_digest"] = "human"
    write_yaml(vault / ".llmwiki" / "config.yaml", config)

    with pytest.raises(Exception, match="HumanProvider"):
        resume_ingest(vault=vault, operation_id=manifest.operation_id, from_step="source_digest")

    resumed = status(vault, manifest.operation_id)
    digest_step = [step for step in resumed.steps if step.name == "source_digest"][0]
    assert digest_step.status == StepStatus.failed
    assert digest_step.attempts[-1].provider_context_source == "resume_current_config"
    assert digest_step.attempts[-1].provider_spec == "human"
    assert digest_step.attempts[-1].provider_record_id == "provider-context-002"
    assert (run_dir / "prepared_raw_review" / "approved_prepared.md").exists()
    assert not (run_dir / "draft_rendering").exists()
    assert not (run_dir / "apply_preview").exists()


def test_resume_mode_is_immutable_and_rejects_before_writing(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="mode")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    stale = run_dir / "draft_rendering" / "draft_pages" / "stale.md"
    stale.write_text("stale", encoding="utf-8")
    before = read_json(RunStore(vault).manifest_path(manifest.operation_id))

    with pytest.raises(PipelineError, match="run_mode is immutable"):
        resume_ingest(vault=vault, operation_id=manifest.operation_id, from_step="source_digest", run_mode=RunMode.standard)

    after = read_json(RunStore(vault).manifest_path(manifest.operation_id))
    assert after == before
    assert stale.exists()

    vault2, raw2 = make_vault(tmp_path / "standard")
    standard = run_simplified_ingest(
        vault=vault2,
        raw_file=raw2,
        fixture_dir=FIXTURE_ROOT / "mock",
        slug="standard-mode",
        run_mode=RunMode.standard,
    )
    before_standard = read_json(RunStore(vault2).manifest_path(standard.operation_id))
    with pytest.raises(PipelineError, match="run_mode is immutable"):
        resume_ingest(vault=vault2, operation_id=standard.operation_id, from_step="validation", run_mode=RunMode.dev)
    assert read_json(RunStore(vault2).manifest_path(standard.operation_id)) == before_standard


def test_resume_mode_can_confirm_existing_mode(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="same-mode")
    resumed = resume_ingest(vault=vault, operation_id=manifest.operation_id, from_step="validation", run_mode=RunMode.dev)
    assert resumed.status == OperationStatus.drafted
    assert resumed.run_mode == RunMode.dev


def test_raw_and_artifact_drift_block_resume(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="drift")
    raw.write_text(raw.read_text(encoding="utf-8") + "\nchanged", encoding="utf-8")
    with pytest.raises(VerifyError) as raw_error:
        resume_ingest(vault=vault, operation_id=manifest.operation_id)
    assert raw_error.value.result.issues[0].code == VerificationStatus.raw_changed

    vault2, raw2 = make_vault(tmp_path / "second")
    manifest2 = run_simplified_ingest(vault=vault2, raw_file=raw2, fixture_dir=FIXTURE_ROOT / "mock", slug="artifact")
    digest = RunStore(vault2).run_dir(manifest2.operation_id) / "source_digest" / "source_digest.json"
    digest.write_text(digest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
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


def test_applied_operation_rejects_review_mutations(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="review-immutable")
    apply_operation(vault, manifest.operation_id)

    with pytest.raises(PipelineError, match="Applied operations are immutable"):
        approve_review(vault, manifest.operation_id, "merge_plan_review")
    with pytest.raises(PipelineError, match="Applied operations are immutable"):
        revise_review(vault, manifest.operation_id, "draft_review")


def test_review_approve_requires_awaiting_review_state(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="review-state")

    with pytest.raises(PipelineError, match="merge_plan_review is not awaiting_review"):
        approve_review(vault, manifest.operation_id, "merge_plan_review")


def test_model_related_pages_are_deterministically_resolved_before_rendering(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    fixture_dir = tmp_path / "related-fixture"
    fixture_dir.mkdir()
    for name in ["raw_prepare.json", "source_digest.json", "candidate_resolution.json", "wiki_merge_planning.json", "draft_rendering.json"]:
        data = read_json(FIXTURE_ROOT / "mock" / name)
        if name == "wiki_merge_planning.json":
            data["items"][0]["related_pages"] = [
                {
                    "target_path": "designs/Design_简化 Ingest 草稿流程.md",
                    "display_title": "简化 Ingest 草稿流程",
                    "source": "source_digest",
                    "reason": "工程骨架和草稿流程互相依赖。",
                },
                {
                    "target_path": "sources/Source_Bad.md",
                    "display_title": "Bad Source",
                    "source": "wiki_context",
                    "reason": "source page must stay out of related.",
                },
                {
                    "target_path": "concepts/Concept_Missing.md",
                    "display_title": "Missing",
                    "source": "wiki_context",
                    "reason": "unknown page must stay unresolved.",
                },
            ]
        write_json(fixture_dir / name, data)

    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=fixture_dir, slug="related-resolve")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    plan = read_json(run_dir / "wiki_merge_planning" / "wiki_merge_plan.json")
    first = plan["items"][0]
    assert [item["target_path"] for item in first["related_pages"]] == ["designs/Design_简化 Ingest 草稿流程.md"]
    assert any("Source_Bad" in item for item in first["related_unresolved"])
    assert any("Concept_Missing" in item for item in first["related_unresolved"])

    concept_text = (run_dir / "draft_rendering" / "draft_pages" / "concepts" / "Concept_知识编译工程骨架.md").read_text(
        encoding="utf-8"
    )
    assert "[[designs/Design_简化 Ingest 草稿流程|简化 Ingest 草稿流程]]" in concept_text
    assert "[[sources/" not in concept_text
    assert "Concept_Missing" not in concept_text


def test_draft_rendering_normalizes_model_section_keys(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    fixture_dir = tmp_path / "draft-section-fixture"
    fixture_dir.mkdir()
    for name in ["raw_prepare.json", "source_digest.json", "candidate_resolution.json", "wiki_merge_planning.json", "draft_rendering.json"]:
        data = read_json(FIXTURE_ROOT / "mock" / name)
        if name == "draft_rendering.json":
            data["pages"][0]["section_bodies"] = {
                "Summary": "这是模型用英文 key 写出的摘要。",
                "Background": "模型补充了背景，但没有使用 canonical key。",
                "Product Philosophy": "模型拆出了产品哲学观察。",
                "Collaboration with Boris Cherny": "模型拆出了协作背景。",
                "Advice for PMs": ["PM 应该把建议写到价值点中。", "数组也要转成 Markdown 字符串。"],
                "Additional Notes": "这是模型明确放进自由发挥区的观察。",
                "Open Questions": "这个主题还有一个未决问题。",
            }
        write_json(fixture_dir / name, data)

    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=fixture_dir, slug="draft-sections")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    draft = read_json(run_dir / "draft_rendering" / "draft_rendering.json")
    first_page = draft["pages"][0]
    assert set(first_page["section_bodies"]) == {
        "summary",
        "detail",
        "value_points",
        "additional_notes",
        "open_questions",
    }
    assert first_page["section_bodies"]["summary"] == "这是模型用英文 key 写出的摘要。"
    assert "### Background" in first_page["section_bodies"]["detail"]
    assert "### Product Philosophy" in first_page["section_bodies"]["detail"]
    assert first_page["section_bodies"]["value_points"] == "- PM 应该把建议写到价值点中。\n- 数组也要转成 Markdown 字符串。"
    assert first_page["section_bodies"]["additional_notes"] == "这是模型明确放进自由发挥区的观察。"

    concept_text = (run_dir / "draft_rendering" / "draft_pages" / "concepts" / "Concept_知识编译工程骨架.md").read_text(
        encoding="utf-8"
    )
    assert "## 补充观察" in concept_text
    assert "这是模型明确放进自由发挥区的观察。" in concept_text


def test_blocked_apply_eligibility_stops_at_merge_plan_review(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    fixture_dir = tmp_path / "blocked-fixture"
    fixture_dir.mkdir()
    for name in ["raw_prepare.json", "source_digest.json", "candidate_resolution.json", "wiki_merge_planning.json", "draft_rendering.json"]:
        data = read_json(FIXTURE_ROOT / "mock" / name)
        if name == "wiki_merge_planning.json":
            data["items"][0]["action"] = "create"
            data["items"][0]["apply_eligibility"] = "blocked"
            data["items"][0]["blocked_reason"] = "需要先人工确认该主题是否应该写入。"
        write_json(fixture_dir / name, data)

    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=fixture_dir, slug="blocked")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    plan = read_json(run_dir / "wiki_merge_planning" / "wiki_merge_plan.json")

    assert plan["items"][0]["action"] == "needs_human_decision"
    assert manifest.status == OperationStatus.awaiting_review
    assert [step for step in manifest.steps if step.status == StepStatus.awaiting_review][0].name == "merge_plan_review"
    assert not (run_dir / "draft_rendering").exists()


def test_wiki_merge_planning_rejects_duplicate_writable_targets(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    fixture_dir = tmp_path / "duplicate-target-fixture"
    fixture_dir.mkdir()
    for name in ["raw_prepare.json", "source_digest.json", "candidate_resolution.json", "wiki_merge_planning.json", "draft_rendering.json"]:
        data = read_json(FIXTURE_ROOT / "mock" / name)
        if name == "wiki_merge_planning.json":
            data["items"][1]["action"] = "update"
            data["items"][1]["matched_page"] = "concepts/Concept_知识编译工程骨架.md"
            data["items"][1]["canonical_target_path"] = "concepts/Concept_知识编译工程骨架.md"
        write_json(fixture_dir / name, data)

    with pytest.raises(PipelineError, match="duplicate writable target paths"):
        run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=fixture_dir, slug="duplicate-target")


def test_wiki_merge_planning_rejects_source_graph_links_in_merge_fields(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    fixture_dir = tmp_path / "source-graph-fixture"
    fixture_dir.mkdir()
    for name in ["raw_prepare.json", "source_digest.json", "candidate_resolution.json", "wiki_merge_planning.json", "draft_rendering.json"]:
        data = read_json(FIXTURE_ROOT / "mock" / name)
        if name == "wiki_merge_planning.json":
            data["items"][0]["value_points"] = ["不要把 [[sources/Source_Bad]] 写入知识页。"]
        write_json(fixture_dir / name, data)

    with pytest.raises(PipelineError, match="must not contain source graph links"):
        run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=fixture_dir, slug="source-graph")


def test_m3_update_target_stops_at_draft_review_without_apply_preview(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    target = vault / "wiki" / "concepts" / "Concept_知识编译工程骨架.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("existing knowledge\n", encoding="utf-8")

    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="existing")

    assert target.read_text(encoding="utf-8") == "existing knowledge\n"
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    plan = read_json(run_dir / "wiki_merge_planning" / "wiki_merge_plan.json")
    update_items = [item for item in plan["items"] if item["canonical_target_path"] == "concepts/Concept_知识编译工程骨架.md"]
    assert update_items
    assert update_items[0]["action"] == "update"
    assert update_items[0]["matched_page"] == "concepts/Concept_知识编译工程骨架.md"
    manifest = status(vault, manifest.operation_id)
    assert manifest.status == OperationStatus.awaiting_review
    awaiting_step = [step for step in manifest.steps if step.status == StepStatus.awaiting_review][0]
    assert awaiting_step.name == "draft_review"
    assert (run_dir / "draft_rendering" / "draft_write_manifest.json").exists()
    assert (run_dir / "draft_review" / "pending_write_manifest.json").exists()
    assert not (run_dir / "apply_preview").exists()


def test_wiki_merge_planning_normalizes_model_wiki_prefix_on_update_target(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    target = vault / "wiki" / "concepts" / "Concept_知识编译工程骨架.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("existing knowledge\n", encoding="utf-8")
    fixture_dir = tmp_path / "wiki-prefixed-target-fixture"
    fixture_dir.mkdir()
    for name in ["raw_prepare.json", "source_digest.json", "candidate_resolution.json", "wiki_merge_planning.json", "draft_rendering.json"]:
        data = read_json(FIXTURE_ROOT / "mock" / name)
        if name == "wiki_merge_planning.json":
            data["items"][0]["action"] = "update"
            data["items"][0]["canonical_target_path"] = "concepts/Concept_知识编译工程骨架.md"
            data["items"][0]["matched_page"] = "wiki/concepts/Concept_知识编译工程骨架.md"
        write_json(fixture_dir / name, data)

    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=fixture_dir, slug="wiki-prefixed-target")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    plan = read_json(run_dir / "wiki_merge_planning" / "wiki_merge_plan.json")
    first = plan["items"][0]

    assert first["action"] == "update"
    assert first["canonical_target_path"] == "concepts/Concept_知识编译工程骨架.md"
    assert first["matched_page"] == "concepts/Concept_知识编译工程骨架.md"
    assert manifest.status == OperationStatus.awaiting_review
    assert [step for step in manifest.steps if step.status == StepStatus.awaiting_review][0].name == "draft_review"


def test_source_recorded_operation_cannot_resume(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    for rel, title in [
        ("wiki/concepts/Concept_知识编译工程骨架.md", "知识编译工程骨架"),
        ("wiki/designs/Design_简化 Ingest 草稿流程.md", "简化 Ingest 草稿流程"),
    ]:
        target = vault / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "---\n"
            f"llmwiki_type: {'design' if '/designs/' in rel else 'concept'}\n"
            f"title: {title}\n"
            "aliases: []\n"
            f"summary: {title} 已覆盖。\n"
            "created: 2026-01-01\n"
            "updated: 2026-01-01\n"
            "---\n\n"
            f"# {title}\n",
            encoding="utf-8",
        )
    fixture_dir = tmp_path / "noop-fixture"
    fixture_dir.mkdir()
    for name in ["raw_prepare.json", "source_digest.json", "candidate_resolution.json", "wiki_merge_planning.json", "draft_rendering.json"]:
        data = read_json(FIXTURE_ROOT / "mock" / name)
        if name == "wiki_merge_planning.json":
            for item in data["items"]:
                item["action"] = "noop"
                item["apply_eligibility"] = "source_only"
                item["matched_page"] = None
        if name == "draft_rendering.json":
            data["pages"] = []
        write_json(fixture_dir / name, data)

    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=fixture_dir, slug="source-recorded")
    apply_operation(vault, manifest.operation_id)

    assert status(vault, manifest.operation_id).status == OperationStatus.source_recorded
    with pytest.raises(PipelineError, match="Applied operations are immutable"):
        resume_ingest(vault=vault, operation_id=manifest.operation_id, from_step="validation")


def test_update_draft_preserves_existing_provenance_frontmatter(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    target = vault / "wiki" / "concepts" / "Concept_知识编译工程骨架.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "---\n"
        "llmwiki_type: concept\n"
        "title: 知识编译工程骨架\n"
        "aliases:\n"
        "  - 工程骨架\n"
        "summary: old summary\n"
        "created: 2026-01-01\n"
        "updated: 2026-01-02\n"
        "source_raw_paths:\n"
        "  - raw/old.md\n"
        "source_raw_hashes:\n"
        "  - old-raw-hash\n"
        "source_prepared_hashes:\n"
        "  - old-prepared-hash\n"
        "source_operation_ids:\n"
        "  - OLD-OP\n"
        "last_ingest_operation: OLD-OP\n"
        "---\n\n"
        "# 知识编译工程骨架\n\nold content\n",
        encoding="utf-8",
    )

    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="provenance")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    draft_text = (run_dir / "draft_rendering" / "draft_pages" / "concepts" / "Concept_知识编译工程骨架.md").read_text(
        encoding="utf-8"
    )
    frontmatter = yaml.safe_load(draft_text.split("---\n", 2)[1])

    assert frontmatter["aliases"] == ["工程骨架"]
    assert str(frontmatter["created"]) == "2026-01-01"
    assert frontmatter["source_raw_paths"] == ["raw/old.md", "raw/raw_project_note.md"]
    assert frontmatter["source_raw_hashes"][0] == "old-raw-hash"
    assert frontmatter["source_raw_hashes"][1] == sha256_file(raw)
    assert frontmatter["source_prepared_hashes"][0] == "old-prepared-hash"
    assert frontmatter["source_operation_ids"] == ["OLD-OP", manifest.operation_id]
    assert frontmatter["last_ingest_operation"] == manifest.operation_id


@pytest.mark.parametrize("from_step", ["validation", "apply_preview"])
def test_resume_cannot_skip_awaiting_draft_review(tmp_path: Path, from_step: str) -> None:
    vault, raw = make_vault(tmp_path)
    target = vault / "wiki" / "concepts" / "Concept_知识编译工程骨架.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("existing knowledge\n", encoding="utf-8")

    run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug=f"skip-{from_step}")
    operation_id = latest_operation(vault)
    assert operation_id is not None
    before = read_json(RunStore(vault).manifest_path(operation_id))

    with pytest.raises(PipelineError, match="Cannot resume from .*upstream step draft_review is awaiting_review"):
        resume_ingest(vault=vault, operation_id=operation_id, from_step=from_step)

    run_dir = RunStore(vault).run_dir(operation_id)
    after = read_json(RunStore(vault).manifest_path(operation_id))
    assert after == before
    assert (run_dir / "draft_rendering").exists()
    assert not (run_dir / "apply_preview").exists()
    assert read_jsonl(vault / ".llmwiki" / "applied" / "operations.jsonl") == []


def test_merge_plan_review_pending_can_be_approved_after_manual_edit(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    fixture_dir = tmp_path / "needs-human-fixture"
    fixture_dir.mkdir()
    for name in ["raw_prepare.json", "source_digest.json", "candidate_resolution.json", "wiki_merge_planning.json", "draft_rendering.json"]:
        data = read_json(FIXTURE_ROOT / "mock" / name)
        if name == "wiki_merge_planning.json":
            data["items"][0]["action"] = "needs_human_decision"
            data["items"][0]["apply_eligibility"] = "blocked"
            data["items"][0]["blocked_reason"] = "需要人工决定是否创建。"
        write_json(fixture_dir / name, data)

    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=fixture_dir, slug="needs-human")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    assert manifest.status == OperationStatus.awaiting_review
    assert [step for step in manifest.steps if step.status == StepStatus.awaiting_review][0].name == "merge_plan_review"
    assert not (run_dir / "draft_rendering").exists()

    pending_path = run_dir / "merge_plan_review" / "pending_merge_plan.json"
    pending = read_json(pending_path)
    pending["items"][0]["action"] = "create"
    pending["items"][0]["apply_eligibility"] = "applyable"
    pending["items"][0]["blocked_reason"] = ""
    write_json(pending_path, pending)

    approved = approve_review(vault, manifest.operation_id, "merge_plan_review")
    assert approved.status == OperationStatus.running
    assert (run_dir / "merge_plan_review" / "approved_merge_plan.json").exists()

    config = read_yaml(vault / ".llmwiki" / "config.yaml")
    config["providers"]["default"] = {
        "spec": "mock:fixture",
        "fixture_dir": str(fixture_dir),
    }
    write_yaml(vault / ".llmwiki" / "config.yaml", config)
    resumed = resume_ingest(vault=vault, operation_id=manifest.operation_id)
    assert resumed.status == OperationStatus.drafted
    assert (run_dir / "apply_preview" / "apply_preview.json").exists()


def test_apply_rejects_incomplete_steps_even_if_manifest_is_marked_drafted(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="tampered")
    manifest_path = RunStore(vault).manifest_path(manifest.operation_id)
    data = read_json(manifest_path)
    data["status"] = "drafted"
    for step in data["steps"]:
        if step["name"] == "wiki_merge_planning":
            step["status"] = "failed"
            step["error"] = "tampered failed planning"
        if step["name"] == "draft_rendering":
            step["status"] = "pending"
    write_json(manifest_path, data)

    with pytest.raises(ApplyError, match="incomplete step\\(s\\).*wiki_merge_planning=failed"):
        apply_operation(vault, manifest.operation_id)


def test_apply_rejects_empty_preview_targets(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="empty-preview")
    preview_path = RunStore(vault).run_dir(manifest.operation_id) / "apply_preview" / "apply_preview.json"
    preview = read_json(preview_path)
    preview["targets"] = []
    write_json(preview_path, preview)
    manifest_path = RunStore(vault).manifest_path(manifest.operation_id)
    manifest_data = read_json(manifest_path)
    for step in manifest_data["steps"]:
        if step["name"] == "apply_preview":
            step["outputs"][0]["sha256"] = sha256_file(preview_path)
            step["outputs"][0]["size_bytes"] = preview_path.stat().st_size
            step["attempts"][-1]["outputs"][0]["sha256"] = sha256_file(preview_path)
            step["attempts"][-1]["outputs"][0]["size_bytes"] = preview_path.stat().st_size
    write_json(manifest_path, manifest_data)

    with pytest.raises(ApplyError, match="Apply preview has no targets"):
        apply_operation(vault, manifest.operation_id)


def test_source_digest_v2_strips_formal_candidate_suggested_action(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    fixture_dir = tmp_path / "fixture"
    fixture_dir.mkdir()
    for name in ["raw_prepare.json", "source_digest.json", "candidate_resolution.json", "wiki_merge_planning.json", "draft_rendering.json"]:
        data = read_json(FIXTURE_ROOT / "mock" / name)
        if name == "source_digest.json":
            data["concepts"][0]["suggested_action"] = "update"
        write_json(fixture_dir / name, data)

    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=fixture_dir, slug="update")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    digest = read_json(run_dir / "source_digest" / "source_digest.json")
    approved_digest = read_json(run_dir / "source_digest_review" / "approved_digest.json")
    resolution = read_json(run_dir / "candidate_resolution" / "candidate_resolution.json")
    plan = read_json(run_dir / "wiki_merge_planning" / "wiki_merge_plan.json")
    digest_markdown = (run_dir / "source_digest" / "source_digest.md").read_text(encoding="utf-8")

    assert digest["schema_version"] == "source_digest.v2"
    assert "suggested_action" not in digest["concepts"][0]
    assert "suggested_action" not in approved_digest["concepts"][0]
    assert "Suggested" not in digest_markdown
    assert "action" not in resolution["items"][0]
    assert plan["items"][0]["action"] == "create"


def test_source_page_neutralizes_model_generated_graph_links(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    fixture_dir = tmp_path / "fixture-links"
    fixture_dir.mkdir()
    for name in ["raw_prepare.json", "source_digest.json", "candidate_resolution.json", "wiki_merge_planning.json", "draft_rendering.json"]:
        data = read_json(FIXTURE_ROOT / "mock" / name)
        if name == "source_digest.json":
            data["summary"] = "摘要提到 [[concepts/Concept_X|X]] 和 [Y](concepts/Concept_Y.md)。"
            data["key_takeaways"] = [
                "关键收获链接 [[sources/Source_Bad]]。",
                "另一个链接 [Z](wiki/concepts/Concept_Z.md)。",
            ]
        write_json(fixture_dir / name, data)

    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=fixture_dir, slug="source-links")
    source_text = (
        RunStore(vault).run_dir(manifest.operation_id)
        / "draft_rendering"
        / "draft_pages"
        / "sources"
        / "Source_raw_project_note.md"
    ).read_text(encoding="utf-8")

    assert "[[" not in source_text
    assert "](concepts/" not in source_text
    assert "](wiki/concepts/" not in source_text
    assert "`concepts/Concept_X / X`" in source_text
    assert "Y (`concepts/Concept_Y.md`)" in source_text


def test_display_title_strips_type_prefix_without_changing_target_path(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    fixture_dir = tmp_path / "fixture"
    fixture_dir.mkdir()
    for name in ["raw_prepare.json", "source_digest.json", "candidate_resolution.json", "wiki_merge_planning.json", "draft_rendering.json"]:
        data = read_json(FIXTURE_ROOT / "mock" / name)
        if name == "source_digest.json":
            data["concepts"][0]["suggested_page_title"] = "Concept_Prefixed Title"
            data["concepts"][0]["name"] = "Prefixed Title"
        if name == "candidate_resolution.json":
            data["items"][0]["display_title"] = "Concept_Prefixed Title"
        if name == "wiki_merge_planning.json":
            data["items"][0]["display_title"] = "Concept_Prefixed Title"
        if name == "draft_rendering.json":
            data["pages"][0]["canonical_target_path"] = "concepts/Concept_Prefixed Title.md"
        write_json(fixture_dir / name, data)

    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=fixture_dir, slug="prefixed")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    resolution = read_json(run_dir / "candidate_resolution" / "candidate_resolution.json")
    item = [item for item in resolution["items"] if "CAND001" in item["source_basis"]["source_candidate_ids"]][0]
    assert item["candidate_target_path"] == "concepts/Concept_Prefixed Title.md"
    assert item["display_title"] == "Prefixed Title"

    draft_root = run_dir / "draft_rendering" / "draft_pages"
    concept_text = (draft_root / "concepts" / "Concept_Prefixed Title.md").read_text(encoding="utf-8")
    index_text = (draft_root / "index.md").read_text(encoding="utf-8")
    assert "# Prefixed Title" in concept_text
    assert "Prefixed Title" in index_text
    assert "[[concepts/Concept_Prefixed Title]]" in index_text
    assert "Concept_Prefixed Title|Concept_Prefixed Title" not in index_text


def test_cross_type_title_prefix_does_not_pollute_target_path(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    fixture_dir = tmp_path / "fixture"
    fixture_dir.mkdir()
    for name in ["raw_prepare.json", "source_digest.json", "candidate_resolution.json", "wiki_merge_planning.json", "draft_rendering.json"]:
        data = read_json(FIXTURE_ROOT / "mock" / name)
        if name == "source_digest.json":
            data["concepts"][0]["type"] = "entity"
            data["concepts"][0]["suggested_page_title"] = "Concept_Foo"
            data["concepts"][0]["name"] = "Foo"
        if name == "candidate_resolution.json":
            data["items"][0]["page_type"] = "entity"
            data["items"][0]["display_title"] = "Concept_Foo"
        if name == "wiki_merge_planning.json":
            data["items"][0]["page_type"] = "entity"
            data["items"][0]["display_title"] = "Concept_Foo"
        if name == "draft_rendering.json":
            data["pages"][0]["canonical_target_path"] = "entities/Entity_Foo.md"
        write_json(fixture_dir / name, data)

    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=fixture_dir, slug="cross-prefix")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    resolution = read_json(run_dir / "candidate_resolution" / "candidate_resolution.json")
    item = [item for item in resolution["items"] if "CAND001" in item["source_basis"]["source_candidate_ids"]][0]

    assert item["page_type"] == "entity"
    assert item["candidate_target_path"] == "entities/Entity_Foo.md"
    assert item["display_title"] == "Foo"


def test_m2_blocks_overwriting_incompatible_system_page(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    (vault / "wiki" / "index.md").write_text("# My human index\n", encoding="utf-8")

    with pytest.raises(PipelineError, match="system page is incompatible with current MVP page contract"):
        run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="human-index")

    assert (vault / "wiki" / "index.md").read_text(encoding="utf-8") == "# My human index\n"


def test_m2_blocks_v1_system_marker(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    (vault / "wiki" / "index.md").write_text(
        "# index\n\n<!-- llmwiki:system-page:v1 -->\n",
        encoding="utf-8",
    )

    with pytest.raises(PipelineError, match="system page is incompatible with current MVP page contract"):
        run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="v1-index")


def test_wiki_context_snapshot_includes_source_daily_target_and_existing_metadata(tmp_path: Path) -> None:
    vault, _ = make_vault(tmp_path)
    existing = vault / "wiki" / "concepts" / "Concept_Existing.md"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text(
        "---\n"
        "llmwiki_type: concept\n"
        "title: Existing Concept\n"
        "aliases:\n"
        "  - Existing Alias\n"
        "summary: Existing summary.\n"
        "created: 2026-01-01\n"
        "updated: 2026-01-02\n"
        "---\n\n"
        "# Existing Concept\n",
        encoding="utf-8",
    )
    resolution = CandidateResolutionArtifact(
        items=[
            resolution_item(
                "CAND001",
                page_type="concept",
                target_path="concepts/Concept_Test.md",
                display_title="Concept Test",
            )
        ]
    )
    snapshot = build_wiki_context_snapshot(
        vault,
        resolution,
        log_date="2026-06-03",
        source_target_path="sources/Source_Test.md",
    )
    paths = {entry.path for entry in snapshot.entries}
    assert {
        "wiki/index.md",
        "wiki/log.md",
        "wiki/logs/2026-06-03.md",
        "wiki/sources/Source_Test.md",
        "wiki/concepts/Concept_Test.md",
        "wiki/concepts/Concept_Existing.md",
    } <= paths
    existing_entry = [entry for entry in snapshot.entries if entry.path == "wiki/concepts/Concept_Existing.md"][0]
    assert existing_entry.metadata is not None
    assert existing_entry.metadata.title == "Existing Concept"
    assert existing_entry.metadata.aliases == ["Existing Alias"]


def test_related_pages_resolve_deterministically_from_candidates_and_snapshot(tmp_path: Path) -> None:
    vault, _ = make_vault(tmp_path)
    existing = vault / "wiki" / "concepts" / "Concept_Existing.md"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text(
        "---\n"
        "llmwiki_type: concept\n"
        "title: Concept_Existing Concept\n"
        "aliases:\n"
        "  - Existing Alias\n"
        "summary: Existing summary.\n"
        "updated: 2026-01-02\n"
        "---\n\n"
        "# Existing Concept\n",
        encoding="utf-8",
    )
    digest = SourceDigestArtifact(
        source_raw_path="raw/sample.md",
        summary="Digest summary.",
        concepts=[
            SourceDigestCandidate(
                candidate_id="CAND001",
                name="Knowledge digestion",
                type="concept",
                one_sentence_summary="Knowledge digestion summary.",
                why_matters="It matters.",
                wiki_value="It belongs in the wiki.",
                suggested_page_title="Knowledge digestion",
                related_candidates=["CAND002", "Existing Alias", "Knowledge digestion", "Unknown Related"],
            ),
            SourceDigestCandidate(
                candidate_id="CAND002",
                name="Review loop",
                type="concept",
                one_sentence_summary="Review loop summary.",
                why_matters="It matters.",
                wiki_value="It belongs in the wiki.",
                suggested_page_title="Review loop",
            ),
        ],
    )
    resolution = CandidateResolutionArtifact(
        items=[
            resolution_item(
                "CAND001",
                page_type="concept",
                target_path="concepts/Concept_Knowledge digestion.md",
                display_title="Knowledge digestion",
            ),
            resolution_item(
                "CAND002",
                page_type="concept",
                target_path="concepts/Concept_Review loop.md",
                display_title="Review loop",
            ),
        ]
    )
    snapshot = build_wiki_context_snapshot(
        vault,
        resolution,
        log_date="2026-06-03",
        source_target_path="sources/Source_Test.md",
    )

    plan = build_wiki_merge_plan(resolution, digest, snapshot, log_date="2026-06-03")
    first = plan.items[0]
    assert [(item.target_path, item.display_title, item.source) for item in first.related_pages] == [
        ("concepts/Concept_Review loop.md", "Review loop", "source_digest"),
        ("concepts/Concept_Existing.md", "Existing Concept", "wiki_context"),
    ]
    assert "Unknown Related" in first.related_unresolved
    assert all(item.target_path != first.canonical_target_path for item in first.related_pages)


def test_source_type_plan_items_are_defensively_excluded_from_index_and_related(tmp_path: Path) -> None:
    vault, _ = make_vault(tmp_path)
    digest = SourceDigestArtifact(
        source_raw_path="raw/sample.md",
        summary="Digest summary.",
        concepts=[
            SourceDigestCandidate(
                candidate_id="CAND001",
                name="Knowledge digestion",
                type="concept",
                one_sentence_summary="Knowledge digestion summary.",
                why_matters="It matters.",
                wiki_value="It belongs in the wiki.",
                suggested_page_title="Knowledge digestion",
                related_candidates=["CAND_SOURCE"],
            ),
            SourceDigestCandidate(
                candidate_id="CAND_SOURCE",
                name="Bad source candidate",
                type="source",
                one_sentence_summary="A source-like candidate that should not become graph knowledge.",
                why_matters="It should be filtered defensively.",
                wiki_value="It should not become a knowledge node.",
                suggested_page_title="Bad source candidate",
            ),
        ],
    )
    resolution = CandidateResolutionArtifact(
        items=[
            resolution_item(
                "CAND001",
                page_type="concept",
                target_path="concepts/Concept_Knowledge digestion.md",
                display_title="Knowledge digestion",
            ),
            resolution_item(
                "CAND_SOURCE",
                page_type="source",
                target_path="sources/Source_Bad source candidate.md",
                display_title="Bad source candidate",
            ),
        ]
    )
    snapshot = build_wiki_context_snapshot(
        vault,
        resolution,
        log_date="2026-06-03",
        source_target_path="sources/Source_Test.md",
    )
    plan = build_wiki_merge_plan(resolution, digest, snapshot, log_date="2026-06-03")
    concept_item = [item for item in plan.items if "CAND001" in item.source_basis.source_candidate_ids][0]
    profile = pipeline_module.load_profile(vault / ".llmwiki" / "profiles" / "project_basic")
    rows = build_index_rows(profile, plan, DraftRenderingArtifact(pages=[]), snapshot)

    assert concept_item.related_pages == []
    assert "CAND_SOURCE" in concept_item.related_unresolved
    assert not any("Source_Bad source candidate" in row["page"] for row in rows)


def test_ambiguous_existing_related_alias_stays_unresolved(tmp_path: Path) -> None:
    vault, _ = make_vault(tmp_path)
    for name in ["First", "Second"]:
        existing = vault / "wiki" / "concepts" / f"Concept_{name}.md"
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_text(
            "---\n"
            "llmwiki_type: concept\n"
            f"title: {name}\n"
            "aliases:\n"
            "  - Shared Alias\n"
            f"summary: {name} summary.\n"
            "updated: 2026-01-02\n"
            "---\n\n"
            f"# {name}\n",
            encoding="utf-8",
        )
    digest = SourceDigestArtifact(
        source_raw_path="raw/sample.md",
        summary="Digest summary.",
        concepts=[
            SourceDigestCandidate(
                candidate_id="CAND001",
                name="Knowledge digestion",
                type="concept",
                one_sentence_summary="Knowledge digestion summary.",
                why_matters="It matters.",
                wiki_value="It belongs in the wiki.",
                suggested_page_title="Knowledge digestion",
                related_candidates=["Shared Alias"],
            )
        ],
    )
    resolution = CandidateResolutionArtifact(
        items=[
            resolution_item(
                "CAND001",
                page_type="concept",
                target_path="concepts/Concept_Knowledge digestion.md",
                display_title="Knowledge digestion",
            )
        ]
    )
    snapshot = build_wiki_context_snapshot(
        vault,
        resolution,
        log_date="2026-06-03",
        source_target_path="sources/Source_Test.md",
    )

    plan = build_wiki_merge_plan(resolution, digest, snapshot, log_date="2026-06-03")

    assert plan.items[0].related_pages == []
    assert "Shared Alias" in plan.items[0].related_unresolved


def test_index_rebuilds_from_snapshot_metadata_and_drops_stale_rows(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    today = local_date()
    (vault / "wiki" / "index.md").write_text(
        "# index\n\n"
        "<!-- llmwiki:system-page:v3 -->\n\n"
        "## Concepts\n\n"
        "| Page | Summary | Updated |\n"
        "| --- | --- | --- |\n"
        "| [[concepts/Concept_Stale|Stale]] | stale summary | 2026-01-01 |\n\n"
        "## Open Questions\n\n"
        "| Question | Page | Updated |\n"
        "| --- | --- | --- |\n",
        encoding="utf-8",
    )
    existing = vault / "wiki" / "concepts" / "Concept_Old.md"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text(
        "---\n"
        "llmwiki_type: concept\n"
        "title: Old Concept\n"
        "aliases:\n"
        "  - Old Alias\n"
        "summary: old summary\n"
        "created: 2026-01-01\n"
        "updated: 2026-01-01\n"
        "---\n\n"
        "# Old Concept\n",
        encoding="utf-8",
    )
    misplaced_source = vault / "wiki" / "misc" / "Source_Misplaced.md"
    misplaced_source.parent.mkdir(parents=True, exist_ok=True)
    misplaced_source.write_text(
        "---\n"
        "llmwiki_type: source\n"
        "title: Misplaced Source\n"
        "aliases: []\n"
        "summary: source summary\n"
        "updated: 2026-01-02\n"
        "---\n\n"
        "# Misplaced Source\n",
        encoding="utf-8",
    )
    (vault / "wiki" / "log.md").write_text(
        "# log\n\n"
        "<!-- llmwiki:system-page:v3 -->\n\n"
        "| Date | Operations | Sources |\n"
        "| --- | ---: | --- |\n"
        f"| [[logs/{today}]] | 1 | `raw/old.md` |\n\n"
        "Latest operation: `OLD`\n",
        encoding="utf-8",
    )
    daily = vault / "wiki" / "logs" / f"{today}.md"
    daily.write_text(
        f"# {today}\n\n"
        "<!-- llmwiki:system-page:v3 -->\n\n"
        "| Operation | Raw | Created | Updated | No-op duplicate | Tensions |\n"
        "| --- | --- | ---: | ---: | ---: | ---: |\n"
        "| `OLD` | `raw/old.md` | 1 | 0 | 0 | 0 |\n",
        encoding="utf-8",
    )

    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="merge-system")
    apply_operation(vault, manifest.operation_id)

    index_text = (vault / "wiki" / "index.md").read_text(encoding="utf-8")
    assert "Old Concept" in index_text
    assert "[[concepts/Concept_Old]]" in index_text
    assert "知识编译工程骨架" in index_text
    assert "[[concepts/Concept_知识编译工程骨架]]" in index_text
    assert "Concept_Stale" not in index_text
    assert "Misplaced Source" not in index_text
    assert "Source Pages" not in index_text
    assert "[[sources/" not in index_text
    log_text = (vault / "wiki" / "log.md").read_text(encoding="utf-8")
    assert f"| [[logs/{today}]] | 2 | `raw/old.md`, `raw/raw_project_note.md` |" in log_text
    daily_text = daily.read_text(encoding="utf-8")
    assert "| `OLD` | `raw/old.md` | 1 | 0 | 0 | 0 |" in daily_text
    assert f"| `{manifest.operation_id}` | `raw/raw_project_note.md` | 2 | 0 | 0 | 0 |" in daily_text


def test_wiki_context_drift_blocks_rerender_from_stale_plan(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="context")
    (vault / "wiki" / "index.md").write_text("changed after planning\n", encoding="utf-8")

    with pytest.raises(PipelineError, match="wiki context changed after planning"):
        resume_ingest(vault=vault, operation_id=manifest.operation_id, from_step="draft_rendering")


def test_multiple_drafts_apply_requires_latest_wiki_context(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    raw_b = vault / "raw" / "second_project_note.md"
    raw_b.write_text(raw.read_text(encoding="utf-8") + "\nSecond raw variant.\n", encoding="utf-8")
    fixture_b = make_variant_fixture(tmp_path, "raw/second_project_note.md", "Second")
    first = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="first")
    second = run_simplified_ingest(vault=vault, raw_file=raw_b, fixture_dir=fixture_b, slug="second")

    apply_operation(vault, first.operation_id)

    with pytest.raises(ApplyError, match="当前 operation 的 apply plan 已过期"):
        apply_operation(vault, second.operation_id)

    config = read_yaml(vault / ".llmwiki" / "config.yaml")
    config["providers"]["default"] = {
        "spec": "mock:fixture",
        "fixture_dir": str(fixture_b),
    }
    write_yaml(vault / ".llmwiki" / "config.yaml", config)
    resumed = resume_ingest(vault=vault, operation_id=second.operation_id)
    assert resumed.status == OperationStatus.drafted
    written = apply_operation(vault, second.operation_id)
    assert vault / "wiki" / "sources" / "Source_second_project_note.md" in written


@pytest.mark.parametrize(
    ("path", "mutation", "message"),
    [
        ("wiki/index.md", "changed", "wiki context changed after planning"),
        ("wiki/log.md", "disappeared", "wiki context disappeared after planning"),
        ("wiki/logs/{log_date}.md", "appeared", "wiki context appeared after planning"),
        ("wiki/sources/Source_raw_project_note.md", "appeared", "wiki context appeared after planning"),
        ("wiki/concepts/Concept_知识编译工程骨架.md", "appeared", "wiki context appeared after planning"),
    ],
)
def test_wiki_context_drift_detects_changed_appeared_and_disappeared(
    tmp_path: Path,
    path: str,
    mutation: str,
    message: str,
) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug=f"drift-{mutation}")
    plan = read_json(RunStore(vault).run_dir(manifest.operation_id) / "wiki_merge_planning" / "wiki_merge_plan.json")
    target = vault / path.format(log_date=plan["log_date"])
    if mutation == "changed":
        target.write_text(target.read_text(encoding="utf-8") + "\nchanged", encoding="utf-8")
    elif mutation == "disappeared":
        target.unlink()
    elif mutation == "appeared":
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("appeared\n", encoding="utf-8")
    else:
        raise AssertionError(mutation)

    with pytest.raises(PipelineError, match=message):
        resume_ingest(vault=vault, operation_id=manifest.operation_id, from_step="draft_rendering")


def test_log_date_is_pinned_by_merge_plan_for_downstream_rendering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    vault, raw = make_vault(tmp_path)
    monkeypatch.setattr(pipeline_module, "local_date", lambda: "2026-06-03")
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="pinned-date")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    plan = read_json(run_dir / "wiki_merge_planning" / "wiki_merge_plan.json")
    assert plan["log_date"] == "2026-06-03"

    def fail_if_called() -> str:
        raise AssertionError("downstream rendering must use plan.log_date")

    monkeypatch.setattr(pipeline_module, "local_date", fail_if_called)
    config = read_yaml(vault / ".llmwiki" / "config.yaml")
    config["providers"]["default"] = {
        "spec": "mock:fixture",
        "fixture_dir": str(FIXTURE_ROOT / "mock"),
    }
    write_yaml(vault / ".llmwiki" / "config.yaml", config)
    resumed = resume_ingest(vault=vault, operation_id=manifest.operation_id, from_step="draft_rendering")
    assert resumed.status == OperationStatus.drafted
    draft_root = run_dir / "draft_rendering" / "draft_pages"
    assert (draft_root / "logs" / "2026-06-03.md").exists()
    for rel in [
        "sources/Source_raw_project_note.md",
        "concepts/Concept_知识编译工程骨架.md",
        "designs/Design_简化 Ingest 草稿流程.md",
        "index.md",
        "log.md",
        "logs/2026-06-03.md",
    ]:
        assert "2026-06-03" in (draft_root / rel).read_text(encoding="utf-8")


def test_apply_rejects_preview_paths_outside_expected_roots(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="bad-path")
    preview_path = RunStore(vault).run_dir(manifest.operation_id) / "apply_preview" / "apply_preview.json"
    preview = read_json(preview_path)
    preview["targets"][0]["target_path"] = "../outside.md"
    write_json(preview_path, preview)
    manifest_path = RunStore(vault).manifest_path(manifest.operation_id)
    manifest_data = read_json(manifest_path)
    for step in manifest_data["steps"]:
        if step["name"] == "apply_preview":
            step["outputs"][0]["sha256"] = sha256_file(preview_path)
            step["outputs"][0]["size_bytes"] = preview_path.stat().st_size
            step["attempts"][-1]["outputs"][0]["sha256"] = sha256_file(preview_path)
            step["attempts"][-1]["outputs"][0]["size_bytes"] = preview_path.stat().st_size
    write_json(manifest_path, manifest_data)

    with pytest.raises(ApplyError, match="targets do not match approved draft manifest"):
        apply_operation(vault, manifest.operation_id)
    assert not (vault.parent / "outside.md").exists()


def test_apply_commit_is_disabled_before_verify_or_writing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="commit")
    target = vault / "wiki" / "concepts" / "Concept_知识编译工程骨架.md"
    assert not target.exists()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("verify should not run for disabled commit apply")

    monkeypatch.setattr(apply_module, "require_verified", fail_if_called)
    with pytest.raises(ApplyError, match="apply --commit is disabled"):
        apply_operation(vault, manifest.operation_id, commit=True)

    assert not target.exists()
    assert read_json(RunStore(vault).manifest_path(manifest.operation_id))["status"] == "drafted"
    assert read_jsonl(vault / ".llmwiki" / "applied" / "operations.jsonl") == []


def test_plain_apply_records_apply_failed_on_write_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="rollback")
    preview = read_json(RunStore(vault).run_dir(manifest.operation_id) / "apply_preview" / "apply_preview.json")
    real_write = apply_module._write_bytes_atomic
    calls = {"count": 0}

    def fail_on_second_write(path: Path, data: bytes) -> None:
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("simulated write failure")
        real_write(path, data)

    monkeypatch.setattr(apply_module, "_write_bytes_atomic", fail_on_second_write)

    with pytest.raises(ApplyError, match="Apply write failed; inspect written targets before retry"):
        apply_operation(vault, manifest.operation_id)

    assert calls["count"] == 2
    manifest_data = read_json(RunStore(vault).manifest_path(manifest.operation_id))
    assert manifest_data["status"] == "apply_failed"
    failure = read_json(RunStore(vault).run_dir(manifest.operation_id) / "apply_failed.json")
    assert len(failure["written_targets"]) == 1
    assert (vault / failure["written_targets"][0]).exists()
    assert read_jsonl(vault / ".llmwiki" / "applied" / "operations.jsonl") == []
    with pytest.raises(PipelineError, match="apply_failed operations cannot be resumed"):
        resume_ingest(vault=vault, operation_id=manifest.operation_id, from_step="validation")


def test_plain_apply_records_apply_failed_on_receipt_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="receipt-failure")
    preview = read_json(RunStore(vault).run_dir(manifest.operation_id) / "apply_preview" / "apply_preview.json")

    def fail_append(*args, **kwargs):
        raise OSError("simulated receipt failure")

    monkeypatch.setattr(apply_module, "append_jsonl", fail_append)

    with pytest.raises(ApplyError, match="Apply write failed; inspect written targets before retry"):
        apply_operation(vault, manifest.operation_id)

    manifest_data = read_json(RunStore(vault).manifest_path(manifest.operation_id))
    assert manifest_data["status"] == "apply_failed"
    failure = read_json(RunStore(vault).run_dir(manifest.operation_id) / "apply_failed.json")
    assert failure["error"] == "simulated receipt failure"
    assert len(failure["written_targets"]) == len(preview["targets"])
    assert read_jsonl(vault / ".llmwiki" / "applied" / "operations.jsonl") == []


def test_standard_apply_and_commit_are_rejected_before_writing(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(
        vault=vault,
        raw_file=raw,
        fixture_dir=FIXTURE_ROOT / "mock",
        slug="standard",
        run_mode=RunMode.standard,
    )
    target = vault / "wiki" / "concepts" / "Concept_知识编译工程骨架.md"
    with pytest.raises(ApplyError, match="standard mode does not allow manual apply"):
        apply_operation(vault, manifest.operation_id)
    with pytest.raises(ApplyError, match="apply --commit is disabled"):
        apply_operation(vault, manifest.operation_id, commit=True)
    assert not target.exists()


@pytest.mark.parametrize("schema_version", ["operation_manifest.v4", "operation_manifest.v7"])
def test_unsupported_manifest_schema_is_rejected_with_clear_error(tmp_path: Path, schema_version: str) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="v2")
    manifest_path = RunStore(vault).manifest_path(manifest.operation_id)
    data = read_json(manifest_path)
    data["schema_version"] = schema_version
    write_json(manifest_path, data)
    with pytest.raises(ValueError, match="operation is incompatible with current MVP pipeline"):
        read_manifest(manifest_path)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda names: names[:-1],
        lambda names: [*names, "extra_step"],
        lambda names: [*names, names[-1]],
        lambda names: [names[1], names[0], *names[2:]],
    ],
)
def test_manifest_step_topology_must_match_current_mvp_pipeline(tmp_path: Path, mutator) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="topology")
    manifest_path = RunStore(vault).manifest_path(manifest.operation_id)
    data = read_json(manifest_path)
    steps_by_name = {step["name"]: step for step in data["steps"]}
    mutated_names = mutator([step["name"] for step in data["steps"]])
    data["steps"] = [dict(steps_by_name.get(name, data["steps"][0]), name=name) for name in mutated_names]
    write_json(manifest_path, data)

    with pytest.raises(ValueError, match="operation is incompatible with current MVP pipeline"):
        read_manifest(manifest_path)


@pytest.mark.parametrize("missing_key", ["run_mode", "status", "provider_contexts", "updated_at"])
def test_manifest_v4_requires_persisted_top_level_fields(tmp_path: Path, missing_key: str) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="missing-field")
    manifest_path = RunStore(vault).manifest_path(manifest.operation_id)
    data = read_json(manifest_path)
    data.pop(missing_key)
    write_json(manifest_path, data)

    with pytest.raises(ValueError, match="operation is incompatible with current MVP pipeline"):
        read_manifest(manifest_path)


def test_manifest_v4_rejects_extra_top_level_fields_with_mvp_message(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="extra-field")
    manifest_path = RunStore(vault).manifest_path(manifest.operation_id)
    data = read_json(manifest_path)
    data["legacy_status_summary"] = {"old": True}
    write_json(manifest_path, data)

    with pytest.raises(ValueError, match="operation is incompatible with current MVP pipeline"):
        read_manifest(manifest_path)


def test_manifest_reader_rejects_non_object_json_with_mvp_message(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, fixture_dir=FIXTURE_ROOT / "mock", slug="bad-root")
    manifest_path = RunStore(vault).manifest_path(manifest.operation_id)
    manifest_path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="operation is incompatible with current MVP pipeline"):
        read_manifest(manifest_path)
