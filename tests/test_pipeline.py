import json
import logging
import os
import sys
import types
import warnings
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from helpers import copy_fixture_raw
import llmwiki_engine.apply as apply_module
import llmwiki_engine.draft_validation as draft_validation_module
import llmwiki_engine.draft_grounding as draft_grounding
import llmwiki_engine.draft_outputs as draft_outputs_module
import llmwiki_engine.draft_rendering_payloads as draft_rendering_payloads_module
import llmwiki_engine.draft_reviewing as draft_reviewing_module
import llmwiki_engine.merge_plan_refinement as merge_plan_refinement_module
import llmwiki_engine.merge_reporting as merge_reporting_module
import llmwiki_engine.pipeline as pipeline_module
import llmwiki_engine.planning_payloads as planning_payloads_module
import llmwiki_engine.related_pages as related_pages_module
import llmwiki_engine.run_metrics as run_metrics_module
import llmwiki_engine.source_digest_budget as source_digest_budget
import llmwiki_engine.source_digest_payload as source_digest_payload_module
import llmwiki_engine.source_refs as source_refs_module
import llmwiki_engine.steps as steps_module
from llmwiki_engine import open_questions as open_questions_module
from llmwiki_engine import page_sections as page_sections_module
from llmwiki_engine import retrieval as retrieval_module
from llmwiki_engine import section_merge as section_merge_module
from llmwiki_engine import source_records as source_records_module
from llmwiki_engine import update_preservation as update_preservation_module
from llmwiki_engine.apply import ApplyError, apply_operation
from llmwiki_engine.hash_utils import sha256_file
from llmwiki_engine.io import read_json, read_jsonl, read_yaml, write_json, write_yaml
from llmwiki_engine.manifest import read_manifest
from llmwiki_engine.errors import PipelineError
from llmwiki_engine.models import (
    CandidateResolutionArtifact,
    CandidateResolutionItem,
    CandidateContextHit,
    CandidateContextItem,
    CandidateContextsArtifact,
    DraftGroundingReview,
    DraftRenderingArtifact,
    GroundingClaim,
    OperationStatus,
    RawPreparePolicy,
    SourceBasis,
    SourceDigestArtifact,
    SourceDigestCandidate,
    StepStatus,
    VerificationStatus,
    WeakOrNoiseItem,
    WikiKnowledgePoolEntry,
    WikiMergePlanArtifact,
)
from llmwiki_engine.pipeline import (
    STEP_RUNNERS,
    _STEP_RUN_FUNCTIONS,
    backfill_missing_candidate_resolution_items,
    build_wiki_context_snapshot,
    build_wiki_merge_plan,
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
from llmwiki_engine.validators import validate_wiki_merge_plan
from llmwiki_engine.verify import VerifyError, verify_run
from llmwiki_engine.workspace import RunStore


ROOT = Path(__file__).parent
FIXTURE_ROOT = ROOT / "fixtures" / "simple_project"
CURRENT_DRAFT_PAGE_FIELDS = {
    "page_plan_id",
    "action",
    "canonical_target_path",
    "preimage_sha256",
    "summary",
    "body_markdown",
    "open_questions",
    "change_summary",
    "source_coverage_notes",
    "quality_risks",
}


def draft_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(f"- {draft_text(item).strip()}" for item in value if draft_text(item).strip())
    return str(value)


def assert_draft_rendering_schema_page_fields(schema: dict) -> None:
    page_schema = schema["$defs"]["DraftPageItem"]
    assert set(page_schema["properties"]) == CURRENT_DRAFT_PAGE_FIELDS
    assert page_schema["additionalProperties"] is False


def draft_body(
    *,
    detail: object = "",
    examples: object = "",
    value_points: object = "",
    additional_notes: object = "",
) -> str:
    blocks = []
    detail_text = draft_text(detail).strip()
    if detail_text:
        blocks.append(detail_text)
    for title, value in [
        ("例子", examples),
        ("价值点", value_points),
        ("补充观察", additional_notes),
    ]:
        text = draft_text(value).strip()
        if text:
            blocks.append(f"### {title}\n\n{text}")
    return "\n\n".join(blocks)


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
    profile_yaml = vault / ".llmwiki" / "profiles" / "project_basic" / "profile.yaml"
    assert profile_yaml.exists()
    assert not (vault / ".llmwiki" / "profiles" / "project_basic" / "templates").exists()
    written_profile = read_yaml(profile_yaml)
    assert written_profile["version"] == "2"
    assert all("template" not in spec for spec in written_profile["page_types"].values())
    assert all("name" not in spec for spec in written_profile["page_types"].values())
    assert (vault / ".llmwiki" / "applied" / "operations.jsonl").exists()
    assert (vault / ".llmwiki" / "runs").exists()
    assert (vault / "wiki" / "index.md").exists()
    assert (vault / "wiki" / "log.md").exists()
    assert (vault / "wiki" / "logs").is_dir()
    config_json = read_json(vault / ".llmwiki" / "config.json")
    assert config_json["embedding_retrieval"]["backend"] == "sentence_transformers"
    assert config_json["embedding_retrieval"]["model"] == "Qwen/Qwen3-Embedding-0.6B"
    assert config_json["embedding_retrieval"]["local_files_only"] is True
    assert config_json["embedding_retrieval"]["cache_dir"] == "~/.llmwiki/cache/embeddings"
    assert config_json["max_ingest_candidates"] == 12
    assert "raw_prepare_policy" not in config_json
    gitignore_lines = (vault / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".llmwiki/" in gitignore_lines
    assert ".llmwiki/runs/" not in gitignore_lines
    assert not (vault / ".git").exists()


def test_embedding_cache_dir_expands_user_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", home.as_posix())
    resolved = retrieval_module.resolve_cache_dir(tmp_path / "vault", "~/.llmwiki/cache/embeddings")
    assert resolved == home / ".llmwiki" / "cache" / "embeddings"
    assert resolved.is_dir()


def test_huggingface_quiet_mode_suppresses_unauthenticated_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [
        "HF_HUB_DISABLE_PROGRESS_BARS",
        "HF_HUB_DISABLE_TELEMETRY",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN",
        "HF_HUB_DISABLE_SYMLINKS_WARNING",
    ]:
        monkeypatch.delenv(name, raising=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        retrieval_module.configure_huggingface_quiet_mode()
        warnings.warn("Warning: You are sending unauthenticated requests to the HF Hub.", UserWarning)

    assert not caught
    assert all(
        os.environ[name] == "1"
        for name in [
            "HF_HUB_DISABLE_PROGRESS_BARS",
            "HF_HUB_DISABLE_TELEMETRY",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN",
            "HF_HUB_DISABLE_SYMLINKS_WARNING",
        ]
    )
    assert logging.getLogger("huggingface_hub").level == logging.ERROR


def test_sentence_transformer_ranker_uses_local_cache_only_and_reports_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vault = tmp_path / "vault"
    page = vault / "wiki" / "concepts" / "Concept_Runtime.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        "---\n"
        "llmwiki_type: concept\n"
        "title: Runtime\n"
        "aliases: []\n"
        "summary: 运行时摘要。\n"
        "created: 2026-06-06\n"
        "updated: 2026-06-06\n"
        "---\n\n"
        "# Runtime\n\n"
        "Claude Code harness 负责工具执行和安全边界。\n",
        encoding="utf-8",
    )
    item = CandidateResolutionItem(
        page_plan_id="PP-RUNTIME",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        page_type="concept",
        display_title="Runtime",
        path_stem="Runtime",
        candidate_target_path="concepts/Concept_Runtime.md",
        topic_summary="Claude Code harness 运行时边界。",
        why_this_page="验证 embedding 本地缓存加载。",
        reason="test",
    )
    captured: dict[str, object] = {}
    encode_calls: list[int] = []

    class FakeSentenceTransformer:
        revision = "abcdef1"

        def __init__(self, model_name: str, **kwargs: object) -> None:
            captured["model_name"] = model_name
            captured.update(kwargs)

        def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
            encode_calls.append(len(texts))
            return [[1.0, 0.0] for _ in texts]

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    config = retrieval_module.EmbeddingRetrievalConfig(local_files_only=True)
    scores, revision, metrics = retrieval_module.SentenceTransformerRanker(config).rank_inputs(
        [item],
        retrieval_module.build_knowledge_pool(vault),
        vault,
    )
    _, _, cached_metrics = retrieval_module.SentenceTransformerRanker(config).rank_inputs(
        [item],
        retrieval_module.build_knowledge_pool(vault),
        vault,
    )

    assert captured["model_name"] == "Qwen/Qwen3-Embedding-0.6B"
    assert captured["local_files_only"] is True
    assert captured["token"] is False
    assert Path(str(captured["cache_folder"])).is_absolute()
    assert revision == "abcdef1"
    assert scores["PP-RUNTIME"]["concepts/Concept_Runtime.md"] == pytest.approx(1.0)
    assert metrics["total_duration_ms"] >= metrics["load_duration_ms"] >= 0
    assert metrics["total_duration_ms"] >= metrics["encode_duration_ms"] >= 0
    assert metrics["page_vector_cache_hit"] == 0
    assert cached_metrics["page_vector_cache_hit"] == 1
    assert encode_calls == [1, 1, 1]


def test_retrieval_candidate_text_includes_later_body_terms(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    page = vault / "wiki" / "concepts" / "Concept_Runtime.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        "---\n"
        "llmwiki_type: concept\n"
        "title: Runtime\n"
        "aliases: []\n"
        "summary: 运行时摘要。\n"
        "created: 2026-06-06\n"
        "updated: 2026-06-06\n"
        "---\n\n"
        "# Runtime\n\n"
        "开头段落没有关键召回词。\n\n"
        "## 后续架构线索\n\n"
        "这里记录 Claude Code harness 如何承担安全边界和工具执行。\n",
        encoding="utf-8",
    )
    entry = WikiKnowledgePoolEntry(
        path="concepts/Concept_Runtime.md",
        rel_path="wiki/concepts/Concept_Runtime.md",
        preimage_sha256="sha",
        display_title="Runtime",
        summary="运行时摘要。",
        llmwiki_type="concept",
    )

    text = retrieval_module.candidate_text(vault, entry)

    assert "Claude Code harness" in text
    assert retrieval_module.lexical_score("Claude Code harness 安全边界", text) > 0.5


def test_retrieval_lexical_expansion_boosts_agent_harness_terms(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    wiki = vault / "wiki"
    managed = wiki / "entities" / "Entity_Managed Agents.md"
    managed.parent.mkdir(parents=True)
    managed.write_text(
        "---\n"
        "llmwiki_type: entity\n"
        "title: Managed Agents\n"
        "aliases:\n"
        "  - 托管代理\n"
        "summary: Anthropic 的托管智能体运行时。\n"
        "created: 2026-06-06\n"
        "updated: 2026-06-06\n"
        "---\n\n"
        "# Managed Agents\n\n"
        "托管代理提供 agent runtime，用 harness 负责工具编排和安全边界。\n",
        encoding="utf-8",
    )
    generic = wiki / "concepts" / "Concept_Claude Code Updates.md"
    generic.parent.mkdir(parents=True)
    generic.write_text(
        "---\n"
        "llmwiki_type: concept\n"
        "title: Claude Code Updates\n"
        "aliases: []\n"
        "summary: Claude Code 产品更新记录。\n"
        "created: 2026-06-06\n"
        "updated: 2026-06-06\n"
        "---\n\n"
        "# Claude Code Updates\n\n"
        "Claude Code 的普通产品更新。\n",
        encoding="utf-8",
    )
    item = CandidateResolutionItem(
        page_plan_id="PP-HARNESS",
        source_basis=SourceBasis(source_candidate_ids=["design_todo"]),
        page_type="design",
        display_title="待办事项列表在 Claude Code 中的应用",
        path_stem="待办事项列表在 Claude Code 中的应用",
        candidate_target_path="designs/Design_待办事项列表在 Claude Code 中的应用.md",
        topic_summary="Claude Code 用 TODO list 推动大型重构。",
        why_this_page="体现如何用 Harness 弥补模型能力不足，是 agent runtime 设计模式。",
        reason="test",
    )

    hits = retrieval_module.rank_candidates(
        item=item,
        query=retrieval_module.query_for_item(item),
        knowledge_pool=retrieval_module.build_knowledge_pool(vault),
        vault=vault,
        config=retrieval_module.EmbeddingRetrievalConfig(backend="exact", top_k=5),
    )

    assert hits[0].path == "entities/Entity_Managed Agents.md"
    assert hits[0].match_basis == "lexical_expansion"
    assert hits[0].score < retrieval_module.EmbeddingRetrievalConfig().medium_score
    assert hits[0].strength == "weak"
    assert hits[0].score_bucket == retrieval_module.retrieval_score_bucket(hits[0].score)
    assert "basis_rank=" in hits[0].sort_explanation


def test_retrieval_lexical_expansion_does_not_expand_bare_product_name() -> None:
    score = retrieval_module.lexical_expansion_score(
        "Claude Code",
        "Managed Agents 托管代理提供 agent runtime 和 harness。",
    )

    assert score == 0.0


def test_retrieval_cjk_bigram_score_does_not_inflate_generic_overlap() -> None:
    query = "待办事项列表在 Claude Code 中的应用\n体现如何用 Harness 弥补模型能力不足，是 agent runtime 设计模式。"
    generic = "未来模型的上下文工程不可预测性。模型在长上下文中可能产生不可预测行为，需要评估。"
    relevant = "Managed Agents 托管代理提供 agent runtime，用 harness 负责工具编排和安全边界。"

    generic_score = retrieval_module.lexical_score(query, generic)
    relevant_expansion = retrieval_module.lexical_expansion_score(query, relevant)

    assert generic_score < retrieval_module.EmbeddingRetrievalConfig().medium_score
    assert relevant_expansion > generic_score


def test_candidate_context_sort_explains_score_bucket_tie_breaks() -> None:
    item = CandidateResolutionItem(
        page_plan_id="PP-TIE",
        source_basis=SourceBasis(source_candidate_ids=["CAND-TIE"]),
        page_type="concept",
        display_title="Runtime Tie",
        path_stem="Runtime Tie",
        candidate_target_path="concepts/Concept_Runtime_Tie.md",
        topic_summary="测试同一分数桶排序。",
        why_this_page="需要解释 score 不严格递减的 tie-break。",
        reason="test",
    )
    entries = {
        "concepts/Concept_Runtime_Tie.md": WikiKnowledgePoolEntry(
            path="concepts/Concept_Runtime_Tie.md",
            rel_path="wiki/concepts/Concept_Runtime_Tie.md",
            preimage_sha256="a",
            display_title="Runtime Tie",
            summary="same type and dir",
            llmwiki_type="concept",
        ),
        "entities/Entity_Runtime_Tie.md": WikiKnowledgePoolEntry(
            path="entities/Entity_Runtime_Tie.md",
            rel_path="wiki/entities/Entity_Runtime_Tie.md",
            preimage_sha256="b",
            display_title="Runtime Tie",
            summary="different type and dir",
            llmwiki_type="entity",
        ),
    }
    lower_score_same_type = retrieval_module.CandidateContextHit(
        page_plan_id="PP-TIE",
        rank=0,
        path="concepts/Concept_Runtime_Tie.md",
        display_title="Runtime Tie",
        score=0.501,
        score_bucket=retrieval_module.retrieval_score_bucket(0.501),
        strength="weak",
        match_basis="lexical",
        sort_explanation=retrieval_module.retrieval_sort_explanation_for_values(
            score=0.501,
            strength="weak",
            match_basis="lexical",
            path="concepts/Concept_Runtime_Tie.md",
            item=item,
            entry=entries["concepts/Concept_Runtime_Tie.md"],
        ),
        page_sha256="a",
    )
    higher_score_different_type = retrieval_module.CandidateContextHit(
        page_plan_id="PP-TIE",
        rank=0,
        path="entities/Entity_Runtime_Tie.md",
        display_title="Runtime Tie",
        score=0.509,
        score_bucket=retrieval_module.retrieval_score_bucket(0.509),
        strength="weak",
        match_basis="lexical",
        sort_explanation=retrieval_module.retrieval_sort_explanation_for_values(
            score=0.509,
            strength="weak",
            match_basis="lexical",
            path="entities/Entity_Runtime_Tie.md",
            item=item,
            entry=entries["entities/Entity_Runtime_Tie.md"],
        ),
        page_sha256="b",
    )

    sorted_hits = sorted(
        [higher_score_different_type, lower_score_same_type],
        key=lambda hit: retrieval_module.retrieval_sort_key(hit, item, entries),
    )
    assert sorted_hits[0].score < sorted_hits[1].score
    assert sorted_hits[0].path == "concepts/Concept_Runtime_Tie.md"
    assert "type=same" in sorted_hits[0].sort_explanation
    assert "dir=same" in sorted_hits[0].sort_explanation
    assert "type=different" in sorted_hits[1].sort_explanation


@pytest.mark.parametrize("sort_explanation", [None, ""])
def test_candidate_context_hit_requires_sort_explanation(sort_explanation: str | None) -> None:
    data = {
        "page_plan_id": "PP-TIE",
        "rank": 1,
        "path": "concepts/Concept_Runtime_Tie.md",
        "display_title": "Runtime Tie",
        "score": 0.5,
        "score_bucket": 50,
        "strength": "weak",
        "match_basis": "lexical",
        "page_sha256": "a",
    }
    if sort_explanation is not None:
        data["sort_explanation"] = sort_explanation

    with pytest.raises(Exception, match="sort_explanation"):
        retrieval_module.CandidateContextHit.model_validate(data)


def test_candidate_context_score_bucket_uses_persisted_rounded_score(tmp_path: Path) -> None:
    vault, _ = make_vault(tmp_path)
    page = vault / "wiki" / "concepts" / "Concept_Boundary.md"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(
        "---\n"
        "llmwiki_type: concept\n"
        "title: Boundary\n"
        "aliases: []\n"
        "summary: Boundary summary.\n"
        "created: 2026-06-08\n"
        "updated: 2026-06-08\n"
        "---\n\n"
        "# Boundary\n\n"
        "Boundary content.\n",
        encoding="utf-8",
    )
    item = CandidateResolutionItem(
        page_plan_id="PP-BOUNDARY",
        source_basis=SourceBasis(source_candidate_ids=["CAND-BOUNDARY"]),
        page_type="concept",
        display_title="No lexical overlap",
        path_stem="No lexical overlap",
        candidate_target_path="concepts/Concept_No_Lexical_Overlap.md",
        topic_summary="No lexical overlap.",
        why_this_page="Forces embedding score to determine bucket.",
        reason="test",
    )
    entry = retrieval_module.build_knowledge_pool(vault)[0]

    hits = retrieval_module.rank_candidates(
        item=item,
        query="unmatched query",
        knowledge_pool=[entry],
        vault=vault,
        config=retrieval_module.EmbeddingRetrievalConfig(backend="sentence_transformers"),
        embedding_scores={entry.path: 0.579999},
    )

    assert hits[0].score == 0.58
    assert hits[0].score_bucket == retrieval_module.retrieval_score_bucket(hits[0].score)
    assert f"bucket={hits[0].score_bucket}" in hits[0].sort_explanation


def test_retrieval_metadata_uses_shared_frontmatter_list_parser() -> None:
    metadata = retrieval_module.metadata_from_text(
        "---\n"
        "llmwiki_type: concept\n"
        "title: Scalar Source\n"
        "summary: Scalar source summary.\n"
        "updated: 2026-06-09\n"
        "source_raw_paths: raw/scalar.md\n"
        "source_operation_ids: OP-SCALAR\n"
        "---\n\n"
        "# Scalar Source\n",
        "wiki/concepts/Concept_Scalar_Source.md",
    )

    assert metadata is not None
    assert metadata.source_raw_paths == ["raw/scalar.md"]
    assert metadata.source_operation_ids == ["OP-SCALAR"]
    assert not hasattr(retrieval_module, "frontmatter_list")
    assert not hasattr(retrieval_module, "parse_frontmatter")
    assert not hasattr(source_records_module, "frontmatter_list")
    assert not hasattr(source_records_module, "parse_frontmatter")
    moved_pipeline_exports = {
        "parse_frontmatter", "DRAFT_RENDERING_GROUNDING_RISK_RULES", "build_draft_grounding_review",
        "quote_supported_by_text", "render_draft_grounding_review", "augment_source_digest_anchor_entities",
        "cap_source_digest_candidates", "render_source_digest_budget_report", "build_source_digest_source_map",
        "project_source_digest_source_map_for_payload", "build_source_kind_hints", "source_digest_language_contract",
        "render_source_digest_source_map_markdown", "render_source_kind_hints_markdown", "build_source_digest_payload",
        "SOURCE_DIGEST_FULL_SOURCE_CHAR_LIMIT", "PAPER_CAPTION_RE", "render_source_digest_markdown", "render_candidate_table",
        "build_candidate_resolution_source_pack", "render_candidate_resolution_source_pack_markdown",
        "build_merge_planning_context_pack", "build_merge_planning_source_pack", "compact_candidate_contexts_for_merge_planning",
        "merge_planning_hit_excerpt_limit", "merge_planning_relevant_wiki_paths", "compact_snapshot_for_merge_planning",
        "merge_planning_payload_pack_summary", "render_merge_planning_context_pack_markdown", "json_char_count",
        "CANDIDATE_RESOLUTION_FULL_SOURCE_CHAR_LIMIT", "CANDIDATE_RESOLUTION_GLOBAL_EXCERPT_LIMIT",
        "CANDIDATE_RESOLUTION_PER_CANDIDATE_EXCERPT_LIMIT", "MERGE_PLANNING_FULL_SOURCE_CHAR_LIMIT",
        "MERGE_PLANNING_SOURCE_GLOBAL_EXCERPT_LIMIT", "MERGE_PLANNING_SOURCE_PER_PAGE_EXCERPT_LIMIT",
        "MERGE_PLANNING_CONTEXT_HIT_EXCERPT_LIMIT", "MERGE_PLANNING_WEAK_CONTEXT_HIT_EXCERPT_LIMIT",
        "MERGE_PLANNING_WEAK_CONTEXT_EXCERPT_MAX_RANK", "MERGE_PLANNING_CONTEXT_QUERY_LIMIT",
        "MERGE_PLANNING_ENTRY_EXCERPT_LIMIT", "source_basis_candidate_refs", "source_digest_candidate_lookup",
        "source_digest_candidate_id_closure", "first_source_basis_candidate", "build_draft_source_excerpt_pack",
        "render_draft_source_excerpt_pack_markdown", "build_draft_rendering_payload", "project_merge_plan_for_draft_rendering",
        "project_merge_plan_item_for_draft_rendering", "draft_rendering_relevant_wiki_paths",
        "should_include_draft_inspected_context", "normalize_wiki_snapshot_path", "compact_snapshot_for_draft_rendering",
        "compact_optional_dict", "project_source_digest_for_merge_plan", "source_digest_candidate_ids_for_merge_plan",
        "DRAFT_RENDERING_FULL_SOURCE_CHAR_LIMIT", "DRAFT_RENDERING_EXCERPT_TOTAL_CHAR_LIMIT",
        "DRAFT_RENDERING_EXCERPT_PER_PAGE_LIMIT", "DRAFT_RENDERING_GLOBAL_EXCERPT_LIMIT",
        "DRAFT_RENDERING_EXCERPT_MAX_SOURCE_RATIO", "DRAFT_RENDERING_EXCERPT_MIN_PAGE_CHARS",
        "DRAFT_RENDERING_CONTEXT_ENTRY_EXCERPT_LIMIT", "render_merge_planning_shortcut_report",
        "render_merge_plan_markdown", "render_merge_plan_review_prompt", "render_candidate_contexts_markdown",
        "render_merge_decision_report", "merge_plan_create_overlap_risk_items", "FINAL_RELATED_LIMIT",
        "render_related_pages", "assemble_knowledge_page", "render_source_page", "build_index_rows",
        "render_update_merge_report", "render_related_merge_report", "render_update_diff", "build_draft_approval",
        "render_draft_review_prompt", "draft_review_reason", "draft_review_requires_manual",
        "update_manual_resolution_count", "update_reinforcement_count", "update_reinforcement_report_ref",
        "draft_diff_ref", "draft_change_summary", "build_apply_preview", "merge_plan_all_create_review_reason",
        "MAX_AUTO_APPROVED_ALL_CREATE_ITEMS", "LOCAL_MEDIUM_CREATE_REASON_MARKER", "merge_same_source_duplicate_creates",
        "merge_update_noop_same_targets", "normalize_model_wiki_target_path", "synthesize_medium_create_why_not_update",
        "medium_create_generic_old_title_review_reason",
    }
    leaked = sorted(name for name in moved_pipeline_exports if hasattr(pipeline_module, name))
    assert leaked == []


def test_source_digest_candidate_budget_defers_overflow_by_group() -> None:
    def candidate(candidate_id: str, name: str) -> SourceDigestCandidate:
        return SourceDigestCandidate(
            candidate_id=candidate_id,
            name=name,
            type="concept",
            one_sentence_summary=f"{name} 摘要。",
            why_matters=f"{name} 重要。",
            wiki_value=f"{name} 可复用。",
            suggested_page_title=name,
        )

    digest = SourceDigestArtifact(
        source_raw_path="raw/sample.md",
        summary="测试预算。",
        concepts=[candidate("C1", "概念一"), candidate("C2", "概念二")],
        designs=[candidate("D1", "设计一"), candidate("D2", "设计二")],
        comparisons=[candidate("CMP1", "对比一")],
        open_questions=[candidate("O1", "问题一")],
        entities=[candidate("E1", "实体一")],
    )

    capped, report = source_digest_budget.cap_source_digest_candidates(digest, 4)

    assert [item.candidate_id for item in capped.concepts] == ["C1"]
    assert [item.candidate_id for item in capped.designs] == ["D1"]
    assert [item.candidate_id for item in capped.comparisons] == ["CMP1"]
    assert [item.candidate_id for item in capped.open_questions] == ["O1"]
    assert capped.entities == []
    assert [item.candidate_id for item in capped.budget_deferred_candidates] == ["E1", "C2", "D2"]
    assert capped.weak_or_noise_items == []
    assert all("page_budget_deferred" in item.resolution_hint for item in capped.budget_deferred_candidates)
    assert report["selected_count"] == 4
    assert report["deferred_count"] == 3
    assert report["deferred"]["entities"] == ["E1"]
    assert report["deferred_details"]["entities"][0]["candidate_id"] == "E1"
    assert report["deferred_details"]["entities"][0]["wiki_value"] == "实体一 可复用。"
    assert report["followup_batches"][0]["group"] == "entities"
    assert "overview/comparison" in report["followup_batches"][0]["suggested_action"]
    assert report["deferred_aggregations"][0]["group"] == "concepts"
    assert report["deferred_aggregations"][0]["suggested_page_type"] == "concept_overview"
    assert "概念二" in report["deferred_aggregations"][0]["suggested_title"]
    assert report["deferred_aggregations"][-1]["group"] == "entities"
    markdown = source_digest_budget.render_source_digest_budget_report(report)
    assert "## 延后候选详情" in markdown
    assert "实体一 可复用。" in markdown
    assert "## 后续处理批次" in markdown
    assert "## 延后聚合建议" in markdown
    assert "concept_overview" in markdown


def test_source_digest_candidate_budget_promotes_deferred_aggregation_without_increasing_budget(tmp_path: Path) -> None:
    def concept(candidate_id: str, name: str, summary: str) -> SourceDigestCandidate:
        return SourceDigestCandidate(
            candidate_id=candidate_id,
            name=name,
            type="concept",
            one_sentence_summary=summary,
            why_matters=f"{summary} 重要。",
            wiki_value=f"{summary} 可与其他同组概念先聚合成总览。",
            source_locator=f"section {candidate_id}",
            suggested_page_title=name,
        )

    digest = SourceDigestArtifact(
        source_raw_path="raw/sample.md",
        summary="测试延后候选聚合。",
        concepts=[
            concept("C1", "产品品味校准", "团队用用户反馈和设计 critique 校准产品判断。"),
            concept("C2", "AGI PM 协作边界", "AGI 后 PM 更关注问题判断、评估设计和组织协作。"),
            concept("C3", "AGI PM 协作职责", "AGI 后 PM 从写需求转向判断问题、组织评估和协调智能体执行。"),
            concept("C4", "AGI PM 协作评估", "AGI 产品需要用任务成功率、可控性和反馈循环评估。"),
        ],
    )

    capped, report = source_digest_budget.cap_source_digest_candidates(digest, 2)

    assert len(capped.concepts) == 2
    assert capped.concepts[0].candidate_id == "C1"
    aggregate = capped.concepts[1]
    assert aggregate.candidate_id.startswith("AGG-concepts-")
    assert aggregate.type == "overview"
    assert aggregate.related_candidates == ["C3", "C4"]
    assert "source_digest_deferred_aggregation" in aggregate.resolution_hint
    assert [item.candidate_id for item in capped.budget_deferred_candidates] == ["C3", "C4"]
    assert all("represented_by_aggregation" in item.resolution_hint for item in capped.budget_deferred_candidates)
    selected_aggregation = report["selected_deferred_aggregations"][0]
    assert selected_aggregation["candidate_id"] == aggregate.candidate_id
    assert selected_aggregation["represented_candidate_ids"] == ["C2", "C3", "C4"]
    assert selected_aggregation["replaced_candidate_id"] == "C2"
    projected = draft_rendering_payloads_module.project_source_digest_for_merge_plan(
        capped,
        WikiMergePlanArtifact(
            log_date="2026-06-06",
            items=[
                pipeline_module.WikiMergePlanItem(
                    page_plan_id="PP-AGG",
                    source_basis=SourceBasis(source_candidate_ids=[aggregate.candidate_id]),
                    action="create",
                    canonical_target_path="overviews/Overview_AGG.md",
                    display_title=aggregate.suggested_page_title,
                    page_type="overview",
                    new_understanding="测试。",
                    section_plans={"summary": "摘要"},
                    reason="test",
                )
            ],
        ),
    )
    projected_lookup = source_refs_module.source_digest_candidate_lookup(projected)
    assert all(candidate_id in projected_lookup for candidate_id in aggregate.related_candidates)
    assert report["selected_count"] == 2
    assert report["deferred_count"] == 2
    markdown = source_digest_budget.render_source_digest_budget_report(report)
    assert "## 本轮已选聚合候选" in markdown
    assert aggregate.candidate_id in markdown

    vault, _ = make_vault(tmp_path)
    profile = pipeline_module.load_profile(vault / ".llmwiki" / "profiles" / "project_basic")
    resolution = pipeline_module.backfill_missing_candidate_resolution_items(
        CandidateResolutionArtifact(items=[]),
        capped,
        profile,
    )
    finalized = pipeline_module.finalize_candidate_resolution(vault, profile, resolution, capped)
    aggregate_plan = [item for item in finalized.items if item.source_basis.source_candidate_ids == [aggregate.candidate_id]][0]
    assert aggregate_plan.page_type == "overview"
    assert aggregate_plan.candidate_target_path.startswith("overviews/Overview_")


def test_source_digest_anchor_entities_are_added_before_page_budget() -> None:
    def candidate(candidate_id: str, title: str, group_type: str) -> SourceDigestCandidate:
        return SourceDigestCandidate(
            candidate_id=candidate_id,
            name=title,
            type=group_type,
            one_sentence_summary=f"{title} 摘要。",
            why_matters=f"{title} 重要。",
            wiki_value=f"{title} 可复用。",
            suggested_page_title=title,
        )

    text = (
        "---\n"
        'title: "Scaling Managed Agents: 将大脑与双手解耦"\n'
        'description: "介绍 Managed Agents 的架构设计。"\n'
        "---\n\n"
        "# Scaling Managed Agents: Decoupling the brain from the hands\n\n"
        "Managed Agents is a meta-harness around Claude.\n\n"
        "For example, Claude Code is an excellent harness that we use widely across tasks.\n"
        "> 例如，**Claude Code** 是一个出色的 harness，我们在各种任务中广泛使用它。\n"
    )
    digest = SourceDigestArtifact(
        source_raw_path="raw/scaling.md",
        summary="Managed Agents 架构摘要。",
        entities=[
            candidate("ent-001", "Harness（适配框架）", "entity"),
            candidate("ent-002", "Session（会话）", "entity"),
        ],
        concepts=[
            candidate("con-001", "大脑与双手解耦", "concept"),
            candidate("con-002", "会话作为持久化上下文", "concept"),
            candidate("con-003", "安全令牌隔离", "concept"),
            candidate("con-004", "元适配框架", "concept"),
        ],
        designs=[
            candidate("des-001", "Managed Agents 架构设计", "design"),
            candidate("des-002", "大脑与双手交互接口设计", "design"),
        ],
        comparisons=[candidate("cmp-001", "耦合架构 vs 解耦架构", "comparison")],
        open_questions=[
            candidate("oq-001", "长期任务上下文管理", "open_question"),
            candidate("oq-002", "窄范围作用域与智能增长", "open_question"),
        ],
    )

    augmented = source_digest_budget.augment_source_digest_anchor_entities(digest, text)
    assert [item.suggested_page_title for item in augmented.entities[:2]] == ["Managed Agents", "Claude Code"]
    assert augmented.entities[0].candidate_id == "auto-ent-managedagents"
    assert augmented.entities[1].source_locator.startswith("L")
    assert "generic_source_anchor_entity" in augmented.entities[1].resolution_hint
    assert augmented.entities[0].related_candidates == []
    assert augmented.entities[1].related_candidates == []
    assert not hasattr(source_digest_budget, "SOURCE_DIGEST_ANCHOR_ENTITIES")

    capped, report = source_digest_budget.cap_source_digest_candidates(augmented, 12)

    selected_titles = [item.suggested_page_title for item in capped.entities]
    assert "Managed Agents" in selected_titles
    assert "Claude Code" in selected_titles
    deferred_titles = [item.suggested_page_title for item in capped.budget_deferred_candidates]
    assert "Managed Agents" not in deferred_titles
    assert "Claude Code" not in deferred_titles
    assert "Session（会话）" in deferred_titles
    assert report["total_formal_candidates_before_budget"] == 13
    assert report["applied"] is True


def test_source_digest_anchor_entities_do_not_duplicate_parenthetical_translation() -> None:
    digest = SourceDigestArtifact(
        source_raw_path="raw/scaling.md",
        summary="Managed Agents 架构摘要。",
        entities=[
            SourceDigestCandidate(
                candidate_id="c001",
                name="Managed Agents",
                type="entity",
                one_sentence_summary="Anthropic 的托管智能体平台，采用大脑与双手解耦的模块化架构。",
                why_matters="它是材料中的核心实体。",
                wiki_value="适合沉淀为实体页。",
                suggested_page_title="Managed Agents（托管智能体）",
            )
        ],
    )
    text = (
        "---\n"
        'title: "Scaling Managed Agents: 将大脑与双手解耦"\n'
        "---\n\n"
        "# Scaling Managed Agents\n\n"
        "Managed Agents is a meta-harness around Claude.\n"
        "Managed Agents can host Claude Code as an excellent harness.\n"
    )

    augmented = source_digest_budget.augment_source_digest_anchor_entities(digest, text)
    capped, report = source_digest_budget.cap_source_digest_candidates(augmented, 12)

    assert [item.candidate_id for item in augmented.entities if item.suggested_page_title.startswith("Managed Agents")] == ["c001"]
    assert [item.suggested_page_title for item in capped.entities] == ["Claude Code", "Managed Agents（托管智能体）"]
    assert report["deduped_count"] == 0


def test_source_digest_anchor_entities_do_not_merge_across_sentence_periods() -> None:
    digest = SourceDigestArtifact(source_raw_path="raw/scaling.md", summary="实体摘要。")
    text = (
        "Managed Agents is a meta-harness around Claude. "
        "Claude Code is an excellent harness for coding work. "
        "OpenAI. Anthropic is a company building AI systems."
    )

    augmented = source_digest_budget.augment_source_digest_anchor_entities(digest, text)

    assert [item.suggested_page_title for item in augmented.entities] == ["Managed Agents", "Claude Code"]
    assert all("." not in item.suggested_page_title for item in augmented.entities)


def test_source_digest_anchor_entities_ignore_weak_single_mentions() -> None:
    digest = SourceDigestArtifact(source_raw_path="raw/sample.md", summary="普通摘要。")
    text = "这篇材料只是随口提了一次 Claude Code，没有说明产品、harness 或团队上下文。"

    augmented = source_digest_budget.augment_source_digest_anchor_entities(digest, text)

    assert augmented.entities == []


def test_source_digest_anchor_entities_ignore_readme_section_noise() -> None:
    digest = SourceDigestArtifact(source_raw_path="raw/readme.md", summary="README 摘要。")
    text = (
        "# Quick Start\n\n"
        "Quick Start is easy.\n\n"
        "## Installation\n\n"
        "Installation provides setup commands.\n\n"
        "## Examples\n\n"
        "Examples show common usage.\n\n"
        "## Documentation\n\n"
        "Documentation links to API references.\n"
    )

    augmented = source_digest_budget.augment_source_digest_anchor_entities(digest, text)

    assert augmented.entities == []


def test_source_digest_anchor_entities_ignore_contextless_document_titles() -> None:
    digest = SourceDigestArtifact(source_raw_path="raw/readme.md", summary="README 摘要。")
    text = (
        "# Hello-Agents\n\n"
        "![GitHub stars](https://img.shields.io/github/stars/datawhalechina/Hello-Agents)\n"
        "[GitHub Project](https://github.com/datawhalechina/Hello-Agents)\n\n"
        "# Long Research Note\n\n"
        "This note studies agent memory evaluation and durable wiki candidates.\n"
    )

    augmented = source_digest_budget.augment_source_digest_anchor_entities(digest, text)

    assert augmented.entities == []


def test_source_digest_anchor_entities_ignore_metadata_and_repeated_mentions_without_definition() -> None:
    digest = SourceDigestArtifact(source_raw_path="raw/repeated.md", summary="重复摘要。")
    text = (
        "---\n"
        'title: "Qwen-Agent Qwen-Agent Qwen-Agent"\n'
        'description: "Qwen-Agent appears several times in metadata."\n'
        "---\n\n"
        "# Qwen-Agent\n\n"
        "Qwen-Agent appears in this note. Qwen-Agent appears again. Qwen-Agent appears a third time.\n"
        "The note lists names but does not define what the project is or how it behaves.\n"
    )

    augmented = source_digest_budget.augment_source_digest_anchor_entities(digest, text)

    assert augmented.entities == []


def test_source_digest_anchor_entities_ignore_positive_marker_noise() -> None:
    digest = SourceDigestArtifact(source_raw_path="raw/noise.md", summary="噪声摘要。")
    for text in [
        "# How AI Agents Can Automate Workflows\n\n"
        "The article discusses workflow automation patterns.",
        "# Hello-Agents\n\n"
        "[GitHub Project](https://github.com/datawhalechina/Hello-Agents) provides examples and docs.",
        "# FooBar\n\n"
        "[FooBar Project](https://example.test/foo) provides examples and docs.",
        "# FooBar\n\n"
        "[FooBar Docs](https://example.test/foo/docs) provides documentation.",
        "# Paper Index\n\n"
        "[PDF Download](https://example.test/paper.pdf) provides the full paper.",
        "# Long Research Note\n\n"
        "Long Research Note is a collection of reading notes, not a product or organization.",
        "# Product Development\n\n"
        "Product Development is hard.",
        "# Creating Documents\n\n"
        "Creating Documents is a workflow.",
        "# Release Planning\n\n"
        "Release Planning supports teams.",
    ]:
        augmented = source_digest_budget.augment_source_digest_anchor_entities(digest, text)
        assert augmented.entities == []


def test_source_digest_candidate_budget_promotes_multiple_topic_aggregations_without_cross_cluster() -> None:
    def concept(candidate_id: str, title: str, summary: str) -> SourceDigestCandidate:
        return SourceDigestCandidate(
            candidate_id=candidate_id,
            name=title,
            type="concept",
            one_sentence_summary=summary,
            why_matters=f"{summary} 重要。",
            wiki_value=f"{summary} 适合沉淀为可复用知识。",
            source_locator=f"section {candidate_id}",
            suggested_page_title=title,
        )

    digest = SourceDigestArtifact(
        source_raw_path="raw/sample.md",
        summary="测试延后候选按主题簇聚合。",
        concepts=[
            concept("C1", "产品品味校准", "团队用用户反馈和设计 critique 校准产品判断。"),
            concept("C2", "客户场景研究", "团队从真实客户任务中提炼场景、约束和优先级。"),
            concept("C3", "模型能力产品边界", "模型能力提升后，产品边界转向工作流、信任和可控性。"),
            concept("C4", "AGI PM 协作边界", "AGI 后 PM 更关注问题判断、评估设计和组织协作。"),
            concept("C5", "AGI PM 协作职责", "AGI 让产品组织重新分配发现问题、定义方案和交付验证的职责。"),
            concept("C6", "AGI PM 协作评估", "AGI 后 PM 从写需求转向判断问题、组织评估和协调智能体执行。"),
            concept("C7", "模型能力吞噬产品功能", "模型能力提升会把一部分产品功能变成提示词、评估和数据飞轮问题。"),
            concept("C8", "模型替代产品功能后的边界", "模型直接完成任务后，产品功能边界转向工作流、信任和可控性。"),
        ],
    )

    capped, report = source_digest_budget.cap_source_digest_candidates(digest, 4)

    assert report["selected_count"] == 4
    assert report["deferred_count"] == 4
    assert [item.candidate_id for item in capped.concepts[:2]] == ["C1", "C2"]
    aggregates = capped.concepts[2:]
    assert len(aggregates) == 2
    assert all(item.candidate_id.startswith("AGG-concepts-") for item in aggregates)
    selected_aggregations = report["selected_deferred_aggregations"]
    assert len(selected_aggregations) == 2
    represented_sets = {
        tuple(aggregation["represented_candidate_ids"])
        for aggregation in selected_aggregations
    }
    assert represented_sets == {
        ("C4", "C5", "C6"),
        ("C3", "C7", "C8"),
    }
    assert {aggregation["replaced_candidate_id"] for aggregation in selected_aggregations} == {"C3", "C4"}
    assert all(aggregation["cluster_terms"] for aggregation in selected_aggregations)
    deferred_by_id = {item.candidate_id: item for item in capped.budget_deferred_candidates}
    assert set(deferred_by_id) == {"C5", "C6", "C7", "C8"}
    assert all("represented_by_aggregation" in item.resolution_hint for item in deferred_by_id.values())
    assert len({item.resolution_hint.rsplit("`", 2)[1] for item in deferred_by_id.values()}) == 2


def test_source_digest_deferred_aggregation_does_not_replace_unrelated_selected_core_page() -> None:
    def concept(candidate_id: str, title: str, summary: str, tension: str = "") -> SourceDigestCandidate:
        return SourceDigestCandidate(
            candidate_id=candidate_id,
            name=title,
            type="concept",
            one_sentence_summary=summary,
            why_matters=f"{summary} 是重要 AI PM 知识。",
            wiki_value=f"{title} 适合沉淀为可复用知识页。",
            source_locator=f"section {candidate_id}",
            suggested_page_title=title,
            open_question_or_tension=tension,
        )

    digest = SourceDigestArtifact(
        source_raw_path="raw/aipm.md",
        summary="测试不相关 selected 核心页不被 deferred 聚合替换。",
        concepts=[
            concept("CON-001", "AI PM 角色分类", "AI PM 分为带 AI 功能的传统 PM 和 AI 原生 PM。"),
            concept("CON-002", "概率性产品体验", "AI 产品输出是概率性的，会改变信任、错误容忍度和产品设计。"),
            concept("CON-003", "AI 技术选择框架", "AI PM 需要在传统 ML、深度学习和 GenAI 之间做技术选择。"),
            concept("CON-004", "RAG (检索增强生成)", "RAG 通过检索外部知识库为 LLM 提供上下文。", "RAG 与 fine-tuning 的适用边界是什么？"),
            concept("CON-005", "AI Agent 架构", "Agent 是能自主感知环境、决策并采取行动的 AI 系统。", "Agent 的可靠性如何保证？"),
            concept(
                "CON-006",
                "提示词工程与上下文工程",
                "提示词工程设计输入，上下文工程选择和组织模型需要的信息。",
                "上下文工程如何与 RAG 结合？",
            ),
        ],
    )

    capped, report = source_digest_budget.cap_source_digest_candidates(digest, 3)

    assert [item.candidate_id for item in capped.concepts] == ["CON-001", "CON-002", "CON-003"]
    assert report["selected_deferred_aggregations"] == []
    assert [item.candidate_id for item in capped.budget_deferred_candidates] == ["CON-004", "CON-005", "CON-006"]
    assert report["deferred_aggregations"][0]["candidate_ids"] == ["CON-004", "CON-005", "CON-006"]


def test_deferred_topic_clusters_do_not_merge_on_single_broad_anchor() -> None:
    def concept(candidate_id: str, title: str, summary: str) -> SourceDigestCandidate:
        return SourceDigestCandidate(
            candidate_id=candidate_id,
            name=title,
            type="concept",
            one_sentence_summary=summary,
            why_matters=f"{summary} 重要。",
            wiki_value=f"{summary} 可复用。",
            suggested_page_title=title,
        )

    agi_org = concept("C1", "AGI 产品组织变化", "AGI 会改变产品组织的职责分配。")
    agi_investment = concept("C2", "AGI 投资节奏", "AGI 会改变基础设施投入和资本节奏。")
    model_eval = concept("C3", "模型评估方法", "模型评估需要离线指标和人工检查。")
    model_boundary = concept("C4", "模型产品边界", "模型进入产品后会改变功能边界。")

    assert source_digest_budget.source_digest_candidate_topic_similarity("concepts", agi_org, agi_investment) == 0.0
    assert source_digest_budget.source_digest_candidate_topic_similarity("concepts", model_eval, model_boundary) == 0.0
    assert source_digest_budget.deferred_candidate_topic_clusters("concepts", [agi_org, agi_investment, model_eval, model_boundary]) == []


def test_draft_source_excerpt_pack_expands_aggregation_child_candidate_cues() -> None:
    agg = SourceDigestCandidate(
        candidate_id="AGG-concepts-demo",
        name="聚合候选",
        type="overview",
        one_sentence_summary="聚合候选摘要。",
        why_matters="聚合候选重要。",
        wiki_value="聚合候选可复用。",
        suggested_page_title="聚合候选",
        related_candidates=["C3", "C4"],
    )
    child = SourceDigestCandidate(
        candidate_id="C3",
        name="子候选 Alpha",
        type="concept",
        one_sentence_summary="子候选 Alpha 解释 evaluator harness 的可靠性。",
        why_matters="子候选 Alpha 很重要。",
        wiki_value="子候选 Alpha 可复用。",
        source_locator="## 子候选 Alpha",
        suggested_page_title="子候选 Alpha",
    )
    other_child = SourceDigestCandidate(
        candidate_id="C4",
        name="子候选 Beta",
        type="concept",
        one_sentence_summary="子候选 Beta 解释运行时边界。",
        why_matters="子候选 Beta 很重要。",
        wiki_value="子候选 Beta 可复用。",
        source_locator="## 子候选 Beta",
        suggested_page_title="子候选 Beta",
    )
    digest = SourceDigestArtifact(
        source_raw_path="raw/sample.md",
        summary="测试 source excerpt closure。",
        concepts=[agg],
        budget_deferred_candidates=[child, other_child],
    )
    merge_plan = WikiMergePlanArtifact(
        log_date="2026-06-06",
        items=[
            pipeline_module.WikiMergePlanItem(
                page_plan_id="PP-AGG",
                source_basis=SourceBasis(source_candidate_ids=["AGG-concepts-demo"], source_locator="聚合"),
                action="create",
                canonical_target_path="overviews/Overview_AGG.md",
                display_title="聚合候选",
                page_type="overview",
                new_understanding="测试。",
                section_plans={"summary": "摘要"},
                reason="test",
            )
        ],
    )
    text = (
        "# 测试材料\n\n"
        "开头内容。\n\n"
        "## 子候选 Alpha\n\n"
        "这里记录 evaluator harness 的可靠性和聚合页必须吸收的子候选 Alpha 细节。\n\n"
        "## 子候选 Beta\n\n"
        "这里记录运行时边界和子候选 Beta 细节。\n"
    )

    pack = draft_rendering_payloads_module.build_draft_source_excerpt_pack(text, digest, merge_plan, full_source_limit=10)

    item = pack["items"][0]
    assert item["expanded_source_candidate_ids"] == ["AGG-concepts-demo", "C3", "C4"]
    assert {"## 子候选 Alpha", "## 子候选 Beta"} <= set(item["source_locators"])
    assert "evaluator harness" in "\n".join(snippet["text"] for snippet in item["snippets"])


def test_source_excerpt_packs_expand_prepared_discovered_budget_deferred_cues() -> None:
    deferred = SourceDigestCandidate(
        candidate_id="C005",
        name="多脑多手架构",
        type="concept",
        one_sentence_summary="多脑多手架构把大脑与双手解耦。",
        why_matters="它帮助理解 agent harness 的组织方式。",
        wiki_value="可用于解释 managed agents。",
        source_locator="## 多脑多手架构",
        suggested_page_title="多脑多手架构",
    )
    digest = SourceDigestArtifact(
        source_raw_path="raw/sample.md",
        summary="测试 prepared discovered source pack。",
        budget_deferred_candidates=[deferred],
    )
    resolution = CandidateResolutionArtifact(
        items=[
            CandidateResolutionItem(
                page_plan_id="PP-C005",
                source_basis=SourceBasis(prepared_discovered_candidates=["C005"], source_locator="prepared discovered"),
                page_type="concept",
                display_title="多脑多手架构",
                candidate_target_path="concepts/Concept_多脑多手架构.md",
                topic_summary="多脑多手架构摘要。",
                why_this_page="值得记录。",
                reason="prepared_discovered",
            )
        ]
    )
    merge_plan = WikiMergePlanArtifact(
        log_date="2026-06-06",
        items=[
            pipeline_module.WikiMergePlanItem(
                page_plan_id="PP-C005",
                source_basis=SourceBasis(prepared_discovered_candidates=["C005"], source_locator="prepared discovered"),
                action="create",
                canonical_target_path="concepts/Concept_多脑多手架构.md",
                display_title="多脑多手架构",
                page_type="concept",
                new_understanding="多脑多手架构摘要。",
                section_plans={"summary": "摘要"},
                reason="prepared_discovered",
            )
        ],
    )
    text = (
        "# 测试材料\n\n"
        + ("背景填充段落。\n" * 80)
        + "\n## 多脑多手架构\n\n"
        "这里描述多脑多手架构如何把大脑与双手解耦，并通过 agent harness 组织 managed agents。\n"
    )

    merge_pack = planning_payloads_module.build_merge_planning_context_pack(
        approved_prepared_text=text + ("\n额外填充段落。\n" * 2000),
        digest=digest,
        resolution=resolution,
        snapshot=pipeline_module.WikiContextSnapshot(log_date="2026-06-06", source_target_path="sources/Source_Test.md"),
        candidate_contexts=pipeline_module.CandidateContextsArtifact(retrieval_backend="exact"),
        snapshot_ref="wiki_context_snapshot/wiki_context_snapshot.json",
    )
    draft_pack = draft_rendering_payloads_module.build_draft_source_excerpt_pack(text, digest, merge_plan, full_source_limit=10)
    merge_item = merge_pack["source_excerpt_pack"]["items"][0]
    draft_item = draft_pack["items"][0]

    assert merge_item["source_candidate_ids"] == []
    assert merge_item["prepared_discovered_candidates"] == ["C005"]
    assert merge_item["source_candidate_refs"] == ["C005"]
    assert "## 多脑多手架构" in merge_item["source_locators"]
    assert "agent harness" in "\n".join(snippet["text"] for snippet in merge_item["snippets"])
    assert draft_item["source_candidate_ids"] == []
    assert draft_item["prepared_discovered_candidates"] == ["C005"]
    assert draft_item["source_candidate_refs"] == ["C005"]
    assert "## 多脑多手架构" in draft_item["source_locators"]
    assert "managed agents" in "\n".join(snippet["text"] for snippet in draft_item["snippets"])


def test_draft_rendering_digest_projection_keeps_batch_candidates_and_related_deferred() -> None:
    def concept(candidate_id: str, title: str, related: list[str] | None = None) -> SourceDigestCandidate:
        return SourceDigestCandidate(
            candidate_id=candidate_id,
            name=title,
            type="concept",
            one_sentence_summary=f"{title} 摘要。",
            why_matters=f"{title} 重要。",
            wiki_value=f"{title} 可复用。",
            suggested_page_title=title,
            related_candidates=related or [],
        )

    digest = SourceDigestArtifact(
        source_raw_path="raw/sample.md",
        summary="测试 draft rendering digest projection。",
        concepts=[
            concept("C1", "保留的普通候选"),
            concept("AGG-concepts-demo", "聚合候选", related=["C3", "C4"]),
            concept("C9", "无关候选"),
        ],
        designs=[concept("D1", "无关设计")],
        budget_deferred_candidates=[
            concept("C3", "聚合代表的 deferred 一"),
            concept("C4", "聚合代表的 deferred 二"),
            concept("C10", "无关 deferred"),
        ],
        weak_or_noise_items=[
            WeakOrNoiseItem(
                candidate_id="noise-1",
                name="噪声",
                one_sentence_summary="不应进入 draft payload。",
            )
        ],
    )
    merge_plan = WikiMergePlanArtifact(
        log_date="2026-06-06",
        items=[
            pipeline_module.WikiMergePlanItem(
                page_plan_id="PP-C1",
                source_basis=SourceBasis(source_candidate_ids=["C1"]),
                action="create",
                canonical_target_path="concepts/Concept_C1.md",
                display_title="保留的普通候选",
                page_type="concept",
                new_understanding="测试。",
                section_plans={"summary": "摘要"},
                reason="test",
            ),
            pipeline_module.WikiMergePlanItem(
                page_plan_id="PP-AGG",
                source_basis=SourceBasis(source_candidate_ids=["AGG-concepts-demo"]),
                action="create",
                canonical_target_path="overviews/Overview_AGG.md",
                display_title="聚合候选",
                page_type="overview",
                new_understanding="测试。",
                section_plans={"summary": "摘要"},
                reason="test",
            ),
            pipeline_module.WikiMergePlanItem(
                page_plan_id="PP-NOOP",
                source_basis=SourceBasis(source_candidate_ids=["D1"]),
                action="noop",
                canonical_target_path="designs/Design_D1.md",
                display_title="无关设计",
                page_type="design",
                new_understanding="测试。",
                section_plans={"summary": "摘要"},
                reason="test",
            ),
        ],
    )

    projected = draft_rendering_payloads_module.project_source_digest_for_merge_plan(digest, merge_plan)

    assert [candidate.candidate_id for candidate in projected.concepts] == ["C1", "AGG-concepts-demo"]
    assert projected.designs == []
    assert [candidate.candidate_id for candidate in projected.budget_deferred_candidates] == ["C3", "C4"]
    assert projected.weak_or_noise_items == []


def test_source_digest_candidate_budget_semantically_dedupes_open_questions_before_budget() -> None:
    def open_question(candidate_id: str, title: str) -> SourceDigestCandidate:
        return SourceDigestCandidate(
            candidate_id=candidate_id,
            name=title,
            type="open_question",
            one_sentence_summary=f"{title} 摘要。",
            why_matters=f"{title} 重要。",
            wiki_value=f"{title} 可复用。",
            source_locator=f"section {candidate_id}",
            suggested_page_title=title,
            open_question_or_tension=title,
        )

    digest = SourceDigestArtifact(
        source_raw_path="raw/sample.md",
        summary="测试去重。",
        open_questions=[
            open_question("O1", "AGI到来后PM角色会消失吗？"),
            open_question("O2", "AGI后PM是否必要？"),
            open_question("O3", "AGI到来后PM角色是否会消失？"),
            open_question("O4", "AI 时代 PM 如何训练产品判断？"),
        ],
    )

    capped, report = source_digest_budget.cap_source_digest_candidates(digest, 12)

    assert [item.candidate_id for item in capped.open_questions] == ["O1", "O4"]
    assert capped.open_questions[0].related_candidates == ["O2", "O3"]
    assert "source_digest_semantic_dedupe" in capped.open_questions[0].resolution_hint
    assert "section O2" in capped.open_questions[0].resolution_hint
    assert report["total_formal_candidates_before_dedupe"] == 4
    assert report["total_formal_candidates_before_budget"] == 2
    assert report["deduped_count"] == 2
    assert report["dedupe_applied"] is True
    assert [item["merged_candidate_id"] for item in report["deduped_candidates"]] == ["O2", "O3"]
    markdown = source_digest_budget.render_source_digest_budget_report(report)
    assert "## 语义去重候选" in markdown
    assert "semantic:agi_pm_role_necessity" in markdown
    assert "O2" in markdown


def test_source_digest_open_question_dedupe_prefers_tension_over_generic_title() -> None:
    def open_question(candidate_id: str, title: str, tension: str) -> SourceDigestCandidate:
        return SourceDigestCandidate(
            candidate_id=candidate_id,
            name=title,
            type="open_question",
            one_sentence_summary=f"{title} 摘要。",
            why_matters=f"{title} 重要。",
            wiki_value=f"{title} 可复用。",
            source_locator=f"section {candidate_id}",
            suggested_page_title=title,
            open_question_or_tension=tension,
        )

    digest = SourceDigestArtifact(
        source_raw_path="raw/sample.md",
        summary="测试去重。",
        open_questions=[
            open_question("O1", "PM 角色问题", "AGI到来后PM角色会消失吗？"),
            open_question("O2", "AI 产品组织问题", "AGI后PM是否必要？"),
        ],
    )

    capped, report = source_digest_budget.cap_source_digest_candidates(digest, 12)

    assert [item.candidate_id for item in capped.open_questions] == ["O1"]
    assert capped.open_questions[0].related_candidates == ["O2"]
    assert report["deduped_count"] == 1
    assert report["deduped_candidates"][0]["dedupe_key"] == "open_questions:semantic:agi_pm_role_necessity"


def test_source_digest_candidate_budget_semantically_dedupes_similar_concepts_before_budget() -> None:
    def concept(candidate_id: str, title: str, summary: str) -> SourceDigestCandidate:
        return SourceDigestCandidate(
            candidate_id=candidate_id,
            name=title,
            type="concept",
            one_sentence_summary=summary,
            why_matters=f"{summary} 对 AI PM 训练有复用价值。",
            wiki_value=f"{summary} 可沉淀为能力判断框架。",
            source_locator=f"section {candidate_id}",
            suggested_page_title=title,
        )

    digest = SourceDigestArtifact(
        source_raw_path="raw/sample.md",
        summary="测试 concept 去重。",
        concepts=[
            concept("C1", "AI PM 能力模型", "AI PM 能力模型覆盖判断力、技术协作和面试准备。"),
            concept("C2", "AI PM 能力框架", "AI PM 能力框架覆盖判断力、技术协作和面试准备。"),
            concept("C3", "AI PM 商业模式", "AI PM 商业模式关注收入结构、定价和客户价值。"),
        ],
    )

    capped, report = source_digest_budget.cap_source_digest_candidates(digest, 12)

    assert [item.candidate_id for item in capped.concepts] == ["C1", "C3"]
    assert capped.concepts[0].related_candidates == ["C2"]
    assert "source_digest_semantic_dedupe" in capped.concepts[0].resolution_hint
    assert "section C2" in capped.concepts[0].resolution_hint
    assert report["total_formal_candidates_before_dedupe"] == 3
    assert report["total_formal_candidates_before_budget"] == 2
    assert report["deduped_count"] == 1
    assert report["deduped_candidates"][0]["group"] == "concepts"
    assert report["deduped_candidates"][0]["merged_candidate_id"] == "C2"


def test_draft_source_excerpt_pack_truncates_long_source_by_page_cues() -> None:
    digest = SourceDigestArtifact.model_validate(read_json(FIXTURE_ROOT / "mock" / "source_digest.json"))
    merge_plan = WikiMergePlanArtifact.model_validate(read_json(FIXTURE_ROOT / "mock" / "wiki_merge_planning.json"))
    approved_text = (
        "# 长文材料\n\n"
        + ("背景填充段落，用来模拟很长的播客或论文转写。\n" * 240)
        + "\n## 知识编译工程骨架\n\n"
        "第一阶段先证明 CLI、状态机、artifact 和 validator 能跑通，避免只追求生成内容数量。\n"
        + ("中间填充段落。\n" * 160)
        + "\n## 简化 Ingest 草稿流程\n\n"
        "简化 Ingest 不直接写入正式 wiki，而是先生成 source 和 concept 草稿，再通过 review/apply 进入知识库。\n"
    )

    pack = draft_rendering_payloads_module.build_draft_source_excerpt_pack(
        approved_text,
        digest,
        merge_plan,
        full_source_limit=1_000,
        total_limit=1_400,
        per_page_limit=420,
        global_limit=180,
    )

    assert pack["full_source_in_payload"] is False
    assert pack["truncated_for_payload"] is True
    assert pack["original_char_count"] > pack["included_char_count"]
    assert pack["approved_prepared_ref"] == "prepared_raw_review/approved_prepared.md"
    by_title = {item["display_title"]: item for item in pack["items"]}
    first_snippets = "\n".join(snippet["text"] for snippet in by_title["知识编译工程骨架"]["snippets"])
    second_snippets = "\n".join(snippet["text"] for snippet in by_title["简化 Ingest 草稿流程"]["snippets"])
    assert "CLI、状态机、artifact 和 validator" in first_snippets
    assert "review/apply" in second_snippets

    markdown = draft_rendering_payloads_module.render_draft_source_excerpt_pack_markdown(pack)
    assert "## 页面摘录索引" in markdown
    assert "## 分页摘录" in markdown
    assert "知识编译工程骨架" in markdown


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
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=fixture_dir, slug="cleanup")
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
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=fixture_dir, slug="cleanup-resume")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    cleanup_report = run_dir / "raw_link_cleanup" / "raw_link_cleanup.json"
    assert cleanup_report.exists()

    with pytest.raises(PipelineError, match="Cannot resume from raw_link_cleanup"):
        resume_ingest(vault=vault, operation_id=manifest.operation_id, from_step="raw_link_cleanup")

    resumed = resume_ingest(vault=vault, operation_id=manifest.operation_id, from_step="raw_prepare")
    assert resumed.status == OperationStatus.drafted
    assert cleanup_report.exists()


def test_init_ingest_status_apply_closes_loop(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(
        vault=vault,
        raw_file=raw,
        mock_fixture_dir=FIXTURE_ROOT / "mock",
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
    assert prepared_decision["schema_version"] == "review_decision.v2"
    assert prepared_decision["decision"] == "approved"
    assert "review_mode" not in prepared_decision
    assert prepared_decision["auto_approved"] is True
    assert (run_dir / "source_digest" / "source_digest.json").exists()
    assert (run_dir / "source_digest" / "source_digest_budget_report.json").exists()
    assert (run_dir / "source_digest" / "source_digest_budget_report.md").exists()
    assert (run_dir / "source_digest_review" / "approved_digest.json").exists()
    digest_decision = read_json(run_dir / "source_digest_review" / "review_decision.json")
    assert digest_decision["schema_version"] == "review_decision.v2"
    assert digest_decision["decision"] == "approved"
    assert "review_mode" not in digest_decision
    assert digest_decision["auto_approved"] is True
    assert (run_dir / "candidate_resolution" / "candidate_resolution.json").exists()
    assert (run_dir / "source_duplicate_guard" / "source_duplicate_guard.json").exists()
    assert (run_dir / "wiki_context_snapshot" / "wiki_context_snapshot.json").exists()
    assert (run_dir / "merge_plan_review" / "approved_merge_plan.json").exists()
    merge_decision = read_json(run_dir / "merge_plan_review" / "review_decision.json")
    assert merge_decision["schema_version"] == "review_decision.v2"
    assert merge_decision["decision"] == "approved"
    assert "review_mode" not in merge_decision
    assert merge_decision["auto_approved"] is True
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
    assert "## 核心内容" in knowledge_text
    assert "\n## 详情\n" not in knowledge_text
    assert "### 价值点" in knowledge_text
    assert "\n## 价值点\n" not in knowledge_text
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
    assert loaded.schema_version == "operation_manifest.v10"
    assert [ref.schema_version for ref in loaded.steps[0].outputs if ref.kind == "json"] == ["raw_link_cleanup.v1"]
    assert "raw_preparation.v1" in [ref.schema_version for ref in loaded.steps[1].outputs if ref.kind == "json"]
    assert "structured_repair_report.v1" in [ref.schema_version for ref in loaded.steps[1].outputs if ref.kind == "json"]
    assert "source_digest.v2" in [ref.schema_version for ref in loaded.steps[3].outputs if ref.kind == "json"]
    assert "source_digest_budget_report.v1" in [ref.schema_version for ref in loaded.steps[3].outputs if ref.kind == "json"]
    assert "structured_repair_report.v1" in [ref.schema_version for ref in loaded.steps[3].outputs if ref.kind == "json"]
    assert [ref.schema_version for ref in loaded.steps[4].outputs if ref.relative_path.endswith("approved_digest.json")] == [
        "source_digest.v2"
    ]
    assert "candidate_resolution.v3" in [ref.schema_version for ref in loaded.steps[6].outputs if ref.kind == "json"]
    assert "structured_repair_report.v1" in [ref.schema_version for ref in loaded.steps[6].outputs if ref.kind == "json"]
    assert [ref.schema_version for ref in loaded.steps[7].outputs if ref.relative_path.endswith("wiki_context_snapshot.json")] == [
        "wiki_context_snapshot.v3"
    ]
    assert [ref.schema_version for ref in loaded.steps[7].outputs if ref.relative_path.endswith("candidate_contexts.json")] == [
        "candidate_contexts.v2"
    ]
    assert [ref.schema_version for ref in loaded.steps[8].outputs if ref.relative_path.endswith("wiki_merge_plan.json")] == [
        "wiki_merge_plan.v5"
    ]
    assert "structured_repair_report.v1" in [ref.schema_version for ref in loaded.steps[8].outputs if ref.kind == "json"]
    for review_name in ["prepared_raw_review", "source_digest_review", "merge_plan_review", "draft_review"]:
        review_step = [step for step in loaded.steps if step.name == review_name][0]
        assert review_step.review_state == "approved"
        assert review_step.review_decision_ref
    plan = read_json(run_dir / "wiki_merge_planning" / "wiki_merge_plan.json")
    assert plan["schema_version"] == "wiki_merge_plan.v5"
    assert (run_dir / "wiki_context_snapshot" / "candidate_contexts.json").exists()
    assert (run_dir / "wiki_context_snapshot" / "candidate_contexts.md").exists()
    assert (run_dir / "draft_rendering" / "update_merge_report.json").exists()
    assert (run_dir / "draft_rendering" / "draft_grounding_review.json").exists()
    assert (run_dir / "draft_rendering" / "related_merge_report.json").exists()
    contexts_markdown = (run_dir / "wiki_context_snapshot" / "candidate_contexts.md").read_text(encoding="utf-8")
    assert "resolved cache path" in contexts_markdown
    assert "query count" in contexts_markdown
    assert "编码页面数" in contexts_markdown
    assert "排序说明" in contexts_markdown
    assert "Score Bucket" in contexts_markdown
    assert "Sort Key" in contexts_markdown
    assert "title_distance" in contexts_markdown
    assert "lexical_expansion" in contexts_markdown
    assert (run_dir / "wiki_merge_planning" / "merge_decision_report.md").exists()
    snapshot = read_json(run_dir / "wiki_context_snapshot" / "wiki_context_snapshot.json")
    assert snapshot["schema_version"] == "wiki_context_snapshot.v3"
    assert snapshot["candidate_contexts"]["schema_version"] == "candidate_contexts.v2"
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
    assert metrics["current_attempt_duration_ms"] >= metrics["steps"][0]["last_duration_ms"]
    assert metrics["current_model_duration_ms"] >= 0
    assert metrics["internal_model_payload_char_count"] > 0
    assert "archived_model_duration_ms" not in metrics
    assert "total_model_duration_ms" not in metrics
    assert "archived_internal_model_payload_char_count" not in metrics
    assert "total_internal_model_payload_char_count" not in metrics
    assert metrics["payload_by_step"]
    assert metrics["largest_payload_step"]
    assert metrics["largest_payload_char_count"] > 0
    assert any(step.get("payload_char_count", 0) > 0 for step in metrics["steps"])
    metrics_markdown = (run_dir / "run_metrics.md").read_text(encoding="utf-8")
    assert "## Payload By Step" in metrics_markdown
    assert metrics["largest_payload_step"] in metrics_markdown
    assert metrics["candidate_page_budget"] == 12
    assert metrics["candidate_count_before_dedupe"] >= metrics["candidate_count_before_budget"]
    assert metrics["candidate_selected_count"] == metrics["candidate_count_before_budget"]
    assert metrics["candidate_deferred_count"] == 0
    assert metrics["candidate_deduped_count"] >= 0
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
    assert receipts[-1]["profile_version"] == "2"
    assert "profile_snapshot_hash" not in receipts[-1]
    assert "prepared_raw" in receipts[-1]
    assert receipts[-1]["raw_cleanup_artifact_ref"] == "raw_link_cleanup/raw_link_cleanup.json"
    assert receipts[-1]["raw_cleanup_diff_ref"] == "raw_link_cleanup/cleanup.diff"
    assert receipts[-1]["raw_cleanup_changed"] is False
    assert receipts[-1]["raw_cleanup_cleaned_link_count"] == 0
    assert status(vault, manifest.operation_id).status == OperationStatus.applied
    metrics_after_apply = read_json(run_dir / "run_metrics.json")
    assert metrics_after_apply["status"] == "applied"


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
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=fixture_dir, slug="profile")
    assert manifest.profile == "research_basic"


def test_resume_after_failed_step(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    broken_fixture = tmp_path / "broken"
    broken_fixture.mkdir()
    for name in ["raw_prepare.json"]:
        (broken_fixture / name).write_text((FIXTURE_ROOT / "mock" / name).read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(Exception):
        run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=broken_fixture, slug="broken")
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
    empty_fixture = tmp_path / "empty-fixture"
    empty_fixture.mkdir()
    config_path = vault / ".llmwiki" / "config.yaml"
    config = read_yaml(config_path)
    config["providers"]["default"] = {
        "spec": "mock:fixture",
        "fixture_dir": str(FIXTURE_ROOT / "mock"),
    }
    config["providers"]["source_digest"] = {
        "spec": "mock:fixture",
        "fixture_dir": empty_fixture.as_posix(),
    }
    write_yaml(config_path, config)

    with pytest.raises(Exception, match="Mock fixture missing"):
        run_simplified_ingest(vault=vault, raw_file=raw, slug="provider")
    operation_id = next(RunStore(vault).runs_root.iterdir()).name
    manifest = status(vault, operation_id)
    assert manifest.status == OperationStatus.failed
    failed_step = [step for step in manifest.steps if step.status == StepStatus.failed][0]
    assert failed_step.name == "source_digest"
    assert failed_step.attempts[-1].provider_spec == "mock:fixture"
    assert manifest.provider_contexts[0].providers["source_digest"].fixture_dir == empty_fixture.as_posix()


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
        run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=FIXTURE_ROOT / "mock", slug="missing-config")


def test_provider_construction_failure_records_attempt_provider(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    config_path = vault / ".llmwiki" / "config.yaml"
    config = read_yaml(config_path)
    config["providers"]["default"] = {
        "spec": "mock:fixture",
        "fixture_dir": str(FIXTURE_ROOT / "mock"),
    }
    config["providers"]["source_digest"] = {
        "spec": "openai_compatible:planner",
        "endpoint": "http://127.0.0.1:1/v1/chat/completions",
        "api_key": "secret-provider-key",
    }
    write_yaml(config_path, config)

    with pytest.raises(Exception):
        run_simplified_ingest(vault=vault, raw_file=raw, slug="provider-build")
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
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=FIXTURE_ROOT / "mock", slug="attempts")
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



@pytest.mark.parametrize(
    ("raw_prepare_policy", "expected_policy_value"),
    [
        (RawPreparePolicy.auto, "auto"),
        (RawPreparePolicy.force, "force"),
    ],
)
def test_raw_prepare_auto_and_force_use_model_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw_prepare_policy: RawPreparePolicy,
    expected_policy_value: str,
) -> None:
    vault, raw = make_vault(tmp_path)
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
    captured_payloads: dict[str, dict] = {}

    def fake_generate_raw(self, task, payload, output_model):
        captured_payloads[task] = payload
        data = read_json(FIXTURE_ROOT / "mock" / f"{task}.json")
        return json.dumps(data, ensure_ascii=False)

    monkeypatch.setattr(OpenAICompatibleProvider, "generate_raw", fake_generate_raw)

    manifest = run_simplified_ingest(
        vault=vault,
        raw_file=raw,
        slug=f"raw-prepare-{expected_policy_value}",
        raw_prepare_policy=raw_prepare_policy,
    )
    run_dir = RunStore(vault).run_dir(manifest.operation_id)

    assert "raw_prepare" in captured_payloads
    assert captured_payloads["raw_prepare"]["raw_prepare_policy"] == expected_policy_value
    assert captured_payloads["raw_prepare"]["raw_markdown"].strip() == raw.read_text(encoding="utf-8").strip()
    assert captured_payloads["raw_prepare"]["contract"]
    preparation = read_json(run_dir / "raw_prepare" / "raw_preparation.json")
    assert preparation["operations_applied"] == ["kept_clean_markdown"]
    assert (run_dir / "raw_prepare" / "prepared.md").exists()
    assert (run_dir / "raw_prepare" / "provider_result.json").exists()


def test_raw_prepare_skip_policy_writes_local_passthrough_without_model_fixture(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    fixture_dir = tmp_path / "mock-without-raw-prepare"
    fixture_dir.mkdir()
    for name in ["source_digest.json", "candidate_resolution.json", "wiki_merge_planning.json", "draft_rendering.json"]:
        write_json(fixture_dir / name, read_json(FIXTURE_ROOT / "mock" / name))

    manifest = run_simplified_ingest(
        vault=vault,
        raw_file=raw,
        mock_fixture_dir=fixture_dir,
        slug="skip-prepare-local-passthrough",
        raw_prepare_policy=RawPreparePolicy.skip,
    )
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    loaded = status(vault, manifest.operation_id)
    raw_prepare_step = [step for step in loaded.steps if step.name == "raw_prepare"][0]
    metrics = read_json(run_dir / "run_metrics.json")
    raw_prepare_metrics = [step for step in metrics["steps"] if step["name"] == "raw_prepare"][0]
    preparation = read_json(run_dir / "raw_prepare" / "raw_preparation.json")

    assert raw_prepare_step.attempts[-1].provider_spec is None
    assert raw_prepare_metrics["provider"] == "local"
    assert "raw_prepare" not in manifest.provider_contexts[0].providers
    assert preparation["operations_applied"] == ["user_skip_markdown_passthrough"]
    assert preparation["requires_human_review"] is False
    assert (run_dir / "raw_prepare" / "prepared.md").read_text(encoding="utf-8") == raw.read_text(encoding="utf-8").rstrip() + "\n"
    assert not (run_dir / "raw_prepare" / "provider_result.json").exists()
    assert not (run_dir / "raw_prepare" / "structured_repair_report.json").exists()


def test_raw_prepare_skip_policy_rejects_empty_markdown_without_provider_fallback(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    raw.write_text("", encoding="utf-8")

    with pytest.raises(PipelineError, match="--prepare skip requires non-empty raw Markdown"):
        run_simplified_ingest(
            vault=vault,
            raw_file=raw,
            mock_fixture_dir=FIXTURE_ROOT / "mock",
            slug="skip-prepare-empty",
            raw_prepare_policy=RawPreparePolicy.skip,
        )


def test_raw_prepare_skip_policy_rejects_non_markdown_without_provider_fallback(tmp_path: Path) -> None:
    vault, _ = make_vault(tmp_path)
    raw = vault / "raw" / "note.txt"
    raw.write_text("Plain text raw should use model prepare, not skip passthrough.\n", encoding="utf-8")

    with pytest.raises(PipelineError, match="--prepare skip requires Markdown raw"):
        run_simplified_ingest(
            vault=vault,
            raw_file=raw,
            mock_fixture_dir=FIXTURE_ROOT / "mock",
            slug="skip-prepare-non-markdown",
            raw_prepare_policy=RawPreparePolicy.skip,
        )


def test_vault_config_raw_prepare_policy_is_not_an_input(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    config_path = vault / ".llmwiki" / "config.json"
    config = read_json(config_path)
    config["raw_prepare_policy"] = "skip"
    write_json(config_path, config)

    with pytest.raises(ValidationError, match="raw_prepare_policy"):
        run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=FIXTURE_ROOT / "mock", slug="vault-config-policy")


def test_default_raw_prepare_policy_is_auto_and_recorded_in_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vault, raw = make_vault(tmp_path)
    provider_config_path = vault / ".llmwiki" / "config.yaml"
    provider_config = read_yaml(provider_config_path)
    provider_config["providers"] = {
        "default": {
            "spec": "openai_compatible:test-model",
            "endpoint": "https://example.test/v1/chat/completions",
            "api_key": "sk-test",
        }
    }
    write_yaml(provider_config_path, provider_config)
    captured_payloads: dict[str, dict] = {}

    def fake_generate_raw(self, task, payload, output_model):
        captured_payloads[task] = payload
        data = read_json(FIXTURE_ROOT / "mock" / f"{task}.json")
        return json.dumps(data, ensure_ascii=False)

    monkeypatch.setattr(OpenAICompatibleProvider, "generate_raw", fake_generate_raw)

    manifest = run_simplified_ingest(vault=vault, raw_file=raw, slug="default-prepare-auto")
    manifest_data = read_json(RunStore(vault).manifest_path(manifest.operation_id))

    assert manifest.vault_config_snapshot.raw_prepare_policy == RawPreparePolicy.auto
    assert manifest_data["vault_config_snapshot"]["raw_prepare_policy"] == "auto"
    assert captured_payloads["raw_prepare"]["raw_prepare_policy"] == "auto"


@pytest.mark.parametrize(
    ("initial_policy", "resume_policy", "slug"),
    [
        (None, RawPreparePolicy.skip, "resume-to-skip"),
        (RawPreparePolicy.skip, None, "resume-reuse-skip"),
    ],
)
def test_resume_from_raw_prepare_uses_explicit_or_snapshot_prepare_policy(
    tmp_path: Path,
    initial_policy: RawPreparePolicy | None,
    resume_policy: RawPreparePolicy | None,
    slug: str,
) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(
        vault=vault,
        raw_file=raw,
        mock_fixture_dir=FIXTURE_ROOT / "mock",
        slug=slug,
        raw_prepare_policy=initial_policy,
    )
    config = read_yaml(vault / ".llmwiki" / "config.yaml")
    config["providers"]["default"] = {
        "spec": "mock:fixture",
        "fixture_dir": str(FIXTURE_ROOT / "mock"),
    }
    write_yaml(vault / ".llmwiki" / "config.yaml", config)

    resumed = resume_ingest(
        vault=vault,
        operation_id=manifest.operation_id,
        from_step="raw_prepare",
        raw_prepare_policy=resume_policy,
    )
    raw_prepare_step = [step for step in resumed.steps if step.name == "raw_prepare"][0]

    assert resumed.vault_config_snapshot.raw_prepare_policy == RawPreparePolicy.skip
    assert "raw_prepare" not in resumed.provider_contexts[-1].providers
    assert raw_prepare_step.attempts[-1].provider_spec is None


def test_resume_prepare_override_requires_rerunning_raw_prepare(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=FIXTURE_ROOT / "mock", slug="late-prepare")

    with pytest.raises(PipelineError, match="raw prepare override only applies"):
        resume_ingest(
            vault=vault,
            operation_id=manifest.operation_id,
            from_step="source_digest",
            raw_prepare_policy=RawPreparePolicy.skip,
        )


def test_prepared_raw_review_for_skip_policy_is_plain_auto_approval(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)

    manifest = run_simplified_ingest(
        vault=vault,
        raw_file=raw,
        mock_fixture_dir=FIXTURE_ROOT / "mock",
        slug="skip-prepare-review",
        raw_prepare_policy=RawPreparePolicy.skip,
    )
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    prompt = (run_dir / "prepared_raw_review" / "review_prompt.md").read_text(encoding="utf-8")
    decision = read_json(run_dir / "prepared_raw_review" / "review_decision.json")

    assert "Skip Prepare 风险提示" not in prompt
    assert "policy_suppressed" not in prompt
    assert decision["notes"] == "当前运行自动批准；交互式审核尚未接入。"


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
    assert source_payload["language_contract"]["vault_language"] == "zh-CN"
    assert "summary" in source_payload["language_contract"]["fields_must_be_chinese"]
    assert "Do not answer source_digest in English" in source_payload["language_contract"]["hard_requirement"]
    assert "Claude Code" in source_payload["language_contract"]["stable_terms_may_remain_english"]
    assert "candidate_coverage_required_ids" in captured_payloads["candidate_resolution"]


def test_source_digest_payload_includes_readme_source_kind_hints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vault, raw = make_vault(tmp_path)
    raw.rename(vault / "raw" / "README.md")
    raw = vault / "raw" / "README.md"
    raw.write_text(
        "# Hello-Agents\n\n"
        "![GitHub stars](https://img.shields.io/github/stars/datawhalechina/Hello-Agents)\n"
        "[GitHub Project](https://github.com/datawhalechina/Hello-Agents)\n"
        "[PDF 下载](https://github.com/datawhalechina/hello-agents/releases/latest/)\n\n"
        "## 内容导航\n\n"
        + "\n".join(
            f"| [第{index}章](./docs/chapter{index}/第{index}章.md) | 智能体教程章节 {index} | ✅ |"
            for index in range(1, 13)
        )
        + "\n\n## 🙏 致谢\n\n"
        "- 陈思州 - 项目负责人，全文写作和校对。\n"
        "- 孙韬 - 联合发起者。\n",
        encoding="utf-8",
    )
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
        if task == "source_digest":
            data["source_raw_path"] = "raw/README.md"
        return json.dumps(data, ensure_ascii=False)

    monkeypatch.setattr(OpenAICompatibleProvider, "generate_raw", fake_generate_raw)

    manifest = run_simplified_ingest(
        vault=vault,
        raw_file=raw,
        slug="readme-kind-hints",
        raw_prepare_policy=RawPreparePolicy.skip,
    )
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    hints = captured_payloads["source_digest"]["source_kind_hints"]
    rules = "\n".join(captured_payloads["source_digest"]["contract"]["rules"])

    assert hints["github_url_present"] is True
    assert hints["repository_readme"] is True
    assert hints["tutorial_index"] is True
    assert hints["navigation_heavy"] is True
    assert hints["contributor_section_present"] is True
    assert hints["badge_or_download_heavy"] is True
    assert hints["counts"]["toc_link_count"] >= 12
    assert "do not create formal candidates for badges" in rules
    assert "contributor/acknowledgement people" in rules
    assert (run_dir / "source_digest" / "source_kind_hints.json").exists()
    assert (run_dir / "source_digest" / "source_kind_hints.md").exists()
    source_step = [step for step in manifest.steps if step.name == "source_digest"][0]
    assert "source_kind_hints.v1" in [ref.schema_version for ref in source_step.outputs if ref.kind == "json"]


def test_source_kind_hints_do_not_mark_article_with_single_github_link_as_readme() -> None:
    text = (
        "# Building Effective Agents\n\n"
        "这篇文章解释如何组合工作流、工具调用和评估。"
        "代码示例放在 [example repo](https://github.com/example/agent-demo) 里，"
        "但主体讨论的是设计原则、权衡和失败模式。\n\n"
        "## 何时使用工作流\n\n"
        "固定路径适合确定性高的任务，Agent 适合开放任务。\n"
    )

    hints = source_digest_payload_module.build_source_kind_hints(text, "raw/building-effective-agents.md")

    assert hints["github_url_present"] is True
    assert hints["repository_readme"] is False
    assert hints["tutorial_index"] is False
    assert hints["navigation_heavy"] is False


def test_source_kind_hints_do_not_mark_article_quickstart_heading_as_tutorial_index() -> None:
    text = (
        "# Agent 设计笔记\n\n"
        "这篇文章先讨论为什么简单工作流经常比复杂 Agent 更可靠。\n\n"
        "## 快速开始\n\n"
        "先定义任务边界，再接入一个工具调用示例。"
        "完整代码见 [repo](https://github.com/example/agent-note)。\n"
    )

    hints = source_digest_payload_module.build_source_kind_hints(text, "raw/agent-design-note.md")

    assert hints["github_url_present"] is True
    assert hints["tutorial_index"] is False
    assert hints["repository_readme"] is False


def test_source_kind_hints_do_not_mark_reference_heavy_article_as_navigation_index() -> None:
    links = "\n".join(f"- [参考资料 {index}](https://example.com/ref-{index})" for index in range(1, 27))
    text = (
        "# Agent 评估综述\n\n"
        "这篇文章比较多个评估框架的适用场景、失败模式和落地成本。\n\n"
        "## 参考资料\n\n"
        f"{links}\n"
    )

    hints = source_digest_payload_module.build_source_kind_hints(text, "raw/agent-evaluation-review.md")

    assert hints["counts"]["markdown_link_count"] >= 24
    assert hints["tutorial_index"] is False
    assert hints["navigation_heavy"] is False
    assert hints["repository_readme"] is False


def test_source_kind_hints_do_not_mark_deep_dive_related_docs_as_tutorial_index() -> None:
    links = "\n".join(f"- [相关实现 {index}](./docs/pattern-{index}.md)" for index in range(1, 10))
    text = (
        "# 上下文工程深度分析\n\n"
        "正文讨论上下文压缩、记忆选择、工具调用边界和评估设计。\n\n"
        "## 更多阅读\n\n"
        f"{links}\n"
    )

    hints = source_digest_payload_module.build_source_kind_hints(text, "raw/context-engineering-deep-dive.md")

    assert hints["counts"]["toc_link_count"] >= 6
    assert hints["tutorial_index"] is False
    assert hints["navigation_heavy"] is False
    assert hints["repository_readme"] is False


def test_source_digest_source_map_triggers_for_long_structured_interview() -> None:
    sections = []
    for index in range(1, 34):
        sections.append(
            f"### 访谈主题 {index}\n\n"
            + (f"这是一段关于 AI 原生产品、PM 工作方式、发布流程和团队协作的访谈内容 {index}。 " * 38)
        )
    text = "---\ntitle: Long Interview\n---\n\n## 访谈全文\n\n" + "\n\n".join(sections)

    source_map = source_digest_payload_module.build_source_digest_source_map(
        text,
        approved_prepared_ref="prepared_raw_review/approved_prepared.md",
    )

    assert len(text) > source_digest_payload_module.SOURCE_DIGEST_FULL_SOURCE_CHAR_LIMIT
    assert source_map["full_source_in_payload"] is False
    assert source_map["included_char_count"] < source_map["original_char_count"]
    assert source_map["section_excerpt_limit"] >= source_digest_payload_module.SOURCE_DIGEST_SOURCE_MAP_MIN_SECTION_EXCERPT_LIMIT
    assert any(section["heading"] == "访谈主题 20" for section in source_map["sections"])


def test_source_digest_payload_uses_source_map_for_long_prepared_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vault, raw = make_vault(tmp_path)
    raw.write_text(
        "# Long Research Note\n\n"
        "###### Abstract\n\n"
        + ("This note studies agent memory evaluation and durable wiki candidates. " * 120)
        + "\n\n## 1 Introduction\n\n"
        + ("The introduction explains the motivation, benchmark gap, and reusable concepts. " * 260)
        + "\n\n## 2 Method\n\n"
        + ("The method section describes dataset construction, scenarios, metrics, and comparisons. " * 260)
        + "\n\n## 3 Experiments\n\n"
        + ("Table 1: Accuracy and memory capacity across mechanisms.\n" * 40)
        + ("The experiments compare retrieval memory, reflective memory, and factual memory. " * 260)
        + "\n\n## 4 Conclusion\n\n"
        + ("The conclusion summarizes reusable implications for agent memory systems. " * 120),
        encoding="utf-8",
    )
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
    captured_payloads: dict[str, dict] = {}

    def fake_generate_raw(self, task, payload, output_model):
        captured_payloads[task] = payload
        data = read_json(FIXTURE_ROOT / "mock" / f"{task}.json")
        return json.dumps(data, ensure_ascii=False)

    monkeypatch.setattr(OpenAICompatibleProvider, "generate_raw", fake_generate_raw)

    manifest = run_simplified_ingest(
        vault=vault,
        raw_file=raw,
        slug="source-digest-map",
        raw_prepare_policy=RawPreparePolicy.skip,
    )
    run_dir = RunStore(vault).run_dir(manifest.operation_id)

    payload = captured_payloads["source_digest"]
    source_map = payload["source_digest_source_map"]
    assert payload["approved_prepared_markdown"] == ""
    assert payload["approved_prepared_ref"] == "prepared_raw_review/approved_prepared.md"
    assert source_map["schema_version"] == "source_digest_source_map_payload.v1"
    assert source_map["full_source_map_ref"] == "source_digest/source_digest_source_map.json"
    assert source_map["full_source_in_payload"] is False
    assert source_map["original_char_count"] > source_digest_payload_module.SOURCE_DIGEST_FULL_SOURCE_CHAR_LIMIT
    assert source_map["included_char_count"] < source_map["original_char_count"]
    assert any(section["heading"] == "2 Method" for section in source_map["sections"])
    assert source_map["captions"]
    assert "source_digest_source_map" in " ".join(payload["contract"]["rules"])

    sidecar = read_json(run_dir / "source_digest" / "source_digest_source_map.json")
    assert sidecar["schema_version"] == "source_digest_source_map.v1"
    payload_sidecar = read_json(run_dir / "source_digest" / "source_digest_source_map_payload.json")
    assert payload_sidecar["schema_version"] == "source_digest_source_map_payload.v1"
    assert (run_dir / "source_digest" / "source_digest_source_map.md").exists()
    digest_step = [step for step in manifest.steps if step.name == "source_digest"][0]
    source_map_ref = [
        ref
        for ref in digest_step.outputs
        if ref.relative_path == "source_digest/source_digest_source_map.json"
    ][0]
    assert source_map_ref.schema_version == "source_digest_source_map.v1"


def test_candidate_resolution_payload_uses_excerpt_pack_for_long_prepared_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vault, raw = make_vault(tmp_path)
    raw.write_text(
        raw.read_text(encoding="utf-8")
        + "\n\n"
        + ("长文填充段落，用来模拟论文或播客正文。\n" * 1_200)
        + "\n## 知识编译工程骨架\n\nCLI、状态机、artifact 和 validator 是第一阶段要验证的骨架。\n",
        encoding="utf-8",
    )
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
    captured_payloads: dict[str, dict] = {}

    def fake_generate_raw(self, task, payload, output_model):
        captured_payloads[task] = payload
        data = read_json(FIXTURE_ROOT / "mock" / f"{task}.json")
        return json.dumps(data, ensure_ascii=False)

    monkeypatch.setattr(OpenAICompatibleProvider, "generate_raw", fake_generate_raw)

    manifest = run_simplified_ingest(
        vault=vault,
        raw_file=raw,
        slug="candidate-resolution-pack",
        raw_prepare_policy=RawPreparePolicy.skip,
    )
    run_dir = RunStore(vault).run_dir(manifest.operation_id)

    payload = captured_payloads["candidate_resolution"]
    assert payload["approved_prepared_markdown"] == ""
    assert payload["approved_prepared_ref"] == "prepared_raw_review/approved_prepared.md"
    assert payload["source_excerpt_pack"]["schema_version"] == "candidate_resolution_source_excerpt_pack.v1"
    assert payload["source_excerpt_pack"]["full_source_in_payload"] is False
    assert payload["source_excerpt_pack"]["original_char_count"] > planning_payloads_module.CANDIDATE_RESOLUTION_FULL_SOURCE_CHAR_LIMIT
    assert payload["source_excerpt_pack"]["included_char_count"] < payload["source_excerpt_pack"]["original_char_count"]
    assert "source_excerpt_pack" in " ".join(payload["contract"]["rules"])

    sidecar = read_json(run_dir / "candidate_resolution" / "candidate_resolution_source_excerpt_pack.json")
    assert sidecar["schema_version"] == "candidate_resolution_source_excerpt_pack.v1"
    assert (run_dir / "candidate_resolution" / "candidate_resolution_source_excerpt_pack.md").exists()
    candidate_step = [step for step in manifest.steps if step.name == "candidate_resolution"][0]
    source_pack_ref = [
        ref
        for ref in candidate_step.outputs
        if ref.relative_path == "candidate_resolution/candidate_resolution_source_excerpt_pack.json"
    ][0]
    assert source_pack_ref.schema_version == "candidate_resolution_source_excerpt_pack.v1"


def test_draft_rendering_payload_uses_excerpt_pack_for_long_prepared_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vault, raw = make_vault(tmp_path)
    raw.write_text(
        raw.read_text(encoding="utf-8")
        + "\n\n"
        + ("长文填充段落，用来模拟论文摘录和产品分析材料的冗长上下文。\n" * 1_200)
        + "\n## 知识编译工程骨架\n\nCLI、状态机、artifact 和 validator 是第一阶段要验证的骨架。\n",
        encoding="utf-8",
    )
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
    captured_payloads: dict[str, dict] = {}

    def fake_generate_raw(self, task, payload, output_model):
        captured_payloads[task] = payload
        data = read_json(FIXTURE_ROOT / "mock" / f"{task}.json")
        return json.dumps(data, ensure_ascii=False)

    monkeypatch.setattr(OpenAICompatibleProvider, "generate_raw", fake_generate_raw)

    manifest = run_simplified_ingest(
        vault=vault,
        raw_file=raw,
        slug="long-draft-payload",
        raw_prepare_policy=RawPreparePolicy.skip,
    )
    run_dir = RunStore(vault).run_dir(manifest.operation_id)

    payload = captured_payloads["draft_rendering"]
    assert payload["approved_prepared_markdown"] == ""
    assert payload["approved_prepared_ref"] == "prepared_raw_review/approved_prepared.md"
    assert payload["approved_digest_ref"] == "source_digest_review/approved_digest.json"
    assert payload["approved_merge_plan_ref"] == "merge_plan_review/approved_merge_plan.json"
    assert payload["approved_merge_plan"]["schema_version"] == "draft_merge_plan_projection.v1"
    assert payload["required_page_plan_ids"]
    assert payload["required_target_paths"]
    assert "exactly required_page_plan_ids" in " ".join(payload["contract"]["rules"])
    assert payload["wiki_context_snapshot"]["schema_version"] == "wiki_context_snapshot_projection.v1"
    assert "candidate_contexts" not in payload["wiki_context_snapshot"]
    assert all("content" not in entry for entry in payload["wiki_context_snapshot"]["entries"])
    assert payload["source_excerpt_pack"]["full_source_in_payload"] is False
    assert payload["source_excerpt_pack"]["original_char_count"] > draft_rendering_payloads_module.DRAFT_RENDERING_FULL_SOURCE_CHAR_LIMIT
    assert payload["source_excerpt_pack"]["included_char_count"] < payload["source_excerpt_pack"]["original_char_count"]
    contract_rules = " ".join(payload["contract"]["rules"])
    grounding_risk_rules = " ".join(payload["contract"]["grounding_risk_rules"])
    assert payload["contract"]["grounding_risk_rules"] == list(draft_grounding.DRAFT_RENDERING_GROUNDING_RISK_RULES)
    assert "source_excerpt_pack" in contract_rules
    assert "satisfy update_preservation_pack in the first draft" in contract_rules
    assert "and reusable key phrases into body_markdown" in contract_rules
    assert "change_summary may summarize retention but does not satisfy the obligation" in contract_rules
    assert "Do not wrap paraphrases" in contract_rules
    assert "Do not wrap paraphrases" in grounding_risk_rules
    assert "conversational source text" in grounding_risk_rules
    assert "speaker-like wording as paraphrase" in grounding_risk_rules
    assert "popularity/adoption/authority claims" in grounding_risk_rules
    assert "source-local capabilities" in grounding_risk_rules
    assert "broader phrasing" in grounding_risk_rules
    assert "allowed as hypotheses" in grounding_risk_rules
    assert "最佳实践" in grounding_risk_rules
    assert "phrase them with uncertainty or 待补来源" in grounding_risk_rules
    assert "For causal/scope terms" in grounding_risk_rules
    assert "keep the wording proportional to the source" in grounding_risk_rules
    assert "可能伴随" in grounding_risk_rules
    assert "translate or paraphrase English raw examples into Chinese" in contract_rules
    assert "Across body_markdown/open_questions" in contract_rules
    assert "source_coverage_notes" in contract_rules
    assert "张三" in contract_rules
    assert "user-123" in contract_rules
    assert "avoid presenting it as an observed user fact" in contract_rules
    assert "用户偏好 X" in contract_rules
    assert "user has already approved this material for ingest" in contract_rules
    assert "Grounding should protect source fidelity" in contract_rules
    assert "only contradiction with the approved source should create review" in contract_rules
    assert "CLI/API/code examples" in contract_rules
    assert "<memory_text>" in contract_rules
    assert "<user_id>" in contract_rules
    assert "<memory_query>" in contract_rules
    assert "explicit exception to the zh-CN translation rule" in contract_rules
    assert "do not translate a source literal into a new concrete preference" in contract_rules
    assert "do not invent causal outcomes" in contract_rules
    assert "导致" in contract_rules
    assert "recommendation or best practice" in contract_rules
    assert "Stable English product/protocol terms" in contract_rules

    sidecar = read_json(run_dir / "draft_rendering" / "draft_source_excerpt_pack.json")
    assert sidecar["schema_version"] == "draft_source_excerpt_pack.v1"
    assert (run_dir / "draft_rendering" / "draft_source_excerpt_pack.md").exists()
    assert len(payload["approved_merge_plan"]["items"]) == len(payload["required_page_plan_ids"])
    draft_step = [step for step in manifest.steps if step.name == "draft_rendering"][0]
    excerpt_ref = [
        ref
        for ref in draft_step.outputs
        if ref.relative_path == "draft_rendering/draft_source_excerpt_pack.json"
    ][0]
    assert excerpt_ref.schema_version == "draft_source_excerpt_pack.v1"


def test_draft_context_projection_keeps_related_metadata_and_omits_weak_inspected() -> None:
    update_item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-UPDATE",
        source_basis=SourceBasis(source_candidate_ids=["C001"]),
        action="update",
        canonical_target_path="entities/Entity_Claude Code.md",
        matched_page="entities/Entity_Claude Code.md",
        display_title="Claude Code",
        page_type="entity",
        strongest_overlap=pipeline_module.ContextOverlapSignal(
            strength="weak",
            path="concepts/Concept_Update_Strongest.md",
            reason="weak update overlap should still keep metadata",
        ),
        inspected_context_paths=["concepts/Concept_Update_Inspected.md"],
        new_understanding="补充产品视角。",
        section_plans={"detail": "详情"},
        reason="测试 draft context projection。",
    )
    create_item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-CREATE",
        source_basis=SourceBasis(source_candidate_ids=["C002"]),
        action="create",
        canonical_target_path="concepts/Concept_New.md",
        display_title="New Concept",
        page_type="concept",
        new_understanding="新增概念。",
        section_plans={"detail": "详情"},
        related_pages=[
            pipeline_module.RelatedPageRef(
                target_path="entities/Entity_Managed Agents.md",
                display_title="Managed Agents",
                source="wiki_context",
                reason="相关旧页。",
            )
        ],
        inspected_context_paths=["concepts/Concept_大脑与双手解耦.md"],
        reason="测试 create related 保留 metadata，但 weak inspected 不进入 draft payload。",
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/entities/Entity_Claude Code.md",
                expected_state="present",
                preimage_sha256="old-claude",
                content="# Claude Code\n\n旧页 Managed Agents / harness 架构视角。\n",
            ),
            pipeline_module.WikiContextEntry(
                path="wiki/entities/Entity_Managed Agents.md",
                expected_state="present",
                preimage_sha256="old-managed",
                content="# Managed Agents\n\n" + ("相关旧页正文不应进入 create draft payload。\n" * 40),
            ),
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_Update_Strongest.md",
                expected_state="present",
                preimage_sha256="old-update-strongest",
                content="# Update Strongest\n\n" + ("update inspected context 正文不应进入 draft payload。\n" * 40),
            ),
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_Update_Inspected.md",
                expected_state="present",
                preimage_sha256="old-update-inspected",
                content="# Update Inspected\n\n" + ("update inspected context 正文不应进入 draft payload。\n" * 40),
            ),
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_大脑与双手解耦.md",
                expected_state="present",
                preimage_sha256="old-brain",
                content="# 大脑与双手解耦\n\n" + ("inspected context 正文也不应进入 create draft payload。\n" * 40),
            ),
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_New.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            ),
        ],
    )
    metadata_paths, content_paths = draft_rendering_payloads_module.draft_rendering_relevant_wiki_paths(
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[update_item, create_item])
    )
    projection = draft_rendering_payloads_module.compact_snapshot_for_draft_rendering(
        snapshot,
        metadata_paths,
        "wiki_context_snapshot/wiki_context_snapshot.json",
        content_paths=content_paths,
    )
    entries = {entry["path"]: entry for entry in projection["entries"]}

    assert "wiki/entities/Entity_Claude Code.md" in content_paths
    assert draft_rendering_payloads_module.should_include_draft_inspected_context(update_item) is True
    assert entries["wiki/entities/Entity_Claude Code.md"]["content_excerpt"]
    assert entries["wiki/entities/Entity_Claude Code.md"]["content_role"] == "draft_context"
    assert entries["wiki/concepts/Concept_Update_Strongest.md"]["content_excerpt"] == ""
    assert entries["wiki/concepts/Concept_Update_Strongest.md"]["content_role"] == "metadata_only"
    assert entries["wiki/concepts/Concept_Update_Inspected.md"]["content_excerpt"] == ""
    assert entries["wiki/concepts/Concept_Update_Inspected.md"]["content_role"] == "metadata_only"
    assert entries["wiki/entities/Entity_Managed Agents.md"]["content_excerpt"] == ""
    assert entries["wiki/entities/Entity_Managed Agents.md"]["content_role"] == "metadata_only"
    assert "wiki/concepts/Concept_大脑与双手解耦.md" not in entries
    assert "wiki/concepts/Concept_大脑与双手解耦.md" not in metadata_paths
    assert projection["included_content_entry_count"] == 1
    assert projection["included_entry_content_chars"] == len(entries["wiki/entities/Entity_Claude Code.md"]["content_excerpt"])


def test_draft_context_projection_keeps_medium_metadata_and_strong_content() -> None:
    medium_item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-MEDIUM",
        source_basis=SourceBasis(source_candidate_ids=["C001"]),
        action="create",
        canonical_target_path="concepts/Concept_Medium_New.md",
        display_title="Medium New",
        page_type="concept",
        strongest_overlap=pipeline_module.ContextOverlapSignal(
            strength="medium",
            path="concepts/Concept_Medium_Context.md",
            reason="medium overlap should keep metadata",
        ),
        inspected_context_paths=["concepts/Concept_Medium_Inspected.md"],
        new_understanding="新增 medium create。",
        section_plans={"detail": "详情"},
        reason="测试 medium create 保留 inspected metadata。",
    )
    strong_item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-STRONG",
        source_basis=SourceBasis(source_candidate_ids=["C002"]),
        action="create",
        canonical_target_path="concepts/Concept_Strong_New.md",
        display_title="Strong New",
        page_type="concept",
        strongest_overlap=pipeline_module.ContextOverlapSignal(
            strength="strong",
            path="concepts/Concept_Strong_Context.md",
            reason="strong overlap should keep content",
        ),
        inspected_context_paths=["concepts/Concept_Strong_Inspected.md"],
        new_understanding="新增 strong create。",
        section_plans={"detail": "详情"},
        reason="测试 strong create 保留 strongest overlap 正文。",
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_Medium_Context.md",
                expected_state="present",
                preimage_sha256="medium-context",
                content="# Medium Context\n\n" + ("medium overlap 正文不应进入 create payload。\n" * 20),
            ),
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_Medium_Inspected.md",
                expected_state="present",
                preimage_sha256="medium-inspected",
                content="# Medium Inspected\n\n" + ("medium inspected 正文不应进入 create payload。\n" * 20),
            ),
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_Strong_Context.md",
                expected_state="present",
                preimage_sha256="strong-context",
                content="# Strong Context\n\nstrong overlap 正文应进入 create payload。\n",
            ),
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_Strong_Inspected.md",
                expected_state="present",
                preimage_sha256="strong-inspected",
                content="# Strong Inspected\n\n" + ("strong inspected 只保留 metadata。\n" * 20),
            ),
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_Medium_New.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            ),
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_Strong_New.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            ),
        ],
    )

    metadata_paths, content_paths = draft_rendering_payloads_module.draft_rendering_relevant_wiki_paths(
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[medium_item, strong_item])
    )
    projection = draft_rendering_payloads_module.compact_snapshot_for_draft_rendering(
        snapshot,
        metadata_paths,
        "wiki_context_snapshot/wiki_context_snapshot.json",
        content_paths=content_paths,
    )
    entries = {entry["path"]: entry for entry in projection["entries"]}

    assert draft_rendering_payloads_module.should_include_draft_inspected_context(medium_item) is True
    assert draft_rendering_payloads_module.should_include_draft_inspected_context(strong_item) is True
    assert entries["wiki/concepts/Concept_Medium_Context.md"]["content_role"] == "metadata_only"
    assert entries["wiki/concepts/Concept_Medium_Inspected.md"]["content_role"] == "metadata_only"
    assert entries["wiki/concepts/Concept_Strong_Context.md"]["content_role"] == "draft_context"
    assert entries["wiki/concepts/Concept_Strong_Context.md"]["content_excerpt"]
    assert entries["wiki/concepts/Concept_Strong_Inspected.md"]["content_role"] == "metadata_only"
    assert "wiki/concepts/Concept_Strong_Context.md" in content_paths
    assert "wiki/concepts/Concept_Medium_Context.md" not in content_paths


def test_wiki_merge_planning_payload_uses_compact_context_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vault, raw = make_vault(tmp_path)
    raw.write_text(
        raw.read_text(encoding="utf-8")
        + "\n\n"
        + ("长文填充段落，用来模拟需要规划合并的长 raw。\n" * 1_000)
        + "\n## 知识编译工程骨架\n\nCLI、状态机、artifact 和 validator 是第一阶段要验证的骨架。\n",
        encoding="utf-8",
    )
    config_json = read_json(vault / ".llmwiki" / "config.json")
    config_json["embedding_retrieval"]["backend"] = "exact"
    write_json(vault / ".llmwiki" / "config.json", config_json)
    existing = vault / "wiki" / "concepts" / "Concept_Existing_Runtime.md"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text(
        "---\n"
        "llmwiki_type: concept\n"
        "title: Existing Runtime\n"
        "aliases:\n"
        "  - 知识编译工程骨架\n"
        "summary: Existing runtime page that mentions CLI, 状态机, artifact, validator, and ingest planning.\n"
        "updated: 2026-01-02\n"
        "---\n\n"
        "# Existing Runtime\n\n"
        + ("CLI、状态机、artifact、validator 和 ingest planning 需要稳定的工程骨架。\n" * 80),
        encoding="utf-8",
    )
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
    captured_payloads: dict[str, dict] = {}

    def fake_generate_raw(self, task, payload, output_model):
        captured_payloads[task] = payload
        data = read_json(FIXTURE_ROOT / "mock" / f"{task}.json")
        return json.dumps(data, ensure_ascii=False)

    monkeypatch.setattr(OpenAICompatibleProvider, "generate_raw", fake_generate_raw)

    manifest = run_simplified_ingest(
        vault=vault,
        raw_file=raw,
        slug="planning-projection",
        raw_prepare_policy=RawPreparePolicy.skip,
    )
    run_dir = RunStore(vault).run_dir(manifest.operation_id)

    payload = captured_payloads["wiki_merge_planning"]
    assert payload["approved_prepared_markdown"] == ""
    assert payload["approved_prepared_ref"] == "prepared_raw_review/approved_prepared.md"
    assert payload["approved_digest_ref"] == "source_digest_review/approved_digest.json"
    assert payload["candidate_resolution_ref"] == "candidate_resolution/candidate_resolution.json"
    assert payload["approved_digest"]["schema_version"] == "source_digest.v2"
    assert payload["candidate_resolution"]["schema_version"] == "candidate_resolution.v3"
    assert "approved_digest_projection" not in payload
    assert "candidate_resolution_projection" not in payload
    assert "source_excerpt_pack" not in payload["merge_planning_context_pack"]
    assert "wiki_context_projection" not in payload["merge_planning_context_pack"]
    assert "candidate_contexts_projection" not in payload["merge_planning_context_pack"]
    assert payload["source_excerpt_pack"]["full_source_in_payload"] is False
    projection = payload["wiki_context_snapshot"]
    assert projection["schema_version"] == "wiki_context_snapshot_projection.v1"
    assert "candidate_contexts" not in projection
    assert projection["included_entry_count"] <= projection["full_entry_count"]
    assert projection["included_entry_content_chars"] < sum(
        len(entry["content"]) for entry in read_json(run_dir / "wiki_context_snapshot" / "wiki_context_snapshot.json")["entries"]
    )
    contexts = payload["candidate_contexts"]
    assert contexts["schema_version"] == "candidate_contexts_projection.v1"
    assert contexts["query_limit"] == planning_payloads_module.MERGE_PLANNING_CONTEXT_QUERY_LIMIT
    assert contexts["hit_excerpt_limit"] == planning_payloads_module.MERGE_PLANNING_CONTEXT_HIT_EXCERPT_LIMIT
    assert contexts["weak_hit_excerpt_limit"] == planning_payloads_module.MERGE_PLANNING_WEAK_CONTEXT_HIT_EXCERPT_LIMIT
    assert contexts["weak_hit_excerpt_max_rank"] == planning_payloads_module.MERGE_PLANNING_WEAK_CONTEXT_EXCERPT_MAX_RANK
    assert contexts["hit_excerpt_role"] == "match_preview"
    assert contexts["content_evidence_ref"] == "wiki_context_projection.entries"
    assert any(hit["path"] == "concepts/Concept_Existing_Runtime.md" for item in contexts["items"] for hit in item["hits"])
    projected_hit = next(hit for item in contexts["items"] for hit in item["hits"])
    assert isinstance(projected_hit["score_bucket"], int)
    assert "bucket=" in projected_hit["sort_explanation"]
    assert all(len(item["query"]) <= planning_payloads_module.MERGE_PLANNING_CONTEXT_QUERY_LIMIT for item in contexts["items"])
    assert all(
        len(hit["excerpt"]) <= planning_payloads_module.MERGE_PLANNING_CONTEXT_HIT_EXCERPT_LIMIT
        for item in contexts["items"]
        for hit in item["hits"]
    )
    assert all(
        len(hit["excerpt"]) <= planning_payloads_module.MERGE_PLANNING_WEAK_CONTEXT_HIT_EXCERPT_LIMIT
        for item in contexts["items"]
        for hit in item["hits"]
        if hit["strength"] == "weak" and not hit["forced"]
    )
    assert all(
        hit["excerpt_limit"]
        == (
            0
            if hit["strength"] == "weak"
            and not hit["forced"]
            and hit["rank"] > planning_payloads_module.MERGE_PLANNING_WEAK_CONTEXT_EXCERPT_MAX_RANK
            else (
                planning_payloads_module.MERGE_PLANNING_WEAK_CONTEXT_HIT_EXCERPT_LIMIT
                if hit["strength"] == "weak" and not hit["forced"]
                else planning_payloads_module.MERGE_PLANNING_CONTEXT_HIT_EXCERPT_LIMIT
            )
        )
        for item in contexts["items"]
        for hit in item["hits"]
    )
    assert all(
        hit["excerpt"] == ""
        for item in contexts["items"]
        for hit in item["hits"]
        if hit["strength"] == "weak" and not hit["forced"] and hit["rank"] > planning_payloads_module.MERGE_PLANNING_WEAK_CONTEXT_EXCERPT_MAX_RANK
    )

    sidecar = read_json(run_dir / "wiki_merge_planning" / "merge_planning_context_pack.json")
    assert sidecar["schema_version"] == "merge_planning_context_pack.v1"
    assert sidecar["original_counts"]["candidate_contexts_json_chars"] > sidecar["projected_counts"]["candidate_contexts_projection_json_chars"]
    assert sidecar["original_counts"]["approved_digest_json_chars"] == planning_payloads_module.json_char_count(
        read_json(run_dir / "source_digest_review" / "approved_digest.json")
    )
    assert sidecar["original_counts"]["candidate_resolution_json_chars"] == planning_payloads_module.json_char_count(
        read_json(run_dir / "candidate_resolution" / "candidate_resolution.json")
    )
    assert (run_dir / "wiki_merge_planning" / "merge_planning_context_pack.md").exists()
    planning_step = [step for step in manifest.steps if step.name == "wiki_merge_planning"][0]
    pack_ref = [
        ref
        for ref in planning_step.outputs
        if ref.relative_path == "wiki_merge_planning/merge_planning_context_pack.json"
    ][0]
    assert pack_ref.schema_version == "merge_planning_context_pack.v1"


def test_wiki_merge_planning_skips_model_for_empty_vault_all_create(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vault, raw = make_vault(tmp_path)
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
    calls: list[str] = []

    def tracking_generate_raw(self, task, payload, output_model):
        if task == "wiki_merge_planning":
            raise AssertionError("wiki_merge_planning should use the empty-vault local shortcut")
        calls.append(task)
        data = read_json(FIXTURE_ROOT / "mock" / f"{task}.json")
        return json.dumps(data, ensure_ascii=False)

    monkeypatch.setattr(OpenAICompatibleProvider, "generate_raw", tracking_generate_raw)

    manifest = run_simplified_ingest(vault=vault, raw_file=raw, slug="empty-plan-shortcut")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)

    assert "source_digest" in calls
    assert "candidate_resolution" in calls
    assert "draft_rendering" in calls
    assert "wiki_merge_planning" not in calls
    assert not (run_dir / "wiki_merge_planning" / "provider_result.json").exists()
    shortcut = read_json(run_dir / "wiki_merge_planning" / "merge_planning_shortcut_report.json")
    assert shortcut["used"] is True
    assert shortcut["shortcut"] == "empty_vault_all_create"
    assert shortcut["knowledge_metadata_pool_count"] == 0
    assert shortcut["candidate_context_hit_count"] == 0
    plan = read_json(run_dir / "wiki_merge_planning" / "wiki_merge_plan.json")
    assert {item["action"] for item in plan["items"]} == {"create"}
    metrics = read_json(run_dir / "run_metrics.json")
    planning_row = next(row for row in metrics["steps"] if row["name"] == "wiki_merge_planning")
    assert planning_row["local_shortcut"] is True
    assert planning_row["local_shortcut_rule"] == "empty_vault_all_create"
    assert "payload_char_count" not in planning_row


def test_empty_vault_merge_shortcut_accepts_prepared_discovered_candidate_refs() -> None:
    digest = SourceDigestArtifact(
        source_raw_path="raw/sample.md",
        summary="Digest summary.",
        budget_deferred_candidates=[
            SourceDigestCandidate(
                candidate_id="C005",
                name="多脑多手架构",
                type="concept",
                one_sentence_summary="多脑多手架构摘要。",
                why_matters="它是来源中发现的架构主题。",
                wiki_value="应成为概念页。",
                suggested_page_title="多脑多手架构",
            )
        ],
    )
    resolution = CandidateResolutionArtifact(
        items=[
            CandidateResolutionItem(
                page_plan_id="PP-C005",
                source_basis=SourceBasis(prepared_discovered_candidates=["C005"], source_locator="S010-S011"),
                page_type="concept",
                display_title="多脑多手架构",
                candidate_target_path="concepts/Concept_多脑多手架构.md",
                topic_summary="多脑多手架构摘要。",
                why_this_page="值得记录。",
                reason="prepared_discovered",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_多脑多手架构.md",
                expected_state="missing",
            )
        ],
    )
    contexts = pipeline_module.CandidateContextsArtifact(
        retrieval_backend="sentence_transformers",
        items=[],
    )

    report = pipeline_module.empty_vault_create_merge_planning_shortcut_report(digest, resolution, snapshot, contexts)
    plan = pipeline_module.build_wiki_merge_plan(resolution, digest, snapshot, log_date="2026-06-06")

    assert report["used"] is True
    assert report["blocking_conditions"] == []
    assert plan.items[0].action == "create"
    assert plan.items[0].display_title == "多脑多手架构"
    assert plan.items[0].source_basis.prepared_discovered_candidates == ["C005"]
    assert report["missing_source_candidate_page_plan_count"] == 0
    assert report["unknown_source_candidate_id_count"] == 0


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
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=FIXTURE_ROOT / "mock", slug="rerun")
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
    assert not stale.exists()
    assert (run_dir / "source_digest" / "source_digest.json").exists()
    assert (run_dir / "draft_rendering" / "draft_pages").exists()
    assert (run_dir / "prepared_raw_review" / "approved_prepared.md").exists()
    digest_step = [step for step in resumed.steps if step.name == "source_digest"][0]
    assert digest_step.attempts[0].outputs
    assert digest_step.attempts[0].completed_at is not None
    assert digest_step.attempts[0].duration_ms is not None
    assert not (run_dir / "attempt_archive").exists()
    assert digest_step.attempts[-1].outputs
    metrics = read_json(run_dir / "run_metrics.json")
    assert metrics["internal_model_call_count"] == 5
    assert metrics["current_attempt_duration_ms"] >= 0
    assert metrics["current_model_duration_ms"] >= 0
    assert metrics["internal_model_payload_char_count"] > 0
    assert "archived_internal_model_call_count" not in metrics
    assert "total_internal_model_call_count" not in metrics
    assert "archived_model_duration_ms" not in metrics
    assert "total_model_duration_ms" not in metrics
    assert "archived_internal_model_payload_char_count" not in metrics
    assert "total_internal_model_payload_char_count" not in metrics
    digest_metrics = [step for step in metrics["steps"] if step["name"] == "source_digest"][0]
    assert digest_metrics["internal_model_call_count"] == 1
    assert digest_metrics["payload_char_count"] > 0
    assert "archived_internal_model_call_count" not in digest_metrics
    assert "total_internal_model_call_count" not in digest_metrics
    assert "archived_payload_char_count" not in digest_metrics
    assert "total_payload_char_count" not in digest_metrics


def test_resume_can_force_mock_fixture_over_live_config(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=FIXTURE_ROOT / "mock", slug="resume-force-mock")
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

    resumed = resume_ingest(
        vault=vault,
        operation_id=manifest.operation_id,
        from_step="draft_rendering",
        mock_fixture_dir=FIXTURE_ROOT / "mock",
    )

    context = resumed.provider_contexts[-1]
    assert context.from_step == "draft_rendering"
    assert set(context.providers) == {"draft_rendering"}
    assert context.providers["draft_rendering"].spec == "mock:fixture"
    assert context.providers["draft_rendering"].fixture_dir == (FIXTURE_ROOT / "mock").resolve().as_posix()


def test_step_repair_metrics_uses_per_step_attempts_for_current_provider_counts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"

    def write_report(step_dir: Path, attempt_count: int) -> None:
        step_dir.mkdir(parents=True, exist_ok=True)
        for index in range(1, attempt_count + 1):
            nested_result = step_dir / f"model_batches/batch-{index:03d}/provider_result.json"
            nested_result.parent.mkdir(parents=True, exist_ok=True)
            write_json(
                nested_result,
                {
                    "task": "draft_rendering",
                    "payload_char_count": index * 100,
                    "json_repair_applied": index % 2 == 0,
                    "http_attempt_count": index + 1,
                },
            )
        write_json(
            step_dir / "structured_repair_report.json",
            {
                "schema_version": "structured_repair_report.v1",
                "task": "draft_rendering",
                "provider": "batched:mock",
                "attempt_count": attempt_count,
                "repair_count": 0,
                "duration_ms": attempt_count * 10,
                "attempts": [
                    {
                        "attempt": index,
                        "provider_result_ref": f"model_batches/batch-{index:03d}/provider_result.json",
                        "issues": [],
                        "parse_success": True,
                        "schema_valid": True,
                    }
                    for index in range(1, attempt_count + 1)
                ],
                "final_provider_result_ref": "provider_result.json",
            },
        )
        write_json(step_dir / "provider_result.json", {"task": "draft_rendering"})

    write_report(run_dir / "draft_rendering", 2)

    current = run_metrics_module.step_repair_metrics(run_dir, "draft_rendering")

    assert current["attempt_count"] == 2
    assert current["provider_result_count"] == 2
    assert current["http_attempt_count"] == 5
    assert current["payload_char_count"] == 300
    assert current["json_repair_count"] == 1


def test_resume_invalid_provider_config_does_not_delete_outputs(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=FIXTURE_ROOT / "mock", slug="invalid")
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
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=FIXTURE_ROOT / "mock", slug="bad-fallback")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    digest = run_dir / "source_digest" / "source_digest.json"
    config = read_yaml(vault / ".llmwiki" / "config.yaml")
    config["providers"]["default"] = {
        "spec": "mock:fixture",
        "fixture_dir": str(FIXTURE_ROOT / "mock"),
    }
    config["providers"]["source_digest"] = ""
    write_yaml(vault / ".llmwiki" / "config.yaml", config)
    with pytest.raises(Exception, match="Invalid provider config for task: source_digest"):
        resume_ingest(vault=vault, operation_id=manifest.operation_id, from_step="source_digest")
    assert digest.exists()
    assert len(status(vault, manifest.operation_id).provider_contexts) == 1


def test_resume_from_outputless_step_does_not_require_provider_context(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=FIXTURE_ROOT / "mock", slug="no-model-resume")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    apply_preview = run_dir / "apply_preview" / "apply_preview.json"
    assert apply_preview.exists()
    resumed = resume_ingest(vault=vault, operation_id=manifest.operation_id, from_step="validation")
    assert resumed.status == OperationStatus.drafted
    assert apply_preview.exists()
    assert len(status(vault, manifest.operation_id).provider_contexts) == 1


def test_resume_mock_provider_requires_current_fixture_dir(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=FIXTURE_ROOT / "mock", slug="mock-resume")
    with pytest.raises(Exception, match="requires fixture_dir"):
        resume_ingest(vault=vault, operation_id=manifest.operation_id, from_step="source_digest")


def test_provider_config_rejects_unknown_field_before_deleting_outputs(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=FIXTURE_ROOT / "mock", slug="secret")
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
    empty_fixture = tmp_path / "empty-fixture"
    empty_fixture.mkdir()
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=FIXTURE_ROOT / "mock", slug="provider-fail")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    config = read_yaml(vault / ".llmwiki" / "config.yaml")
    config["providers"]["default"] = {
        "spec": "mock:fixture",
        "fixture_dir": str(FIXTURE_ROOT / "mock"),
    }
    config["providers"]["source_digest"] = {
        "spec": "mock:fixture",
        "fixture_dir": empty_fixture.as_posix(),
    }
    write_yaml(vault / ".llmwiki" / "config.yaml", config)

    with pytest.raises(Exception, match="Mock fixture missing"):
        resume_ingest(vault=vault, operation_id=manifest.operation_id, from_step="source_digest")

    resumed = status(vault, manifest.operation_id)
    digest_step = [step for step in resumed.steps if step.name == "source_digest"][0]
    assert digest_step.status == StepStatus.failed
    assert digest_step.attempts[-1].provider_context_source == "resume_current_config"
    assert digest_step.attempts[-1].provider_spec == "mock:fixture"
    assert digest_step.attempts[-1].provider_record_id == "provider-context-002"
    assert resumed.provider_contexts[-1].providers["source_digest"].fixture_dir == empty_fixture.as_posix()
    assert (run_dir / "prepared_raw_review" / "approved_prepared.md").exists()
    assert not (run_dir / "draft_rendering").exists()
    assert not (run_dir / "apply_preview").exists()


def test_resume_can_rerun_from_step_without_mode_parameter(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=FIXTURE_ROOT / "mock", slug="resume")
    resumed = resume_ingest(vault=vault, operation_id=manifest.operation_id, from_step="validation")
    assert resumed.status == OperationStatus.drafted


def test_raw_and_artifact_drift_block_resume(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=FIXTURE_ROOT / "mock", slug="drift")
    raw.write_text(raw.read_text(encoding="utf-8") + "\nchanged", encoding="utf-8")
    with pytest.raises(VerifyError) as raw_error:
        resume_ingest(vault=vault, operation_id=manifest.operation_id)
    assert raw_error.value.result.issues[0].code == VerificationStatus.raw_changed

    vault2, raw2 = make_vault(tmp_path / "second")
    manifest2 = run_simplified_ingest(vault=vault2, raw_file=raw2, mock_fixture_dir=FIXTURE_ROOT / "mock", slug="artifact")
    digest = RunStore(vault2).run_dir(manifest2.operation_id) / "source_digest" / "source_digest.json"
    digest.write_text(digest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(VerifyError):
        resume_ingest(vault=vault2, operation_id=manifest2.operation_id)

    vault3, raw3 = make_vault(tmp_path / "third")
    manifest3 = run_simplified_ingest(vault=vault3, raw_file=raw3, mock_fixture_dir=FIXTURE_ROOT / "mock", slug="artifact-dir")
    digest_dir = RunStore(vault3).run_dir(manifest3.operation_id) / "source_digest" / "source_digest.json"
    digest_dir.unlink()
    digest_dir.mkdir()
    result = verify_run(vault3, status(vault3, manifest3.operation_id))
    assert not result.ok
    assert any(issue.path == "source_digest/source_digest.json" and "not a file" in issue.message for issue in result.issues)


def test_apply_preimage_repeat_and_applied_resume_block(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=FIXTURE_ROOT / "mock", slug="apply")
    target = vault / "wiki" / "concepts" / "Concept_知识编译工程骨架.md"
    target.write_text("user edit", encoding="utf-8")
    with pytest.raises(ApplyError):
        apply_operation(vault, manifest.operation_id)

    vault2, raw2 = make_vault(tmp_path / "clean")
    manifest2 = run_simplified_ingest(vault=vault2, raw_file=raw2, mock_fixture_dir=FIXTURE_ROOT / "mock", slug="apply")
    apply_operation(vault2, manifest2.operation_id)
    with pytest.raises(ApplyError):
        apply_operation(vault2, manifest2.operation_id)
    with pytest.raises(Exception):
        resume_ingest(vault=vault2, operation_id=manifest2.operation_id)


def test_applied_operation_rejects_review_mutations(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=FIXTURE_ROOT / "mock", slug="review-immutable")
    apply_operation(vault, manifest.operation_id)

    with pytest.raises(PipelineError, match="Applied operations are immutable"):
        approve_review(vault, manifest.operation_id, "merge_plan_review")
    with pytest.raises(PipelineError, match="Applied operations are immutable"):
        revise_review(vault, manifest.operation_id, "draft_review")


def test_review_approve_requires_awaiting_review_state(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=FIXTURE_ROOT / "mock", slug="review-state")

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

    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=fixture_dir, slug="related-resolve")
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


def test_related_renderer_filters_and_caps_candidates() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-CAND001",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Current.md",
        display_title="Current",
        page_type="concept",
        new_understanding="当前主题。",
        section_plans={"summary": "Summary"},
        related_pages=[
            pipeline_module.RelatedPageRef(
                target_path="concepts/Concept_A.md",
                display_title="A",
                source="wiki_context",
                reason="A 与当前主题最相关。",
            ),
            pipeline_module.RelatedPageRef(
                target_path="concepts/Concept_A.md",
                display_title="A duplicate",
                source="wiki_context",
                reason="重复链接。",
            ),
            pipeline_module.RelatedPageRef(
                target_path="concepts/Concept_B.md",
                display_title="B",
                source="source_digest",
                reason="B 是同源互补主题。",
            ),
            pipeline_module.RelatedPageRef(
                target_path="concepts/Concept_C.md",
                display_title="C",
                source="wiki_context",
                reason="C 是召回到的补充背景。",
            ),
            pipeline_module.RelatedPageRef(
                target_path="concepts/Concept_D.md",
                display_title="D",
                source="wiki_context",
                reason="D 会因为 cap 被截断。",
            ),
            pipeline_module.RelatedPageRef(
                target_path="sources/Source_Bad.md",
                display_title="Source",
                source="wiki_context",
                reason="source page 不应进入 Related。",
            ),
        ],
        reason="test",
    )
    report: list[pipeline_module.RelatedCandidateReport] = []
    rendered = related_pages_module.render_related_pages(
        item,
        existing_entry=pipeline_module.WikiContextEntry(
            path="wiki/concepts/Concept_Current.md",
            expected_state="present",
            preimage_sha256="old",
            content="## 相关页面\n\n- [[concepts/Concept_Old|旧链接]]：旧页中仍强相关的链接。\n",
        ),
        report_list=report,
        known_paths={
            "concepts/Concept_Old.md",
            "concepts/Concept_A.md",
            "concepts/Concept_B.md",
            "concepts/Concept_C.md",
            "concepts/Concept_D.md",
        },
    )

    assert rendered.count("[[") == 3
    assert "[[concepts/Concept_Old|旧链接]]" in rendered
    assert "Concept_D" not in rendered
    assert "sources/" not in rendered
    assert any(row.decision == "cutoff" and row.target_path == "concepts/Concept_D.md" for row in report)
    assert any(row.decision == "filtered" and row.reject_reason == "duplicate" for row in report)
    assert any(row.decision == "filtered" and row.reject_reason == "unknown_path" for row in report)


def test_related_renderer_scrubs_internal_candidate_ids_from_public_reason() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-CURRENT",
        source_basis=SourceBasis(source_candidate_ids=["C001"]),
        action="create",
        canonical_target_path="concepts/Concept_Current.md",
        display_title="Current",
        page_type="concept",
        new_understanding="当前主题。",
        section_plans={"summary": "Summary"},
        related_pages=[
            pipeline_module.RelatedPageRef(
                target_path="concepts/Concept_A.md",
                display_title="A",
                source="source_digest",
                reason="来源摘要把 `E001` 标记为相关候选，本页与该候选属于同一材料中的互补主题。",
            ),
            pipeline_module.RelatedPageRef(
                target_path="concepts/Concept_B.md",
                display_title="B",
                source="source_digest",
                reason="source digest says CON-001 is related.",
            ),
            pipeline_module.RelatedPageRef(
                target_path="concepts/Concept_C.md",
                display_title="C",
                source="source_digest",
                reason="AGG-concepts-demo 与候选页面 ent-001 互补。",
            ),
        ],
        reason="test",
    )

    rendered = related_pages_module.render_related_pages(
        item,
        known_paths={"concepts/Concept_A.md", "concepts/Concept_B.md", "concepts/Concept_C.md"},
    )

    assert "[[concepts/Concept_A|A]]" in rendered
    assert "[[concepts/Concept_B|B]]" in rendered
    assert "[[concepts/Concept_C|C]]" in rendered
    assert "同属本次材料中的互补主题" in rendered
    assert "E001" not in rendered
    assert "CON-001" not in rendered
    assert "AGG-concepts-demo" not in rendered
    assert "ent-001" not in rendered
    assert "source digest" not in rendered
    assert "候选页面" not in rendered


def test_draft_rendering_model_schema_uses_current_page_fields() -> None:
    schema = pipeline_module.DraftRenderingArtifact.model_json_schema()
    assert_draft_rendering_schema_page_fields(schema)

    page = read_json(FIXTURE_ROOT / "mock" / "draft_rendering.json")["pages"][0]
    page["unexpected_page_field"] = "This field is not part of the current draft page contract."
    with pytest.raises(Exception, match="unexpected_page_field"):
        pipeline_module.DraftRenderingArtifact.model_validate({"schema_version": "draft_rendering.v3", "pages": [page]})


def test_draft_rendering_model_schema_rejects_unexpected_source_coverage_field() -> None:
    page = read_json(FIXTURE_ROOT / "mock" / "draft_rendering.json")["pages"][0]
    page.pop("source_coverage_notes", None)
    page["unexpected_source_coverage_field"] = "This field is not part of the current draft page contract."

    with pytest.raises(Exception, match="unexpected_source_coverage_field"):
        pipeline_module.DraftRenderingArtifact.model_validate({"schema_version": "draft_rendering.v3", "pages": [page]})


def test_draft_rendering_model_output_is_canonicalized_for_internal_pipeline() -> None:
    page = read_json(FIXTURE_ROOT / "mock" / "draft_rendering.json")["pages"][0]
    page["page_plan_id"] = "PP-QWEN"
    page["canonical_target_path"] = "entities/Entity_Qwen-Agent.md"
    page["source_coverage_notes"] = "严格按照源内容，无额外添加。"
    draft = pipeline_module.DraftRenderingArtifact.model_validate(
        {"schema_version": "draft_rendering.v3", "pages": [page]}
    )

    finalized = draft_validation_module.canonicalize_draft_artifact(draft, qwen_related_block_plan())
    first_page = finalized.pages[0]

    assert first_page.summary == page["summary"]
    assert first_page.body_markdown == page["body_markdown"]
    assert "### 例子" in first_page.body_markdown
    assert first_page.source_coverage_notes == "严格按照源内容，无额外添加。"


def test_validate_draft_rendering_accepts_freeform_body_markdown() -> None:
    plan = qwen_related_block_plan()
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-QWEN",
                action="create",
                canonical_target_path="entities/Entity_Qwen-Agent.md",
                summary="Qwen-Agent 是 Agent 开发框架。",
                body_markdown=(
                    "### 能力边界\n\n"
                    "Qwen-Agent 把工具调用、规划和记忆能力组织成可运行的 Agent 框架。"
                    "这不是单纯罗列功能，而是强调开发者可以围绕具体任务把模型能力和外部工具组合起来。\n\n"
                    "### 使用场景\n\n"
                    "例如，团队可以用它搭建一个处理内部知识查询的助手，并用占位符描述用户输入。"
                ),
                open_questions="- 待补来源：不同工具组合方式的可靠性如何验证？",
                change_summary="创建 Qwen-Agent 页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    draft_validation_module.validate_draft_rendering(draft, plan, language="zh-CN")


def test_finalize_draft_rendering_preserves_freeform_body_markdown() -> None:
    plan = qwen_related_block_plan()
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/entities/Entity_Qwen-Agent.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-QWEN",
                action="create",
                canonical_target_path="entities/Entity_Qwen-Agent.md",
                summary="Qwen-Agent 是 Agent 开发框架。",
                body_markdown=(
                    "### 自定义核心\n\n"
                    "模型自己写的核心判断。\n\n"
                    "### 例子\n\n"
                    "这个例子由自由正文承载。\n\n"
                    "### 价值点\n\n"
                    "这个价值判断也由自由正文承载。"
                ),
                change_summary="创建 Qwen-Agent 页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    finalized = pipeline_module.finalize_draft_rendering(draft, plan, snapshot)
    body = finalized.pages[0].body_markdown

    assert "### 自定义核心" in body
    assert "这个例子由自由正文承载。" in body
    assert "这个价值判断也由自由正文承载。" in body


def test_validate_draft_rendering_strips_system_heading_inside_body_markdown() -> None:
    plan = qwen_related_block_plan()
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-QWEN",
                action="create",
                canonical_target_path="entities/Entity_Qwen-Agent.md",
                summary="Qwen-Agent 是 Agent 开发框架。",
                body_markdown=(
                    "### 能力边界\n\n"
                    "Qwen-Agent 支持工具使用、规划和记忆能力。\n\n"
                    "## 相关页面\n\n"
                    "- [[entities/Entity_Qwen-Agent|Qwen-Agent]]"
                ),
                change_summary="创建 Qwen-Agent 页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    canonical = draft_validation_module.canonicalize_draft_artifact(draft, plan)

    draft_validation_module.validate_draft_rendering(canonical, plan, language="zh-CN")
    assert "相关页面" not in canonical.pages[0].body_markdown
    assert "Entity_Qwen-Agent" not in canonical.pages[0].body_markdown


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

    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=fixture_dir, slug="blocked")
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
        run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=fixture_dir, slug="duplicate-target")


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
        run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=fixture_dir, slug="source-graph")


def test_m3_update_target_auto_approves_when_no_review_risks(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    target = vault / "wiki" / "concepts" / "Concept_知识编译工程骨架.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("existing knowledge\n", encoding="utf-8")

    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=FIXTURE_ROOT / "mock", slug="existing")

    assert target.read_text(encoding="utf-8") == "existing knowledge\n"
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    plan = read_json(run_dir / "wiki_merge_planning" / "wiki_merge_plan.json")
    update_items = [item for item in plan["items"] if item["canonical_target_path"] == "concepts/Concept_知识编译工程骨架.md"]
    assert update_items
    assert update_items[0]["action"] == "update"
    assert update_items[0]["matched_page"] == "concepts/Concept_知识编译工程骨架.md"
    manifest = status(vault, manifest.operation_id)
    assert manifest.status == OperationStatus.drafted
    assert not [step for step in manifest.steps if step.status == StepStatus.awaiting_review]
    assert (run_dir / "draft_rendering" / "draft_write_manifest.json").exists()
    assert (run_dir / "draft_review" / "approved_write_manifest.json").exists()
    approval = read_json(run_dir / "draft_review" / "draft_approval.json")
    assert approval["schema_version"] == "draft_review.v2"
    assert approval["decision"] == "approved"
    assert "review_mode" not in approval
    assert approval["auto_approved"] is True
    assert "update operation 未发现" in approval["notes"]
    assert (run_dir / "apply_preview").exists()
    preview = read_json(run_dir / "apply_preview" / "apply_preview.json")
    assert preview["has_updates"] is True
    assert preview["requires_draft_review"] is False
    draft_step = [step for step in manifest.steps if step.name == "draft_rendering"][0]
    diff_refs = [ref for ref in draft_step.outputs if ref.relative_path.endswith(".diff")]
    assert diff_refs
    assert {ref.kind for ref in diff_refs} == {"diff"}


def test_update_preserves_and_reports_existing_summary_detail_and_index_title(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    target = vault / "wiki" / "concepts" / "Concept_知识编译工程骨架.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "---\n"
        "llmwiki_type: concept\n"
        "title: 旧工程骨架标题\n"
        "aliases: []\n"
        "summary: 旧摘要用于索引。\n"
        "created: 2026-01-01\n"
        "updated: 2026-01-02\n"
        "---\n\n"
        "# 旧工程骨架标题\n\n"
        "## 摘要\n\n"
        "旧摘要正文应该参与 update 审计。\n\n"
        "## 详情\n\n"
        "旧详情正文应该参与 update 审计。\n\n"
        "## 相关页面\n\n"
        "- [[concepts/Concept_旧相关|旧相关]]：旧链接仍然重要。\n\n"
        "## 矛盾与未决问题\n\n"
        "暂无矛盾与未决问题记录。\n",
        encoding="utf-8",
    )
    related = vault / "wiki" / "concepts" / "Concept_旧相关.md"
    related.write_text(
        "---\n"
        "llmwiki_type: concept\n"
        "title: 旧相关\n"
        "aliases: []\n"
        "summary: 旧相关摘要。\n"
        "created: 2026-01-01\n"
        "updated: 2026-01-02\n"
        "---\n\n"
        "# 旧相关\n",
        encoding="utf-8",
    )
    fixture_dir = tmp_path / "update-core-fixture"
    fixture_dir.mkdir()
    for name in ["raw_prepare.json", "source_digest.json", "candidate_resolution.json", "wiki_merge_planning.json", "draft_rendering.json"]:
        data = read_json(FIXTURE_ROOT / "mock" / name)
        if name == "wiki_merge_planning.json":
            data["items"][0]["action"] = "update"
            data["items"][0]["matched_page"] = "concepts/Concept_知识编译工程骨架.md"
            data["items"][0]["canonical_target_path"] = "concepts/Concept_知识编译工程骨架.md"
        if name == "draft_rendering.json":
            data["pages"][0]["change_summary"] = "补充并澄清旧工程骨架页面，将新材料中的 MVP 编译重点整合进完整替换草稿。"
        write_json(fixture_dir / name, data)

    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=fixture_dir, slug="update-core")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    draft_path = run_dir / "draft_rendering" / "draft_pages" / "concepts" / "Concept_知识编译工程骨架.md"
    draft_text = draft_path.read_text(encoding="utf-8")
    update_report = read_json(run_dir / "draft_rendering" / "update_merge_report.json")
    reinforcement_report = read_json(run_dir / "draft_rendering" / "update_preservation_reinforcement_report.json")
    repair_report = read_json(run_dir / "draft_rendering" / "structured_repair_report.json")
    report_markdown = (run_dir / "draft_rendering" / "update_merge_report.md").read_text(encoding="utf-8")
    review_prompt = (run_dir / "draft_review" / "review_prompt.md").read_text(encoding="utf-8")
    index_text = (run_dir / "draft_rendering" / "draft_pages" / "index.md").read_text(encoding="utf-8")

    assert "# 旧工程骨架标题" in draft_text
    assert "旧页保留观察" not in draft_text
    assert "与旧页架构视角相衔接" in draft_text
    assert "旧摘要正文应该参与 update 审计。" in draft_text
    assert "知识编译工程骨架强调" in draft_text
    assert "旧详情正文应该参与 update 审计。" in draft_text
    assert "这个判断把 MVP 的重点" in draft_text
    sections = {section["section_key"]: section for section in update_report["pages"][0]["sections"]}
    assert "旧摘要正文应该参与 update 审计。" in sections["summary"]["retained"]
    assert sections["summary"]["preserved_old"] == []
    assert sections["summary"]["needs_manual_resolution"] is False
    assert sections["summary"]["removed"] == []
    assert any("知识编译工程骨架强调" in item for item in sections["summary"]["added"])
    assert any("与旧页架构视角相衔接" in item for item in sections["summary"]["added"])
    assert "旧详情正文应该参与 update 审计。" in sections["core_content"]["retained"]
    assert sections["core_content"]["preserved_old"] == []
    assert sections["core_content"]["needs_manual_resolution"] is False
    assert sections["core_content"]["removed"] == []
    assert any("这个判断把 MVP 的重点" in item for item in sections["core_content"]["added"])
    assert any("与旧页架构视角相衔接" in item for item in sections["core_content"]["added"])
    assert reinforcement_report["changed"] is True
    assert reinforcement_report["reinforced_section_count"] == 2
    assert repair_report["final_outcome"] == "success"
    assert repair_report["repair_attempted"] is True
    attempt_issue_codes = {
        issue["issue_code"]
        for attempt in repair_report["attempts"]
        for issue in attempt["issues"]
    }
    assert "old_knowledge_not_absorbed" in attempt_issue_codes
    previews = [
        section["reinforcement_preview"]
        for page in reinforcement_report["pages"]
        for section in page["sections"]
    ]
    assert any("旧摘要正文应该参与 update 审计。" in preview for preview in previews)
    assert any("旧详情正文应该参与 update 审计。" in preview for preview in previews)
    assert "| 段落 | 保留 | 新增 | 删除 | 旧页保留观察 | 需人工消化 | 原因 |" in report_markdown
    assert "模型完整重写后未显式吸收该旧段落" not in report_markdown
    assert "Removal Reason" not in report_markdown
    assert "旧页保留观察需人工消化：否" in review_prompt
    assert "本地旧知识补强已执行：是（2 段旧页知识已由系统本地补强并记录）" in review_prompt
    assert "## 本地旧知识补强提示" in review_prompt
    assert "## 旧页保留观察警示" not in review_prompt
    manifest = status(vault, manifest.operation_id)
    assert not [step for step in manifest.steps if step.status == StepStatus.awaiting_review]
    approval = read_json(run_dir / "draft_review" / "draft_approval.json")
    assert approval["decision"] == "approved"
    assert approval["auto_approved"] is True
    assert "本地旧知识补强已写入审计报告" in approval["notes"]
    assert (run_dir / "draft_rendering" / "update_preservation_pack.json").exists()
    assert (run_dir / "draft_rendering" / "update_preservation_pack.md").exists()
    assert (run_dir / "draft_rendering" / "update_preservation_reinforcement_report.json").exists()
    assert (run_dir / "draft_rendering" / "update_preservation_reinforcement_report.md").exists()
    preservation_pack = read_json(run_dir / "draft_rendering" / "update_preservation_pack.json")
    assert preservation_pack["schema_version"] == "update_preservation_pack.v1"
    assert preservation_pack["pages"][0]["sections"]
    draft_step = [step for step in manifest.steps if step.name == "draft_rendering"][0]
    preservation_ref = [
        ref
        for ref in draft_step.outputs
        if ref.relative_path == "draft_rendering/update_preservation_pack.json"
    ][0]
    assert preservation_ref.schema_version == "update_preservation_pack.v1"
    reinforcement_ref = [
        ref
        for ref in draft_step.outputs
        if ref.relative_path == "draft_rendering/update_preservation_reinforcement_report.json"
    ][0]
    assert reinforcement_ref.schema_version == "update_preservation_reinforcement_report.v1"
    assert "| 旧工程骨架标题 | [[concepts/Concept_知识编译工程骨架]] |" in index_text


def test_merge_update_section_semantic_absorption_avoids_old_observation() -> None:
    old = "Managed Agents / harness 视角强调安全边界、隔离容器、工具权限和会话对象。"
    new = "新版页面补充产品视角，同时保留 Managed Agents、harness、安全边界、隔离容器、工具权限和会话对象这些架构约束。"

    merged, change = section_merge_module.merge_update_section("detail", old, new)

    assert "旧页保留观察" not in merged
    assert change.retained == [old]
    assert change.preserved_old == []
    assert change.needs_manual_resolution is False
    assert "关键短语" in change.removal_reason


def test_merge_update_section_cross_section_absorption_avoids_old_observation() -> None:
    old = "Managed Agents / harness 视角强调安全边界、隔离容器、工具权限和会话对象。"
    new_summary = "新版摘要补充 Claude Code 的产品发布速度和团队协作视角。"
    new_detail = (
        "详情保留旧页架构判断：Claude Code 仍处在 Managed Agents / harness 视角中，"
        "模型负责推理和规划，工具执行通过工具权限、隔离容器和会话对象来承接。"
    )

    merged, change = section_merge_module.merge_update_section(
        "summary",
        old,
        new_summary,
        absorption_context=f"{new_summary}\n\n{new_detail}",
    )

    assert "旧页保留观察" not in merged
    assert change.retained == [old]
    assert change.preserved_old == []
    assert change.needs_manual_resolution is False
    assert "其他章节吸收旧段落" in change.removal_reason


def test_draft_review_prompt_points_to_batch_reinforcement_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    draft_root = run_dir / "draft_rendering"
    draft_root.mkdir(parents=True)
    write_json(
        draft_root / "draft_rendering_batch_report.json",
        {
            "schema_version": "draft_rendering_batch_report.v1",
            "batches": [
                {
                    "batch_id": "batch-001",
                    "reinforced_section_count": 2,
                }
            ],
        },
    )
    (draft_root / "draft_rendering_batch_report.md").write_text("# Draft Rendering 分批报告\n", encoding="utf-8")

    prompt = draft_reviewing_module.render_draft_review_prompt(
        run_dir,
        pipeline_module.DraftWriteManifest(targets=[]),
    )

    assert "本地旧知识补强已执行：是（2 段旧页知识已由系统本地补强并记录）" in prompt
    assert "`draft_rendering/draft_rendering_batch_report.md`" in prompt
    assert "`draft_rendering/update_preservation_reinforcement_report.md`" not in prompt


def test_draft_aux_report_writes_only_when_active(tmp_path: Path) -> None:
    output_dir = tmp_path / "draft_rendering"
    output_dir.mkdir()

    skipped = pipeline_module.write_draft_aux_report_if_active(
        output_dir=output_dir,
        stem="grounding_paraphrase_rewrite_report",
        report={
            "schema_version": "grounding_paraphrase_rewrite_report.v1",
            "changed": False,
            "rewrite_count": 0,
            "pages": [],
        },
        renderer=draft_grounding.render_grounding_paraphrase_rewrite_report,
        count_keys=["rewrite_count"],
    )

    assert skipped is None
    assert not (output_dir / "grounding_paraphrase_rewrite_report.json").exists()
    assert not (output_dir / "grounding_paraphrase_rewrite_report.md").exists()

    written = pipeline_module.write_draft_aux_report_if_active(
        output_dir=output_dir,
        stem="grounding_paraphrase_rewrite_report",
        report={
            "schema_version": "grounding_paraphrase_rewrite_report.v1",
            "changed": False,
            "rewrite_count": 1,
            "pages": [
                {
                    "page_plan_id": "PP-1",
                    "target_path": "concepts/Concept_Test.md",
                    "fields": [
                        {
                            "field": "summary",
                            "rewrites": [
                                {
                                    "original_quote": "旧引号短语",
                                    "replacement": "改成来源内表述",
                                    "source_sentence": "来源里有这一句。",
                                }
                            ],
                        }
                    ],
                }
            ],
        },
        renderer=draft_grounding.render_grounding_paraphrase_rewrite_report,
        count_keys=["rewrite_count"],
    )

    assert written is not None
    assert (output_dir / "grounding_paraphrase_rewrite_report.json").exists()
    assert (output_dir / "grounding_paraphrase_rewrite_report.md").exists()
    markdown = (output_dir / "grounding_paraphrase_rewrite_report.md").read_text(encoding="utf-8")
    assert "旧引号短语" in markdown
    assert "改成来源内表述" in markdown
    assert "summary" in markdown


def test_merge_update_section_absorbs_live_brain_hands_summary_across_sections() -> None:
    old = (
        "Claude Code is useful because it shows how a Managed Agents system can separate the model "
        "brain from execution hands while preserving a focused coding experience."
    )
    new_summary = (
        "Claude Code 是 Anthropic 推出的编程辅助产品，最初作为 Managed Agents 框架下的适配层"
        "（harness）提供工具权限、沙盒执行、仓库上下文和持久会话状态。"
    )
    new_detail = (
        "从原有 Managed Agents 适配框架视角看，Claude Code 不仅是聊天界面，而是一个通过"
        "工具权限、沙盒执行、仓库上下文和持久会话状态来路由模型意图的适配层（harness）。"
    )

    concepts = update_preservation_module.update_preservation_concepts(old)
    concept_names = {str(concept["name"]) for concept in concepts}
    merged, change = section_merge_module.merge_update_section(
        "summary",
        old,
        new_summary,
        absorption_context=f"{new_summary}\n\n{new_detail}",
    )

    assert {"managed_agents", "brain_hands_decoupling"} <= concept_names
    assert "旧页保留观察" not in merged
    assert change.retained == [old]
    assert change.preserved_old == []
    assert change.needs_manual_resolution is False
    assert "Managed Agents / 托管智能体" in change.removal_reason
    assert "大脑与双手解耦" in change.removal_reason


def test_merge_update_section_cross_section_absorption_still_requires_core_concepts() -> None:
    old = "Managed Agents / harness 视角强调安全边界、隔离容器、工具权限和会话对象。"
    new_summary = "新版摘要补充 Claude Code 的产品发布速度和团队协作视角。"
    shallow_context = "新版详情只顺带提到 Managed Agents 和 harness。"

    merged, change = section_merge_module.merge_update_section(
        "summary",
        old,
        new_summary,
        absorption_context=f"{new_summary}\n\n{shallow_context}",
    )

    assert "旧页保留观察" in merged
    assert old in change.preserved_old
    assert change.needs_manual_resolution is True


def test_merge_update_section_live_brain_hands_summary_rejects_shallow_context() -> None:
    old = (
        "Claude Code is useful because it shows how a Managed Agents system can separate the model "
        "brain from execution hands while preserving a focused coding experience."
    )
    new_summary = "新版摘要补充 Claude Code 的产品发布速度和团队协作视角。"
    shallow_context = "新版详情只顺带提到 Managed Agents、harness 和模型意图。"

    merged, change = section_merge_module.merge_update_section(
        "summary",
        old,
        new_summary,
        absorption_context=f"{new_summary}\n\n{shallow_context}",
    )

    assert "旧页保留观察" in merged
    assert old in change.preserved_old
    assert change.needs_manual_resolution is True


def test_merge_update_section_live_brain_hands_summary_rejects_broad_intent_phrase() -> None:
    old = (
        "Claude Code is useful because it shows how a Managed Agents system can separate the model "
        "brain from execution hands while preserving a focused coding experience."
    )
    new_summary = "新版摘要补充 Claude Code 的产品发布速度和团队协作视角。"
    broad_context = "新版详情提到 Managed Agents 中模型意图通过更清晰的产品界面表达。"

    merged, change = section_merge_module.merge_update_section(
        "summary",
        old,
        new_summary,
        absorption_context=f"{new_summary}\n\n{broad_context}",
    )

    assert "旧页保留观察" in merged
    assert old in change.preserved_old
    assert change.needs_manual_resolution is True


def test_merge_update_section_requires_old_concept_obligations_not_shallow_terms() -> None:
    old = "Managed Agents / harness 视角强调安全边界、隔离容器、工具权限和会话对象。"
    new = "新版页面补充产品视角，只顺带提到 Managed Agents 和 harness。"

    merged, change = section_merge_module.merge_update_section("detail", old, new)

    assert "旧页保留观察" in merged
    assert old in change.preserved_old
    assert change.needs_manual_resolution is True


def test_merge_update_section_does_not_preserve_non_core_old_section() -> None:
    old = "来源在文章末尾提及：Claude Code is an excellent harness。"
    new = "新版例子讨论 CLI、桌面版和 Cowork 的使用场景。"

    merged, change = section_merge_module.merge_update_section("examples", old, new)

    assert "旧页保留观察" not in merged
    assert change.retained == []
    assert change.removed == [old]
    assert change.preserved_old == []
    assert change.needs_manual_resolution is False
    assert "不属于 update preservation 核心义务" in change.removal_reason


def test_merge_update_section_does_not_cross_absorb_non_core_old_section() -> None:
    old = "旧例子强调 harness 检查工具权限，并把命令交给隔离容器执行。"
    new = "新版例子讨论 CLI、桌面版和 Cowork 的使用场景。"
    context = "详情保留 harness、工具权限和隔离容器等架构视角。"

    merged, change = section_merge_module.merge_update_section(
        "examples",
        old,
        new,
        absorption_context=f"{new}\n\n{context}",
    )

    assert "旧页保留观察" not in merged
    assert change.retained == []
    assert change.removed == [old]
    assert change.preserved_old == []
    assert change.needs_manual_resolution is False


def test_merge_update_section_open_questions_unions_old_questions() -> None:
    old = (
        "- 如何量化记忆召回的置信度？Agent Memory API 是否提供？\n"
        "- 是否存在记忆回滚或修正机制以应对错误记忆？"
    )
    new = "- 如何设计用户确认交互？"

    merged, change = section_merge_module.merge_update_section("open_questions", old, new)

    assert merged.splitlines() == [
        "- 如何设计用户确认交互？",
        "- 如何量化记忆召回的置信度？Agent Memory API 是否提供？",
        "- 是否存在记忆回滚或修正机制以应对错误记忆？",
    ]
    assert change.retained == [
        "如何量化记忆召回的置信度？Agent Memory API 是否提供？",
        "是否存在记忆回滚或修正机制以应对错误记忆？",
    ]
    assert change.removed == []
    assert change.preserved_old == []
    assert change.needs_manual_resolution is False
    assert "union/dedupe" in change.removal_reason


def test_merge_update_section_open_questions_dedupes_semantic_repeats() -> None:
    old = "- AGI后PM是否必要？\n- 记忆回滚机制如何设计？"
    new = "- AGI到来后PM角色是否会消失？"

    merged, change = section_merge_module.merge_update_section("open_questions", old, new)

    assert "AGI后PM是否必要" not in merged
    assert "AGI到来后PM角色是否会消失" in merged
    assert "记忆回滚机制如何设计" in merged
    assert change.retained == ["记忆回滚机制如何设计？"]


def test_merge_update_section_open_questions_filters_placeholders_and_low_signal_old_questions() -> None:
    old = "- 暂无矛盾与未决问题记录。\n- 待补来源：需要继续确认。"
    new = "- 如何设计用户确认交互？"

    merged, change = section_merge_module.merge_update_section("open_questions", old, new)

    assert merged == "- 如何设计用户确认交互？"
    assert change.retained == []
    assert change.removed == []
    assert change.needs_manual_resolution is False


@pytest.mark.parametrize("new", ["", "暂无矛盾与未决问题记录。"])
def test_merge_update_section_open_questions_does_not_fallback_to_low_signal_old(new: str) -> None:
    old = "- 待补来源：需要继续确认。"

    merged, change = section_merge_module.merge_update_section("open_questions", old, new)

    assert merged == "暂无矛盾与未决问题记录。"
    assert "待补来源：需要继续确认" not in merged
    assert change.retained == []
    assert change.removed == []
    assert change.needs_manual_resolution is False


def test_merge_update_section_additional_notes_preserves_high_signal_boundary_note() -> None:
    old = "文档提醒：回忆的记忆应视为有帮助的上下文而非绝对真实，重要决定需要用户确认。"
    new = "本页面补充 Redis 等实现方式。"

    merged, change = section_merge_module.merge_update_section("additional_notes", old, new)

    assert "旧页补充观察" in merged
    assert old in merged
    assert change.retained == [old]
    assert change.preserved_old == [old]
    assert change.removed == []
    assert change.needs_manual_resolution is False
    assert "高信号旧补充观察" in change.removal_reason


def test_merge_update_section_additional_notes_keeps_only_high_signal_units() -> None:
    high = "重要决定必须由用户确认，不能只依赖召回记忆。"
    low = "本页面可与 Redis 页面联动阅读。"
    old = f"- {low}\n- {high}"
    new = "本页面补充通用记忆架构。"

    merged, change = section_merge_module.merge_update_section("additional_notes", old, new)

    assert high in merged
    assert low not in merged
    assert change.retained == [high]
    assert change.preserved_old == [high]
    assert change.removed == [low]
    assert change.needs_manual_resolution is False


def test_merge_update_section_additional_notes_does_not_preserve_low_signal_note() -> None:
    old = "本页面从通用概念出发，可与 Redis 页面联动阅读。"
    new = "本页面补充通用记忆架构。"

    merged, change = section_merge_module.merge_update_section("additional_notes", old, new)

    assert merged == new
    assert change.retained == []
    assert change.preserved_old == []
    assert change.removed == [old]
    assert change.needs_manual_resolution is False
    assert "不属于 update preservation 核心义务" in change.removal_reason


def test_merge_update_section_additional_notes_does_not_duplicate_absorbed_note() -> None:
    old = "重要决定需要用户确认，不能只依赖召回记忆。"
    new = f"本页延续旧边界：{old}"

    merged, change = section_merge_module.merge_update_section("additional_notes", old, new)

    assert merged == new
    assert "旧页补充观察" not in merged
    assert change.retained == [old]
    assert change.preserved_old == []
    assert change.removed == []


def test_merge_update_section_additional_notes_does_not_duplicate_paraphrased_boundary() -> None:
    old = "重要决定需要用户确认，不能只依赖召回记忆。"
    new = "召回的记忆只能作为辅助上下文，重要决策仍应由用户确认。"

    merged, change = section_merge_module.merge_update_section("additional_notes", old, new)

    assert merged == new
    assert "旧页补充观察" not in merged
    assert change.retained == [old]
    assert change.preserved_old == []
    assert change.removed == []


def test_merge_update_section_additional_notes_scattered_signals_do_not_absorb_boundary() -> None:
    old = "重要决定需要用户确认，不能只依赖召回记忆。"
    new = "用户确认邮箱后才能登录。产品决策由团队流程处理。记忆召回作为上下文用于推荐。"

    merged, change = section_merge_module.merge_update_section("additional_notes", old, new)

    assert "旧页补充观察" in merged
    assert old in merged
    assert change.preserved_old == [old]


def test_merge_update_section_additional_notes_login_confirmation_does_not_absorb_boundary() -> None:
    old = "重要决定需要用户确认，不能只依赖召回记忆。"
    new = "系统决定需要用户确认邮箱后才能登录，召回的记忆只能作为辅助上下文。"

    merged, change = section_merge_module.merge_update_section("additional_notes", old, new)

    assert "旧页补充观察" in merged
    assert old in merged
    assert change.preserved_old == [old]


def test_merge_update_section_additional_notes_negative_memory_context_does_not_absorb_boundary() -> None:
    old = "重要决定需要用户确认，不能只依赖召回记忆。"
    new = "重要决策仍应由用户确认，召回的记忆不是辅助上下文。"

    merged, change = section_merge_module.merge_update_section("additional_notes", old, new)

    assert "旧页补充观察" in merged
    assert old in merged
    assert change.preserved_old == [old]


def test_merge_update_section_additional_notes_does_not_keep_question_like_note() -> None:
    old = "是否需要为召回记忆设计用户确认机制？"
    new = "本页面补充通用记忆架构。"

    merged, change = section_merge_module.merge_update_section("additional_notes", old, new)

    assert merged == new
    assert change.retained == []
    assert change.preserved_old == []
    assert change.removed == [old]
    assert change.needs_manual_resolution is False


def test_merge_update_section_additional_notes_does_not_keep_confirm_whether_note() -> None:
    old = "需要确认是否存在记忆回滚或修正机制。"
    new = "本页面补充通用记忆架构。"

    merged, change = section_merge_module.merge_update_section("additional_notes", old, new)

    assert merged == new
    assert change.retained == []
    assert change.preserved_old == []
    assert change.removed == [old]


def test_merge_update_section_additional_notes_does_not_keep_generic_must_note() -> None:
    old = "本页面必须与 Redis 页面联动阅读。"
    new = "本页面补充通用记忆架构。"

    merged, change = section_merge_module.merge_update_section("additional_notes", old, new)

    assert merged == new
    assert change.retained == []
    assert change.preserved_old == []
    assert change.removed == [old]


def test_merge_update_section_additional_notes_does_not_reintroduce_superseded_note() -> None:
    old = "重要决定需要用户确认，不能只依赖召回记忆。"
    new = "新版系统已改为自动校验召回记忆，重要决定不再需要用户确认。"

    merged, change = section_merge_module.merge_update_section("additional_notes", old, new)

    assert merged == new
    assert "旧页补充观察" not in merged
    assert change.retained == []
    assert change.preserved_old == []
    assert change.removed == [old]


def test_merge_update_section_additional_notes_user_confirmation_deprecated_is_superseded() -> None:
    old = "重要决定需要用户确认，不能只依赖召回记忆。"
    new = "用户确认机制已废弃，系统改为自动校验。"

    merged, change = section_merge_module.merge_update_section("additional_notes", old, new)

    assert merged == new
    assert "旧页补充观察" not in merged
    assert change.preserved_old == []
    assert change.removed == [old]


def test_merge_update_section_additional_notes_unrelated_no_longer_needed_clause_preserves_note() -> None:
    old = "重要决定需要用户确认，不能只依赖召回记忆。"
    new = "重要决定需要用户确认，但旧 API 不再需要。"

    merged, change = section_merge_module.merge_update_section("additional_notes", old, new)

    assert "旧页补充观察" in merged
    assert old in merged
    assert change.preserved_old == [old]


def test_merge_update_section_additional_notes_unrelated_approval_change_preserves_note() -> None:
    old = "重要决定需要用户确认，不能只依赖召回记忆。"
    new = "重要决定不再需要额外审批，但仍需要用户确认。"

    merged, change = section_merge_module.merge_update_section("additional_notes", old, new)

    assert "旧页补充观察" in merged
    assert old in merged
    assert change.preserved_old == [old]


def test_merge_update_section_additional_notes_superseded_anchor_not_hidden_by_preserved_anchor() -> None:
    old = "重要决定需要用户确认，不能只依赖召回记忆。"
    new = "重要决定不再需要用户确认，但召回记忆仍可作为上下文。"

    merged, change = section_merge_module.merge_update_section("additional_notes", old, new)

    assert merged == new
    assert "旧页补充观察" not in merged
    assert change.preserved_old == []
    assert change.removed == [old]


def test_merge_update_section_additional_notes_negative_confirmation_is_superseded_not_absorbed() -> None:
    old = "重要决定需要用户确认，不能只依赖召回记忆。"
    new = "重要决定不需要用户确认，召回的记忆只能作为辅助上下文。"

    merged, change = section_merge_module.merge_update_section("additional_notes", old, new)

    assert merged == new
    assert "旧页补充观察" not in merged
    assert change.retained == []
    assert change.preserved_old == []
    assert change.removed == [old]


def test_merge_update_section_additional_notes_strips_legacy_label_before_preserving() -> None:
    note = "文档提醒：回忆的记忆应视为有帮助的上下文而非绝对真实，重要决定需要用户确认。"
    old = f"旧页补充观察：旧页补充观察：{note}"
    new = "本页面补充 Redis 等实现方式。"

    merged, change = section_merge_module.merge_update_section("additional_notes", old, new)

    assert merged.count("旧页补充观察") == 1
    assert f"旧页补充观察：{note}" in merged
    assert f"旧页补充观察：旧页补充观察：{note}" not in merged
    assert change.retained == [note]
    assert change.preserved_old == [note]


def test_merge_update_section_additional_notes_keeps_boundary_even_when_open_question_overlaps() -> None:
    old = "重要决定需要用户确认，不能只依赖召回记忆。"
    new = "本页面补充通用记忆架构。"
    context = f"{new}\n\n- 重要决定是否需要用户确认？"

    merged, change = section_merge_module.merge_update_section(
        "additional_notes",
        old,
        new,
        absorption_context=context,
    )

    assert "旧页补充观察" in merged
    assert old in merged
    assert change.retained == [old]
    assert change.preserved_old == [old]
    assert change.removed == []


def test_merge_update_section_additional_notes_generic_new_version_does_not_supersede_note() -> None:
    old = "重要决定需要用户确认，不能只依赖召回记忆。"
    new = "新版页面补充通用记忆架构。"

    merged, change = section_merge_module.merge_update_section("additional_notes", old, new)

    assert "旧页补充观察" in merged
    assert old in merged
    assert change.preserved_old == [old]


def test_merge_update_section_additional_notes_unrelated_replacement_does_not_supersede_note() -> None:
    old = "重要决定需要用户确认，不能只依赖召回记忆。"
    new = "新版 API 已改为支持记忆元数据。"

    merged, change = section_merge_module.merge_update_section("additional_notes", old, new)

    assert "旧页补充观察" in merged
    assert old in merged
    assert change.preserved_old == [old]


def test_merge_update_section_additional_notes_unrelated_replacement_with_anchor_preserves_note() -> None:
    old = "重要决定需要用户确认，不能只依赖召回记忆。"
    new = "新版 API 已改为支持用户记忆元数据。重要决定仍需要用户确认。"

    merged, change = section_merge_module.merge_update_section("additional_notes", old, new)

    assert "旧页补充观察" in merged
    assert old in merged
    assert change.preserved_old == [old]


def test_merge_update_section_additional_notes_recall_memory_metadata_change_preserves_note() -> None:
    old = "重要决定需要用户确认，不能只依赖召回记忆。"
    new = "新版 API 已改为支持召回记忆元数据。"

    merged, change = section_merge_module.merge_update_section("additional_notes", old, new)

    assert "旧页补充观察" in merged
    assert old in merged
    assert change.preserved_old == [old]


def test_merge_update_section_additional_notes_recall_memory_field_rename_preserves_note() -> None:
    old = "重要决定需要用户确认，不能只依赖召回记忆。"
    new = "召回记忆字段已改为 memories。"

    merged, change = section_merge_module.merge_update_section("additional_notes", old, new)

    assert "旧页补充观察" in merged
    assert old in merged
    assert change.preserved_old == [old]


def test_merge_update_section_additional_notes_reports_absorbed_and_removed_units_separately() -> None:
    absorbed = "重要决定需要用户确认，不能只依赖召回记忆。"
    low = "本页面可与 Redis 页面联动阅读。"
    old = f"- {absorbed}\n- {low}"
    new = f"本页延续旧边界：{absorbed}"

    merged, change = section_merge_module.merge_update_section("additional_notes", old, new)

    assert merged == new
    assert change.retained == [absorbed]
    assert change.preserved_old == []
    assert change.removed == [low]
    assert old not in change.removed


def test_stable_brand_typos_are_normalized_in_draft_and_related() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-TYPO",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="entities/Entity_Cat Wu.md",
        display_title="Cat Wu",
        page_type="entity",
        new_understanding="测试。",
        section_plans={"detail": "详情"},
        reason="测试 typo 修正。",
        related_pages=[
            pipeline_module.RelatedPageRef(
                target_path="entities/Entity_Claude Code.md",
                display_title="Claude Code",
                source="source_digest",
                reason="Cat Wu 是 Clade Code 产品负责人。",
            )
        ],
    )
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-TYPO",
                action="create",
                canonical_target_path="entities/Entity_Cat Wu.md",
                summary="Cat Wu 负责 Clade Code，任职于 Anropinic，并与 Borris Cherny 协作。",
                body_markdown="Cat Wu 负责 Clade Code，任职于 Anropinic，并与 Borris Cherny 协作。",
                change_summary="创建 Borris 相关页面。",
                source_coverage_notes="Borris 与 Cat Wu 的访谈。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/entities/Entity_Cat Wu.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )

    finalized = pipeline_module.finalize_draft_rendering(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
    )
    related = related_pages_module.render_related_pages(
        item,
        known_paths={"entities/Entity_Claude Code.md"},
    )

    assert "Claude Code" in finalized.pages[0].summary
    assert "Anthropic" in finalized.pages[0].summary
    assert "Boris Cherny" in finalized.pages[0].summary
    assert "Clade Code" not in finalized.pages[0].summary
    assert "Borris" not in finalized.pages[0].summary
    assert finalized.pages[0].change_summary == "创建 Boris 相关页面。"
    assert finalized.pages[0].source_coverage_notes == "Boris 与 Cat Wu 的访谈。"
    assert "Cat Wu 是 Claude Code 产品负责人" in related
    assert draft_validation_module.normalize_stable_brand_typos("Borrison builds Clade Codebase tools") == "Borrison builds Clade Codebase tools"


def test_validate_draft_rendering_rejects_model_self_talk() -> None:
    plan = pipeline_module.WikiMergePlanArtifact(
        log_date="2026-06-06",
        items=[
            pipeline_module.WikiMergePlanItem(
                page_plan_id="PP-SELF-TALK",
                source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
                action="create",
                canonical_target_path="concepts/Concept_静态基准评估.md",
                display_title="静态基准评估",
                page_type="concept",
                new_understanding="静态基准可能高估智能体表现。",
                section_plans={"detail": "详情"},
                reason="测试 draft 自我推理污染。",
            )
        ],
    )
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-SELF-TALK",
                action="create",
                canonical_target_path="concepts/Concept_静态基准评估.md",
                summary="静态基准可能高估智能体表现。",
                body_markdown=draft_body(detail="静态基准会受污染影响。检查原文后我会修正数字方向，这里需要谨慎。", examples="例如，静态结果可能高于实时结果。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    draft_validation_module.validate_draft_rendering(draft, plan, language="zh-CN")
    issues = draft_validation_module.draft_self_talk_issues(draft)

    assert [issue.issue_code for issue in issues] == ["model_self_talk_leak"]
    assert issues[0].field_path == "pages.PP-SELF-TALK.body_markdown"
    assert "检查原文" in issues[0].message


def test_validate_draft_rendering_rejects_wiki_state_leak() -> None:
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-STATE-LEAK",
                action="create",
                canonical_target_path="entities/Entity_Cowork.md",
                summary="Cowork 是知识工作协作者产品。",
                body_markdown=draft_body(detail="Cowork 用于综合信息和创建文档。", additional_notes="目前 wiki 中无此页面，创建后可与 Claude Code、Cat Wu 等页面互链。"),
                change_summary="创建 Cowork 页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    issues = draft_validation_module.draft_self_talk_issues(draft)

    assert [issue.issue_code for issue in issues] == ["model_self_talk_leak"]
    assert issues[0].field_path == "pages.PP-STATE-LEAK.body_markdown"
    assert "目前wiki中无此页面" in issues[0].message


def test_validate_draft_rendering_allows_normal_caution_wording() -> None:
    plan = pipeline_module.WikiMergePlanArtifact(
        log_date="2026-06-06",
        items=[
            pipeline_module.WikiMergePlanItem(
                page_plan_id="PP-CAUTION",
                source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
                action="create",
                canonical_target_path="concepts/Concept_高风险部署.md",
                display_title="高风险部署",
                page_type="concept",
                new_understanding="高风险部署需要额外审查。",
                section_plans={"detail": "详情"},
                reason="测试正常谨慎措辞。",
            )
        ],
    )
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-CAUTION",
                action="create",
                canonical_target_path="concepts/Concept_高风险部署.md",
                summary="高风险部署需要额外审查。",
                body_markdown=draft_body(detail="高风险部署需要谨慎处理，尤其是在权限、用户数据和自动化执行边界不清楚时。", examples="例如，生产环境自动化执行前应先做人工审批。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    draft_validation_module.validate_draft_rendering(draft, plan, language="zh-CN")


def qwen_related_block_plan() -> pipeline_module.WikiMergePlanArtifact:
    return pipeline_module.WikiMergePlanArtifact(
        log_date="2026-06-06",
        items=[
            pipeline_module.WikiMergePlanItem(
                page_plan_id="PP-QWEN",
                source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
                action="create",
                canonical_target_path="entities/Entity_Qwen-Agent.md",
                display_title="Qwen-Agent",
                page_type="entity",
                new_understanding="Qwen-Agent 是 Agent 开发框架。",
                section_plans={"detail": "详情"},
                reason="测试 related block 泄漏。",
            )
        ],
    )


def qwen_related_block_draft(additional_notes: str) -> pipeline_module.DraftRenderingArtifact:
    return pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-QWEN",
                action="create",
                canonical_target_path="entities/Entity_Qwen-Agent.md",
                summary="Qwen-Agent 是 Agent 开发框架。",
                body_markdown=draft_body(detail="Qwen-Agent 支持工具使用、规划和记忆能力。", examples="例如，开发者可以用它把 LLM、工具和智能体抽象组合成一个可运行助手。", additional_notes=additional_notes),
                change_summary="创建 Qwen-Agent 页面。",
                source_coverage_notes="测试。",
            )
        ]
    )


def test_validate_draft_rendering_rejects_related_block_inside_content() -> None:
    draft = qwen_related_block_draft(
        "相关页面：\n"
        "- [[concepts/Concept_Agent 开发框架（Qwen-Agent）.md]]\n"
        "- [[concepts/Concept_代码解释器（Qwen-Agent）.md]]"
    )

    with pytest.raises(pipeline_module.ContractValidationError) as exc_info:
        draft_validation_module.validate_draft_rendering(draft, qwen_related_block_plan(), language="zh-CN")

    assert exc_info.value.issues[0].issue_code == "stray_related_links_in_content"
    assert exc_info.value.issues[0].field_path == "pages.PP-QWEN.body_markdown"


def test_validate_draft_rendering_rejects_decorated_related_markdown_links_inside_content() -> None:
    draft = qwen_related_block_draft(
        "- **相关页面**：建议参见 [Agent 开发框架](concepts/Concept_Agent 开发框架（Qwen-Agent）.md)"
    )

    with pytest.raises(pipeline_module.ContractValidationError) as exc_info:
        draft_validation_module.validate_draft_rendering(draft, qwen_related_block_plan(), language="zh-CN")

    assert exc_info.value.issues[0].issue_code == "stray_related_links_in_content"


def test_validate_draft_rendering_rejects_related_markdown_link_bullets_inside_content() -> None:
    draft = qwen_related_block_draft(
        "**Related Pages**\n"
        "- [Agent framework](concepts/Concept_Agent framework.md)\n"
        "- [Code interpreter](concepts/Concept_Code interpreter.md)"
    )

    with pytest.raises(pipeline_module.ContractValidationError) as exc_info:
        draft_validation_module.validate_draft_rendering(draft, qwen_related_block_plan(), language="zh-CN")

    assert exc_info.value.issues[0].issue_code == "stray_related_links_in_content"


def test_validate_draft_rendering_rejects_self_wikilink_inside_content() -> None:
    draft = qwen_related_block_draft("可与 [[entities/Entity_Qwen-Agent.md|Qwen-Agent]] 页面保持一致。")

    with pytest.raises(pipeline_module.ContractValidationError) as exc_info:
        draft_validation_module.validate_draft_rendering(draft, qwen_related_block_plan(), language="zh-CN")

    assert exc_info.value.issues[0].issue_code == "stray_related_links_in_content"


def test_validate_draft_rendering_rejects_basename_self_wikilink_inside_content() -> None:
    draft = qwen_related_block_draft("可与 [[Entity_Qwen-Agent.md|Qwen-Agent]] 页面保持一致。")

    with pytest.raises(pipeline_module.ContractValidationError) as exc_info:
        draft_validation_module.validate_draft_rendering(draft, qwen_related_block_plan(), language="zh-CN")

    assert exc_info.value.issues[0].issue_code == "stray_related_links_in_content"


def test_validate_draft_rendering_rejects_display_title_self_wikilink_inside_content() -> None:
    draft = qwen_related_block_draft("可与 [[Qwen-Agent|Qwen-Agent]] 页面保持一致。")

    with pytest.raises(pipeline_module.ContractValidationError) as exc_info:
        draft_validation_module.validate_draft_rendering(draft, qwen_related_block_plan(), language="zh-CN")

    assert exc_info.value.issues[0].issue_code == "stray_related_links_in_content"


def test_validate_draft_rendering_allows_external_markdown_link_with_display_title() -> None:
    draft = qwen_related_block_draft(
        "项目仓库可以写作 [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent)，"
        "这里它只是外部参考链接，不是指向当前 wiki 页面的自链。"
    )

    draft_validation_module.validate_draft_rendering(draft, qwen_related_block_plan(), language="zh-CN")


def test_validate_draft_rendering_allows_related_word_without_link_block() -> None:
    draft = qwen_related_block_draft(
        "Related work 这个英文短语只作为普通说明出现，没有相关页面列表。"
        "这里补充说明框架适合用来观察工具调用、规划和记忆抽象之间的边界。"
    )

    draft_validation_module.validate_draft_rendering(draft, qwen_related_block_plan(), language="zh-CN")


def test_canonicalize_draft_rendering_strips_system_sections_from_free_body() -> None:
    plan = qwen_related_block_plan()
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-QWEN",
                action="create",
                canonical_target_path="entities/Entity_Qwen-Agent.md",
                summary="Qwen-Agent 是 Agent 开发框架。",
                body_markdown=(
                    "Qwen-Agent 把模型、工具和智能体运行时组织在一起，适合说明 Agent 框架的工程边界。\n\n"
                    "## 相关页面\n\n"
                    "- [[concepts/Concept_Agent 开发框架（Qwen-Agent）.md]]：模型误写的系统段落。\n\n"
                    "## 后续说明\n\n"
                    "这部分仍属于自由正文，应该保留。\n\n"
                    "## 矛盾与未决问题\n\n"
                    "- 这个系统段落也应交给系统统一渲染。"
                ),
                change_summary="创建 Qwen-Agent 页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    canonical = draft_validation_module.canonicalize_draft_artifact(draft, plan)
    body = canonical.pages[0].body_markdown

    draft_validation_module.validate_draft_rendering(canonical, plan, language="zh-CN")
    assert "相关页面" not in body
    assert "矛盾与未决问题" not in body
    assert "Concept_Agent 开发框架" not in body
    assert "后续说明" in body
    assert "这部分仍属于自由正文，应该保留。" in body


def test_validate_draft_rendering_allows_tilde_fenced_related_markdown_example() -> None:
    draft = qwen_related_block_draft(
        "下面只是一个 Markdown 示例，不代表页面正文关系。\n"
        "   ~~~md\n"
        "相关页面：\n"
        "- [[concepts/Concept_Agent 开发框架（Qwen-Agent）.md]]\n"
        "   ~~~~\n"
        "示例外的正文继续说明 Qwen-Agent 的页面内容边界。"
    )

    draft_validation_module.validate_draft_rendering(draft, qwen_related_block_plan(), language="zh-CN")


def test_stray_related_links_issue_is_page_scoped_repairable() -> None:
    issues = [
        pipeline_module.StructuredIssue(
            issue_code="stray_related_links_in_content",
            field_path="pages.PP-QWEN.body_markdown",
            validator_id="validate_draft_rendering",
            message="stray related links",
            repairability="repairable",
        )
    ]

    assert pipeline_module.draft_repair_page_plan_ids_from_issues(issues, qwen_related_block_plan()) == {"PP-QWEN"}


def test_update_preservation_issues_detect_missing_old_key_phrases() -> None:
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-UPDATE",
                action="update",
                canonical_target_path="entities/Entity_Claude Code.md",
                summary="Claude Code 是一个编码助手，本轮只补充产品功能。",
                body_markdown=draft_body(detail="新材料讨论 PowerUp、TODO List 和发布速度。"),
                change_summary="补充产品视角。",
                source_coverage_notes="测试。",
            )
        ]
    )
    pack = {
        "schema_version": "update_preservation_pack.v1",
        "pages": [
            {
                "page_plan_id": "PP-UPDATE",
                "target_path": "entities/Entity_Claude Code.md",
                "sections": [
                    {
                        "section_key": "detail",
                        "old_text": "Managed Agents / harness 视角强调安全边界、隔离容器、工具权限和会话对象。",
                        "key_phrases": ["Managed Agents / harness", "安全边界", "会话对象"],
                        "min_required_matches": 2,
                    }
                ],
            }
        ],
    }

    issues = update_preservation_module.update_preservation_issues(draft, pack)

    assert [issue.issue_code for issue in issues] == ["old_knowledge_not_absorbed"]
    assert issues[0].repairability == "repairable"
    assert "Managed Agents / harness" in issues[0].message
    assert "Required old concept obligations" in issues[0].message
    assert "安全边界/权限限制" in issues[0].message


def test_partial_draft_extraction_rejects_update_missing_old_knowledge() -> None:
    update_item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-UPDATE",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="update",
        canonical_target_path="entities/Entity_Claude Code.md",
        matched_page="entities/Entity_Claude Code.md",
        display_title="Claude Code",
        page_type="entity",
        new_understanding="补充产品视角。",
        section_plans={"detail": "详情"},
        reason="测试 partial draft。",
    )
    missing_item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-MISSING",
        source_basis=SourceBasis(source_candidate_ids=["CAND002"]),
        action="create",
        canonical_target_path="concepts/Concept_Missing.md",
        display_title="Missing",
        page_type="concept",
        new_understanding="另一个待生成页面。",
        section_plans={"detail": "详情"},
        reason="测试 partial draft。",
    )
    partial = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-UPDATE",
                action="update",
                canonical_target_path="entities/Entity_Claude Code.md",
                summary="Claude Code 是一个编码助手，本轮只补充产品功能。",
                body_markdown=draft_body(detail="新材料讨论 PowerUp、TODO List 和发布速度。"),
                change_summary="补充产品视角。",
                source_coverage_notes="测试。",
            )
        ]
    )
    pack = {
        "schema_version": "update_preservation_pack.v1",
        "pages": [
            {
                "page_plan_id": "PP-UPDATE",
                "target_path": "entities/Entity_Claude Code.md",
                "sections": [
                    {
                        "section_key": "detail",
                        "old_text": "Managed Agents / harness 视角强调安全边界、隔离容器、工具权限和会话对象。",
                        "key_phrases": ["Managed Agents / harness", "安全边界", "会话对象"],
                        "min_required_matches": 2,
                    }
                ],
            }
        ],
    }

    extracted = pipeline_module.extract_valid_partial_draft_rendering(
        json.dumps(partial.model_dump(mode="json"), ensure_ascii=False),
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[update_item, missing_item]),
        pipeline_module.WikiContextSnapshot(
            log_date="2026-06-06",
            source_target_path="sources/Source_Test.md",
            entries=[],
        ),
        update_preservation_pack=pack,
        approved_prepared_text="",
        language="zh-CN",
    )

    assert extracted is None


def test_partial_draft_extraction_preserves_example_literals_without_cleanup_report() -> None:
    ok_item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-OK",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_OK.md",
        display_title="OK",
        page_type="concept",
        new_understanding="通过页摘要。",
        section_plans={"summary": "摘要", "examples": "例子"},
        reason="测试 partial draft。",
    )
    missing_item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-MISSING",
        source_basis=SourceBasis(source_candidate_ids=["CAND002"]),
        action="create",
        canonical_target_path="concepts/Concept_Missing.md",
        display_title="Missing",
        page_type="concept",
        new_understanding="另一个待生成页面。",
        section_plans={"detail": "详情"},
        reason="测试 partial draft。",
    )
    partial = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-OK",
                action="create",
                canonical_target_path="concepts/Concept_OK.md",
                summary="通过页摘要。",
                body_markdown=draft_body(detail="这个页面用于说明示例值的使用场景和边界：examples 里的示例参数不应被改写成观察到的事实。", examples="- 示例构建编号是 “ABC123”。"),
                change_summary="创建通过页。",
                source_coverage_notes="测试。",
            )
        ]
    )
    plan = pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[ok_item, missing_item])
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(path="wiki/concepts/Concept_OK.md", expected_state="missing"),
            pipeline_module.WikiContextEntry(path="wiki/concepts/Concept_Missing.md", expected_state="missing"),
        ],
    )

    extracted = pipeline_module.extract_valid_partial_draft_rendering(
        json.dumps(partial.model_dump(mode="json"), ensure_ascii=False),
        plan,
        snapshot,
        update_preservation_pack={"schema_version": "update_preservation_pack.v1", "pages": []},
        approved_prepared_text="",
        language="zh-CN",
    )

    assert extracted is not None
    assert "ABC123" in extracted.pages[0].body_markdown


def test_draft_page_scoped_repair_payload_keeps_accepted_pages_and_targets_failing_page(tmp_path: Path) -> None:
    vault, _raw = make_vault(tmp_path)
    profile = pipeline_module.load_profile(vault / ".llmwiki" / "profiles" / "project_basic")
    ctx = types.SimpleNamespace(
        profile=profile,
        manifest=types.SimpleNamespace(vault_config_snapshot=pipeline_module.OperationConfigSnapshot()),
    )
    digest = SourceDigestArtifact(
        source_raw_path="raw/sample.md",
        summary="测试 page scoped repair。",
        concepts=[
            SourceDigestCandidate(
                candidate_id="C-OK",
                name="通过页",
                type="concept",
                one_sentence_summary="通过页摘要。",
                why_matters="通过页重要。",
                wiki_value="通过页可复用。",
                suggested_page_title="通过页",
            ),
            SourceDigestCandidate(
                candidate_id="C-BAD",
                name="失败页",
                type="concept",
                one_sentence_summary="失败页摘要。",
                why_matters="失败页重要。",
                wiki_value="失败页可复用。",
                suggested_page_title="失败页",
            ),
        ],
    )
    ok_item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-OK",
        source_basis=SourceBasis(source_candidate_ids=["C-OK"]),
        action="create",
        canonical_target_path="concepts/Concept_OK.md",
        display_title="通过页",
        page_type="concept",
        new_understanding="通过页摘要。",
        section_plans={"summary": "摘要", "detail": "详情"},
        reason="test",
    )
    bad_item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-BAD",
        source_basis=SourceBasis(source_candidate_ids=["C-BAD"]),
        action="create",
        canonical_target_path="concepts/Concept_BAD.md",
        display_title="失败页",
        page_type="concept",
        new_understanding="失败页摘要。",
        section_plans={"summary": "摘要", "detail": "详情"},
        reason="test",
    )
    merge_plan = WikiMergePlanArtifact(log_date="2026-06-06", items=[ok_item, bad_item])
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(path="wiki/concepts/Concept_OK.md", expected_state="missing"),
            pipeline_module.WikiContextEntry(path="wiki/concepts/Concept_BAD.md", expected_state="missing"),
        ],
    )
    raw_text = (
        "# 测试材料\n\n"
        "## 通过页\n\n通过页用于验证 accepted partial pages 会被保留。\n\n"
        "## 失败页\n\n失败页用于验证 repair payload 只重写失败页面。\n"
    )
    source_excerpt_pack = draft_rendering_payloads_module.build_draft_source_excerpt_pack(raw_text, digest, merge_plan, full_source_limit=10)
    update_preservation_pack = update_preservation_module.build_update_preservation_pack(merge_plan, snapshot)
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-OK",
                action="create",
                canonical_target_path="concepts/Concept_OK.md",
                summary="通过页摘要。",
                body_markdown=draft_body(detail="通过页用于验证 accepted partial pages 会被保留。"),
                change_summary="创建通过页。",
                source_coverage_notes="依据测试材料生成。",
            ),
            pipeline_module.DraftPageItem(
                page_plan_id="PP-BAD",
                action="create",
                canonical_target_path="concepts/Concept_BAD.md",
                summary="失败页摘要。",
                body_markdown=draft_body(detail="失败页用于验证 repair payload。", examples="例如，“用户喜欢蓝色”。"),
                change_summary="创建失败页。",
                source_coverage_notes="依据测试材料生成。",
            ),
        ]
    )
    issues = [
        pipeline_module.StructuredIssue(
            issue_code="unsupported_new_fact",
            field_path="pages.PP-BAD.examples",
            validator_id="draft_grounding_review",
            message="unsupported",
            repairability="repairable",
        )
    ]

    repair_prompt = pipeline_module.build_draft_rendering_page_repair_payload(
        task="draft_rendering",
        raw=json.dumps(draft.model_dump(mode="json"), ensure_ascii=False),
        issues=issues,
        output_model=pipeline_module.DraftRenderingArtifact,
        ctx=ctx,
        digest=digest,
        merge_plan=merge_plan,
        snapshot=snapshot,
        source_excerpt_pack=source_excerpt_pack,
        update_preservation_pack=update_preservation_pack,
        approved_prepared_text=raw_text,
    )

    assert repair_prompt is not None
    assert repair_prompt["repair_contract"]["mode"] == "page_scoped_repair"
    assert_draft_rendering_schema_page_fields(repair_prompt["repair_contract"]["schema"])
    assert repair_prompt["repair_contract"]["accepted_page_plan_ids"] == ["PP-OK"]
    assert repair_prompt["repair_contract"]["repair_page_plan_ids"] == ["PP-BAD"]
    assert "accepted_partial_pages" not in repair_prompt
    assert repair_prompt["accepted_page_refs"] == [
        {
            "page_plan_id": "PP-OK",
            "action": "create",
            "target_path": "concepts/Concept_OK.md",
            "display_title": "通过页",
            "page_type": "concept",
        }
    ]
    assert repair_prompt["repair_page_payload"]["required_page_plan_ids"] == ["PP-BAD"]
    assert "PP-OK" not in repair_prompt["repair_page_payload"]["required_page_plan_ids"]


def test_merge_repaired_draft_with_accepted_pages_ignores_returned_accepted_copy() -> None:
    ok_item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-OK",
        source_basis=SourceBasis(source_candidate_ids=["C-OK"]),
        action="create",
        canonical_target_path="concepts/Concept_OK.md",
        display_title="通过页",
        page_type="concept",
        new_understanding="通过页摘要。",
        section_plans={"summary": "摘要", "detail": "详情"},
        reason="test",
    )
    bad_item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-BAD",
        source_basis=SourceBasis(source_candidate_ids=["C-BAD"]),
        action="create",
        canonical_target_path="concepts/Concept_BAD.md",
        display_title="失败页",
        page_type="concept",
        new_understanding="失败页摘要。",
        section_plans={"summary": "摘要", "detail": "详情"},
        reason="test",
    )
    merge_plan = WikiMergePlanArtifact(log_date="2026-06-06", items=[ok_item, bad_item])
    accepted_ok = pipeline_module.DraftPageItem(
        page_plan_id="PP-OK",
        action="create",
        canonical_target_path="concepts/Concept_OK.md",
        summary="本地保留的通过页。",
        body_markdown=draft_body(detail="不要被模型覆盖。"),
        change_summary="创建页面。",
        source_coverage_notes="本地 accepted。",
    )
    returned_ok = accepted_ok.model_copy(
        update={"summary": "模型错误改写的通过页。", "body_markdown": "不应采纳。"}
    )
    repaired_bad = pipeline_module.DraftPageItem(
        page_plan_id="PP-BAD",
        action="create",
        canonical_target_path="concepts/Concept_BAD.md",
        summary="修复后的失败页。",
        body_markdown=draft_body(detail="只采纳修复页。"),
        change_summary="修复页面。",
        source_coverage_notes="repair。",
    )

    merged = pipeline_module.merge_repaired_draft_with_accepted_pages(
        pipeline_module.DraftRenderingArtifact(pages=[returned_ok, repaired_bad]),
        accepted_pages_by_id={"PP-OK": accepted_ok.model_dump(mode="json")},
        repair_page_plan_ids={"PP-BAD"},
        merge_plan=merge_plan,
    )

    assert [page.page_plan_id for page in merged.pages] == ["PP-OK", "PP-BAD"]
    assert merged.pages[0].summary == "本地保留的通过页。"
    assert merged.pages[1].summary == "修复后的失败页。"


def test_run_single_draft_rendering_merges_repair_only_result(tmp_path: Path) -> None:
    vault, raw_path = make_vault(tmp_path)
    profile = pipeline_module.load_profile(vault / ".llmwiki" / "profiles" / "project_basic")
    raw_text = (
        "# 测试材料\n\n"
        "## 通过页\n\n通过页摘要。通过页可以用来验证本地 accepted 页面在 repair 后仍被保留。\n\n"
        "## 失败页\n\n失败页摘要。失败页可以用来验证局部修复只重写坏页。\n"
    )
    raw_path.write_text(raw_text, encoding="utf-8")
    digest = SourceDigestArtifact(
        source_raw_path="raw/raw_project_note.md",
        summary="测试 repair-only handoff。",
        concepts=[
            SourceDigestCandidate(
                candidate_id="C-OK",
                name="通过页",
                type="concept",
                one_sentence_summary="通过页摘要。",
                why_matters="通过页重要。",
                wiki_value="通过页可复用。",
                suggested_page_title="通过页",
            ),
            SourceDigestCandidate(
                candidate_id="C-BAD",
                name="失败页",
                type="concept",
                one_sentence_summary="失败页摘要。",
                why_matters="失败页重要。",
                wiki_value="失败页可复用。",
                suggested_page_title="失败页",
            ),
        ],
    )
    ok_item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-OK",
        source_basis=SourceBasis(source_candidate_ids=["C-OK"]),
        action="create",
        canonical_target_path="concepts/Concept_OK.md",
        display_title="通过页",
        page_type="concept",
        new_understanding="通过页摘要。",
        section_plans={"summary": "摘要", "detail": "详情"},
        reason="test",
    )
    bad_item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-BAD",
        source_basis=SourceBasis(source_candidate_ids=["C-BAD"]),
        action="create",
        canonical_target_path="concepts/Concept_BAD.md",
        display_title="失败页",
        page_type="concept",
        new_understanding="失败页摘要。",
        section_plans={"summary": "摘要", "detail": "详情"},
        reason="test",
    )
    merge_plan = WikiMergePlanArtifact(log_date="2026-06-06", items=[ok_item, bad_item])
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(path="wiki/concepts/Concept_OK.md", expected_state="missing"),
            pipeline_module.WikiContextEntry(path="wiki/concepts/Concept_BAD.md", expected_state="missing"),
        ],
    )
    accepted_ok = pipeline_module.DraftPageItem(
        page_plan_id="PP-OK",
        action="create",
        canonical_target_path="concepts/Concept_OK.md",
        summary="通过页摘要。",
        body_markdown=draft_body(detail="通过页可以用来验证本地 accepted 页面在 repair 后仍被保留。"),
        change_summary="通过页摘要。",
        source_coverage_notes="通过页可以用来验证本地 accepted 页面在 repair 后仍被保留。",
    )
    bad_with_self_talk = pipeline_module.DraftPageItem(
        page_plan_id="PP-BAD",
        action="create",
        canonical_target_path="concepts/Concept_BAD.md",
        summary="失败页摘要。",
        body_markdown=draft_body(detail="失败页可以用来验证局部修复只重写坏页。检查原文后我会修正。"),
        change_summary="失败页摘要。",
        source_coverage_notes="失败页可以用来验证局部修复只重写坏页。",
    )
    repaired_bad = pipeline_module.DraftPageItem(
        page_plan_id="PP-BAD",
        action="create",
        canonical_target_path="concepts/Concept_BAD.md",
        summary="失败页摘要。",
        body_markdown=draft_body(detail="失败页可以用来验证局部修复只重写坏页。"),
        change_summary="失败页摘要。",
        source_coverage_notes="失败页可以用来验证局部修复只重写坏页。",
    )

    class SequenceProvider:
        name = "sequence"

        def __init__(self) -> None:
            self.payloads: list[dict[str, object]] = []
            self.outputs = [
                pipeline_module.DraftRenderingArtifact(pages=[accepted_ok, bad_with_self_talk]),
                pipeline_module.DraftRenderingArtifact(pages=[repaired_bad]),
            ]

        def generate_raw(self, task: str, payload: dict[str, object], output_model: type[object]) -> str:
            self.payloads.append(payload)
            output = self.outputs[len(self.payloads) - 1]
            output = draft_validation_module.canonicalize_draft_artifact(output, merge_plan)
            return json.dumps(output.model_dump(mode="json"), ensure_ascii=False)

    class NoopRedactor:
        def redact(self, data: object) -> object:
            return data

        def redact_text(self, text: str) -> str:
            return text

    provider = SequenceProvider()
    output_dir = tmp_path / "draft-output"
    output_dir.mkdir()
    ctx = types.SimpleNamespace(
        vault=vault,
        run_dir=tmp_path / "run",
        raw_path=raw_path,
        profile=profile,
        manifest=types.SimpleNamespace(vault_config_snapshot=pipeline_module.OperationConfigSnapshot()),
        execution_context=types.SimpleNamespace(redactor=NoopRedactor()),
    )

    result = pipeline_module.run_single_draft_rendering_model_call(
        ctx=ctx,
        provider=provider,
        output_dir=output_dir,
        digest=digest,
        merge_plan=merge_plan,
        snapshot=snapshot,
        source_excerpt_pack=draft_rendering_payloads_module.build_draft_source_excerpt_pack(raw_text, digest, merge_plan, full_source_limit=20),
        update_preservation_pack=update_preservation_module.build_update_preservation_pack(merge_plan, snapshot),
        approved_prepared_text=raw_text,
    )

    assert [page.page_plan_id for page in result.pages] == ["PP-OK", "PP-BAD"]
    assert result.pages[0].body_markdown == "通过页可以用来验证本地 accepted 页面在 repair 后仍被保留。"
    assert result.pages[1].body_markdown == "失败页可以用来验证局部修复只重写坏页。"
    assert len(provider.payloads) == 2
    repair_payload = provider.payloads[1]
    assert repair_payload["repair_contract"]["mode"] == "page_scoped_repair"
    assert repair_payload["repair_contract"]["repair_page_plan_ids"] == ["PP-BAD"]
    assert "accepted_page_refs" in repair_payload
    assert "accepted_partial_pages" not in repair_payload
    persisted_repair_prompt = read_json(output_dir / "repair_prompts" / "attempt-2.json")
    assert persisted_repair_prompt["repair_contract"]["repair_page_plan_ids"] == ["PP-BAD"]
    assert "accepted_partial_pages" not in persisted_repair_prompt
    final_provider_result = read_json(output_dir / "provider_result.json")
    assert [page["page_plan_id"] for page in final_provider_result["parsed_output"]["pages"]] == ["PP-BAD"]


def test_missing_repair_page_issue_reuses_local_accepted_pages_for_page_scoped_repair(tmp_path: Path) -> None:
    vault, _raw = make_vault(tmp_path)
    profile = pipeline_module.load_profile(vault / ".llmwiki" / "profiles" / "project_basic")
    ctx = types.SimpleNamespace(
        profile=profile,
        manifest=types.SimpleNamespace(vault_config_snapshot=pipeline_module.OperationConfigSnapshot()),
    )
    digest = SourceDigestArtifact(
        source_raw_path="raw/sample.md",
        summary="测试 missing repair page。",
        concepts=[
            SourceDigestCandidate(
                candidate_id="C-OK",
                name="通过页",
                type="concept",
                one_sentence_summary="通过页摘要。",
                why_matters="通过页重要。",
                wiki_value="通过页可复用。",
                suggested_page_title="通过页",
            ),
            SourceDigestCandidate(
                candidate_id="C-BAD",
                name="失败页",
                type="concept",
                one_sentence_summary="失败页摘要。",
                why_matters="失败页重要。",
                wiki_value="失败页可复用。",
                suggested_page_title="失败页",
            ),
        ],
    )
    ok_item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-OK",
        source_basis=SourceBasis(source_candidate_ids=["C-OK"]),
        action="create",
        canonical_target_path="concepts/Concept_OK.md",
        display_title="通过页",
        page_type="concept",
        new_understanding="通过页摘要。",
        section_plans={"summary": "摘要"},
        reason="test",
    )
    bad_item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-BAD",
        source_basis=SourceBasis(source_candidate_ids=["C-BAD"]),
        action="create",
        canonical_target_path="concepts/Concept_BAD.md",
        display_title="失败页",
        page_type="concept",
        new_understanding="失败页摘要。",
        section_plans={"summary": "摘要"},
        reason="test",
    )
    merge_plan = WikiMergePlanArtifact(log_date="2026-06-06", items=[ok_item, bad_item])
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(path="wiki/concepts/Concept_OK.md", expected_state="missing"),
            pipeline_module.WikiContextEntry(path="wiki/concepts/Concept_BAD.md", expected_state="missing"),
        ],
    )
    raw_text = "# 测试材料\n\n## 通过页\n\n通过页。\n\n## 失败页\n\n失败页。\n"
    source_excerpt_pack = draft_rendering_payloads_module.build_draft_source_excerpt_pack(raw_text, digest, merge_plan, full_source_limit=10)
    update_preservation_pack = update_preservation_module.build_update_preservation_pack(merge_plan, snapshot)
    accepted_ok = pipeline_module.DraftPageItem(
        page_plan_id="PP-OK",
        action="create",
        canonical_target_path="concepts/Concept_OK.md",
        summary="通过页摘要。",
        body_markdown=draft_body(detail="通过页。"),
        change_summary="创建通过页。",
        source_coverage_notes="accepted。",
    )
    issues = [
        pipeline_module.StructuredIssue(
            issue_code="missing_repair_page",
            field_path="pages.PP-BAD",
            validator_id="draft_page_scoped_repair",
            message="missing repair page",
            repairability="repairable",
        )
    ]

    repair_prompt = pipeline_module.build_draft_rendering_page_repair_payload(
        task="draft_rendering",
        raw=json.dumps({"schema_version": "draft_rendering.v3", "pages": []}, ensure_ascii=False),
        issues=issues,
        output_model=pipeline_module.DraftRenderingArtifact,
        ctx=ctx,
        digest=digest,
        merge_plan=merge_plan,
        snapshot=snapshot,
        source_excerpt_pack=source_excerpt_pack,
        update_preservation_pack=update_preservation_pack,
        approved_prepared_text=raw_text,
        accepted_partial_pages_override=[accepted_ok],
    )

    assert repair_prompt is not None
    assert repair_prompt["repair_contract"]["mode"] == "page_scoped_repair"
    assert repair_prompt["repair_contract"]["repair_page_plan_ids"] == ["PP-BAD"]
    assert "accepted_partial_pages" not in repair_prompt


def test_preserve_active_repair_page_issues_keeps_full_active_repair_set_for_missing_issue() -> None:
    issues = [
        pipeline_module.StructuredIssue(
            issue_code="missing_repair_page",
            field_path="pages.PP-BAD-2",
            validator_id="draft_page_scoped_repair",
            message="missing second repair page",
            repairability="repairable",
        )
    ]

    expanded = pipeline_module.preserve_active_repair_page_issues(
        issues,
        {"PP-BAD-1", "PP-BAD-2"},
    )

    assert [issue.field_path for issue in expanded] == ["pages.PP-BAD-2", "pages.PP-BAD-1"]
    assert {issue.issue_code for issue in expanded} == {"missing_repair_page"}


def test_preserve_active_repair_page_issues_keeps_full_active_repair_set_for_page_issue() -> None:
    bad_1_issue = pipeline_module.StructuredIssue(
        issue_code="unsupported_new_fact",
        field_path="pages.PP-BAD-1.examples.0",
        validator_id="draft_grounding",
        message="unsupported example",
        repairability="repairable",
    )

    expanded = pipeline_module.preserve_active_repair_page_issues(
        [bad_1_issue],
        {"PP-BAD-1", "PP-BAD-2"},
    )

    assert [issue.field_path for issue in expanded] == ["pages.PP-BAD-1.examples.0", "pages.PP-BAD-2"]
    assert expanded[0] is bad_1_issue
    assert expanded[1].issue_code == "missing_repair_page"
    assert pipeline_module.draft_repair_page_plan_ids_from_issues(
        expanded,
        WikiMergePlanArtifact(
            log_date="2026-06-06",
            items=[
                pipeline_module.WikiMergePlanItem(
                    page_plan_id="PP-BAD-1",
                    source_basis=SourceBasis(source_candidate_ids=["C-BAD-1"]),
                    action="create",
                    canonical_target_path="concepts/Concept_BAD_1.md",
                    display_title="失败页一",
                    page_type="concept",
                    new_understanding="失败页一摘要。",
                    section_plans={"summary": "摘要"},
                    reason="test",
                ),
                pipeline_module.WikiMergePlanItem(
                    page_plan_id="PP-BAD-2",
                    source_basis=SourceBasis(source_candidate_ids=["C-BAD-2"]),
                    action="create",
                    canonical_target_path="concepts/Concept_BAD_2.md",
                    display_title="失败页二",
                    page_type="concept",
                    new_understanding="失败页二摘要。",
                    section_plans={"summary": "摘要"},
                    reason="test",
                ),
            ],
        ),
    ) == {"PP-BAD-1", "PP-BAD-2"}


def test_cleanup_open_question_unsupported_scope_claims_moves_fact_to_question() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-OQ",
        source_basis=SourceBasis(source_candidate_ids=["O001"]),
        action="create",
        canonical_target_path="open_questions/Open_Question_记忆准确性.md",
        display_title="记忆准确性",
        page_type="open_question",
        new_understanding="讨论记忆准确性。",
        section_plans={"summary": "摘要", "examples": "例子", "open_questions": "问题"},
        reason="test",
    )
    plan = WikiMergePlanArtifact(log_date="2026-06-06", items=[item])
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[pipeline_module.WikiContextEntry(path="wiki/open_questions/Open_Question_记忆准确性.md", expected_state="missing")],
    )
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-OQ",
                action="create",
                canonical_target_path="open_questions/Open_Question_记忆准确性.md",
                summary="讨论记忆准确性。",
                body_markdown=draft_body(examples="例如，模型可能召回到某个用户的偏好，但该偏好记忆不准确，导致错误响应。如何确保召回准确性？"),
                open_questions="- 如何确认召回结果？",
                change_summary="创建开放问题。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review_before = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "记忆召回结果需要确认。")
    cleaned, report = draft_grounding.cleanup_open_question_unsupported_scope_claims(
        draft,
        plan,
        snapshot,
        "记忆召回结果需要确认。",
    )
    review_after = draft_grounding.build_draft_grounding_review(cleaned, plan, snapshot, "记忆召回结果需要确认。")

    assert review_before.requires_review is False
    assert review_before.warnings
    assert cleaned == draft
    assert report["changed"] is False
    assert report["relocation_count"] == 0
    assert not review_after.requires_review

    cleaned_again, report_again = draft_grounding.cleanup_open_question_unsupported_scope_claims(
        cleaned,
        plan,
        snapshot,
        "记忆召回结果需要确认。",
    )
    assert cleaned_again == cleaned
    assert report_again["changed"] is False
    assert report_again["relocation_count"] == 0


def test_cleanup_open_question_duplicate_question_still_removes_fact() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-OQ",
        source_basis=SourceBasis(source_candidate_ids=["O001"]),
        action="create",
        canonical_target_path="open_questions/Open_Question_记忆准确性.md",
        display_title="记忆准确性",
        page_type="open_question",
        new_understanding="讨论记忆准确性。",
        section_plans={"summary": "摘要", "examples": "例子", "open_questions": "问题"},
        reason="test",
    )
    plan = WikiMergePlanArtifact(log_date="2026-06-06", items=[item])
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[pipeline_module.WikiContextEntry(path="wiki/open_questions/Open_Question_记忆准确性.md", expected_state="missing")],
    )
    duplicate_question = "待补来源：召回到不准确的用户偏好时，系统应如何确认与纠正？"
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-OQ",
                action="create",
                canonical_target_path="open_questions/Open_Question_记忆准确性.md",
                summary="讨论记忆准确性。",
                body_markdown=draft_body(examples="例如，模型可能召回到某个用户的偏好，但该偏好记忆不准确，导致错误响应。如何确保召回准确性？"),
                open_questions=f"- 如何确认召回结果？\n- {duplicate_question}",
                change_summary="创建开放问题。",
                source_coverage_notes="测试。",
            )
        ]
    )

    cleaned, report = draft_grounding.cleanup_open_question_unsupported_scope_claims(
        draft,
        plan,
        snapshot,
        "记忆召回结果需要确认。",
    )

    page = cleaned.pages[0]
    assert cleaned == draft
    assert report["changed"] is False
    assert report["relocation_count"] == 0
    assert "导致错误响应" in page.body_markdown
    assert page.open_questions.count(duplicate_question) == 1
    assert not draft_grounding.build_draft_grounding_review(cleaned, plan, snapshot, "记忆召回结果需要确认。").requires_review


def test_cleanup_open_question_scope_claim_uses_reason_marker_after_sentence_split() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-OQ",
        source_basis=SourceBasis(source_candidate_ids=["O001"]),
        action="create",
        canonical_target_path="open_questions/Open_Question_记忆准确性.md",
        display_title="记忆准确性",
        page_type="open_question",
        new_understanding="讨论记忆准确性。",
        section_plans={"summary": "摘要", "examples": "例子", "open_questions": "问题"},
        reason="test",
    )
    plan = WikiMergePlanArtifact(log_date="2026-06-06", items=[item])
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[pipeline_module.WikiContextEntry(path="wiki/open_questions/Open_Question_记忆准确性.md", expected_state="missing")],
    )
    unsupported_sentence = "正确的做法是使用抽象占位符描述模式：模型应通过工具搜索记忆，然后在涉及重要决策时向用户提问。"
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-OQ",
                action="create",
                canonical_target_path="open_questions/Open_Question_记忆准确性.md",
                summary="讨论记忆准确性。",
                body_markdown=draft_body(examples=f"模型可能需要确认记忆。{unsupported_sentence}"),
                open_questions="- 如何确认召回结果？",
                change_summary="创建开放问题。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review_before = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "记忆召回结果需要确认。")
    cleaned, report = draft_grounding.cleanup_open_question_unsupported_scope_claims(
        draft,
        plan,
        snapshot,
        "记忆召回结果需要确认。",
    )

    assert review_before.requires_review is False
    assert review_before.warnings[0].text == unsupported_sentence
    assert cleaned == draft
    assert report["changed"] is False
    assert report["relocation_count"] == 0
    assert not draft_grounding.build_draft_grounding_review(cleaned, plan, snapshot, "记忆召回结果需要确认。").requires_review


def test_open_question_scope_cleanup_claim_requires_reason_marker_in_text() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-OQ",
        source_basis=SourceBasis(source_candidate_ids=["O001"]),
        action="create",
        canonical_target_path="open_questions/Open_Question_记忆准确性.md",
        display_title="记忆准确性",
        page_type="open_question",
        new_understanding="讨论记忆准确性。",
        section_plans={"summary": "摘要"},
        reason="test",
    )
    claim = GroundingClaim(
        page_plan_id="PP-OQ",
        target_path="open_questions/Open_Question_记忆准确性.md",
        section_key="examples",
        claim_type="new_fact",
        text="这句话不包含被 reason 标出的词。",
        support="unsupported",
        action="needs_review",
        reason="新增影响范围/受影响对象推测 `涉及` 未被 raw 或 inspected wiki 同句级支撑；请删除该推测。",
    )

    assert not draft_grounding.open_question_scope_cleanup_claim(claim, item)


def test_cleanup_open_question_unsupported_scope_claims_does_not_touch_concepts() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-CONCEPT",
        source_basis=SourceBasis(source_candidate_ids=["C001"]),
        action="create",
        canonical_target_path="concepts/Concept_记忆准确性.md",
        display_title="记忆准确性",
        page_type="concept",
        new_understanding="讨论记忆准确性。",
        section_plans={"summary": "摘要", "examples": "例子", "open_questions": "问题"},
        reason="test",
    )
    plan = WikiMergePlanArtifact(log_date="2026-06-06", items=[item])
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[pipeline_module.WikiContextEntry(path="wiki/concepts/Concept_记忆准确性.md", expected_state="missing")],
    )
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-CONCEPT",
                action="create",
                canonical_target_path="concepts/Concept_记忆准确性.md",
                summary="讨论记忆准确性。",
                body_markdown=draft_body(examples="例如，模型可能召回到某个用户的偏好，但该偏好记忆不准确，导致错误响应。"),
                open_questions="- 如何确认召回结果？",
                change_summary="创建概念页。",
                source_coverage_notes="测试。",
            )
        ]
    )

    cleaned, report = draft_grounding.cleanup_open_question_unsupported_scope_claims(
        draft,
        plan,
        snapshot,
        "记忆召回结果需要确认。",
    )

    assert cleaned == draft
    assert report["changed"] is False
    review = draft_grounding.build_draft_grounding_review(cleaned, plan, snapshot, "记忆召回结果需要确认。")
    assert review.requires_review is False
    assert review.warnings


def test_draft_rendering_batch_refs_include_open_question_cleanup_schema(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    step_root = run_dir / "draft_rendering"
    report_path = step_root / "model_batches" / "batch-001" / "open_question_grounding_cleanup_report.json"
    report_path.parent.mkdir(parents=True)
    write_json(
        report_path,
        {
            "schema_version": "open_question_grounding_cleanup_report.v1",
            "changed": False,
            "relocation_count": 0,
            "skipped_count": 0,
            "pages": [],
        },
    )

    refs = pipeline_module.draft_rendering_model_batch_refs(run_dir, step_root, "draft_rendering")
    cleanup_ref = next(ref for ref in refs if ref.relative_path.endswith("open_question_grounding_cleanup_report.json"))

    assert cleanup_ref.schema_version == "open_question_grounding_cleanup_report.v1"


def test_draft_page_scoped_repair_falls_back_for_global_or_mixed_issues() -> None:
    merge_plan = WikiMergePlanArtifact(
        log_date="2026-06-06",
        items=[
            pipeline_module.WikiMergePlanItem(
                page_plan_id="PP-OK",
                source_basis=SourceBasis(source_candidate_ids=["C-OK"]),
                action="create",
                canonical_target_path="concepts/Concept_OK.md",
                display_title="通过页",
                page_type="concept",
                new_understanding="通过页摘要。",
                section_plans={"summary": "摘要", "detail": "详情"},
                reason="test",
            )
        ],
    )
    issues = [
        pipeline_module.StructuredIssue(
            issue_code="unsupported_new_fact",
            field_path="pages.PP-OK.examples",
            validator_id="draft_grounding_review",
            message="unsupported",
            repairability="repairable",
        ),
        pipeline_module.StructuredIssue(
            issue_code="missing_page_plan_coverage",
            field_path="pages",
            validator_id="validate_draft_rendering",
            message="global",
            repairability="repairable",
        ),
    ]

    assert pipeline_module.draft_repair_page_plan_ids_from_issues(issues, merge_plan) is None


def test_accepted_partial_page_copy_issues_detect_rewritten_accepted_page() -> None:
    accepted = pipeline_module.DraftPageItem(
        page_plan_id="PP-OK",
        action="create",
        canonical_target_path="concepts/Concept_OK.md",
        summary="原摘要。",
        body_markdown=draft_body(detail="原详情。"),
        change_summary="创建原页面。",
        source_coverage_notes="测试。",
    )
    changed = accepted.model_copy(update={"change_summary": "被模型改写。"})

    issues = pipeline_module.accepted_partial_page_copy_issues(
        pipeline_module.DraftRenderingArtifact(pages=[changed]),
        {"PP-OK": accepted.model_dump(mode="json")},
    )

    assert [issue.issue_code for issue in issues] == ["accepted_partial_page_changed"]
    assert issues[0].field_path == "pages.PP-OK"


def test_update_preservation_reinforcement_fills_missing_old_knowledge() -> None:
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-UPDATE",
                action="update",
                canonical_target_path="entities/Entity_Claude Code.md",
                summary="Claude Code 是一个编码助手，本轮补充待办事项列表和发布速度。",
                body_markdown=draft_body(examples="例如，待办事项列表用于帮助模型跟踪任务。"),
                change_summary="补充产品视角。",
                source_coverage_notes="测试。",
            )
        ]
    )
    pack = {
        "schema_version": "update_preservation_pack.v1",
        "pages": [
            {
                "page_plan_id": "PP-UPDATE",
                "target_path": "entities/Entity_Claude Code.md",
                "display_title": "Claude Code",
                "sections": [
                    {
                        "section_key": "summary",
                        "old_text": "旧页观点：Claude Code 在 Managed Agents 语境中不是单一聊天窗口，而是被 harness 调度的大脑；真正的执行手由隔离容器、工具权限和会话对象承担。",
                        "key_phrases": ["Managed Agents", "harness", "隔离容器", "会话对象"],
                        "min_required_matches": 2,
                    },
                    {
                        "section_key": "examples",
                        "old_text": "当 Claude Code 需要写文件时，旧页要求先经由 harness 检查路径和权限，再把命令交给隔离执行环境，而不是让模型直接拥有无限本机权限。",
                        "key_phrases": ["harness 检查路径和权限", "隔离执行环境", "无限本机权限"],
                        "min_required_matches": 2,
                    },
                ],
            }
        ],
    }

    reinforced, report = update_preservation_module.reinforce_update_preservation(draft, pack)

    assert report["changed"] is True
    assert report["reinforced_section_count"] == 2
    assert update_preservation_module.update_preservation_issues(reinforced, pack) == []
    page = reinforced.pages[0]
    assert "从旧页保留的架构视角看" in page.summary
    assert "从旧页保留的架构视角看" in page.body_markdown
    assert "从旧页保留的架构视角看" in page.body_markdown


def test_update_preservation_reinforcement_synthesizes_concept_bridge_without_english_dump() -> None:
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-UPDATE",
                action="update",
                canonical_target_path="entities/Entity_Claude Code.md",
                body_markdown=draft_body(detail="Claude Code 本轮补充产品发布速度和 PM 协作流程。"),
                change_summary="补充产品视角。",
                source_coverage_notes="测试。",
            )
        ]
    )
    old_text = (
        "From the old Managed Agents / harness perspective, Claude Code is not just a chat interface. "
        "It routes model intent through tool permissions, sandboxed execution, repository context, and session state."
    )
    pack = {
        "schema_version": "update_preservation_pack.v1",
        "pages": [
            {
                "page_plan_id": "PP-UPDATE",
                "target_path": "entities/Entity_Claude Code.md",
                "sections": [
                    {
                        "section_key": "detail",
                        "old_text": old_text,
                        "key_phrases": ["From the old Managed Agents / harness perspective", "sandboxed execution"],
                        "min_required_matches": 0,
                        "concept_obligations": [
                            {"name": "managed_agents", "label": "Managed Agents / 托管智能体"},
                            {"name": "harness", "label": "harness / 适配框架"},
                            {"name": "session_context", "label": "会话/持久上下文"},
                            {"name": "isolated_execution", "label": "隔离执行/容器"},
                        ],
                        "min_required_concept_matches": 3,
                    }
                ],
            }
        ],
    }

    reinforced, report = update_preservation_module.reinforce_update_preservation(draft, pack)

    body = reinforced.pages[0].body_markdown
    assert report["changed"] is True
    assert update_preservation_module.update_preservation_issues(reinforced, pack) == []
    assert "从旧页保留的架构视角看" in body
    assert "会话/持久上下文" in body
    assert "隔离执行/容器" in body
    assert "From the old Managed Agents" not in body
    assert "sandboxed execution" not in body


def test_update_preservation_reinforcement_single_concept_does_not_invent_other_concepts() -> None:
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-UPDATE",
                action="update",
                canonical_target_path="entities/Entity_Claude Code.md",
                summary="Claude Code 本轮补充产品发布速度。",
                body_markdown=draft_body(detail="Claude Code 本轮补充产品发布速度。"),
                change_summary="补充产品视角。",
                source_coverage_notes="测试。",
            )
        ]
    )
    pack = {
        "schema_version": "update_preservation_pack.v1",
        "pages": [
            {
                "page_plan_id": "PP-UPDATE",
                "target_path": "entities/Entity_Claude Code.md",
                "sections": [
                    {
                        "section_key": "detail",
                        "old_text": "旧页只要求保留 Managed Agents 这一系统定位。",
                        "key_phrases": ["Managed Agents"],
                        "min_required_matches": 0,
                        "concept_obligations": [
                            {"name": "managed_agents", "label": "Managed Agents / 托管智能体"},
                        ],
                        "min_required_concept_matches": 1,
                    }
                ],
            }
        ],
    }

    reinforced, _report = update_preservation_module.reinforce_update_preservation(draft, pack)

    body = reinforced.pages[0].body_markdown
    assert update_preservation_module.update_preservation_issues(reinforced, pack) == []
    assert "Managed Agents / 托管智能体" in body
    assert "会话/持久上下文" not in body
    assert "隔离执行环境" not in body
    assert "隔离执行/容器" not in body


def test_update_preservation_pack_records_concept_obligations() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-UPDATE",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="update",
        canonical_target_path="entities/Entity_Claude Code.md",
        display_title="Claude Code",
        page_type="entity",
        new_understanding="补充产品视角。",
        section_plans={"detail": "详情"},
        reason="测试旧知识保留。",
        matched_page="entities/Entity_Claude Code.md",
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/entities/Entity_Claude Code.md",
                expected_state="present",
                preimage_sha256="old",
                content=(
                    "---\nllmwiki_type: entity\ntitle: Claude Code\nsummary: 旧页。\n---\n\n"
                    "# Claude Code\n\n"
                    "## 详情\n\n"
                    "Managed Agents / harness 视角强调安全边界、隔离容器、工具权限和会话对象。\n"
                ),
            )
        ],
    )

    pack = update_preservation_module.build_update_preservation_pack(
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
    )

    detail = pack["pages"][0]["sections"][0]
    labels = [concept["label"] for concept in detail["concept_obligations"]]
    assert "Managed Agents / 托管智能体" in labels
    assert "harness / 适配框架" in labels
    assert "安全边界/权限限制" in labels
    assert detail["min_required_concept_matches"] >= 3
    assert detail["min_required_matches"] == 0


def test_update_preservation_pack_reads_current_core_content_section() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-UPDATE",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="update",
        canonical_target_path="entities/Entity_Claude Code.md",
        display_title="Claude Code",
        page_type="entity",
        new_understanding="补充产品视角。",
        section_plans={"detail": "详情"},
        reason="测试当前正式页核心内容保留。",
        matched_page="entities/Entity_Claude Code.md",
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/entities/Entity_Claude Code.md",
                expected_state="present",
                preimage_sha256="old",
                content=(
                    "---\nllmwiki_type: entity\ntitle: Claude Code\nsummary: 旧页。\n---\n\n"
                    "# Claude Code\n\n"
                    "## 摘要\n\n旧页摘要。\n\n"
                    "## 核心内容\n\n"
                    "Managed Agents / harness 视角强调安全边界、隔离容器、工具权限和会话对象。\n\n"
                    "### 例子\n\n旧页还记录了 Claude Code 可作为 harness 示例承接托管智能体任务。\n\n"
                    "## 相关页面\n\n- [[entities/Entity_Anthropic|Anthropic]]\n\n"
                    "## 矛盾与未决问题\n\n暂无矛盾与未决问题记录。\n"
                ),
            )
        ],
    )

    pack = update_preservation_module.build_update_preservation_pack(
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
    )

    sections = {section["section_key"]: section for section in pack["pages"][0]["sections"]}
    core = sections["core_content"]
    assert core["section_key"] == "core_content"
    assert "Managed Agents / harness" in core["old_text"]
    assert "Claude Code 可作为 harness 示例" in core["old_text"]
    labels = [concept["label"] for concept in core["concept_obligations"]]
    assert "Managed Agents / 托管智能体" in labels
    assert "harness / 适配框架" in labels


def test_update_preservation_concepts_do_not_match_bare_session_substrings() -> None:
    concepts = update_preservation_module.update_preservation_concepts(
        "Possession of product context, user interview sessions, session stateless notes, "
        "session contextual comments, session 和 state, session和state, session 与 context, "
        "session & context, session. State, and session, state reveal product friction. "
        "Managed. Agents is not a phrase."
    )

    labels = [concept["label"] for concept in concepts]
    assert "会话/持久上下文" not in labels
    assert "Managed Agents / 托管智能体" not in labels


def test_update_preservation_concepts_match_brain_ampersand_hands_without_global_ampersand() -> None:
    concepts = update_preservation_module.update_preservation_concepts(
        "Managed Agents decouple brain & hands in the execution architecture."
    )

    labels = [concept["label"] for concept in concepts]
    assert "Managed Agents / 托管智能体" in labels
    assert "大脑与双手解耦" in labels

    dirty_concepts = update_preservation_module.update_preservation_concepts(
        "session & context are discussed separately. brain && hands is dirty shorthand."
    )
    dirty_labels = [concept["label"] for concept in dirty_concepts]
    assert "会话/持久上下文" not in dirty_labels
    assert "大脑与双手解耦" not in dirty_labels


def test_update_preservation_pack_does_not_turn_interview_sessions_into_context_obligation() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-UPDATE",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="update",
        canonical_target_path="entities/Entity_User Research.md",
        display_title="User Research",
        page_type="entity",
        new_understanding="补充研究视角。",
        section_plans={"detail": "详情"},
        reason="测试普通 sessions 不应变成持久上下文。",
        matched_page="entities/Entity_User Research.md",
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/entities/Entity_User Research.md",
                expected_state="present",
                preimage_sha256="old",
                content=(
                    "---\nllmwiki_type: entity\ntitle: User Research\nsummary: old.\n---\n\n"
                    "# User Research\n\n"
                    "## Detail\n\n"
                    "Managed Agents user interview sessions reveal product friction in onboarding workflows.\n"
                ),
            )
        ],
    )

    pack = update_preservation_module.build_update_preservation_pack(
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
    )

    detail = pack["pages"][0]["sections"][0]
    labels = [concept["label"] for concept in detail["concept_obligations"]]
    assert "Managed Agents / 托管智能体" in labels
    assert "会话/持久上下文" not in labels


def test_update_preservation_concepts_match_specific_session_terms() -> None:
    concepts = update_preservation_module.update_preservation_concepts(
        "The harness keeps a session object with persistent-session state, session/context, "
        "session_state, session-state, and a durable context."
    )

    session = next(concept for concept in concepts if concept["name"] == "session_context")
    assert "session object" in session["matched_terms"]
    assert "persistent session" in session["matched_terms"]
    assert "durable context" in session["matched_terms"]
    assert "session state" in session["matched_terms"]
    assert "session context" in session["matched_terms"]


def test_update_preservation_concepts_match_persistent_context_and_sandboxed_execution() -> None:
    concepts = update_preservation_module.update_preservation_concepts(
        "Managed Agents use the session as a persistent context object inside sandboxed execution."
    )

    labels = [concept["label"] for concept in concepts]
    assert "会话/持久上下文" in labels
    assert "隔离执行/容器" in labels

    morphology_concepts = update_preservation_module.update_preservation_concepts(
        "Harnesses run isolated containers, containerized execution, and sandboxes."
    )
    morphology_labels = [concept["label"] for concept in morphology_concepts]
    assert "harness / 适配框架" in morphology_labels
    assert "隔离执行/容器" in morphology_labels

    sandboxing_concepts = update_preservation_module.update_preservation_concepts("A sandboxing strategy is discussed.")
    sandboxing_labels = [concept["label"] for concept in sandboxing_concepts]
    assert "隔离执行/容器" not in sandboxing_labels


def test_update_preservation_session_absorption_uses_token_boundaries() -> None:
    section = {
        "section_key": "detail",
        "old_text": "Managed Agents keep session state as a persistent context.",
        "key_phrases": [],
        "min_required_matches": 0,
        "concept_obligations": update_preservation_module.update_preservation_concepts(
            "Managed Agents keep session state as a persistent context."
        ),
        "min_required_concept_matches": 2,
    }

    absorption = update_preservation_module.update_preservation_section_absorption(
        section,
        "Managed Agents are mentioned with session stateless notes, session contextual comments, "
        "and session 与 context 分开介绍。",
    )

    assert "Managed Agents / 托管智能体" in absorption["matched_concepts"]
    assert "会话/持久上下文" not in absorption["matched_concepts"]
    assert "会话/持久上下文" in absorption["missing_concepts"]
    assert absorption["absorbed"] is False


def test_update_preservation_morphology_absorption_uses_explicit_variants() -> None:
    section = {
        "section_key": "detail",
        "old_text": "Harnesses run isolated containers and sandboxes.",
        "key_phrases": [],
        "min_required_matches": 0,
        "concept_obligations": update_preservation_module.update_preservation_concepts(
            "Harnesses run isolated containers and sandboxes."
        ),
        "min_required_concept_matches": 2,
    }

    absorption = update_preservation_module.update_preservation_section_absorption(
        section,
        "The architecture still uses harnesses and containerized execution.",
    )

    assert "harness / 适配框架" in absorption["matched_concepts"]
    assert "隔离执行/容器" in absorption["matched_concepts"]
    assert absorption["absorbed"] is True


def test_update_preservation_issues_detect_missing_persistent_context_obligation() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-UPDATE",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="update",
        canonical_target_path="entities/Entity_Claude Code.md",
        display_title="Claude Code",
        page_type="entity",
        new_understanding="补充产品视角。",
        section_plans={"detail": "详情"},
        reason="测试 persistent context 旧知识保留。",
        matched_page="entities/Entity_Claude Code.md",
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/entities/Entity_Claude Code.md",
                expected_state="present",
                preimage_sha256="old",
                content=(
                    "---\nllmwiki_type: entity\ntitle: Claude Code\nsummary: old.\n---\n\n"
                    "# Claude Code\n\n"
                    "## Detail\n\n"
                    "Managed Agents use the session as a persistent context object so execution state survives between turns.\n"
                ),
            )
        ],
    )
    pack = update_preservation_module.build_update_preservation_pack(
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
    )
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-UPDATE",
                action="update",
                canonical_target_path="entities/Entity_Claude Code.md",
                summary="Claude Code 延续 Managed Agents 产品视角。",
                body_markdown=draft_body(detail="Claude Code 延续 Managed Agents 产品视角，但这里只讨论发布速度。"),
                change_summary="补充产品视角。",
                source_coverage_notes="测试。",
            )
        ]
    )

    section = pack["pages"][0]["sections"][0]
    labels = [concept["label"] for concept in section["concept_obligations"]]
    assert "Managed Agents / 托管智能体" in labels
    assert "会话/持久上下文" in labels
    issues = update_preservation_module.update_preservation_issues(draft, pack)

    assert [issue.issue_code for issue in issues] == ["old_knowledge_not_absorbed"]
    assert "会话/持久上下文" in issues[0].message


def test_update_preservation_issues_detect_missing_brain_ampersand_hands_obligation() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-UPDATE",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="update",
        canonical_target_path="entities/Entity_Claude Code.md",
        display_title="Claude Code",
        page_type="entity",
        new_understanding="补充产品视角。",
        section_plans={"detail": "详情"},
        reason="测试 brain & hands 旧知识保留。",
        matched_page="entities/Entity_Claude Code.md",
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/entities/Entity_Claude Code.md",
                expected_state="present",
                preimage_sha256="old",
                content=(
                    "---\nllmwiki_type: entity\ntitle: Claude Code\nsummary: old.\n---\n\n"
                    "# Claude Code\n\n"
                    "## Detail\n\n"
                    "Managed Agents decouple brain & hands so model reasoning stays separate from execution.\n"
                ),
            )
        ],
    )
    pack = update_preservation_module.build_update_preservation_pack(
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
    )
    section = pack["pages"][0]["sections"][0]
    labels = [concept["label"] for concept in section["concept_obligations"]]
    assert "Managed Agents / 托管智能体" in labels
    assert "大脑与双手解耦" in labels

    absorbed = update_preservation_module.update_preservation_section_absorption(
        section,
        "Claude Code keeps Managed Agents architecture and explicitly decouples brain & hands.",
    )
    assert absorbed["absorbed"] is True

    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-UPDATE",
                action="update",
                canonical_target_path="entities/Entity_Claude Code.md",
                summary="Claude Code 延续 Managed Agents 产品视角。",
                body_markdown=draft_body(detail="Claude Code 延续 Managed Agents 产品视角，但这里只讨论发布速度。"),
                change_summary="补充产品视角。",
                source_coverage_notes="测试。",
            )
        ]
    )
    issues = update_preservation_module.update_preservation_issues(draft, pack)

    assert [issue.issue_code for issue in issues] == ["old_knowledge_not_absorbed"]
    assert "大脑与双手解耦" in issues[0].message


def test_update_preservation_pack_reads_english_section_headings() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-UPDATE",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="update",
        canonical_target_path="entities/Entity_Claude Code.md",
        display_title="Claude Code",
        page_type="entity",
        new_understanding="补充产品视角。",
        section_plans={"detail": "详情"},
        reason="测试英文旧页标题。",
        matched_page="entities/Entity_Claude Code.md",
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/entities/Entity_Claude Code.md",
                expected_state="present",
                preimage_sha256="old",
                content=(
                    "---\nllmwiki_type: entity\ntitle: Claude Code\nsummary: old.\n---\n\n"
                    "# Claude Code\n\n"
                    "## summary\n\n"
                    "Claude Code connects Managed Agents architecture to product practice.\n\n"
                    "## DETAIL\n\n"
                    "From the old Managed Agents / harness perspective, Claude Code routes model intent through sandboxed execution, tool permissions, and persistent session state.\n\n"
                    "## Value points\n\n"
                    "- Preserves the distinction between model reasoning and execution environment.\n"
                    "- Connects session context to product workflows.\n\n"
                    "## Open Questions\n\n"
                    "- Open questions should stay outside preservation core obligations.\n\n"
                    "## RELATED PAGES\n\n"
                    "- [[entities/Entity_Other|Other]]: should not be swallowed by Detail.\n"
                ),
            )
        ],
    )

    pack = update_preservation_module.build_update_preservation_pack(
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
    )

    sections = {section["section_key"]: section for section in pack["pages"][0]["sections"]}
    assert set(sections) == {"detail"}
    detail_labels = [concept["label"] for concept in sections["detail"]["concept_obligations"]]
    assert "Managed Agents / 托管智能体" in detail_labels
    assert "harness / 适配框架" in detail_labels
    assert "会话/持久上下文" in detail_labels
    assert "Related Pages" not in sections["detail"]["old_text"]
    assert "Other" not in sections["detail"]["old_text"]
    assert "Open questions" not in sections["detail"]["old_text"]


def test_parse_existing_sections_ignores_noncanonical_english_headings() -> None:
    sections = page_sections_module.parse_existing_sections(
        "# Page\n\n"
        "Summary is mentioned in body text but is not a section.\n\n"
        "### Summary\n\n"
        "This tertiary heading should not start a section.\n\n"
        "## Product Summary\n\n"
        "This noncanonical heading should not start a section.\n\n"
        "## Detail\n\n"
        "Actual detail text.\n"
    )

    assert sections == {"detail": "Actual detail text."}


def test_parse_existing_sections_reads_casefold_english_headings() -> None:
    sections = page_sections_module.parse_existing_sections(
        "# Page\n\n"
        "## summary\n\n"
        "Lowercase summary.\n\n"
        "## VALUE POINTS\n\n"
        "Uppercase value points.\n\n"
        "## detail\n\n"
        "Lowercase detail.\n\n"
        "## RELATED PAGES\n\n"
        "Related text.\n"
    )

    assert sections == {
        "summary": "Lowercase summary.",
        "value_points": "Uppercase value points.",
        "detail": "Lowercase detail.",
        "related": "Related text.",
    }


def test_update_preservation_pack_keeps_mixed_placeholder_section_with_core_knowledge() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-UPDATE",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="update",
        canonical_target_path="entities/Entity_Claude Code.md",
        display_title="Claude Code",
        page_type="entity",
        new_understanding="补充产品视角。",
        section_plans={"detail": "详情"},
        reason="测试旧知识保留。",
        matched_page="entities/Entity_Claude Code.md",
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/entities/Entity_Claude Code.md",
                expected_state="present",
                preimage_sha256="old",
                content=(
                    "---\nllmwiki_type: entity\ntitle: Claude Code\nsummary: 旧页。\n---\n\n"
                    "# Claude Code\n\n"
                    "## 详情\n\n"
                    "待补来源：这句是旧页里的占位提醒。\n\n"
                    "Managed Agents / harness 视角强调安全边界、隔离容器、工具权限和会话对象。\n"
                ),
            )
        ],
    )

    pack = update_preservation_module.build_update_preservation_pack(
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
    )

    detail = pack["pages"][0]["sections"][0]
    labels = [concept["label"] for concept in detail["concept_obligations"]]
    assert detail["section_key"] == "detail"
    assert "Managed Agents / 托管智能体" in labels
    assert "待补来源" not in detail["old_text"]


def test_update_preservation_pack_keeps_mixed_empty_placeholder_section_with_core_knowledge() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-UPDATE",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="update",
        canonical_target_path="entities/Entity_Claude Code.md",
        display_title="Claude Code",
        page_type="entity",
        new_understanding="补充产品视角。",
        section_plans={"detail": "详情"},
        reason="测试旧知识保留。",
        matched_page="entities/Entity_Claude Code.md",
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/entities/Entity_Claude Code.md",
                expected_state="present",
                preimage_sha256="old",
                content=(
                    "---\nllmwiki_type: entity\ntitle: Claude Code\nsummary: 旧页。\n---\n\n"
                    "# Claude Code\n\n"
                    "## 详情\n\n"
                    "暂无相关补充。\n\n"
                    "Managed Agents / harness 视角强调安全边界、隔离容器、工具权限和会话对象。\n"
                ),
            )
        ],
    )

    pack = update_preservation_module.build_update_preservation_pack(
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
    )

    detail = pack["pages"][0]["sections"][0]
    labels = [concept["label"] for concept in detail["concept_obligations"]]
    assert detail["section_key"] == "detail"
    assert "Managed Agents / 托管智能体" in labels
    assert "暂无相关补充" not in detail["old_text"]


def test_update_preservation_pack_keeps_core_after_placeholder_prefix_colon() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-UPDATE",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="update",
        canonical_target_path="entities/Entity_Claude Code.md",
        display_title="Claude Code",
        page_type="entity",
        new_understanding="补充产品视角。",
        section_plans={"detail": "详情"},
        reason="测试旧知识保留。",
        matched_page="entities/Entity_Claude Code.md",
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/entities/Entity_Claude Code.md",
                expected_state="present",
                preimage_sha256="old",
                content=(
                    "---\nllmwiki_type: entity\ntitle: Claude Code\nsummary: 旧页。\n---\n\n"
                    "# Claude Code\n\n"
                    "## 详情\n\n暂无相关：Managed Agents / harness 视角强调安全边界。\n"
                ),
            )
        ],
    )

    pack = update_preservation_module.build_update_preservation_pack(
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
    )

    detail = pack["pages"][0]["sections"][0]
    labels = [concept["label"] for concept in detail["concept_obligations"]]
    assert "Managed Agents / 托管智能体" in labels
    assert "暂无相关" not in detail["old_text"]


@pytest.mark.parametrize("placeholder", ["暂无相关补充。", "没有相关补充。", "N/A"])
def test_update_preservation_pack_skips_pure_placeholder_sections(placeholder: str) -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-UPDATE",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="update",
        canonical_target_path="entities/Entity_Claude Code.md",
        display_title="Claude Code",
        page_type="entity",
        new_understanding="补充产品视角。",
        section_plans={"detail": "详情"},
        reason="测试旧知识保留。",
        matched_page="entities/Entity_Claude Code.md",
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/entities/Entity_Claude Code.md",
                expected_state="present",
                preimage_sha256="old",
                content=(
                    "---\nllmwiki_type: entity\ntitle: Claude Code\nsummary: 旧页。\n---\n\n"
                    "# Claude Code\n\n"
                    f"## 详情\n\n{placeholder}\n"
                ),
            )
        ],
    )

    pack = update_preservation_module.build_update_preservation_pack(
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
    )

    assert pack["pages"] == []


def test_update_preservation_pack_skips_placeholder_even_when_it_mentions_known_concept() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-UPDATE",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="update",
        canonical_target_path="entities/Entity_Claude Code.md",
        display_title="Claude Code",
        page_type="entity",
        new_understanding="补充产品视角。",
        section_plans={"detail": "详情"},
        reason="测试旧知识保留。",
        matched_page="entities/Entity_Claude Code.md",
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/entities/Entity_Claude Code.md",
                expected_state="present",
                preimage_sha256="old",
                content=(
                    "---\nllmwiki_type: entity\ntitle: Claude Code\nsummary: 旧页。\n---\n\n"
                    "# Claude Code\n\n"
                    "## 详情\n\n暂无 Managed Agents 相关补充。\n"
                ),
            )
        ],
    )

    pack = update_preservation_module.build_update_preservation_pack(
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
    )

    assert pack["pages"] == []


def test_update_preservation_pack_keeps_english_phrase_with_na_substring() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-UPDATE",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="update",
        canonical_target_path="entities/Entity_Claude Code.md",
        display_title="Claude Code",
        page_type="entity",
        new_understanding="补充产品视角。",
        section_plans={"detail": "详情"},
        reason="测试旧知识保留。",
        matched_page="entities/Entity_Claude Code.md",
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/entities/Entity_Claude Code.md",
                expected_state="present",
                preimage_sha256="old",
                content=(
                    "---\nllmwiki_type: entity\ntitle: Claude Code\nsummary: 旧页。\n---\n\n"
                    "# Claude Code\n\n"
                    "## 详情\n\n"
                    "A/B testing analysis pipeline preserves experiment insights for future product reviews.\n"
                ),
            )
        ],
    )

    pack = update_preservation_module.build_update_preservation_pack(
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
    )

    detail = pack["pages"][0]["sections"][0]
    assert "analysis pipeline" in detail["old_text"]
    assert any("analysis pipeline" in phrase for phrase in detail["key_phrases"])


def test_update_preservation_pack_does_not_use_placeholder_segment_concepts() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-UPDATE",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="update",
        canonical_target_path="entities/Entity_Claude Code.md",
        display_title="Claude Code",
        page_type="entity",
        new_understanding="补充产品视角。",
        section_plans={"detail": "详情"},
        reason="测试旧知识保留。",
        matched_page="entities/Entity_Claude Code.md",
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/entities/Entity_Claude Code.md",
                expected_state="present",
                preimage_sha256="old",
                content=(
                    "---\nllmwiki_type: entity\ntitle: Claude Code\nsummary: 旧页。\n---\n\n"
                    "# Claude Code\n\n"
                    "## 详情\n\n"
                    "暂无 Managed Agents 相关补充。\n\n"
                    "A/B testing analysis pipeline preserves experiment insights for future product reviews.\n"
                ),
            )
        ],
    )

    pack = update_preservation_module.build_update_preservation_pack(
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
    )

    detail = pack["pages"][0]["sections"][0]
    labels = [concept["label"] for concept in detail["concept_obligations"]]
    assert labels == []
    assert "analysis pipeline" in detail["old_text"]
    assert "Managed Agents" not in detail["old_text"]


def test_update_preservation_pack_filters_mixed_ascii_placeholder_segment() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-UPDATE",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="update",
        canonical_target_path="entities/Entity_Claude Code.md",
        display_title="Claude Code",
        page_type="entity",
        new_understanding="补充产品视角。",
        section_plans={"detail": "详情"},
        reason="测试旧知识保留。",
        matched_page="entities/Entity_Claude Code.md",
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/entities/Entity_Claude Code.md",
                expected_state="present",
                preimage_sha256="old",
                content=(
                    "---\nllmwiki_type: entity\ntitle: Claude Code\nsummary: 旧页。\n---\n\n"
                    "# Claude Code\n\n"
                    "## 详情\n\n"
                    "N/A\n\n"
                    "A/B testing analysis pipeline preserves experiment insights for future product reviews.\n"
                ),
            )
        ],
    )

    pack = update_preservation_module.build_update_preservation_pack(
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
    )

    detail = pack["pages"][0]["sections"][0]
    assert "analysis pipeline" in detail["old_text"]
    assert "N/A" not in detail["old_text"]
    assert any("analysis pipeline" in phrase for phrase in detail["key_phrases"])


def test_update_preservation_pack_keeps_only_reusable_core_sections() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-UPDATE",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="update",
        canonical_target_path="entities/Entity_Claude Code.md",
        display_title="Claude Code",
        page_type="entity",
        new_understanding="补充产品视角。",
        section_plans={"detail": "详情"},
        reason="测试旧知识保留。",
        matched_page="entities/Entity_Claude Code.md",
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/entities/Entity_Claude Code.md",
                expected_state="present",
                preimage_sha256="old",
                content=(
                    "---\nllmwiki_type: entity\ntitle: Claude Code\nsummary: 旧页。\n---\n\n"
                    "# Claude Code\n\n"
                    "## 摘要\n\n"
                    "Claude Code 是 Managed Agents 生态中的具体 harness 实现。\n\n"
                    "## 详情\n\n"
                    "来源描述 Claude Code 为 excellent harness that we use widely across tasks，是 Managed Agents 元框架可容纳的多个 harness 之一。\n\n"
                    "## 例子\n\n"
                    "来源在文章末尾提及：Claude Code is an excellent harness。\n\n"
                    "## 价值点\n\n"
                    "展示了 Managed Agents 的开放设计。\n\n"
                    "## 补充观察\n\n"
                    "来源未提供 Claude Code 内部架构细节。\n\n"
                    "## 矛盾与未决问题\n\n"
                    "Claude Code 如何与会话接口集成？是否有特殊要求？（待补来源）\n"
                ),
            )
        ],
    )

    pack = update_preservation_module.build_update_preservation_pack(
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
    )

    section_keys = [section["section_key"] for section in pack["pages"][0]["sections"]]
    assert section_keys == ["detail", "examples"]
    assert pack["pages"][0]["sections"][0]["min_required_matches"] == 0


def test_update_preservation_uses_pack_concepts_when_old_text_is_truncated() -> None:
    tail = "Managed Agents / harness 视角强调安全边界、隔离容器、工具权限和会话对象。"
    old_detail = ("普通背景。" * 400) + tail
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-UPDATE",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="update",
        canonical_target_path="entities/Entity_Claude Code.md",
        display_title="Claude Code",
        page_type="entity",
        new_understanding="补充产品视角。",
        section_plans={"detail": "详情"},
        reason="测试旧知识保留。",
        matched_page="entities/Entity_Claude Code.md",
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/entities/Entity_Claude Code.md",
                expected_state="present",
                preimage_sha256="old",
                content=(
                    "---\nllmwiki_type: entity\ntitle: Claude Code\nsummary: 旧页。\n---\n\n"
                    "# Claude Code\n\n"
                    f"## 详情\n\n{old_detail}\n"
                ),
            )
        ],
    )
    pack = update_preservation_module.build_update_preservation_pack(
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
    )
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-UPDATE",
                action="update",
                canonical_target_path="entities/Entity_Claude Code.md",
                summary="新材料只讨论待办事项列表和发布速度。",
                body_markdown=draft_body(detail="新材料只讨论待办事项列表和发布速度。"),
                change_summary="补充待办事项列表和发布速度。",
                source_coverage_notes="测试。",
            )
        ]
    )

    section = pack["pages"][0]["sections"][0]
    assert tail not in section["old_text"]
    labels = [concept["label"] for concept in section["concept_obligations"]]
    assert "安全边界/权限限制" in labels
    issues = update_preservation_module.update_preservation_issues(draft, pack)

    assert issues
    assert "安全边界/权限限制" in issues[0].message


def test_index_update_uses_snapshot_title_not_model_display_title() -> None:
    metadata = pipeline_module.WikiPageMetadata(
        path="concepts/Concept_X.md",
        llmwiki_type="concept",
        title="旧标题",
        summary="旧摘要。",
        created="2026-01-01",
        updated="2026-01-02",
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-05",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_X.md",
                expected_state="present",
                preimage_sha256="old",
                metadata=metadata,
                content=(
                    "---\n"
                    "llmwiki_type: concept\n"
                    "title: 旧标题\n"
                    "summary: 旧摘要。\n"
                    "updated: 2026-01-02\n"
                    "---\n\n"
                    "# 旧标题\n\n"
                    "## 摘要\n\n"
                    "旧摘要正文。\n"
                ),
            )
        ],
        knowledge_metadata_pool=[
            WikiKnowledgePoolEntry(
                path="wiki/concepts/Concept_X.md",
                rel_path="concepts/Concept_X.md",
                preimage_sha256="old",
                metadata=metadata,
                display_title="旧标题",
                summary="旧摘要。",
                llmwiki_type="concept",
            )
        ],
    )
    plan = pipeline_module.WikiMergePlanArtifact(
        log_date="2026-06-05",
        context_snapshot_ref="wiki_context_snapshot/wiki_context_snapshot.json",
        items=[
            pipeline_module.WikiMergePlanItem(
                page_plan_id="PP-X",
                source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
                action="update",
                canonical_target_path="concepts/Concept_X.md",
                display_title="模型新标题",
                page_type="concept",
                matched_page="concepts/Concept_X.md",
                new_understanding="模型新摘要。",
                section_plans={"summary": "摘要"},
                reason="测试 update 索引标题。",
            )
        ],
    )
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-X",
                action="update",
                canonical_target_path="concepts/Concept_X.md",
                preimage_sha256="old",
                summary="模型新摘要。",
                body_markdown=draft_body(detail="模型新详情。"),
                change_summary="更新页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    profile = type("ProfileStub", (), {"page_types": {"concept": object()}})()

    rows = draft_outputs_module.build_index_rows(profile, plan, draft, snapshot)

    assert rows[0]["title"] == "旧标题"
    assert rows[0]["summary"] == "模型新摘要。"


def test_index_open_questions_keeps_high_signal_and_filters_source_gaps() -> None:
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_Product_Taste.md",
                expected_state="present",
                preimage_sha256="a",
                metadata=pipeline_module.WikiPageMetadata(
                    path="concepts/Concept_Product_Taste.md",
                    llmwiki_type="concept",
                    title="Product Taste",
                    summary="产品品味摘要。",
                    updated="2026-06-05",
                ),
                content="# Product Taste\n\n## 矛盾与未决问题\n\n- 产品品味能否通过系统化训练提升？\n- 待补来源：MIT报告的具体引用。\n",
            ),
            pipeline_module.WikiContextEntry(
                path="wiki/designs/Design_AI_PM.md",
                expected_state="present",
                preimage_sha256="b",
                metadata=pipeline_module.WikiPageMetadata(
                    path="designs/Design_AI_PM.md",
                    llmwiki_type="design",
                    title="AI PM",
                    summary="AI PM 摘要。",
                    updated="2026-06-06",
                ),
                content="# AI PM\n\n## 矛盾与未决问题\n\n- 产品品味能否通过系统化训练提升？\n",
            ),
            pipeline_module.WikiContextEntry(
                path="wiki/open_questions/Open_Question_PM角色.md",
                expected_state="present",
                preimage_sha256="c",
                metadata=pipeline_module.WikiPageMetadata(
                    path="open_questions/Open_Question_PM角色.md",
                    llmwiki_type="open_question",
                    title="PM角色如何演变",
                    summary="PM角色问题。",
                    updated="2026-06-04",
                ),
                content="# PM角色如何演变\n\n## 矛盾与未决问题\n\n- PM 角色在 AI 时代如何演变？\n",
            ),
        ],
    )
    plan = pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", context_snapshot_ref="x", items=[])
    draft = pipeline_module.DraftRenderingArtifact(pages=[])

    rows, report = open_questions_module.build_open_question_rows_with_report(plan, draft, snapshot)
    questions = [row["question"] for row in rows]

    assert questions.count("产品品味能否通过系统化训练提升？") == 1
    assert "PM 角色在 AI 时代如何演变？" in questions
    assert all("待补来源" not in question for question in questions)
    filtered = [item for item in report["items"] if item["decision"] == "filtered"]
    assert any("MIT报告" in item["question"] for item in filtered)


def test_index_open_questions_representative_prefers_high_signal_over_newer_source_gap() -> None:
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_Product_Taste.md",
                expected_state="present",
                preimage_sha256="a",
                metadata=pipeline_module.WikiPageMetadata(
                    path="concepts/Concept_Product_Taste.md",
                    llmwiki_type="concept",
                    title="Product Taste",
                    summary="产品品味摘要。",
                    updated="2026-06-05",
                ),
                content="# Product Taste\n\n## 矛盾与未决问题\n\n- 产品品味能否通过系统化训练提升？\n",
            ),
            pipeline_module.WikiContextEntry(
                path="wiki/designs/Design_AI_PM.md",
                expected_state="present",
                preimage_sha256="b",
                metadata=pipeline_module.WikiPageMetadata(
                    path="designs/Design_AI_PM.md",
                    llmwiki_type="design",
                    title="AI PM",
                    summary="AI PM 摘要。",
                    updated="2026-06-06",
                ),
                content="# AI PM\n\n## 矛盾与未决问题\n\n- 待补来源：产品品味能否通过系统化训练提升？\n",
            ),
        ],
    )

    rows, report = open_questions_module.build_open_question_rows_with_report(
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", context_snapshot_ref="x", items=[]),
        pipeline_module.DraftRenderingArtifact(pages=[]),
        snapshot,
    )

    assert rows[0]["question"] == "产品品味能否通过系统化训练提升？"
    kept = [item for item in report["items"] if item["decision"] == "kept"]
    assert kept[0]["question"] == "产品品味能否通过系统化训练提升？"
    assert kept[0]["occurrences"] == 2


def test_index_open_questions_semantically_dedupes_common_ai_pm_variants() -> None:
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_A.md",
                expected_state="present",
                preimage_sha256="a",
                metadata=pipeline_module.WikiPageMetadata(
                    path="concepts/Concept_A.md",
                    llmwiki_type="concept",
                    title="AI PM A",
                    summary="A。",
                    updated="2026-06-04",
                ),
                content="# AI PM A\n\n## 矛盾与未决问题\n\n- AGI到来后PM角色会消失吗？\n",
            ),
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_B.md",
                expected_state="present",
                preimage_sha256="b",
                metadata=pipeline_module.WikiPageMetadata(
                    path="concepts/Concept_B.md",
                    llmwiki_type="concept",
                    title="AI PM B",
                    summary="B。",
                    updated="2026-06-05",
                ),
                content="# AI PM B\n\n## 矛盾与未决问题\n\n- AGI后PM是否必要？\n",
            ),
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_C.md",
                expected_state="present",
                preimage_sha256="c",
                metadata=pipeline_module.WikiPageMetadata(
                    path="concepts/Concept_C.md",
                    llmwiki_type="concept",
                    title="模型能力",
                    summary="C。",
                    updated="2026-06-06",
                ),
                content="# 模型能力\n\n## 矛盾与未决问题\n\n- 模型能力吞噬产品功能后，产品边界在哪里？\n- 产品功能会被模型能力替代吗？\n",
            ),
        ],
    )
    rows, report = open_questions_module.build_open_question_rows_with_report(
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", context_snapshot_ref="x", items=[]),
        pipeline_module.DraftRenderingArtifact(pages=[]),
        snapshot,
    )

    questions = [row["question"] for row in rows]
    assert sum(1 for question in questions if "AGI" in question and "PM" in question) == 1
    assert sum(1 for question in questions if "模型能力" in question and "产品" in question) == 1
    keys = [item["normalized_key"] for item in report["items"]]
    assert "semantic:agi_pm_role_necessity" in keys
    assert "semantic:model_capability_product_function_boundary" in keys


def test_index_open_questions_semantically_dedupes_product_judgement_training_variants() -> None:
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_Taste.md",
                expected_state="present",
                preimage_sha256="a",
                metadata=pipeline_module.WikiPageMetadata(
                    path="concepts/Concept_Taste.md",
                    llmwiki_type="concept",
                    title="产品品味",
                    summary="A。",
                    updated="2026-06-04",
                ),
                content="# 产品品味\n\n## 矛盾与未决问题\n\n- 产品品味能否通过系统化训练提升？\n",
            ),
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_Judgement.md",
                expected_state="present",
                preimage_sha256="b",
                metadata=pipeline_module.WikiPageMetadata(
                    path="concepts/Concept_Judgement.md",
                    llmwiki_type="concept",
                    title="产品判断",
                    summary="B。",
                    updated="2026-06-05",
                ),
                content="# 产品判断\n\n## 矛盾与未决问题\n\n- 产品判断可以被训练出来吗？\n",
            ),
        ],
    )
    rows, report = open_questions_module.build_open_question_rows_with_report(
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", context_snapshot_ref="x", items=[]),
        pipeline_module.DraftRenderingArtifact(pages=[]),
        snapshot,
    )

    assert len(rows) == 1
    assert rows[0]["question"] == "产品判断可以被训练出来吗？"
    items = [item for item in report["items"] if item["decision"] == "kept"]
    assert items[0]["normalized_key"] == "semantic:product_judgement_training"
    assert items[0]["occurrences"] == 2


def test_index_open_questions_keeps_pm_necessity_and_evolution_separate() -> None:
    necessity = open_questions_module.open_question_key("AGI到来后PM角色会消失吗？")
    evolution = open_questions_module.open_question_key("AI 时代 PM 角色会如何演变？")

    assert necessity == "semantic:agi_pm_role_necessity"
    assert evolution == "semantic:ai_pm_role_evolution"
    assert necessity != evolution


def test_index_open_questions_dedupes_catwu_harness_and_iteration_variants() -> None:
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/entities/Entity_Claude_Code.md",
                expected_state="present",
                preimage_sha256="a",
                metadata=pipeline_module.WikiPageMetadata(
                    path="entities/Entity_Claude_Code.md",
                    llmwiki_type="entity",
                    title="Claude Code",
                    summary="Claude Code。",
                    updated="2026-06-05",
                ),
                content=(
                    "# Claude Code\n\n## 矛盾与未决问题\n\n"
                    "- Claude Code 的产品体验提升，会不会掩盖 harness 安全边界的重要性？\n"
                ),
            ),
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_Fast_Iteration.md",
                expected_state="present",
                preimage_sha256="b",
                metadata=pipeline_module.WikiPageMetadata(
                    path="concepts/Concept_Fast_Iteration.md",
                    llmwiki_type="concept",
                    title="快速迭代",
                    summary="快速迭代。",
                    updated="2026-06-05",
                ),
                content=(
                    "# 快速迭代\n\n## 矛盾与未决问题\n\n"
                    "- 快速发布是否带来质量风险？如何平衡速度与安全？研究预览策略对长期产品一致性有何影响？\n"
                ),
            ),
        ],
    )
    plan = pipeline_module.WikiMergePlanArtifact(
        log_date="2026-06-06",
        context_snapshot_ref="x",
        items=[
            pipeline_module.WikiMergePlanItem(
                page_plan_id="PP-CLAUDE",
                source_basis=SourceBasis(source_candidate_ids=["C1"]),
                action="update",
                canonical_target_path="entities/Entity_Claude_Code.md",
                display_title="Claude Code",
                page_type="entity",
                new_understanding="Claude Code 更新。",
                section_plans={"open_questions": "问题"},
                reason="测试。",
            ),
            pipeline_module.WikiMergePlanItem(
                page_plan_id="PP-ITERATION",
                source_basis=SourceBasis(source_candidate_ids=["C2"]),
                action="create",
                canonical_target_path="concepts/Concept_AI产品快速迭代.md",
                display_title="AI产品快速迭代",
                page_type="concept",
                new_understanding="快速迭代更新。",
                section_plans={"open_questions": "问题"},
                reason="测试。",
            ),
            pipeline_module.WikiMergePlanItem(
                page_plan_id="PP-EVAL",
                source_basis=SourceBasis(source_candidate_ids=["C3"]),
                action="create",
                canonical_target_path="concepts/Concept_Eval.md",
                display_title="Eval",
                page_type="concept",
                new_understanding="Eval。",
                section_plans={"open_questions": "问题"},
                reason="测试。",
            ),
        ],
    )
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-CLAUDE",
                action="update",
                canonical_target_path="entities/Entity_Claude_Code.md",
                summary="摘要。",
                open_questions="Claude Code的产品体验提升会不会掩盖harness安全边界的重要性？Eval的设计如何避免过度拟合？",
                change_summary="更新。",
                source_coverage_notes="测试。",
            ),
            pipeline_module.DraftPageItem(
                page_plan_id="PP-ITERATION",
                action="create",
                canonical_target_path="concepts/Concept_AI产品快速迭代.md",
                summary="摘要。",
                open_questions="快速迭代是否可能牺牲长期质量或安全？研究预览策略如何管理用户预期？流程扩展到更大团队时是否仍有效？",
                change_summary="创建。",
                source_coverage_notes="测试。",
            ),
            pipeline_module.DraftPageItem(
                page_plan_id="PP-EVAL",
                action="create",
                canonical_target_path="concepts/Concept_Eval.md",
                summary="摘要。",
                open_questions="1. Eval的维护成本是否随产品复杂度线性增长？",
                change_summary="创建。",
                source_coverage_notes="测试。",
            ),
        ]
    )

    rows, report = open_questions_module.build_open_question_rows_with_report(plan, draft, snapshot)
    questions = [row["question"] for row in rows]

    assert sum(1 for question in questions if "harness" in question and "安全边界" in question) == 1
    assert sum(1 for question in questions if "研究预览" in question and "安全" in question) == 1
    assert "Eval的维护成本是否随产品复杂度线性增长？" in questions
    assert all(not question.startswith("1.") for question in questions)
    assert report["deduped_count"] >= 2
    keys = [item["normalized_key"] for item in report["items"]]
    assert "semantic:claude_code_product_experience_harness_boundary" in keys
    assert "semantic:rapid_iteration_quality_safety_research_preview" in keys


def test_index_open_questions_dedupes_agent_hand_transfer_but_keeps_concurrency_separate() -> None:
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/entities/Entity_Managed_Agents.md",
                expected_state="present",
                preimage_sha256="a",
                metadata=pipeline_module.WikiPageMetadata(
                    path="entities/Entity_Managed_Agents.md",
                    llmwiki_type="entity",
                    title="Managed Agents",
                    summary="Managed Agents。",
                    updated="2026-06-05",
                ),
                content=(
                    "# Managed Agents\n\n## 矛盾与未决问题\n\n"
                    "- 多大脑间如何高效传递双手（hand）？文中提及但未深入实现细节\n"
                    "- 当多个大脑共享同一双手时，并发和状态同步如何保证？\n"
                ),
            ),
            pipeline_module.WikiContextEntry(
                path="wiki/designs/Design_Managed_Agents.md",
                expected_state="present",
                preimage_sha256="b",
                metadata=pipeline_module.WikiPageMetadata(
                    path="designs/Design_Managed_Agents.md",
                    llmwiki_type="design",
                    title="Managed Agents 架构",
                    summary="Managed Agents 架构。",
                    updated="2026-06-06",
                ),
                content=(
                    "# Managed Agents 架构\n\n## 矛盾与未决问题\n\n"
                    "- 大脑间传递 hand 的具体机制？仅提及“can pass hands”，无细节\n"
                ),
            ),
        ],
    )
    rows, report = open_questions_module.build_open_question_rows_with_report(
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", context_snapshot_ref="x", items=[]),
        pipeline_module.DraftRenderingArtifact(pages=[]),
        snapshot,
    )

    questions = [row["question"] for row in rows]
    assert sum(1 for question in questions if "传递" in question and "hand" in question.lower()) == 1
    assert any("并发和状态同步" in question for question in questions)
    items = [item for item in report["items"] if item["normalized_key"] == "semantic:agent_hand_transfer_mechanism"]
    assert len(items) == 1
    assert items[0]["occurrences"] == 2


def test_source_page_empty_unwritten_section_does_not_repeat_touched_pages() -> None:
    digest = SourceDigestArtifact(
        source_raw_path="raw/sample.md",
        summary="这是一篇用于测试 source 页空态的材料。",
        key_takeaways=["关键收获。"],
    )
    cleanup = pipeline_module.RawLinkCleanupArtifact(
        raw_path="raw/sample.md",
        changed=False,
        pre_cleanup_sha256="pre",
        post_cleanup_sha256="post",
    )

    markdown = draft_outputs_module.render_source_page(
        title="Source sample",
        digest=digest,
        operation_id="ING-TEST",
        linked_pages=["concepts/Concept_A.md"],
        no_change_pages=[],
        log_date="2026-06-06",
        raw_hash="raw-hash",
        prepared_hash="prepared-hash",
        cleanup=cleanup,
    )

    section = markdown.split("## 未写入说明", 1)[1]
    assert "暂无未写入页面。" in section
    assert "触达页面" not in section
    assert "concepts/Concept_A.md" not in section


def test_source_page_unwritten_section_keeps_budget_deferred_candidates() -> None:
    digest = SourceDigestArtifact(
        source_raw_path="raw/sample.md",
        summary="这是一篇用于测试 source 页预算延后摘要的材料。",
        key_takeaways=["关键收获。"],
        budget_deferred_candidates=[
            SourceDigestCandidate(
                candidate_id="C-DEFER",
                name="延后概念",
                type="concept",
                one_sentence_summary="这是一个有价值但本轮不独立建页的概念。",
                why_matters="它可以后续复用。",
                wiki_value="保留为后续总览或对比页素材。",
                suggested_page_title="延后概念",
            ),
            SourceDigestCandidate(
                candidate_id="C-DEFER-2",
                name="延后概念二",
                type="concept",
                one_sentence_summary="第二个延后概念可以与前一个聚合。",
                why_matters="它和前一个概念同属一组。",
                wiki_value="适合先放进概念总览。",
                suggested_page_title="延后概念二",
                resolution_hint="represented_by_aggregation: `AGG-concepts-demo` 已在本轮用聚合候选代表该候选的核心价值。",
            ),
        ],
    )
    cleanup = pipeline_module.RawLinkCleanupArtifact(
        raw_path="raw/sample.md",
        changed=False,
        pre_cleanup_sha256="pre",
        post_cleanup_sha256="post",
    )

    markdown = draft_outputs_module.render_source_page(
        title="Source sample",
        digest=digest,
        operation_id="ING-TEST",
        linked_pages=["concepts/Concept_A.md"],
        no_change_pages=[],
        log_date="2026-06-06",
        raw_hash="raw-hash",
        prepared_hash="prepared-hash",
        cleanup=cleanup,
    )
    section = markdown.split("## 未写入说明", 1)[1]

    assert "### 预算延后候选（未独立建页）" in section
    assert "### 延后候选聚合建议" in section
    assert "C-DEFER" in section
    assert "C-DEFER-2" in section
    assert "保留为后续总览或对比页素材。" in section
    assert "处理提示" in section
    assert "represented_by_aggregation" in section
    assert "concept_overview" in section
    assert "延后概念 等 2 个延后概念聚合页" in section
    assert "concepts/Concept_A.md" not in section


def test_embedding_model_revision_falls_back_to_unknown() -> None:
    class FakeModel:
        model_card_data = "tags:\n- sentence-transformers\nvery long model card"

    assert retrieval_module.model_revision(FakeModel()) == "unknown"


def test_create_draft_with_raw_contradiction_stops_at_draft_review(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    fixture_dir = tmp_path / "grounding-fixture"
    fixture_dir.mkdir()
    for name in ["raw_prepare.json", "source_digest.json", "candidate_resolution.json", "wiki_merge_planning.json", "draft_rendering.json"]:
        data = read_json(FIXTURE_ROOT / "mock" / name)
        if name == "raw_prepare.json":
            data["prepared_markdown"] += "\n\nAnthropic 收购了 OpenAI。"
        if name == "draft_rendering.json":
            data["pages"][0]["body_markdown"] += "\n\nOpenAI 收购了 Anthropic。"
        write_json(fixture_dir / name, data)

    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=fixture_dir, slug="grounding")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    grounding = read_json(run_dir / "draft_rendering" / "draft_grounding_review.json")
    repair_report = read_json(run_dir / "draft_rendering" / "structured_repair_report.json")
    grounding_markdown = (run_dir / "draft_rendering" / "draft_grounding_review.md").read_text(encoding="utf-8")

    assert grounding["requires_review"] is True
    assert grounding["unsupported_new_facts"]
    unsupported = grounding["unsupported_new_facts"][0]
    assert unsupported["text"] == "OpenAI 收购了 Anthropic。"
    assert "明显不符" in unsupported["reason"]
    assert "# 草稿来源支撑审查" in grounding_markdown
    assert "unsupported new_fact" not in grounding_markdown
    assert "Draft Grounding Review" not in grounding_markdown
    assert repair_report["repair_count"] == 2
    assert "触发文本：OpenAI 收购了 Anthropic。" in repair_report["attempts"][0]["issues"][0]["message"]
    assert repair_report["attempts"][1]["repair_prompt_ref"] == "repair_prompts/attempt-2.json"
    assert (run_dir / "draft_rendering" / "repair_prompts" / "attempt-2.json").exists()
    manifest = status(vault, manifest.operation_id)
    assert manifest.status == OperationStatus.awaiting_review
    awaiting_step = [step for step in manifest.steps if step.status == StepStatus.awaiting_review][0]
    assert awaiting_step.name == "draft_review"
    assert "Grounding review" in (awaiting_step.review_reason or "")
    assert not (run_dir / "apply_preview").exists()

    approve_review(vault, manifest.operation_id, "draft_review")
    resumed = resume_ingest(vault=vault, operation_id=manifest.operation_id, from_step="validation")
    preview = read_json(run_dir / "apply_preview" / "apply_preview.json")
    assert resumed.status == OperationStatus.drafted
    assert preview["requires_draft_review"] is True


def test_draft_review_refreshes_stale_grounding_artifacts(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=FIXTURE_ROOT / "mock", slug="stale-grounding")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    draft_manifest_path = run_dir / "draft_rendering" / "draft_write_manifest.json"
    stale_manifest = read_json(draft_manifest_path)
    stale_manifest["requires_grounding_review"] = True
    write_json(draft_manifest_path, stale_manifest)
    write_json(
        run_dir / "draft_rendering" / "draft_grounding_review.json",
        {
            "schema_version": "draft_grounding_review.v1",
            "unsupported_new_facts": [
                {
                    "page_plan_id": "PP-001",
                    "target_path": "concepts/Concept_Knowledge_Compilation.md",
                    "section_key": "detail",
                    "claim_type": "new_fact",
                    "text": "过期误杀",
                    "support": "unsupported",
                    "action": "needs_review",
                    "reason": "旧规则误判。",
                }
            ],
            "claims": [],
            "requires_review": True,
        },
    )

    class Ctx:
        pass

    ctx = Ctx()
    ctx.run_dir = run_dir
    ctx.manifest = read_manifest(run_dir / "manifest.json")
    refreshed = pipeline_module.refresh_current_draft_grounding_artifacts(
        ctx,
        pipeline_module.DraftWriteManifest.model_validate(stale_manifest),
        draft_manifest_path,
    )
    grounding = read_json(run_dir / "draft_rendering" / "draft_grounding_review.json")
    draft_step = [step for step in ctx.manifest.steps if step.name == "draft_rendering"][0]
    write_manifest_ref = [
        ref
        for ref in draft_step.outputs
        if ref.relative_path == "draft_rendering/draft_write_manifest.json"
    ][0]

    assert refreshed.requires_grounding_review is False
    assert grounding["requires_review"] is False
    assert read_json(draft_manifest_path)["requires_grounding_review"] is False
    assert write_manifest_ref.sha256 == sha256_file(draft_manifest_path)


def test_grounding_examples_do_not_require_raw_exact_match_for_generic_prompts() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-X",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_X.md",
        display_title="示例提示",
        page_type="concept",
        new_understanding="示例提示帮助说明 Agent 使用边界。",
        section_plans={"summary": "摘要"},
        reason="测试 grounding。",
    )
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-X",
                action="create",
                canonical_target_path="concepts/Concept_X.md",
                summary="示例提示。",
                body_markdown=draft_body(detail="这个页面说明如何处理通用问题。", examples="- “公司报销政策是什么？”\n- “你是一位客服代表，用礼貌的语气回答。”"),
                change_summary="创建示例提示页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_X.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    review = draft_grounding.build_draft_grounding_review(draft, pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]), snapshot, "")

    assert review.requires_review is False
    assert {claim.claim_type for claim in review.claims} == {"inference"}


def build_examples_grounding_case(
    examples: str,
    *,
    detail: str = "这个页面说明例子 grounding。",
) -> tuple[pipeline_module.DraftRenderingArtifact, pipeline_module.WikiMergePlanArtifact, pipeline_module.WikiContextSnapshot]:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-EXAMPLES",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Examples.md",
        display_title="例子页",
        page_type="concept",
        new_understanding="例子页用于测试 grounding。",
        section_plans={"examples": "例子"},
        reason="测试 grounding examples。",
    )
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="例子页。",
                body_markdown=draft_body(detail=detail, examples=examples),
                change_summary="创建例子页。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_Examples.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    plan = pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item])
    return draft, plan, snapshot


def build_examples_grounding_review(examples: str) -> DraftGroundingReview:
    draft, plan, snapshot = build_examples_grounding_case(examples)
    return draft_grounding.build_draft_grounding_review(
        draft,
        plan,
        snapshot,
        "",
    )


def test_grounding_examples_allow_abstract_placeholder_quotes() -> None:
    review = build_examples_grounding_review("- “某个用户曾在某家店消费过”\n- “该用户表示喜欢某类产品”")

    assert review.requires_review is False
    assert [claim.action for claim in review.warnings] == ["warn", "warn"]


def test_grounding_examples_allow_user_preference_placeholder() -> None:
    review = build_examples_grounding_review("- “用户偏好 X”\n- “<example_id>”\n- “<time_period>”")

    assert review.requires_review is False
    assert [claim.text for claim in review.claims] == ["用户偏好 X", "<example_id>", "<time_period>"]


def test_grounding_examples_allow_abstract_memory_query_literals() -> None:
    review = build_examples_grounding_review(
        '- `recall("用户最近的工单信息")`\n'
        '- `search_context("用户之前提到的项目截止日期")`'
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.claims] == [
        "用户最近的工单信息",
        "用户之前提到的项目截止日期",
    ]
    assert review.requires_review is False


def test_grounding_examples_allow_short_query_template_quotes() -> None:
    review = build_examples_grounding_review(
        "一个问答代理经常被问及“Redis 的安装方法”。\n"
        "用户如果用“怎么安装Redis”询问，语义缓存可命中。\n"
        "类似“查询某个用户的记忆片段”的请求可以作为模板。\n"
        "用户如果用“Redis 连接地址配置方法”询问，也是在描述技术主题。"
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.claims] == [
        "Redis 的安装方法",
        "怎么安装Redis",
        "查询某个用户的记忆片段",
        "Redis 连接地址配置方法",
    ]
    reasons = {claim.text: claim.reason for claim in review.claims}
    assert reasons["Redis 的安装方法"] == "例子区的短查询/请求模板按 illustrative example 处理，不要求 raw exact match。"
    assert reasons["怎么安装Redis"] == "例子区的短查询/请求模板按 illustrative example 处理，不要求 raw exact match。"


@pytest.mark.parametrize(
    "examples",
    [
        "例如“某个用户的账户余额是多少？”",
        "类似“查看某个用户的账户余额”的请求",
        "例如“如何重置密码”",
        "比如“忘记密码怎么办”",
        "类似“query account balance”的请求",
        "例如“reset password”",
    ],
)
def test_grounding_examples_sensitive_dynamic_queries_do_not_block_ingest(examples: str) -> None:
    review = build_examples_grounding_review(examples)

    assert review.requires_review is False
    assert review.unsupported_new_facts == []
    assert all(claim.action != "needs_review" for claim in review.claims)


def test_grounding_detail_sensitive_dynamic_query_does_not_block_ingest() -> None:
    draft, plan, snapshot = build_examples_grounding_case(
        "暂无相关例子记录。",
        detail="语义缓存可以处理常见问题，例如“忘记密码怎么办”。",
    )
    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


@pytest.mark.parametrize(
    "detail",
    [
        "记忆评估问题“某个用户的账户余额是多少？”用于测试召回。",
        "记忆评估问题“我的账户余额是多少？”用于测试召回。",
        "记忆评估问句“查询某个用户的手机号”用于测试召回。",
        "语义缓存可以处理常见问题，例如“密码忘了怎么办”。",
        "语义缓存可以处理常见问题，例如“密码找不回怎么办”。",
        "语义缓存可以处理常见问题，例如“forgot my password”。",
        "语义缓存可以处理常见问题，例如“how to reset my password”。",
        "语义缓存可以处理常见问题，例如“change my password”。",
        "语义缓存可以处理常见问题，例如“cannot login”。",
        "语义缓存可以处理常见问题，例如“登录失败怎么处理”。",
        "语义缓存可以处理常见问题，例如“登录报错怎么处理”。",
        "语义缓存可以处理常见问题，例如“登录问题”。",
        "语义缓存可以处理常见问题，例如“账号问题”。",
        "语义缓存可以处理常见问题，例如“账户问题”。",
        "语义缓存可以处理常见问题，例如“账号登录问题”。",
        "语义缓存可以处理常见问题，例如“login failed”。",
        "语义缓存可以处理常见问题，例如“login error”。",
        "语义缓存可以处理常见问题，例如“login problem”。",
        "语义缓存可以处理常见问题，例如“sign in failed”。",
        "语义缓存可以处理常见问题，例如“sign-in failed”。",
        "语义缓存可以处理常见问题，例如“sign-in problem”。",
        "语义缓存可以处理常见问题，例如“sign-in error”。",
        "语义缓存可以处理常见问题，例如“failed login”。",
        "语义缓存可以处理常见问题，例如“failed sign-in”。",
        "语义缓存可以处理常见问题，例如“error log-in”。",
        "语义缓存可以处理常见问题，例如“log-in failed”。",
        "语义缓存可以处理常见问题，例如“log-in problem”。",
        "语义缓存可以处理常见问题，例如“account login error”。",
        "语义缓存可以处理常见问题，例如“account login problem”。",
        "语义缓存可以处理常见问题，例如“user login problem”。",
        "语义缓存可以处理常见问题，例如“account problems”。",
        "语义缓存可以处理常见问题，例如“user account problems”。",
        "技术排障示例可以写成“登不上账号”。",
        "技术排障示例可以写成“登不上账户”。",
        "技术排障示例可以写成“登不上后台”。",
        '技术排障示例可以写成"登不上账号"。',
        '技术排障示例可以写成"login failed"。',
        '技术排障示例可以写成"login problem"。',
        '技术排障示例可以写成"account problem"。',
        "语义缓存可以处理常见问题，例如“query user emails”。",
        "语义缓存可以处理常见问题，例如“query users emails”。",
        "语义缓存可以处理常见问题，例如“query user addresses”。",
        "语义缓存可以处理常见问题，例如“query user profiles”。",
    ],
)
def test_grounding_detail_memory_examples_sensitive_dynamic_queries_do_not_block_ingest(detail: str) -> None:
    draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。", detail=detail)
    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert review.unsupported_new_facts == []
    assert all(claim.action != "needs_review" for claim in review.claims)


def test_grounding_body_markdown_sensitive_dynamic_query_does_not_block_ingest() -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="语义缓存需要区分稳定知识和动态查询。",
                body_markdown="### 风险边界\n\n用户查询订单状态时，系统不能把这个动态场景当成可缓存知识。",
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


@pytest.mark.parametrize(
    "body_markdown, expected_marker",
    [
        ("### 医疗建议\n\n患者每天服用阿司匹林可以预防心梗。", "阿司匹林"),
        ("### 法律判断\n\n员工签署竞业协议后一定不能加入竞争公司。", "竞业"),
        ("### 金融建议\n\n普通用户应该把大部分存款投入高收益债券。", "存款"),
    ],
)
def test_grounding_body_markdown_high_risk_domain_advice_does_not_block_ingest(
    body_markdown: str,
    expected_marker: str,
) -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="高风险建议需要来源支撑。",
                body_markdown=body_markdown,
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert review.unsupported_new_facts == []
    assert expected_marker


def test_grounding_body_markdown_high_risk_domain_advice_allows_source_supported_claim() -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    sentence = "患者每天服用阿司匹林可以预防心梗。"
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="高风险建议需要来源支撑。",
                body_markdown=f"### 医疗建议\n\n{sentence}",
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, sentence)

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


def test_grounding_open_questions_high_risk_domain_gap_is_not_blocked_as_fact() -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="高风险建议需要来源支撑。",
                body_markdown="### 边界\n\n这里不把高风险建议写成事实。",
                open_questions="- 待补来源：患者每天服用阿司匹林是否可以预防心梗？",
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False


def test_grounding_low_risk_open_question_quote_warns_without_review() -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="PM 角色演变仍待讨论。",
                body_markdown="### 背景\n\n这里把问题保留为待研究方向。",
                open_questions="- 待补来源：“AGI后PM是否必要？”",
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == ["AGI后PM是否必要？"]
    assert review.warnings[0].action == "warn"


def test_grounding_low_risk_body_quote_warns_without_review() -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="产品直觉需要长期训练。",
                body_markdown="### 表达方式\n\n这里把“好的产品判断往往来自长期实践中形成的经验直觉”当作一个低风险表述来记录。",
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == ["好的产品判断往往来自长期实践中形成的经验直觉"]


@pytest.mark.parametrize(
    "body_markdown, expected",
    [
        ("### 公司关系\n\n“OpenAI 收购了 Anthropic”是一个需要来源支撑的公司关系。", "OpenAI 收购了 Anthropic"),
        ("### 身份关系\n\n“Sam Altman 担任 Anthropic CEO”是一个需要来源支撑的身份关系。", "Sam Altman 担任 Anthropic CEO"),
        ("### 产品关系\n\n“Claude 由 Google 发布”是一个需要来源支撑的产品关系。", "Claude 由 Google 发布"),
    ],
)
def test_grounding_severe_factual_relationship_quote_warns_without_raw_contradiction(body_markdown: str, expected: str) -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="严重事实关系需要来源支撑。",
                body_markdown=body_markdown,
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == [expected]


@pytest.mark.parametrize(
    "body_markdown, expected",
    [
        ("### 公司关系\n\nOpenAI 收购了 Anthropic。", "OpenAI 收购了 Anthropic。"),
        ("### 身份关系\n\nSam Altman 担任 Anthropic CEO。", "Sam Altman 担任 Anthropic CEO。"),
        ("### 产品关系\n\nClaude 由 Google 发布。", "Claude 由 Google 发布。"),
    ],
)
def test_grounding_severe_factual_relationship_unquoted_warns_without_raw_contradiction(body_markdown: str, expected: str) -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="严重事实关系需要来源支撑。",
                body_markdown=body_markdown,
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == [expected]


def test_grounding_severe_factual_relationship_quote_warning_is_not_duplicated_by_unquoted_scan() -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="严重事实关系需要来源支撑。",
                body_markdown="### 公司关系\n\n“OpenAI 收购了 Anthropic”是一个需要来源支撑的公司关系。",
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == ["OpenAI 收购了 Anthropic"]


def test_grounding_severe_factual_relationship_blocks_when_raw_contradicts() -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="严重事实关系与 raw 不符才阻断。",
                body_markdown="### 公司关系\n\nOpenAI 收购了 Anthropic。",
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "Anthropic 收购了 OpenAI。")

    assert review.requires_review is True
    assert [claim.text for claim in review.unsupported_new_facts] == ["OpenAI 收购了 Anthropic。"]
    assert "明显不符" in review.unsupported_new_facts[0].reason


def test_grounding_scans_body_markdown_heading_text_for_contradictions() -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="标题里的事实关系也需要来源一致。",
                body_markdown="### OpenAI 收购了 Anthropic\n\n正文只补充说明这个标题。",
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "Anthropic 收购了 OpenAI。")

    assert review.requires_review is True
    assert [claim.text for claim in review.unsupported_new_facts] == ["OpenAI 收购了 Anthropic"]
    assert "明显不符" in review.unsupported_new_facts[0].reason


def test_grounding_role_relationship_blocks_when_raw_names_different_org() -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="身份关系与 raw 不符才阻断。",
                body_markdown="### 身份关系\n\nSam Altman 担任 Anthropic CEO。",
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "Sam Altman 担任 OpenAI CEO。")

    assert review.requires_review is True
    assert [claim.text for claim in review.unsupported_new_facts] == ["Sam Altman 担任 Anthropic CEO。"]


def test_grounding_severe_factual_relationship_blocks_explicit_raw_negation() -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="显式否定与肯定冲突才阻断。",
                body_markdown="### 公司关系\n\nOpenAI 收购了 Anthropic。",
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "OpenAI 没有收购 Anthropic。")

    assert review.requires_review is True
    assert [claim.text for claim in review.unsupported_new_facts] == ["OpenAI 收购了 Anthropic。"]


def test_grounding_creator_relationship_blocks_same_object_different_creator() -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="同一对象不同创建方与 raw 不符才阻断。",
                body_markdown="### 创建关系\n\nOpenAI 创建了 Claude。",
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "Anthropic 创建了 Claude。")

    assert review.requires_review is True
    assert [claim.text for claim in review.unsupported_new_facts] == ["OpenAI 创建了 Claude。"]


def test_grounding_release_relationship_same_subject_different_object_only_warns() -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="同一主体发布另一个对象只是缺支撑，不算 raw 矛盾。",
                body_markdown="### 发布关系\n\nOpenAI 发布了 Sora。",
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "OpenAI 发布了 ChatGPT。")

    assert review.requires_review is False
    assert review.unsupported_new_facts == []
    assert [claim.text for claim in review.warnings] == ["OpenAI 发布了 Sora。"]


@pytest.mark.parametrize(
    ("body_markdown", "approved_raw", "expected"),
    [
        ("### 创建关系\n\nClaude 由 OpenAI 创建。", "Claude 由 Anthropic 创建。", "Claude 由 OpenAI 创建。"),
        ("### 发布关系\n\nSora 由 OpenAI 发布。", "Sora 由 Anthropic 发布。", "Sora 由 OpenAI 发布。"),
        ("### Creation\n\nClaude was developed by OpenAI.", "Claude was developed by Anthropic.", "Claude was developed by OpenAI."),
    ],
)
def test_grounding_by_actor_relationship_blocks_same_object_different_actor(
    body_markdown: str,
    approved_raw: str,
    expected: str,
) -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="由某方创建/发布的同一对象不同主体与 raw 不符才阻断。",
                body_markdown=body_markdown,
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, approved_raw)

    assert review.requires_review is True
    assert [claim.text for claim in review.unsupported_new_facts] == [expected]


@pytest.mark.parametrize(
    ("body_markdown", "approved_raw"),
    [
        ("### 收购关系\n\nAnthropic 被 OpenAI 收购。", "OpenAI 收购了 Anthropic。"),
        ("### 创建关系\n\nClaude 由 Anthropic 创建。", "Anthropic 创建了 Claude。"),
    ],
)
def test_grounding_active_passive_paraphrase_does_not_count_as_raw_contradiction(
    body_markdown: str,
    approved_raw: str,
) -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="主动/被动同义转述不应被当成 raw 矛盾。",
                body_markdown=body_markdown,
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, approved_raw)

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


def test_grounding_severe_factual_relationship_unquoted_allows_source_supported_claim() -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    sentence = "OpenAI 收购了 Anthropic。"
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="严重事实关系如果来自来源则保留。",
                body_markdown=f"### 公司关系\n\n{sentence}",
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, sentence)

    assert review.requires_review is False
    assert [claim.action for claim in review.claims if claim.text == sentence] == ["kept"]


@pytest.mark.parametrize(
    "body_markdown",
    [
        "### 技术解释\n\nAI Agent 由模型、工具和记忆组成。",
        "### 技术解释\n\nPython 支持异步编程。",
        "### 技术解释\n\nRedis 支持语义缓存。",
    ],
)
def test_grounding_weak_technical_relationships_do_not_require_review(body_markdown: str) -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="普通技术解释不应该被严重事实关系误杀。",
                body_markdown=body_markdown,
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


@pytest.mark.parametrize(
    "body_markdown",
    [
        "### 导入流程\n\nLLM 读取源文档，提取关键信息，并更新或创建相关 wiki 页面。",
        "### Wiki 层\n\nLLM 完全拥有这一层：创建、更新、删除页面，维护交叉引用，保持一致性。",
        "### 摘要撰写\n\nLLM 在 wiki 中创建该源的摘要页面，记录来源信息及主要贡献。",
        "### 查询流程\n\n当用户向 wiki 提出问题时，LLM 会搜索相关页面并合成答案。",
        "### 自定义工具\n\n开发者可以注册自定义工具（使用 `@register_tool` 装饰器），例如创建一个图像生成工具，然后实例化 `Assistant` 并配置 LLM 服务。",
        "### 智能体创建\n\n3. **创建智能体**：通过 `Assistant` 类实例化，集成工具使用与文件读取能力。",
    ],
)
def test_grounding_wiki_operation_create_and_question_flow_do_not_require_review(body_markdown: str) -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="Wiki 操作流程不应该被严重事实关系误杀。",
                body_markdown=body_markdown,
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


@pytest.mark.parametrize(
    "body_markdown",
    [
        "### LLM 层\n\n通过 `BaseChatModel` 基类封装大语言模型接口，提供统一的 `chat` 方法，支持流式输出和函数调用。",
        "### 工具调用\n\n默认模板支持并行工具调用。",
        "### API 兼容\n\nQwen-Agent 支持 OpenAI-compatible API。",
    ],
)
def test_grounding_technical_support_capabilities_do_not_require_review(body_markdown: str) -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="技术能力说明不应该被支持关系误杀。",
                body_markdown=body_markdown,
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


@pytest.mark.parametrize(
    "body_markdown",
    [
        "### 可选依赖\n\n支持可选依赖，如 GUI（基于 Gradio）、RAG（检索增强生成）、代码解释器、MCP（模型上下文协议）等。",
        "### MCP 集成\n\nMCP 集成：支持模型上下文协议，使用 MCP 工具需要安装 Node.js、uv、Git 等依赖（详见 README）。",
        "### 版本更新\n\n框架持续更新，近期版本包括 Qwen3.5 支持、DeepPlanning 评测基准发布等。",
        "### 智能体创建\n\n`Assistant` 是一个能够使用工具并读取文件的智能体，其创建示例如下（来自源代码步骤 3）：",
        "### 工具循环\n\n工具调用后，结果返回给 Agent，再由 Agent 决定下一步动作。",
        "### 初始化示例\n\n示例中通过 `Assistant(llm, function_list, files)` 创建（见 README 步骤3代码）。",
        "### 模板配置\n\n工具调用支持多种模板，通过 `fncall_prompt_type` 参数配置，默认为 `nous`（Qwen3 推荐）。",
        "### 代码解释器\n\n当智能体决定使用代码解释器时，框架会在本地 Docker 环境中创建一个隔离容器。",
        "### 框架定位\n\nQwen-Agent 是一个基于 Qwen 模型的 Agent 开发框架，提供 LLM、Tool、Agent 等组件，支持自定义工具、代码解释器、MCP 集成，并作为 Qwen Chat 的后端运行。",
        "### 模型服务\n\nQwen-Agent 支持接入阿里云 DashScope 服务提供的 Qwen 模型服务，也支持通过 OpenAI API 方式接入开源的 Qwen 模型服务。",
        "### 工具解析\n\n部署时注意：对于 QwQ 和 Qwen3 模型，建议不开启 vLLM 的 `--enable-auto-tool-choice` 和 `--tool-call-parser hermes`，由 Qwen-Agent 自行解析工具输出。",
        "### DeepPlanning\n\nDeepPlanning 是用于评估 Agent 规划能力的开源基准测试，由 Qwen 团队发布。",
        "### DeepPlanning\n\nDeepPlanning 是一个用于评估大语言模型智能体规划能力的开源基准测试，由 Qwen 团队在 2026 年 1 月发布。",
    ],
)
def test_grounding_qwen_agent_technical_documentation_does_not_require_review(body_markdown: str) -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="Qwen-Agent README 的技术说明不应该被严重事实关系误杀。",
                body_markdown=body_markdown,
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


def test_grounding_allows_placeholder_api_key_in_configuration_example() -> None:
    draft, plan, snapshot = build_examples_grounding_case(
        "暂无相关例子记录。",
        detail=(
            "2. 配置 LLM，例如使用 DashScope："
            "`{'model': 'qwen3-32b', 'model_type': 'qwen_dashscope', 'api_key': '<DASHSCOPE_API_KEY>'}`。"
        ),
    )
    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False


def test_grounding_blocks_real_api_key_literal_in_configuration_example() -> None:
    draft, plan, snapshot = build_examples_grounding_case(
        "暂无相关例子记录。",
        detail="示例配置里写了 api_key: sk-live-secret-value。",
    )
    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


@pytest.mark.parametrize(
    "detail",
    [
        "配置：api_key: <DASHSCOPE_API_KEY>; token: sk-live-secret-value。",
        "配置：token: sk-live-secret-value; api_key: <DASHSCOPE_API_KEY>。",
    ],
)
def test_grounding_placeholder_secret_does_not_hide_real_secret(detail: str) -> None:
    draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。", detail=detail)
    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


@pytest.mark.parametrize(
    "body_markdown, expected_marker",
    [
        ("### 方案归属\n\nKarpathy 提出 llm-wiki 方案。", "提出"),
        ("### 产品归属\n\nOpenAI 创建了一个 Anthropic 竞品。", "创建"),
        ("### 服务关系\n\nOpenAI 支持 Anthropic 服务。", "支持"),
        ("### 产品关系\n\nOpenAI 创建了 Assistant 产品。", "创建"),
    ],
)
def test_grounding_real_create_and_propose_relationships_warn_without_raw_contradiction(
    body_markdown: str,
    expected_marker: str,
) -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="真实创建/提出关系缺支撑时只提醒。",
                body_markdown=body_markdown,
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert review.unsupported_new_facts == []
    assert review.warnings[0].action == "warn"
    assert expected_marker in review.warnings[0].reason


def test_grounding_security_sandbox_advice_does_not_require_review() -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="防御性安全工程描述不应自动 review。",
                body_markdown=(
                    "### 沙箱边界\n\n"
                    "Docker 沙箱提供了一定程度的隔离，但生产环境中可能需要更严格的沙箱方案（如 gVisor、Firecracker）来增强安全性。"
                ),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


def test_grounding_quote_uses_sentence_context_without_blocking_high_risk_domain() -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="高风险建议需要来源支撑。",
                body_markdown="### 医疗建议\n\n医疗上通常推荐“方案A”。",
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


@pytest.mark.parametrize(
    "body_markdown",
    [
        "### 安全协作\n\nAI 安全团队可以把评估结果作为产品发布节奏的参考。",
        "### 金融产品表达\n\n金融产品可以用更清晰的解释降低用户理解成本。",
        "### 法务协作\n\n法务团队可以提前参与合同评审流程。",
        "### 医疗表述\n\n这段材料适合讨论阿司匹林风险。",
        "### 金融表述\n\n这段材料适合讨论高收益债券。",
        "### 法律表述\n\n这段材料适用于合同评审案例。",
        "### 安全表述\n\n这个案例适合讨论明文密码风险。",
        "### Security Framing\n\nThis setup is suitable for discussing passwords in plaintext.",
    ],
)
def test_grounding_high_risk_domain_meta_statements_do_not_require_review(body_markdown: str) -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="普通领域元叙述不应该被高风险建议误杀。",
                body_markdown=body_markdown,
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


@pytest.mark.parametrize(
    "body_markdown",
    [
        "### 医疗建议\n\n患者可以服用阿司匹林。",
        "### 医疗建议\n\n患者可以吃阿司匹林。",
        "### 医疗建议\n\n患者可以口服阿司匹林。",
        "### Medical Advice\n\nPatients can use aspirin for chest pain.",
        "### 金融建议\n\n普通用户可以投资高收益债券。",
        "### 金融建议\n\n普通用户可以配置高收益理财产品。",
        "### 法律建议\n\n法律上可以解除这份合同。",
        "### 法律建议\n\n员工可以起诉雇主要求赔偿。",
        "### 法律建议\n\n员工可以签竞业协议。",
        "### Legal Advice\n\nYou can sue your employer without an attorney.",
        "### 安全建议\n\n管理员可以绕过认证后删除日志。",
        "### 安全建议\n\n管理员可以删日志。",
        "### 安全建议\n\n管理员推荐用于删除日志。",
        "### Security Advice\n\nUsers can store passwords in plaintext for convenience.",
        "### Security Advice\n\nThis setup is recommended for storing passwords in plaintext.",
    ],
)
def test_grounding_high_risk_domain_actionable_can_statements_do_not_block_ingest(body_markdown: str) -> None:
    _draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-EXAMPLES",
                action="create",
                canonical_target_path="concepts/Concept_Examples.md",
                summary="可执行高风险建议需要来源支撑。",
                body_markdown=body_markdown,
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


def test_render_draft_grounding_review_shows_warning_section() -> None:
    claim = GroundingClaim(
        page_plan_id="PP-WARN",
        target_path="concepts/Concept_Warn.md",
        section_key="detail",
        claim_type="new_fact",
        text="好的产品判断往往来自长期实践中形成的经验直觉",
        support="unsupported",
        action="warn",
        reason="低风险未支撑引号内容仅记录为 warning，不阻塞自动 ingest；如需严谨可人工回看来源。",
    )
    review = DraftGroundingReview(warnings=[claim], claims=[claim], requires_review=False)

    markdown = draft_grounding.render_draft_grounding_review(review)

    assert "- 结果：通过，有非阻塞提醒" in markdown
    assert "- 非阻塞提醒数量：1" in markdown
    assert "## 非阻塞提醒" in markdown
    assert "好的产品判断往往来自长期实践中形成的经验直觉" in markdown


@pytest.mark.parametrize(
    "detail",
    [
        "例如，在 AI 代理的客服场景中，用户反复询问与某个订单状态相关的相似问题时，语义缓存可识别语义相似性。",
        "例如，在 Redis 语义缓存场景中，用户反复询问与某个订单状态相关的相似问题时，语义缓存可复用回答。",
        "例如，Redis 语义缓存可以帮助客服处理用户想修改那个订单的地址的请求。",
        "例如，客服排查用户登录问题时会查询登录状态。",
        "例如，API 场景中用户反复询问某个订单状态。",
        "例如，客服接口场景中用户想修改那个订单的地址。",
        "例如，参数配置场景中客户查询订单信息。",
        "例如，API 场景中用户查看某个订单状态。",
        "例如，接口场景中客户获取订单信息。",
        "例如，参数配置场景中用户搜索订单状态。",
        "例如，客服 API 中客户申请订单退款。",
        "In a Redis semantic cache scenario, a user asks about order status repeatedly.",
        "In an API scenario, a user asks about order status repeatedly.",
        "In an API scenario, a user checks order status.",
        "In a support API scenario, a customer looks up order details.",
        "场景：用户想修改那个订单的地址，代理需要关联长期记忆。",
        "在连续对话场景中，假设用户先询问某个过去的订单信息，随后用户说修改那个订单的地址。",
    ],
)
def test_grounding_detail_unquoted_dynamic_user_scenarios_do_not_block_ingest(detail: str) -> None:
    draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。", detail=detail)
    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


def test_grounding_examples_unquoted_dynamic_user_scenario_does_not_block_ingest() -> None:
    review = build_examples_grounding_review("例如用户反复询问与某个订单状态相关的相似问题。")

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


def test_grounding_sensitive_dynamic_query_passes_when_source_supported() -> None:
    draft, plan, snapshot = build_examples_grounding_case(
        "暂无相关例子记录。",
        detail="客服文档原文示例是“忘记密码怎么办”。",
    )
    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "忘记密码怎么办")

    assert review.requires_review is False
    assert any(claim.text == "忘记密码怎么办" and claim.support == "raw" for claim in review.claims)


@pytest.mark.parametrize(
    ("detail", "raw"),
    [
        ('客服文档原文示例是"登不上账号"。', "登不上账号"),
        ('客服文档原文示例是"login problem"。', "login problem"),
        ('客服文档原文示例是"sign-in problem"。', "sign-in problem"),
        ("客服文档原文示例是“账号问题”。", "账号问题"),
    ],
)
def test_grounding_sensitive_dynamic_query_short_quote_passes_when_source_supported(detail: str, raw: str) -> None:
    draft, plan, snapshot = build_examples_grounding_case(
        "暂无相关例子记录。",
        detail=detail,
    )
    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, raw)

    assert review.requires_review is False
    assert any(claim.text == raw and claim.support == "raw" for claim in review.claims)


def test_grounding_unquoted_dynamic_user_scenario_passes_when_source_supported() -> None:
    detail = "例如，在 AI 代理的客服场景中，用户反复询问与某个订单状态相关的相似问题时，语义缓存可识别语义相似性。"
    draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。", detail=detail)
    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, detail)

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


def test_grounding_examples_still_allow_safe_technical_query_template_after_sensitive_guard() -> None:
    review = build_examples_grounding_review("类似“Redis 地址配置方法”的请求")

    assert review.requires_review is False
    assert [claim.text for claim in review.claims] == ["Redis 地址配置方法"]


@pytest.mark.parametrize(
    "detail",
    [
        "技术排障示例可以写成“无法连接 Redis”。",
        "技术排障示例可以写成“不能安装 Redis”。",
        "技术排障示例可以写成“打不开配置文件”。",
        "技术排障示例可以写成“Redis 无法启动”。",
        '技术排障示例可以写成"cache"。',
        '技术排障示例可以写成"Redis config"。',
        '技术排障示例可以写成"Redis problem"。',
        '技术排障示例可以写成"config problem"。',
        '技术排障示例可以写成"cache problem"。',
        '技术排障示例可以写成"service account config"。',
        '技术排障示例可以写成"service account issue"。',
        '技术排障示例可以写成"service account error"。',
        '技术排障示例可以写成"service account problems"。',
        '技术排障示例可以写成"login configuration"。',
        '技术排障示例可以写成"sign-in configuration"。',
        '技术排障示例可以写成"log-in configuration"。',
        "例如，Redis 配置问题可以通过文档排查。",
        "订单状态字段用于排序。",
        "订单状态 schema 示例用于说明字段。",
        "服务会缓存订单状态字段。",
        "订单状态 API 示例用于说明接口。",
        "OAuth 回调接口说明包含订单状态参数。",
        "Redis 数据库字段 order_status 用于缓存订单状态。",
        "用户字段 API 参数说明包含 user_id。",
        "API docs show order status lookup parameters.",
        "payment API parameter describes refund status.",
    ],
)
def test_grounding_detail_allows_safe_technical_troubleshooting_queries(detail: str) -> None:
    draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。", detail=detail)
    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False


def test_grounding_unquoted_dynamic_scenario_scanner_ignores_quoted_claims() -> None:
    review = build_examples_grounding_review("例如“查看某个用户的账户余额”这类问题。")

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


def test_grounding_unquoted_dynamic_scenario_scanner_ignores_open_questions_section() -> None:
    draft, plan, snapshot = build_examples_grounding_case("暂无相关例子记录。")
    page = draft.pages[0]
    draft = draft.model_copy(
        update={"pages": [page.model_copy(update={"open_questions": "- 待补来源：用户订单状态场景是否适合语义缓存？"})]}
    )
    review = draft_grounding.build_draft_grounding_review(draft, plan, snapshot, "")

    assert review.requires_review is False


def test_grounding_examples_query_template_without_local_context_warns() -> None:
    review = build_examples_grounding_review("- “Redis 的安装方法”")

    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == ["Redis 的安装方法"]


def test_grounding_examples_query_template_direct_quote_warns_without_support() -> None:
    review = build_examples_grounding_review("原文称：“Redis 的安装方法”。")

    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == ["Redis 的安装方法"]


@pytest.mark.parametrize(
    "examples",
    [
        "类似“查询某个用户的记忆片段”的请求",
        "类似“查询某个用户的记忆片段”的请求可以作为模板。",
        "类似“Redis setup guide”的请求",
        "类似“query Redis memory”的请求",
        "类似“query redis memory”的请求",
        "类似“查询Redis记忆”的请求",
        "类似“Redis IP address config”的请求",
        "类似“Redis address configuration”的请求",
        "类似“Redis configuration guide”的请求",
    ],
)
def test_grounding_examples_allow_isolated_query_template_contexts(examples: str) -> None:
    review = build_examples_grounding_review(examples)

    assert review.requires_review is False


@pytest.mark.parametrize(
    "examples",
    [
        "类似“Redis 支持集群模式”的问题",
        "类似“用户 1234 删除了凭证”的请求",
        "类似“Alice uses MacBook in 2026”的请求",
        "类似“Alice uses MacBook”的请求",
        "类似“用户使用华为手机”的请求",
        "类似“查询王小明的记忆片段”的请求",
        "类似“查询某个用户的手机号”的请求",
        "类似“查询某个用户的邮箱地址”的请求",
        "类似“查询用户登录记录”的请求",
        "类似“query a user's email address”的请求",
        "类似“查询Alice的记忆片段”的请求",
        "类似“查询Charlie的记忆片段”的请求",
        "类似“查询alice的记忆片段”的请求",
        "类似“查询alice的对话摘要”的请求",
        "类似“查找Alice记忆片段”的请求",
        "类似“查找王小明记忆片段”的请求",
        "类似“query a user's address”的请求",
        "类似“query user's address”的请求",
        "类似“query customer address”的请求",
        "类似“query user IP address”的请求",
        "类似“query a user's birthday”的请求",
        "类似“query a user's name”的请求",
        "类似“query person profile”的请求",
        "类似“query user IP”的请求",
        "类似“query customer IP”的请求",
        "类似“query users profiles”的请求",
        "类似“query users addresses”的请求",
        "类似“search customer profiles”的请求",
        "类似“find user addresses”的请求",
        "类似“query user IP config”的请求",
        "类似“query users IPs”的请求",
        "类似“query users emails”的请求",
        "类似“query customer cookies”的请求",
        "类似“query user IDs”的请求",
        "类似“query customer card”的请求",
        "类似“query user credit card”的请求",
        "类似“query users tokens”的请求",
        "类似“query user sessions”的请求",
        "类似“query user passwords”的请求",
        "类似“query people's addresses”的请求",
        "类似“find people profiles”的请求",
        "类似“search persons addresses”的请求",
        "类似“query people locations”的请求",
        "类似“query users' emails”的请求",
        "类似“search persons' addresses”的请求",
        "类似“Redis config supports cluster”的请求",
        "类似“Redis config improves latency”的请求",
        "类似“Redis configuration is best”的请求",
        "例如“Redis config supports cluster”",
        "比如“Redis config improves latency”",
        "示例“Redis configuration is best”",
        "例如“Redis supports cluster”",
        "例如“Redis 支持集群模式”",
        "比如“Redis 最佳实践”",
        "示例“Redis 配置是最佳方案”",
        "例如“Redis 已经发布新功能”",
        "例如“Redis 推出企业版”",
        "例如“Redis 配置推荐用于生产环境”",
        "比如“Redis 配置导致错误”",
        "类似“query user SSN”的请求",
        "类似“query users SSNs”的请求",
        "类似“query user social security number”的请求",
        "类似“query customer passport number”的请求",
        "类似“query customer license number”的请求",
        "类似“query user api key”的请求",
        "类似“query user API key”的请求",
        "类似“query user api keys”的请求",
        "类似“query users API keys”的请求",
        "类似“query user secrets”的请求",
        "类似“query customer passwds”的请求",
        "例如“query users API keys”",
    ],
)
def test_grounding_examples_query_template_does_not_block_without_raw_contradiction(examples: str) -> None:
    review = build_examples_grounding_review(examples)

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


@pytest.mark.parametrize(
    "examples",
    [
        "- “权限配置方法”",
        "例如“权限配置方法”",
        "例如“payment setup guide”",
        "类似“payment setup guide”的请求",
    ],
)
def test_grounding_examples_sensitive_quotes_do_not_block_ingest(examples: str) -> None:
    review = build_examples_grounding_review(examples)

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


@pytest.mark.parametrize(
    "examples",
    [
        "问句“查询某个用户的手机号”用于评估。",
        "记忆评估问题“用户登录记录是什么？”",
        "记忆评估问题“王小明的手机号是多少？”",
    ],
)
def test_grounding_examples_sensitive_memory_eval_quotes_do_not_block_ingest(examples: str) -> None:
    review = build_examples_grounding_review(examples)

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


@pytest.mark.parametrize(
    "examples",
    [
        "- “该用户喜欢蓝色”",
        "- “Alice 在 2026 年 3 月购买了 MacBook。”",
        "- “Build number 1234 completed with status success”",
        "- 原文称：“某个用户曾在某家店消费过”",
        "- “Alice uses MacBook”",
        "- “Alice likes coffee”",
        "- “MacBook syncs memory”",
        "- “某个用户在星巴克消费过”",
        "- “某个用户购买了华为手机”",
        "- “某个用户在北京门店消费过”",
        "- “某个用户购买了小米手机”",
        "- “某个用户在南京门店消费过”",
        "- “某个用户喜欢黄色”",
        "- “某个用户购买了OPPO手机”",
        "- “某个用户在成都门店消费过”",
        "- “某个用户喜欢紫色”",
        '- `recall("张三的工单 1234")`',
        '- `search_context("Alice order 1234")`',
        '- `recall("用户喜欢蓝色")`',
        '- `recall("某个用户在北京门店消费过")`',
        '- `recall("用户最近的订单状态")`',
        '- `recall("查询某个用户的手机号")`',
        '- `mem0 search "用户最近的工单信息" --user-id user123`',
    ],
)
def test_grounding_examples_placeholder_bypass_warns_for_concrete_or_attributed_quotes(examples: str) -> None:
    review = build_examples_grounding_review(examples)

    assert review.requires_review is False
    assert review.unsupported_new_facts == []
    assert review.warnings or review.claims


def test_grounding_examples_hard_facts_warn_without_support() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-FACT-EXAMPLE",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Fact_Example.md",
        display_title="事实例子",
        page_type="concept",
        new_understanding="事实例子需要来源支撑。",
        section_plans={"summary": "摘要"},
        reason="测试 grounding。",
    )
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-FACT-EXAMPLE",
                action="create",
                canonical_target_path="concepts/Concept_Fact_Example.md",
                summary="事实例子。",
                body_markdown=draft_body(detail="这个页面说明事实型例子需要来源。", examples="- “销量增长三倍”"),
                change_summary="创建事实例子页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_Fact_Example.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )

    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == ["销量增长三倍"]


def test_grounding_examples_cli_argument_literals_warn_without_repair() -> None:
    review = build_examples_grounding_review('- `mem0 add --user-id user123 --text "用户喜欢科技类文章"`')

    assert review.requires_review is False
    assert review.warnings


def test_grounding_detail_illustrative_examples_do_not_require_raw_exact_match() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-STYLE",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Style.md",
        display_title="写作风格",
        page_type="concept",
        new_understanding="写作风格描述 AI 上下文文件中的表达方式。",
        section_plans={"summary": "摘要"},
        reason="测试 grounding。",
    )
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-STYLE",
                action="create",
                canonical_target_path="concepts/Concept_Style.md",
                summary="写作风格示例。",
                body_markdown=draft_body(detail="解释性风格提供理由，如“因为性能原因，使用列表推导”；条件性风格指定条件，如“如果代码量超过 100 行，请拆分”。", examples="暂无相关例子记录。"),
                change_summary="创建写作风格页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_Style.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == ["因为性能原因，使用列表推导", "如果代码量超过 100 行，请拆分"]


def test_grounding_memory_example_questions_do_not_require_raw_exact_match() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-MEMORY",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_记忆评估示例.md",
        display_title="记忆评估示例",
        page_type="concept",
        new_understanding="记忆评估常用短问句和偏好样例解释不同记忆层级。",
        section_plans={"summary": "摘要"},
        reason="测试 grounding memory examples。",
    )
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-MEMORY",
                action="create",
                canonical_target_path="concepts/Concept_记忆评估示例.md",
                summary="MemBench 用短样例解释不同记忆任务。",
                body_markdown=draft_body(detail="事实记忆的问题示例包括“用户哥哥的名字是什么？”，反思记忆示例包括“用户喜欢重口味”。", examples="暂无相关例子记录。"),
                change_summary="创建记忆评估示例页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_记忆评估示例.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.claims] == ["用户哥哥的名字是什么？", "用户喜欢重口味"]
    assert {claim.reason for claim in review.claims} == {"记忆评估中的短问句/用户偏好/对话样例按 illustrative example 处理，不要求 raw exact match。"}


def test_grounding_detail_memory_examples_do_not_require_raw_exact_match() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-MEM-DETAIL",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Memory_Detail.md",
        display_title="事实记忆",
        page_type="concept",
        new_understanding="事实记忆包含不同评估子任务。",
        section_plans={"detail": "详情"},
        reason="测试 detail memory examples。",
    )
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-MEM-DETAIL",
                action="create",
                canonical_target_path="concepts/Concept_Memory_Detail.md",
                summary="摘要。",
                body_markdown=draft_body(detail="事实记忆的子任务包括单跳（如“用户表哥的名字？”）和知识更新"
                        "（如“用户修改了年龄后，现在多大？”）。在参与场景中，例如，用户说"
                        "“我的表哥Ethan身高162cm”，智能体回应“明白了，Ethan身高162厘米”。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_Memory_Detail.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.claims] == [
        "用户表哥的名字？",
        "用户修改了年龄后，现在多大？",
        "我的表哥Ethan身高162cm",
        "明白了，Ethan身高162厘米",
    ]
    assert {claim.reason for claim in review.claims} == {"记忆评估中的短问句/用户偏好/对话样例按 illustrative example 处理，不要求 raw exact match。"}


def test_grounding_short_concept_phrases_do_not_require_raw_exact_match() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-SCALING",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Scaling.md",
        display_title="Scaling Managed Agents",
        page_type="concept",
        new_understanding="Scaling 讨论管理型 Agent 的协作边界。",
        section_plans={"summary": "摘要"},
        reason="测试 grounding 短语。",
    )
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-SCALING",
                action="create",
                canonical_target_path="concepts/Concept_Scaling.md",
                summary="页面围绕“宠物 vs 牛”和“解耦大脑与双手”两个概念展开。",
                body_markdown=draft_body(detail="还保留“会话作为持久上下文对象”这个标题式表达。", examples="暂无相关例子记录。"),
                change_summary="创建 Scaling 页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_Scaling.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.claims] == ["宠物 vs 牛", "解耦大脑与双手", "会话作为持久上下文对象"]
    assert {claim.claim_type for claim in review.claims} == {"inference"}


def test_grounding_short_concept_phrases_in_body_markdown_do_not_require_raw_exact_match() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-SCALING",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Scaling.md",
        display_title="Scaling Managed Agents",
        page_type="concept",
        new_understanding="Scaling 讨论管理型 Agent 的协作边界。",
        section_plans={"summary": "摘要"},
        reason="测试 grounding 短语。",
    )
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-SCALING",
                action="create",
                canonical_target_path="concepts/Concept_Scaling.md",
                summary="页面围绕“宠物 vs 牛”和“解耦大脑与双手”两个概念展开。",
                body_markdown="### 概念框架\n\n还保留“会话作为持久上下文对象”这个标题式表达。",
                change_summary="创建 Scaling 页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_Scaling.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.claims] == ["宠物 vs 牛", "解耦大脑与双手", "会话作为持久上下文对象"]
    assert {claim.claim_type for claim in review.claims} == {"inference"}


def test_grounding_external_backing_claim_uses_trigger_sentence() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-EVAL",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Eval.md",
        display_title="评估（Eval）",
        page_type="concept",
        new_understanding="Eval 在产品开发中用于判断功能风险。",
        section_plans={"detail": "详情"},
        reason="测试 grounding 外部背书。",
    )
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-EVAL",
                action="create",
                canonical_target_path="concepts/Concept_Eval.md",
                summary="摘要。",
                body_markdown=draft_body(detail="在Anthropic，评估被广泛使用于产品开发。Cat Wu指出，评估的重要性因功能而异。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_Eval.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    raw = "Cat Wu指出，评估的重要性因功能而异。"
    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        raw,
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == ["在Anthropic，评估被广泛使用于产品开发。"]
    assert review.warnings[0].action == "warn"
    assert "被广泛使用" in review.warnings[0].reason
    assert "非阻塞提醒" in review.warnings[0].reason


def test_grounding_external_backing_issue_message_rejects_synonym_swap() -> None:
    claim = GroundingClaim(
        page_plan_id="PP-REDIS",
        target_path="concepts/Concept_Redis.md",
        section_key="detail",
        claim_type="new_fact",
        text="Redis最初作为高性能缓存、分析和消息代理广泛使用。",
        support="unsupported",
        action="needs_review",
        reason="新增外部背书/强事实标记 `广泛使用` 未在 raw 或 inspected wiki 中出现；请删除该背书词，或改写为 source-local 表达。",
    )

    message = draft_grounding.grounding_issue_message(claim)

    assert "非阻塞提醒" in message
    assert "adoption/authority 表达最好有来源意识" in message
    assert "source-local 表达" in message
    assert "触发文本：Redis最初作为高性能缓存、分析和消息代理广泛使用。" in message


def test_grounding_external_backing_issue_message_neutralizes_open_question_premise() -> None:
    claim = GroundingClaim(
        page_plan_id="PP-SEARCH",
        target_path="open_questions/Open_Question_混合搜索策略.md",
        section_key="open_questions",
        claim_type="new_fact",
        text="目前是否存在公认的最佳融合策略？",
        support="unsupported",
        action="needs_review",
        reason="新增外部背书/强事实标记 `公认` 未在 raw 或 inspected wiki 中出现；请删除该背书词，或改写为 source-local 表达。",
    )

    message = draft_grounding.grounding_issue_message(claim)

    assert "中性的 `待补来源` 问题" in message
    assert "不要保留 公认、广泛、业界普遍、最佳实践、行业最佳 作为问题前提" in message
    assert "非阻塞提醒" in message
    assert "触发文本：目前是否存在公认的最佳融合策略？" in message


def test_grounding_external_backing_detects_adoption_and_best_practice_real_path() -> None:
    concept_item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-REDIS",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Redis.md",
        display_title="Redis",
        page_type="concept",
        new_understanding="Redis 可以用作缓存。",
        section_plans={"detail": "说明 Redis 能力。"},
        reason="测试广泛采用 marker。",
    )
    question_item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-SEARCH",
        source_basis=SourceBasis(source_candidate_ids=["CAND002"]),
        action="create",
        canonical_target_path="open_questions/Open_Question_混合搜索策略.md",
        display_title="混合搜索策略",
        page_type="open_question",
        new_understanding="混合搜索策略仍需确认。",
        section_plans={"open_questions": "记录待补来源问题。"},
        reason="测试最佳实践 marker。",
    )
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-REDIS",
                action="create",
                canonical_target_path="concepts/Concept_Redis.md",
                summary="摘要。",
                body_markdown=draft_body(detail="Redis 被广泛采用作为缓存和消息代理。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            ),
            pipeline_module.DraftPageItem(
                page_plan_id="PP-SEARCH",
                action="create",
                canonical_target_path="open_questions/Open_Question_混合搜索策略.md",
                summary="摘要。",
                body_markdown=draft_body(detail="整理仍需确认的策略问题。"),
                open_questions="- 目前是否存在最佳实践？",
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            ),
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_Redis.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            ),
            pipeline_module.WikiContextEntry(
                path="wiki/open_questions/Open_Question_混合搜索策略.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            ),
        ],
    )
    raw = "Redis 可以作为缓存和消息代理使用。材料讨论了混合搜索与向量搜索的融合策略需要继续验证。"

    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[concept_item, question_item]),
        snapshot,
        raw,
    )

    assert review.requires_review is False
    assert {claim.action for claim in review.warnings} == {"warn"}
    reasons_by_text = {claim.text: claim.reason for claim in review.warnings}
    assert "Redis 被广泛采用作为缓存和消息代理。" in reasons_by_text
    assert "目前是否存在最佳实践？" in reasons_by_text
    assert "被广泛采用" in reasons_by_text["Redis 被广泛采用作为缓存和消息代理。"]
    assert "最佳实践" in reasons_by_text["目前是否存在最佳实践？"]


@pytest.mark.parametrize(
    "examples",
    [
        "例子写成“Redis 被广泛采用”。",
        "类似“是否存在广泛采用的方案？”的问题",
        "类似“是否存在公认方案？”的问题",
    ],
)
def test_grounding_examples_external_backing_quotes_do_not_bypass_as_inference(examples: str) -> None:
    review = build_examples_grounding_review(examples)

    assert review.requires_review is False
    assert [claim.action for claim in review.warnings] == ["warn"]


def test_grounding_external_backing_quote_only_detail_does_not_bypass_as_concept_label() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-REDIS",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Redis.md",
        display_title="Redis",
        page_type="concept",
        new_understanding="Redis 作为缓存能力被讨论。",
        section_plans={"detail": "说明 Redis 能力。"},
        reason="测试引号内外部背书 marker。",
    )
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-REDIS",
                action="create",
                canonical_target_path="concepts/Concept_Redis.md",
                summary="摘要。",
                body_markdown=draft_body(detail="主题写成“Redis 被广泛采用”。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_Redis.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )

    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == ["Redis 被广泛采用"]
    assert review.warnings[0].action == "warn"


def test_grounding_external_backing_supported_quote_does_not_hide_later_unsupported_marker() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-REDIS",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Redis.md",
        display_title="Redis",
        page_type="concept",
        new_understanding="Redis 作为缓存能力被讨论。",
        section_plans={"detail": "说明 Redis 能力。"},
        reason="测试 supported quote 后的额外 marker。",
    )
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-REDIS",
                action="create",
                canonical_target_path="concepts/Concept_Redis.md",
                summary="摘要。",
                body_markdown=draft_body(detail="材料写到“Redis 被广泛采用作为缓存”，因此 MongoDB 被广泛采用。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_Redis.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )

    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "Redis 被广泛采用作为缓存。",
    )

    assert review.requires_review is False
    assert any("MongoDB 被广泛采用" in claim.text for claim in review.warnings)
    assert any("被广泛采用" in claim.reason for claim in review.warnings)


@pytest.mark.parametrize(
    ("outside_claim", "expected_marker"),
    [
        ("因此 MongoDB 广泛采用。", "广泛采用"),
        ("因此 MongoDB 公认可靠。", "公认"),
        ("这说明 Redis 是行业最佳。", "行业最佳"),
        ("这说明 MongoDB 有最佳实践明确支持。", "最佳实践"),
    ],
)
def test_grounding_external_backing_supported_quote_does_not_hide_different_later_marker(
    outside_claim: str,
    expected_marker: str,
) -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-REDIS",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Redis.md",
        display_title="Redis",
        page_type="concept",
        new_understanding="Redis 作为缓存能力被讨论。",
        section_plans={"detail": "说明 Redis 能力。"},
        reason="测试 supported quote 后的不同 marker。",
    )
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-REDIS",
                action="create",
                canonical_target_path="concepts/Concept_Redis.md",
                summary="摘要。",
                body_markdown=draft_body(detail=f"材料写到“Redis 被广泛采用作为缓存”，{outside_claim}", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_Redis.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )

    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "Redis 被广泛采用作为缓存。",
    )

    assert review.requires_review is False
    assert any(outside_claim.rstrip("。") in claim.text for claim in review.warnings)
    assert any(expected_marker in claim.reason for claim in review.warnings)


def test_grounding_external_backing_does_not_flag_internal_multiple_components() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-MANY-HANDS",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Many_Hands.md",
        display_title="多脑多手扩展",
        page_type="concept",
        new_understanding="多脑多手扩展描述大脑和沙箱的组合方式。",
        section_plans={"detail": "详情"},
        reason="测试 `被多个` 不误伤内部组件关系。",
    )
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-MANY-HANDS",
                action="create",
                canonical_target_path="concepts/Concept_Many_Hands.md",
                summary="摘要。",
                body_markdown=draft_body(detail="一个沙箱可以被多个适配框架共享以保持状态一致性。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_Many_Hands.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


def test_grounding_external_backing_still_flags_multiple_community_claim() -> None:
    assert draft_grounding.unsupported_backing_marker("该方案被多个社区引用。") == "被多个"


def test_grounding_flags_unsupported_scope_speculation() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-COWORK",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="entities/Entity_Cowork.md",
        display_title="Cowork",
        page_type="entity",
        new_understanding="Cowork 是知识工作产品。",
        section_plans={"additional_notes": "补充观察"},
        reason="测试 grounding 范围推测。",
    )
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-COWORK",
                action="create",
                canonical_target_path="entities/Entity_Cowork.md",
                summary="Cowork 是知识工作产品。",
                body_markdown=draft_body(detail="Cowork 用于综合信息和创建文档。", additional_notes="源代码泄露事件中，Cowork 的组件可能也受到影响，但访谈中未详细说明。"),
                change_summary="创建 Cowork 页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/entities/Entity_Cowork.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    raw = "Claude Code 的源代码泄露被归因于人为错误。Cowork 是另一款知识工作产品。"

    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        raw,
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == [
        "源代码泄露事件中，Cowork 的组件可能也受到影响，但访谈中未详细说明。"
    ]
    assert review.warnings[0].action == "warn"
    assert "受影响对象推测" in review.warnings[0].reason


def test_grounding_scope_speculation_allows_open_question() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-QUESTION",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="open_questions/Open_Question_发布一致性.md",
        display_title="发布一致性",
        page_type="open_question",
        new_understanding="快速发布有一致性问题。",
        section_plans={"open_questions": "未决问题"},
        reason="测试 grounding 未决问题。",
    )
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-QUESTION",
                action="create",
                canonical_target_path="open_questions/Open_Question_发布一致性.md",
                summary="快速发布和产品一致性之间存在张力。",
                body_markdown=draft_body(detail="访谈提到团队追求快速发布。"),
                open_questions="快速发布是否可能影响长期产品一致性？",
                change_summary="创建未决问题页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/open_questions/Open_Question_发布一致性.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )

    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "团队追求快速发布。",
    )

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


def test_grounding_open_question_repair_message_moves_speculation_to_open_questions() -> None:
    claim = GroundingClaim(
        page_plan_id="PP-QUESTION",
        target_path="wiki/open_questions/Open_Question_记忆可信度.md",
        section_key="examples",
        claim_type="new_fact",
        text="如果记忆不准确，可能导致错误交易。",
        support="unsupported",
        action="needs_review",
        reason="新增影响范围/受影响对象推测 `导致` 未被 raw 或 inspected wiki 同句级支撑；请删除该推测，或改写为来源明确陈述。",
    )

    message = draft_grounding.grounding_issue_message(claim)

    assert "open_questions" in message
    assert "改写成问题" in message
    assert "待补来源" in message
    assert "detail/examples" in message
    assert "不要换成另一个具体后果" in message
    assert "<user_id>" in message


def test_grounding_external_backing_accepts_english_widely_used_anchor() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-NYU-CTF",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="entities/Entity_NYU CTF Bench.md",
        display_title="NYU CTF Bench",
        page_type="entity",
        new_understanding="NYU CTF Bench 是静态 CTF benchmark。",
        section_plans={"detail": "详情"},
        reason="测试英文论文 backing marker。",
    )
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-NYU-CTF",
                action="create",
                canonical_target_path="entities/Entity_NYU CTF Bench.md",
                summary="摘要。",
                body_markdown=draft_body(detail="NYU CTF Bench 被广泛用于评估 LLM 智能体在网络安全任务中的表现。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/entities/Entity_NYU CTF Bench.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    raw = (
        "To evaluate these agents, CTF benchmarks have become the de-facto standard. "
        "These benchmarks have also been widely used in evaluating recent LLM models."
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        raw,
    )

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


def test_grounding_external_backing_accepts_widely_across_tasks_anchor() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-CLAUDE-CODE",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="update",
        canonical_target_path="entities/Entity_Claude Code.md",
        matched_page="entities/Entity_Claude Code.md",
        display_title="Claude Code",
        page_type="entity",
        new_understanding="Claude Code 是 Managed Agents 生态中的 harness。",
        section_plans={"detail": "详情"},
        reason="测试英文 widely across tasks 支撑。",
    )
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-CLAUDE-CODE",
                action="update",
                canonical_target_path="entities/Entity_Claude Code.md",
                summary="Claude Code 是 Anthropic 开发的 harness，在团队内部被广泛使用。",
                body_markdown=draft_body(detail="Claude Code 作为 Managed Agents 的一个 harness 示例，被广泛用于多种任务。"),
                change_summary="更新页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/entities/Entity_Claude Code.md",
                expected_state="present",
                preimage_sha256="old",
                content="",
            )
        ],
    )
    raw = "For example, Claude Code is an excellent harness that we use widely across tasks."

    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        raw,
    )

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


def test_grounding_external_backing_accepts_retained_existing_fact_with_bridge_prefix() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-CLAUDE-CODE",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="update",
        canonical_target_path="entities/Entity_Claude Code.md",
        matched_page="entities/Entity_Claude Code.md",
        display_title="Claude Code",
        page_type="entity",
        new_understanding="Cat Wu 访谈补充 Claude Code 产品管理细节。",
        section_plans={"detail": "详情"},
        reason="测试 update preservation 旧事实桥接前缀。",
    )
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-CLAUDE-CODE",
                action="update",
                canonical_target_path="entities/Entity_Claude Code.md",
                summary="Claude Code 是 Anthropic 开发的一款编程助手产品。",
                body_markdown=draft_body(detail="从 Managed Agents / 托管智能体 等旧页视角看，本材料将 Claude Code 描述为“出色的 harness”，在各种任务中广泛使用。"),
                change_summary="更新页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/entities/Entity_Claude Code.md",
                expected_state="present",
                preimage_sha256="old",
                content="# Claude Code\n\n## 详细说明\n\n本材料将 Claude Code 描述为“出色的 harness”，在各种任务中广泛使用。\n",
            )
        ],
    )

    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "Cat Wu 访谈讨论 Claude Code 产品团队。",
    )

    assert review.requires_review is False
    assert review.unsupported_new_facts == []
    assert any(claim.claim_type == "retained_fact" and claim.support == "existing_wiki" for claim in review.claims)


def test_grounding_external_backing_uses_same_line_pronoun_context() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-NYU-PRONOUN",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="entities/Entity_NYU CTF Bench.md",
        display_title="NYU CTF Bench",
        page_type="entity",
        new_understanding="NYU CTF Bench 是静态 CTF benchmark。",
        section_plans={"summary": "摘要"},
        reason="测试英文论文 backing marker 的代词上下文。",
    )
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-NYU-PRONOUN",
                action="create",
                canonical_target_path="entities/Entity_NYU CTF Bench.md",
                summary="NYU CTF Bench 是用于评估 LLM 智能体的 CTF 基准。它被广泛使用，但存在数据污染风险。",
                body_markdown=draft_body(detail="静态 CTF 基准可能高估模型表现，实时 CTF 可以降低公开题解带来的污染。", examples="例如，公开 write-up 会影响静态题库。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/entities/Entity_NYU CTF Bench.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    raw = (
        "NYU CTF Bench was the first benchmark to use CTF problems for evaluating cybersecurity agents. "
        "These benchmarks have also been widely used in evaluating recent LLM models."
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        raw,
    )

    assert review.requires_review is False
    assert review.unsupported_new_facts == []


def test_grounding_external_backing_requires_specific_anchor_not_only_generic_widely_used() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-FAKE-BENCH",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="entities/Entity_FooBench.md",
        display_title="FooBench",
        page_type="entity",
        new_understanding="FooBench 是一个评估基准。",
        section_plans={"summary": "摘要"},
        reason="测试英文 backing 不能只凭泛词放行。",
    )
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-FAKE-BENCH",
                action="create",
                canonical_target_path="entities/Entity_FooBench.md",
                summary="FooBench 是用于评估 LLM 智能体的基准。它被广泛使用，但存在数据污染风险。",
                body_markdown=draft_body(detail="静态基准可能高估模型表现。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/entities/Entity_FooBench.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    raw = "These benchmarks have also been widely used in evaluating recent LLM models."
    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        raw,
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == ["它被广泛使用，但存在数据污染风险。"]
    assert review.warnings[0].action == "warn"


def test_grounding_quoted_conceptual_release_process_is_not_direct_quote() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-RESEARCH-PREVIEW",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="designs/Design_Research_Preview.md",
        display_title="研究预览发布模式",
        page_type="design",
        new_understanding="研究预览是一种发布模式。",
        section_plans={"detail": "详情"},
        reason="测试 grounding 发布流程概念短语。",
    )
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-RESEARCH-PREVIEW",
                action="create",
                canonical_target_path="designs/Design_Research_Preview.md",
                summary="摘要。",
                body_markdown=draft_body(detail="该模式与“可重复发布流程”和“设定清晰目标”形成配套。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/designs/Design_Research_Preview.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.claims] == ["可重复发布流程", "设定清晰目标"]
    assert {claim.claim_type for claim in review.claims} == {"inference"}


def test_grounding_quoted_product_choice_label_context_is_not_direct_quote() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-PRODUCT-CHOICE",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Claude_Code_Cowork_Choice.md",
        display_title="Claude Code 与 Cowork 的产品选择",
        page_type="concept",
        new_understanding="产品选择标签不应被当成直接引用。",
        section_plans={"additional_notes": "补充观察"},
        reason="测试 grounding 产品选择标签。",
    )
    quote = "何时使用Claude Code与Cowork"
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-PRODUCT-CHOICE",
                action="create",
                canonical_target_path="concepts/Concept_Claude_Code_Cowork_Choice.md",
                summary="摘要。",
                body_markdown=draft_body(examples="暂无相关例子记录。", additional_notes=f"本概念源自访谈中关于“{quote}”的讨论。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_Claude_Code_Cowork_Choice.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.claims] == [quote]
    assert review.claims[0].claim_type == "inference"


def test_grounding_concept_label_after_broad_mention_is_not_direct_quote() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-HARNESS",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="entities/Entity_Claude Code.md",
        display_title="Claude Code",
        page_type="entity",
        new_understanding="Claude Code 是 harness 示例。",
        section_plans={"detail": "详情"},
        reason="测试 grounding 宽泛提到短概念。",
    )
    quote = "优秀的适配框架"
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-HARNESS",
                action="create",
                canonical_target_path="entities/Entity_Claude Code.md",
                summary="摘要。",
                body_markdown=draft_body(detail=f"文中提到它是“{quote}”，展示了元适配框架可以容纳不同类型的 harness。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/entities/Entity_Claude Code.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.claims] == [quote]
    assert review.claims[0].claim_type == "inference"


def test_grounding_attributed_concept_label_is_not_dequoted_or_bypassed() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-ATTRIBUTED-LABEL",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Context_Object.md",
        display_title="会话上下文对象",
        page_type="concept",
        new_understanding="会话可被理解成上下文对象。",
        section_plans={"detail": "详情"},
        reason="测试 attributed concept label。",
    )
    quote = "会话作为持久上下文对象"
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-ATTRIBUTED-LABEL",
                action="create",
                canonical_target_path="concepts/Concept_Context_Object.md",
                summary="摘要。",
                body_markdown=draft_body(detail=f"文中称“{quote}”，因此该页面保留这个概念。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_Context_Object.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )

    rewritten, report = draft_grounding.rewrite_grounding_sensitive_paraphrases(draft, "")
    review = draft_grounding.build_draft_grounding_review(
        rewritten,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert report["changed"] is False
    assert f"“{quote}”" in rewritten.pages[0].body_markdown
    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == [quote]


def test_grounding_attributed_concept_label_with_punctuation_is_not_bypassed() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-ATTRIBUTED-PUNCTUATION",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Context_Object.md",
        display_title="会话上下文对象",
        page_type="concept",
        new_understanding="会话可被理解成上下文对象。",
        section_plans={"detail": "详情"},
        reason="测试 attributed punctuation。",
    )
    quote = "会话作为持久上下文对象"
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-ATTRIBUTED-PUNCTUATION",
                action="create",
                canonical_target_path="concepts/Concept_Context_Object.md",
                summary="摘要。",
                body_markdown=draft_body(detail=f"文中称：“{quote}”，因此该页面保留这个概念。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_Context_Object.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )

    rewritten, report = draft_grounding.rewrite_grounding_sensitive_paraphrases(draft, "")
    review = draft_grounding.build_draft_grounding_review(
        rewritten,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert report["changed"] is False
    assert f"“{quote}”" in rewritten.pages[0].body_markdown
    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == [quote]


def test_grounding_quoted_abstract_trend_label_context_is_not_direct_quote() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-PM-SKILLS",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="open_questions/Open_Question_AI_PM_Skills.md",
        display_title="AI PM 技能变化",
        page_type="open_question",
        new_understanding="抽象趋势标签不应被当成直接引用。",
        section_plans={"additional_notes": "补充观察"},
        reason="测试 grounding 抽象趋势标签。",
    )
    quote = "技术壁垒正在降低"
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-PM-SKILLS",
                action="create",
                canonical_target_path="open_questions/Open_Question_AI_PM_Skills.md",
                summary="摘要。",
                body_markdown=draft_body(examples="暂无相关例子记录。", additional_notes=f"本问题源自她提到的“{quote}”的趋势。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/open_questions/Open_Question_AI_PM_Skills.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.claims] == [quote]
    assert review.claims[0].claim_type == "inference"


def test_grounding_explicit_direct_quote_mismatch_warns_without_review() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-QUOTE",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Quote.md",
        display_title="直接引用",
        page_type="concept",
        new_understanding="直接引用需要来源支撑。",
        section_plans={"summary": "摘要"},
        reason="测试 grounding。",
    )
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-QUOTE",
                action="create",
                canonical_target_path="concepts/Concept_Quote.md",
                summary="原文说“解耦大脑与双手”。",
                body_markdown=draft_body(detail="暂无更多细节。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_Quote.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert review.requires_review is False
    assert review.warnings[0].text == "解耦大脑与双手"
    assert "直接引用/作者归因" in review.warnings[0].reason


def test_grounding_direct_quote_accepts_normalized_source_match() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-PAPER",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="designs/Design_Paper.md",
        display_title="论文方法",
        page_type="design",
        new_understanding="论文方法句需要来源支撑。",
        section_plans={"detail": "详情"},
        reason="测试 grounding 规范化 exact match。",
    )
    quote = "Based on MemEngine (Zhang et al., 2025), we implement seven memory mechanisms, using Qwen2.5-7B as the base model"
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-PAPER",
                action="create",
                canonical_target_path="designs/Design_Paper.md",
                summary="摘要。",
                body_markdown=draft_body(detail=f"源摘录中提到“{quote}”。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/designs/Design_Paper.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    raw = (
        "To eliminate other designs on results, we make no modifications to components."
        "Based on MemEngine (Zhang et al., 2025 ), we implement seven memory mechanisms, "
        "using Qwen2.5-7B as the base model for the agent applications on our benchmark."
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        raw,
    )

    assert review.requires_review is False
    assert review.claims[0].support == "raw"
    assert review.claims[0].reason == "直接引用已在 raw 或已有 wiki 中规范化 exact match。"


def test_grounding_direct_quote_accepts_time_range_transcript_variant() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-PM-ROLE",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_PM角色演变.md",
        display_title="PM角色演变",
        page_type="concept",
        new_understanding="PM负责从当前状态到长期愿景之间的路径。",
        section_plans={"detail": "详情"},
        reason="测试 transcript 数字范围近似直引。",
    )
    quote = "弄清楚从今天到3-6个月后愿景之间的路径"
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-PM-ROLE",
                action="create",
                canonical_target_path="concepts/Concept_PM角色演变.md",
                summary="摘要。",
                body_markdown=draft_body(detail=f"Cat Wu提到，PM的工作是“{quote}”。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_PM角色演变.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    raw = (
        "Boris 非常擅长设定方向，比如这就是产品在3个月、6个月后需要成为的样子。"
        "而我的很多职责是弄清楚从今天到那个3到6个月后的愿景之间的路径是什么。"
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        raw,
    )

    assert review.requires_review is False
    assert review.claims[0].support == "raw"
    assert review.claims[0].reason == "直接引用已在 raw 或已有 wiki 中规范化 exact match。"


def test_grounding_direct_quote_accepts_paired_month_enumeration_as_range() -> None:
    quote = "产品在3-6个月后需要成为的样子"
    raw = "Boris 非常擅长设定方向，比如这就是产品在3个月、6个月后需要成为的样子。"

    assert draft_grounding.quote_supported_by_text(quote, raw) is True


def test_grounding_direct_quote_paired_month_enumeration_requires_same_numbers() -> None:
    quote = "产品在3-9个月后需要成为的样子"
    raw = "Boris 非常擅长设定方向，比如这就是产品在3个月、6个月后需要成为的样子。"

    assert draft_grounding.quote_supported_by_text(quote, raw) is False


def test_grounding_direct_quote_does_not_collapse_three_item_timeline() -> None:
    quote = "产品在3-6个月后需要成为的样子"
    raw = "路线图分别记录产品在3个月、6个月、9个月后需要成为的样子。"

    assert draft_grounding.quote_supported_by_text(quote, raw) is False


def test_grounding_direct_quote_range_does_not_match_partial_numeric_token() -> None:
    quote = "产品在3-6个月后需要成为的样子"
    raw = "Boris 讨论的是产品在13个月、6个月后需要成为的样子。"

    assert draft_grounding.quote_supported_by_text(quote, raw) is False


def test_grounding_direct_quote_time_range_variant_requires_same_numbers() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-PM-ROLE",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_PM角色演变.md",
        display_title="PM角色演变",
        page_type="concept",
        new_understanding="PM负责从当前状态到长期愿景之间的路径。",
        section_plans={"detail": "详情"},
        reason="测试 transcript 数字范围不能误配。",
    )
    quote = "弄清楚从今天到3-9个月后愿景之间的路径"
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-PM-ROLE",
                action="create",
                canonical_target_path="concepts/Concept_PM角色演变.md",
                summary="摘要。",
                body_markdown=draft_body(detail=f"Cat Wu提到，PM的工作是“{quote}”。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_PM角色演变.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    raw = "我的职责是弄清楚从今天到那个3到6个月后的愿景之间的路径是什么。"
    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        raw,
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == [quote]


def test_grounding_short_domain_quote_accepts_normalized_source_match() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-CLAUDE-CODE",
        source_basis=SourceBasis(source_candidate_ids=["auto-ent-claudecode"]),
        action="create",
        canonical_target_path="entities/Entity_Claude Code.md",
        display_title="Claude Code",
        page_type="entity",
        new_understanding="Claude Code 是 Managed Agents 生态中的 harness。",
        section_plans={"detail": "详情"},
        reason="测试短 domain quote 的规范化 exact match。",
    )
    quote = "出色的 harness"
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-CLAUDE-CODE",
                action="create",
                canonical_target_path="entities/Entity_Claude Code.md",
                summary="摘要。",
                body_markdown=draft_body(detail=f"源材料在正文中提到，Claude Code 已经作为“{quote}”被集成到 Managed Agents 架构中。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/entities/Entity_Claude Code.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    raw = "例如，**Claude Code** 是一个出色的 **harness（适配框架）**，我们在各种任务中广泛使用它。"

    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        raw,
    )

    assert review.requires_review is False
    assert review.claims[0].text == quote
    assert review.claims[0].support == "raw"


def test_grounding_domain_quote_accepts_source_match_with_parenthetical_translation() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-CLAUDE-CODE-LONG",
        source_basis=SourceBasis(source_candidate_ids=["auto-ent-claudecode"]),
        action="create",
        canonical_target_path="entities/Entity_Claude Code.md",
        display_title="Claude Code",
        page_type="entity",
        new_understanding="Claude Code 是 Managed Agents 生态中的 harness。",
        section_plans={"detail": "详情"},
        reason="测试 domain quote 可省略英文术语后的中文括注。",
    )
    quote = "一个出色的 harness，我们在各种任务中广泛使用它"
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-CLAUDE-CODE-LONG",
                action="create",
                canonical_target_path="entities/Entity_Claude Code.md",
                summary="摘要。",
                body_markdown=draft_body(detail=f"原文提到“{quote}”。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/entities/Entity_Claude Code.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    raw = "例如，**Claude Code** 是一个出色的 **harness（适配框架）**，我们在各种任务中广泛使用它。"

    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        raw,
    )

    assert review.requires_review is False
    assert review.claims[0].text == quote
    assert review.claims[0].support == "raw"


def test_grounding_short_numeric_quote_accepts_exact_numeric_source_match() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-BORIS",
        source_basis=SourceBasis(source_candidate_ids=["E002"]),
        action="create",
        canonical_target_path="entities/Entity_Boris Cherny.md",
        display_title="Boris Cherny",
        page_type="entity",
        new_understanding="Boris 与 Cat Wu 的协作模式。",
        section_plans={"detail": "详情"},
        reason="测试短数字 quote 的规范化 exact match。",
    )
    quote = "80% 是心灵融合"
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-BORIS",
                action="create",
                canonical_target_path="entities/Entity_Boris Cherny.md",
                summary="摘要。",
                body_markdown=draft_body(detail=f"Cat 形容他们的合作“{quote}”，剩余 20% 由各自在意的事情驱动。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/entities/Entity_Boris Cherny.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    raw = '我觉得我们大概80%是心灵融合，然后有20%的事情我更在意，我就多推动那些。'

    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        raw,
    )

    assert review.requires_review is False
    assert review.claims[0].text == quote
    assert review.claims[0].support == "raw"


def test_grounding_short_numeric_quote_does_not_match_decimal_collapse() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-BORIS",
        source_basis=SourceBasis(source_candidate_ids=["E002"]),
        action="create",
        canonical_target_path="entities/Entity_Boris Cherny.md",
        display_title="Boris Cherny",
        page_type="entity",
        new_understanding="Boris 与 Cat Wu 的协作模式。",
        section_plans={"detail": "详情"},
        reason="测试短数字 quote 不把小数错配成整数百分比。",
    )
    quote = "9.5% 是心灵融合"
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-BORIS",
                action="create",
                canonical_target_path="entities/Entity_Boris Cherny.md",
                summary="摘要。",
                body_markdown=draft_body(detail=f"Cat 形容他们的合作“{quote}”。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/entities/Entity_Boris Cherny.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    raw = "我觉得我们大概95%是心灵融合。"

    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        raw,
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == [quote]


def test_grounding_ascii_closing_quote_is_not_treated_as_new_quote_start() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-PETS-CATTLE",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Pets_Cattle.md",
        display_title="Pets vs Cattle",
        page_type="concept",
        new_understanding="容器失败应像 cattle 一样被自动替换。",
        section_plans={"detail": "详情"},
        reason="测试 grounding 半角引号边界。",
    )
    quote = (
        "If the container died, the harness caught the failure as a tool-call error "
        "and passed it back to Claude."
    )
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-PETS-CATTLE",
                action="create",
                canonical_target_path="concepts/Concept_Pets_Cattle.md",
                summary="摘要。",
                body_markdown=draft_body(detail='解耦后，container 变成"牲畜"——如果它死了，harness 将失败捕获为工具调用错误，'
                        f'传回 Claude。原文描述："{quote}"', examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_Pets_Cattle.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        quote,
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.claims] == [quote]
    assert review.claims[0].support == "raw"


def test_grounding_quoted_evaluation_question_template_is_not_direct_quote() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-EVAL-QUESTION",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="open_questions/Open_Question_Eval.md",
        display_title="概率性 AI 产品评估",
        page_type="open_question",
        new_understanding="评估问题模板不是直接事实引用。",
        section_plans={"additional_notes": "补充观察"},
        reason="测试 grounding 评估问句模板。",
    )
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-EVAL-QUESTION",
                action="create",
                canonical_target_path="open_questions/Open_Question_Eval.md",
                summary="摘要。",
                body_markdown=draft_body(examples="暂无相关例子记录。", additional_notes="不是“是否回答正确”，而是“在多少比例下用户满意”。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/open_questions/Open_Question_Eval.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.claims] == ["是否回答正确", "在多少比例下用户满意"]
    assert {claim.claim_type for claim in review.claims} == {"inference"}


def test_grounding_quoted_compact_paraphrase_uses_nearby_source_support() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-FAST-ITERATION",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Fast_Iteration.md",
        display_title="快速迭代流程",
        page_type="concept",
        new_understanding="清晰目标帮助团队快速迭代。",
        section_plans={"detail": "详情"},
        reason="测试 grounding 压缩概括。",
    )
    quote = "核心用户是专业开发者，主要问题是权限提示疲劳"
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-FAST-ITERATION",
                action="create",
                canonical_target_path="concepts/Concept_Fast_Iteration.md",
                summary="摘要。",
                body_markdown=draft_body(detail=f"设定清晰目标（如“{quote}”）可以减少 LLM 通用性带来的模糊。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_Fast_Iteration.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    raw = (
        "所以我认为一个优秀的PM能够说：好的，我们的核心用户是专业开发者。"
        "我们这个功能要解决的主要问题可能是权限提示太多了，人们感到疲劳。"
        "我们的用例是：我们希望企业里的专业开发者能够安全地实现零权限提示。"
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        raw,
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.claims] == [quote]
    assert review.claims[0].claim_type == "inference"
    assert review.claims[0].support == "raw"
    assert "压缩概括" in review.claims[0].reason


def test_grounding_quoted_method_goal_paraphrase_uses_nearby_source_support() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-FAST-SHIPPING",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Fast_Shipping.md",
        display_title="快速交付方法",
        page_type="concept",
        new_understanding="团队用目标短语总结快速交付方法。",
        section_plans={"detail": "详情"},
        reason="测试 grounding 方法目标短语。",
    )
    quote = "找到最快将功能交到用户手中的方法"
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-FAST-SHIPPING",
                action="create",
                canonical_target_path="concepts/Concept_Fast_Shipping.md",
                summary="摘要。",
                body_markdown=draft_body(detail=f"PM 关注的是“{quote}”。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_Fast_Shipping.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    raw = (
        "我们怎样才能找到最快把东西推出去的方法？"
        "我们怎样才能创建一个产品套件的概念角落，让工程师或 PM 有一个想法，"
        "到周末就能把功能交到用户手中。"
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        raw,
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.claims] == [quote]
    assert review.claims[0].claim_type == "inference"
    assert review.claims[0].support == "raw"
    assert "压缩概括" in review.claims[0].reason


def test_grounding_quoted_method_goal_paraphrase_warns_without_nearby_support() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-FAST-SHIPPING",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Fast_Shipping.md",
        display_title="快速交付方法",
        page_type="concept",
        new_understanding="团队用目标短语总结快速交付方法。",
        section_plans={"detail": "详情"},
        reason="测试 grounding 方法目标短语。",
    )
    quote = "找到最快将功能交到用户手中的方法"
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-FAST-SHIPPING",
                action="create",
                canonical_target_path="concepts/Concept_Fast_Shipping.md",
                summary="摘要。",
                body_markdown=draft_body(detail=f"PM 关注的是“{quote}”。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_Fast_Shipping.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    raw = "我们怎样才能找到最快把东西推出去的方法？这里没有说明最终交付给谁。"
    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        raw,
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == [quote]


def test_grounding_numeric_reliability_paraphrase_rewrites_to_source_sentence() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-AUTOMATION-RELIABILITY",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_AI自动化可靠性.md",
        display_title="AI自动化可靠性",
        page_type="concept",
        new_understanding="100%可靠性原则来自访谈。",
        section_plans={"additional_notes": "补充观察"},
        reason="测试 grounding 百分比 paraphrase 改写。",
    )
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-AUTOMATION-RELIABILITY",
                action="create",
                canonical_target_path="concepts/Concept_AI自动化可靠性.md",
                summary="摘要。",
                body_markdown=draft_body(examples="暂无相关例子记录。", additional_notes="100%可靠性原则也适用于AI产品自身的质量，正如Cat Wu所说“95%对AI来说就是失败”。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    raw = "如果自动化不是100%有效，它真的不是自动化。95%的自动化真的没什么价值。"
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_AI自动化可靠性.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )

    rewritten, report = draft_grounding.rewrite_grounding_sensitive_paraphrases(draft, raw)
    body = rewritten.pages[0].body_markdown
    review = draft_grounding.build_draft_grounding_review(
        rewritten,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        raw,
    )

    assert report["changed"] is True
    assert report["rewrite_count"] == 1
    assert "95%对AI来说就是失败" not in body
    assert "正如Cat Wu所说，95%的自动化真的没什么价值。" in body
    assert review.requires_review is False


def test_grounding_numeric_reliability_paraphrase_warns_without_source_sentence() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-AUTOMATION-RELIABILITY",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_AI自动化可靠性.md",
        display_title="AI自动化可靠性",
        page_type="concept",
        new_understanding="100%可靠性原则来自访谈。",
        section_plans={"additional_notes": "补充观察"},
        reason="测试 grounding 百分比 paraphrase 改写。",
    )
    quote = "95%对AI来说就是失败"
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-AUTOMATION-RELIABILITY",
                action="create",
                canonical_target_path="concepts/Concept_AI自动化可靠性.md",
                summary="摘要。",
                body_markdown=draft_body(examples="暂无相关例子记录。", additional_notes=f"正如Cat Wu所说“{quote}”。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_AI自动化可靠性.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )

    rewritten, report = draft_grounding.rewrite_grounding_sensitive_paraphrases(draft, "AI工具需要继续提升可靠性。")
    review = draft_grounding.build_draft_grounding_review(
        rewritten,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "AI工具需要继续提升可靠性。",
    )

    assert report["changed"] is False
    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == [quote]


def test_grounding_rewrite_translates_known_english_harness_quote() -> None:
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-CLAUDE-CODE",
                action="create",
                canonical_target_path="entities/Entity_Claude Code.md",
                summary="摘要。",
                body_markdown=draft_body(detail="Claude Code 被描述为“an excellent harness that provides a focused coding experience”，可接入 Managed Agents。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    rewritten, report = draft_grounding.rewrite_grounding_sensitive_paraphrases(draft, "")
    body = rewritten.pages[0].body_markdown

    assert report["changed"] is True
    assert "an excellent harness" not in body
    assert "被描述为一种优秀的 harness，提供聚焦的编码体验" in body


def test_grounding_rewrite_dequotes_internal_digest_paraphrase() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-SECURITY",
        source_basis=SourceBasis(source_candidate_ids=["D001"]),
        action="create",
        canonical_target_path="designs/Design_Security.md",
        display_title="安全令牌隔离",
        page_type="design",
        new_understanding="安全令牌隔离减少凭证暴露。",
        section_plans={"additional_notes": "补充观察"},
        reason="测试内部 artifact paraphrase 去引号。",
    )
    quote = "在耦合架构中，sandbox 与凭证共存，攻击者可通过提示注入窃取令牌；此设计从结构上消除了该风险。"
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-SECURITY",
                action="create",
                canonical_target_path="designs/Design_Security.md",
                summary="摘要。",
                body_markdown=draft_body(examples="暂无相关例子记录。", additional_notes=f"该设计对应 approved_digest 中 D001 的 why_matters 描述：“{quote}”"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/designs/Design_Security.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )

    rewritten, report = draft_grounding.rewrite_grounding_sensitive_paraphrases(draft, "")
    body = rewritten.pages[0].body_markdown
    review = draft_grounding.build_draft_grounding_review(
        rewritten,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert report["changed"] is True
    assert "approved_digest" not in body
    assert f"“{quote}”" not in body
    assert "对应的来源要点是：" in body
    assert review.requires_review is False


def test_grounding_rewrite_dequotes_non_explicit_scope_paraphrase() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-CONTEXT",
        source_basis=SourceBasis(source_candidate_ids=["Q001"]),
        action="create",
        canonical_target_path="open_questions/Open_Question_Context.md",
        display_title="长时上下文管理",
        page_type="open_question",
        new_understanding="长时上下文管理仍有开放问题。",
        section_plans={"additional_notes": "补充观察"},
        reason="测试 scope paraphrase 去引号。",
    )
    quote = "文章仅提出会话作为持久化存储，但未深入讨论智能压缩、索引或预取"
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-CONTEXT",
                action="create",
                canonical_target_path="open_questions/Open_Question_Context.md",
                summary="摘要。",
                body_markdown=draft_body(examples="暂无相关例子记录。", additional_notes=f"本材料中提到“{quote}”，这正是该问题的来源。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/open_questions/Open_Question_Context.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )

    rewritten, report = draft_grounding.rewrite_grounding_sensitive_paraphrases(draft, "")
    review = draft_grounding.build_draft_grounding_review(
        rewritten,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert report["changed"] is True
    assert f"“{quote}”" not in rewritten.pages[0].body_markdown
    assert review.requires_review is False


def test_grounding_rewrite_dequotes_long_non_explicit_paraphrase_with_fact_markers() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-MANY-HANDS",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Many_Hands.md",
        display_title="多大脑多手",
        page_type="concept",
        new_understanding="多大脑多手来自大脑与双手解耦。",
        section_plans={"detail": "详情"},
        reason="测试长 paraphrase 去引号。",
    )
    quote = (
        "将大脑与双手解耦解决了我们最早的客户投诉之一。当团队希望 Claude 使用他们自己 VPC 中的资源时，"
        "唯一的路径是将他们的网络与我们的做对等互连，因为持有 harness 的容器假定每个资源都在它旁边。"
    )
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-MANY-HANDS",
                action="create",
                canonical_target_path="concepts/Concept_Many_Hands.md",
                summary="摘要。",
                body_markdown=draft_body(detail=f"材料指出：“{quote}”解耦后，资源可以位于任何位置。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_Many_Hands.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )

    rewritten, report = draft_grounding.rewrite_grounding_sensitive_paraphrases(draft, "")
    review = draft_grounding.build_draft_grounding_review(
        rewritten,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert report["changed"] is True
    assert f"“{quote}”" not in rewritten.pages[0].body_markdown
    assert review.requires_review is False


def test_grounding_rewrite_dequotes_open_question_quote_with_growth_marker() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-LOG-GROWTH",
        source_basis=SourceBasis(source_candidate_ids=["Q001"]),
        action="create",
        canonical_target_path="open_questions/Open_Question_Log_Growth.md",
        display_title="会话日志增长管理",
        page_type="open_question",
        new_understanding="会话日志增长管理仍待设计。",
        section_plans={"additional_notes": "补充观察"},
        reason="测试开放问题问句去引号。",
    )
    quote = "日志大小增长如何管理？是否需要引入日志压缩或归档策略？"
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-LOG-GROWTH",
                action="create",
                canonical_target_path="open_questions/Open_Question_Log_Growth.md",
                summary="摘要。",
                body_markdown=draft_body(examples="暂无相关例子记录。", additional_notes=f"该开放问题来自来源材料中明确提到的“{quote}”。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/open_questions/Open_Question_Log_Growth.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )

    rewritten, report = draft_grounding.rewrite_grounding_sensitive_paraphrases(draft, "")
    review = draft_grounding.build_draft_grounding_review(
        rewritten,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert report["changed"] is True
    assert f"“{quote}”" not in rewritten.pages[0].body_markdown
    assert review.requires_review is False


def test_grounding_rewrite_dequotes_source_local_context_window_paraphrase() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-CONTEXT-WINDOW",
        source_basis=SourceBasis(source_candidate_ids=["Q001"]),
        action="create",
        canonical_target_path="open_questions/Open_Question_Context_Window.md",
        display_title="未来上下文工程不可预测性",
        page_type="open_question",
        new_understanding="会话和上下文窗口的边界可能变化。",
        section_plans={"examples": "例子"},
        reason="测试 source-local concept paraphrase 去引号。",
    )
    quote = "会话不是 Claude 的上下文窗口"
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-CONTEXT-WINDOW",
                action="create",
                canonical_target_path="open_questions/Open_Question_Context_Window.md",
                summary="摘要。",
                body_markdown=draft_body(detail="暂无更多细节。", examples=f"原文提到“{quote}”，但未说明未来原生长上下文是否会改变当前架构。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/open_questions/Open_Question_Context_Window.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )

    rewritten, report = draft_grounding.rewrite_grounding_sensitive_paraphrases(draft, "")
    review = draft_grounding.build_draft_grounding_review(
        rewritten,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert report["changed"] is True
    assert f"“{quote}”" not in rewritten.pages[0].body_markdown
    assert review.requires_review is False


def test_grounding_rewrite_dequotes_short_slogan_label() -> None:
    quote = "快速行动，打破常规"
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-META-CULTURE",
                action="create",
                canonical_target_path="comparisons/Comparison_Meta_Culture.md",
                summary="摘要。",
                body_markdown=draft_body(detail=f"Meta 速度实验驱动：推崇“{quote}”，产品决策依赖 A/B 测试和快速迭代。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    rewritten, report = draft_grounding.rewrite_grounding_sensitive_paraphrases(
        draft,
        "Meta 的文化是速度驱动的。Move fast and break things。你不需要完美的文档。",
    )

    assert report["changed"] is True
    assert f"“{quote}”" not in rewritten.pages[0].body_markdown
    assert "推崇快速行动，打破常规" in rewritten.pages[0].body_markdown


def test_grounding_rewrite_does_not_dequote_short_hard_fact_label() -> None:
    quote = "用户增长，收入下降"
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-HARD-FACT",
                action="create",
                canonical_target_path="concepts/Concept_Hard_Fact.md",
                summary="摘要。",
                body_markdown=draft_body(detail=f"报告称“{quote}”。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    rewritten, report = draft_grounding.rewrite_grounding_sensitive_paraphrases(draft, "源材料没有这句话。")

    assert report["changed"] is False
    assert f"“{quote}”" in rewritten.pages[0].body_markdown


def test_grounding_numeric_reliability_rewrite_does_not_match_decimal_percent() -> None:
    quote = "9.5%对AI来说就是失败"
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-AUTOMATION-RELIABILITY",
                action="create",
                canonical_target_path="concepts/Concept_AI自动化可靠性.md",
                summary="摘要。",
                body_markdown=draft_body(additional_notes=f"正如Cat Wu所说“{quote}”。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )

    rewritten, report = draft_grounding.rewrite_grounding_sensitive_paraphrases(
        draft,
        "95%的自动化真的没什么价值。",
    )

    assert report["changed"] is False
    assert rewritten.pages[0].body_markdown == draft.pages[0].body_markdown


def test_grounding_attributed_paraphrase_warns_without_exact_match() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-FAST-SHIPPING",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Fast_Shipping.md",
        display_title="快速交付方法",
        page_type="concept",
        new_understanding="团队用目标短语总结快速交付方法。",
        section_plans={"detail": "详情"},
        reason="测试 grounding 人物归因短语。",
    )
    quote = "找到最快将功能交到用户手中的方法"
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-FAST-SHIPPING",
                action="create",
                canonical_target_path="concepts/Concept_Fast_Shipping.md",
                summary="摘要。",
                body_markdown=draft_body(detail=f"Cat Wu指出“{quote}”。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_Fast_Shipping.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    raw = (
        "我们怎样才能找到最快把东西推出去的方法？"
        "让工程师或 PM 有一个想法，到周末就能把功能交到用户手中。"
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        raw,
    )
    rewritten, report = draft_grounding.rewrite_grounding_sensitive_paraphrases(draft, raw)

    assert report["changed"] is False
    assert rewritten.pages[0].body_markdown == draft.pages[0].body_markdown
    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == [quote]


def test_grounding_named_tool_concept_label_with_digits_is_not_direct_quote() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-AGENT",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_AI_Agent.md",
        display_title="AI Agent 架构",
        page_type="concept",
        new_understanding="Agent 和工作流适用场景不同。",
        section_plans={"additional_notes": "补充观察"},
        reason="测试 grounding 工具标题标签。",
    )
    quote = "N8N工作流与Agent构建对比"
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-AGENT",
                action="create",
                canonical_target_path="concepts/Concept_AI_Agent.md",
                summary="摘要。",
                body_markdown=draft_body(examples="暂无相关例子记录。", additional_notes=f"本页可与设计模式“{quote}”联动阅读。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_AI_Agent.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.claims] == [quote]
    assert review.claims[0].claim_type == "inference"


def test_grounding_named_tool_label_with_numeric_fact_warns_without_support() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-AGENT-FACT",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_AI_Agent.md",
        display_title="AI Agent 架构",
        page_type="concept",
        new_understanding="Agent 系统包含多个步骤。",
        section_plans={"additional_notes": "补充观察"},
        reason="测试带数字的短事实不能伪装成概念标题。",
    )
    quote = "Agent系统有3个步骤"
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-AGENT-FACT",
                action="create",
                canonical_target_path="concepts/Concept_AI_Agent.md",
                summary="摘要。",
                body_markdown=draft_body(examples="暂无相关例子记录。", additional_notes=f"本页暂以“{quote}”作为结构提示。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_AI_Agent.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == [quote]


def test_grounding_quoted_compact_paraphrase_warns_without_support_for_each_part() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-FAST-ITERATION",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Fast_Iteration.md",
        display_title="快速迭代流程",
        page_type="concept",
        new_understanding="清晰目标帮助团队快速迭代。",
        section_plans={"detail": "详情"},
        reason="测试 grounding 压缩概括。",
    )
    quote = "核心用户是专业开发者，主要问题是权限提示疲劳"
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-FAST-ITERATION",
                action="create",
                canonical_target_path="concepts/Concept_Fast_Iteration.md",
                summary="摘要。",
                body_markdown=draft_body(detail=f"设定清晰目标（如“{quote}”）可以减少 LLM 通用性带来的模糊。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_Fast_Iteration.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    raw = "团队原则中写到，核心用户是专业开发者。"
    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        raw,
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == [quote]


def test_grounding_short_fact_phrases_warn_without_exact_match() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-FACT",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Fact.md",
        display_title="短事实",
        page_type="concept",
        new_understanding="短事实需要来源支撑。",
        section_plans={"summary": "摘要"},
        reason="测试 grounding。",
    )
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-FACT",
                action="create",
                canonical_target_path="concepts/Concept_Fact.md",
                summary="结果包括例如“销量增长三倍”、“裁撤一半团队”和“预算超过百万”。",
                body_markdown=draft_body(detail="暂无更多细节。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_Fact.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == ["销量增长三倍", "裁撤一半团队", "预算超过百万"]


def test_grounding_quoted_release_event_warns_without_exact_match() -> None:
    item = pipeline_module.WikiMergePlanItem(
        page_plan_id="PP-RELEASE-FACT",
        source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
        action="create",
        canonical_target_path="concepts/Concept_Release_Fact.md",
        display_title="发布事实",
        page_type="concept",
        new_understanding="发布事件需要来源支撑。",
        section_plans={"summary": "摘要"},
        reason="测试 grounding 发布事实。",
    )
    draft = pipeline_module.DraftRenderingArtifact(
        pages=[
            pipeline_module.DraftPageItem(
                page_plan_id="PP-RELEASE-FACT",
                action="create",
                canonical_target_path="concepts/Concept_Release_Fact.md",
                summary="团队“发布了重大功能”。",
                body_markdown=draft_body(detail="暂无更多细节。", examples="暂无相关例子记录。"),
                change_summary="创建页面。",
                source_coverage_notes="测试。",
            )
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_Release_Fact.md",
                expected_state="missing",
                preimage_sha256=None,
                content="",
            )
        ],
    )
    review = draft_grounding.build_draft_grounding_review(
        draft,
        pipeline_module.WikiMergePlanArtifact(log_date="2026-06-06", items=[item]),
        snapshot,
        "",
    )

    assert review.requires_review is False
    assert [claim.text for claim in review.warnings] == ["发布了重大功能"]


def test_draft_page_item_coerces_quality_risks_string_to_list() -> None:
    page = pipeline_module.DraftPageItem(
        page_plan_id="PP-RISK",
        action="create",
        canonical_target_path="concepts/Concept_Risk.md",
        summary="摘要",
        body_markdown="风险页面正文。",
        change_summary="创建页面。",
        source_coverage_notes="测试。",
        quality_risks="样本范围有限；需要后续验证",
    )

    assert page.quality_risks == ["样本范围有限；需要后续验证"]


def test_draft_rendering_business_validation_repairs_before_persisting(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    fixture_dir = tmp_path / "draft-repair-fixture"
    fixture_dir.mkdir()
    for name in ["raw_prepare.json", "source_digest.json", "candidate_resolution.json", "wiki_merge_planning.json"]:
        write_json(fixture_dir / name, read_json(FIXTURE_ROOT / "mock" / name))
    bad_draft = read_json(FIXTURE_ROOT / "mock" / "draft_rendering.json")
    bad_draft["pages"][0].pop("summary")
    write_json(fixture_dir / "draft_rendering.1.json", bad_draft)
    write_json(fixture_dir / "draft_rendering.2.json", read_json(FIXTURE_ROOT / "mock" / "draft_rendering.json"))

    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=fixture_dir, slug="draft-repair")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    repair_report = read_json(run_dir / "draft_rendering" / "structured_repair_report.json")
    manifest_after = status(vault, manifest.operation_id)
    draft_step = [step for step in manifest_after.steps if step.name == "draft_rendering"][0]

    assert manifest_after.status == OperationStatus.drafted
    assert repair_report["repair_count"] == 1
    assert repair_report["attempts"][0]["issues"][0]["issue_code"] == "missing_field"
    assert repair_report["attempts"][0]["issues"][0]["field_path"] == "summary"
    assert repair_report["attempts"][1]["repair_prompt_ref"] == "repair_prompts/attempt-2.json"
    repair_prompt = read_json(run_dir / "draft_rendering" / "repair_prompts" / "attempt-2.json")
    assert repair_prompt["repair_contract"]["issues"][0]["field_path"] == "summary"
    assert_draft_rendering_schema_page_fields(repair_prompt["repair_contract"]["schema"])
    assert any(ref.relative_path.endswith("repair_prompts/attempt-2.json") for ref in draft_step.outputs)


def test_draft_rendering_create_change_summary_is_filled_without_repair(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    fixture_dir = tmp_path / "draft-change-summary-fixture"
    fixture_dir.mkdir()
    for name in ["raw_prepare.json", "source_digest.json", "candidate_resolution.json", "wiki_merge_planning.json", "draft_rendering.json"]:
        data = read_json(FIXTURE_ROOT / "mock" / name)
        if name == "draft_rendering.json":
            data["pages"][0]["change_summary"] = ""
        write_json(fixture_dir / name, data)

    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=fixture_dir, slug="draft-change-summary")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    repair_report = read_json(run_dir / "draft_rendering" / "structured_repair_report.json")
    draft = read_json(run_dir / "draft_rendering" / "draft_rendering.json")

    assert manifest.status == OperationStatus.drafted
    assert repair_report["repair_count"] == 0
    assert draft["pages"][0]["change_summary"] == "创建 知识编译工程骨架 页面。"


def test_draft_rendering_create_english_change_summary_is_filled_without_repair(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    fixture_dir = tmp_path / "draft-english-change-summary-fixture"
    fixture_dir.mkdir()
    for name in ["raw_prepare.json", "source_digest.json", "candidate_resolution.json", "wiki_merge_planning.json", "draft_rendering.json"]:
        data = read_json(FIXTURE_ROOT / "mock" / name)
        if name == "draft_rendering.json":
            data["pages"][0]["change_summary"] = (
                "Create a new page about knowledge compilation scaffolding from the approved source and explain why the project matters."
            )
        write_json(fixture_dir / name, data)

    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=fixture_dir, slug="draft-english-change-summary")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    repair_report = read_json(run_dir / "draft_rendering" / "structured_repair_report.json")
    draft = read_json(run_dir / "draft_rendering" / "draft_rendering.json")

    assert manifest.status == OperationStatus.drafted
    assert repair_report["repair_count"] == 0
    assert draft["pages"][0]["change_summary"] == "创建 知识编译工程骨架 页面。"


def test_draft_rendering_create_english_source_coverage_notes_is_filled_without_repair(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    fixture_dir = tmp_path / "draft-english-source-coverage-fixture"
    fixture_dir.mkdir()
    for name in ["raw_prepare.json", "source_digest.json", "candidate_resolution.json", "wiki_merge_planning.json", "draft_rendering.json"]:
        data = read_json(FIXTURE_ROOT / "mock" / name)
        if name == "draft_rendering.json":
            data["pages"][0]["source_coverage_notes"] = (
                "Based on global excerpt and snippets from source material; covers introduction and methodology sections."
            )
        write_json(fixture_dir / name, data)

    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=fixture_dir, slug="draft-english-source-coverage")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    repair_report = read_json(run_dir / "draft_rendering" / "structured_repair_report.json")
    draft = read_json(run_dir / "draft_rendering" / "draft_rendering.json")

    assert manifest.status == OperationStatus.drafted
    assert repair_report["repair_count"] == 0
    assert draft["pages"][0]["source_coverage_notes"].startswith("依据本轮来源摘录中与")
    assert "Based on" not in draft["pages"][0]["source_coverage_notes"]


def test_draft_rendering_batches_large_page_sets(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    fixture_dir = tmp_path / "draft-batch-fixture"
    fixture_dir.mkdir()
    write_json(fixture_dir / "raw_prepare.json", read_json(FIXTURE_ROOT / "mock" / "raw_prepare.json"))
    digest = read_json(FIXTURE_ROOT / "mock" / "source_digest.json")
    resolution = read_json(FIXTURE_ROOT / "mock" / "candidate_resolution.json")
    merge = read_json(FIXTURE_ROOT / "mock" / "wiki_merge_planning.json")
    draft = read_json(FIXTURE_ROOT / "mock" / "draft_rendering.json")
    concept_template = digest["concepts"][0]
    resolution_template = resolution["items"][0]
    merge_template = merge["items"][0]
    draft_template = draft["pages"][0]
    digest["concepts"] = []
    digest["designs"] = []
    resolution["items"] = []
    merge["items"] = []
    draft_pages = []
    titles = [
        "队列编排 Alpha",
        "运行时沙箱 Bravo",
        "证据账本 Charlie",
        "审核路由 Delta",
        "漂移哨兵 Echo",
        "写入回执 Foxtrot",
        "上下文裁剪 Golf",
    ]
    summaries = [
        "队列编排 Alpha 协调待处理入库任务穿过确定性门禁。",
        "运行时沙箱 Bravo 隔离 provider 执行与本地 artifact 组装。",
        "证据账本 Charlie 在知识页生成前记录有来源支撑的主张。",
        "审核路由 Delta 把高风险草稿送到合适的人类检查点。",
        "漂移哨兵 Echo 在过期计划 apply 前发现 wiki 变化。",
        "写入回执 Foxtrot 把写入凭证保存在临时 run artifact 之外。",
        "上下文裁剪 Golf 在保留 validator 证据时压缩模型 payload。",
    ]
    for index, (title, summary) in enumerate(zip(titles, summaries), start=1):
        candidate_id = f"CAND-BATCH-{index:03d}"
        source_basis = {
            "source_candidate_ids": [candidate_id],
            "prepared_discovered_candidates": [],
            "source_locator": f"测试 / 第 {index} 段",
        }
        page_plan_id = pipeline_module.stable_page_plan_id(
            "concept",
            title,
            pipeline_module.source_basis_fingerprint(SourceBasis.model_validate(source_basis)),
        )
        target_path = f"concepts/Concept_{title}.md"
        candidate = json.loads(json.dumps(concept_template))
        candidate.update(
            {
                "candidate_id": candidate_id,
                "name": title,
                "suggested_page_title": title,
                "one_sentence_summary": summary,
                "why_matters": f"{title} 代表一个独立的分批渲染测试主题。",
                "wiki_value": f"{title} 让测试页面与相邻主题保持语义区分。",
                "source_locator": f"测试 / 第 {index} 段",
            }
        )
        digest["concepts"].append(candidate)
        resolution_item = json.loads(json.dumps(resolution_template))
        resolution_item["page_plan_id"] = ""
        resolution_item["source_basis"] = source_basis
        resolution_item["display_title"] = title
        resolution_item["topic_summary"] = candidate["one_sentence_summary"]
        resolution["items"].append(resolution_item)
        merge_item = json.loads(json.dumps(merge_template))
        merge_item["page_plan_id"] = page_plan_id
        merge_item["source_basis"] = source_basis
        merge_item["canonical_target_path"] = target_path
        merge_item["display_title"] = title
        merge_item["new_understanding"] = candidate["one_sentence_summary"]
        merge_item["related_pages"] = []
        merge["items"].append(merge_item)
        draft_page = json.loads(json.dumps(draft_template))
        draft_page["page_plan_id"] = page_plan_id
        draft_page["canonical_target_path"] = target_path
        draft_page["summary"] = candidate["one_sentence_summary"]
        draft_page["body_markdown"] = (
            f"### 运行机制\n\n"
            f"{title} 的详情来自测试源材料。它把一个独立流程环节放进可验证的 ingest 管线中，"
            "例如可以用固定输入检查状态推进、artifact 写入和后续审核边界。\n\n"
            "### 价值\n\n"
            "这个主题的价值在于让批量渲染测试能区分相邻页面，避免多个页面挤成同一个泛化摘要。"
        )
        draft_page["change_summary"] = f"创建 {title}。"
        draft_page["source_coverage_notes"] = f"覆盖 {candidate_id}。"
        draft_pages.append(draft_page)
    write_json(fixture_dir / "source_digest.json", digest)
    write_json(fixture_dir / "candidate_resolution.json", resolution)
    write_json(fixture_dir / "wiki_merge_planning.json", merge)
    write_json(fixture_dir / "draft_rendering.1.json", {"schema_version": "draft_rendering.v3", "pages": draft_pages[:4]})
    bad_second_batch = {"schema_version": "draft_rendering.v3", "pages": json.loads(json.dumps(draft_pages[4:6]))}
    write_json(fixture_dir / "draft_rendering.2.json", bad_second_batch)
    write_json(fixture_dir / "draft_rendering.3.json", {"schema_version": "draft_rendering.v3", "pages": draft_pages[4:]})

    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=fixture_dir, slug="draft-batch")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    batch_report = read_json(run_dir / "draft_rendering" / "draft_rendering_batch_report.json")
    draft_artifact = read_json(run_dir / "draft_rendering" / "draft_rendering.json")
    repair_report = read_json(run_dir / "draft_rendering" / "structured_repair_report.json")
    metrics = read_json(run_dir / "run_metrics.json")

    assert manifest.status == OperationStatus.drafted
    assert batch_report["parallel"] is False
    assert batch_report["max_parallel_batches"] == 1
    assert batch_report["batch_count"] == 2
    assert batch_report["page_count"] == 7
    assert [batch["attempt_count"] for batch in batch_report["batches"]] == [1, 2]
    assert [batch["http_attempt_count"] for batch in batch_report["batches"]] == [1, 2]
    assert [batch["repair_count"] for batch in batch_report["batches"]] == [0, 1]
    assert batch_report["http_attempt_count"] == 3
    assert batch_report["model_duration_ms"] == batch_report["duration_ms"]
    assert batch_report["wall_duration_ms"] >= 0
    assert batch_report["payload_char_count"] > 0
    assert batch_report["max_batch_payload_char_count"] > 0
    assert batch_report["avg_batch_payload_char_count"] > 0
    assert all(batch["payload_char_count"] > 0 for batch in batch_report["batches"])
    assert not (run_dir / "draft_rendering" / "draft_source_excerpt_pack.json").exists()
    assert not (run_dir / "draft_rendering" / "draft_source_excerpt_pack.md").exists()
    assert not (run_dir / "draft_rendering" / "update_preservation_pack.json").exists()
    assert not (run_dir / "draft_rendering" / "update_preservation_pack.md").exists()
    first_batch_source_pack = read_json(run_dir / "draft_rendering" / "model_batches" / "batch-001" / "draft_source_excerpt_pack.json")
    second_batch_source_pack = read_json(run_dir / "draft_rendering" / "model_batches" / "batch-002" / "draft_source_excerpt_pack.json")
    assert first_batch_source_pack["force_excerpt"] is True
    assert first_batch_source_pack["full_source_in_payload"] is False
    assert second_batch_source_pack["force_excerpt"] is True
    assert second_batch_source_pack["full_source_in_payload"] is False
    batch_report_markdown = (run_dir / "draft_rendering" / "draft_rendering_batch_report.md").read_text(encoding="utf-8")
    assert "Payload Chars" in batch_report_markdown
    assert "HTTP Attempts" in batch_report_markdown
    assert "墙钟耗时" in batch_report_markdown
    assert "最大单批 payload" in batch_report_markdown
    assert len(draft_artifact["pages"]) == 7
    assert (run_dir / "draft_rendering" / "model_batches" / "batch-001" / "provider_result.json").exists()
    assert (run_dir / "draft_rendering" / "model_batches" / "batch-002" / "provider_result.json").exists()
    aggregate_provider_result = read_json(run_dir / "draft_rendering" / "provider_result.json")
    assert aggregate_provider_result["http_attempt_count"] == 3
    assert (run_dir / "draft_rendering" / "model_batches" / "batch-002" / "repair_prompts" / "attempt-2.json").exists()
    repair_prompt = read_json(run_dir / "draft_rendering" / "model_batches" / "batch-002" / "repair_prompts" / "attempt-2.json")
    assert repair_prompt["repair_contract"]["mode"] == "missing_page_completion"
    assert repair_prompt["repair_contract"]["missing_page_plan_ids"] == [draft_pages[6]["page_plan_id"]]
    assert [page["page_plan_id"] for page in repair_prompt["accepted_partial_pages"]] == [
        draft_pages[4]["page_plan_id"],
        draft_pages[5]["page_plan_id"],
    ]
    assert all(set(page) == CURRENT_DRAFT_PAGE_FIELDS for page in repair_prompt["accepted_partial_pages"])
    assert_draft_rendering_schema_page_fields(repair_prompt["repair_contract"]["schema"])
    assert repair_prompt["missing_page_payload"]["required_page_plan_ids"] == [draft_pages[6]["page_plan_id"]]
    assert repair_report["provider"] == "batched:mock"
    assert repair_report["attempt_count"] == 3
    assert repair_report["repair_count"] == 1
    assert repair_report["attempts"][1]["provider_result_ref"] == "model_batches/batch-002/provider_results/attempt-1.json"
    assert repair_report["attempts"][2]["repair_prompt_ref"] == "model_batches/batch-002/repair_prompts/attempt-2.json"
    draft_step = next(step for step in manifest.steps if step.name == "draft_rendering")
    assert "draft_rendering_batch_report.v1" in [ref.schema_version for ref in draft_step.outputs]
    assert any(ref.relative_path.endswith("model_batches/batch-001/provider_result.json") for ref in draft_step.outputs)
    assert any(ref.relative_path.endswith("model_batches/batch-002/structured_repair_report.json") for ref in draft_step.outputs)
    assert any(
        ref.relative_path.endswith("model_batches/batch-002/repair_prompts/attempt-2.json") and not ref.required_for_resume
        for ref in draft_step.outputs
    )
    draft_metrics = next(step for step in metrics["steps"] if step["name"] == "draft_rendering")
    assert draft_metrics["internal_model_call_count"] == 3
    assert draft_metrics["repair_count"] == 1
    assert draft_metrics["provider_result_count"] == 3
    assert draft_metrics["http_attempt_count"] == 3
    assert draft_metrics["payload_char_count"] > 0


def test_candidate_resolution_sanitizes_weak_noise_formal_item_without_repair(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    fixture_dir = tmp_path / "candidate-noise-repair"
    fixture_dir.mkdir()
    for name in ["raw_prepare.json", "source_digest.json", "wiki_merge_planning.json", "draft_rendering.json"]:
        data = read_json(FIXTURE_ROOT / "mock" / name)
        if name == "source_digest.json":
            data["weak_or_noise_items"] = [
                {
                    "candidate_id": "noise-1",
                    "name": "噪声片段",
                    "type": "noise",
                    "one_sentence_summary": "这只是口播噪声，不应该成为页面。",
                    "why_matters": "用于确认噪声不会泄漏进正式页面规划。",
                    "wiki_value": "不入库。",
                    "suggested_action": "ignore",
                }
            ]
        write_json(fixture_dir / name, data)
    bad_resolution = read_json(FIXTURE_ROOT / "mock" / "candidate_resolution.json")
    bad_resolution["items"].append(
        {
            "page_plan_id": "",
            "source_basis": {
                "source_candidate_ids": ["noise-1"],
                "prepared_discovered_candidates": ["模型声称从噪声里发现的新主题"],
                "source_locator": "noise",
            },
            "page_type": "noise",
            "display_title": "噪声片段",
            "path_stem": "",
            "candidate_target_path": "",
            "topic_summary": "这只是噪声。",
            "why_this_page": "ignore",
            "initial_section_intent": "ignore",
            "coverage_notes": "ignore",
            "reason": "ignore",
        }
    )
    write_json(fixture_dir / "candidate_resolution.1.json", bad_resolution)
    write_json(fixture_dir / "candidate_resolution.2.json", read_json(FIXTURE_ROOT / "mock" / "candidate_resolution.json"))

    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=fixture_dir, slug="candidate-noise")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    repair_report = read_json(run_dir / "candidate_resolution" / "structured_repair_report.json")
    resolution = read_json(run_dir / "candidate_resolution" / "candidate_resolution.json")

    assert repair_report["repair_count"] == 0
    assert repair_report["attempts"][0]["issues"] == []
    assert all(item["page_type"] != "noise" for item in resolution["items"])
    assert "noise-1" not in {
        candidate_id
        for item in resolution["items"]
        for candidate_id in item["source_basis"]["source_candidate_ids"]
    }
    assert any("dropped because it only referenced weak/noise candidates" in note for note in resolution["missed_candidate_risks"])


def test_candidate_resolution_moves_prepared_discovered_unknown_ids_out_of_source_ids(tmp_path: Path) -> None:
    vault, _ = make_vault(tmp_path)
    profile = pipeline_module.load_profile(vault / ".llmwiki" / "profiles" / "project_basic")
    digest = SourceDigestArtifact(
        source_raw_path="raw/sample.md",
        summary="Digest summary.",
        concepts=[
            SourceDigestCandidate(
                candidate_id="CAND001",
                name="Known candidate",
                type="concept",
                one_sentence_summary="Known candidate summary.",
                why_matters="It matters.",
                wiki_value="It belongs in the wiki.",
                suggested_page_title="Known candidate",
            )
        ],
    )
    artifact = CandidateResolutionArtifact(
        items=[
            resolution_item(
                "CAND001",
                page_type="concept",
                target_path="concepts/Concept_Known.md",
                display_title="Known candidate",
            ),
            CandidateResolutionItem(
                source_basis=SourceBasis(
                    source_candidate_ids=["concept-ai-pm-foundation"],
                    prepared_discovered_candidates=["concept-ai-pm-foundation"],
                    source_locator="第四步",
                ),
                page_type="concept",
                display_title="AI PM基础",
                topic_summary="通用产品管理基础对 AI PM 仍然重要。",
                why_this_page="这是 prepared raw 中出现但 source_digest 漏掉的主题。",
                reason="new",
            ),
        ]
    )

    finalized = pipeline_module.finalize_candidate_resolution(vault, profile, artifact, digest)
    pipeline_module.validate_candidate_resolution(digest, finalized)
    discovered = [item for item in finalized.items if item.display_title == "AI PM基础"][0]

    assert discovered.source_basis.source_candidate_ids == []
    assert discovered.source_basis.prepared_discovered_candidates == ["concept-ai-pm-foundation"]
    assert "非 source_digest candidate id" in discovered.coverage_notes
    assert any("moved unknown candidate refs" in note for note in finalized.missed_candidate_risks)


def test_candidate_resolution_moves_unknown_only_source_ids_to_prepared_discovered(tmp_path: Path) -> None:
    vault, _ = make_vault(tmp_path)
    profile = pipeline_module.load_profile(vault / ".llmwiki" / "profiles" / "project_basic")
    digest = SourceDigestArtifact(
        source_raw_path="raw/sample.md",
        summary="Digest summary.",
        concepts=[
            SourceDigestCandidate(
                candidate_id="CAND001",
                name="Known candidate",
                type="concept",
                one_sentence_summary="Known candidate summary.",
                why_matters="It matters.",
                wiki_value="It belongs in the wiki.",
                suggested_page_title="Known candidate",
            )
        ],
    )
    artifact = CandidateResolutionArtifact(
        items=[
            resolution_item(
                "CAND001",
                page_type="concept",
                target_path="concepts/Concept_Known.md",
                display_title="Known candidate",
            ),
            CandidateResolutionItem(
                source_basis=SourceBasis(source_candidate_ids=["new-topic-from-prepared"], source_locator="第五步"),
                page_type="concept",
                display_title="Prepared 新主题",
                topic_summary="prepared raw 中出现的新主题。",
                why_this_page="这是 prepared raw 中出现但 source_digest 漏掉的主题。",
                reason="new",
            )
        ]
    )

    finalized = pipeline_module.finalize_candidate_resolution(vault, profile, artifact, digest)
    pipeline_module.validate_candidate_resolution(digest, finalized)
    item = [item for item in finalized.items if item.display_title == "Prepared 新主题"][0]

    assert item.source_basis.source_candidate_ids == []
    assert item.source_basis.prepared_discovered_candidates == ["new-topic-from-prepared"]
    assert "非 source_digest candidate id" in item.coverage_notes
    assert any("moved unknown-only candidate refs" in note for note in finalized.missed_candidate_risks)


def test_candidate_resolution_moves_budget_deferred_source_ids_to_prepared_discovered(tmp_path: Path) -> None:
    vault, _ = make_vault(tmp_path)
    profile = pipeline_module.load_profile(vault / ".llmwiki" / "profiles" / "project_basic")
    digest = SourceDigestArtifact(
        source_raw_path="raw/sample.md",
        summary="Digest summary.",
        concepts=[
            SourceDigestCandidate(
                candidate_id="CAND001",
                name="Known candidate",
                type="concept",
                one_sentence_summary="Known candidate summary.",
                why_matters="It matters.",
                wiki_value="It belongs in the wiki.",
                suggested_page_title="Known candidate",
            )
        ],
        budget_deferred_candidates=[
            SourceDigestCandidate(
                candidate_id="C005",
                name="Deferred candidate",
                type="concept",
                one_sentence_summary="Deferred candidate summary.",
                why_matters="It matters later.",
                wiki_value="It may belong in the wiki.",
                suggested_page_title="Deferred candidate",
            )
        ],
    )
    artifact = CandidateResolutionArtifact(
        items=[
            resolution_item(
                "CAND001",
                page_type="concept",
                target_path="concepts/Concept_Known.md",
                display_title="Known candidate",
            ),
            CandidateResolutionItem(
                source_basis=SourceBasis(source_candidate_ids=["C005"], source_locator="延后候选"),
                page_type="concept",
                display_title="Deferred candidate",
                topic_summary="延后候选现在被复用。",
                why_this_page="模型误把 budget-deferred id 放进 source_candidate_ids。",
                reason="prepared_discovered",
            ),
        ]
    )

    finalized = pipeline_module.finalize_candidate_resolution(vault, profile, artifact, digest)
    pipeline_module.validate_candidate_resolution(digest, finalized)
    item = [item for item in finalized.items if item.display_title == "Deferred candidate"][0]

    assert item.source_basis.source_candidate_ids == []
    assert item.source_basis.prepared_discovered_candidates == ["C005"]
    assert "非 source_digest candidate id" in item.coverage_notes


def test_candidate_resolution_moves_deferred_source_ids_when_no_selected_candidates(tmp_path: Path) -> None:
    vault, _ = make_vault(tmp_path)
    profile = pipeline_module.load_profile(vault / ".llmwiki" / "profiles" / "project_basic")
    digest = SourceDigestArtifact(
        source_raw_path="raw/sample.md",
        summary="Digest summary.",
        budget_deferred_candidates=[
            SourceDigestCandidate(
                candidate_id="C005",
                name="Deferred candidate",
                type="concept",
                one_sentence_summary="Deferred candidate summary.",
                why_matters="It matters later.",
                wiki_value="It may belong in the wiki.",
                suggested_page_title="Deferred candidate",
            )
        ],
    )
    artifact = CandidateResolutionArtifact(
        items=[
            CandidateResolutionItem(
                source_basis=SourceBasis(source_candidate_ids=["C005"], source_locator="延后候选"),
                page_type="concept",
                display_title="Deferred candidate",
                topic_summary="延后候选现在被复用。",
                why_this_page="模型误把 budget-deferred id 放进 source_candidate_ids。",
                reason="prepared_discovered",
            ),
        ]
    )

    finalized = pipeline_module.finalize_candidate_resolution(vault, profile, artifact, digest)
    pipeline_module.validate_candidate_resolution(digest, finalized)
    item = finalized.items[0]

    assert item.source_basis.source_candidate_ids == []
    assert item.source_basis.prepared_discovered_candidates == ["C005"]
    assert "非 source_digest candidate id" in item.coverage_notes


def test_candidate_resolution_markdown_shows_prepared_discovered_candidates() -> None:
    artifact = CandidateResolutionArtifact(
        items=[
            CandidateResolutionItem(
                page_plan_id="PP-DISCOVERED",
                source_basis=SourceBasis(prepared_discovered_candidates=["new-topic"]),
                page_type="concept",
                display_title="Prepared 新主题",
                candidate_target_path="concepts/Concept_Prepared_新主题.md",
                topic_summary="prepared raw 中出现的新主题。",
                why_this_page="值得记录。",
                reason="new",
            )
        ]
    )

    rendered = pipeline_module.render_candidate_resolution_markdown(artifact)

    assert "Prepared 发现候选" in rendered
    assert "new-topic" in rendered


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

    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=fixture_dir, slug="wiki-prefixed-target")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    plan = read_json(run_dir / "wiki_merge_planning" / "wiki_merge_plan.json")
    first = plan["items"][0]

    assert first["action"] == "update"
    assert first["canonical_target_path"] == "concepts/Concept_知识编译工程骨架.md"
    assert first["matched_page"] == "concepts/Concept_知识编译工程骨架.md"
    assert manifest.status == OperationStatus.drafted
    assert not [step for step in manifest.steps if step.status == StepStatus.awaiting_review]
    assert (run_dir / "draft_review" / "approved_write_manifest.json").exists()
    assert (run_dir / "apply_preview" / "apply_preview.json").exists()


def test_update_and_noop_same_target_are_merged_by_finalizer(tmp_path: Path) -> None:
    vault, _ = make_vault(tmp_path)
    existing = vault / "wiki" / "concepts" / "Concept_知识编译工程骨架.md"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text(
        "---\n"
        "llmwiki_type: concept\n"
        "title: 知识编译工程骨架\n"
        "aliases: []\n"
        "summary: 已有知识编译工程骨架。\n"
        "updated: 2026-01-02\n"
        "---\n\n"
        "# 知识编译工程骨架\n",
        encoding="utf-8",
    )
    resolution = CandidateResolutionArtifact(
        items=[
            resolution_item(
                "CAND001",
                page_type="concept",
                target_path="concepts/Concept_知识编译工程骨架.md",
                display_title="知识编译工程骨架",
            ),
            resolution_item(
                "CAND002",
                page_type="concept",
                target_path="concepts/Concept_知识编译工程骨架.md",
                display_title="知识编译工程骨架",
            ),
        ]
    )
    snapshot = build_wiki_context_snapshot(
        vault,
        resolution,
        log_date="2026-06-03",
        source_target_path="sources/Source_Test.md",
    )
    plan = pipeline_module.finalize_wiki_merge_plan(
        pipeline_module.WikiMergePlanArtifact(
            log_date="",
            context_snapshot_ref="",
            items=[
                pipeline_module.WikiMergePlanItem(
                    page_plan_id="PP-CAND001",
                    source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
                    action="update",
                    canonical_target_path="concepts/Concept_知识编译工程骨架.md",
                    display_title="知识编译工程骨架",
                    page_type="concept",
                    matched_page="concepts/Concept_知识编译工程骨架.md",
                    new_understanding="新增工程骨架理解。",
                    section_plans={"summary": "Summary"},
                    reason="更新已有页。",
                ),
                pipeline_module.WikiMergePlanItem(
                    page_plan_id="PP-CAND002",
                    source_basis=SourceBasis(source_candidate_ids=["CAND002"]),
                    action="noop",
                    canonical_target_path="concepts/Concept_知识编译工程骨架.md",
                    display_title="知识编译工程骨架",
                    page_type="concept",
                    matched_page="concepts/Concept_知识编译工程骨架.md",
                    new_understanding="已有页已覆盖。",
                    section_plans={},
                    reason="已被已有页覆盖。",
                ),
            ],
        ),
        resolution,
        snapshot,
        "wiki_context_snapshot/wiki_context_snapshot.json",
    )

    assert len(plan.items) == 1
    assert plan.items[0].action == "update"
    assert set(plan.items[0].source_basis.source_candidate_ids) == {"CAND001", "CAND002"}
    assert set(plan.items[0].merged_page_plan_ids) == {"PP-CAND001", "PP-CAND002"}
    assert plan.items[0].noop_covered_by_update is True


def test_same_source_duplicate_create_items_are_merged_without_losing_coverage(tmp_path: Path) -> None:
    vault, _ = make_vault(tmp_path)
    digest = SourceDigestArtifact(
        source_raw_path="raw/sample.md",
        summary="Digest summary.",
        concepts=[
            SourceDigestCandidate(
                candidate_id=f"CAND00{index}",
                name=title,
                type=page_type,
                one_sentence_summary=f"{title} 摘要。",
                why_matters="它值得沉淀。",
                wiki_value="它属于 wiki。",
                suggested_page_title=title,
            )
            for index, page_type, title in [
                (1, "concept", "Agent 与 Workflow 对比"),
                (2, "comparison", "Workflow vs Agent"),
                (3, "concept", "RAG 概念"),
                (4, "design", "RAG 系统设计"),
            ]
        ],
    )
    resolution = CandidateResolutionArtifact(
        items=[
            resolution_item("CAND001", page_type="concept", target_path="concepts/Concept_Agent 与 Workflow 对比.md", display_title="Agent 与 Workflow 对比"),
            resolution_item("CAND002", page_type="comparison", target_path="comparisons/Comparison_Workflow vs Agent.md", display_title="Workflow vs Agent"),
            resolution_item("CAND003", page_type="concept", target_path="concepts/Concept_RAG 概念.md", display_title="RAG 概念"),
            resolution_item("CAND004", page_type="design", target_path="designs/Design_RAG 系统设计.md", display_title="RAG 系统设计"),
        ]
    )
    snapshot = build_wiki_context_snapshot(vault, resolution, log_date="2026-06-06", source_target_path="sources/Source_Test.md")

    def plan_item(
        page_plan_id: str,
        candidate_id: str,
        page_type: str,
        target_path: str,
        title: str,
        related_pages: list[pipeline_module.RelatedPageRef] | None = None,
    ) -> pipeline_module.WikiMergePlanItem:
        return pipeline_module.WikiMergePlanItem(
            page_plan_id=page_plan_id,
            source_basis=SourceBasis(source_candidate_ids=[candidate_id]),
            action="create",
            canonical_target_path=target_path,
            display_title=title,
            page_type=page_type,
            new_understanding=f"{title} 说明 Agent 与 Workflow 的执行边界、适用场景和判断价值。",
            knowledge_delta=f"{title} 补充 Agent 与 Workflow 的边界差异。",
            why_this_matters="帮助判断什么时候用稳定流程，什么时候用智能体。",
            value_points=["帮助判断方案边界。"],
            section_plans={"summary": f"{title} 摘要。", "detail": f"{title} 讨论 Agent 与 Workflow 的差异。"},
            related_pages=related_pages or [],
            reason=f"{title} 值得沉淀。",
        )

    plan = pipeline_module.finalize_wiki_merge_plan(
        pipeline_module.WikiMergePlanArtifact(
            log_date="2026-06-06",
            context_snapshot_ref="wiki_context_snapshot/wiki_context_snapshot.json",
                items=[
                    plan_item(
                        "PP-CAND001",
                        "CAND001",
                        "concept",
                        "concepts/Concept_Agent 与 Workflow 对比.md",
                        "Agent 与 Workflow 对比",
                        related_pages=[
                            pipeline_module.RelatedPageRef(
                                target_path="comparisons/Comparison_Workflow vs Agent.md",
                                display_title="Workflow vs Agent",
                                source="source_digest",
                                reason="两页描述同一组边界。",
                            )
                        ],
                    ),
                    plan_item("PP-CAND002", "CAND002", "comparison", "comparisons/Comparison_Workflow vs Agent.md", "Workflow vs Agent"),
                    pipeline_module.WikiMergePlanItem(
                        page_plan_id="PP-CAND003",
                        source_basis=SourceBasis(source_candidate_ids=["CAND003"]),
                        action="create",
                        canonical_target_path="concepts/Concept_RAG 概念.md",
                        display_title="RAG 概念",
                        page_type="concept",
                        new_understanding="RAG 概念解释检索增强生成如何把外部材料带入模型回答。",
                        knowledge_delta="补充 RAG 作为概念的定义和适用边界。",
                        why_this_matters="帮助区分 RAG 概念和具体系统设计。",
                        value_points=["帮助判断什么时候需要检索增强。"],
                        section_plans={"summary": "RAG 概念摘要。", "detail": "RAG 概念讨论检索增强生成的定义和边界。"},
                        related_pages=[
                            pipeline_module.RelatedPageRef(
                                target_path="concepts/Concept_Agent 与 Workflow 对比.md",
                                display_title="Agent 与 Workflow 对比",
                                source="source_digest",
                                reason="RAG 概念可与 Agent/Workflow 边界对比。",
                            )
                        ],
                        reason="RAG 概念值得沉淀。",
                    ),
                pipeline_module.WikiMergePlanItem(
                    page_plan_id="PP-CAND004",
                    source_basis=SourceBasis(source_candidate_ids=["CAND004"]),
                    action="create",
                    canonical_target_path="designs/Design_RAG 系统设计.md",
                    display_title="RAG 系统设计",
                    page_type="design",
                    new_understanding="RAG 系统设计关注检索、编排、评估和工具实现。",
                    knowledge_delta="补充 RAG 方案实现路径。",
                    why_this_matters="帮助判断 RAG 什么时候是系统方案而非单一概念。",
                    value_points=["帮助设计检索增强生成方案。"],
                    section_plans={"summary": "RAG 系统设计摘要。", "detail": "RAG 系统设计讨论检索、编排和评估。"},
                    reason="RAG 系统设计值得沉淀。",
                ),
            ],
        ),
        resolution,
        snapshot,
        "wiki_context_snapshot/wiki_context_snapshot.json",
    )

    paths = {item.canonical_target_path for item in plan.items}
    assert "comparisons/Comparison_Workflow vs Agent.md" in paths
    assert "concepts/Concept_Agent 与 Workflow 对比.md" not in paths
    assert "concepts/Concept_RAG 概念.md" in paths
    assert "designs/Design_RAG 系统设计.md" in paths
    agent_item = next(item for item in plan.items if item.canonical_target_path == "comparisons/Comparison_Workflow vs Agent.md")
    assert set(agent_item.source_basis.source_candidate_ids) == {"CAND001", "CAND002"}
    assert {"PP-CAND001", "PP-CAND002"} <= set(agent_item.merged_page_plan_ids)
    assert "合并自" in "\n".join(agent_item.section_plans.values())
    assert agent_item.related_pages == []
    rag_item = next(item for item in plan.items if item.canonical_target_path == "concepts/Concept_RAG 概念.md")
    assert [(related.target_path, related.display_title) for related in rag_item.related_pages] == [
        ("comparisons/Comparison_Workflow vs Agent.md", "Workflow vs Agent")
    ]
    validate_wiki_merge_plan(digest, plan, resolution, snapshot, language="zh-CN")


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

    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=fixture_dir, slug="source-recorded")
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

    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=FIXTURE_ROOT / "mock", slug="provenance")
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
    fixture_dir = tmp_path / f"grounding-fixture-{from_step}"
    fixture_dir.mkdir()
    for name in ["raw_prepare.json", "source_digest.json", "candidate_resolution.json", "wiki_merge_planning.json", "draft_rendering.json"]:
        data = read_json(FIXTURE_ROOT / "mock" / name)
        if name == "raw_prepare.json":
            data["prepared_markdown"] += "\n\nAnthropic 收购了 OpenAI。"
        if name == "draft_rendering.json":
            data["pages"][0]["body_markdown"] += "\n\nOpenAI 收购了 Anthropic。"
        write_json(fixture_dir / name, data)

    run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=fixture_dir, slug=f"skip-{from_step}")
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

    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=fixture_dir, slug="needs-human")
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
    before_approve = status(vault, manifest.operation_id)
    before_step = [step for step in before_approve.steps if step.name == "merge_plan_review"][0]
    before_duration = before_step.attempts[-1].duration_ms

    approved = approve_review(vault, manifest.operation_id, "merge_plan_review")
    assert approved.status == OperationStatus.running
    assert (run_dir / "merge_plan_review" / "approved_merge_plan.json").exists()
    assert pending_path.exists()
    approved_step = [step for step in approved.steps if step.name == "merge_plan_review"][0]
    assert approved_step.status == StepStatus.approved
    assert approved_step.error is None
    assert approved_step.attempts[-1].error is None
    assert approved_step.attempts[-1].duration_ms == before_duration
    assert any(ref.relative_path.endswith("pending_merge_plan.json") for ref in approved_step.outputs)
    decision = read_json(run_dir / "merge_plan_review" / "review_decision.json")
    assert "取代" in decision["notes"]
    metrics = read_json(run_dir / "run_metrics.json")
    assert metrics["status"] == "running"
    metric_step = [step for step in metrics["steps"] if step["name"] == "merge_plan_review"][0]
    assert metric_step["status"] == "approved"

    config = read_yaml(vault / ".llmwiki" / "config.yaml")
    config["providers"]["default"] = {
        "spec": "mock:fixture",
        "fixture_dir": str(fixture_dir),
    }
    write_yaml(vault / ".llmwiki" / "config.yaml", config)
    resumed = resume_ingest(vault=vault, operation_id=manifest.operation_id)
    assert resumed.status == OperationStatus.drafted
    assert (run_dir / "apply_preview" / "apply_preview.json").exists()


def test_review_approval_rejects_upstream_artifact_replaced_by_directory(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    fixture_dir = tmp_path / "needs-human-dir-fixture"
    fixture_dir.mkdir()
    for name in ["raw_prepare.json", "source_digest.json", "candidate_resolution.json", "wiki_merge_planning.json", "draft_rendering.json"]:
        data = read_json(FIXTURE_ROOT / "mock" / name)
        if name == "wiki_merge_planning.json":
            data["items"][0]["action"] = "needs_human_decision"
            data["items"][0]["apply_eligibility"] = "blocked"
            data["items"][0]["blocked_reason"] = "需要人工决定是否创建。"
        write_json(fixture_dir / name, data)

    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=fixture_dir, slug="review-artifact-dir")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    plan_path = run_dir / "wiki_merge_planning" / "wiki_merge_plan.json"
    plan_path.unlink()
    plan_path.mkdir()

    with pytest.raises(PipelineError, match="wiki_merge_planning/wiki_merge_plan.json is not a file"):
        approve_review(vault, manifest.operation_id, "merge_plan_review")


def test_revise_review_deletes_pending_artifacts_before_reset(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    fixture_dir = tmp_path / "needs-human-revise-fixture"
    fixture_dir.mkdir()
    for name in ["raw_prepare.json", "source_digest.json", "candidate_resolution.json", "wiki_merge_planning.json", "draft_rendering.json"]:
        data = read_json(FIXTURE_ROOT / "mock" / name)
        if name == "wiki_merge_planning.json":
            data["items"][0]["action"] = "needs_human_decision"
            data["items"][0]["apply_eligibility"] = "blocked"
            data["items"][0]["blocked_reason"] = "需要人工决定是否创建。"
        write_json(fixture_dir / name, data)

    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=fixture_dir, slug="review-revise")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)

    revised = revise_review(vault, manifest.operation_id, "merge_plan_review")

    assert revised.status == OperationStatus.running
    assert not (run_dir / "review_archive").exists()
    assert not (run_dir / "merge_plan_review").exists()
    assert not (run_dir / "wiki_merge_planning").exists()


def test_apply_rejects_incomplete_steps_even_if_manifest_is_marked_drafted(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=FIXTURE_ROOT / "mock", slug="tampered")
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
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=FIXTURE_ROOT / "mock", slug="empty-preview")
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


def test_source_digest_english_user_text_fails_for_zh_cn_vault(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    fixture_dir = tmp_path / "english-digest-fixture"
    fixture_dir.mkdir()
    for name in ["raw_prepare.json", "source_digest.json", "candidate_resolution.json", "wiki_merge_planning.json", "draft_rendering.json"]:
        data = read_json(FIXTURE_ROOT / "mock" / name)
        if name == "source_digest.json":
            data["summary"] = (
                "This source explains how product managers should work with agents, workflows, evaluation loops, and career strategy."
            )
        write_json(fixture_dir / name, data)

    with pytest.raises(PipelineError, match="must be Chinese"):
        run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=fixture_dir, slug="english-digest")

    manifest = status(vault, latest_operation(vault) or "")
    failed_step = [step for step in manifest.steps if step.status == StepStatus.failed][0]
    assert failed_step.name == "source_digest"


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

    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=fixture_dir, slug="source-links")
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

    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=fixture_dir, slug="prefixed")
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

    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=fixture_dir, slug="cross-prefix")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    resolution = read_json(run_dir / "candidate_resolution" / "candidate_resolution.json")
    item = [item for item in resolution["items"] if "CAND001" in item["source_basis"]["source_candidate_ids"]][0]

    assert item["page_type"] == "entity"
    assert item["candidate_target_path"] == "entities/Entity_Foo.md"
    assert item["display_title"] == "Foo"


def test_blocks_overwriting_unsupported_system_page(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    (vault / "wiki" / "index.md").write_text("# My human index\n", encoding="utf-8")

    with pytest.raises(PipelineError, match="system page is not supported by this engine"):
        run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=FIXTURE_ROOT / "mock", slug="human-index")

    assert (vault / "wiki" / "index.md").read_text(encoding="utf-8") == "# My human index\n"


def test_blocks_old_system_marker(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    (vault / "wiki" / "index.md").write_text(
        "# index\n\n<!-- llmwiki:system-page:v1 -->\n",
        encoding="utf-8",
    )

    with pytest.raises(PipelineError, match="system page is not supported by this engine"):
        run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=FIXTURE_ROOT / "mock", slug="v1-index")


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


def test_related_pages_resolve_prepared_discovered_candidate_refs() -> None:
    digest = SourceDigestArtifact(
        source_raw_path="raw/sample.md",
        summary="Digest summary.",
        concepts=[
            SourceDigestCandidate(
                candidate_id="C001",
                name="主候选",
                type="concept",
                one_sentence_summary="主候选摘要。",
                why_matters="它是主主题。",
                wiki_value="应成为概念页。",
                suggested_page_title="主候选",
                related_candidates=["C005"],
            )
        ],
        budget_deferred_candidates=[
            SourceDigestCandidate(
                candidate_id="C005",
                name="延后候选",
                type="concept",
                one_sentence_summary="延后候选摘要。",
                why_matters="它是相关主题。",
                wiki_value="应成为概念页。",
                suggested_page_title="延后候选",
            )
        ],
    )
    resolution = CandidateResolutionArtifact(
        items=[
            CandidateResolutionItem(
                page_plan_id="PP-C001",
                source_basis=SourceBasis(source_candidate_ids=["C001"]),
                page_type="concept",
                display_title="主候选",
                candidate_target_path="concepts/Concept_主候选.md",
                topic_summary="主候选摘要。",
                why_this_page="值得记录。",
                reason="new",
            ),
            CandidateResolutionItem(
                page_plan_id="PP-C005",
                source_basis=SourceBasis(prepared_discovered_candidates=["C005"]),
                page_type="concept",
                display_title="延后候选",
                candidate_target_path="concepts/Concept_延后候选.md",
                topic_summary="延后候选摘要。",
                why_this_page="值得记录。",
                reason="prepared_discovered",
            ),
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-06",
        source_target_path="sources/Source_Test.md",
        entries=[
            pipeline_module.WikiContextEntry(path="wiki/concepts/Concept_主候选.md", expected_state="missing"),
            pipeline_module.WikiContextEntry(path="wiki/concepts/Concept_延后候选.md", expected_state="missing"),
        ],
    )

    plan = build_wiki_merge_plan(resolution, digest, snapshot, log_date="2026-06-06")
    first = plan.items[0]

    assert [(item.target_path, item.display_title, item.source) for item in first.related_pages] == [
        ("concepts/Concept_延后候选.md", "延后候选", "source_digest")
    ]
    assert first.related_unresolved == []


def test_wiki_context_snapshot_writes_candidate_contexts_and_metadata_poor_fallback(tmp_path: Path) -> None:
    vault, _ = make_vault(tmp_path)
    hand_written = vault / "wiki" / "concepts" / "Concept_Handwritten_Agent.md"
    hand_written.parent.mkdir(parents=True, exist_ok=True)
    hand_written.write_text(
        "# Handwritten Agent Page\n\n"
        "Workflow and agent execution differ in autonomy, feedback loops, and tool use.\n",
        encoding="utf-8",
    )
    resolution = CandidateResolutionArtifact(
        items=[
            resolution_item(
                "CAND001",
                page_type="concept",
                target_path="concepts/Concept_Workflow vs Agent.md",
                display_title="Workflow vs Agent",
                summary="Workflow and agent execution comparison.",
            )
        ]
    )

    snapshot = build_wiki_context_snapshot(
        vault,
        resolution,
        log_date="2026-06-03",
        source_target_path="sources/Source_Test.md",
    )

    pool_entry = [entry for entry in snapshot.knowledge_metadata_pool if entry.path == "concepts/Concept_Handwritten_Agent.md"][0]
    assert pool_entry.metadata is None
    assert pool_entry.indexable is False
    context_item = snapshot.candidate_contexts.items[0]
    assert "concepts/Concept_Handwritten_Agent.md" in context_item.unindexable_pages
    assert any(hit.path == "concepts/Concept_Handwritten_Agent.md" for hit in context_item.hits)
    assert "wiki/concepts/Concept_Handwritten_Agent.md" in {entry.path for entry in snapshot.entries}


def test_strong_context_create_is_finalized_to_needs_human_decision(tmp_path: Path) -> None:
    vault, _ = make_vault(tmp_path)
    existing = vault / "wiki" / "concepts" / "Concept_Knowledge digestion.md"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text(
        "---\n"
        "llmwiki_type: concept\n"
        "title: Knowledge digestion\n"
        "aliases: []\n"
        "summary: Existing summary.\n"
        "updated: 2026-01-02\n"
        "---\n\n"
        "# Knowledge digestion\n",
        encoding="utf-8",
    )
    resolution = CandidateResolutionArtifact(
        items=[
            resolution_item(
                "CAND001",
                page_type="concept",
                target_path="concepts/Concept_New Knowledge digestion.md",
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
    plan = pipeline_module.finalize_wiki_merge_plan(
        pipeline_module.WikiMergePlanArtifact(
            log_date="",
            context_snapshot_ref="",
            items=[
                pipeline_module.WikiMergePlanItem(
                    page_plan_id="PP-CAND001",
                    source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
                    action="create",
                    canonical_target_path="concepts/Concept_New Knowledge digestion.md",
                    display_title="Knowledge digestion",
                    page_type="concept",
                    new_understanding="New material.",
                    section_plans={"summary": "Summary"},
                    reason="Model tried to create.",
                )
            ],
        ),
        resolution,
        snapshot,
        "wiki_context_snapshot/wiki_context_snapshot.json",
    )

    assert plan.items[0].action == "needs_human_decision"
    assert plan.items[0].apply_eligibility == "blocked"
    assert plan.items[0].strongest_overlap.strength == "strong"


def merge_review_create_item(
    page_plan_id: str,
    *,
    strength: str = "none",
    why_not_update: str = "",
) -> pipeline_module.WikiMergePlanItem:
    return pipeline_module.WikiMergePlanItem(
        page_plan_id=page_plan_id,
        source_basis=SourceBasis(source_candidate_ids=[page_plan_id]),
        action="create",
        model_action="create",
        canonical_target_path=f"concepts/Concept_{page_plan_id}.md",
        display_title=f"Concept {page_plan_id}",
        page_type="concept",
        inspected_context_paths=[f"concepts/Concept_Old_{page_plan_id}.md"] if strength != "none" else [],
        strongest_overlap=pipeline_module.ContextOverlapSignal(
            strength=strength,
            match_basis="embedding" if strength != "none" else "",
            path=f"concepts/Concept_Old_{page_plan_id}.md" if strength != "none" else "",
            score={"none": 0.0, "weak": 0.4, "medium": 0.67, "strong": 0.82}[strength],
            reason="test overlap",
        ),
        why_not_update=why_not_update,
        why_create_or_update="测试 create。",
        new_understanding="新增知识。",
        section_plans={"摘要": "写摘要。"},
        reason="测试 create。",
    )


def merge_review_plan(*items: pipeline_module.WikiMergePlanItem) -> pipeline_module.WikiMergePlanArtifact:
    return pipeline_module.WikiMergePlanArtifact(log_date="2026-06-07", context_snapshot_ref="", items=list(items))


def test_all_create_medium_with_concrete_reason_does_not_force_review() -> None:
    reason = (
        "新页范围是通用 AI 代理记忆；旧页范围是 Mem0 多级记忆实现。"
        "本轮来源增量来自 Redis 播客，直接更新旧页会让 Mem0 页面失焦，"
        "只做 Related 不能承载新增例子和价值点。"
    )
    plan = merge_review_plan(merge_review_create_item("PP-1", strength="medium", why_not_update=reason))

    assert merge_plan_refinement_module.merge_plan_all_create_review_reason(plan) == ""


def test_all_create_medium_with_weak_reason_still_waits_for_review() -> None:
    plan = merge_review_plan(merge_review_create_item("PP-1", strength="medium", why_not_update="更适合新建。"))

    reason = merge_plan_refinement_module.merge_plan_all_create_review_reason(plan)

    assert "中等召回风险" in reason
    assert "理由不充分" in reason


def test_all_create_medium_with_generic_marker_reason_still_waits_for_review() -> None:
    plan = merge_review_plan(
        merge_review_create_item(
            "PP-1",
            strength="medium",
            why_not_update="旧页范围不同，来源材料不同，更新旧页不合适，Related 不够。",
        )
    )

    reason = merge_plan_refinement_module.merge_plan_all_create_review_reason(plan)

    assert "中等召回风险" in reason
    assert "理由不充分" in reason


def test_all_create_medium_with_long_generic_reason_still_waits_for_review() -> None:
    plan = merge_review_plan(
        merge_review_create_item(
            "PP-1",
            strength="medium",
            why_not_update=(
                "新页范围和旧页范围不一样，本轮来源材料也不一样，直接更新旧页会让旧页范围变大，"
                "已有页覆盖不了新增内容，只做 Related 不够承载新增结构和价值点，所以应该创建新页。"
            ),
        )
    )

    reason = merge_plan_refinement_module.merge_plan_all_create_review_reason(plan)

    assert "中等召回风险" in reason
    assert "理由不充分" in reason


def test_all_create_medium_generic_old_title_scope_dismissal_waits_for_review() -> None:
    item = merge_review_create_item(
        "PP-agent-memory",
        strength="medium",
        why_not_update=(
            "旧页《持久记忆（Agent Memory）》是 Cloudflare 平台特定页面；"
            "新页是通用 Agent 记忆系统概念，直接更新旧页会失焦，Related 不够承载新增结构。"
        ),
    ).model_copy(
        update={
            "display_title": "Agent 记忆系统",
            "canonical_target_path": "concepts/Concept_Agent 记忆系统.md",
            "strongest_overlap": pipeline_module.ContextOverlapSignal(
                strength="medium",
                match_basis="embedding",
                path="concepts/Concept_持久记忆（Agent Memory）.md",
                score=0.72,
                reason="Top inspected context: 持久记忆（Agent Memory）",
            ),
        }
    )
    plan = merge_review_plan(item)

    reason = merge_plan_refinement_module.merge_plan_all_create_review_reason(plan)

    assert "中等召回风险" in reason
    assert "旧页标题像通用概念页" in reason


def test_all_create_medium_ai_agent_neicun_alias_waits_for_review() -> None:
    item = merge_review_create_item(
        "PP-agent-memory",
        strength="medium",
        why_not_update=(
            "现有页面「持久记忆（Agent Memory）」聚焦于 Cloudflare 的持久记忆服务，"
            "而新页面需要覆盖更广泛的 AI Agent 内存概念，包括短期对话、摘要和长期事实的完整分类及设计权衡。"
            "更新现有页面会导致其失去焦点，而仅通过相关链接不足以表达独立的概念体系。"
        ),
    ).model_copy(
        update={
            "display_title": "AI Agent 内存",
            "canonical_target_path": "concepts/Concept_AI Agent 内存.md",
            "strongest_overlap": pipeline_module.ContextOverlapSignal(
                strength="medium",
                match_basis="embedding",
                path="concepts/Concept_持久记忆（Agent Memory）.md",
                score=0.7645,
                reason="Top inspected context: 持久记忆（Agent Memory）",
            ),
        }
    )
    plan = merge_review_plan(item)

    reason = merge_plan_refinement_module.merge_plan_all_create_review_reason(plan)

    assert "中等召回风险" in reason
    assert "旧页标题像通用概念页" in reason


def test_all_create_medium_existing_knowledge_page_marker_waits_for_review() -> None:
    item = merge_review_create_item(
        "PP-agent-memory",
        strength="medium",
        why_not_update=(
            "现有知识页「持久记忆（Agent Memory）」聚焦于 Cloudflare 的持久记忆服务，"
            "而新页面需要覆盖更广泛的 AI Agent 内存概念。"
            "来源增量来自 Redis 播客，直接更新旧页会失焦，只做 Related 不能承载新增结构。"
        ),
    ).model_copy(
        update={
            "display_title": "AI Agent 内存",
            "canonical_target_path": "concepts/Concept_AI Agent 内存.md",
            "strongest_overlap": pipeline_module.ContextOverlapSignal(
                strength="medium",
                match_basis="embedding",
                path="concepts/Concept_持久记忆（Agent Memory）.md",
                score=0.7645,
                reason="Top inspected context: 持久记忆（Agent Memory）",
            ),
        }
    )
    plan = merge_review_plan(item)

    reason = merge_plan_refinement_module.merge_plan_all_create_review_reason(plan)

    assert "中等召回风险" in reason
    assert "旧页标题像通用概念页" in reason


def test_all_create_medium_old_title_comma_scope_predicate_waits_for_review() -> None:
    item = merge_review_create_item(
        "PP-agent-memory",
        strength="medium",
        why_not_update=(
            "现有页面「持久记忆（Agent Memory）」，聚焦于 Cloudflare 的持久记忆服务，"
            "而新页面需要覆盖更广泛的 AI Agent 内存概念。"
            "来源增量来自 Redis 播客，直接更新旧页会失焦，只做 Related 不能承载新增结构。"
        ),
    ).model_copy(
        update={
            "display_title": "AI Agent 内存",
            "canonical_target_path": "concepts/Concept_AI Agent 内存.md",
            "strongest_overlap": pipeline_module.ContextOverlapSignal(
                strength="medium",
                match_basis="embedding",
                path="concepts/Concept_持久记忆（Agent Memory）.md",
                score=0.7645,
                reason="Top inspected context: 持久记忆（Agent Memory）",
            ),
        }
    )
    plan = merge_review_plan(item)

    reason = merge_plan_refinement_module.merge_plan_all_create_review_reason(plan)

    assert "中等召回风险" in reason
    assert "旧页标题像通用概念页" in reason


def test_all_create_medium_added_page_marker_waits_for_review() -> None:
    item = merge_review_create_item(
        "PP-agent-memory",
        strength="medium",
        why_not_update=(
            "新增页面是通用 AI Agent 内存概念；来源增量来自 Redis 播客。"
            "直接更新旧页会失焦，只做 Related 不能承载新增结构。"
        ),
    ).model_copy(
        update={
            "display_title": "AI Agent 内存",
            "canonical_target_path": "concepts/Concept_AI Agent 内存.md",
            "strongest_overlap": pipeline_module.ContextOverlapSignal(
                strength="medium",
                match_basis="embedding",
                path="concepts/Concept_持久记忆（Agent Memory）.md",
                score=0.7645,
                reason="Top inspected context: 持久记忆（Agent Memory）",
            ),
        }
    )
    plan = merge_review_plan(item)

    reason = merge_plan_refinement_module.merge_plan_all_create_review_reason(plan)

    assert "中等召回风险" in reason
    assert "旧页标题像通用概念页" in reason


def test_all_create_medium_added_knowledge_page_marker_waits_for_review() -> None:
    item = merge_review_create_item(
        "PP-agent-memory",
        strength="medium",
        why_not_update=(
            "新增知识页是通用 AI Agent 内存概念；来源增量来自 Redis 播客。"
            "直接更新旧页会失焦，只做 Related 不能承载新增结构。"
        ),
    ).model_copy(
        update={
            "display_title": "AI Agent 内存",
            "canonical_target_path": "concepts/Concept_AI Agent 内存.md",
            "strongest_overlap": pipeline_module.ContextOverlapSignal(
                strength="medium",
                match_basis="embedding",
                path="concepts/Concept_持久记忆（Agent Memory）.md",
                score=0.7645,
                reason="Top inspected context: 持久记忆（Agent Memory）",
            ),
        }
    )
    plan = merge_review_plan(item)

    reason = merge_plan_refinement_module.merge_plan_all_create_review_reason(plan)

    assert "中等召回风险" in reason
    assert "旧页标题像通用概念页" in reason


def test_all_create_medium_pure_english_memory_old_title_waits_for_review() -> None:
    item = merge_review_create_item(
        "PP-agent-memory",
        strength="medium",
        why_not_update=(
            "The old page is Cloudflare platform-specific; the new page is a generic Agent Memory System concept. "
            "Updating the old page would blur scope, and Related is not enough for the new structure."
        ),
    ).model_copy(
        update={
            "display_title": "Agent Memory System",
            "canonical_target_path": "concepts/Concept_Agent Memory System.md",
            "strongest_overlap": pipeline_module.ContextOverlapSignal(
                strength="medium",
                match_basis="embedding",
                path="concepts/Concept_Persistent Memory.md",
                score=0.72,
                reason="Top inspected context: Persistent Memory",
            ),
        }
    )
    plan = merge_review_plan(item)

    reason = merge_plan_refinement_module.merge_plan_all_create_review_reason(plan)

    assert "中等召回风险" in reason
    assert "旧页标题像通用概念页" in reason


def test_all_create_medium_rapid_memory_is_not_api_specific() -> None:
    item = merge_review_create_item(
        "PP-rapid-memory",
        strength="medium",
        why_not_update=(
            "The old page is platform-specific; the new page is a generic Rapid Memory concept. "
            "Updating the old page would blur scope, and Related is not enough for the new structure."
        ),
    ).model_copy(
        update={
            "display_title": "Rapid Memory",
            "canonical_target_path": "concepts/Concept_Rapid Memory.md",
            "strongest_overlap": pipeline_module.ContextOverlapSignal(
                strength="medium",
                match_basis="embedding",
                path="concepts/Concept_Rapid Memory.md",
                score=0.72,
                reason="Top inspected context: Rapid Memory",
            ),
        }
    )
    plan = merge_review_plan(item)

    reason = merge_plan_refinement_module.merge_plan_all_create_review_reason(plan)

    assert "中等召回风险" in reason
    assert "旧页标题像通用概念页" in reason


def test_all_create_medium_old_specific_negation_with_generic_old_scope_does_not_force_review() -> None:
    item = merge_review_create_item(
        "PP-agent-memory",
        strength="medium",
        why_not_update=(
            "旧页不是平台特定页面，旧页本身像通用概念页；"
            "新页范围是 Agent 记忆系统的写入策略，直接更新旧页会混淆持久记忆总览与写入策略，"
            "只做 Related 不能承载新增步骤。"
        ),
    ).model_copy(
        update={
            "display_title": "Agent 记忆系统写入策略",
            "canonical_target_path": "concepts/Concept_Agent 记忆系统写入策略.md",
            "strongest_overlap": pipeline_module.ContextOverlapSignal(
                strength="medium",
                match_basis="embedding",
                path="concepts/Concept_持久记忆（Agent Memory）.md",
                score=0.72,
                reason="Top inspected context: 持久记忆（Agent Memory）",
            ),
        }
    )
    plan = merge_review_plan(item)

    assert merge_plan_refinement_module.merge_plan_all_create_review_reason(plan) == ""


def test_all_create_medium_old_title_no_product_token_does_not_force_review() -> None:
    item = merge_review_create_item(
        "PP-agent-memory",
        strength="medium",
        why_not_update=(
            "旧页标题没有 Cloudflare/Mem0 平台词，本身像通用概念页；"
            "新页范围是 Agent 记忆系统的淘汰策略，直接更新旧页会混淆总览与策略页，"
            "只做 Related 不能承载新增策略步骤。"
        ),
    ).model_copy(
        update={
            "display_title": "Agent 记忆系统淘汰策略",
            "canonical_target_path": "concepts/Concept_Agent 记忆系统淘汰策略.md",
            "strongest_overlap": pipeline_module.ContextOverlapSignal(
                strength="medium",
                match_basis="embedding",
                path="concepts/Concept_持久记忆（Agent Memory）.md",
                score=0.72,
                reason="Top inspected context: 持久记忆（Agent Memory）",
            ),
        }
    )
    plan = merge_review_plan(item)

    assert merge_plan_refinement_module.merge_plan_all_create_review_reason(plan) == ""


def test_all_create_medium_bare_negation_still_reviews_when_new_is_generic() -> None:
    item = merge_review_create_item(
        "PP-agent-memory",
        strength="medium",
        why_not_update=(
            "旧页不是平台特定页面；"
            "新页是通用 Agent 记忆系统概念，直接更新旧页会失焦，Related 不够承载新增结构。"
        ),
    ).model_copy(
        update={
            "display_title": "Agent 记忆系统",
            "canonical_target_path": "concepts/Concept_Agent 记忆系统.md",
            "strongest_overlap": pipeline_module.ContextOverlapSignal(
                strength="medium",
                match_basis="embedding",
                path="concepts/Concept_持久记忆（Agent Memory）.md",
                score=0.72,
                reason="Top inspected context: 持久记忆（Agent Memory）",
            ),
        }
    )
    plan = merge_review_plan(item)

    assert "中等召回风险" in merge_plan_refinement_module.merge_plan_all_create_review_reason(plan)


def test_all_create_medium_adversarial_negation_still_waits_for_review() -> None:
    item = merge_review_create_item(
        "PP-agent-memory",
        strength="medium",
        why_not_update=(
            "旧页不是普通平台页，而是 Cloudflare 官方实现；"
            "新页是通用 Agent 记忆系统概念，直接更新旧页会失焦，Related 不够承载新增结构。"
        ),
    ).model_copy(
        update={
            "display_title": "Agent 记忆系统",
            "canonical_target_path": "concepts/Concept_Agent 记忆系统.md",
            "strongest_overlap": pipeline_module.ContextOverlapSignal(
                strength="medium",
                match_basis="embedding",
                path="concepts/Concept_持久记忆（Agent Memory）.md",
                score=0.72,
                reason="Top inspected context: 持久记忆（Agent Memory）",
            ),
        }
    )
    plan = merge_review_plan(item)

    assert "中等召回风险" in merge_plan_refinement_module.merge_plan_all_create_review_reason(plan)


def test_all_create_medium_adversarial_negation_without_new_generic_still_waits_for_review() -> None:
    item = merge_review_create_item(
        "PP-agent-memory",
        strength="medium",
        why_not_update=(
            "旧页不是普通平台页，而是 Cloudflare 官方实现；"
            "本轮来源增量来自 Redis 播客，直接更新旧页会失焦，Related 不够承载新增结构。"
        ),
    ).model_copy(
        update={
            "display_title": "Agent 记忆系统",
            "canonical_target_path": "concepts/Concept_Agent 记忆系统.md",
            "strongest_overlap": pipeline_module.ContextOverlapSignal(
                strength="medium",
                match_basis="embedding",
                path="concepts/Concept_持久记忆（Agent Memory）.md",
                score=0.72,
                reason="Top inspected context: 持久记忆（Agent Memory）",
            ),
        }
    )
    plan = merge_review_plan(item)

    assert "中等召回风险" in merge_plan_refinement_module.merge_plan_all_create_review_reason(plan)


def test_all_create_medium_english_not_merely_specific_still_waits_for_review() -> None:
    item = merge_review_create_item(
        "PP-agent-memory",
        strength="medium",
        why_not_update=(
            "The old page is not merely platform-specific; it is a Cloudflare implementation. "
            "The Redis source adds a separate scope, and Related is not enough for the new structure."
        ),
    ).model_copy(
        update={
            "display_title": "Agent Memory System",
            "canonical_target_path": "concepts/Concept_Agent Memory System.md",
            "strongest_overlap": pipeline_module.ContextOverlapSignal(
                strength="medium",
                match_basis="embedding",
                path="concepts/Concept_Persistent Memory.md",
                score=0.72,
                reason="Top inspected context: Persistent Memory",
            ),
        }
    )
    plan = merge_review_plan(item)

    assert "中等召回风险" in merge_plan_refinement_module.merge_plan_all_create_review_reason(plan)


def test_all_create_medium_old_not_generic_new_generic_still_waits_for_review() -> None:
    item = merge_review_create_item(
        "PP-agent-memory",
        strength="medium",
        why_not_update=(
            "旧页不是通用概念页；"
            "新页是通用 Agent 记忆系统概念，直接更新旧页会失焦，Related 不够承载新增结构。"
        ),
    ).model_copy(
        update={
            "display_title": "Agent 记忆系统",
            "canonical_target_path": "concepts/Concept_Agent 记忆系统.md",
            "strongest_overlap": pipeline_module.ContextOverlapSignal(
                strength="medium",
                match_basis="embedding",
                path="concepts/Concept_持久记忆（Agent Memory）.md",
                score=0.72,
                reason="Top inspected context: 持久记忆（Agent Memory）",
            ),
        }
    )
    plan = merge_review_plan(item)

    assert "中等召回风险" in merge_plan_refinement_module.merge_plan_all_create_review_reason(plan)


def test_all_create_medium_english_old_not_generic_new_generic_still_waits_for_review() -> None:
    item = merge_review_create_item(
        "PP-agent-memory",
        strength="medium",
        why_not_update=(
            "The old page is not generic; the new page is a generic Agent Memory System concept. "
            "Updating the old page would blur scope, and Related is not enough for the new structure."
        ),
    ).model_copy(
        update={
            "display_title": "Agent Memory System",
            "canonical_target_path": "concepts/Concept_Agent Memory System.md",
            "strongest_overlap": pipeline_module.ContextOverlapSignal(
                strength="medium",
                match_basis="embedding",
                path="concepts/Concept_Persistent Memory.md",
                score=0.72,
                reason="Top inspected context: Persistent Memory",
            ),
        }
    )
    plan = merge_review_plan(item)

    assert "中等召回风险" in merge_plan_refinement_module.merge_plan_all_create_review_reason(plan)


def test_all_create_medium_product_specific_old_title_can_auto_pass() -> None:
    item = merge_review_create_item(
        "PP-agent-memory",
        strength="medium",
        why_not_update=(
            "新页范围是通用 AI Agent 记忆系统；旧页范围是 Mem0 多级记忆实现。"
            "本轮来源增量来自 Redis 播客，直接更新旧页会让 Mem0 页面失焦，"
            "只做 Related 不能承载新增例子和价值点。"
        ),
    ).model_copy(
        update={
            "display_title": "AI Agent 记忆系统",
            "canonical_target_path": "concepts/Concept_AI Agent 记忆系统.md",
            "strongest_overlap": pipeline_module.ContextOverlapSignal(
                strength="medium",
                match_basis="embedding",
                path="concepts/Concept_Mem0 多级记忆实现.md",
                score=0.70,
                reason="Top inspected context: Mem0 多级记忆实现",
            ),
        }
    )
    plan = merge_review_plan(item)

    assert merge_plan_refinement_module.merge_plan_all_create_review_reason(plan) == ""


def test_all_create_medium_source_specific_old_title_with_neicun_can_auto_pass() -> None:
    item = merge_review_create_item(
        "PP-redis-memory-architecture",
        strength="medium",
        why_not_update=(
            "新页范围是基于 Redis 的 Agent 内存架构设计；旧页范围是 Cloudflare Agent Memory 产品实现。"
            "本轮来源增量来自 Redis 播客中的向量搜索和语义缓存架构，"
            "直接更新旧页会让 Cloudflare 页面失焦，只做 Related 不能承载 Redis 集成步骤。"
        ),
    ).model_copy(
        update={
            "display_title": "基于 Redis 的 Agent 内存架构",
            "canonical_target_path": "designs/Design_基于 Redis 的 Agent 内存架构.md",
            "page_type": "design",
            "strongest_overlap": pipeline_module.ContextOverlapSignal(
                strength="medium",
                match_basis="embedding",
                path="entities/Entity_Cloudflare Agent Memory.md",
                score=0.6379,
                reason="Top inspected context: Cloudflare Agent Memory",
            ),
        }
    )
    plan = merge_review_plan(item)

    assert merge_plan_refinement_module.merge_plan_all_create_review_reason(plan) == ""


def test_all_create_medium_runtime_neicun_does_not_become_agent_memory_review() -> None:
    item = merge_review_create_item(
        "PP-redis-runtime-memory",
        strength="medium",
        why_not_update=(
            "新页范围是 Redis 运行时内存配置和 maxmemory 策略；旧页范围是 Agent 持久记忆概念。"
            "本轮来源增量来自 Redis 配置文档，直接更新旧页会混淆运行时资源配置与 Agent 记忆能力，"
            "只做 Related 不能承载配置步骤。"
        ),
    ).model_copy(
        update={
            "display_title": "Redis 内存配置",
            "canonical_target_path": "concepts/Concept_Redis 内存配置.md",
            "strongest_overlap": pipeline_module.ContextOverlapSignal(
                strength="medium",
                match_basis="embedding",
                path="concepts/Concept_持久记忆（Agent Memory）.md",
                score=0.69,
                reason="Top inspected context: 持久记忆（Agent Memory）",
            ),
        }
    )
    plan = merge_review_plan(item)

    assert merge_plan_refinement_module.merge_plan_all_create_review_reason(plan) == ""


def test_all_create_medium_runtime_neicun_generic_wording_still_auto_passes() -> None:
    item = merge_review_create_item(
        "PP-redis-runtime-memory",
        strength="medium",
        why_not_update=(
            "新页面是通用 Redis 内存配置概念；来源增量来自 Redis 配置文档。"
            "直接更新旧页会混淆运行时资源配置与 Agent 记忆能力，只做 Related 不能承载配置步骤。"
        ),
    ).model_copy(
        update={
            "display_title": "Redis 内存配置",
            "canonical_target_path": "concepts/Concept_Redis 内存配置.md",
            "strongest_overlap": pipeline_module.ContextOverlapSignal(
                strength="medium",
                match_basis="embedding",
                path="concepts/Concept_持久记忆（Agent Memory）.md",
                score=0.69,
                reason="Top inspected context: 持久记忆（Agent Memory）",
            ),
        }
    )
    plan = merge_review_plan(item)

    assert merge_plan_refinement_module.merge_plan_all_create_review_reason(plan) == ""


def test_all_create_medium_later_source_delta_does_not_make_old_page_specific() -> None:
    item = merge_review_create_item(
        "PP-agent-memory-policy",
        strength="medium",
        why_not_update=(
            "旧页范围是 Agent 持久记忆总览，本轮来源增量来自 Redis 播客；"
            "新页范围是 AI Agent 内存写入策略，直接更新旧页会混淆总览与策略，"
            "只做 Related 不能承载写入策略步骤。"
        ),
    ).model_copy(
        update={
            "display_title": "AI Agent 内存写入策略",
            "canonical_target_path": "concepts/Concept_AI Agent 内存写入策略.md",
            "strongest_overlap": pipeline_module.ContextOverlapSignal(
                strength="medium",
                match_basis="embedding",
                path="concepts/Concept_持久记忆（Agent Memory）.md",
                score=0.70,
                reason="Top inspected context: 持久记忆（Agent Memory）",
            ),
        }
    )
    plan = merge_review_plan(item)

    assert merge_plan_refinement_module.merge_plan_all_create_review_reason(plan) == ""


def test_all_create_medium_agent_only_title_overlap_does_not_force_review() -> None:
    item = merge_review_create_item(
        "PP-agent-routing",
        strength="medium",
        why_not_update=(
            "新页范围是通用 Agent 路由设计；旧页范围是 Agent 监控指标。"
            "本轮来源增量来自调度材料，直接更新旧页会混淆指标与路由策略，"
            "只做 Related 不能承载新增设计步骤。"
        ),
    ).model_copy(
        update={
            "display_title": "Agent 路由设计",
            "canonical_target_path": "designs/Design_Agent 路由设计.md",
            "strongest_overlap": pipeline_module.ContextOverlapSignal(
                strength="medium",
                match_basis="embedding",
                path="concepts/Concept_Agent 监控指标.md",
                score=0.66,
                reason="Top inspected context: Agent 监控指标",
            ),
        }
    )
    plan = merge_review_plan(item)

    assert merge_plan_refinement_module.merge_plan_all_create_review_reason(plan) == ""


def test_all_create_medium_with_locally_synthesized_reason_still_waits_for_review() -> None:
    item = merge_review_create_item(
        "PP-1",
        strength="medium",
        why_not_update=(
            "本地补充：scope_delta：新页范围是通用概念；旧页范围是产品实现。"
            "source_delta：本轮来源增量不同。why_update_not_enough：直接更新会失焦。"
            "why_related_link_not_enough：只做 Related 不能承载新增结构。"
        ),
    ).model_copy(
        update={
            "finalization_reason": (
                f"medium overlap create 缺少 why_not_update，已{merge_plan_refinement_module.LOCAL_MEDIUM_CREATE_REASON_MARKER}。"
            )
        }
    )
    plan = merge_review_plan(item)

    reason = merge_plan_refinement_module.merge_plan_all_create_review_reason(plan)

    assert "中等召回风险" in reason
    assert "仅由本地补充" in reason


def test_all_create_strong_overlap_still_waits_for_review() -> None:
    reason = (
        "新页范围和旧页有差异，来源也不同；更新旧页会扩大旧页范围，"
        "只做 Related 不能承载新增结构。"
    )
    plan = merge_review_plan(merge_review_create_item("PP-1", strength="strong", why_not_update=reason))

    assert "强召回风险" in merge_plan_refinement_module.merge_plan_all_create_review_reason(plan)


def test_all_create_page_count_cap_waits_for_review_at_thirteen() -> None:
    allowed = [
        merge_review_create_item(f"PP-{index}", strength="none")
        for index in range(merge_plan_refinement_module.MAX_AUTO_APPROVED_ALL_CREATE_ITEMS)
    ]
    blocked = [
        merge_review_create_item(f"PP-{index}", strength="none")
        for index in range(merge_plan_refinement_module.MAX_AUTO_APPROVED_ALL_CREATE_ITEMS + 1)
    ]

    assert merge_plan_refinement_module.merge_plan_all_create_review_reason(merge_review_plan(*allowed)) == ""
    assert "超过自动通过上限" in merge_plan_refinement_module.merge_plan_all_create_review_reason(merge_review_plan(*blocked))


def test_all_create_page_count_cap_can_use_vault_budget() -> None:
    items = [merge_review_create_item(f"PP-{index}", strength="none") for index in range(5)]

    assert (
        "超过自动通过上限 4"
        in merge_plan_refinement_module.merge_plan_all_create_review_reason(merge_review_plan(*items), max_auto_create_items=4)
    )


def test_medium_context_create_without_why_not_update_stops_for_review(tmp_path: Path) -> None:
    vault, _ = make_vault(tmp_path)
    existing = vault / "wiki" / "concepts" / "Concept_AI_PM_Career.md"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text(
        "---\n"
        "llmwiki_type: concept\n"
        "title: AI PM Career Skills\n"
        "aliases: []\n"
        "summary: Existing AI PM career skill summary.\n"
        "updated: 2026-01-02\n"
        "---\n\n"
        "# AI PM Career Skills\n",
        encoding="utf-8",
    )
    resolution = CandidateResolutionArtifact(
        items=[
            resolution_item(
                "CAND001",
                page_type="concept",
                target_path="concepts/Concept_AI PM Interview.md",
                display_title="AI PM",
            )
        ]
    )
    snapshot = build_wiki_context_snapshot(
        vault,
        resolution,
        log_date="2026-06-03",
        source_target_path="sources/Source_Test.md",
    )
    plan = pipeline_module.finalize_wiki_merge_plan(
        pipeline_module.WikiMergePlanArtifact(
            log_date="",
            context_snapshot_ref="",
            items=[
                pipeline_module.WikiMergePlanItem(
                    page_plan_id="PP-CAND001",
                    source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
                    action="create",
                    canonical_target_path="concepts/Concept_AI PM Interview.md",
                    display_title="AI PM",
                    page_type="concept",
                    new_understanding="New AI PM material.",
                    section_plans={"summary": "Summary"},
                    reason="Model tried to create without explaining why not update.",
                )
            ],
        ),
        resolution,
        snapshot,
        "wiki_context_snapshot/wiki_context_snapshot.json",
    )

    assert plan.items[0].strongest_overlap.strength == "medium"
    assert plan.items[0].action == "needs_human_decision"
    assert plan.items[0].apply_eligibility == "blocked"
    assert "理由不充分" in plan.items[0].blocked_reason
    report = merge_reporting_module.render_merge_decision_report(plan, snapshot)
    assert "## Create/Update 风险摘要" in report
    assert "Concept_AI_PM_Career.md" in report
    assert "未提供" in report
    assert "理由不充分" in report


def test_wiki_merge_planning_locally_fills_missing_medium_create_reason(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    existing = vault / "wiki" / "concepts" / "Concept_知识编译.md"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text(
        "---\n"
        "llmwiki_type: concept\n"
        "title: 知识编译\n"
        "aliases: []\n"
        "summary: 已有知识编译概念。\n"
        "updated: 2026-01-02\n"
        "---\n\n"
        "# 知识编译\n",
        encoding="utf-8",
    )
    fixture_dir = tmp_path / "repair-fixture"
    fixture_dir.mkdir()
    for name in ["raw_prepare.json", "source_digest.json", "candidate_resolution.json", "draft_rendering.json"]:
        write_json(fixture_dir / name, read_json(FIXTURE_ROOT / "mock" / name))
    initial_plan = read_json(FIXTURE_ROOT / "mock" / "wiki_merge_planning.json")
    initial_plan["items"][0].pop("why_not_update", None)
    write_json(fixture_dir / "wiki_merge_planning.json", initial_plan)

    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=fixture_dir, slug="repair-why")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    plan = read_json(run_dir / "wiki_merge_planning" / "wiki_merge_plan.json")

    repair_report = read_json(run_dir / "wiki_merge_planning" / "structured_repair_report.json")
    assert repair_report["repair_count"] == 0
    assert len(list((run_dir / "wiki_merge_planning" / "provider_results").glob("attempt-*.json"))) == 1
    assert plan["items"][0]["why_not_update"].startswith("本地补充：scope_delta")
    assert "source_delta" in plan["items"][0]["why_not_update"]
    assert "why_update_not_enough" in plan["items"][0]["why_not_update"]
    assert "why_related_link_not_enough" in plan["items"][0]["why_not_update"]
    assert "本地补充结构化 create/update 对比理由" in plan["items"][0]["finalization_reason"]
    saved_manifest = read_manifest(RunStore(vault).manifest_path(manifest.operation_id))
    assert saved_manifest.steps[8].status == StepStatus.completed
    assert saved_manifest.steps[9].status == StepStatus.awaiting_review
    assert manifest.status == OperationStatus.awaiting_review


def test_mixed_plan_medium_generic_old_title_create_stops_for_review() -> None:
    resolution = CandidateResolutionArtifact(
        items=[
            resolution_item(
                "CAND001",
                page_type="concept",
                target_path="concepts/Concept_Agent 记忆系统.md",
                display_title="Agent 记忆系统",
            ),
            resolution_item(
                "CAND002",
                page_type="concept",
                target_path="concepts/Concept_已有更新.md",
                display_title="已有更新",
            ),
        ]
    )
    snapshot = pipeline_module.WikiContextSnapshot(
        log_date="2026-06-07",
        source_target_path="sources/Source_Test.md",
        candidate_contexts=CandidateContextsArtifact(
            retrieval_backend="exact",
            items=[
                CandidateContextItem(
                    page_plan_id="PP-CAND001",
                    query="Agent 记忆系统",
                    hits=[
                        CandidateContextHit(
                            page_plan_id="PP-CAND001",
                            rank=1,
                            path="concepts/Concept_持久记忆（Agent Memory）.md",
                            display_title="持久记忆（Agent Memory）",
                            score=0.72,
                            score_bucket=72,
                            strength="medium",
                            match_basis="embedding",
                            sort_explanation=(
                                "bucket=72; strength_rank=2; basis_rank=1; type=same; dir=same; "
                                "title_distance=2; path=concepts/Concept_持久记忆（Agent Memory）.md"
                            ),
                            page_sha256="old-memory",
                        )
                    ],
                ),
                CandidateContextItem(page_plan_id="PP-CAND002", query="已有更新", hits=[]),
            ],
        ),
        entries=[
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_Agent 记忆系统.md",
                expected_state="missing",
                content="",
            ),
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_持久记忆（Agent Memory）.md",
                expected_state="present",
                preimage_sha256="old-memory",
                content="# 持久记忆（Agent Memory）\n\n旧页已有通用 Agent memory 概念。\n",
            ),
            pipeline_module.WikiContextEntry(
                path="wiki/concepts/Concept_已有更新.md",
                expected_state="present",
                preimage_sha256="old-update",
                content="# 已有更新\n\n旧页。\n",
            ),
        ],
    )
    plan = pipeline_module.WikiMergePlanArtifact(
        log_date="2026-06-07",
        context_snapshot_ref="wiki_context_snapshot/wiki_context_snapshot.json",
        items=[
            pipeline_module.WikiMergePlanItem(
                page_plan_id="PP-CAND001",
                source_basis=SourceBasis(source_candidate_ids=["CAND001"]),
                action="create",
                canonical_target_path="concepts/Concept_Agent 记忆系统.md",
                display_title="Agent 记忆系统",
                page_type="concept",
                why_not_update=(
                    "旧页《持久记忆（Agent Memory）》是 Cloudflare 平台特定页面；"
                    "新页是通用 Agent 记忆系统概念，直接更新旧页会失焦，Related 不够承载新增结构。"
                ),
                new_understanding="补充 Agent 记忆系统的通用视角。",
                section_plans={"summary": "摘要。"},
                reason="模型尝试新建。",
            ),
            pipeline_module.WikiMergePlanItem(
                page_plan_id="PP-CAND002",
                source_basis=SourceBasis(source_candidate_ids=["CAND002"]),
                action="update",
                matched_page="concepts/Concept_已有更新.md",
                canonical_target_path="concepts/Concept_已有更新.md",
                display_title="已有更新",
                page_type="concept",
                why_create_or_update="补充已有页。",
                new_understanding="补充已有页。",
                section_plans={"summary": "摘要。"},
                reason="模型更新已有页。",
            ),
        ],
    )

    finalized = pipeline_module.finalize_wiki_merge_plan(
        plan,
        resolution,
        snapshot,
        "wiki_context_snapshot/wiki_context_snapshot.json",
    )

    create_item = finalized.items[0]
    assert create_item.action == "needs_human_decision"
    assert create_item.apply_eligibility == "blocked"
    assert "旧页标题《持久记忆（Agent Memory）》像通用概念页" in create_item.blocked_reason
    assert finalized.items[1].action == "update"
    report = merge_reporting_module.render_merge_decision_report(finalized, snapshot)
    assert "旧页标题《持久记忆（Agent Memory）》像通用概念页" in report


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
    rows = draft_outputs_module.build_index_rows(profile, plan, DraftRenderingArtifact(pages=[]), snapshot)

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
        "# Old Concept\n\n"
        "## 矛盾与未决问题\n\n"
        "- 旧概念是否还适用于新的 Agent 工作流？\n",
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

    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=FIXTURE_ROOT / "mock", slug="merge-system")
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
    assert "旧概念是否还适用于新的 Agent 工作流？" in index_text
    assert "暂无未决问题记录" not in index_text
    log_text = (vault / "wiki" / "log.md").read_text(encoding="utf-8")
    assert f"| [[logs/{today}]] | 2 | `raw/old.md`, `raw/raw_project_note.md` |" in log_text
    daily_text = daily.read_text(encoding="utf-8")
    assert "| `OLD` | `raw/old.md` | 1 | 0 | 0 | 0 |" in daily_text
    assert f"| `{manifest.operation_id}` | `raw/raw_project_note.md` | 2 | 0 | 0 | 0 |" in daily_text


def test_wiki_context_drift_blocks_rerender_from_stale_plan(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=FIXTURE_ROOT / "mock", slug="context")
    (vault / "wiki" / "index.md").write_text("changed after planning\n", encoding="utf-8")

    with pytest.raises(PipelineError, match="wiki context changed after planning"):
        resume_ingest(vault=vault, operation_id=manifest.operation_id, from_step="draft_rendering")


def test_multiple_drafts_apply_requires_latest_wiki_context(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    raw_b = vault / "raw" / "second_project_note.md"
    raw_b.write_text(raw.read_text(encoding="utf-8") + "\nSecond raw variant.\n", encoding="utf-8")
    fixture_b = make_variant_fixture(tmp_path, "raw/second_project_note.md", "Second")
    first = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=FIXTURE_ROOT / "mock", slug="first")
    second = run_simplified_ingest(vault=vault, raw_file=raw_b, mock_fixture_dir=fixture_b, slug="second")

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
    assert resumed.status == OperationStatus.awaiting_review
    assert read_manifest(RunStore(vault).manifest_path(second.operation_id)).steps[9].status == StepStatus.awaiting_review
    approve_review(vault, second.operation_id, "merge_plan_review")
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
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=FIXTURE_ROOT / "mock", slug=f"drift-{mutation}")
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
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=FIXTURE_ROOT / "mock", slug="pinned-date")
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
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=FIXTURE_ROOT / "mock", slug="bad-path")
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


def test_apply_rejects_draft_missing_grounding_sidecar(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=FIXTURE_ROOT / "mock", slug="missing-sidecar")
    run_dir = RunStore(vault).run_dir(manifest.operation_id)
    missing = run_dir / "draft_rendering" / "draft_grounding_review.json"
    missing.unlink()
    manifest_path = RunStore(vault).manifest_path(manifest.operation_id)
    manifest_data = read_json(manifest_path)
    for step in manifest_data["steps"]:
        if step["name"] != "draft_rendering":
            continue
        step["outputs"] = [ref for ref in step["outputs"] if ref["relative_path"] != "draft_rendering/draft_grounding_review.json"]
        for attempt in step["attempts"]:
            attempt["outputs"] = [
                ref for ref in attempt["outputs"] if ref["relative_path"] != "draft_rendering/draft_grounding_review.json"
            ]
    write_json(manifest_path, manifest_data)

    with pytest.raises(ApplyError, match="Draft rendering sidecar artifacts are missing"):
        apply_operation(vault, manifest.operation_id)


def test_plain_apply_records_apply_failed_on_write_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=FIXTURE_ROOT / "mock", slug="rollback")
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
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=FIXTURE_ROOT / "mock", slug="receipt-failure")
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


def test_unsupported_manifest_schema_is_rejected_with_clear_error(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=FIXTURE_ROOT / "mock", slug="invalid-schema")
    manifest_path = RunStore(vault).manifest_path(manifest.operation_id)
    data = read_json(manifest_path)
    data["schema_version"] = "operation_manifest.invalid"
    write_json(manifest_path, data)
    with pytest.raises(ValueError, match="operation manifest is not supported by this engine"):
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
def test_manifest_step_topology_must_match_current_engine(tmp_path: Path, mutator) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=FIXTURE_ROOT / "mock", slug="topology")
    manifest_path = RunStore(vault).manifest_path(manifest.operation_id)
    data = read_json(manifest_path)
    steps_by_name = {step["name"]: step for step in data["steps"]}
    mutated_names = mutator([step["name"] for step in data["steps"]])
    data["steps"] = [dict(steps_by_name.get(name, data["steps"][0]), name=name) for name in mutated_names]
    write_json(manifest_path, data)

    with pytest.raises(ValueError, match="operation manifest is not supported by this engine"):
        read_manifest(manifest_path)


@pytest.mark.parametrize("missing_key", ["status", "provider_contexts", "updated_at"])
def test_manifest_v10_requires_persisted_top_level_fields(tmp_path: Path, missing_key: str) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=FIXTURE_ROOT / "mock", slug="missing-field")
    manifest_path = RunStore(vault).manifest_path(manifest.operation_id)
    data = read_json(manifest_path)
    data.pop(missing_key)
    write_json(manifest_path, data)

    with pytest.raises(ValueError, match="operation manifest is not supported by this engine"):
        read_manifest(manifest_path)


def test_manifest_v10_rejects_extra_top_level_fields_with_engine_message(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=FIXTURE_ROOT / "mock", slug="extra-field")
    manifest_path = RunStore(vault).manifest_path(manifest.operation_id)
    data = read_json(manifest_path)
    data["unexpected_status_summary"] = {"unexpected": True}
    write_json(manifest_path, data)

    with pytest.raises(ValueError, match="operation manifest is not supported by this engine"):
        read_manifest(manifest_path)


def test_manifest_reader_rejects_non_object_json_with_engine_message(tmp_path: Path) -> None:
    vault, raw = make_vault(tmp_path)
    manifest = run_simplified_ingest(vault=vault, raw_file=raw, mock_fixture_dir=FIXTURE_ROOT / "mock", slug="bad-root")
    manifest_path = RunStore(vault).manifest_path(manifest.operation_id)
    manifest_path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="operation manifest is not supported by this engine"):
        read_manifest(manifest_path)
