from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from llmwiki_engine.cli import app
from llmwiki_engine.lite import embeddings
from llmwiki_engine.lite import prompts
from llmwiki_engine.lite import related as related_logic
from llmwiki_engine.lite.io import sha256_file, sha256_text
from llmwiki_engine.lite.models import (
    CandidateContext,
    CandidateContextHit,
    CandidateContexts,
    CandidateMergePlan,
    CandidateMergeUnit,
    CandidatePage,
    CandidatePages,
    CompositionItem,
    CompositionPlan,
    FinalPage,
    FinalPages,
    MergeDecision,
    MergePlan,
    OperationManifest,
    RawBinding,
    SourceDigest,
    SourceDigestCandidate,
    SourceRef,
    WikiKnowledgeEntry,
    WikiSnapshot,
)
from llmwiki_engine.lite.pipeline import (
    _assert_candidate_pages_chinese,
    _assert_merge_plan_consumes_candidates,
    _assert_merge_plan_chinese,
    _assert_source_digest_chinese,
    _canonical_final_markdown,
    _normalize_candidate_pages,
    _normalize_final_pages,
    _normalize_merge_plan,
    _page_generation_parallelism,
    _step_candidate_merge,
    _step_index_log_write,
    _validate_before_write,
    init_vault,
    PipelineError,
)
from llmwiki_engine.lite.profile import load_profile
from llmwiki_engine.lite.providers import ProviderCallResult, ProviderSpec, build_chat_payload


def write_raw(vault: Path, name: str = "project_note.md") -> Path:
    raw = vault / "raw" / name
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(
        "# 项目笔记\n\n"
        "这个项目的核心目标是把知识库维护流程拆成可测试的模块。"
        "第一阶段先证明 CLI、状态机、artifact 和 validator 能跑通。\n\n"
        "## 简化 Ingest 流程\n\n"
        "简化 Ingest 不进行人工审核，也不需要手动 apply。"
        "系统会先合并候选，再生成候选页面，然后通过向量召回旧页面并自动写入 wiki。\n",
        encoding="utf-8",
    )
    return raw


def make_decision(
    *,
    decision_id: str,
    candidate_page_id: str,
    action: str,
    target_path: str | None,
    title: str,
    ref: SourceRef,
    matched_existing_paths: list[str] | None = None,
) -> MergeDecision:
    return MergeDecision(
        decision_id=decision_id,
        candidate_page_id=candidate_page_id,
        action=action,  # type: ignore[arg-type]
        target_path=target_path,
        title=title,
        page_type="concept",
        content_scope=f"写入 {title} 对应的候选内容。",
        candidate_path_index=["摘要", "核心内容"],
        matched_existing_paths=matched_existing_paths or [],
        reason="按候选内容生成合并决策。",
        source_refs=[ref],
    )


def test_source_digest_rejects_english_user_facing_text() -> None:
    ref = SourceRef(raw_path="raw/project_note.md", raw_sha256="abc", locator="whole_file")
    digest = SourceDigest(
        source_raw_path="raw/project_note.md",
        raw_sha256="abc",
        summary="English summary",
        key_takeaways=["English takeaway"],
        concepts=[
            SourceDigestCandidate(
                candidate_id="CAND-001",
                kind="concept",
                name="English Name",
                suggested_page_title="English Title",
                summary="English candidate summary",
                source_basis="English source basis",
                source_refs=[ref],
            )
        ],
    )

    with pytest.raises(PipelineError, match="必须使用中文用户可读文本"):
        _assert_source_digest_chinese(digest)


def test_source_digest_allows_proper_noun_name_with_chinese_context() -> None:
    ref = SourceRef(raw_path="raw/project_note.md", raw_sha256="abc", locator="whole_file")
    digest = SourceDigest(
        source_raw_path="raw/project_note.md",
        raw_sha256="abc",
        summary="本文解释 RAG 的核心流程。",
        key_takeaways=["RAG 先检索，再增强 prompt，最后生成回答。"],
        concepts=[
            SourceDigestCandidate(
                candidate_id="CAND-001",
                kind="concept",
                name="RAG",
                suggested_page_title="RAG（检索增强生成）",
                summary="RAG 是检索增强生成流程。",
                source_basis="表格说明了三个字母的含义。",
                source_refs=[ref],
            )
        ],
    )

    _assert_source_digest_chinese(digest)


def test_cli_json_run(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", home.as_posix())
    vault = tmp_path / "vault"
    runner = CliRunner()
    result = runner.invoke(app, ["init", str(vault)])
    assert result.exit_code == 0, result.output
    raw = write_raw(vault)

    result = runner.invoke(app, ["ingest", "run", str(vault), str(raw), "--slug", "cli", "--json"])

    assert result.exit_code != 0
    assert "必须使用真实模型 provider" in result.output


def test_log_pages_are_date_sharded_without_global_log(tmp_path: Path) -> None:
    vault = init_vault(tmp_path / "vault")
    raw = write_raw(vault)
    raw_path = raw.relative_to(vault).as_posix()
    assert not (vault / "wiki" / "log.md").exists()

    binding = RawBinding(
        raw_path=raw_path,
        raw_sha256=sha256_file(raw),
        size_bytes=raw.stat().st_size,
        mtime_ns=raw.stat().st_mtime_ns,
        bound_at="2026-06-16T00:00:00Z",
    )
    manifest = OperationManifest(
        operation_id="ING-20260616T000000Z-log-test",
        status="running",
        created_at="2026-06-16T00:00:00Z",
        updated_at="2026-06-16T00:00:00Z",
        vault=vault.as_posix(),
        raw_path=raw_path,
        profile_name="default",
        engine_version="test",
    )
    merge_plan = MergePlan(action_counts={"create": 2, "update": 1, "noop": 0}, decisions=[])

    output = _step_index_log_write(
        vault,
        tmp_path / "run",
        {"raw_binding": binding, "merge_plan": merge_plan, "profile": load_profile(vault)},
        manifest,
    )

    assert output.counts["system_written_count"] == 2
    assert output.counts["written_target_count"] == 2
    assert (vault / "wiki" / "index.md").exists()
    assert not (vault / "wiki" / "log.md").exists()
    daily_log = vault / "wiki" / "logs" / "2026-06-16.md"
    assert daily_log.exists()
    assert "`ING-20260616T000000Z-log-test`" in daily_log.read_text(encoding="utf-8")
    write_result = json.loads((tmp_path / "run" / "index_log_write" / "write_result.json").read_text(encoding="utf-8"))
    assert write_result["written_targets"] == ["index.md", "logs/2026-06-16.md"]


def test_sentence_transformers_embedding_backend_uses_qwen_cache_contract(tmp_path: Path, monkeypatch) -> None:
    vault = init_vault(tmp_path / "vault")
    page_path = vault / "wiki" / "concepts" / "Concept_Qwen.md"
    page_path.parent.mkdir(parents=True, exist_ok=True)
    page_path.write_text("# Qwen Embedding\n\nA page about local Qwen retrieval.\n", encoding="utf-8")
    entry = WikiKnowledgeEntry(
        path="concepts/Concept_Qwen.md",
        title="Qwen Embedding",
        page_type="concept",
        sha256=sha256_file(page_path),
        summary="A page about local Qwen retrieval.",
        text_excerpt="A page about local Qwen retrieval.",
    )
    config = embeddings.EmbeddingConfig(
        backend="sentence_transformers",
        dimensions=3,
        max_page_chars=200,
        max_query_chars=200,
    )
    calls: list[tuple[bool, list[str]]] = []

    def fake_embed_texts(texts: list[str], config: embeddings.EmbeddingConfig, *, is_query: bool) -> list[list[float]]:
        calls.append((is_query, list(texts)))
        vector = [0.8, 0.6, 0.0] if is_query else [1.0, 0.0, 0.0]
        return [vector for _ in texts]

    monkeypatch.setattr(embeddings, "embed_texts", fake_embed_texts)

    records, metrics = embeddings.sync_page_embedding_cache(vault, [entry], config)

    assert metrics["backend"] == "sentence_transformers"
    assert metrics["model"] == embeddings.DEFAULT_QWEN_EMBEDDING_MODEL
    assert metrics["cache_created_paths"] == [entry.path]
    assert metrics["cache_refreshed_paths"] == [entry.path]
    assert records[entry.path]["dimensions"] == 3
    assert records[entry.path]["max_page_chars"] == 200
    assert len(records[entry.path]["vector"]) == 3
    assert calls[0][0] is False

    calls.clear()
    cached_records, cached_metrics = embeddings.sync_page_embedding_cache(vault, [entry], config)

    assert cached_metrics["cache_hit"] == 1
    assert cached_metrics["cache_hit_paths"] == [entry.path]
    assert cached_records[entry.path]["cache_hit"] is True
    assert calls == []

    ref = SourceRef(raw_path="raw/qwen.md", raw_sha256="abc", locator="whole_file")
    candidate_pages = CandidatePages(
        pages=[
            CandidatePage(
                candidate_page_id="CP-001",
                candidate_unit_id="CM-001",
                source_candidate_ids=["CAND-001"],
                title="Local Embedding Retrieval",
                proposed_page_type="concept",
                proposed_path_hint="concepts/Concept_Local_Embedding_Retrieval.md",
                summary="Use Qwen locally for retrieval.",
                body_markdown="## Summary\n\nUse Qwen locally for retrieval.",
                source_refs=[ref],
                confidence=0.8,
            )
        ]
    )

    contexts = embeddings.build_candidate_contexts(candidate_pages, [entry], cached_records, config)

    assert calls[0][0] is True
    assert contexts.retrieval_backend == "sentence_transformers"
    assert contexts.model == embeddings.DEFAULT_QWEN_EMBEDDING_MODEL
    assert contexts.items[0].hits[0].match_basis == "embedding"
    assert contexts.items[0].hits[0].path == entry.path


def test_candidate_contexts_rejects_non_embedding_backend(tmp_path: Path) -> None:
    vault = init_vault(tmp_path / "vault")
    ref = SourceRef(raw_path="raw/qwen.md", raw_sha256="abc", locator="whole_file")
    candidate_pages = CandidatePages(
        pages=[
            CandidatePage(
                candidate_page_id="CP-001",
                candidate_unit_id="CM-001",
                source_candidate_ids=["CAND-001"],
                title="RAG 系统",
                proposed_page_type="concept",
                proposed_path_hint="concepts/Concept_RAG系统.md",
                summary="RAG 系统使用检索结果增强生成。",
                body_markdown="## 摘要\n\nRAG 系统使用检索结果增强生成。",
                source_refs=[ref],
                confidence=0.8,
            )
        ]
    )
    config = embeddings.EmbeddingConfig(backend="unsupported-test", dimensions=256)

    with pytest.raises(RuntimeError, match="真实 embedding"):
        embeddings.build_candidate_contexts(candidate_pages, [], {}, config)


def test_candidate_open_question_locator_resolves_to_chinese_body_question() -> None:
    ref = SourceRef(raw_path="raw/managed_agents.md", raw_sha256="abc", locator="whole_file")
    artifact = CandidatePages(
        pages=[
            CandidatePage(
                candidate_page_id="CP-005",
                candidate_unit_id="CM-005",
                source_candidate_ids=["C-003"],
                title="会话作为外部上下文",
                proposed_page_type="concept",
                proposed_path_hint="concepts/Concept_Session_Context.md",
                summary="会话日志是外部上下文对象。",
                body_markdown=(
                    "## 矛盾与未决问题\n\n"
                    "- 未来模型需要怎样的上下文工程，harness 应如何演进以支持不可预见的上下文操作？【O-001】\n"
                ),
                open_questions=["O-001", "Claude-3 是否需要特殊上下文工程？"],
                source_refs=[ref],
                evidence_notes=["section"],
                confidence=0.9,
            )
        ]
    )

    normalized = _normalize_candidate_pages(artifact)

    assert normalized.pages[0].open_questions == [
        "未来模型需要怎样的上下文工程，harness 应如何演进以支持不可预见的上下文操作？",
        "Claude-3 是否需要特殊上下文工程？",
    ]
    assert normalized.pages[0].evidence_notes == ["来源定位：section"]
    _assert_candidate_pages_chinese(normalized)


def test_candidate_open_question_orphan_locator_is_dropped() -> None:
    ref = SourceRef(raw_path="raw/managed_agents.md", raw_sha256="abc", locator="whole_file")
    artifact = CandidatePages(
        pages=[
            CandidatePage(
                candidate_page_id="CP-001",
                candidate_unit_id="CM-001",
                source_candidate_ids=["C-001"],
                title="上下文工程",
                proposed_page_type="concept",
                proposed_path_hint="concepts/Concept_Context_Engineering.md",
                summary="上下文工程需要根据任务演进。",
                body_markdown="## 摘要\n\n上下文工程需要根据任务演进。\n",
                open_questions=["O-001"],
                source_refs=[ref],
                confidence=0.8,
            )
        ]
    )

    normalized = _normalize_candidate_pages(artifact)

    assert normalized.pages[0].open_questions == []
    _assert_candidate_pages_chinese(normalized)


def test_merge_plan_candidate_path_index_allows_locator_values() -> None:
    ref = SourceRef(raw_path="raw/a.md", raw_sha256="abc", locator="whole_file")
    plan = MergePlan(
        decisions=[
            MergeDecision(
                decision_id="",
                candidate_page_id="CP-001",
                action="create",
                target_path="concepts/Concept_Context.md",
                title="上下文工程",
                page_type="concept",
                content_scope="写入候选页中关于上下文工程的内容。",
                candidate_path_index=["O-001", "摘要"],
                reason="候选页包含新的上下文工程说明。",
                source_refs=[ref],
            )
        ],
        action_counts={},
    )

    normalized = _normalize_merge_plan(plan)

    assert normalized.decisions[0].decision_id == "MD-001"
    assert normalized.decisions[0].candidate_path_index == ["候选页定位：O-001", "摘要"]
    _assert_merge_plan_chinese(normalized)


def test_candidate_prompts_do_not_receive_wiki_snapshot_before_embedding(tmp_path: Path) -> None:
    vault = init_vault(tmp_path / "vault")
    raw = write_raw(vault)
    ref = SourceRef(raw_path="raw/project_note.md", raw_sha256=sha256_file(raw), locator="whole_file")
    digest = SourceDigest(
        source_raw_path="raw/project_note.md",
        raw_sha256=sha256_file(raw),
        summary="这份材料说明自动化知识库入库流程。",
        key_takeaways=["候选页必须先忠于原文，再进行旧 wiki 召回。"],
        concepts=[
            SourceDigestCandidate(
                candidate_id="CAND-001",
                kind="concept",
                name="自动化入库",
                suggested_page_title="自动化入库",
                summary="自动化入库强调去掉人工审核节点。",
                source_basis="原文说明流程不再人工审核。",
                source_refs=[ref],
            )
        ],
    )
    unit = CandidateMergeUnit(
        candidate_unit_id="CM-001",
        source_candidate_ids=["CAND-001"],
        title="自动化入库",
        page_type="concept",
        path_hint="concepts/Concept_Auto_Ingest.md",
        summary="合并后的候选页单元。",
        merge_reason="只有一个同义候选，直接生成页面。",
        must_cover_points=["解释为什么候选页先忠于原文。"],
        source_refs=[ref],
    )
    profile = load_profile(vault)

    merge_prompt = prompts.candidate_merge_prompt(digest=digest, profile=profile)
    page_prompt = prompts.candidate_page_prompt(
        digest=digest,
        candidate_unit=unit,
        raw_path=ref.raw_path,
        raw_sha256=ref.raw_sha256,
        raw_text=raw.read_text(encoding="utf-8"),
        profile=profile,
    )
    candidate_pages = CandidatePages(
        pages=[
            CandidatePage(
                candidate_page_id="CP-001",
                candidate_unit_id="CM-001",
                source_candidate_ids=["CAND-001"],
                title="自动化入库",
                proposed_page_type="concept",
                proposed_path_hint="concepts/Concept_Auto_Ingest.md",
                summary="候选页正文忠于原文。",
                body_markdown="## 摘要\n\n候选页正文忠于原文。",
                source_refs=[ref],
                confidence=0.9,
            )
        ]
    )
    contexts = CandidateContexts(
        retrieval_backend="sentence_transformers",
        model=embeddings.DEFAULT_QWEN_EMBEDDING_MODEL,
        input_version="test",
        top_k=5,
        knowledge_pool_size=0,
        candidate_page_count=1,
        candidate_pool_hash="empty",
        items=[],
    )
    merge_plan_prompt = prompts.merge_plan_prompt(candidate_pages=candidate_pages, candidate_contexts=contexts, profile=profile)

    assert merge_prompt.schema_name == "llmwiki_lite_candidate_merge"
    assert "wiki_snapshot" not in merge_prompt.user_payload
    assert "wiki_snapshot_entries" not in page_prompt.user_payload
    assert "wiki_snapshot" not in merge_plan_prompt.user_payload
    assert "wiki_snapshot_entries" not in merge_plan_prompt.user_payload
    assert page_prompt.user_payload["candidate_unit"]["candidate_unit_id"] == "CM-001"
    assert page_prompt.cache_prefix_payload
    assert page_prompt.cache_prefix_payload["raw_text"]


def test_candidate_merge_retries_unknown_source_candidate_ids(tmp_path: Path) -> None:
    vault = init_vault(tmp_path / "vault")
    ref = SourceRef(raw_path="raw/project_note.md", raw_sha256="abc", locator="whole_file")
    digest = SourceDigest(
        source_raw_path="raw/project_note.md",
        raw_sha256="abc",
        summary="这份材料说明自动化知识库入库流程。",
        key_takeaways=["候选合并只能引用真实候选 ID。"],
        concepts=[
            SourceDigestCandidate(
                candidate_id="CAND-001",
                kind="concept",
                name="自动化入库",
                suggested_page_title="自动化入库",
                summary="自动化入库强调去掉人工审核节点。",
                source_basis="原文说明流程不再人工审核。",
                source_refs=[ref],
            )
        ],
    )
    invalid_plan = CandidateMergePlan(
        units=[
            CandidateMergeUnit(
                candidate_unit_id="CM-999",
                source_candidate_ids=["B-001"],
                title="错误候选",
                page_type="concept",
                path_hint="concepts/Concept_Bad.md",
                summary="这一轮错误引用了不存在的候选。",
                merge_reason="模型误把说明性编号当成候选 ID。",
                must_cover_points=["修正不存在的候选 ID。"],
                source_refs=[ref],
            )
        ]
    )
    valid_plan = CandidateMergePlan(
        units=[
            CandidateMergeUnit(
                candidate_unit_id="CM-999",
                source_candidate_ids=["CAND-001"],
                title="自动化入库",
                page_type="concept",
                path_hint="concepts/Concept_Auto_Ingest.md",
                summary="这一轮使用真实候选 ID。",
                merge_reason="只保留 source_digest 中存在的候选 ID。",
                must_cover_points=["解释候选合并为什么不能编造 ID。"],
                source_refs=[ref],
            )
        ]
    )

    class FakeRegistry:
        def __init__(self) -> None:
            self.spec = ProviderSpec(
                spec="openai_compatible:deepseek-v4-flash",
                endpoint="https://api.deepseek.com/v1/chat/completions",
                api_key="test-key",
            )
            self.requests = []

        def provider_for(self, step: str) -> ProviderSpec:
            return self.spec

        def call_structured(self, step: str, request, output_model):
            self.requests.append(request)
            output = invalid_plan if len(self.requests) == 1 else valid_plan
            return ProviderCallResult(
                output=output,
                prompt_artifact={"step": step, "request": request.model_dump(mode="json")},
                provider_result={"parsed": output.model_dump(mode="json")},
                sanitized_context=self.spec.sanitized_context(),
                api_calls=[
                    {
                        "step": step,
                        "request_key": "",
                        "model": "deepseek-v4-flash",
                        "attempt": 0,
                        "call_index": len(self.requests),
                        "status": "success",
                        "finish_reason": "stop",
                        "duration_ms": 1.0,
                        "response_status_code": 200,
                        "error": "",
                        "prompt_tokens": 10,
                        "prompt_cache_hit_tokens": 0,
                        "prompt_cache_miss_tokens": 10,
                        "completion_tokens": 5,
                        "reasoning_tokens": 0,
                        "total_tokens": 15,
                        "cache_hit_rate_percent": 0.0,
                        "price_cny": 0.00002,
                    }
                ],
            )

    registry = FakeRegistry()
    state = {
        "source_digest": digest,
        "profile": load_profile(vault),
        "provider_registry": registry,
        "provider_contexts": {},
    }

    output = _step_candidate_merge(tmp_path / "run", state)
    plan = state["candidate_merge"]

    assert isinstance(plan, CandidateMergePlan)
    assert [unit.source_candidate_ids for unit in plan.units] == [["CAND-001"]]
    assert len(registry.requests) == 2
    assert registry.requests[1].user_payload["allowed_source_candidate_ids"] == ["CAND-001"]
    assert "B-001" in registry.requests[1].user_payload["validation_error"]
    assert output.counts["semantic_retry_count"] == 1
    assert output.counts["api_call_count"] == 2
    assert output.counts["api_success_count"] == 1
    assert output.counts["api_paused_count"] == 1


def test_candidate_page_warmup_and_generation_share_cache_prefix(tmp_path: Path) -> None:
    vault = init_vault(tmp_path / "vault")
    raw = write_raw(vault)
    raw_text = raw.read_text(encoding="utf-8")
    raw_sha = sha256_file(raw)
    ref = SourceRef(raw_path="raw/project_note.md", raw_sha256=raw_sha, locator="whole_file")
    digest = SourceDigest(
        source_raw_path="raw/project_note.md",
        raw_sha256=raw_sha,
        summary="这份材料说明自动化知识库入库流程。",
        key_takeaways=["候选页必须先忠于原文，再进行旧 wiki 召回。"],
        concepts=[
            SourceDigestCandidate(
                candidate_id="CAND-001",
                kind="concept",
                name="自动化入库",
                suggested_page_title="自动化入库",
                summary="自动化入库强调去掉人工审核节点。",
                source_basis="原文说明流程不再人工审核。",
                source_refs=[ref],
            )
        ],
    )
    unit = CandidateMergeUnit(
        candidate_unit_id="CM-001",
        source_candidate_ids=["CAND-001"],
        title="自动化入库",
        page_type="concept",
        path_hint="concepts/Concept_Auto_Ingest.md",
        summary="合并后的候选页单元。",
        merge_reason="只有一个同义候选，直接生成页面。",
        must_cover_points=["解释为什么候选页先忠于原文。"],
        source_refs=[ref],
    )
    profile = load_profile(vault)
    spec = ProviderSpec(spec="openai_compatible:deepseek-v4-flash", endpoint="https://api.deepseek.com/v1/chat/completions", api_key="test-key")

    warmup_prompt = prompts.candidate_pages_warmup_prompt(digest=digest, raw_path=ref.raw_path, raw_sha256=raw_sha, raw_text=raw_text, profile=profile)
    page_prompt = prompts.candidate_page_prompt(digest=digest, candidate_unit=unit, raw_path=ref.raw_path, raw_sha256=raw_sha, raw_text=raw_text, profile=profile)
    warmup_payload = build_chat_payload(spec, warmup_prompt)
    page_payload = build_chat_payload(spec, page_prompt)

    assert warmup_prompt.schema_name == "llmwiki_lite_candidate_pages_warmup"
    assert page_prompt.schema_name == "llmwiki_lite_candidate_pages"
    assert warmup_payload["messages"][1]["content"] == page_payload["messages"][1]["content"]
    assert len(warmup_payload["messages"]) == 3
    assert len(page_payload["messages"]) == 3
    shared_prefix = json.loads(page_payload["messages"][1]["content"])["cache_prefix"]
    variable_suffix = json.loads(page_payload["messages"][2]["content"])
    warmup_suffix = json.loads(warmup_payload["messages"][2]["content"])
    assert shared_prefix["task"] == "candidate_pages"
    assert shared_prefix["raw_text"] == raw_text
    assert "candidate_unit" not in shared_prefix
    assert variable_suffix["input"]["candidate_unit"]["candidate_unit_id"] == "CM-001"
    assert "raw_text" not in variable_suffix["input"]
    assert warmup_suffix["input"]["warmup"] is True
    assert warmup_suffix["json_output_example"] == {"status": "OK"}


def test_merge_plan_allows_split_decisions_but_update_must_use_top5() -> None:
    ref = SourceRef(raw_path="raw/a.md", raw_sha256="abc", locator="whole_file")
    candidate_pages = CandidatePages(
        pages=[
            CandidatePage(
                candidate_page_id="CP-001",
                candidate_unit_id="CM-001",
                source_candidate_ids=["CAND-001"],
                title="自动化入库",
                proposed_page_type="concept",
                proposed_path_hint="concepts/Concept_Auto_Ingest.md",
                summary="候选页包含可更新旧页和可新建页面的两部分。",
                body_markdown="## 摘要\n\n这是一页候选页。",
                source_refs=[ref],
                confidence=0.9,
            )
        ]
    )
    contexts = CandidateContexts(
        retrieval_backend="sentence_transformers",
        model=embeddings.DEFAULT_QWEN_EMBEDDING_MODEL,
        input_version="test",
        top_k=5,
        knowledge_pool_size=1,
        candidate_page_count=1,
        candidate_pool_hash="hash",
        items=[
            CandidateContext(
                candidate_page_id="CP-001",
                query="自动化入库",
                hits=[CandidateContextHit(path="concepts/Concept_Old.md", title="旧页", rank=1, score=0.9, reason="向量相近。")],
            )
        ],
    )
    plan = MergePlan(
        action_counts={"create": 1, "update": 1, "noop": 0},
        decisions=[
            make_decision(
                decision_id="MD-001",
                candidate_page_id="CP-001",
                action="update",
                target_path="concepts/Concept_Old.md",
                title="旧页",
                ref=ref,
                matched_existing_paths=["concepts/Concept_Old.md"],
            ),
            make_decision(decision_id="MD-002", candidate_page_id="CP-001", action="create", target_path="concepts/Concept_New.md", title="新页", ref=ref),
        ],
    )

    _assert_merge_plan_consumes_candidates(plan, candidate_pages, contexts)

    bad_plan = plan.model_copy(
        update={
            "decisions": [
                plan.decisions[0].model_copy(update={"decision_id": "MD-BAD", "target_path": "concepts/Concept_Not_In_Top5.md"}),
            ]
        }
    )
    with pytest.raises(PipelineError, match="TopK 召回"):
        _assert_merge_plan_consumes_candidates(bad_plan, candidate_pages, contexts)


def test_source_digest_related_candidates_resolve_to_sibling_pages() -> None:
    ref = SourceRef(raw_path="raw/a.md", raw_sha256="abc", locator="whole_file")
    first = SourceDigestCandidate(
        candidate_id="CAND-001",
        kind="concept",
        name="First",
        suggested_page_title="First",
        summary="First summary.",
        source_basis="First basis.",
        source_refs=[ref],
        related_candidates=["CAND-002"],
    )
    second = SourceDigestCandidate(
        candidate_id="CAND-002",
        kind="concept",
        name="Second",
        suggested_page_title="Second",
        summary="Second summary.",
        source_basis="Second basis.",
        source_refs=[ref],
    )
    digest = SourceDigest(source_raw_path="raw/a.md", raw_sha256="abc", summary="summary", key_takeaways=["one"], concepts=[first, second])
    candidate_pages = CandidatePages(
        pages=[
            CandidatePage(
                candidate_page_id="CP-001",
                candidate_unit_id="CM-001",
                source_candidate_ids=["CAND-001"],
                title="First",
                proposed_page_type="concept",
                proposed_path_hint="concepts/Concept_First.md",
                summary="First summary.",
                body_markdown="## Summary\n\nFirst.",
                source_refs=[ref],
                confidence=0.8,
            ),
            CandidatePage(
                candidate_page_id="CP-002",
                candidate_unit_id="CM-002",
                source_candidate_ids=["CAND-002"],
                title="Second",
                proposed_page_type="concept",
                proposed_path_hint="concepts/Concept_Second.md",
                summary="Second summary.",
                body_markdown="## Summary\n\nSecond.",
                source_refs=[ref],
                confidence=0.8,
            ),
        ]
    )
    plan = MergePlan(
        action_counts={"create": 2, "update": 0, "noop": 0},
        decisions=[
            make_decision(decision_id="MD-001", candidate_page_id="CP-001", action="create", target_path="concepts/Concept_First.md", title="First", ref=ref),
            make_decision(decision_id="MD-002", candidate_page_id="CP-002", action="create", target_path="concepts/Concept_Second.md", title="Second", ref=ref),
        ],
    )
    contexts = CandidateContexts(
        retrieval_backend="sentence_transformers",
        model=embeddings.DEFAULT_QWEN_EMBEDDING_MODEL,
        input_version="test",
        top_k=5,
        knowledge_pool_size=0,
        candidate_page_count=2,
        candidate_pool_hash="hash",
        items=[],
    )
    snapshot = WikiSnapshot(wiki_root="wiki", pool_hash="empty", generated_at="2026-06-11T00:00:00Z", entries=[])

    resolved, report = related_logic.finalize_merge_plan_related(plan, candidate_pages=candidate_pages, digest=digest, snapshot=snapshot, contexts=contexts)

    assert resolved.decisions[0].related_pages[0].target_path == "concepts/Concept_Second.md"
    assert resolved.decisions[0].related_pages[0].source == "source_digest"
    assert any(item.decision == "kept" and item.target_path == "concepts/Concept_Second.md" for item in report.candidates)


def test_top5_context_can_become_related_but_unknown_paths_are_filtered() -> None:
    ref = SourceRef(raw_path="raw/a.md", raw_sha256="abc", locator="whole_file")
    digest = SourceDigest(
        source_raw_path="raw/a.md",
        raw_sha256="abc",
        summary="summary",
        key_takeaways=["one"],
        concepts=[
            SourceDigestCandidate(
                candidate_id="CAND-001",
                kind="concept",
                name="New",
                suggested_page_title="New",
                summary="New summary.",
                source_basis="basis",
                source_refs=[ref],
            )
        ],
    )
    candidate_pages = CandidatePages(
        pages=[
            CandidatePage(
                candidate_page_id="CP-001",
                candidate_unit_id="CM-001",
                source_candidate_ids=["CAND-001"],
                title="New",
                proposed_page_type="concept",
                proposed_path_hint="concepts/Concept_New.md",
                summary="New summary.",
                body_markdown="## Summary\n\nNew.",
                source_refs=[ref],
                confidence=0.8,
            )
        ]
    )
    old_entry = WikiKnowledgeEntry(
        path="concepts/Concept_Old.md",
        title="Old",
        page_type="concept",
        sha256="oldsha",
        summary="Old summary.",
        text_excerpt="Old page.",
    )
    snapshot = WikiSnapshot(wiki_root="wiki", pool_hash="pool", generated_at="2026-06-11T00:00:00Z", entries=[old_entry])
    contexts = CandidateContexts(
        retrieval_backend="sentence_transformers",
        model=embeddings.DEFAULT_QWEN_EMBEDDING_MODEL,
        input_version="test",
        top_k=5,
        knowledge_pool_size=1,
        candidate_page_count=1,
        candidate_pool_hash="hash",
        items=[
            CandidateContext(
                candidate_page_id="CP-001",
                query="New",
                hits=[
                    CandidateContextHit(path="concepts/Concept_Old.md", title="Old", rank=1, score=0.42, reason="related old page"),
                    CandidateContextHit(path="concepts/Concept_Missing.md", title="Missing", rank=2, score=0.41, reason="missing"),
                ],
            )
        ],
    )
    plan = MergePlan(
        action_counts={"create": 1, "update": 0, "noop": 0},
        decisions=[make_decision(decision_id="MD-001", candidate_page_id="CP-001", action="create", target_path="concepts/Concept_New.md", title="New", ref=ref)],
    )

    resolved, report = related_logic.finalize_merge_plan_related(plan, candidate_pages=candidate_pages, digest=digest, snapshot=snapshot, contexts=contexts)

    assert [ref.target_path for ref in resolved.decisions[0].related_pages] == ["concepts/Concept_Old.md"]
    assert any(item.target_path == "concepts/Concept_Missing.md" and item.decision == "filtered" for item in report.candidates)


def test_validation_rejects_model_related_sections_and_self_wikilinks(tmp_path: Path) -> None:
    vault = init_vault(tmp_path / "vault")
    raw = write_raw(vault)
    ref = SourceRef(raw_path="raw/project_note.md", raw_sha256=sha256_file(raw), locator="whole_file")
    markdown = (
        "---\n"
        "title: Bad\n"
        "type: concept\n"
        "source_refs:\n"
        f"- raw_path: {ref.raw_path}\n"
        f"  raw_sha256: {ref.raw_sha256}\n"
        f"  locator: {ref.locator}\n"
        "---\n\n"
        "# Bad\n\n"
        "[[concepts/Concept_Bad]]\n\n"
        "## Related\n\n"
        "- [[concepts/Concept_Other]]\n"
    )
    page = FinalPage(
        final_page_id="FP-001",
        target_path="concepts/Concept_Bad.md",
        action="create",
        title="Bad",
        page_type="concept",
        markdown=markdown,
        content_sha256=sha256_text(markdown),
        source_refs=[ref],
    )
    binding = RawBinding(
        raw_path="raw/project_note.md",
        raw_sha256=sha256_file(raw),
        size_bytes=raw.stat().st_size,
        mtime_ns=raw.stat().st_mtime_ns,
        bound_at="2026-06-11T00:00:00Z",
    )

    report = _validate_before_write(
        vault,
        {
            "raw_abs": raw,
            "raw_binding": binding,
            "final_pages": FinalPages(pages=[page]),
            "profile": load_profile(vault),
        },
    )

    codes = {issue.code for issue in report.issues}
    assert "model_related_section" in codes
    assert "self_wikilink" in codes


def test_canonicalization_rejects_raw_source_and_system_graph_links(tmp_path: Path) -> None:
    vault = init_vault(tmp_path / "vault")
    raw = write_raw(vault)
    ref = SourceRef(raw_path="raw/project_note.md", raw_sha256=sha256_file(raw), locator="whole_file")
    page = FinalPage(
        final_page_id="FP-001",
        target_path="concepts/Concept_Graph_Bad.md",
        action="create",
        title="Graph Bad",
        page_type="concept",
        markdown=(
            "# Graph Bad\n\n"
            "[[raw/project_note]]\n"
            "[source](sources/Source_project_note.md)\n"
            '<a href="logs/2026-06-11.md">log</a>\n'
        ),
        content_sha256="",
        source_refs=[ref],
    )

    with pytest.raises(PipelineError, match="raw/source/system 图谱链接"):
        _canonical_final_markdown(page, None, operation_id="OP-GRAPH")


def test_canonicalization_strips_model_related_section(tmp_path: Path) -> None:
    vault = init_vault(tmp_path / "vault")
    raw = write_raw(vault)
    ref = SourceRef(raw_path="raw/project_note.md", raw_sha256=sha256_file(raw), locator="whole_file")
    page = FinalPage(
        final_page_id="FP-001",
        target_path="concepts/Concept_Model_Related.md",
        action="create",
        title="模型相关页",
        page_type="concept",
        markdown="# 模型相关页\n\n## 摘要\n\n这是正文。\n\n## 相关页面\n\n- 模型自己写的相关页面。\n",
        content_sha256="",
        source_refs=[ref],
    )

    markdown = _canonical_final_markdown(page, None, operation_id="OP-RELATED")

    assert "模型自己写的相关页面" not in markdown
    assert "## 相关页面" not in markdown


def test_validation_rejects_source_ref_mismatch_and_unsafe_overwrite(tmp_path: Path) -> None:
    vault = init_vault(tmp_path / "vault")
    raw = write_raw(vault)
    existing = vault / "wiki" / "concepts" / "Concept_Bad.md"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text("# Existing\n", encoding="utf-8")
    ref = SourceRef(raw_path="raw/other.md", raw_sha256="wrong", locator="whole_file")
    page = FinalPage(
        final_page_id="FP-001",
        target_path="concepts/Concept_Bad.md",
        action="create",
        title="Bad",
        page_type="concept",
        markdown="# Bad\n\nBody",
        source_refs=[ref],
    )
    markdown = _canonical_final_markdown(page, None, operation_id="OP-MISMATCH")
    page = page.model_copy(update={"markdown": markdown, "content_sha256": sha256_text(markdown)})
    binding = RawBinding(
        raw_path="raw/project_note.md",
        raw_sha256=sha256_file(raw),
        size_bytes=raw.stat().st_size,
        mtime_ns=raw.stat().st_mtime_ns,
        bound_at="2026-06-11T00:00:00Z",
    )

    report = _validate_before_write(
        vault,
        {
            "raw_abs": raw,
            "raw_binding": binding,
            "final_pages": FinalPages(pages=[page]),
            "profile": load_profile(vault),
        },
    )

    codes = {issue.code for issue in report.issues}
    assert report.ok is False
    assert "source_ref_mismatch" in codes
    assert "unsafe_overwrite" in codes


def test_update_frontmatter_preserves_existing_provenance_and_aliases(tmp_path: Path) -> None:
    vault = init_vault(tmp_path / "vault")
    raw = write_raw(vault)
    existing = WikiKnowledgeEntry(
        path="concepts/Concept_Update.md",
        title="Existing Title",
        page_type="concept",
        sha256="oldsha",
        summary="old summary",
        aliases=["Old Alias"],
        created="2026-01-01",
        source_raw_paths=["raw/old.md"],
        source_raw_hashes=["old-raw-hash"],
        source_prepared_hashes=["old-prepared-hash"],
        source_operation_ids=["OLD-OP"],
        updated="2026-01-02",
        text_excerpt="# Existing Title\n\nOld body.",
    )
    ref = SourceRef(raw_path="raw/project_note.md", raw_sha256=sha256_file(raw), locator="whole_file")
    page = FinalPage(
        final_page_id="FP-001",
        target_path="concepts/Concept_Update.md",
        action="update",
        title="Model Suggested Title",
        page_type="concept",
        markdown="# Existing Title\n\n## Summary\n\nnew summary",
        content_sha256="",
        source_refs=[ref],
        preimage_sha256="oldsha",
    )

    markdown = _canonical_final_markdown(page.model_copy(update={"title": existing.title}), None, operation_id="OP-NEW", existing_entry=existing, model_title=page.title)
    frontmatter = yaml.safe_load(markdown.split("---", 2)[1])

    assert frontmatter["title"] == "Existing Title"
    assert frontmatter["aliases"] == ["Old Alias", "Model Suggested Title"]
    assert str(frontmatter["created"]) == "2026-01-01"
    assert frontmatter["source_raw_paths"] == ["raw/old.md", "raw/project_note.md"]
    assert frontmatter["source_raw_hashes"] == ["old-raw-hash", sha256_file(raw)]
    assert frontmatter["source_prepared_hashes"] == ["old-prepared-hash", sha256_file(raw)]
    assert frontmatter["source_operation_ids"] == ["OLD-OP", "OP-NEW"]
    assert frontmatter["last_ingest_operation"] == "OP-NEW"


def test_page_generation_parallelism_defaults_to_request_count_and_supports_limit() -> None:
    assert _page_generation_parallelism({}, 16) == 16
    assert _page_generation_parallelism({"config": {"page_generation": {}}}, 23) == 23
    assert _page_generation_parallelism({"config": {"page_generation": {"max_parallel_requests": 7}}}, 23) == 7


def test_composition_and_final_page_prompt_runtime_contracts(tmp_path: Path) -> None:
    vault = init_vault(tmp_path / "vault")
    ref = SourceRef(raw_path="raw/project_note.md", raw_sha256="abc", locator="whole_file")
    candidate_pages = CandidatePages(
        pages=[
            CandidatePage(
                candidate_page_id="CP-001",
                candidate_unit_id="CM-001",
                source_candidate_ids=["CAND-001"],
                title="项目知识库",
                proposed_page_type="concept",
                proposed_path_hint="concepts/Concept_Project_Wiki.md",
                summary="项目知识库用于沉淀长期知识。",
                body_markdown="# 项目知识库\n\n项目知识库用于沉淀长期知识。",
                source_refs=[ref],
                confidence=0.9,
            )
        ]
    )
    plan = MergePlan(
        action_counts={"create": 1, "update": 0, "noop": 0},
        decisions=[make_decision(decision_id="MD-001", candidate_page_id="CP-001", action="create", target_path="concepts/Concept_Project_Wiki.md", title="项目知识库", ref=ref)],
    )
    snapshot = WikiSnapshot(wiki_root="wiki", pool_hash="empty", generated_at="2026-06-12T00:00:00Z", entries=[])
    profile = load_profile(vault)

    composition_prompt = prompts.composition_plan_prompt(merge_plan=plan, candidate_pages=candidate_pages, profile=profile)

    assert composition_prompt.schema_name == "llmwiki_lite_composition_plan"
    item = CompositionItem(
        final_page_id="FP-001",
        target_path="concepts/Concept_Project_Wiki.md",
        action="create",
        merge_decision_ids=["MD-001"],
        candidate_page_ids=["CP-001"],
        section_order=["摘要"],
        source_ref_rules=["保留本次 raw 的来源引用。"],
        readability_goal="整理成一篇可读的中文知识页。",
    )
    final_prompt = prompts.final_page_prompt(composition_item=item, candidate_pages=candidate_pages, snapshot=snapshot, profile=profile)
    normalized = _normalize_final_pages(
        FinalPages(
            pages=[
                FinalPage(
                    final_page_id="FP-001",
                    target_path="concepts/Concept_Project_Wiki.md",
                    action="create",
                    title="项目知识库",
                    page_type="concept",
                    markdown="# 项目知识库\n\n项目知识库用于沉淀长期知识。",
                    source_refs=[ref],
                )
            ]
        ),
        CompositionPlan(items=[item]),
        snapshot=snapshot,
        operation_id="ING-20260612T000000Z-test",
    )

    assert final_prompt.schema_name == "llmwiki_lite_final_pages"
    assert normalized.pages[0].content_sha256 == sha256_text(normalized.pages[0].markdown)
