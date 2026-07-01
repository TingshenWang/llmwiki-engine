from __future__ import annotations

import json
from threading import Lock
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from llmwiki_engine.cli import app
from llmwiki_engine.lite import embeddings
from llmwiki_engine.lite import prompts
from llmwiki_engine.lite import related as related_logic
from llmwiki_engine.lite.io import read_json, sha256_file, sha256_text, write_json
from llmwiki_engine.lite.models import (
    ArtifactRef,
    CandidateContext,
    CandidateContextHit,
    CandidateContexts,
    CandidatePage,
    CandidatePages,
    ClaimCoverageItem,
    CompositionItem,
    CompositionPlan,
    CoverageJudge,
    DigestCoverageRepair,
    DigestCoverageJudge,
    FinalCoverageRepair,
    FinalPage,
    FinalPages,
    MergeDecision,
    MergePlan,
    OperationManifest,
    PageState,
    PageStateClaim,
    PageStateNode,
    PageUpdatePlan,
    PreimageCoverageItem,
    RawBinding,
    SourceDigest,
    SourceClaim,
    SourceContentUnit,
    SourceRef,
    StepRecord,
    WikiKnowledgeEntry,
    WikiSnapshot,
)
from llmwiki_engine.lite.pipeline import (
    _assert_candidate_pages_chinese,
    _assert_digest_coverage_complete,
    _assert_digest_coverage_judge_valid,
    _assert_final_coverage_judge_chinese,
    _assert_merge_plan_consumes_candidates,
    _assert_merge_plan_chinese,
    _assert_source_digest_chinese,
    _assert_source_digest_granularity,
    _assert_final_coverage_judge_thresholds,
    _build_page_update_plan,
    _canonical_final_markdown,
    _digest_coverage_report,
    _final_coverage_report,
    _normalize_candidate_pages,
    _normalize_final_pages,
    _normalize_merge_plan,
    _page_generation_parallelism,
    _repair_merge_plan_candidate_content_locators,
    _step_candidate_pages,
    _step_composition_plan,
    _step_digest_coverage_judge,
    _step_final_coverage_judge,
    _step_final_pages,
    _step_index_log_write,
    _step_merge_plan,
    _step_page_state_write,
    _step_related_maintenance,
    _step_related_refresh,
    _step_source_digest,
    _source_granularity_stats,
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


def make_claim(ref: SourceRef, claim_id: str = "C-001", text: str = "说明自动化入库的核心知识点。", *, importance: int = 4) -> SourceClaim:
    return SourceClaim(
        claim_id=claim_id,
        text=text,
        kind="concept",
        importance=importance,
        concept_terms=["自动化入库"],
        raw_locator=ref.locator,
        source_refs=[ref],
    )


def make_source_unit(
    ref: SourceRef,
    *,
    content_unit_id: str = "CU-001",
    title: str = "自动化入库",
    path_hint: str = "concepts/Concept_Auto_Ingest.md",
    claim_ids: list[str] | None = None,
    content_scope: str = "覆盖自动化入库的核心内容。",
    content_role: str = "主干",
    absorption_decision: str = "独立成页",
    anchor_unit_id: str | None = None,
    section_hint: str = "核心内容",
    absorption_reason: str = "这是原文中的主干知识对象，适合独立成页。",
) -> SourceContentUnit:
    return SourceContentUnit(
        content_unit_id=content_unit_id,
        title=title,
        content_role=content_role,  # type: ignore[arg-type]
        absorption_decision=absorption_decision,  # type: ignore[arg-type]
        anchor_unit_id=anchor_unit_id or content_unit_id,
        section_hint=section_hint,
        page_type="concept",
        path_hint=path_hint,
        summary=f"{title} 是有来源支撑的知识点。",
        absorption_reason=absorption_reason,
        content_scope=content_scope,
        claim_ids=claim_ids or ["C-001"],
        source_refs=[ref],
    )


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
        candidate_content_locators=["摘要", "核心内容"],
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
        claims=[make_claim(ref, text="English claim")],
        content_units=[
            SourceContentUnit(
                content_unit_id="CU-001",
                title="English Title",
                content_role="主干",
                absorption_decision="独立成页",
                anchor_unit_id="CU-001",
                section_hint="English section",
                page_type="concept",
                path_hint="concepts/Concept_English.md",
                summary="English candidate summary",
                absorption_reason="English reason",
                content_scope="English content scope",
                claim_ids=["C-001"],
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
        claims=[make_claim(ref, text="RAG 先检索，再增强 prompt，最后生成回答。")],
        content_units=[
            SourceContentUnit(
                content_unit_id="CU-001",
                title="RAG（检索增强生成）",
                content_role="主干",
                absorption_decision="独立成页",
                anchor_unit_id="CU-001",
                section_hint="核心流程",
                page_type="concept",
                path_hint="concepts/Concept_RAG.md",
                summary="RAG 是检索增强生成流程。",
                absorption_reason="这是原文主干概念，适合独立成页。",
                content_scope="覆盖 RAG 的定义和流程。",
                claim_ids=["C-001"],
                source_refs=[ref],
            )
        ],
    )

    _assert_source_digest_chinese(digest)


def test_source_digest_retries_chinese_semantic_validation(tmp_path: Path) -> None:
    vault = init_vault(tmp_path / "vault")
    raw = write_raw(vault)
    raw_rel = raw.relative_to(vault).as_posix()
    raw_sha = sha256_file(raw)
    binding = RawBinding(
        raw_path=raw_rel,
        raw_sha256=raw_sha,
        size_bytes=raw.stat().st_size,
        mtime_ns=raw.stat().st_mtime_ns,
        bound_at="2026-06-18T00:00:00Z",
    )
    ref = SourceRef(raw_path=raw_rel, raw_sha256=raw_sha, locator="whole_file")
    invalid_digest = SourceDigest(
        source_raw_path=raw_rel,
        raw_sha256=raw_sha,
        summary="这份材料说明自动化知识库入库流程。",
        key_takeaways=["内容单元必须使用中文用户可读文本。"],
        claims=[make_claim(ref, text="自动化入库会去掉人工审核节点。")],
        content_units=[
            SourceContentUnit(
                content_unit_id="CU-001",
                title="自动化入库",
                content_role="主干",
                absorption_decision="独立成页",
                anchor_unit_id="CU-001",
                section_hint="核心流程",
                page_type="concept",
                path_hint="concepts/Concept_Auto_Ingest.md",
                summary="自动化入库强调去掉人工审核节点。",
                absorption_reason="这是原文主干概念，适合独立成页。",
                content_scope="English content scope",
                claim_ids=["C-001"],
                source_refs=[ref],
            )
        ],
    )
    valid_digest = invalid_digest.model_copy(
        update={
            "content_units": [
                invalid_digest.content_units[0].model_copy(update={"content_scope": "覆盖原文中关于自动化入库去掉人工审核节点的内容。"})
            ]
        }
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
            output = invalid_digest if len(self.requests) == 1 else valid_digest
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
        "raw_abs": raw,
        "raw_rel": raw_rel,
        "raw_binding": binding,
        "profile": load_profile(vault),
        "provider_registry": registry,
        "provider_contexts": {},
    }

    output = _step_source_digest(tmp_path / "run", state)
    digest = state["source_digest"]

    assert isinstance(digest, SourceDigest)
    assert digest.content_units[0].content_scope == "覆盖原文中关于自动化入库去掉人工审核节点的内容。"
    assert len(registry.requests) == 2
    retry_payload = registry.requests[1].user_payload
    assert "content_scope" in retry_payload["validation_error"]
    assert output.counts["content_unit_count"] == 1
    assert output.counts["semantic_retry_count"] == 1
    assert output.counts["api_call_count"] == 2
    assert output.counts["api_success_count"] == 1
    assert output.counts["api_paused_count"] == 1


def test_source_digest_retries_source_only_error_page(tmp_path: Path) -> None:
    vault = init_vault(tmp_path / "vault")
    raw = vault / "raw" / "voice_agents_quickstart.md"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(
        "---\n"
        "title: \"voice_agents_quickstart\"\n"
        "source_url: \"https://cookbook.openai.com/examples/agents_sdk/voice_agents_quickstart\"\n"
        "---\n\n"
        "# voice_agents_quickstart\n\n"
        "Title: Page not found\n\n"
        "URL Source: https://cookbook.openai.com/examples/agents_sdk/voice_agents_quickstart\n\n"
        "Warning: Target URL returned error 404: Not Found\n\n"
        "Markdown Content:\n"
        "# Page not found\n\n"
        "[Home](https://cookbook.openai.com/)\n"
        "[API](https://cookbook.openai.com/api)\n"
        "[Docs](https://cookbook.openai.com/api/docs)\n",
        encoding="utf-8",
    )
    raw_rel = raw.relative_to(vault).as_posix()
    raw_sha = sha256_file(raw)
    binding = RawBinding(
        raw_path=raw_rel,
        raw_sha256=raw_sha,
        size_bytes=raw.stat().st_size,
        mtime_ns=raw.stat().st_mtime_ns,
        bound_at="2026-06-18T00:00:00Z",
    )
    ref = SourceRef(raw_path=raw_rel, raw_sha256=raw_sha, locator="whole_file")
    invalid_digest = SourceDigest(
        source_raw_path=raw_rel,
        raw_sha256=raw_sha,
        summary="原始页面返回 404，没有语音代理快速入门内容。",
        key_takeaways=["页面失效不能提供语音代理教程。"],
        claims=[
            make_claim(
                ref,
                text="OpenAI Cookbook 的 voice_agents_quickstart 页面返回 404。",
                importance=3,
            )
        ],
        content_units=[
            SourceContentUnit(
                content_unit_id="CU-001",
                title="voice_agents_quickstart 页面失效事件",
                content_role="主干",
                absorption_decision="独立成页",
                anchor_unit_id="CU-001",
                section_hint="页面失效",
                page_type="event",
                path_hint="events/Event_voice_agents_quickstart_404.md",
                summary="记录 voice_agents_quickstart 页面返回 404。",
                absorption_reason="这是错误示例，source-only 校验应拦截它。",
                content_scope="覆盖 raw 中关于该页面失效的唯一事实。",
                claim_ids=["C-001"],
                source_refs=[ref],
            )
        ],
        weak_or_noise_items=[{"text": "导航菜单", "reason": "页面只有导航菜单和 Page not found 提示。"}],
    )
    valid_digest = SourceDigest(
        source_raw_path=raw_rel,
        raw_sha256=raw_sha,
        summary="原始页面返回 404，没有可入库知识。",
        key_takeaways=[],
        claims=[],
        content_units=[],
        weak_or_noise_items=[{"text": "404 页面", "reason": "原始链接返回 404，页面只有导航菜单和错误提示，不应写入知识页。"}],
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
            output = invalid_digest if len(self.requests) == 1 else valid_digest
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
        "raw_abs": raw,
        "raw_rel": raw_rel,
        "raw_binding": binding,
        "profile": load_profile(vault),
        "provider_registry": registry,
        "provider_contexts": {},
    }

    output = _step_source_digest(tmp_path / "run", state)
    digest = state["source_digest"]

    assert isinstance(digest, SourceDigest)
    assert digest.claims == []
    assert digest.content_units == []
    assert len(registry.requests) == 2
    assert "只能记录来源" in registry.requests[1].user_payload["validation_error"]
    assert output.counts["content_unit_count"] == 0
    assert output.counts["semantic_retry_count"] == 1


def test_empty_candidate_flow_skips_merge_and_composition_providers(tmp_path: Path) -> None:
    vault = init_vault(tmp_path / "vault")

    class NoProviderRegistry:
        def provider_for(self, step: str) -> ProviderSpec:
            raise AssertionError(f"不应调用 provider：{step}")

        def call_structured(self, step: str, request, output_model):
            raise AssertionError(f"不应调用 provider：{step}")

    state = {
        "candidate_pages": CandidatePages(pages=[]),
        "candidate_contexts": CandidateContexts(
            retrieval_backend="sentence_transformers",
            model="Qwen/Qwen3-Embedding-0.6B",
            input_version="page_card_v2",
            top_k=5,
            knowledge_pool_size=0,
            candidate_page_count=0,
            candidate_pool_hash="empty",
            items=[],
        ),
        "profile": load_profile(vault),
        "provider_registry": NoProviderRegistry(),
        "provider_contexts": {},
    }

    merge_output = _step_merge_plan(tmp_path / "run", state)
    composition_output = _step_composition_plan(tmp_path / "run", state)

    merge_plan = state["merge_plan"]
    composition_plan = state["composition_plan"]
    assert isinstance(merge_plan, MergePlan)
    assert isinstance(composition_plan, CompositionPlan)
    assert merge_plan.decisions == []
    assert merge_plan.action_counts == {"create": 0, "update": 0, "noop": 0}
    assert composition_plan.items == []
    assert merge_output.model_calls == 0
    assert composition_output.model_calls == 0
    assert merge_output.counts["api_call_count"] == 0
    assert composition_output.counts["api_call_count"] == 0


def test_source_digest_rejects_duplicate_claim_assignment() -> None:
    ref = SourceRef(raw_path="raw/example.md", raw_sha256="sha", locator="whole_file")
    digest = SourceDigest(
        source_raw_path="raw/example.md",
        raw_sha256="sha",
        summary="这份材料说明自动化知识库入库流程。",
        key_takeaways=["每个知识点只能被一个内容单元消费。"],
        claims=[
            make_claim(ref, "C-001", "说明自动化入库流程。"),
            make_claim(ref, "C-002", "说明覆盖审查机制。"),
        ],
        content_units=[
            make_source_unit(ref, content_unit_id="CU-001", claim_ids=["C-001"]),
            make_source_unit(ref, content_unit_id="CU-002", claim_ids=["C-001", "C-002"]),
        ],
    )
    stats = _source_granularity_stats("这是一段足够进入知识库的中文内容。" * 500, raw_size_bytes=10000)

    with pytest.raises(PipelineError, match="重复消费"):
        _assert_source_digest_granularity(digest, stats)


def test_source_digest_retries_over_split_granularity(tmp_path: Path) -> None:
    vault = init_vault(tmp_path / "vault")
    raw = write_raw(vault)
    raw_rel = raw.relative_to(vault).as_posix()
    raw_sha = sha256_file(raw)
    binding = RawBinding(
        raw_path=raw_rel,
        raw_sha256=raw_sha,
        size_bytes=raw.stat().st_size,
        mtime_ns=raw.stat().st_mtime_ns,
        bound_at="2026-06-18T00:00:00Z",
    )
    ref = SourceRef(raw_path=raw_rel, raw_sha256=raw_sha, locator="whole_file")

    def unit(index: int, title: str) -> SourceContentUnit:
        return SourceContentUnit(
            content_unit_id=f"CU-{index:03d}",
            title=title,
            content_role="主干",
            absorption_decision="独立成页",
            anchor_unit_id=f"CU-{index:03d}",
            section_hint="核心内容",
            page_type="concept",
            path_hint=f"concepts/Concept_{index}.md",
            summary=f"{title} 是自动化入库流程的一部分。",
            absorption_reason="这是用于触发粒度校验的过细主干规划。",
            content_scope=f"覆盖 {title} 的具体内容。",
            claim_ids=[f"C-{index:03d}"],
            source_refs=[ref],
        )

    invalid_digest = SourceDigest(
        source_raw_path=raw_rel,
        raw_sha256=raw_sha,
        summary="这份材料说明自动化知识库入库流程。",
        key_takeaways=["短文应该合并为粗粒度页面。"],
        claims=[
            make_claim(ref, "C-001", "说明自动化入库。"),
            make_claim(ref, "C-002", "说明去掉人工审核。"),
            make_claim(ref, "C-003", "说明自动写入 Wiki。"),
        ],
        content_units=[unit(1, "自动化入库"), unit(2, "去掉人工审核"), unit(3, "自动写入 Wiki")],
    )
    valid_digest = SourceDigest(
        source_raw_path=raw_rel,
        raw_sha256=raw_sha,
        summary="这份材料说明自动化知识库入库流程。",
        key_takeaways=["短文应该合并为一篇粗粒度页面。"],
        claims=[
            make_claim(ref, "C-001", "说明为什么不再人工审核。"),
            make_claim(ref, "C-002", "说明自动写入 wiki 的基本顺序。"),
        ],
        content_units=[
            SourceContentUnit(
                content_unit_id="CU-001",
                title="自动化知识库入库流程",
                content_role="主干",
                absorption_decision="独立成页",
                anchor_unit_id="CU-001",
                section_hint="核心流程",
                page_type="concept",
                path_hint="concepts/Concept_Auto_Ingest.md",
                summary="自动化知识库入库流程把人工审核、候选页生成和自动写入合并为可运行链路。",
                absorption_reason="这是合并后的主干流程页，适合独立成页。",
                content_scope="覆盖原文中自动化入库流程的核心设计。",
                claim_ids=["C-001", "C-002"],
                source_refs=[ref],
            )
        ],
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
            output = invalid_digest if len(self.requests) == 1 else valid_digest
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
        "raw_abs": raw,
        "raw_rel": raw_rel,
        "raw_binding": binding,
        "profile": load_profile(vault),
        "provider_registry": registry,
        "provider_contexts": {},
    }

    output = _step_source_digest(tmp_path / "run", state)
    digest = state["source_digest"]

    assert isinstance(digest, SourceDigest)
    assert len(digest.content_units) == 1
    assert len(registry.requests) == 2
    assert "candidate_page_count 超出经验粒度区间" in registry.requests[1].user_payload["validation_error"]
    assert "granularity_stats" in registry.requests[0].user_payload
    assert output.counts["semantic_retry_count"] == 1
    assert output.counts["granularity_suggested_max_candidate_pages"] == 1
    assert (tmp_path / "run" / "source_digest" / "granularity.json").exists()


def test_source_digest_allows_zero_content_units_for_explicit_noise() -> None:
    raw_text = "# 404\n\nNavigation\n\nSearch\n\nPage not found"
    stats = _source_granularity_stats(raw_text, raw_size_bytes=len(raw_text.encode()))
    digest = SourceDigest(
        source_raw_path="raw/404.md",
        raw_sha256="abc",
        summary="原文是无法访问的错误页。",
        key_takeaways=[],
        content_units=[],
        weak_or_noise_items=[{"text": "404 页面", "reason": "原始链接返回 404，页面只有导航菜单。"}],
    )

    _assert_source_digest_granularity(digest, stats)


def test_digest_coverage_judge_repairs_incomplete_source_digest(tmp_path: Path) -> None:
    vault = init_vault(tmp_path / "vault")
    raw = vault / "raw" / "project_note.md"
    raw.write_text(
        "# 项目笔记\n\n"
        "自动化入库会在覆盖校验通过后自动写入 wiki。"
        "系统必须保留 raw 文件，方便人类追溯原始信息。\n",
        encoding="utf-8",
    )
    ref = SourceRef(raw_path="raw/project_note.md", raw_sha256=sha256_file(raw), locator="whole_file")
    initial_digest = SourceDigest(
        source_raw_path=ref.raw_path,
        raw_sha256=ref.raw_sha256,
        summary="这份材料说明自动化入库流程。",
        key_takeaways=["自动化入库会自动写入 wiki。"],
        claims=[make_claim(ref, "C-001", "自动化入库会在覆盖校验通过后自动写入 wiki。")],
        content_units=[make_source_unit(ref, claim_ids=["C-001"])],
    )
    repaired_digest = SourceDigest(
        source_raw_path=ref.raw_path,
        raw_sha256=ref.raw_sha256,
        summary="这份材料说明自动化入库和 raw 追溯。",
        key_takeaways=["自动化入库会自动写入 wiki。", "raw 文件用于追溯原始信息。"],
        claims=[
            make_claim(ref, "C-001", "自动化入库会在覆盖校验通过后自动写入 wiki。"),
            make_claim(ref, "C-002", "系统必须保留 raw 文件，方便人类追溯原始信息。"),
        ],
        content_units=[make_source_unit(ref, claim_ids=["C-001", "C-002"])],
    )
    covered_judge = DigestCoverageJudge(
        coverage_items=[
            {
                "item_id": "RI-001",
                "text": "自动化入库会在覆盖校验通过后自动写入 wiki。",
                "importance": 4,
                "status": "covered",
                "covered_by_claim_ids": ["C-001"],
                "covered_by_content_unit_ids": ["CU-001"],
                "raw_locator": "whole_file",
                "evidence": "digest 已记录自动写入 wiki。",
                "reason": "claim 和 content_unit 覆盖了该信息。",
            },
            {
                "item_id": "RI-002",
                "text": "系统必须保留 raw 文件，方便人类追溯原始信息。",
                "importance": 4,
                "status": "covered",
                "covered_by_claim_ids": ["C-002"],
                "covered_by_content_unit_ids": ["CU-001"],
                "raw_locator": "whole_file",
                "evidence": "digest 已补充 raw 追溯要求。",
                "reason": "新增 claim 覆盖了 raw 保留与追溯。",
            },
        ]
    )
    coverage_repair = DigestCoverageRepair(
        coverage_items=covered_judge.coverage_items,
        repaired_source_digest=repaired_digest,
        repair_actions=[
            {
                "item_id": "RI-002",
                "action": "added_claim",
                "claim_ids": ["C-002"],
                "content_unit_ids": ["CU-001"],
                "reason": "补充 raw 保留与追溯的知识点。",
            }
        ],
    )

    class FakeRegistry:
        def __init__(self) -> None:
            self.spec = ProviderSpec(
                spec="openai_compatible:deepseek-v4-flash",
                endpoint="https://api.deepseek.com/v1/chat/completions",
                api_key="test-key",
            )
            self.coverage_repair_calls = 0

        def provider_for(self, step: str) -> ProviderSpec:
            return self.spec

        def call_structured(self, step: str, request, output_model):
            if output_model is DigestCoverageRepair:
                self.coverage_repair_calls += 1
                output = coverage_repair
            else:
                raise AssertionError(output_model)
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
                        "call_index": self.coverage_repair_calls,
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
        "raw_abs": raw,
        "raw_rel": ref.raw_path,
        "raw_binding": RawBinding(
            raw_path=ref.raw_path,
            raw_sha256=ref.raw_sha256,
            size_bytes=raw.stat().st_size,
            mtime_ns=raw.stat().st_mtime_ns,
            bound_at="2026-06-30T00:00:00Z",
        ),
        "source_granularity_stats": _source_granularity_stats(raw.read_text(encoding="utf-8"), raw_size_bytes=raw.stat().st_size),
        "source_digest": initial_digest,
        "profile": load_profile(vault),
        "provider_registry": registry,
        "provider_contexts": {},
    }

    output = _step_digest_coverage_judge(tmp_path / "run", state)

    assert registry.coverage_repair_calls == 1
    assert state["source_digest"].claims[1].claim_id == "C-002"
    assert output.counts["digest_repair_count"] == 1
    assert output.counts["digest_repair_action_count"] == 1
    assert output.counts["digest_verify_count"] == 0
    assert output.counts["digest_raw_coverage_percent"] == 100.0
    assert (tmp_path / "run" / "digest_coverage_judge" / "digest_coverage_report.json").exists()
    assert (tmp_path / "run" / "digest_coverage_judge" / "repaired_source_digest.json").exists()


def test_digest_coverage_complete_rejects_low_importance_gap() -> None:
    ref = SourceRef(raw_path="raw/project_note.md", raw_sha256="abc", locator="whole_file")
    digest = SourceDigest(
        source_raw_path=ref.raw_path,
        raw_sha256=ref.raw_sha256,
        summary="这份材料说明自动化入库流程。",
        key_takeaways=["自动化入库会自动写入 wiki。"],
        claims=[make_claim(ref, "C-001"), make_claim(ref, "C-002")],
        content_units=[make_source_unit(ref, claim_ids=["C-001", "C-002"])],
    )
    judge = DigestCoverageJudge(
        coverage_items=[
            {
                "item_id": "RI-001",
                "text": "自动化入库会自动写入 wiki。",
                "importance": 5,
                "status": "covered",
                "covered_by_claim_ids": ["C-001"],
                "covered_by_content_unit_ids": ["CU-001"],
                "raw_locator": "whole_file",
                "evidence": "digest 已覆盖自动写入 wiki。",
                "reason": "核心信息已覆盖。",
            },
            {
                "item_id": "RI-002",
                "text": "系统会展示一个低价值工具演示。",
                "importance": 1,
                "status": "missing",
                "covered_by_claim_ids": [],
                "covered_by_content_unit_ids": [],
                "raw_locator": "whole_file",
                "evidence": "digest 没有提到工具演示。",
                "reason": "这是低重要度缺口，应进入报告但不应在覆盖率达标时阻断。",
            },
            {
                "item_id": "RI-003",
                "text": "raw 文件用于追溯原始信息。",
                "importance": 5,
                "status": "covered",
                "covered_by_claim_ids": ["C-002"],
                "covered_by_content_unit_ids": ["CU-001"],
                "raw_locator": "whole_file",
                "evidence": "digest 已覆盖 raw 追溯。",
                "reason": "核心信息已覆盖。",
            },
        ]
    )

    report = _digest_coverage_report(judge, digest)

    assert report["digest_raw_coverage_percent"] == 90.91
    with pytest.raises(PipelineError, match="修复后仍存在未完整覆盖"):
        _assert_digest_coverage_complete(report)


def test_digest_coverage_allows_raw_english_evidence() -> None:
    ref = SourceRef(raw_path="raw/paper.md", raw_sha256="abc", locator="results")
    digest = SourceDigest(
        source_raw_path=ref.raw_path,
        raw_sha256=ref.raw_sha256,
        summary="这份材料说明 ReAct 的实验设置。",
        key_takeaways=["ReAct 同时使用推理轨迹和文本动作。"],
        claims=[make_claim(ref, "C-001", "ReAct 在 HotpotQA 中使用 6-shot 设置评估。")],
        content_units=[make_source_unit(ref, claim_ids=["C-001"])],
    )
    judge = DigestCoverageJudge(
        coverage_items=[
            {
                "item_id": "RI-001",
                "text": "ReAct 在 HotpotQA 中使用 6-shot 设置评估。",
                "importance": 4,
                "status": "covered",
                "covered_by_claim_ids": ["C-001"],
                "covered_by_content_unit_ids": ["CU-001"],
                "raw_locator": "results",
                "evidence": "HotpotQA (6-shot): ReAct exact match 27.4.",
                "reason": "英文是 raw 原文短证据，中文字段已经说明覆盖判断。",
            }
        ]
    )

    raw_text = "HotpotQA (6-shot): ReAct exact match 27.4."
    stats = _source_granularity_stats(raw_text, raw_size_bytes=len(raw_text.encode("utf-8")))
    _assert_digest_coverage_judge_valid(judge, digest, raw_text, stats)


def test_final_coverage_judge_writes_quantified_report(tmp_path: Path) -> None:
    ref = SourceRef(raw_path="raw/project_note.md", raw_sha256="abc", locator="whole_file")
    digest = SourceDigest(
        source_raw_path=ref.raw_path,
        raw_sha256=ref.raw_sha256,
        summary="这份材料说明自动化知识库入库流程。",
        key_takeaways=["自动化入库去掉人工审核。"],
        claims=[make_claim(ref, text="自动化入库去掉人工审核并自动写入 wiki。")],
        content_units=[make_source_unit(ref)],
    )
    final_pages = FinalPages(
        pages=[
            FinalPage(
                final_page_id="FP-001",
                target_path="concepts/Concept_Auto_Ingest.md",
                action="create",
                title="自动化入库",
                page_type="concept",
                markdown="# 自动化入库\n\n## 摘要\n\n自动化入库去掉人工审核，并自动写入 wiki。\n",
                source_refs=[ref],
            )
        ]
    )
    judge = CoverageJudge(
        claim_results=[
            ClaimCoverageItem(
                claim_id="C-001",
                status="covered",
                covered_by=["concepts/Concept_Auto_Ingest.md#摘要"],
                evidence="最终页面写明自动化入库去掉人工审核，并自动写入 wiki。",
                reason="正文明确覆盖了 claim 的事实含义。",
            )
        ]
    )
    repair = FinalCoverageRepair(claim_results=judge.claim_results, repaired_final_pages=final_pages, repair_actions=[])
    composition = CompositionPlan(
        items=[
            CompositionItem(
                final_page_id="FP-001",
                target_path="concepts/Concept_Auto_Ingest.md",
                action="create",
                merge_decision_ids=["MD-001"],
                candidate_page_ids=["CP-001"],
                section_order=["摘要"],
                source_ref_rules=["保留 raw 来源引用。"],
                readability_goal="生成可读的中文 wiki 笔记。",
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

        def provider_for(self, step: str) -> ProviderSpec:
            return self.spec

        def call_structured(self, step: str, request, output_model):
            return ProviderCallResult(
                output=repair,
                prompt_artifact={"step": step, "request": request.model_dump(mode="json")},
                provider_result={"parsed": repair.model_dump(mode="json")},
                sanitized_context=self.spec.sanitized_context(),
                api_calls=[
                    {
                        "step": step,
                        "request_key": "",
                        "model": "deepseek-v4-flash",
                        "attempt": 0,
                        "call_index": 1,
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

    state = {
        "source_digest": digest,
        "final_pages": final_pages,
        "composition_plan": composition,
        "wiki_snapshot": WikiSnapshot(wiki_root="wiki", pool_hash="empty", generated_at="2026-06-23T00:00:00Z", entries=[]),
        "provider_registry": FakeRegistry(),
        "provider_contexts": {},
        "operation_id": "ING-test",
    }

    output = _step_final_coverage_judge(tmp_path / "run", state)

    assert output.counts["claim_count"] == 1
    assert output.counts["raw_claim_coverage_percent"] == 100.0
    assert output.counts["core_claim_coverage_percent"] == 100.0
    assert (tmp_path / "run" / "final_coverage_judge" / "final_coverage_report.json").exists()


def test_final_coverage_judge_retries_empty_evidence(tmp_path: Path) -> None:
    ref = SourceRef(raw_path="raw/project_note.md", raw_sha256="abc", locator="whole_file")
    digest = SourceDigest(
        source_raw_path=ref.raw_path,
        raw_sha256=ref.raw_sha256,
        summary="这份材料说明自动化知识库入库流程。",
        key_takeaways=["自动化入库去掉人工审核。"],
        claims=[make_claim(ref, text="自动化入库去掉人工审核并自动写入 wiki。")],
        content_units=[make_source_unit(ref)],
    )
    final_pages = FinalPages(
        pages=[
            FinalPage(
                final_page_id="FP-001",
                target_path="concepts/Concept_Auto_Ingest.md",
                action="create",
                title="自动化入库",
                page_type="concept",
                markdown="# 自动化入库\n\n## 摘要\n\n自动化入库去掉人工审核，并自动写入 wiki。\n",
                source_refs=[ref],
            )
        ]
    )
    invalid_judge = CoverageJudge(
        claim_results=[
            ClaimCoverageItem(
                claim_id="C-001",
                status="covered",
                covered_by=["concepts/Concept_Auto_Ingest.md#摘要"],
                evidence="",
                reason="判断理由有中文，但证据为空。",
            )
        ]
    )
    valid_judge = CoverageJudge(
        claim_results=[
            ClaimCoverageItem(
                claim_id="C-001",
                status="covered",
                covered_by=["concepts/Concept_Auto_Ingest.md#摘要"],
                evidence="最终页面写明自动化入库去掉人工审核，并自动写入 wiki。",
                reason="正文明确覆盖了 claim 的事实含义。",
            )
        ]
    )
    invalid_repair = FinalCoverageRepair(claim_results=invalid_judge.claim_results, repaired_final_pages=final_pages, repair_actions=[])
    valid_repair = FinalCoverageRepair(claim_results=valid_judge.claim_results, repaired_final_pages=final_pages, repair_actions=[])
    composition = CompositionPlan(
        items=[
            CompositionItem(
                final_page_id="FP-001",
                target_path="concepts/Concept_Auto_Ingest.md",
                action="create",
                merge_decision_ids=["MD-001"],
                candidate_page_ids=["CP-001"],
                section_order=["摘要"],
                source_ref_rules=["保留 raw 来源引用。"],
                readability_goal="生成可读的中文 wiki 笔记。",
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
            call_index = len(self.requests)
            output = invalid_repair if call_index == 1 else valid_repair
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
                        "call_index": call_index,
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
        "final_pages": final_pages,
        "composition_plan": composition,
        "wiki_snapshot": WikiSnapshot(wiki_root="wiki", pool_hash="empty", generated_at="2026-06-23T00:00:00Z", entries=[]),
        "provider_registry": registry,
        "provider_contexts": {},
        "operation_id": "ING-test",
    }

    output = _step_final_coverage_judge(tmp_path / "run", state)

    assert len(registry.requests) == 2
    assert "evidence 不能为空" in registry.requests[1].user_payload["validation_error"]
    assert output.counts["semantic_retry_count"] == 1
    assert output.counts["api_call_count"] == 2
    assert output.counts["api_success_count"] == 1
    assert output.counts["api_paused_count"] == 1


def test_final_coverage_judge_threshold_fails_when_core_claim_missing() -> None:
    ref = SourceRef(raw_path="raw/project_note.md", raw_sha256="abc", locator="whole_file")
    digest = SourceDigest(
        source_raw_path=ref.raw_path,
        raw_sha256=ref.raw_sha256,
        summary="这份材料说明自动化知识库入库流程。",
        key_takeaways=["自动化入库去掉人工审核。"],
        claims=[make_claim(ref, text="自动化入库去掉人工审核并自动写入 wiki。")],
        content_units=[make_source_unit(ref)],
    )
    judge = CoverageJudge(
        claim_results=[
            ClaimCoverageItem(
                claim_id="C-001",
                status="missing",
                evidence="最终页面没有写出该知识点。",
                reason="没有找到对应正文。",
            )
        ]
    )
    report = _final_coverage_report(judge, digest)

    with pytest.raises(PipelineError, match="final_coverage_judge 未通过"):
        _assert_final_coverage_judge_thresholds(report)


def test_final_coverage_judge_allows_technical_evidence_fragments() -> None:
    judge = CoverageJudge(
        claim_results=[
            ClaimCoverageItem(
                claim_id="C-001",
                status="covered",
                covered_by=["concepts/Concept_ADK.md#语言支持与安装"],
                evidence=(
                    "Python (`pip install google-adk`)、TypeScript (`npm install @google/adk`)、"
                    "Go (`go get google.golang.org/adk`)、Java (`com.google.adk:google-adk`)、"
                    "Kotlin (`com.google.adk:google-adk-kotlin-core`)。"
                ),
                reason="最终页面列出了各语言对应的安装命令，因此覆盖该知识点。",
            )
        ]
    )

    _assert_final_coverage_judge_chinese(judge)


def test_final_coverage_judge_allows_product_name_evidence_fragments() -> None:
    judge = CoverageJudge(
        claim_results=[
            ClaimCoverageItem(
                claim_id="C-001",
                status="covered",
                covered_by=["concepts/Concept_Model_Context_Protocol.md#预构建集成"],
                evidence="Google Drive、Slack、GitHub、Git、Postgres、Puppeteer",
                reason="最终页面列出了这些预构建集成，因此覆盖该知识点。",
            )
        ]
    )

    _assert_final_coverage_judge_chinese(judge)


def test_final_coverage_judge_rejects_english_explanatory_evidence() -> None:
    judge = CoverageJudge(
        claim_results=[
            ClaimCoverageItem(
                claim_id="C-001",
                status="covered",
                covered_by=["concepts/Concept_ADK.md#语言支持与安装"],
                evidence="The final page describes installation commands for several languages.",
                reason="判断理由是中文，但证据字段本身仍是英文解释句。",
            )
        ]
    )

    with pytest.raises(PipelineError, match="必须使用中文用户可读文本"):
        _assert_final_coverage_judge_chinese(judge)


def test_final_coverage_judge_repairs_missing_claim_in_single_call(tmp_path: Path) -> None:
    vault = init_vault(tmp_path / "vault")
    ref = SourceRef(raw_path="raw/project_note.md", raw_sha256="abc", locator="whole_file")
    digest = SourceDigest(
        source_raw_path=ref.raw_path,
        raw_sha256=ref.raw_sha256,
        summary="这份材料说明自动化知识库入库流程。",
        key_takeaways=["自动化入库会自动写入 wiki，并保留 raw 用于追溯。"],
        claims=[
            make_claim(ref, "C-001", "自动化入库去掉人工审核，并在覆盖校验通过后自动写入 wiki。"),
            make_claim(ref, "C-002", "自动化入库会保留 raw 文件用于追溯原始信息。"),
        ],
        content_units=[make_source_unit(ref, claim_ids=["C-001", "C-002"])],
    )
    candidate_pages = CandidatePages(
        pages=[
            CandidatePage(
                candidate_page_id="CP-001",
                content_unit_id="CU-001",
                title="自动化入库",
                proposed_page_type="concept",
                proposed_path_hint="concepts/Concept_Auto_Ingest.md",
                summary="自动化入库去掉人工审核。",
                body_markdown="## 摘要\n\n自动化入库去掉人工审核，并在覆盖校验通过后自动写入 wiki，同时保留 raw 文件用于追溯。",
                source_refs=[ref],
                confidence=0.9,
            )
        ]
    )
    composition = CompositionPlan(
        items=[
            CompositionItem(
                final_page_id="FP-001",
                target_path="concepts/Concept_Auto_Ingest.md",
                action="create",
                merge_decision_ids=["MD-001"],
                candidate_page_ids=["CP-001"],
                section_order=["摘要", "核心内容"],
                source_ref_rules=["保留 raw 来源引用。"],
                readability_goal="生成可读的中文 wiki 笔记。",
            )
        ]
    )
    initial_final_pages = FinalPages(
        pages=[
            FinalPage(
                final_page_id="FP-001",
                target_path="concepts/Concept_Auto_Ingest.md",
                action="create",
                title="自动化入库",
                page_type="concept",
                markdown="# 自动化入库\n\n## 摘要\n\n自动化入库去掉人工审核。\n",
                source_refs=[ref],
            )
        ]
    )
    missing_judge = CoverageJudge(
        claim_results=[
            ClaimCoverageItem(
                claim_id="C-001",
                status="missing",
                evidence="最终页面没有写出自动写入 wiki。",
                reason="缺少覆盖校验通过后自动写入 wiki 的信息。",
            ),
            ClaimCoverageItem(
                claim_id="C-002",
                status="covered",
                covered_by=["concepts/Concept_Auto_Ingest.md#摘要"],
                evidence="最终页面体现 raw 可用于追溯。",
                reason="raw 追溯信息已覆盖。",
            )
        ]
    )
    newly_exposed_judge = CoverageJudge(
        claim_results=[
            ClaimCoverageItem(
                claim_id="C-001",
                status="covered",
                covered_by=["concepts/Concept_Auto_Ingest.md#摘要"],
                evidence="最终页面写明覆盖校验通过后自动写入 wiki。",
                reason="正文补齐了缺失的 claim。",
            ),
            ClaimCoverageItem(
                claim_id="C-002",
                status="partial",
                covered_by=["concepts/Concept_Auto_Ingest.md#摘要"],
                evidence="最终页面只泛化提到 raw，没有明确保留 raw 文件用于追溯原始信息。",
                reason="raw 追溯信息不够具体。",
            ),
        ]
    )
    covered_judge = CoverageJudge(
        claim_results=[
            ClaimCoverageItem(
                claim_id="C-001",
                status="covered",
                covered_by=["concepts/Concept_Auto_Ingest.md#摘要"],
                evidence="最终页面写明覆盖校验通过后自动写入 wiki。",
                reason="正文补齐了缺失的 claim。",
            ),
            ClaimCoverageItem(
                claim_id="C-002",
                status="covered",
                covered_by=["concepts/Concept_Auto_Ingest.md#追溯"],
                evidence="最终页面写明保留 raw 文件用于追溯原始信息。",
                reason="raw 追溯信息已完整覆盖。",
            )
        ]
    )
    repaired_pages = FinalPages(
        pages=[
            FinalPage(
                final_page_id="FP-001",
                target_path="concepts/Concept_Auto_Ingest.md",
                action="create",
                title="自动化入库",
                page_type="concept",
                markdown="# 自动化入库\n\n## 摘要\n\n自动化入库去掉人工审核，并在覆盖校验通过后自动写入 wiki。\n",
                source_refs=[ref],
            )
        ]
    )
    second_repaired_pages = FinalPages(
        pages=[
            FinalPage(
                final_page_id="FP-001",
                target_path="concepts/Concept_Auto_Ingest.md",
                action="create",
                title="自动化入库",
                page_type="concept",
                markdown="# 自动化入库\n\n## 摘要\n\n自动化入库去掉人工审核，并在覆盖校验通过后自动写入 wiki。\n\n## 追溯\n\n自动化入库会保留 raw 文件用于追溯原始信息。\n",
                source_refs=[ref],
            )
        ]
    )
    final_repair = FinalCoverageRepair(
        claim_results=covered_judge.claim_results,
        repaired_final_pages=second_repaired_pages,
        repair_actions=[
            {
                "claim_id": "C-001",
                "action": "updated_content",
                "final_page_ids": ["FP-001"],
                "reason": "补齐覆盖校验通过后自动写入 wiki 的信息。",
            },
            {
                "claim_id": "C-002",
                "action": "added_content",
                "final_page_ids": ["FP-001"],
                "reason": "补齐保留 raw 文件用于追溯原始信息的表述。",
            },
        ],
    )

    class FakeRegistry:
        def __init__(self) -> None:
            self.spec = ProviderSpec(
                spec="openai_compatible:deepseek-v4-flash",
                endpoint="https://api.deepseek.com/v1/chat/completions",
                api_key="test-key",
            )
            self.coverage_repair_calls = 0

        def provider_for(self, step: str) -> ProviderSpec:
            return self.spec

        def call_structured(self, step: str, request, output_model):
            if output_model is FinalCoverageRepair:
                self.coverage_repair_calls += 1
                output = final_repair
            else:
                raise AssertionError(output_model)
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
                        "call_index": self.coverage_repair_calls,
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
        "final_pages": initial_final_pages,
        "composition_plan": composition,
        "candidate_pages": candidate_pages,
        "wiki_snapshot": WikiSnapshot(wiki_root="wiki", pool_hash="empty", generated_at="2026-06-23T00:00:00Z", entries=[]),
        "profile": load_profile(vault),
        "provider_registry": registry,
        "provider_contexts": {},
        "operation_id": "ING-test",
    }

    output = _step_final_coverage_judge(tmp_path / "run", state)

    assert output.counts["coverage_repair_count"] == 1
    assert output.counts["coverage_repair_action_count"] == 2
    assert output.counts["coverage_repair_final_page_count"] == 1
    assert output.counts["final_verify_count"] == 0
    assert output.counts["raw_claim_coverage_percent"] == 100.0
    assert registry.coverage_repair_calls == 1
    assert "自动写入 wiki" in state["final_pages"].pages[0].markdown
    assert "保留 raw 文件用于追溯原始信息" in state["final_pages"].pages[0].markdown
    assert (tmp_path / "run" / "final_coverage_judge" / "repaired_final_pages.json").exists()


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
                content_unit_id="CU-001",
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
                content_unit_id="CU-001",
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
                content_unit_id="CU-005",
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
                content_unit_id="CU-001",
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


def test_merge_plan_candidate_content_locators_allow_locator_values() -> None:
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
                candidate_content_locators=["O-001", "摘要"],
                reason="候选页包含新的上下文工程说明。",
                source_refs=[ref],
            )
        ],
        action_counts={},
    )

    normalized = _normalize_merge_plan(plan)

    assert normalized.decisions[0].decision_id == "MD-001"
    assert normalized.decisions[0].candidate_content_locators == ["候选内容定位：O-001", "摘要"]
    _assert_merge_plan_chinese(normalized)


def test_merge_plan_missing_candidate_content_locators_are_repaired_from_candidate_page() -> None:
    ref = SourceRef(raw_path="raw/a.md", raw_sha256="abc", locator="whole_file")
    decision = MergeDecision.model_validate(
        {
            "decision_id": "MD-001",
            "candidate_page_id": "CP-001",
            "action": "create",
            "target_path": "concepts/Concept_Context.md",
            "title": "上下文工程",
            "page_type": "concept",
            "content_scope": "写入候选页中关于上下文工程的内容。",
            "reason": "候选页包含新的上下文工程说明。",
            "source_refs": [ref.model_dump(mode="json")],
        }
    )
    plan = MergePlan(decisions=[decision], action_counts={})
    candidate_pages = CandidatePages(
        pages=[
            CandidatePage(
                candidate_page_id="CP-001",
                content_unit_id="CU-001",
                title="上下文工程",
                proposed_page_type="concept",
                proposed_path_hint="concepts/Concept_Context.md",
                summary="上下文工程负责组织模型可用信息。",
                body_markdown="## 摘要\n\n上下文工程负责组织模型可用信息。\n",
                source_refs=[ref],
                confidence=0.8,
            )
        ]
    )
    contexts = CandidateContexts(
        retrieval_backend="sentence_transformers",
        model="Qwen/Qwen3-Embedding-0.6B",
        input_version="test",
        top_k=5,
        knowledge_pool_size=0,
        candidate_page_count=1,
        candidate_pool_hash="pool",
        items=[CandidateContext(candidate_page_id="CP-001", query="上下文工程", hits=[])],
    )

    repaired = _repair_merge_plan_candidate_content_locators(_normalize_merge_plan(plan), candidate_pages)

    assert repaired.decisions[0].candidate_content_locators
    assert "候选内容定位由引擎根据候选页自动补齐。" in repaired.decisions[0].warnings
    _assert_merge_plan_chinese(repaired)
    _assert_merge_plan_consumes_candidates(repaired, candidate_pages, contexts)


def test_candidate_prompts_do_not_receive_wiki_snapshot_before_embedding(tmp_path: Path) -> None:
    vault = init_vault(tmp_path / "vault")
    raw = write_raw(vault)
    ref = SourceRef(raw_path="raw/project_note.md", raw_sha256=sha256_file(raw), locator="whole_file")
    digest = SourceDigest(
        source_raw_path="raw/project_note.md",
        raw_sha256=sha256_file(raw),
        summary="这份材料说明自动化知识库入库流程。",
        key_takeaways=["候选页必须先忠于原文，再进行旧 wiki 召回。"],
        claims=[make_claim(ref, text="候选页必须先忠于原文，再进行旧 wiki 召回。")],
        content_units=[make_source_unit(ref, content_scope="覆盖原文中关于自动化入库的流程说明。")],
    )
    unit = make_source_unit(ref, content_scope="覆盖原文中关于自动化入库的流程说明。")
    profile = load_profile(vault)

    page_prompt = prompts.candidate_page_prompt(
        digest=digest,
        anchor_unit=unit,
        attached_units=[unit],
        raw_path=ref.raw_path,
        raw_sha256=ref.raw_sha256,
        raw_text=raw.read_text(encoding="utf-8"),
        profile=profile,
    )
    candidate_pages = CandidatePages(
        pages=[
            CandidatePage(
                candidate_page_id="CP-001",
                content_unit_id="CU-001",
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

    assert "wiki_snapshot_entries" not in page_prompt.user_payload
    assert "wiki_snapshot" not in merge_plan_prompt.user_payload
    assert "wiki_snapshot_entries" not in merge_plan_prompt.user_payload
    assert page_prompt.user_payload["anchor_content_unit"]["content_unit_id"] == "CU-001"
    assert page_prompt.user_payload["attached_content_units"][0]["content_unit_id"] == "CU-001"
    assert page_prompt.cache_prefix_payload
    assert page_prompt.cache_prefix_payload["raw_text"]


def test_candidate_pages_retries_only_invalid_content_unit(tmp_path: Path) -> None:
    vault = init_vault(tmp_path / "vault")
    raw = write_raw(vault)
    raw_rel = raw.relative_to(vault).as_posix()
    raw_sha = sha256_file(raw)
    ref = SourceRef(raw_path=raw_rel, raw_sha256=raw_sha, locator="whole_file")
    digest = SourceDigest(
        source_raw_path=raw_rel,
        raw_sha256=raw_sha,
        summary="这份材料说明自动化知识库入库流程。",
        key_takeaways=["候选页生成应该按内容单元局部重试。"],
        claims=[
            make_claim(ref, "C-001", "说明为什么不再人工审核。"),
            make_claim(ref, "C-002", "说明向量缓存服务下一轮召回。"),
        ],
        content_units=[
            make_source_unit(ref, content_scope="覆盖原文中关于自动化入库的流程说明。"),
            make_source_unit(
                ref,
                content_unit_id="CU-002",
                title="向量缓存",
                path_hint="concepts/Concept_Vector_Cache.md",
                content_scope="覆盖原文中关于写入后刷新向量缓存的说明。",
                claim_ids=["C-002"],
            ),
        ],
    )
    units = [
        make_source_unit(ref, content_scope="覆盖原文中关于自动化入库的流程说明。"),
        make_source_unit(
            ref,
            content_unit_id="CU-002",
            title="向量缓存",
            path_hint="concepts/Concept_Vector_Cache.md",
            content_scope="覆盖原文中关于写入后刷新向量缓存的说明。",
            claim_ids=["C-002"],
        ),
    ]

    def page(unit: SourceContentUnit, suffix: str = "") -> CandidatePage:
        return CandidatePage(
            candidate_page_id=f"MODEL-{unit.content_unit_id}{suffix}",
            content_unit_id=unit.content_unit_id,
            title=unit.title,
            proposed_page_type=unit.page_type,
            proposed_path_hint=unit.path_hint,
            summary=unit.summary,
            body_markdown=f"# {unit.title}\n\n## 摘要\n\n{unit.summary}\n\n## 核心内容\n\n这是一页中文候选知识页。",
            source_refs=[ref],
            evidence_notes=["来源定位：整篇材料。"],
            confidence=0.8,
        )

    invalid_cm001 = CandidatePages(pages=[page(units[0], "-A"), page(units[0], "-B")])
    valid_cm001 = CandidatePages(pages=[page(units[0])])
    valid_cm002 = CandidatePages(pages=[page(units[1])])

    class FakeRegistry:
        def __init__(self) -> None:
            self.spec = ProviderSpec(
                spec="openai_compatible:deepseek-v4-flash",
                endpoint="https://api.deepseek.com/v1/chat/completions",
                api_key="test-key",
            )
            self.lock = Lock()
            self.calls_by_unit: dict[str, int] = {}
            self.requests_by_unit: dict[str, list] = {}

        def provider_for(self, step: str) -> ProviderSpec:
            return self.spec

        def call_structured(self, step: str, request, output_model):
            unit_id = request.user_payload["anchor_content_unit"]["content_unit_id"]
            with self.lock:
                call_index = sum(self.calls_by_unit.values()) + 1
                count = self.calls_by_unit.get(unit_id, 0) + 1
                self.calls_by_unit[unit_id] = count
                self.requests_by_unit.setdefault(unit_id, []).append(request)
            output = invalid_cm001 if unit_id == "CU-001" and count == 1 else valid_cm001 if unit_id == "CU-001" else valid_cm002
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
                        "call_index": call_index,
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
        "raw_abs": raw,
        "raw_rel": raw_rel,
        "raw_binding": RawBinding(
            raw_path=raw_rel,
            raw_sha256=raw_sha,
            size_bytes=raw.stat().st_size,
            mtime_ns=raw.stat().st_mtime_ns,
            bound_at="2026-06-18T00:00:00Z",
        ),
        "profile": load_profile(vault),
        "provider_registry": registry,
        "provider_contexts": {},
    }

    output = _step_candidate_pages(tmp_path / "run", state)
    artifact = state["candidate_pages"]

    assert isinstance(artifact, CandidatePages)
    assert [page.content_unit_id for page in artifact.pages] == ["CU-001", "CU-002"]
    assert output.counts["covered_content_unit_count"] == 2
    assert registry.calls_by_unit == {"CU-001": 2, "CU-002": 1}
    retry_payload = registry.requests_by_unit["CU-001"][1].user_payload
    assert "必须且只能返回 1 页" in retry_payload["validation_error"]
    assert output.counts["semantic_retry_count"] == 1
    assert output.counts["api_call_count"] == 3
    assert output.counts["api_success_count"] == 2
    assert output.counts["api_paused_count"] == 1


def test_merge_plan_retries_semantic_validation(tmp_path: Path) -> None:
    vault = init_vault(tmp_path / "vault")
    ref = SourceRef(raw_path="raw/project_note.md", raw_sha256="abc", locator="whole_file")
    candidate_pages = CandidatePages(
        pages=[
            CandidatePage(
                candidate_page_id="CP-001",
                content_unit_id="CU-001",
                title="自动化入库",
                proposed_page_type="concept",
                proposed_path_hint="concepts/Concept_Auto_Ingest.md",
                summary="候选页包含可更新旧页的内容。",
                body_markdown="## 摘要\n\n候选页包含可更新旧页的内容。",
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
    digest = SourceDigest(
        source_raw_path=ref.raw_path,
        raw_sha256=ref.raw_sha256,
        summary="这份材料说明自动化知识库入库流程。",
        key_takeaways=["候选合并后需要用召回结果判断更新或新建。"],
    )
    snapshot = WikiSnapshot(
        wiki_root="wiki",
        pool_hash="pool",
        generated_at="2026-06-18T00:00:00Z",
        entries=[
            WikiKnowledgeEntry(
                path="concepts/Concept_Old.md",
                title="旧页",
                page_type="concept",
                sha256="old",
                summary="旧页已有部分内容。",
                text_excerpt="旧页已有部分内容。",
            )
        ],
    )
    invalid_plan = MergePlan(
        action_counts={"create": 0, "update": 1, "noop": 0},
        decisions=[
            make_decision(
                decision_id="MD-001",
                candidate_page_id="CP-001",
                action="update",
                target_path="concepts/Concept_Old.md",
                title="旧页",
                ref=ref,
                matched_existing_paths=[],
            )
        ],
    )
    valid_plan = MergePlan(
        action_counts={"create": 0, "update": 1, "noop": 0},
        decisions=[
            make_decision(
                decision_id="MD-001",
                candidate_page_id="CP-001",
                action="update",
                target_path="concepts/Concept_Old.md",
                title="旧页",
                ref=ref,
                matched_existing_paths=["concepts/Concept_Old.md"],
            )
        ],
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
        "candidate_pages": candidate_pages,
        "source_digest": digest,
        "wiki_snapshot": snapshot,
        "candidate_contexts": contexts,
        "profile": load_profile(vault),
        "provider_registry": registry,
        "provider_contexts": {},
    }

    output = _step_merge_plan(tmp_path / "run", state)
    plan = state["merge_plan"]

    assert isinstance(plan, MergePlan)
    assert plan.decisions[0].matched_existing_paths == ["concepts/Concept_Old.md"]
    assert len(registry.requests) == 2
    retry_payload = registry.requests[1].user_payload
    assert retry_payload["allowed_update_targets_by_candidate_page"]["CP-001"] == ["concepts/Concept_Old.md"]
    assert "matched_existing_paths" in retry_payload["validation_error"]
    assert output.counts["semantic_retry_count"] == 1
    assert output.counts["api_call_count"] == 2
    assert output.counts["api_success_count"] == 1
    assert output.counts["api_paused_count"] == 1


def test_merge_plan_retries_high_overlap_create_only(tmp_path: Path) -> None:
    vault = init_vault(tmp_path / "vault")
    ref = SourceRef(raw_path="raw/project_note.md", raw_sha256="abc", locator="whole_file")
    candidate_pages = CandidatePages(
        pages=[
            CandidatePage(
                candidate_page_id="CP-001",
                content_unit_id="CU-001",
                title="OpenAI Agents SDK 概览",
                proposed_page_type="overview",
                proposed_path_hint="overviews/Overview_OpenAI_Agents_SDK.md",
                summary="候选页与旧的 Agents SDK 概览高度重叠。",
                body_markdown="## 摘要\n\n候选页与旧的 Agents SDK 概览高度重叠，但补充了新工具链信息。",
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
                query="OpenAI Agents SDK 概览",
                hits=[
                    CandidateContextHit(
                        path="overviews/Overview_OpenAI_Agents_SDK.md",
                        title="OpenAI Agents SDK 概览",
                        rank=1,
                        score=0.86,
                        reason="候选页与旧页高度相近。",
                    )
                ],
            )
        ],
    )
    invalid_plan = MergePlan(
        action_counts={"create": 1, "update": 0, "noop": 0},
        decisions=[
            make_decision(
                decision_id="MD-001",
                candidate_page_id="CP-001",
                action="create",
                target_path="overviews/Overview_OpenAI_Agents_SDK_New.md",
                title="OpenAI Agents SDK 新概览",
                ref=ref,
            )
        ],
    )
    valid_plan = MergePlan(
        action_counts={"create": 0, "update": 1, "noop": 0},
        decisions=[
            make_decision(
                decision_id="MD-001",
                candidate_page_id="CP-001",
                action="update",
                target_path="overviews/Overview_OpenAI_Agents_SDK.md",
                title="OpenAI Agents SDK 概览",
                ref=ref,
                matched_existing_paths=["overviews/Overview_OpenAI_Agents_SDK.md"],
            )
        ],
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
        "candidate_pages": candidate_pages,
        "candidate_contexts": contexts,
        "profile": load_profile(vault),
        "provider_registry": registry,
        "provider_contexts": {},
    }

    output = _step_merge_plan(tmp_path / "run", state)
    plan = state["merge_plan"]

    assert isinstance(plan, MergePlan)
    assert plan.decisions[0].action == "update"
    assert plan.decisions[0].target_path == "overviews/Overview_OpenAI_Agents_SDK.md"
    assert len(registry.requests) == 2
    assert "不能只 create" in registry.requests[1].user_payload["validation_error"]
    assert output.counts["semantic_retry_count"] == 1


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
        claims=[make_claim(ref, text="候选页必须先忠于原文，再进行旧 wiki 召回。")],
        content_units=[make_source_unit(ref, content_scope="覆盖原文中关于自动化入库的流程说明。")],
    )
    unit = make_source_unit(ref, content_scope="覆盖原文中关于自动化入库的流程说明。")
    profile = load_profile(vault)
    spec = ProviderSpec(spec="openai_compatible:deepseek-v4-flash", endpoint="https://api.deepseek.com/v1/chat/completions", api_key="test-key")

    warmup_prompt = prompts.candidate_pages_warmup_prompt(digest=digest, raw_path=ref.raw_path, raw_sha256=raw_sha, raw_text=raw_text, profile=profile)
    page_prompt = prompts.candidate_page_prompt(
        digest=digest,
        anchor_unit=unit,
        attached_units=[unit],
        raw_path=ref.raw_path,
        raw_sha256=raw_sha,
        raw_text=raw_text,
        profile=profile,
    )
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
    assert "anchor_content_unit" not in shared_prefix
    assert "attached_content_units" not in shared_prefix
    assert variable_suffix["input"]["anchor_content_unit"]["content_unit_id"] == "CU-001"
    assert variable_suffix["input"]["attached_content_units"][0]["content_unit_id"] == "CU-001"
    assert "raw_text" not in variable_suffix["input"]
    assert warmup_suffix["input"]["warmup"] is True
    assert warmup_suffix["json_output_example"] == {"status": "OK"}


def test_page_state_guides_update_and_is_refreshed_after_write(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault)
    target_path = "concepts/Concept_Auto_Ingest.md"
    ref = SourceRef(raw_path="raw/project_note.md", raw_sha256="abc", locator="whole_file")
    old_state = PageState(
        page_path=target_path,
        title="自动化入库",
        page_type="concept",
        page_sha256="old-sha",
        updated_at="2026-06-30T00:00:00Z",
        source_raw_paths=["raw/old.md"],
        source_operation_ids=["ING-OLD"],
        nodes=[
            PageStateNode(
                node_id="ING-OLD:CU-001",
                title="自动化入库",
                kind="主干",
                section_hint="核心内容",
                source_content_unit_ids=["CU-001"],
                claim_ids=["ING-OLD:C-001"],
            )
        ],
        claims=[
            PageStateClaim(
                state_claim_id="ING-OLD:C-001",
                source_claim_id="C-001",
                text="自动化入库需要保留 raw 以便追溯。",
                importance=5,
                node_id="ING-OLD:CU-001",
                source_content_unit_id="CU-001",
                source_operation_id="ING-OLD",
                source_refs=[ref],
                coverage_policy="必须保留",
                current_anchor="核心内容",
            )
        ],
    )
    write_json(vault / ".llmwiki" / "page_state" / f"{target_path}.json", old_state)

    new_claim = make_claim(ref, claim_id="C-002", text="自动化入库可以通过 page_state 防止旧内容被无声删除。", importance=4)
    branch_claim = make_claim(ref, claim_id="C-003", text="低价值实现细节应降级为段落，不应该撑开新页面。", importance=2)
    digest = SourceDigest(
        source_raw_path="raw/project_note.md",
        raw_sha256="abc",
        summary="本文补充自动化入库的页面状态维护方式。",
        key_takeaways=["page_state 用于防止 update 丢失旧内容。"],
        claims=[new_claim, branch_claim],
        content_units=[
            make_source_unit(ref, claim_ids=["C-002"], content_scope="补充 page_state 防丢逻辑。"),
            make_source_unit(
                ref,
                content_unit_id="CU-002",
                title="低价值实现细节",
                claim_ids=["C-003"],
                content_scope="把低价值实现细节压缩进主干段落。",
                content_role="附属",
                absorption_decision="降级为段落",
                anchor_unit_id="CU-001",
                section_hint="实现边界",
                absorption_reason="这是主干知识的补充说明，不适合独立成页。",
            ),
        ],
    )
    candidate_pages = CandidatePages(
        pages=[
            CandidatePage(
                candidate_page_id="CP-001",
                content_unit_id="CU-001",
                title="自动化入库",
                proposed_page_type="concept",
                proposed_path_hint=target_path,
                summary="补充 page_state 防丢逻辑。",
                body_markdown="## 核心内容\n\npage_state 用于防止 update 丢失旧内容。\n",
                source_refs=[ref],
                confidence=0.8,
            )
        ]
    )
    composition = CompositionPlan(
        items=[
            CompositionItem(
                final_page_id="FP-001",
                target_path=target_path,
                action="update",
                merge_decision_ids=["MD-001"],
                candidate_page_ids=["CP-001"],
                section_order=["摘要", "核心内容"],
                preserve_rules=[],
                insert_rules=["补充 page_state 防丢逻辑。"],
                delete_rules=[],
                source_ref_rules=["保留 raw 来源引用。"],
                readability_goal="更新自动化入库页面。",
            )
        ]
    )

    plan = _build_page_update_plan(vault, composition, candidate_pages, digest, operation_id="ING-NEW")
    item = plan.items[0]
    assert item.state_exists is True
    assert item.previous_active_claim_count == 1
    assert item.previous_active_claims[0].state_claim_id == "ING-OLD:C-001"
    assert [unit.content_unit_id for unit in item.incoming_content_units] == ["CU-001", "CU-002"]
    assert [claim.claim_id for claim in item.incoming_claims] == ["C-002", "C-003"]
    prompt = prompts.final_page_prompt(
        composition_item=composition.items[0],
        candidate_pages=candidate_pages,
        snapshot=WikiSnapshot(
            wiki_root="wiki",
            pool_hash="pool",
            generated_at="2026-06-30T00:00:00Z",
            entries=[
                WikiKnowledgeEntry(
                    path=target_path,
                    title="自动化入库",
                    page_type="concept",
                    sha256="old-sha",
                    summary="自动化入库需要保留 raw。",
                    text_excerpt="# 自动化入库\n\n## 核心内容\n\n自动化入库需要保留 raw 以便追溯。\n",
                )
            ],
        ),
        profile=load_profile(vault),
        page_update_plan_item=item,
    )
    assert prompt.user_payload["preimage_coverage_requirements"] == []
    assert prompt.user_payload["page_update_plan"]["previous_active_claim_count"] == 1

    final_page = FinalPage(
        final_page_id="FP-001",
        target_path=target_path,
        action="update",
        title="自动化入库",
        page_type="concept",
        markdown="# 自动化入库\n\n## 核心内容\n\n自动化入库需要保留 raw，也可以通过 page_state 防止旧内容被无声删除。\n",
        source_refs=[ref],
    )
    output = _step_page_state_write(
        vault,
        tmp_path / "run",
        {"final_pages": FinalPages(pages=[final_page]), "page_update_plan": plan, "operation_id": "ING-NEW"},
    )
    assert output.counts["page_state_written_count"] == 1
    refreshed = PageState.model_validate(read_json(vault / ".llmwiki" / "page_state" / f"{target_path}.json"))
    assert [claim.state_claim_id for claim in refreshed.claims] == ["ING-OLD:C-001", "ING-NEW:C-002", "ING-NEW:C-003"]
    assert {node.node_id for node in refreshed.nodes} == {"ING-OLD:CU-001", "ING-NEW:CU-001", "ING-NEW:CU-002"}
    branch_state_claim = next(claim for claim in refreshed.claims if claim.state_claim_id == "ING-NEW:C-003")
    assert branch_state_claim.coverage_policy == "低价值"
    assert refreshed.source_operation_ids == ["ING-OLD", "ING-NEW"]


def test_merge_plan_allows_split_decisions_but_update_must_use_top5() -> None:
    ref = SourceRef(raw_path="raw/a.md", raw_sha256="abc", locator="whole_file")
    candidate_pages = CandidatePages(
        pages=[
            CandidatePage(
                candidate_page_id="CP-001",
                content_unit_id="CU-001",
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


def test_related_refresh_keeps_one_related_and_excludes_body_links(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ref = SourceRef(raw_path="raw/a.md", raw_sha256="abc", locator="whole_file")
    page = FinalPage(
        final_page_id="FP-001",
        target_path="concepts/Concept_New.md",
        action="create",
        title="新页",
        page_type="concept",
        markdown="# 新页\n\n正文已经链接 [[concepts/Concept_Body|正文旧页]]，所以 Related 不能重复它。",
        source_refs=[ref],
    )
    snapshot = WikiSnapshot(
        wiki_root="wiki",
        pool_hash="pool",
        generated_at="2026-06-22T00:00:00Z",
        entries=[
            WikiKnowledgeEntry(path="concepts/Concept_Body.md", title="正文旧页", page_type="concept", sha256="body", summary="正文已链接。", text_excerpt="正文已链接。"),
            WikiKnowledgeEntry(path="concepts/Concept_Related.md", title="相关旧页", page_type="concept", sha256="related", summary="应该成为 Related。", text_excerpt="应该成为 Related。"),
        ],
    )
    state = {
        "final_pages": FinalPages(pages=[page]),
        "wiki_snapshot": snapshot,
        "embedding_config": embeddings.EmbeddingConfig(dimensions=2),
        "embedding_page_records": {
            "concepts/Concept_Body.md": {"vector": [1.0, 0.0]},
            "concepts/Concept_Related.md": {"vector": [0.9, 0.1]},
        },
        "config": {},
    }

    def fake_embed_texts(texts, config, *, is_query):
        assert is_query is False
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr("llmwiki_engine.lite.pipeline.embed_texts", fake_embed_texts)

    output = _step_related_refresh(tmp_path / "run", state)
    refreshed = state["final_pages"]
    report = state["related_refresh_report"]

    assert isinstance(refreshed, FinalPages)
    assert "[[concepts/Concept_Related|相关旧页]]" in refreshed.pages[0].markdown
    assert "全文向量相似度" not in refreshed.pages[0].markdown
    assert output.counts["related_link_count"] == 1
    assert any(item.target_path == "concepts/Concept_Body.md" and item.reject_reason == "already_body_link" for item in report.candidates)
    assert any(item.target_path == "concepts/Concept_Related.md" and item.decision == "kept" for item in report.candidates)


def test_related_refresh_omits_related_below_threshold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ref = SourceRef(raw_path="raw/a.md", raw_sha256="abc", locator="whole_file")
    page = FinalPage(
        final_page_id="FP-001",
        target_path="concepts/Concept_New.md",
        action="create",
        title="新页",
        page_type="concept",
        markdown="# 新页\n\n正文没有足够相似的页面。",
        source_refs=[ref],
    )
    old_entry = WikiKnowledgeEntry(
        path="concepts/Concept_Old.md",
        title="旧页",
        page_type="concept",
        sha256="oldsha",
        summary="相似度不足。",
        text_excerpt="相似度不足。",
    )
    state = {
        "final_pages": FinalPages(pages=[page]),
        "wiki_snapshot": WikiSnapshot(wiki_root="wiki", pool_hash="pool", generated_at="2026-06-22T00:00:00Z", entries=[old_entry]),
        "embedding_config": embeddings.EmbeddingConfig(dimensions=2),
        "embedding_page_records": {"concepts/Concept_Old.md": {"vector": [0.6, 0.8]}},
        "config": {},
    }

    monkeypatch.setattr("llmwiki_engine.lite.pipeline.embed_texts", lambda texts, config, *, is_query: [[1.0, 0.0] for _ in texts])

    output = _step_related_refresh(tmp_path / "run", state)
    refreshed = state["final_pages"]

    assert isinstance(refreshed, FinalPages)
    assert "## 相关页面" not in refreshed.pages[0].markdown
    assert output.counts["related_link_count"] == 0


def test_related_maintenance_replaces_old_related_when_new_page_is_stronger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    vault = init_vault(tmp_path / "vault")
    wiki = vault / "wiki"
    (wiki / "concepts").mkdir(parents=True, exist_ok=True)
    old_path = wiki / "concepts/Concept_AI_PM职业路径与作品集策略.md"
    weak_path = wiki / "concepts/Concept_AI产品经理类型.md"
    new_path = wiki / "concepts/Concept_AI_PM职业路径规划与求职.md"
    old_path.write_text(
        "# AI PM职业路径与作品集策略\n\n"
        "正文保持不变。\n\n"
        "## 相关页面\n\n"
        "- [[concepts/Concept_AI产品经理类型|AI产品经理类型]]\n",
        encoding="utf-8",
    )
    weak_path.write_text("# AI产品经理类型\n\n岗位类型分类。\n", encoding="utf-8")
    new_path.write_text("# AI PM职业路径规划与求职\n\n求职路径规划与作品集策略。\n", encoding="utf-8")
    ref = SourceRef(raw_path="raw/a.md", raw_sha256="abc", locator="whole_file")
    final_page = FinalPage(
        final_page_id="FP-001",
        target_path="concepts/Concept_AI_PM职业路径规划与求职.md",
        action="create",
        title="AI PM职业路径规划与求职",
        page_type="concept",
        markdown=new_path.read_text(encoding="utf-8"),
        source_refs=[ref],
    )
    state = {
        "profile": load_profile(vault),
        "final_pages": FinalPages(pages=[final_page]),
        "embedding_config": embeddings.EmbeddingConfig(dimensions=2),
        "embedding_page_records": {
            "concepts/Concept_AI_PM职业路径与作品集策略.md": {"vector": [1.0, 0.0]},
            "concepts/Concept_AI产品经理类型.md": {"vector": [0.75, 0.6614]},
            "concepts/Concept_AI_PM职业路径规划与求职.md": {"vector": [0.95, 0.3122]},
        },
        "config": {"related": {"min_similarity": 0.72, "replacement_margin": 0.04}},
        "write_set_items": [],
        "written_targets": [final_page.target_path],
        "embedding_refresh_metrics": {},
    }

    monkeypatch.setattr(
        "llmwiki_engine.lite.pipeline.sync_page_embedding_cache",
        lambda vault, entries, config: (state["embedding_page_records"], {"cache_hit": len(entries), "cache_refreshed": 0}),
    )

    output = _step_related_maintenance(vault, tmp_path / "run", state)
    updated = old_path.read_text(encoding="utf-8")
    report = json.loads((tmp_path / "run" / "related_maintenance" / "related_maintenance_report.json").read_text(encoding="utf-8"))

    assert "[[concepts/Concept_AI_PM职业路径规划与求职|AI PM职业路径规划与求职]]" in updated
    assert "全文向量相似度" not in updated
    assert "正文保持不变。" in updated
    assert output.counts["related_maintenance_replaced_count"] == 1
    assert "concepts/Concept_AI_PM职业路径与作品集策略.md" in state["written_targets"]
    assert report["checked_pages"][0]["action"] == "replace"


def test_related_maintenance_keeps_existing_when_margin_is_too_small(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    vault = init_vault(tmp_path / "vault")
    wiki = vault / "wiki"
    (wiki / "concepts").mkdir(parents=True, exist_ok=True)
    old_path = wiki / "concepts/Concept_Old.md"
    weak_path = wiki / "concepts/Concept_Current.md"
    new_path = wiki / "concepts/Concept_New.md"
    original = (
        "# 旧页\n\n"
        "正文保持不变。\n\n"
        "## 相关页面\n\n"
        "- [[concepts/Concept_Current|当前相关]]\n"
    )
    old_path.write_text(original, encoding="utf-8")
    weak_path.write_text("# 当前相关\n\n已有相关页面。\n", encoding="utf-8")
    new_path.write_text("# 新页\n\n略强一点但没有明显更强。\n", encoding="utf-8")
    ref = SourceRef(raw_path="raw/a.md", raw_sha256="abc", locator="whole_file")
    final_page = FinalPage(
        final_page_id="FP-001",
        target_path="concepts/Concept_New.md",
        action="create",
        title="新页",
        page_type="concept",
        markdown=new_path.read_text(encoding="utf-8"),
        source_refs=[ref],
    )
    state = {
        "profile": load_profile(vault),
        "final_pages": FinalPages(pages=[final_page]),
        "embedding_config": embeddings.EmbeddingConfig(dimensions=2),
        "embedding_page_records": {
            "concepts/Concept_Old.md": {"vector": [1.0, 0.0]},
            "concepts/Concept_Current.md": {"vector": [0.78, 0.6258]},
            "concepts/Concept_New.md": {"vector": [0.80, 0.6]},
        },
        "config": {"related": {"min_similarity": 0.72, "replacement_margin": 0.04}},
        "write_set_items": [],
        "written_targets": [final_page.target_path],
        "embedding_refresh_metrics": {},
    }

    monkeypatch.setattr(
        "llmwiki_engine.lite.pipeline.sync_page_embedding_cache",
        lambda vault, entries, config: (state["embedding_page_records"], {"cache_hit": len(entries), "cache_refreshed": 0}),
    )

    output = _step_related_maintenance(vault, tmp_path / "run", state)
    report = json.loads((tmp_path / "run" / "related_maintenance" / "related_maintenance_report.json").read_text(encoding="utf-8"))

    assert old_path.read_text(encoding="utf-8") == original
    assert output.counts["related_maintenance_replaced_count"] == 0
    old_row = next(item for item in report["checked_pages"] if item["path"] == "concepts/Concept_Old.md")
    assert old_row["action"] == "keep_existing"


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
        _canonical_final_markdown(page, operation_id="OP-GRAPH")


def test_final_pages_retries_only_failed_page_semantic_validation(tmp_path: Path) -> None:
    vault = init_vault(tmp_path / "vault")
    ref = SourceRef(raw_path="raw/project_note.md", raw_sha256="abc", locator="whole_file")
    candidate_pages = CandidatePages(
        pages=[
            CandidatePage(
                candidate_page_id="CP-001",
                content_unit_id="CU-001",
                title="链接契约",
                proposed_page_type="concept",
                proposed_path_hint="concepts/Concept_Link_Contract.md",
                summary="候选页说明最终页不能链接 raw。",
                body_markdown="## 摘要\n\n最终页不能链接 raw。",
                source_refs=[ref],
                confidence=0.9,
            ),
            CandidatePage(
                candidate_page_id="CP-002",
                content_unit_id="CU-002",
                title="稳定写作",
                proposed_page_type="concept",
                proposed_path_hint="concepts/Concept_Stable_Writing.md",
                summary="候选页说明稳定写作。",
                body_markdown="## 摘要\n\n稳定写作需要中文正文。",
                source_refs=[ref],
                confidence=0.9,
            ),
        ]
    )
    composition = CompositionPlan(
        items=[
            CompositionItem(
                final_page_id="FP-001",
                target_path="concepts/Concept_Link_Contract.md",
                action="create",
                merge_decision_ids=["MD-001"],
                candidate_page_ids=["CP-001"],
                section_order=["摘要"],
                source_ref_rules=["保留 raw 来源引用。"],
                readability_goal="生成中文知识页。",
            ),
            CompositionItem(
                final_page_id="FP-002",
                target_path="concepts/Concept_Stable_Writing.md",
                action="create",
                merge_decision_ids=["MD-002"],
                candidate_page_ids=["CP-002"],
                section_order=["摘要"],
                source_ref_rules=["保留 raw 来源引用。"],
                readability_goal="生成中文知识页。",
            ),
        ]
    )
    snapshot = WikiSnapshot(wiki_root="wiki", pool_hash="empty", generated_at="2026-06-18T00:00:00Z", entries=[])

    class FakeRegistry:
        def __init__(self) -> None:
            self.spec = ProviderSpec(
                spec="openai_compatible:deepseek-v4-flash",
                endpoint="https://api.deepseek.com/v1/chat/completions",
                api_key="test-key",
            )
            self.lock = Lock()
            self.requests: list[tuple[str, bool]] = []

        def provider_for(self, step: str) -> ProviderSpec:
            return self.spec

        def call_structured(self, step: str, request, output_model):
            final_page_id = request.user_payload["composition_item"]["final_page_id"]
            target_path = request.user_payload["composition_item"]["target_path"]
            is_retry = "validation_error" in request.user_payload
            with self.lock:
                self.requests.append((final_page_id, is_retry))
                call_index = len(self.requests)
            if final_page_id == "FP-001" and not is_retry:
                markdown = "# 链接契约\n\n## 摘要\n\n这段错误地链接到 [raw](raw/project_note.md)。"
            else:
                title = "链接契约" if final_page_id == "FP-001" else "稳定写作"
                markdown = f"# {title}\n\n## 摘要\n\n这是通过校验的中文最终页面。"
            page = FinalPage(
                final_page_id=final_page_id,
                target_path=target_path,
                action="create",
                title="链接契约" if final_page_id == "FP-001" else "稳定写作",
                page_type="concept",
                markdown=markdown,
                source_refs=[ref],
            )
            output = FinalPages(pages=[page])
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
                        "call_index": call_index,
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
        "composition_plan": composition,
        "candidate_pages": candidate_pages,
        "wiki_snapshot": snapshot,
        "profile": load_profile(vault),
        "provider_registry": registry,
        "provider_contexts": {},
        "operation_id": "ING-20260618T000000Z-test",
    }

    output = _step_final_pages(vault, tmp_path / "run", state)
    final_pages = state["final_pages"]

    assert isinstance(final_pages, FinalPages)
    assert len(final_pages.pages) == 2
    assert all("](raw/project_note.md)" not in page.markdown for page in final_pages.pages)
    assert sorted(registry.requests) == [("FP-001", False), ("FP-001", True), ("FP-002", False)]
    assert output.counts["semantic_retry_count"] == 1
    assert output.counts["api_call_count"] == 3
    assert output.counts["api_success_count"] == 2
    assert output.counts["api_paused_count"] == 1
    retry_prompt = tmp_path / "run" / "final_pages" / "model_calls" / "final_pages_FP-001_retry_1.prompt.json"
    assert retry_prompt.exists()


def test_final_pages_retry_update_that_drops_preimage_coverage(tmp_path: Path) -> None:
    vault = init_vault(tmp_path / "vault")
    existing_path = vault / "wiki" / "concepts" / "Concept_Agents_SDK.md"
    existing_path.parent.mkdir(parents=True, exist_ok=True)
    existing_body = (
        "# Agents SDK\n\n"
        "## 摘要\n\nAgents SDK 用于构建生产级智能体应用。\n\n"
        "### 使用案例\n\nCoinbase 和 Box 使用 Agents SDK 构建企业级智能体。\n\n"
        "### 开源与社区愿景\n\nOpenAI 将 Agents SDK 作为开源框架持续发展。\n"
    )
    existing_path.write_text(existing_body, encoding="utf-8")
    existing_sha = sha256_file(existing_path)
    ref = SourceRef(raw_path="raw/project_note.md", raw_sha256="abc", locator="whole_file")
    candidate_pages = CandidatePages(
        pages=[
            CandidatePage(
                candidate_page_id="CP-001",
                content_unit_id="CU-001",
                title="Agents SDK Python 快速开始",
                proposed_page_type="concept",
                proposed_path_hint="concepts/Concept_Agents_SDK.md",
                summary="补充 Python 安装和 Hello World。",
                body_markdown="# Agents SDK Python 快速开始\n\n## 安装\n\n使用 pip install openai-agents。",
                source_refs=[ref],
                confidence=0.9,
            )
        ]
    )
    composition = CompositionPlan(
        items=[
            CompositionItem(
                final_page_id="FP-001",
                target_path="concepts/Concept_Agents_SDK.md",
                action="update",
                merge_decision_ids=["MD-001"],
                candidate_page_ids=["CP-001"],
                section_order=["摘要", "安装", "使用案例", "开源与社区愿景"],
                preserve_rules=["保留旧页使用案例和开源愿景。"],
                source_ref_rules=["追加本次 raw 来源引用。"],
                readability_goal="在旧页基础上补充 Python 快速开始。",
            )
        ]
    )
    snapshot = WikiSnapshot(
        wiki_root="wiki",
        pool_hash="old",
        generated_at="2026-06-18T00:00:00Z",
        entries=[
            WikiKnowledgeEntry(
                path="concepts/Concept_Agents_SDK.md",
                title="Agents SDK",
                page_type="concept",
                sha256=existing_sha,
                summary="Agents SDK 用于构建生产级智能体应用，包含使用案例和开源社区愿景。",
                created="2026-06-01",
                source_raw_paths=["raw/old_agents_sdk.md"],
                source_raw_hashes=["oldhash"],
                source_prepared_hashes=["oldhash"],
                source_operation_ids=["OLD-OP"],
                updated="2026-06-01",
                text_excerpt=existing_body,
            )
        ],
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
            is_retry = "validation_error" in request.user_payload
            if not is_retry:
                page = FinalPage(
                    final_page_id="FP-001",
                    target_path="concepts/Concept_Agents_SDK.md",
                    action="update",
                    title="Agents SDK",
                    page_type="concept",
                    markdown="# Agents SDK\n\n## 摘要\n\nAgents SDK 支持 Python 快速开始。\n\n## 安装\n\n使用 pip install openai-agents。",
                    source_refs=[ref],
                )
            else:
                page = FinalPage(
                    final_page_id="FP-001",
                    target_path="concepts/Concept_Agents_SDK.md",
                    action="update",
                    title="Agents SDK",
                    page_type="concept",
                    markdown=(
                        "# Agents SDK\n\n"
                        "## SDK 的定位\n\nAgents SDK 用于构建生产级智能体应用，也支持 Python 快速开始。\n\n"
                        "## 安装\n\n使用 pip install openai-agents。\n\n"
                        "## 使用案例\n\nCoinbase 和 Box 使用 Agents SDK 构建企业级智能体。\n\n"
                        "## 开源与社区愿景\n\nOpenAI 将 Agents SDK 作为开源框架持续发展。"
                    ),
                    source_refs=[ref],
                    preimage_coverage_report=[
                        PreimageCoverageItem(
                            requirement_id="OLD-SUMMARY",
                            status="merged",
                            final_anchor="SDK 的定位",
                            evidence="旧页关于生产级智能体应用的定位已合并到 SDK 的定位。",
                        ),
                        PreimageCoverageItem(
                            requirement_id="OLD-SECTION-001",
                            status="preserved",
                            final_anchor="使用案例",
                            evidence="Coinbase 和 Box 的旧使用案例已保留在使用案例小节。",
                        ),
                        PreimageCoverageItem(
                            requirement_id="OLD-SECTION-002",
                            status="preserved",
                            final_anchor="开源与社区愿景",
                            evidence="旧页开源社区愿景已保留在同名小节。",
                        ),
                    ],
                )
            output = FinalPages(pages=[page])
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
        "composition_plan": composition,
        "candidate_pages": candidate_pages,
        "wiki_snapshot": snapshot,
        "profile": load_profile(vault),
        "provider_registry": registry,
        "provider_contexts": {},
        "operation_id": "ING-20260618T000000Z-test",
    }

    output = _step_final_pages(vault, tmp_path / "run", state)
    final_page = state["final_pages"].pages[0]

    assert len(registry.requests) == 2
    first_payload = registry.requests[0].user_payload
    requirements = first_payload["preimage_coverage_requirements"]
    summary_requirement = next(item for item in requirements if item["requirement_id"] == "OLD-SUMMARY")
    assert "摘要" in summary_requirement["forbidden_final_anchors"]
    assert "使用案例" in summary_requirement["anchor_candidates"]
    assert "开源与社区愿景" in summary_requirement["anchor_candidates"]
    assert "OLD-SUMMARY 的 final_anchor 不能写" in summary_requirement["final_anchor_rule"]
    first_instructions = "\n".join(first_payload["instructions"])
    assert "OLD-SUMMARY 也不能把 final_anchor 写成“摘要”" in first_instructions
    assert "不要填写 HTML anchor id" in first_instructions
    assert "缺少旧页覆盖报告" in registry.requests[1].user_payload["validation_error"]
    retry_instructions = "\n".join(registry.requests[1].user_payload["instructions"])
    assert "如果 validation_error 指出 final_anchor 有问题" in retry_instructions
    assert "不要写 HTML anchor id" in retry_instructions
    assert output.counts["semantic_retry_count"] == 1
    assert output.counts["preimage_coverage_requirement_count"] == 3
    assert output.counts["preimage_coverage_report_count"] == 3
    assert "Coinbase" in final_page.markdown
    assert "开源与社区愿景" in final_page.markdown
    assert (tmp_path / "run" / "final_pages" / "preimage_coverage_report.json").exists()


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

    markdown = _canonical_final_markdown(page, operation_id="OP-RELATED")

    assert "模型自己写的相关页面" not in markdown
    assert "## 相关页面" not in markdown


def test_canonicalization_resolves_short_body_wikilinks_to_known_paths(tmp_path: Path) -> None:
    ref = SourceRef(raw_path="raw/a.md", raw_sha256="abc", locator="whole_file")
    page = FinalPage(
        final_page_id="FP-001",
        target_path="concepts/Concept_New.md",
        action="create",
        title="新页",
        page_type="concept",
        markdown="# 新页\n\n正文链接到 [[Concept_智能体与工作流]]。",
        source_refs=[ref],
    )

    markdown = _canonical_final_markdown(
        page,
        operation_id="OP-LINK",
        known_paths={"concepts/Concept_New.md", "concepts/Concept_智能体与工作流.md"},
        path_titles={"concepts/Concept_New.md": "新页", "concepts/Concept_智能体与工作流.md": "智能体与工作流"},
    )

    assert "[[concepts/Concept_智能体与工作流]]" in markdown


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
    markdown = _canonical_final_markdown(page, operation_id="OP-MISMATCH")
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

    markdown = _canonical_final_markdown(page.model_copy(update={"title": existing.title}), operation_id="OP-NEW", existing_entry=existing, model_title=page.title)
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


def test_composition_plan_retries_target_path_drift(tmp_path: Path) -> None:
    vault = init_vault(tmp_path / "vault")
    ref = SourceRef(raw_path="raw/react.md", raw_sha256="abc", locator="whole_file")
    expected_path = "concepts/Concept_ReAct_推理与行动协同的语言模型范式.md"
    wrong_path = "concepts/Concept_ReAct_Synergizing_Reasoning_and_Acting_in_Language_Models.md"
    candidate_pages = CandidatePages(
        pages=[
            CandidatePage(
                candidate_page_id="CP-001",
                content_unit_id="CU-001",
                title="ReAct 推理与行动协同",
                proposed_page_type="concept",
                proposed_path_hint=expected_path,
                summary="ReAct 将推理轨迹与环境行动交织起来。",
                body_markdown="# ReAct 推理与行动协同\n\nReAct 将推理轨迹与环境行动交织起来。",
                source_refs=[ref],
                confidence=0.9,
            )
        ]
    )
    merge_plan = MergePlan(
        action_counts={"create": 1, "update": 0, "noop": 0},
        decisions=[
            make_decision(
                decision_id="MD-001",
                candidate_page_id="CP-001",
                action="create",
                target_path=expected_path,
                title="ReAct 推理与行动协同",
                ref=ref,
            )
        ],
    )

    def composition_item(target_path: str) -> CompositionItem:
        return CompositionItem(
            final_page_id="FP-001",
            target_path=target_path,
            action="create",
            merge_decision_ids=["MD-001"],
            candidate_page_ids=["CP-001"],
            section_order=["摘要", "核心机制"],
            source_ref_rules=["保留本次 raw 的来源引用。"],
            readability_goal="整理成一篇中文概念页。",
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
            target_path = expected_path if "validation_error" in request.user_payload else wrong_path
            output = CompositionPlan(items=[composition_item(target_path)])
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
        "merge_plan": merge_plan,
        "candidate_pages": candidate_pages,
        "profile": load_profile(vault),
        "provider_registry": registry,
        "provider_contexts": {},
    }

    output = _step_composition_plan(tmp_path / "run", state)
    artifact = state["composition_plan"]

    assert isinstance(artifact, CompositionPlan)
    assert artifact.items[0].target_path == expected_path
    assert len(registry.requests) == 2
    assert registry.requests[0].user_payload["expected_writable_targets"] == [expected_path]
    assert "必须覆盖所有可写合并目标" in registry.requests[1].user_payload["validation_error"]
    assert output.counts["semantic_retry_count"] == 1
    assert output.counts["api_call_count"] == 2
    assert output.counts["api_success_count"] == 1
    assert output.counts["api_paused_count"] == 1


def test_composition_and_final_page_prompt_runtime_contracts(tmp_path: Path) -> None:
    vault = init_vault(tmp_path / "vault")
    ref = SourceRef(raw_path="raw/project_note.md", raw_sha256="abc", locator="whole_file")
    candidate_pages = CandidatePages(
        pages=[
            CandidatePage(
                candidate_page_id="CP-001",
                content_unit_id="CU-001",
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
