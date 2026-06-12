from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from llmwiki_engine.cli import app
from llmwiki_engine.lite import embeddings
from llmwiki_engine.lite import related as related_logic
from llmwiki_engine.lite import system_pages
from llmwiki_engine.lite.io import read_json, sha256_file, sha256_text
from llmwiki_engine.lite.models import (
    CandidateContext,
    CandidateContextHit,
    CandidateContexts,
    CandidatePage,
    CandidatePages,
    FinalPage,
    FinalPages,
    MergeDecision,
    MergePlan,
    RawBinding,
    RelatedPageRef,
    SourceDigest,
    SourceDigestCandidate,
    SourceRef,
    WikiKnowledgeEntry,
    WikiSnapshot,
)
from llmwiki_engine.lite.pipeline import (
    _assert_source_digest_chinese,
    _assert_composition_plan_chinese,
    _build_local_composition_plan,
    _canonical_final_markdown,
    _normalize_composition_plan,
    _page_generation_parallelism,
    _validate_before_write,
    init_vault,
    PipelineError,
    run_ingest,
    scan_raw_candidates,
    verify_operation,
)
from llmwiki_engine.lite.profile import load_profile


def write_raw(vault: Path, name: str = "project_note.md") -> Path:
    raw = vault / "raw" / name
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(
        "# 项目笔记\n\n"
        "这个项目的核心目标是把知识库维护流程拆成可测试的模块。"
        "第一阶段先证明 CLI、状态机、artifact 和 validator 能跑通。\n\n"
        "## 简化 Ingest 流程\n\n"
        "简化 Ingest 不进行人工审核，也不需要手动 apply。"
        "系统会先生成候选页面，再生成 merge plan 和 composition plan，最后自动写入 wiki。\n",
        encoding="utf-8",
    )
    return raw


def test_init_and_ingest_full_auto_writes_wiki_and_receipt(tmp_path: Path) -> None:
    vault = init_vault(tmp_path / "vault")
    raw = write_raw(vault)

    manifest = run_ingest(vault, Path("raw") / raw.name, slug="smoke", emit_progress=False, allow_test_providers=True)

    assert manifest.status == "written"
    assert manifest.receipt_path is not None
    assert (vault / manifest.receipt_path).exists()
    assert (vault / "wiki" / "sources" / "Source_project_note.md").exists()
    written_pages = list((vault / "wiki").glob("designs/*.md"))
    assert written_pages
    text = written_pages[0].read_text(encoding="utf-8")
    frontmatter_keys = _frontmatter_keys(text)
    assert frontmatter_keys == [
        "llmwiki_type",
        "title",
        "aliases",
        "summary",
        "created",
        "updated",
        "source_raw_paths",
        "source_raw_hashes",
        "source_prepared_hashes",
        "source_operation_ids",
        "last_ingest_operation",
    ]
    frontmatter = yaml.safe_load(text.split("---", 2)[1])
    assert frontmatter["llmwiki_type"] == "design"
    assert "type" not in frontmatter
    assert "source_refs" not in frontmatter
    assert "llmwiki" not in frontmatter
    assert frontmatter["source_raw_paths"] == ["raw/project_note.md"]
    assert frontmatter["source_raw_hashes"] == [sha256_file(raw)]
    assert frontmatter["source_prepared_hashes"] == [sha256_file(raw)]
    assert frontmatter["source_operation_ids"] == [manifest.operation_id]
    config = read_json(vault / ".llmwiki" / "config.json")
    assert config["embedding"]["backend"] == "sentence_transformers"
    assert config["embedding"]["model"] == embeddings.DEFAULT_QWEN_EMBEDDING_MODEL
    assert not (vault / ".llmwiki" / "config.yaml").exists()
    report = verify_operation(vault, manifest.operation_id)
    assert report.ok, report.issues


def test_cli_status_and_raw_candidates_use_chinese_labels(tmp_path: Path) -> None:
    vault = init_vault(tmp_path / "vault")
    raw = write_raw(vault)
    run_ingest(vault, Path("raw") / raw.name, slug="cli-cn", emit_progress=False, allow_test_providers=True)
    runner = CliRunner()

    status_result = runner.invoke(app, ["ingest", "status", str(vault), "--verify"])
    assert status_result.exit_code == 0
    assert "raw 大小" in status_result.output
    assert "延后数" in status_result.output
    assert "召回后端" in status_result.output
    assert "校验：通过" in status_result.output
    assert "raw_size_bytes" not in status_result.output
    assert "deferred_count" not in status_result.output
    assert "retrieval_backend" not in status_result.output

    raw_result = runner.invoke(app, ["ingest", "raw-candidates", str(vault), "--all"])
    assert raw_result.exit_code == 0
    assert "原始材料候选" in raw_result.output
    assert "是" in raw_result.output
    assert "True" not in raw_result.output


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


def test_raw_candidates_marks_processed_after_ingest(tmp_path: Path) -> None:
    vault = init_vault(tmp_path / "vault")
    raw = write_raw(vault)
    before = scan_raw_candidates(vault)
    assert before["count"] == 1

    run_ingest(vault, raw, slug="processed", emit_progress=False, allow_test_providers=True)
    after = scan_raw_candidates(vault)
    assert after["count"] == 0
    all_items = scan_raw_candidates(vault, include_processed=True)
    assert all_items["items"][0]["processed"] is True


def test_cli_json_run(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    runner = CliRunner()
    result = runner.invoke(app, ["init", str(vault)])
    assert result.exit_code == 0, result.output
    raw = write_raw(vault)

    result = runner.invoke(app, ["ingest", "run", str(vault), str(raw), "--slug", "cli", "--json"])

    assert result.exit_code != 0
    assert "必须使用真实模型 provider" in result.output


def test_candidate_contexts_are_per_generated_page_and_refresh_last(tmp_path: Path) -> None:
    vault = init_vault(tmp_path / "vault")
    existing = vault / "wiki" / "designs" / "Design_简化_Ingest_流程.md"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text(
        "---\n"
        "title: 简化 Ingest 流程\n"
        "type: design\n"
        "source_refs: []\n"
        "---\n\n"
        "# 简化 Ingest 流程\n\n旧页面讨论自动 ingest、状态机、artifact 和 wiki 写入。\n",
        encoding="utf-8",
    )
    raw = write_raw(vault)

    manifest = run_ingest(vault, raw, slug="contexts", emit_progress=False, allow_test_providers=True)

    run_dir = vault / ".llmwiki" / "runs" / "ingest" / manifest.operation_id
    candidate_pages = read_json(run_dir / "candidate_pages" / "candidate_pages.json")
    contexts = read_json(run_dir / "candidate_contexts" / "candidate_contexts.json")
    receipt = read_json(vault / manifest.receipt_path)
    final_cache_report = read_json(run_dir / "embedding_cache_refresh" / "embedding_cache_refresh.json")
    step_names = [step.name for step in manifest.steps]
    assert step_names.index("candidate_pages") < step_names.index("candidate_contexts") < step_names.index("merge_plan")
    assert step_names.index("index_log_write") < step_names.index("embedding_cache_refresh") < step_names.index("receipt")
    assert step_names[-1] == "receipt"
    assert receipt["embedding_metrics"] == final_cache_report
    assert "embedding_cache_refresh" in receipt["artifact_hashes"]
    assert final_cache_report["cache_updated_paths"] == ["designs/Design_简化_Ingest_流程.md"]
    assert len(contexts["items"]) == len(candidate_pages["pages"])
    assert {item["candidate_page_id"] for item in contexts["items"]} == {page["candidate_page_id"] for page in candidate_pages["pages"]}
    assert all(len(item["hits"]) <= 5 for item in contexts["items"])
    assert any(hit["path"] == "designs/Design_简化_Ingest_流程.md" for item in contexts["items"] for hit in item["hits"])


def test_embedding_cache_keeps_only_latest_record_per_page(tmp_path: Path) -> None:
    vault = init_vault(tmp_path / "vault")
    existing = vault / "wiki" / "concepts" / "Concept_Cache_Probe.md"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text("# Cache Probe\n\nFirst body about a stable page.\n", encoding="utf-8")
    raw1 = write_raw(vault, "first.md")
    run_ingest(vault, raw1, slug="cache-first", emit_progress=False, allow_test_providers=True)

    existing.write_text("# Cache Probe\n\nSecond body with changed current content.\n", encoding="utf-8")
    raw2 = write_raw(vault, "second.md")
    manifest = run_ingest(vault, raw2, slug="cache-second", emit_progress=False, allow_test_providers=True)

    run_dir = vault / ".llmwiki" / "runs" / "ingest" / manifest.operation_id
    report = read_json(run_dir / "wiki_snapshot" / "embedding_cache_report.json")
    cache_dir = vault / ".llmwiki" / "cache" / "embeddings" / "sentence_transformers_Qwen_Qwen3-Embedding-0.6B_page_card_v1_1024" / "pages"
    records = list(cache_dir.glob("*concepts__Concept_Cache_Probe.md.json"))
    assert report["cache_stale"] >= 1
    assert "concepts/Concept_Cache_Probe.md" in report["cache_updated_paths"]
    assert len(records) == 1
    assert read_json(records[0])["content_sha256"] == sha256_file(existing)


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
        source_refs=[],
        text_excerpt="A page about local Qwen retrieval.",
    )
    config = embeddings.EmbeddingConfig(
        backend="sentence_transformers",
        model=embeddings.DEFAULT_QWEN_EMBEDDING_MODEL,
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

    contexts = embeddings.build_candidate_contexts(vault, candidate_pages, [entry], cached_records, config)

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
    config = embeddings.EmbeddingConfig(backend="hashing", model="lite-hashing-v1", dimensions=256)

    with pytest.raises(RuntimeError, match="真实 embedding"):
        embeddings.build_candidate_contexts(vault, candidate_pages, [], {}, config)


def test_system_pages_use_index_log_and_daily_log_contract(tmp_path: Path) -> None:
    vault = init_vault(tmp_path / "vault")
    (vault / "wiki" / "index.md").write_text(
        "# 索引\n\n"
        f"{system_pages.SYSTEM_MARKER}\n\n"
        "## 概念\n\n"
        "| 标题 | 页面 | 摘要 | 更新日期 |\n"
        "| --- | --- | --- | --- |\n"
        "| Stale | [[concepts/Concept_Stale]] | stale | 2026-01-01 |\n",
        encoding="utf-8",
    )
    existing = vault / "wiki" / "concepts" / "Concept_Old.md"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text(
        "---\n"
        "title: Old Concept\n"
        "type: concept\n"
        "llmwiki_type: concept\n"
        "summary: old summary\n"
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
        "title: Misplaced Source\n"
        "type: source\n"
        "llmwiki_type: source\n"
        "summary: source summary\n"
        "updated: 2026-01-02\n"
        "---\n\n"
        "# Misplaced Source\n",
        encoding="utf-8",
    )
    raw = write_raw(vault)

    manifest = run_ingest(vault, raw, slug="system-pages", emit_progress=False, allow_test_providers=True)
    log_date = _operation_date_from_id(manifest.operation_id)

    index_text = (vault / "wiki" / "index.md").read_text(encoding="utf-8")
    assert index_text.startswith("# 索引")
    assert system_pages.SYSTEM_MARKER in index_text
    assert "[[concepts/Concept_Old]]" in index_text
    assert "旧概念是否还适用于新的 Agent 工作流？" in index_text
    assert "Concept_Stale" not in index_text
    assert "Misplaced Source" not in index_text
    assert "[[sources/" not in index_text

    log_text = (vault / "wiki" / "log.md").read_text(encoding="utf-8")
    assert f"[[logs/{log_date}]]" in log_text
    assert f"最新 operation: `{manifest.operation_id}`" in log_text
    daily_text = (vault / "wiki" / "logs" / f"{log_date}.md").read_text(encoding="utf-8")
    assert f"`{manifest.operation_id}`" in daily_text
    assert "`raw/project_note.md`" in daily_text

    source_text = (vault / "wiki" / "sources" / "Source_project_note.md").read_text(encoding="utf-8")
    assert _frontmatter_keys(source_text) == [
        "llmwiki_type",
        "title",
        "aliases",
        "summary",
        "created",
        "updated",
        "source_raw_paths",
        "source_raw_hashes",
        "source_prepared_hashes",
        "source_operation_ids",
        "last_ingest_operation",
    ]
    assert "## 派生知识页" in source_text
    assert "[[" not in source_text
    assert "`designs/Design_简化_Ingest_流程.md`" in source_text


def test_related_links_follow_previous_branch_cap_and_filters() -> None:
    existing = "# Current\n\n## Related\n\n- [[concepts/Concept_Existing|Existing]]：旧链接。\n"
    rendered = system_pages.render_related_links(
        current_path="concepts/Concept_Current.md",
        candidate_paths=[
            "concepts/Concept_Current.md",
            "raw/a.md",
            "sources/Source_A.md",
            "Source_Misplaced.md",
            "logs/2026-06-11.md",
            "index.md",
            "concepts/Concept_A.md",
            "concepts/Concept_B.md",
            "concepts/Concept_C.md",
            "concepts/Concept_D.md",
        ],
        known_paths={
            "concepts/Concept_Existing.md",
            "concepts/Concept_A.md",
            "concepts/Concept_B.md",
            "concepts/Concept_C.md",
            "concepts/Concept_D.md",
        },
        existing_markdown=existing,
    )

    assert rendered.count("[[") == system_pages.RELATED_LINK_LIMIT
    assert "[[concepts/Concept_Existing|Existing]]" in rendered
    assert "Concept_Current" not in rendered
    assert "raw/a" not in rendered
    assert "Source_A" not in rendered
    assert "Source_Misplaced" not in rendered
    assert "logs/2026-06-11" not in rendered
    assert "index" not in rendered
    assert "Concept_D" not in rendered


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
        action_counts={"create": 2, "update": 0, "noop": 0, "split": 0, "merge": 0},
        decisions=[
            MergeDecision(candidate_page_id="CP-001", action="create", target_path="concepts/Concept_First.md", reason="create", source_refs=[ref]),
            MergeDecision(candidate_page_id="CP-002", action="create", target_path="concepts/Concept_Second.md", reason="create", source_refs=[ref]),
        ],
    )
    contexts = CandidateContexts(
        retrieval_backend="hashing",
        model="lite",
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
        source_refs=[],
        text_excerpt="Old page.",
    )
    snapshot = WikiSnapshot(wiki_root="wiki", pool_hash="pool", generated_at="2026-06-11T00:00:00Z", entries=[old_entry])
    contexts = CandidateContexts(
        retrieval_backend="hashing",
        model="lite",
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
        action_counts={"create": 1, "update": 0, "noop": 0, "split": 0, "merge": 0},
        decisions=[MergeDecision(candidate_page_id="CP-001", action="create", target_path="concepts/Concept_New.md", reason="create", source_refs=[ref])],
    )

    resolved, report = related_logic.finalize_merge_plan_related(plan, candidate_pages=candidate_pages, digest=digest, snapshot=snapshot, contexts=contexts)

    assert [ref.target_path for ref in resolved.decisions[0].related_pages] == ["concepts/Concept_Old.md"]
    assert any(item.target_path == "concepts/Concept_Missing.md" and item.decision == "filtered" for item in report.candidates)


def test_duplicate_create_merge_drops_related_self_links_after_composition() -> None:
    ref = SourceRef(raw_path="raw/a.md", raw_sha256="abc", locator="whole_file")
    merge_plan = MergePlan(
        action_counts={"create": 0, "update": 0, "noop": 0, "split": 0, "merge": 2},
        decisions=[
            MergeDecision(
                candidate_page_id="CP-001",
                action="merge",
                target_path="concepts/Concept_A.md",
                reason="merge one",
                source_refs=[ref],
                related_pages=[
                    RelatedPageRef(
                        target_path="concepts/Concept_A.md",
                        display_title="Concept A",
                        source="source_digest",
                        reason="分组后会变成自链接。",
                    )
                ],
            ),
            MergeDecision(
                candidate_page_id="CP-002",
                action="merge",
                target_path="concepts/Concept_A.md",
                reason="merge two",
                source_refs=[ref],
            ),
        ],
    )

    composition = _normalize_composition_plan(_build_local_composition_plan(merge_plan))

    assert len(composition.items) == 1
    assert composition.items[0].related_pages == []
    assert composition.items[0].warnings == ["多个合并决策指向同一个目标页面，已合并写作规则。"]
    _assert_composition_plan_chinese(composition)


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
        source_refs=[],
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
    assert _page_generation_parallelism({"config": {"page_generation": {"parallel_requests": 5}}}, 23) == 5


def _operation_date_from_id(operation_id: str) -> str:
    raw = operation_id.split("-", 2)[1].split("T", 1)[0]
    return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"


def _frontmatter_keys(markdown: str) -> list[str]:
    body = markdown.split("---", 2)[1]
    return [line.split(":", 1)[0] for line in body.splitlines() if line and not line.startswith(" ")]
