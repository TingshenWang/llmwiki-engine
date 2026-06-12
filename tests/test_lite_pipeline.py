from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from llmwiki_engine.cli import app
from llmwiki_engine.lite import embeddings
from llmwiki_engine.lite import related as related_logic
from llmwiki_engine.lite import system_pages
from llmwiki_engine.lite.io import sha256_file, sha256_text
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
    _canonical_final_markdown,
    _page_generation_parallelism,
    _validate_before_write,
    init_vault,
    PipelineError,
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
    config = embeddings.EmbeddingConfig(backend="unsupported-test", model="unsupported-test", dimensions=256)

    with pytest.raises(RuntimeError, match="真实 embedding"):
        embeddings.build_candidate_contexts(vault, candidate_pages, [], {}, config)


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
        action_counts={"create": 1, "update": 0, "noop": 0, "split": 0, "merge": 0},
        decisions=[MergeDecision(candidate_page_id="CP-001", action="create", target_path="concepts/Concept_New.md", reason="create", source_refs=[ref])],
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
