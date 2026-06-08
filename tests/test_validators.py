import pytest

from llmwiki_engine.models import (
    CandidateResolutionArtifact,
    CandidateResolutionItem,
    RelatedPageRef,
    SourceBasis,
    SourceDigestArtifact,
    SourceDigestCandidate,
    WeakOrNoiseItem,
    WikiMergePlanArtifact,
    WikiMergePlanItem,
)
from llmwiki_engine.pipeline import render_source_digest_markdown
from llmwiki_engine.validators import ValidationError, validate_candidate_resolution, validate_source_digest, validate_wiki_merge_plan


def test_source_digest_accepts_complete_candidates() -> None:
    validate_source_digest(_digest())


def test_source_digest_rejects_duplicate_candidate_ids() -> None:
    digest = _digest()
    digest.designs.append(
        SourceDigestCandidate(
            candidate_id="CAND001",
            name="Duplicate",
            type="design",
            one_sentence_summary="Duplicate candidate.",
            why_matters="It should be rejected.",
            wiki_value="It would create review confusion.",
            suggested_page_title="Duplicate",
        )
    )

    with pytest.raises(ValidationError, match="duplicate candidate_id: CAND001"):
        validate_source_digest(digest)


def test_source_digest_rejects_empty_candidate_summary() -> None:
    digest = _digest()
    digest.concepts[0].one_sentence_summary = " "

    with pytest.raises(ValidationError, match="one_sentence_summary must not be empty"):
        validate_source_digest(digest)


def test_source_digest_rejects_formal_source_candidate_type() -> None:
    digest = _digest()
    digest.concepts[0].type = "source"

    with pytest.raises(ValidationError, match="formal candidates must not use source page type"):
        validate_source_digest(digest)


def test_source_digest_rejects_whole_english_user_text_for_zh_cn() -> None:
    digest = _digest()
    digest.summary = "This source explains how product managers should work with agents, workflows, evaluation loops, and career strategy."

    with pytest.raises(ValidationError, match="must be Chinese"):
        validate_source_digest(digest, language="zh-CN")


def test_source_digest_allows_domain_terms_inside_chinese_for_zh_cn() -> None:
    digest = _digest()
    digest.summary = "这篇材料讨论 AI PM 如何理解 Workflow、Agent 和 RAG，并把它们转成可复用知识。"
    digest.concepts[0].one_sentence_summary = "Workflow 和 Agent 的差异会影响产品方案、执行边界和评估方式。"
    digest.concepts[0].why_matters = "它能帮助 PM 判断什么时候用流程自动化，什么时候需要智能体。"
    digest.concepts[0].wiki_value = "适合沉淀为对比型知识，并连接到 Agent 产品设计。"

    validate_source_digest(digest, language="zh-CN")


def test_source_digest_accepts_weak_noise_without_suggested_page_title() -> None:
    digest = _digest()
    digest.weak_or_noise_items.append(
        WeakOrNoiseItem(
            candidate_id="NOISE001",
            name="Sponsor transition",
            type="noise",
            one_sentence_summary="A transcript transition that should not become a wiki page.",
            why_matters="It explains why the mention was filtered out.",
            wiki_value="It prevents review confusion.",
        )
    )

    validate_source_digest(digest)
    markdown = render_source_digest_markdown(digest)
    weak_section = markdown.split("## 弱相关或噪声项", 1)[1]
    assert "Suggested" not in weak_section
    assert "create" not in weak_section


def test_source_digest_accepts_current_weak_noise_fields() -> None:
    digest = SourceDigestArtifact.model_validate(
        {
            "source_raw_path": "raw/sample.md",
            "summary": "A useful source digest.",
            "concepts": [
                {
                    "candidate_id": "CAND001",
                    "name": "Useful concept",
                    "type": "concept",
                    "one_sentence_summary": "A useful concept from the source.",
                    "why_matters": "It matters for the wiki.",
                    "wiki_value": "It should become a page.",
                    "suggested_page_title": "Useful Concept",
                }
            ],
            "weak_or_noise_items": [
                {
                    "candidate_id": "NOISE001",
                    "name": "Lightweight aside",
                    "type": "noise",
                    "one_sentence_summary": "An aside that should not become a wiki page.",
                    "why_matters": "The source mentions it, but there is not enough durable knowledge to ingest.",
                    "suggested_action": "ignore",
                }
            ],
        }
    )

    assert digest.weak_or_noise_items[0].why_matters == "The source mentions it, but there is not enough durable knowledge to ingest."
    validate_source_digest(digest)
    markdown = render_source_digest_markdown(digest)
    weak_section = markdown.split("## 弱相关或噪声项", 1)[1]
    assert "ignore" not in weak_section


def test_source_digest_rejects_why_matches_alias() -> None:
    data = _digest().model_dump(mode="json")
    data["weak_or_noise_items"] = [
        {
            "candidate_id": "NOISE001",
            "name": "Lightweight aside",
            "type": "noise",
            "one_sentence_summary": "An aside that should not become a wiki page.",
            "why_matches": "旧字段不再兼容。",
            "suggested_action": "ignore",
        }
    ]

    with pytest.raises(Exception, match="why_matches"):
        SourceDigestArtifact.model_validate(data)


def test_source_digest_rejects_wrong_schema_version() -> None:
    data = _digest().model_dump(mode="json")
    data["schema_version"] = "source_digest.invalid"

    with pytest.raises(Exception, match="source_digest.v2"):
        SourceDigestArtifact.model_validate(data)


def test_source_digest_rejects_formal_candidate_suggested_action_extra() -> None:
    data = _digest().model_dump(mode="json")
    data["concepts"][0]["suggested_action"] = "update"
    with pytest.raises(Exception, match="Extra inputs are not permitted"):
        SourceDigestArtifact.model_validate(data)

    data = _digest().model_dump(mode="json")
    data["concepts"][0]["unexpected"] = "blocked"
    with pytest.raises(Exception, match="Extra inputs are not permitted"):
        SourceDigestArtifact.model_validate(data)


def test_candidate_resolution_must_cover_all_ingest_candidates() -> None:
    digest = _digest()
    resolution = CandidateResolutionArtifact(items=[])

    with pytest.raises(ValidationError, match="misses approved candidates"):
        validate_candidate_resolution(digest, resolution)


def test_candidate_resolution_rejects_unknown_candidate() -> None:
    digest = _digest()
    resolution = CandidateResolutionArtifact(
        items=[
            _resolution_item("CAND001"),
            _resolution_item("CAND999"),
        ]
    )

    with pytest.raises(ValidationError, match="references unknown candidates"):
        validate_candidate_resolution(digest, resolution)


def test_candidate_resolution_target_path_is_relative_to_wiki_root() -> None:
    digest = _digest()
    resolution = CandidateResolutionArtifact(items=[_resolution_item("CAND001", target_path="wiki/concepts/Concept_Test.md")])

    with pytest.raises(ValidationError, match="relative to wiki root"):
        validate_candidate_resolution(digest, resolution)


def test_source_basis_strips_and_drops_empty_candidate_refs() -> None:
    source_basis = SourceBasis(
        source_candidate_ids=[" CAND001 ", "", "CAND001"],
        prepared_discovered_candidates=["  ", "prepared topic", "prepared topic"],
    )

    assert source_basis.source_candidate_ids == ["CAND001"]
    assert source_basis.prepared_discovered_candidates == ["prepared topic"]


def test_candidate_resolution_rejects_empty_prepared_discovered_source_basis() -> None:
    digest = SourceDigestArtifact(source_raw_path="raw/sample.md", summary="No formal candidates.")
    resolution = CandidateResolutionArtifact(
        items=[
            CandidateResolutionItem(
                page_plan_id="PP-EMPTY",
                source_basis=SourceBasis(prepared_discovered_candidates=["  "]),
                page_type="concept",
                display_title="Empty basis",
                candidate_target_path="concepts/Concept_Empty_basis.md",
                topic_summary="Empty basis topic.",
                why_this_page="This should not pass without a real source basis.",
                reason="test",
            )
        ]
    )

    with pytest.raises(ValidationError, match="source_basis must not be empty"):
        validate_candidate_resolution(digest, resolution)


@pytest.mark.parametrize("target_path", ["", "../outside.md", "/tmp/outside.md"])
def test_candidate_resolution_rejects_unsafe_target_paths(target_path: str) -> None:
    digest = _digest()
    resolution = CandidateResolutionArtifact(items=[_resolution_item("CAND001", target_path=target_path)])

    with pytest.raises(ValidationError):
        validate_candidate_resolution(digest, resolution)


def test_wiki_merge_plan_rejects_missing_section_plans() -> None:
    digest = _digest()
    plan = _merge_plan([_plan_item("CAND001", target_path="concepts/Concept_Test.md", section_plans={})])

    with pytest.raises(ValidationError, match="section_plans must not be empty"):
        validate_wiki_merge_plan(digest, plan)


def test_wiki_merge_plan_update_requires_matched_page() -> None:
    digest = _digest()
    plan = _merge_plan([_plan_item("CAND001", action="update", matched_page=None)])

    with pytest.raises(ValidationError, match="update action must include matched_page"):
        validate_wiki_merge_plan(digest, plan)


def test_wiki_merge_plan_rejects_empty_prepared_discovered_source_basis() -> None:
    digest = SourceDigestArtifact(source_raw_path="raw/sample.md", summary="No formal candidates.")
    plan = WikiMergePlanArtifact(
        log_date="2026-06-06",
        context_snapshot_ref="wiki_context_snapshot/wiki_context_snapshot.json",
        items=[
            WikiMergePlanItem(
                page_plan_id="PP-EMPTY",
                source_basis=SourceBasis(prepared_discovered_candidates=[""]),
                page_type="concept",
                canonical_target_path="concepts/Concept_Empty_basis.md",
                display_title="Empty basis",
                action="create",
                new_understanding="This should not pass without a real source basis.",
                section_plans={"summary": "Summary"},
                reason="test",
                apply_eligibility="applyable",
                related_absence_reason="no_candidate",
            )
        ],
    )

    with pytest.raises(ValidationError, match="source_basis must not be empty"):
        validate_wiki_merge_plan(digest, plan)


def test_wiki_merge_plan_rejects_self_related_page() -> None:
    digest = _digest()
    plan = _merge_plan(
        [
            _plan_item(
                "CAND001",
                related_pages=[
                    RelatedPageRef(
                        target_path="concepts/Concept_Knowledge digestion.md",
                        display_title="Knowledge digestion",
                        source="source_digest",
                        reason="self link",
                    )
                ],
            )
        ]
    )

    with pytest.raises(ValidationError, match="related_pages must not include self-link"):
        validate_wiki_merge_plan(digest, plan)


def test_wiki_merge_plan_rejects_source_page_type_items() -> None:
    digest = _digest()
    plan = _merge_plan([_plan_item("CAND001", candidate_type="source", target_path="sources/Source_Test.md")])

    with pytest.raises(ValidationError, match="wiki_merge_plan items must not use source page type"):
        validate_wiki_merge_plan(digest, plan)


def test_wiki_merge_plan_rejects_duplicate_writable_targets() -> None:
    digest = _digest()
    digest.concepts.append(
        SourceDigestCandidate(
            candidate_id="CAND002",
            name="Second concept",
            type="concept",
            one_sentence_summary="Another useful concept.",
            why_matters="It matters.",
            wiki_value="It belongs in the wiki.",
            suggested_page_title="Second Concept",
        )
    )
    plan = _merge_plan(
        [
            _plan_item("CAND001", target_path="concepts/Concept_Knowledge digestion.md"),
            _plan_item("CAND002", target_path="concepts/Concept_Knowledge digestion.md"),
        ]
    )

    with pytest.raises(ValidationError, match="duplicate writable target paths"):
        validate_wiki_merge_plan(digest, plan)


def test_wiki_merge_plan_rejects_source_graph_links() -> None:
    digest = _digest()
    plan = _merge_plan(
        [
            _plan_item(
                "CAND001",
                related_pages=[
                    RelatedPageRef(
                        target_path="sources/Source_Test.md",
                        display_title="Source Test",
                        source="wiki_context",
                        reason="source pages must stay out of related",
                    )
                ],
            )
        ]
    )

    with pytest.raises(ValidationError, match="related_pages must not point to source pages"):
        validate_wiki_merge_plan(digest, plan)


def test_wiki_merge_plan_rejects_english_related_reason_for_zh_cn() -> None:
    digest = _digest()
    plan = _merge_plan(
        [
            _plan_item(
                "CAND001",
                related_pages=[
                    RelatedPageRef(
                        target_path="concepts/Concept_Other.md",
                        display_title="Other",
                        source="source_digest",
                        reason="This page is related because both discuss reusable product knowledge and agent workflows.",
                    )
                ],
            )
        ]
    )

    with pytest.raises(ValidationError, match="must be Chinese"):
        validate_wiki_merge_plan(digest, plan, language="zh-CN")


def test_wiki_merge_plan_must_cover_resolution_page_plans() -> None:
    digest = _digest()
    resolution = CandidateResolutionArtifact(
        items=[
            _resolution_item("CAND001"),
            CandidateResolutionItem(
                page_plan_id="PP-DISCOVERED",
                source_basis=SourceBasis(prepared_discovered_candidates=["全文补发现主题"]),
                page_type="concept",
                display_title="Discovered topic",
                candidate_target_path="concepts/Concept_Discovered topic.md",
                topic_summary="Discovered topic summary.",
                why_this_page="It was found from the prepared raw.",
                reason="Coverage check found it.",
            ),
        ]
    )
    plan = _merge_plan([_plan_item("CAND001")])

    with pytest.raises(ValidationError, match="misses planned pages"):
        validate_wiki_merge_plan(digest, plan, resolution)


def _digest() -> SourceDigestArtifact:
    return SourceDigestArtifact(
        source_raw_path="raw/sample.md",
        summary="A useful source digest.",
        concepts=[
            SourceDigestCandidate(
                candidate_id="CAND001",
                name="Knowledge digestion",
                type="concept",
                one_sentence_summary="Knowledge digestion turns raw material into reusable understanding.",
                why_matters="It is the core product value.",
                wiki_value="It helps create a durable concept page.",
                suggested_page_title="Knowledge digestion",
            )
        ],
    )


def _resolution_item(
    candidate_id: str,
    *,
    target_path: str = "concepts/Concept_Knowledge digestion.md",
) -> CandidateResolutionItem:
    return CandidateResolutionItem(
        page_plan_id=f"PP-{candidate_id}",
        source_basis=SourceBasis(source_candidate_ids=[candidate_id]),
        page_type="concept",
        display_title="Knowledge digestion",
        candidate_target_path=target_path,
        topic_summary="Knowledge digestion topic.",
        why_this_page="It deserves a concept page.",
        reason="Maps the candidate to a concept page.",
    )


def _plan_item(
    candidate_id: str,
    *,
    candidate_type: str = "concept",
    target_path: str = "concepts/Concept_Knowledge digestion.md",
    action: str = "create",
    matched_page: str | None = None,
    section_plans: dict[str, str] | None = None,
    related_pages: list[RelatedPageRef] | None = None,
) -> WikiMergePlanItem:
    return WikiMergePlanItem(
        page_plan_id=f"PP-{candidate_id}",
        source_basis=SourceBasis(source_candidate_ids=[candidate_id]),
        page_type=candidate_type,
        canonical_target_path=target_path,
        display_title="Knowledge digestion",
        action=action,
        matched_page=matched_page,
        new_understanding="Knowledge digestion turns raw material into reusable understanding.",
        section_plans=section_plans if section_plans is not None else {"summary": "Summary"},
        apply_eligibility="blocked" if action == "needs_human_decision" else "applyable",
        blocked_reason="needs human" if action == "needs_human_decision" else "",
        reason="Maps the candidate to a concept page.",
        related_pages=related_pages or [],
    )


def _merge_plan(items: list[WikiMergePlanItem]) -> WikiMergePlanArtifact:
    return WikiMergePlanArtifact(
        log_date="2026-06-03",
        items=items,
        context_snapshot_ref="wiki_context_snapshot/wiki_context_snapshot.json",
    )
