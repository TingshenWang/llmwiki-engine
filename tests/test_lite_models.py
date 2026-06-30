from __future__ import annotations

import pytest
from pydantic import ValidationError

from llmwiki_engine.lite.models import RelatedPageRef, SourceClaim, SourceDigest, SourceContentUnit, SourceRef
from llmwiki_engine.lite.pipeline import normalize_source_digest
from llmwiki_engine.lite.profile import PROJECT_BASIC, Profile


def test_schema_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        SourceRef(raw_path="raw/a.md", raw_sha256="abc", locator="whole_file", surprise=True)  # type: ignore[call-arg]


def test_related_page_ref_rejects_internal_existing_related_source() -> None:
    with pytest.raises(ValidationError):
        RelatedPageRef(target_path="concepts/Concept_A.md", display_title="A", source="existing_related", reason="internal only")  # type: ignore[arg-type]


def test_normalize_source_digest_rewrites_content_unit_path_without_model_repair() -> None:
    ref = SourceRef(raw_path="raw/a.md", raw_sha256="abc", locator="whole_file")
    digest = SourceDigest(
        source_raw_path="raw/a.md",
        raw_sha256="abc",
        summary="summary",
        key_takeaways=["one"],
        claims=[
            SourceClaim(
                claim_id="TEMP-CLAIM",
                text="知识引擎是需要进入 wiki 的核心概念。",
                kind="concept",
                importance=4,
                concept_terms=["知识引擎"],
                raw_locator="whole_file",
                source_refs=[ref],
            )
        ],
        content_units=[
            SourceContentUnit(
                content_unit_id="TEMP",
                title="Knowledge Engine",
                content_role="主干",
                absorption_decision="独立成页",
                anchor_unit_id="TEMP",
                section_hint="核心内容",
                page_type="unknown",
                path_hint="../bad.md",
                summary="summary",
                absorption_reason="reason",
                content_scope="scope",
                claim_ids=["TEMP-CLAIM"],
                source_refs=[ref],
            )
        ],
    )

    normalized, report = normalize_source_digest(digest, Profile.model_validate(PROJECT_BASIC))

    assert normalized.content_units[0].content_unit_id == "CU-001"
    assert normalized.claims[0].claim_id == "C-001"
    assert normalized.content_units[0].page_type == "concept"
    assert normalized.content_units[0].path_hint == "concepts/Concept_Knowledge_Engine.md"
    assert normalized.content_units[0].claim_ids == ["C-001"]
    assert report.model_calls == 0
    assert report.repairs[0].local_fix is True
    assert report.repairs[0].model_called is False
