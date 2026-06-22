from __future__ import annotations

import pytest
from pydantic import ValidationError

from llmwiki_engine.lite.models import RelatedPageRef, SourceDigest, SourcePageUnit, SourceRef
from llmwiki_engine.lite.pipeline import normalize_source_digest
from llmwiki_engine.lite.profile import PROJECT_BASIC, Profile


def test_schema_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        SourceRef(raw_path="raw/a.md", raw_sha256="abc", locator="whole_file", surprise=True)  # type: ignore[call-arg]


def test_related_page_ref_rejects_internal_existing_related_source() -> None:
    with pytest.raises(ValidationError):
        RelatedPageRef(target_path="concepts/Concept_A.md", display_title="A", source="existing_related", reason="internal only")  # type: ignore[arg-type]


def test_normalize_source_digest_rewrites_page_unit_path_without_model_repair() -> None:
    ref = SourceRef(raw_path="raw/a.md", raw_sha256="abc", locator="whole_file")
    digest = SourceDigest(
        source_raw_path="raw/a.md",
        raw_sha256="abc",
        summary="summary",
        key_takeaways=["one"],
        page_units=[
            SourcePageUnit(
                page_unit_id="TEMP",
                title="Knowledge Engine",
                page_type="unknown",
                path_hint="../bad.md",
                summary="summary",
                content_scope="scope",
                must_cover_points=[],
                source_refs=[ref],
            )
        ],
    )

    normalized, report = normalize_source_digest(digest, Profile.model_validate(PROJECT_BASIC))

    assert normalized.page_units[0].page_unit_id == "PU-001"
    assert normalized.page_units[0].page_type == "concept"
    assert normalized.page_units[0].path_hint == "concepts/Concept_Knowledge_Engine.md"
    assert normalized.page_units[0].must_cover_points == ["summary"]
    assert report.model_calls == 0
    assert report.repairs[0].local_fix is True
    assert report.repairs[0].model_called is False
