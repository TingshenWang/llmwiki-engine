from __future__ import annotations

import pytest
from pydantic import ValidationError

from llmwiki_engine.lite.models import RelatedPageRef, SourceDigest, SourceDigestCandidate, SourceRef
from llmwiki_engine.lite.pipeline import normalize_source_digest


def test_schema_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        SourceRef(raw_path="raw/a.md", raw_sha256="abc", locator="whole_file", surprise=True)  # type: ignore[call-arg]


def test_related_page_ref_rejects_internal_existing_related_source() -> None:
    with pytest.raises(ValidationError):
        RelatedPageRef(target_path="concepts/Concept_A.md", display_title="A", source="existing_related", reason="internal only")  # type: ignore[arg-type]


def test_normalize_source_digest_fills_empty_suggested_title_without_model_repair() -> None:
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
                name="Knowledge Engine",
                suggested_page_title=" ",
                summary="summary",
                source_basis="basis",
                source_refs=[ref],
            )
        ],
    )

    normalized, report = normalize_source_digest(digest)

    assert normalized.concepts[0].suggested_page_title == "Knowledge Engine"
    assert report.model_calls == 0
    assert report.repairs[0].local_fix is True
    assert report.repairs[0].model_called is False
