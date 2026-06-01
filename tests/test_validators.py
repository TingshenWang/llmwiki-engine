import pytest

from llmwiki_engine.models import (
    Claim,
    ClaimsArtifact,
    ExtractionWindow,
    ExtractionWindowsArtifact,
    RawIndexArtifact,
    RawSpan,
)
from llmwiki_engine.validators import ValidationError, validate_claims


def test_claim_evidence_can_reference_spans_outside_source_window() -> None:
    raw_index, windows = _raw_index_and_windows()
    claims = ClaimsArtifact(
        claims=[
            Claim(
                claim_id="CLM001",
                source_window_id="W001",
                text="Product taste matters across the article.",
                evidence_span_ids=["S001", "S003"],
                evidence_quote="Product taste decides what to build.",
                target_type="concept",
                target_title="Product taste",
            )
        ]
    )

    validate_claims(raw_index, windows, claims)


def test_claim_source_window_must_exist() -> None:
    raw_index, windows = _raw_index_and_windows()
    claims = ClaimsArtifact(
        claims=[
            Claim(
                claim_id="CLM001",
                source_window_id="W999",
                text="A claim.",
                evidence_span_ids=["S001"],
                evidence_quote="Product taste decides what to build.",
                target_type="concept",
                target_title="Product taste",
            )
        ]
    )

    with pytest.raises(ValidationError, match="references missing extraction window W999"):
        validate_claims(raw_index, windows, claims)


def test_claim_evidence_spans_must_exist_in_raw_index() -> None:
    raw_index, windows = _raw_index_and_windows()
    claims = ClaimsArtifact(
        claims=[
            Claim(
                claim_id="CLM001",
                source_window_id="W001",
                text="A claim.",
                evidence_span_ids=["S999"],
                evidence_quote="Product taste decides what to build.",
                target_type="concept",
                target_title="Product taste",
            )
        ]
    )

    with pytest.raises(ValidationError, match="references missing evidence span S999"):
        validate_claims(raw_index, windows, claims)


def test_claim_quote_is_display_text_not_exact_span_boundary() -> None:
    raw_index, windows = _raw_index_and_windows()
    claims = ClaimsArtifact(
        claims=[
            Claim(
                claim_id="CLM001",
                source_window_id="W001",
                text="A claim.",
                evidence_span_ids=["S001", "S003"],
                evidence_quote="This exact quote is absent.",
                target_type="concept",
                target_title="Product taste",
            )
        ]
    )

    validate_claims(raw_index, windows, claims)


def test_claim_quote_must_not_be_empty() -> None:
    raw_index, windows = _raw_index_and_windows()
    claims = ClaimsArtifact(
        claims=[
            Claim(
                claim_id="CLM001",
                source_window_id="W001",
                text="A claim.",
                evidence_span_ids=["S001"],
                evidence_quote=" ",
                target_type="concept",
                target_title="Product taste",
            )
        ]
    )

    with pytest.raises(ValidationError, match="evidence_quote must not be empty"):
        validate_claims(raw_index, windows, claims)


def _raw_index_and_windows() -> tuple[RawIndexArtifact, ExtractionWindowsArtifact]:
    raw_index = RawIndexArtifact(
        raw_path="raw/sample.md",
        raw_sha256="raw-sha",
        spans=[
            RawSpan(
                span_id="S001",
                raw_path="raw/sample.md",
                raw_sha256="raw-sha",
                start_char=0,
                end_char=45,
                text_sha256="text-sha-1",
                text="Product taste decides what to build.",
            ),
            RawSpan(
                span_id="S002",
                raw_path="raw/sample.md",
                raw_sha256="raw-sha",
                start_char=46,
                end_char=70,
                text_sha256="text-sha-2",
                text="A nearby supporting note.",
            ),
            RawSpan(
                span_id="S003",
                raw_path="raw/sample.md",
                raw_sha256="raw-sha",
                start_char=71,
                end_char=120,
                text_sha256="text-sha-3",
                text="The same point appears later in another window.",
            ),
        ],
    )
    windows = ExtractionWindowsArtifact(
        raw_path=raw_index.raw_path,
        raw_sha256=raw_index.raw_sha256,
        strategy="test",
        max_chars=100,
        windows=[
            ExtractionWindow(window_id="W001", source_span_ids=["S001", "S002"], text="window one"),
            ExtractionWindow(window_id="W002", source_span_ids=["S003"], text="window two"),
        ],
    )
    return raw_index, windows
