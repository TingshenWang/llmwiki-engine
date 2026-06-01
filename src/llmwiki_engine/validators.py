from __future__ import annotations

from .models import ClaimsArtifact, ExtractionWindowsArtifact, PagePlanArtifact, ProfileSpec, RawIndexArtifact, RawPreparationArtifact


class ValidationError(RuntimeError):
    pass


def validate_raw_preparation(preparation: RawPreparationArtifact) -> None:
    if not preparation.source_raw_path.startswith("raw/"):
        raise ValidationError("raw_preparation source_raw_path must point inside raw/")
    if not preparation.prepared_markdown.strip():
        raise ValidationError("raw_preparation prepared_markdown is empty")
    if preparation.risk_level == "high" and not preparation.requires_human_review:
        raise ValidationError("high risk raw_preparation must require human review")


def validate_raw_index(raw_index: RawIndexArtifact) -> None:
    ids = [span.span_id for span in raw_index.spans]
    if len(ids) != len(set(ids)):
        raise ValidationError("raw_index contains duplicate span_id values")
    if not raw_index.spans:
        raise ValidationError("raw_index contains no spans")
    if raw_index.input_kind == "prepared_raw" and not raw_index.original_raw_path:
        raise ValidationError("prepared raw_index must record original_raw_path")


def validate_extraction_windows(raw_index: RawIndexArtifact, windows: ExtractionWindowsArtifact) -> None:
    if windows.raw_path != raw_index.raw_path:
        raise ValidationError("extraction_windows raw_path must match raw_index raw_path")
    if windows.raw_sha256 != raw_index.raw_sha256:
        raise ValidationError("extraction_windows raw_sha256 must match raw_index raw_sha256")
    span_ids = {span.span_id for span in raw_index.spans}
    window_ids = [window.window_id for window in windows.windows]
    if len(window_ids) != len(set(window_ids)):
        raise ValidationError("extraction_windows contains duplicate window_id values")
    if not windows.windows:
        raise ValidationError("extraction_windows contains no windows")
    for window in windows.windows:
        if not window.source_span_ids:
            raise ValidationError(f"{window.window_id} contains no source spans")
        missing = set(window.source_span_ids) - span_ids
        if missing:
            raise ValidationError(f"{window.window_id} references missing spans: {sorted(missing)}")


def validate_claims(raw_index: RawIndexArtifact, windows: ExtractionWindowsArtifact, claims: ClaimsArtifact) -> None:
    span_by_id = {span.span_id: span for span in raw_index.spans}
    window_ids = {window.window_id for window in windows.windows}
    for claim in claims.claims:
        if claim.source_window_id not in window_ids:
            raise ValidationError(f"{claim.claim_id} references missing extraction window {claim.source_window_id}")
        if not claim.evidence_span_ids:
            raise ValidationError(f"{claim.claim_id} must reference at least one evidence span")
        if not claim.evidence_quote.strip():
            raise ValidationError(f"{claim.claim_id} evidence_quote must not be empty")
        for span_id in claim.evidence_span_ids:
            span = span_by_id.get(span_id)
            if span is None:
                raise ValidationError(f"{claim.claim_id} references missing evidence span {span_id}")


def validate_page_plan(profile: ProfileSpec, claims: ClaimsArtifact, plan: PagePlanArtifact) -> None:
    claim_ids = {claim.claim_id for claim in claims.claims}
    page_types = set(profile.page_types)
    for page in plan.pages:
        if page.page_type not in page_types:
            raise ValidationError(f"page {page.title} uses unknown page_type {page.page_type}")
        missing = set(page.claim_ids) - claim_ids
        if missing:
            raise ValidationError(f"page {page.title} references missing claims: {sorted(missing)}")
