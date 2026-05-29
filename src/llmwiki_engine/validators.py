from __future__ import annotations

from .models import ClaimsArtifact, PagePlanArtifact, ProfileSpec, RawIndexArtifact, SemanticAggregationArtifact


class ValidationError(RuntimeError):
    pass


def validate_raw_index(raw_index: RawIndexArtifact) -> None:
    ids = [span.span_id for span in raw_index.spans]
    if len(ids) != len(set(ids)):
        raise ValidationError("raw_index contains duplicate span_id values")
    if not raw_index.spans:
        raise ValidationError("raw_index contains no spans")


def validate_aggregation(raw_index: RawIndexArtifact, aggregation: SemanticAggregationArtifact) -> None:
    span_ids = {span.span_id for span in raw_index.spans}
    for item in aggregation.aggregations:
        missing = set(item.source_span_ids) - span_ids
        if missing:
            raise ValidationError(f"{item.aggregation_id} references missing spans: {sorted(missing)}")


def validate_claims(raw_index: RawIndexArtifact, aggregation: SemanticAggregationArtifact, claims: ClaimsArtifact) -> None:
    span_by_id = {span.span_id: span for span in raw_index.spans}
    aggregation_ids = {item.aggregation_id for item in aggregation.aggregations}
    for claim in claims.claims:
        if claim.aggregation_id not in aggregation_ids:
            raise ValidationError(f"{claim.claim_id} references missing aggregation {claim.aggregation_id}")
        for span_id in claim.evidence_span_ids:
            span = span_by_id.get(span_id)
            if span is None:
                raise ValidationError(f"{claim.claim_id} references missing evidence span {span_id}")
            if claim.evidence_quote not in span.text:
                raise ValidationError(f"{claim.claim_id} evidence_quote is not inside span {span_id}")


def validate_page_plan(profile: ProfileSpec, claims: ClaimsArtifact, plan: PagePlanArtifact) -> None:
    claim_ids = {claim.claim_id for claim in claims.claims}
    page_types = set(profile.page_types)
    for page in plan.pages:
        if page.page_type not in page_types:
            raise ValidationError(f"page {page.title} uses unknown page_type {page.page_type}")
        missing = set(page.claim_ids) - claim_ids
        if missing:
            raise ValidationError(f"page {page.title} references missing claims: {sorted(missing)}")

