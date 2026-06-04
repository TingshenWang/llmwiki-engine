from __future__ import annotations

import re
from pathlib import PurePosixPath

from .models import CandidateResolutionArtifact, RawPreparationArtifact, SourceDigestArtifact, WikiMergePlanArtifact


class ValidationError(RuntimeError):
    pass


def validate_raw_preparation(preparation: RawPreparationArtifact) -> None:
    if not preparation.source_raw_path.startswith("raw/"):
        raise ValidationError("raw_preparation source_raw_path must point inside raw/")
    if not preparation.prepared_markdown.strip():
        raise ValidationError("raw_preparation prepared_markdown is empty")
    if preparation.risk_level == "high" and not preparation.requires_human_review:
        raise ValidationError("high risk raw_preparation must require human review")


def validate_source_digest(digest: SourceDigestArtifact) -> None:
    if not digest.source_raw_path.startswith("raw/"):
        raise ValidationError("source_digest source_raw_path must point inside raw/")
    if not digest.summary.strip():
        raise ValidationError("source_digest summary must not be empty")
    seen: set[str] = set()
    for group_name, candidates in [
        ("entities", digest.entities),
        ("concepts", digest.concepts),
        ("designs", digest.designs),
        ("comparisons", digest.comparisons),
        ("open_questions", digest.open_questions),
        ("weak_or_noise_items", digest.weak_or_noise_items),
    ]:
        for candidate in candidates:
            if not candidate.candidate_id.strip():
                raise ValidationError(f"{group_name} contains a candidate without candidate_id")
            if candidate.candidate_id in seen:
                raise ValidationError(f"source_digest contains duplicate candidate_id: {candidate.candidate_id}")
            seen.add(candidate.candidate_id)
            if not candidate.name.strip():
                raise ValidationError(f"{candidate.candidate_id} name must not be empty")
            if not candidate.type.strip():
                raise ValidationError(f"{candidate.candidate_id} type must not be empty")
            if not candidate.one_sentence_summary.strip():
                raise ValidationError(f"{candidate.candidate_id} one_sentence_summary must not be empty")
            if group_name != "weak_or_noise_items":
                if candidate.type.strip().lower() == "source":
                    raise ValidationError(f"{candidate.candidate_id} formal candidates must not use source page type")
                if not candidate.suggested_page_title.strip():
                    raise ValidationError(f"{candidate.candidate_id} suggested_page_title must not be empty")


def validate_candidate_resolution(digest: SourceDigestArtifact, resolution: CandidateResolutionArtifact) -> None:
    candidate_ids = {candidate.candidate_id for candidate in digest.ingest_candidates()}
    page_plan_ids = [item.page_plan_id for item in resolution.items]
    if len(page_plan_ids) != len(set(page_plan_ids)):
        raise ValidationError("candidate_resolution contains duplicate page_plan_id values")
    covered_ids: set[str] = set()
    for item in resolution.items:
        covered_ids.update(item.source_basis.source_candidate_ids)
    missing = candidate_ids - covered_ids
    if missing:
        raise ValidationError(f"candidate_resolution misses approved candidates: {sorted(missing)}")
    unknown = covered_ids - candidate_ids
    if unknown:
        raise ValidationError(f"candidate_resolution references unknown candidates: {sorted(unknown)}")
    for item in resolution.items:
        validate_wiki_relative_markdown_path(item.page_plan_id or item.display_title, item.candidate_target_path, "candidate_target_path")
        if not item.page_plan_id.strip():
            raise ValidationError("candidate_resolution page_plan_id must not be empty")
        if not item.source_basis.source_candidate_ids and not item.source_basis.prepared_discovered_candidates:
            raise ValidationError(f"{item.page_plan_id} source_basis must not be empty")
        if not item.page_type.strip():
            raise ValidationError(f"{item.page_plan_id} page_type must not be empty")
        if not item.display_title.strip():
            raise ValidationError(f"{item.page_plan_id} display_title must not be empty")
        if not item.topic_summary.strip():
            raise ValidationError(f"{item.page_plan_id} topic_summary must not be empty")
        if not item.why_this_page.strip():
            raise ValidationError(f"{item.page_plan_id} why_this_page must not be empty")
        if not item.reason.strip():
            raise ValidationError(f"{item.page_plan_id} reason must not be empty")


def validate_wiki_merge_plan(
    digest: SourceDigestArtifact,
    plan: WikiMergePlanArtifact,
    resolution: CandidateResolutionArtifact | None = None,
) -> None:
    candidate_ids = {candidate.candidate_id for candidate in digest.ingest_candidates()}
    page_plan_ids = [item.page_plan_id for item in plan.items]
    if len(page_plan_ids) != len(set(page_plan_ids)):
        raise ValidationError("wiki_merge_plan contains duplicate page_plan_id values")
    writable_paths = [item.canonical_target_path for item in plan.items if item.action in {"create", "update"}]
    if len(writable_paths) != len(set(writable_paths)):
        raise ValidationError("wiki_merge_plan contains duplicate writable target paths")
    covered_ids: set[str] = set()
    for item in plan.items:
        covered_ids.update(item.source_basis.source_candidate_ids)
    missing = candidate_ids - covered_ids
    if missing:
        raise ValidationError(f"wiki_merge_plan misses approved candidates: {sorted(missing)}")
    unknown = covered_ids - candidate_ids
    if unknown:
        raise ValidationError(f"wiki_merge_plan references unknown candidates: {sorted(unknown)}")
    if resolution is not None:
        expected_page_plan_ids = {item.page_plan_id for item in resolution.items}
        actual_page_plan_ids = set(page_plan_ids)
        missing_page_plans = expected_page_plan_ids - actual_page_plan_ids
        if missing_page_plans:
            raise ValidationError(f"wiki_merge_plan misses planned pages: {sorted(missing_page_plans)}")
        extra_page_plans = actual_page_plan_ids - expected_page_plan_ids
        if extra_page_plans:
            raise ValidationError(f"wiki_merge_plan references unknown page_plan_id values: {sorted(extra_page_plans)}")
    if not plan.log_date.strip():
        raise ValidationError("wiki_merge_plan log_date must not be empty")
    if not plan.context_snapshot_ref.strip():
        raise ValidationError("wiki_merge_plan context_snapshot_ref must not be empty")
    for item in plan.items:
        validate_wiki_relative_markdown_path(item.page_plan_id, item.canonical_target_path, "canonical_target_path")
        if item.page_type.strip().lower() == "source":
            raise ValidationError(f"{item.page_plan_id} wiki_merge_plan items must not use source page type")
        if not item.source_basis.source_candidate_ids and not item.source_basis.prepared_discovered_candidates:
            raise ValidationError(f"{item.page_plan_id} source_basis must not be empty")
        if not item.display_title.strip():
            raise ValidationError(f"{item.page_plan_id} display_title must not be empty")
        if item.matched_page is not None:
            validate_wiki_relative_markdown_path(item.page_plan_id, item.matched_page, "matched_page")
        if item.action == "update" and not item.matched_page:
            raise ValidationError(f"{item.page_plan_id} update action must include matched_page")
        if item.action == "needs_human_decision" and item.apply_eligibility != "blocked":
            raise ValidationError(f"{item.page_plan_id} needs_human_decision must be blocked")
        if item.action != "needs_human_decision" and item.apply_eligibility == "blocked":
            raise ValidationError(f"{item.page_plan_id} blocked item must use needs_human_decision action")
        if item.action != "noop" and not item.section_plans:
            raise ValidationError(f"{item.page_plan_id} section_plans must not be empty")
        if not item.new_understanding.strip():
            raise ValidationError(f"{item.page_plan_id} new_understanding must not be empty")
        _validate_no_source_graph_links(
            item.page_plan_id,
            [
                item.display_title,
                item.prior_knowledge_state,
                item.new_understanding,
                item.changed_view,
                item.knowledge_delta,
                item.why_this_matters,
                *item.reuse_scenarios,
                *item.value_points,
                *item.section_plans.values(),
                *item.related_unresolved,
                *item.unresolved_related,
                *item.conflicts,
                *item.uncertainties,
                *item.quality_risks,
                item.blocked_reason,
                item.reason,
            ],
        )
        for related in item.related_pages:
            validate_wiki_relative_markdown_path(item.page_plan_id, related.target_path, "related_pages")
            if _is_source_graph_target(related.target_path):
                raise ValidationError(f"{item.page_plan_id} related_pages must not point to source pages")
            if related.target_path == item.canonical_target_path:
                raise ValidationError(f"{item.page_plan_id} related_pages must not include self-link")
            if not related.display_title.strip():
                raise ValidationError(f"{item.page_plan_id} related_pages display_title must not be empty")
            if not related.reason.strip():
                raise ValidationError(f"{item.page_plan_id} related_pages reason must not be empty")
            _validate_no_source_graph_links(item.page_plan_id, [related.display_title, related.reason])


def validate_wiki_relative_markdown_path(candidate_id: str, value: str, field_name: str) -> None:
    if not value.strip():
        raise ValidationError(f"{candidate_id} {field_name} must not be empty")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValidationError(f"{candidate_id} {field_name} must be a safe relative path")
    if path.parts and path.parts[0] == "wiki":
        raise ValidationError(f"{candidate_id} {field_name} must be relative to wiki root")
    if path.suffix != ".md":
        if field_name == "target_path":
            raise ValidationError(f"{candidate_id} target_path must end with .md")
        raise ValidationError(f"{candidate_id} {field_name} entries must end with .md")


def text_contains_source_graph_link(text: str) -> bool:
    lowered = text.lower()
    if "[[source_" in lowered:
        return True
    for match in re.finditer(r"\[\[([^\]]+)\]\]", text):
        target = match.group(1).split("|", 1)[0]
        if _is_source_graph_target(target):
            return True
    for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", text):
        if _is_source_graph_target(match.group(1)):
            return True
    for match in re.finditer(r"""href\s*=\s*["']([^"']+)["']""", text, flags=re.IGNORECASE):
        if _is_source_graph_target(match.group(1)):
            return True
    return False


def _validate_no_source_graph_links(candidate_id: str, values: list[str]) -> None:
    for value in values:
        if text_contains_source_graph_link(value):
            raise ValidationError(f"{candidate_id} must not contain source graph links")


def _is_source_graph_target(value: str) -> bool:
    target = value.strip().strip("`").strip("<>").replace("\\", "/")
    target = target.split("#", 1)[0].split("?", 1)[0]
    while target.startswith("./"):
        target = target[2:]
    target = target.lstrip("/")
    if not target:
        return False
    parts = [part.lower() for part in PurePosixPath(target).parts if part not in {"", "."}]
    if not parts:
        return False
    if parts[0] == "wiki":
        parts = parts[1:]
    if parts and parts[0] == "sources":
        return True
    return parts[-1].startswith("source_")
