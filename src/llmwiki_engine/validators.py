from __future__ import annotations

import re
from pathlib import PurePosixPath

from .models import CandidateResolutionArtifact, RawPreparationArtifact, SourceDigestArtifact, StructuredIssue, WikiContextSnapshot, WikiMergePlanArtifact


class ValidationError(RuntimeError):
    def __init__(self, message: str, *, issues: list[StructuredIssue] | None = None):
        super().__init__(message)
        self.issues = issues or []


def _issue(
    code: str,
    message: str,
    *,
    field_path: str = "",
    validator_id: str = "",
    repairable: bool = False,
) -> StructuredIssue:
    return StructuredIssue(
        issue_code=code,
        field_path=field_path,
        validator_id=validator_id,
        message=message,
        repairability="repairable" if repairable else "non_repairable",
    )


def _raise_issue(
    code: str,
    message: str,
    *,
    field_path: str = "",
    validator_id: str = "",
    repairable: bool = False,
) -> None:
    raise ValidationError(message, issues=[_issue(code, message, field_path=field_path, validator_id=validator_id, repairable=repairable)])


def validate_raw_preparation(preparation: RawPreparationArtifact) -> None:
    if not preparation.source_raw_path.startswith("raw/"):
        _raise_issue("invalid_raw_path", "raw_preparation source_raw_path must point inside raw/", validator_id="validate_raw_preparation")
    if not preparation.prepared_markdown.strip():
        _raise_issue("missing_field", "raw_preparation prepared_markdown is empty", field_path="prepared_markdown", validator_id="validate_raw_preparation", repairable=True)
    if preparation.risk_level == "high" and not preparation.requires_human_review:
        _raise_issue("invalid_review_gate", "high risk raw_preparation must require human review", validator_id="validate_raw_preparation")


def validate_source_digest(digest: SourceDigestArtifact, *, language: str | None = None) -> None:
    if not digest.source_raw_path.startswith("raw/"):
        _raise_issue("invalid_raw_path", "source_digest source_raw_path must point inside raw/", field_path="source_raw_path", validator_id="validate_source_digest")
    if not digest.summary.strip():
        _raise_issue("missing_field", "source_digest summary must not be empty", field_path="summary", validator_id="validate_source_digest", repairable=True)
    if language == "zh-CN":
        require_zh_cn_user_text("source_digest summary", digest.summary)
        for index, takeaway in enumerate(digest.key_takeaways, start=1):
            require_zh_cn_user_text(f"source_digest key_takeaways[{index}]", takeaway)
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
                _raise_issue("missing_field", f"{group_name} contains a candidate without candidate_id", field_path=f"{group_name}.candidate_id", validator_id="validate_source_digest", repairable=True)
            if candidate.candidate_id in seen:
                _raise_issue("duplicate_candidate_id", f"source_digest contains duplicate candidate_id: {candidate.candidate_id}", field_path=f"{group_name}.candidate_id", validator_id="validate_source_digest")
            seen.add(candidate.candidate_id)
            if not candidate.name.strip():
                _raise_issue("missing_field", f"{candidate.candidate_id} name must not be empty", field_path=f"{group_name}.name", validator_id="validate_source_digest", repairable=True)
            if not candidate.type.strip():
                _raise_issue("missing_field", f"{candidate.candidate_id} type must not be empty", field_path=f"{group_name}.type", validator_id="validate_source_digest", repairable=True)
            if not candidate.one_sentence_summary.strip():
                _raise_issue("missing_field", f"{candidate.candidate_id} one_sentence_summary must not be empty", field_path=f"{group_name}.one_sentence_summary", validator_id="validate_source_digest", repairable=True)
            if language == "zh-CN":
                require_zh_cn_user_text(f"{candidate.candidate_id} one_sentence_summary", candidate.one_sentence_summary)
                require_zh_cn_user_text(f"{candidate.candidate_id} why_matters", candidate.why_matters)
                require_zh_cn_user_text(f"{candidate.candidate_id} wiki_value", candidate.wiki_value)
            if group_name != "weak_or_noise_items":
                if candidate.type.strip().lower() == "source":
                    _raise_issue("invalid_page_type", f"{candidate.candidate_id} formal candidates must not use source page type", field_path=f"{group_name}.type", validator_id="validate_source_digest")
                if not candidate.suggested_page_title.strip():
                    _raise_issue("missing_field", f"{candidate.candidate_id} suggested_page_title must not be empty", field_path=f"{group_name}.suggested_page_title", validator_id="validate_source_digest", repairable=True)


def validate_candidate_resolution(digest: SourceDigestArtifact, resolution: CandidateResolutionArtifact) -> None:
    candidate_ids = {candidate.candidate_id for candidate in digest.ingest_candidates()}
    weak_or_noise_ids = {candidate.candidate_id for candidate in digest.weak_or_noise_items}
    page_plan_ids = [item.page_plan_id for item in resolution.items]
    if len(page_plan_ids) != len(set(page_plan_ids)):
        _raise_issue("duplicate_page_plan_id", "candidate_resolution contains duplicate page_plan_id values", validator_id="validate_candidate_resolution")
    covered_ids: set[str] = set()
    for item in resolution.items:
        covered_ids.update(nonempty_source_candidate_ids(item.source_basis))
    missing = candidate_ids - covered_ids
    if missing:
        _raise_issue("missing_candidate_coverage", f"candidate_resolution misses approved candidates: {sorted(missing)}", validator_id="validate_candidate_resolution", repairable=True)
    unknown = covered_ids - candidate_ids
    if unknown:
        weak_or_noise_unknown = {item for item in unknown if item in weak_or_noise_ids or looks_like_noise_candidate_id(item)}
        if weak_or_noise_unknown == unknown:
            _raise_issue(
                "weak_noise_candidate_reference",
                f"candidate_resolution must remove weak/noise candidate references from formal page plans: {sorted(unknown)}",
                field_path="source_basis.source_candidate_ids",
                validator_id="validate_candidate_resolution",
                repairable=True,
            )
        _raise_issue("unknown_candidate_reference", f"candidate_resolution references unknown candidates: {sorted(unknown)}", validator_id="validate_candidate_resolution")
    for item in resolution.items:
        validate_wiki_relative_markdown_path(item.page_plan_id or item.display_title, item.candidate_target_path, "candidate_target_path")
        if not item.page_plan_id.strip():
            _raise_issue("missing_field", "candidate_resolution page_plan_id must not be empty", field_path="page_plan_id", validator_id="validate_candidate_resolution", repairable=True)
        if not nonempty_source_basis_refs(item.source_basis):
            _raise_issue("missing_field", f"{item.page_plan_id} source_basis must not be empty", field_path="source_basis", validator_id="validate_candidate_resolution", repairable=True)
        if not item.page_type.strip():
            _raise_issue("missing_field", f"{item.page_plan_id} page_type must not be empty", field_path="page_type", validator_id="validate_candidate_resolution", repairable=True)
        if not item.display_title.strip():
            _raise_issue("missing_field", f"{item.page_plan_id} display_title must not be empty", field_path="display_title", validator_id="validate_candidate_resolution", repairable=True)
        if not item.topic_summary.strip():
            _raise_issue("missing_field", f"{item.page_plan_id} topic_summary must not be empty", field_path="topic_summary", validator_id="validate_candidate_resolution", repairable=True)
        if not item.why_this_page.strip():
            _raise_issue("missing_field", f"{item.page_plan_id} why_this_page must not be empty", field_path="why_this_page", validator_id="validate_candidate_resolution", repairable=True)
        if not item.reason.strip():
            _raise_issue("missing_field", f"{item.page_plan_id} reason must not be empty", field_path="reason", validator_id="validate_candidate_resolution", repairable=True)
        if item.reason.strip().lower() in {"ignore", "ignored", "noise", "weak", "弱相关", "噪声"}:
            _raise_issue(
                "ignore_as_formal_item",
                f"{item.page_plan_id} uses ignore/noise reason as a formal page plan; remove this item.",
                field_path="reason",
                validator_id="validate_candidate_resolution",
                repairable=True,
            )


def looks_like_noise_candidate_id(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized.startswith(("noise", "weak", "ignore")) or normalized in {"n/a", "na"}


def nonempty_source_candidate_ids(source_basis: object) -> list[str]:
    ids = getattr(source_basis, "source_candidate_ids", [])
    return _nonempty_unique_strings(ids)


def nonempty_prepared_discovered_candidates(source_basis: object) -> list[str]:
    ids = getattr(source_basis, "prepared_discovered_candidates", [])
    return _nonempty_unique_strings(ids)


def nonempty_source_basis_refs(source_basis: object) -> list[str]:
    return _nonempty_unique_strings(
        [
            *nonempty_source_candidate_ids(source_basis),
            *nonempty_prepared_discovered_candidates(source_basis),
        ]
    )


def _nonempty_unique_strings(values: object) -> list[str]:
    refs: list[str] = []
    if not isinstance(values, list):
        return refs
    for value in values:
        ref = str(value).strip()
        if ref and ref not in refs:
            refs.append(ref)
    return refs


def validate_wiki_merge_plan(
    digest: SourceDigestArtifact,
    plan: WikiMergePlanArtifact,
    resolution: CandidateResolutionArtifact | None = None,
    snapshot: WikiContextSnapshot | None = None,
    *,
    language: str | None = None,
) -> None:
    candidate_ids = {candidate.candidate_id for candidate in digest.ingest_candidates()}
    page_plan_ids = [item.page_plan_id for item in plan.items]
    if len(page_plan_ids) != len(set(page_plan_ids)):
        _raise_issue("duplicate_page_plan_id", "wiki_merge_plan contains duplicate page_plan_id values", validator_id="validate_wiki_merge_plan")
    writable_paths = [item.canonical_target_path for item in plan.items if item.action in {"create", "update"}]
    if len(writable_paths) != len(set(writable_paths)):
        _raise_issue("duplicate_writable_target", "wiki_merge_plan contains duplicate writable target paths", validator_id="validate_wiki_merge_plan")
    covered_ids: set[str] = set()
    for item in plan.items:
        covered_ids.update(nonempty_source_candidate_ids(item.source_basis))
    missing = candidate_ids - covered_ids
    if missing:
        _raise_issue("missing_candidate_coverage", f"wiki_merge_plan misses approved candidates: {sorted(missing)}", validator_id="validate_wiki_merge_plan", repairable=True)
    unknown = covered_ids - candidate_ids
    if unknown:
        _raise_issue("unknown_candidate_reference", f"wiki_merge_plan references unknown candidates: {sorted(unknown)}", validator_id="validate_wiki_merge_plan")
    if resolution is not None:
        expected_page_plan_ids = {item.page_plan_id for item in resolution.items}
        actual_page_plan_ids = set(page_plan_ids)
        for item in plan.items:
            actual_page_plan_ids.update(item.merged_page_plan_ids)
        missing_page_plans = expected_page_plan_ids - actual_page_plan_ids
        if missing_page_plans:
            _raise_issue("missing_page_plan_coverage", f"wiki_merge_plan misses planned pages: {sorted(missing_page_plans)}", validator_id="validate_wiki_merge_plan", repairable=True)
        extra_page_plans = actual_page_plan_ids - expected_page_plan_ids
        if extra_page_plans:
            _raise_issue("unknown_page_plan_reference", f"wiki_merge_plan references unknown page_plan_id values: {sorted(extra_page_plans)}", validator_id="validate_wiki_merge_plan")
    if not plan.log_date.strip():
        _raise_issue("missing_field", "wiki_merge_plan log_date must not be empty", field_path="log_date", validator_id="validate_wiki_merge_plan", repairable=True)
    if not plan.context_snapshot_ref.strip():
        _raise_issue("missing_field", "wiki_merge_plan context_snapshot_ref must not be empty", field_path="context_snapshot_ref", validator_id="validate_wiki_merge_plan", repairable=True)
    inspected_by_id = {}
    known_related_targets: set[str] = set()
    if snapshot is not None:
        inspected_by_id = {
            context.page_plan_id: {hit.path for hit in context.hits}
            for context in snapshot.candidate_contexts.items
        }
        known_related_targets.update(entry.path for entry in snapshot.knowledge_metadata_pool)
    known_related_targets.update(
        item.canonical_target_path
        for item in plan.items
        if item.action in {"create", "update", "noop"} and not _is_source_graph_target(item.canonical_target_path)
    )
    for item in plan.items:
        validate_wiki_relative_markdown_path(item.page_plan_id, item.canonical_target_path, "canonical_target_path")
        for inspected_path in item.inspected_context_paths:
            validate_wiki_relative_markdown_path(item.page_plan_id, inspected_path, "inspected_context_paths")
        if item.page_type.strip().lower() == "source":
            _raise_issue("invalid_page_type", f"{item.page_plan_id} wiki_merge_plan items must not use source page type", field_path="page_type", validator_id="validate_wiki_merge_plan")
        if not nonempty_source_basis_refs(item.source_basis):
            _raise_issue("missing_field", f"{item.page_plan_id} source_basis must not be empty", field_path="source_basis", validator_id="validate_wiki_merge_plan", repairable=True)
        if not item.display_title.strip():
            _raise_issue("missing_field", f"{item.page_plan_id} display_title must not be empty", field_path="display_title", validator_id="validate_wiki_merge_plan", repairable=True)
        if item.matched_page is not None:
            validate_wiki_relative_markdown_path(item.page_plan_id, item.matched_page, "matched_page")
        if item.action == "update" and not item.matched_page:
            _raise_issue("missing_field", f"{item.page_plan_id} update action must include matched_page", field_path="matched_page", validator_id="validate_wiki_merge_plan", repairable=True)
        if snapshot is not None:
            inspected_paths = inspected_by_id.get(item.page_plan_id, set())
            if snapshot.knowledge_metadata_pool and not item.inspected_context_paths:
                _raise_issue("missing_field", f"{item.page_plan_id} must include inspected_context_paths from candidate_contexts", field_path="inspected_context_paths", validator_id="validate_wiki_merge_plan", repairable=True)
            unknown_inspected = set(item.inspected_context_paths) - inspected_paths
            if unknown_inspected:
                _raise_issue("unknown_inspected_context", f"{item.page_plan_id} inspected_context_paths must come from candidate_contexts: {sorted(unknown_inspected)}", field_path="inspected_context_paths", validator_id="validate_wiki_merge_plan")
            if item.action in {"update", "noop"} and item.matched_page and item.matched_page not in inspected_paths:
                _raise_issue("unknown_matched_page", f"{item.page_plan_id} matched_page must come from inspected_context_paths", field_path="matched_page", validator_id="validate_wiki_merge_plan")
            if item.action == "noop" and not item.matched_page:
                _raise_issue("missing_field", f"{item.page_plan_id} noop action must include matched_page from inspected_context_paths", field_path="matched_page", validator_id="validate_wiki_merge_plan", repairable=True)
            if item.action == "create" and item.strongest_overlap.strength == "medium" and create_reason_needs_repair(item.why_not_update):
                _raise_issue("medium_create_missing_why_not_update", f"{item.page_plan_id} create action with medium overlap must explain why_not_update", field_path="why_not_update", validator_id="validate_wiki_merge_plan", repairable=True)
            if item.action == "create" and item.strongest_overlap.strength == "strong":
                _raise_issue("strong_overlap_create", f"{item.page_plan_id} strong overlap create must use needs_human_decision", field_path="action", validator_id="validate_wiki_merge_plan")
        if item.action == "needs_human_decision" and item.apply_eligibility != "blocked":
            _raise_issue("needs_human_decision", f"{item.page_plan_id} needs_human_decision must be blocked", field_path="apply_eligibility", validator_id="validate_wiki_merge_plan")
        if item.action != "needs_human_decision" and item.apply_eligibility == "blocked":
            _raise_issue("needs_human_decision", f"{item.page_plan_id} blocked item must use needs_human_decision action", field_path="action", validator_id="validate_wiki_merge_plan")
        if item.action != "noop" and not item.section_plans:
            _raise_issue("missing_field", f"{item.page_plan_id} section_plans must not be empty", field_path="section_plans", validator_id="validate_wiki_merge_plan", repairable=True)
        if not item.new_understanding.strip():
            _raise_issue("missing_field", f"{item.page_plan_id} new_understanding must not be empty", field_path="new_understanding", validator_id="validate_wiki_merge_plan", repairable=True)
        if len(item.related_pages) > 3:
            _raise_issue("related_cap_exceeded", f"{item.page_plan_id} related_pages must contain at most 3 items", field_path="related_pages", validator_id="validate_wiki_merge_plan", repairable=True)
        if not item.related_pages and item.related_absence_reason is None:
            _raise_issue("missing_field", f"{item.page_plan_id} must include related_absence_reason when related_pages is empty", field_path="related_absence_reason", validator_id="validate_wiki_merge_plan", repairable=True)
        _validate_no_source_graph_links(
            item.page_plan_id,
            [
                item.display_title,
                item.finalization_reason,
                item.why_not_update,
                item.why_create_or_update,
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
                _raise_issue("source_related_target", f"{item.page_plan_id} related_pages must not point to source pages", field_path="related_pages", validator_id="validate_wiki_merge_plan", repairable=True)
            if related.target_path == item.canonical_target_path:
                _raise_issue("self_related_target", f"{item.page_plan_id} related_pages must not include self-link", field_path="related_pages", validator_id="validate_wiki_merge_plan", repairable=True)
            if snapshot is not None and related.target_path not in known_related_targets:
                _raise_issue("unknown_related_target", f"{item.page_plan_id} related_pages target is unknown: {related.target_path}", field_path="related_pages", validator_id="validate_wiki_merge_plan")
            if not related.display_title.strip():
                _raise_issue("missing_field", f"{item.page_plan_id} related_pages display_title must not be empty", field_path="related_pages.display_title", validator_id="validate_wiki_merge_plan", repairable=True)
            if not related.reason.strip():
                _raise_issue("missing_field", f"{item.page_plan_id} related_pages reason must not be empty", field_path="related_pages.reason", validator_id="validate_wiki_merge_plan", repairable=True)
            if language == "zh-CN":
                require_zh_cn_user_text(f"{item.page_plan_id} related_pages reason", related.reason)
            _validate_no_source_graph_links(item.page_plan_id, [related.display_title, related.reason])
        related_paths = [related.target_path for related in item.related_pages]
        if len(related_paths) != len(set(related_paths)):
            _raise_issue("duplicate_related_target", f"{item.page_plan_id} related_pages contains duplicate target_path values", field_path="related_pages", validator_id="validate_wiki_merge_plan", repairable=True)


def validate_wiki_relative_markdown_path(candidate_id: str, value: str, field_name: str) -> None:
    if not value.strip():
        _raise_issue("missing_field", f"{candidate_id} {field_name} must not be empty", field_path=field_name, validator_id="validate_wiki_relative_markdown_path", repairable=True)
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        _raise_issue("unsafe_path", f"{candidate_id} {field_name} must be a safe relative path", field_path=field_name, validator_id="validate_wiki_relative_markdown_path")
    if path.parts and path.parts[0] == "wiki":
        _raise_issue("normalizable_target_path", f"{candidate_id} {field_name} must be relative to wiki root", field_path=field_name, validator_id="validate_wiki_relative_markdown_path", repairable=True)
    if path.suffix != ".md":
        if field_name == "target_path":
            _raise_issue("normalizable_target_path", f"{candidate_id} target_path must end with .md", field_path=field_name, validator_id="validate_wiki_relative_markdown_path", repairable=True)
        _raise_issue("normalizable_target_path", f"{candidate_id} {field_name} entries must end with .md", field_path=field_name, validator_id="validate_wiki_relative_markdown_path", repairable=True)


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


def require_zh_cn_user_text(field_name: str, text: str) -> None:
    if looks_like_untranslated_english(text):
        _raise_issue(
            "zh_cn_untranslated_user_text",
            f"{field_name} must be Chinese for zh-CN vault",
            field_path=field_name,
            validator_id="require_zh_cn_user_text",
            repairable=True,
        )


def looks_like_untranslated_english(text: str) -> bool:
    stripped = _strip_markdown_noise(text)
    if not stripped:
        return False
    cjk_count = len(re.findall(r"[\u3400-\u9fff]", stripped))
    ascii_words = re.findall(r"\b[A-Za-z][A-Za-z'-]{2,}\b", stripped)
    if len(ascii_words) < 8:
        return False
    if cjk_count == 0:
        return True
    return len(ascii_words) >= 14 and cjk_count < max(4, len(ascii_words) // 3)


def create_reason_needs_repair(text: str) -> bool:
    normalized = " ".join(text.strip().split())
    if not normalized:
        return True
    if len(normalized) < 18:
        return True
    shallow_patterns = [
        "目标页不存在",
        "目标页面不存在",
        "target page does not exist",
        "页面类型不同",
        "类型不同",
        "不同类型",
        "更适合新建",
        "should be created",
    ]
    lowered = normalized.lower()
    if any(pattern in lowered for pattern in shallow_patterns) and not any(
        marker in normalized
        for marker in ["范围", "边界", "增量", "来源", "材料", "旧页", "已有页", "覆盖", "补充", "价值点", "例子", "适用"]
    ):
        return True
    if not create_reason_has_concrete_anchor(normalized):
        return True
    return False


def create_reason_has_concrete_anchor(text: str) -> bool:
    if any(marker in text for marker in ["`", "“", "”", "《", "》"]):
        return True
    ascii_terms = {
        term.lower()
        for term in re.findall(r"\b[A-Za-z][A-Za-z0-9_-]{2,}\b", text)
        if term.lower() not in {"create", "update", "noop", "related", "source", "scope", "delta"}
    }
    return bool(ascii_terms)


def _strip_markdown_noise(text: str) -> str:
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[\[[^\]]*\]\]", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    return text


def _validate_no_source_graph_links(candidate_id: str, values: list[str]) -> None:
    for value in values:
        if text_contains_source_graph_link(value):
            _raise_issue("source_graph_link", f"{candidate_id} must not contain source graph links", validator_id="validate_no_source_graph_links")


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
