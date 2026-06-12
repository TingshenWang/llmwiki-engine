from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from . import system_pages
from .io import read_text
from .models import (
    CandidateContexts,
    CandidatePages,
    CompositionItem,
    CompositionPlan,
    MergeDecision,
    MergePlan,
    RelatedCandidateReport,
    RelatedMergeReport,
    RelatedPageRef,
    SourceDigest,
    ValidationIssue,
    WikiKnowledgeEntry,
    WikiSnapshot,
)
from .text import strip_frontmatter


MODEL_RELATED_SUGGESTION_LIMIT = 2
FINAL_RELATED_LIMIT = system_pages.RELATED_LINK_LIMIT


def finalize_merge_plan_related(
    plan: MergePlan,
    *,
    candidate_pages: CandidatePages,
    digest: SourceDigest,
    snapshot: WikiSnapshot,
    contexts: CandidateContexts,
) -> tuple[MergePlan, RelatedMergeReport]:
    decisions_by_candidate = {decision.candidate_page_id: decision for decision in plan.decisions}
    candidate_by_id = {page.candidate_page_id: page for page in candidate_pages.pages}
    candidate_by_source_id = {
        source_id: page
        for page in candidate_pages.pages
        for source_id in page.source_candidate_ids
    }
    source_by_id = {candidate.candidate_id: candidate for candidate in digest.candidates()}
    context_by_id = {context.candidate_page_id: context for context in contexts.items}
    known_paths = _known_paths(snapshot.entries, (decision.target_path for decision in plan.decisions))
    path_titles = _path_titles(snapshot.entries, candidate_pages, decisions_by_candidate)
    title_to_path = _title_to_path(path_titles)
    report = RelatedMergeReport()
    updated_decisions: list[MergeDecision] = []

    for decision in plan.decisions:
        page = candidate_by_id.get(decision.candidate_page_id)
        if page is None or decision.action == "noop" or not decision.target_path:
            updated_decisions.append(decision)
            continue
        selector = _RelatedSelector(
            owner_id=decision.candidate_page_id,
            current_path=decision.target_path,
            known_paths=known_paths,
            path_titles=path_titles,
            title_to_path=title_to_path,
            report=report,
        )
        unresolved = list(decision.related_unresolved)
        for index, ref in enumerate(decision.related_pages):
            if index >= MODEL_RELATED_SUGGESTION_LIMIT:
                selector.cutoff(ref.target_path, ref.source, ref.reason, reject_reason="model_suggestion_limit")
                continue
            selector.add(ref.target_path, ref.source, ref.reason, display_title=ref.display_title)

        for source_id in page.source_candidate_ids:
            source_candidate = source_by_id.get(source_id)
            if source_candidate is None:
                continue
            for related_id in source_candidate.related_candidates:
                related_page = candidate_by_source_id.get(related_id)
                related_target = None
                related_title = related_id
                if related_page is not None:
                    related_decision = decisions_by_candidate.get(related_page.candidate_page_id)
                    if related_decision and related_decision.action != "noop":
                        related_target = related_decision.target_path
                    related_title = related_page.title
                else:
                    related_target = _resolve_target(related_id, known_paths=known_paths, title_to_path=title_to_path)
                if not related_target:
                    unresolved.append(related_id)
                    selector.filtered(related_id, "source_digest", "unresolved_source_candidate")
                    continue
                selector.add(
                    related_target,
                    "source_digest",
                    "同一原始材料中被标记为上下游、补充或对照关系。",
                    display_title=related_title,
                )

        context = context_by_id.get(decision.candidate_page_id)
        if context:
            for hit in context.hits:
                if hit.score <= 0:
                    continue
                selector.add(
                    hit.path,
                    "wiki_context",
                    hit.reason or "Top5 召回旧页，可作为背景补充。",
                    display_title=hit.title,
                )

        related_pages = selector.selected
        related_absence_reason = "" if related_pages else (decision.related_absence_reason or _absence_reason(unresolved))
        updated_decisions.append(
            decision.model_copy(
                update={
                    "related_pages": related_pages,
                    "related_absence_reason": related_absence_reason,
                    "related_unresolved": _dedupe([item for item in unresolved if item.strip()]),
                }
            )
        )

    return plan.model_copy(update={"decisions": updated_decisions}), report


def finalize_composition_related(
    composition: CompositionPlan,
    *,
    snapshot: WikiSnapshot,
    vault: Path,
) -> tuple[CompositionPlan, RelatedMergeReport]:
    known_paths = _known_paths(snapshot.entries, (item.target_path for item in composition.items))
    path_titles = _path_titles(snapshot.entries)
    for item in composition.items:
        path_titles.setdefault(item.target_path, _display_title(item.target_path))
    title_to_path = _title_to_path(path_titles)
    report = RelatedMergeReport()
    updated_items: list[CompositionItem] = []

    for item in composition.items:
        selector = _RelatedSelector(
            owner_id=item.final_page_id,
            current_path=item.target_path,
            known_paths=known_paths,
            path_titles=path_titles,
            title_to_path=title_to_path,
            report=report,
        )
        existing_path = vault / "wiki" / item.target_path
        if existing_path.exists():
            for path in system_pages.parse_related_paths(read_text(existing_path)):
                selector.add(
                    path,
                    "existing_related",
                    "既有页面已记录该相关页面，保留可追溯链接。",
                    display_title=path_titles.get(system_pages.normalize_related_path(path) or "", ""),
                )
        for ref in item.related_pages:
            selector.add(ref.target_path, ref.source, ref.reason, display_title=ref.display_title)

        related_pages = selector.selected
        related_absence_reason = "" if related_pages else (item.related_absence_reason or "no_related_candidate_after_filter")
        updated_items.append(item.model_copy(update={"related_pages": related_pages, "related_absence_reason": related_absence_reason}))

    return composition.model_copy(update={"items": updated_items}), report


def merge_reports(*reports: RelatedMergeReport) -> RelatedMergeReport:
    merged: list[RelatedCandidateReport] = []
    for report in reports:
        merged.extend(report.candidates)
    return RelatedMergeReport(candidates=merged)


def render_related_section(
    *,
    current_path: str,
    related_pages: Iterable[RelatedPageRef],
    known_paths: set[str],
    path_titles: dict[str, str] | None = None,
) -> str:
    report = RelatedMergeReport()
    titles = path_titles or {}
    selector = _RelatedSelector(
        owner_id="render",
        current_path=current_path,
        known_paths=known_paths,
        path_titles=titles,
        title_to_path=_title_to_path(titles),
        report=report,
    )
    for ref in related_pages:
        selector.add(ref.target_path, ref.source, ref.reason, display_title=ref.display_title)
    if not selector.selected:
        return "- 暂无相关页面记录。"
    return "\n".join(
        f"- {system_pages.obsidian_link(ref.target_path, ref.display_title)}：{ref.reason}"
        for ref in selector.selected
    )


def validate_related_plan(
    *,
    merge_plan: MergePlan | None,
    composition_plan: CompositionPlan | None,
    snapshot: WikiSnapshot | None,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    entries = snapshot.entries if snapshot else []
    plan_targets = [decision.target_path for decision in merge_plan.decisions] if merge_plan else []
    composition_targets = [item.target_path for item in composition_plan.items] if composition_plan else []
    known_paths = _known_paths(entries, [*plan_targets, *composition_targets])

    if merge_plan:
        for decision in merge_plan.decisions:
            if decision.action == "noop":
                continue
            issues.extend(
                _validate_related_refs(
                    owner_id=decision.candidate_page_id,
                    current_path=decision.target_path or "",
                    refs=decision.related_pages,
                    known_paths=known_paths,
                    path=decision.target_path,
                    code_prefix="merge",
                )
            )
    if composition_plan:
        for item in composition_plan.items:
            issues.extend(
                _validate_related_refs(
                    owner_id=item.final_page_id,
                    current_path=item.target_path,
                    refs=item.related_pages,
                    known_paths=known_paths,
                    path=item.target_path,
                    code_prefix="composition",
                )
            )
    return issues


def final_markdown_link_issues(*, markdown: str, target_path: str, title: str) -> list[ValidationIssue]:
    body = strip_frontmatter(markdown)
    body_without_official = _remove_sections(body, {"相关页面"})
    issues: list[ValidationIssue] = []
    if _has_section(body_without_official, {"Related"}):
        issues.append(ValidationIssue(severity="error", code="model_related_section", message="最终 Markdown 包含模型自己写的 Related 章节。", path=target_path))
    for target in _wikilink_targets(body_without_official):
        if _is_self_link(target, target_path=target_path, title=title):
            issues.append(ValidationIssue(severity="error", code="self_wikilink", message="最终 Markdown 在系统相关页面章节外包含自链接。", path=target_path))
            break
    if _contains_graph_excluded_link(body_without_official):
        issues.append(ValidationIssue(severity="error", code="source_graph_link", message="最终 Markdown 在系统相关页面章节外包含 raw/source/system 图谱链接。", path=target_path))
    return issues


def precanonical_link_errors(*, markdown: str, target_path: str, title: str) -> list[str]:
    body = strip_frontmatter(markdown)
    errors: list[str] = []
    if _has_section(body, {"Related", "相关页面"}):
        errors.append("模型输出不能包含 Related/相关页面；引擎会统一渲染")
    if any(_is_self_link(target, target_path=target_path, title=title) for target in _wikilink_targets(body)):
        errors.append("模型输出不能包含自链接")
    if _contains_graph_excluded_link(body):
        errors.append("模型输出不能包含 raw/source/system 图谱链接")
    return errors


def render_related_report(report: RelatedMergeReport) -> str:
    lines = ["# 相关页面合并报告", ""]
    if not report.candidates:
        lines.append("- 暂无相关页面候选。")
        return "\n".join(lines) + "\n"
    for item in report.candidates:
        target = f"`{item.target_path}`" if item.target_path else "`<empty>`"
        reason = f" ({item.reject_reason})" if item.reject_reason else ""
        lines.append(f"- {item.owner_id}：{_decision_label(item.decision)} {target}；来源={item.source}{reason}")
    return "\n".join(lines) + "\n"


def _decision_label(decision: str) -> str:
    return {"kept": "保留", "filtered": "过滤", "cutoff": "截断"}.get(decision, decision)


class _RelatedSelector:
    def __init__(
        self,
        *,
        owner_id: str,
        current_path: str,
        known_paths: set[str],
        path_titles: dict[str, str],
        title_to_path: dict[str, str],
        report: RelatedMergeReport,
        limit: int = FINAL_RELATED_LIMIT,
    ) -> None:
        self.owner_id = owner_id
        self.current_path = system_pages.normalize_related_path(current_path) or current_path
        self.known_paths = known_paths
        self.path_titles = path_titles
        self.title_to_path = title_to_path
        self.report = report
        self.limit = limit
        self.selected: list[RelatedPageRef] = []
        self.seen: set[str] = set()

    def add(self, target: str, source: str, reason: str, *, display_title: str = "") -> None:
        normalized = _resolve_target(target, known_paths=self.known_paths, title_to_path=self.title_to_path)
        if normalized is None:
            self.filtered(target, source, "unknown_or_system_target", reason=reason)
            return
        if normalized == self.current_path:
            self.filtered(normalized, source, "self_link", reason=reason)
            return
        if normalized not in self.known_paths:
            self.filtered(normalized, source, "unknown_target", reason=reason)
            return
        if normalized in self.seen:
            self.filtered(normalized, source, "duplicate", reason=reason)
            return
        if len(self.selected) >= self.limit:
            self.cutoff(normalized, source, reason, reject_reason="final_related_limit")
            return
        title = display_title.strip() or self.path_titles.get(normalized, "") or _display_title(normalized)
        clean_reason = reason.strip() or "相关主题，可作为背景补充。"
        ref = RelatedPageRef(target_path=normalized, display_title=title, source=_related_ref_source(source), reason=clean_reason)
        self.selected.append(ref)
        self.seen.add(normalized)
        self.report.candidates.append(
            RelatedCandidateReport(
                owner_id=self.owner_id,
                current_path=self.current_path,
                target_path=normalized,
                display_title=title,
                source=source,
                decision="kept",
                reason=clean_reason,
            )
        )

    def filtered(self, target: str, source: str, reject_reason: str, *, reason: str = "") -> None:
        self.report.candidates.append(
            RelatedCandidateReport(
                owner_id=self.owner_id,
                current_path=self.current_path,
                target_path=target,
                source=source,
                decision="filtered",
                reject_reason=reject_reason,
                reason=reason,
            )
        )

    def cutoff(self, target: str, source: str, reason: str, *, reject_reason: str) -> None:
        self.report.candidates.append(
            RelatedCandidateReport(
                owner_id=self.owner_id,
                current_path=self.current_path,
                target_path=target,
                source=source,
                decision="cutoff",
                reject_reason=reject_reason,
                reason=reason,
            )
        )


def _validate_related_refs(
    *,
    owner_id: str,
    current_path: str,
    refs: list[RelatedPageRef],
    known_paths: set[str],
    path: str | None,
    code_prefix: str,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if len(refs) > FINAL_RELATED_LIMIT:
        issues.append(ValidationIssue(severity="error", code=f"{code_prefix}_related_too_many", message=f"{owner_id} 的相关页面超过 {FINAL_RELATED_LIMIT} 个。", path=path))
    current = system_pages.normalize_related_path(current_path) or current_path
    seen: set[str] = set()
    for ref in refs:
        normalized = system_pages.normalize_related_path(ref.target_path)
        if normalized is None:
            issues.append(ValidationIssue(severity="error", code=f"{code_prefix}_related_invalid_target", message=f"{owner_id} 的相关页面目标是 source/system/非法路径。", path=path))
            continue
        if normalized == current:
            issues.append(ValidationIssue(severity="error", code=f"{code_prefix}_related_self_link", message=f"{owner_id} 的相关页面目标指向自身。", path=path))
        if normalized not in known_paths:
            issues.append(ValidationIssue(severity="error", code=f"{code_prefix}_related_unknown_target", message=f"{owner_id} 的相关页面目标不在 wiki 快照或本次写入中。", path=path))
        if normalized in seen:
            issues.append(ValidationIssue(severity="error", code=f"{code_prefix}_related_duplicate", message=f"{owner_id} 的相关页面目标重复。", path=path))
        seen.add(normalized)
        if not ref.display_title.strip() or not ref.reason.strip():
            issues.append(ValidationIssue(severity="error", code=f"{code_prefix}_related_missing_text", message=f"{owner_id} 的相关页面需要 display_title 和 reason。", path=path))
    return issues


def _known_paths(entries: Iterable[WikiKnowledgeEntry], targets: Iterable[str | None] = ()) -> set[str]:
    paths: set[str] = set()
    for entry in entries:
        normalized = system_pages.normalize_related_path(entry.path)
        if normalized:
            paths.add(normalized)
    for target in targets:
        if not target:
            continue
        normalized = system_pages.normalize_related_path(target)
        if normalized:
            paths.add(normalized)
    return paths


def _path_titles(
    entries: Iterable[WikiKnowledgeEntry],
    candidate_pages: CandidatePages | None = None,
    decisions_by_candidate: dict[str, MergeDecision] | None = None,
) -> dict[str, str]:
    titles = {entry.path: entry.title for entry in entries}
    if candidate_pages and decisions_by_candidate:
        for page in candidate_pages.pages:
            decision = decisions_by_candidate.get(page.candidate_page_id)
            if decision and decision.target_path:
                titles[decision.target_path] = page.title
    return titles


def _title_to_path(path_titles: dict[str, str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for path, title in path_titles.items():
        values[_norm(title)] = path
        values[_norm(Path(path).stem)] = path
    return values


def _resolve_target(target: str, *, known_paths: set[str], title_to_path: dict[str, str]) -> str | None:
    text = _strip_wikilink(target)
    normalized = system_pages.normalize_related_path(text)
    if normalized and normalized in known_paths:
        return normalized
    title_hit = title_to_path.get(_norm(text))
    if title_hit:
        return title_hit
    return None


def _display_title(path: str) -> str:
    return Path(path).stem.replace("_", " ")


def _absence_reason(unresolved: list[str]) -> str:
    if unresolved:
        return "unresolved_related_candidate"
    return "no_related_candidate_after_filter"


def _remove_sections(markdown: str, headings: set[str]) -> str:
    wanted = {heading.lower() for heading in headings}
    lines = markdown.splitlines()
    result: list[str] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped.startswith("## ") and stripped[3:].strip().lower() in wanted:
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("## "):
                index += 1
            continue
        result.append(lines[index])
        index += 1
    return "\n".join(result)


def _has_section(markdown: str, headings: set[str]) -> bool:
    wanted = {heading.lower() for heading in headings}
    return any(line.strip().startswith("## ") and line.strip()[3:].strip().lower() in wanted for line in markdown.splitlines())


def _wikilink_targets(markdown: str) -> list[str]:
    targets: list[str] = []
    for match in re.finditer(r"\[\[([^\]]+)\]\]", markdown):
        target = match.group(1).split("|", 1)[0].strip()
        if target:
            targets.append(target)
    return targets


def _contains_graph_excluded_link(markdown: str) -> bool:
    for target in _wikilink_targets(markdown):
        if system_pages.is_graph_excluded_target(target):
            return True
    for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", markdown):
        if system_pages.is_graph_excluded_target(match.group(1)):
            return True
    for match in re.finditer(r"""href\s*=\s*["']([^"']+)["']""", markdown, flags=re.IGNORECASE):
        if system_pages.is_graph_excluded_target(match.group(1)):
            return True
    return False


def _related_ref_source(source: str) -> str:
    return "source_digest" if source == "source_digest" else "wiki_context"


def _strip_wikilink(value: str) -> str:
    text = value.strip()
    if text.startswith("[[") and text.endswith("]]"):
        text = text[2:-2].split("|", 1)[0].strip()
    return text


def _is_self_link(target: str, *, target_path: str, title: str) -> bool:
    normalized_target = system_pages.normalize_related_path(target)
    normalized_current = system_pages.normalize_related_path(target_path)
    if normalized_target and normalized_current and normalized_target == normalized_current:
        return True
    text = _strip_wikilink(target)
    stem = Path(target_path).stem
    return _norm(text) in {_norm(title), _norm(stem)}


def _norm(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
