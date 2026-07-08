from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from . import system_pages
from .models import (
    RelatedCandidateReport,
    RelatedMergeReport,
    RelatedPageRef,
    ValidationIssue,
)
from .text import strip_frontmatter


BODY_WIKILINK_LIMIT = 2


def render_related_section(
    *,
    current_path: str,
    related_pages: Iterable[RelatedPageRef],
    known_paths: set[str],
    path_titles: dict[str, str] | None = None,
    limit: int,
    report: RelatedMergeReport | None = None,
    owner_id: str = "render",
) -> str:
    active_report = report or RelatedMergeReport()
    titles = path_titles or {}
    selector = _RelatedSelector(
        owner_id=owner_id,
        current_path=current_path,
        known_paths=known_paths,
        path_titles=titles,
        title_to_path=_title_to_path(titles),
        report=active_report,
        limit=limit,
    )
    for ref in related_pages:
        selector.add(ref.target_path, ref.source, ref.reason, display_title=ref.display_title)
    if not selector.selected:
        return ""
    return "\n".join(f"- {system_pages.obsidian_link(ref.target_path, ref.display_title)}" for ref in selector.selected)


def final_markdown_link_issues(
    *,
    markdown: str,
    target_path: str,
    title: str,
    known_paths: set[str] | None = None,
    path_titles: dict[str, str] | None = None,
    body_wikilink_limit: int = BODY_WIKILINK_LIMIT,
) -> list[ValidationIssue]:
    body = strip_frontmatter(markdown)
    body_without_official = _remove_sections(body, {"相关页面"})
    titles = path_titles or {}
    title_to_path = _title_to_path(titles)
    issues: list[ValidationIssue] = []
    if _has_section(body_without_official, {"Related"}):
        issues.append(ValidationIssue(severity="error", code="model_related_section", message="最终 Markdown 包含模型自己写的 Related 章节。", path=target_path))
    body_targets = _wikilink_targets(body_without_official)
    if len(body_targets) > body_wikilink_limit:
        issues.append(ValidationIssue(severity="error", code="body_wikilink_too_many", message=f"最终 Markdown 正文 wikilink 超过 {body_wikilink_limit} 条。", path=target_path))
    for target in body_targets:
        if _is_self_link(target, target_path=target_path, title=title):
            issues.append(ValidationIssue(severity="error", code="self_wikilink", message="最终 Markdown 在系统相关页面章节外包含自链接。", path=target_path))
        normalized = system_pages.normalize_related_path(target)
        if normalized is not None and known_paths is not None and normalized not in known_paths:
            resolved = _resolve_target(target, known_paths=known_paths, title_to_path=title_to_path)
            normalized = resolved or normalized
        if normalized is None:
            issues.append(ValidationIssue(severity="error", code="body_wikilink_invalid_target", message="最终 Markdown 正文 wikilink 指向 raw/source/system/非法路径。", path=target_path))
            continue
        if known_paths is not None and normalized not in known_paths:
            issues.append(ValidationIssue(severity="error", code="body_wikilink_unknown_target", message="最终 Markdown 正文 wikilink 目标不在知识页集合中。", path=target_path))
    if _contains_graph_excluded_link(body_without_official):
        issues.append(ValidationIssue(severity="error", code="source_graph_link", message="最终 Markdown 在系统相关页面章节外包含 raw/source/system 图谱链接。", path=target_path))
    return issues


def canonicalize_body_wikilinks(markdown: str, *, known_paths: set[str], path_titles: dict[str, str]) -> str:
    if not known_paths:
        return markdown
    title_to_path = _title_to_path(path_titles)

    def replace(match: re.Match[str]) -> str:
        inner = match.group(1)
        target, separator, alias = inner.partition("|")
        resolved = _resolve_target(target, known_paths=known_paths, title_to_path=title_to_path)
        if resolved is None:
            return _plain_wikilink_text(target, alias if separator else "")
        display_target = resolved[:-3] if resolved.endswith(".md") else resolved
        if separator:
            return f"[[{display_target}|{alias.strip()}]]"
        return f"[[{display_target}]]"

    return re.sub(r"\[\[([^\]]+)\]\]", replace, markdown)


def _plain_wikilink_text(target: str, alias: str = "") -> str:
    if alias.strip():
        return alias.strip()
    text = _strip_wikilink(target)
    text = Path(text.replace("\\", "/")).stem
    for prefix in ["Concept_", "Overview_", "Design_", "Entity_", "Comparison_"]:
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    return text.replace("_", " ").strip() or target.strip()


def precanonical_link_errors(*, markdown: str, target_path: str, title: str, body_wikilink_limit: int = BODY_WIKILINK_LIMIT) -> list[str]:
    body = strip_frontmatter(markdown)
    errors: list[str] = []
    if _has_section(body, {"Related", "相关页面"}):
        errors.append("模型输出不能包含 Related/相关页面；引擎会统一渲染")
    if len(_wikilink_targets(body)) > body_wikilink_limit:
        errors.append(f"正文 wikilink 最多 {body_wikilink_limit} 条")
    if any(_is_self_link(target, target_path=target_path, title=title) for target in _wikilink_targets(body)):
        errors.append("模型输出不能包含自链接")
    if _contains_graph_excluded_link(body):
        errors.append("模型输出不能包含 raw/source/system 图谱链接")
    return errors


def body_wikilink_targets(markdown: str) -> list[str]:
    body = _remove_sections(strip_frontmatter(markdown), {"Related", "相关页面"})
    targets: list[str] = []
    for target in _wikilink_targets(body):
        normalized = system_pages.normalize_related_path(target)
        if normalized:
            targets.append(normalized)
    return targets


def related_section_targets(markdown: str) -> list[str]:
    body = strip_frontmatter(markdown)
    lines = body.splitlines()
    targets: list[str] = []
    in_related = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            heading = stripped[3:].strip().lower()
            in_related = heading in {"related", "相关页面"}
            continue
        if not in_related:
            continue
        for target in _wikilink_targets(line):
            normalized = system_pages.normalize_related_path(target)
            if normalized:
                targets.append(normalized)
    return targets


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
        limit: int,
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
    return "same_ingest" if source == "same_ingest" else "wiki_context"


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
