from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from . import wiki_markup as _wiki_markup
from . import update_preservation as _update_preservation
from .models import (
    DraftPageItem,
    DraftRenderingArtifact,
    StructuredIssue,
    WikiMergePlanArtifact,
    WikiMergePlanItem,
)
from .validators import (
    ValidationError as ContractValidationError,
    looks_like_untranslated_english,
    text_contains_source_graph_link,
)


SYSTEM_CORE_SECTION_TITLES = {
    "相关页面",
    "related",
    "related pages",
    "矛盾与未决问题",
    "未决问题",
    "open questions",
    "tensions / open questions",
}

DRAFT_SELF_TALK_MARKERS = (
    "我记错",
    "可能我记错",
    "检查原文",
    "查看原文",
    "我会修正",
    "输出已经确定",
    "我还没输出",
    "但现在我们无法修改",
    "等等，原文",
    "目前wiki中无此页面",
    "当前wiki没有相关知识页",
    "当前wiki没有此页面",
    "创建后可与",
    "创建后可和",
    "创建后可互链",
)


def finalize_draft_change_summary(summary: str, item: WikiMergePlanItem) -> str:
    summary = normalize_stable_brand_typos(summary.strip())
    if item.action == "create" and (not summary or looks_like_untranslated_english(summary)):
        title = item.display_title.strip() or Path(item.canonical_target_path).stem
        return f"创建 {title} 页面。"
    return summary


def finalize_draft_source_coverage_notes(notes: str, item: WikiMergePlanItem) -> str:
    notes = normalize_stable_brand_typos(notes.strip())
    if not notes or looks_like_untranslated_english(notes):
        title = item.display_title.strip() or Path(item.canonical_target_path).stem
        basis = "本轮来源摘录"
        if item.action == "update":
            basis = "本轮来源摘录与已检查旧页"
        return f"依据{basis}中与「{title}」相关的内容生成；未被来源支撑的细节保留为未决问题。"
    notes = re.sub(r"\bapproved_digest\b", "来源摘要", notes)
    notes = re.sub(r"\bsource_excerpt_pack\b", "来源摘录包", notes)
    notes = re.sub(r"\bwiki_context_snapshot\b", "已检查 wiki 上下文", notes)
    notes = re.sub(r"\bsnippets?\b", "摘录", notes, flags=re.IGNORECASE)
    return notes


def canonical_draft_page_content(page: DraftPageItem, *, item: WikiMergePlanItem) -> dict[str, Any]:
    summary = normalize_stable_brand_typos(page.summary.strip())
    body = page.body_markdown.strip()
    body = normalize_core_body_markdown(body)
    open_questions = normalize_stable_brand_typos(page.open_questions.strip())
    return {
        "summary": summary,
        "body_markdown": body,
        "open_questions": open_questions,
    }


def canonicalize_draft_artifact(artifact: DraftRenderingArtifact, plan: WikiMergePlanArtifact) -> DraftRenderingArtifact:
    plan_by_id = {item.page_plan_id: item for item in plan.items}
    pages: list[DraftPageItem] = []
    changed = False
    for page in artifact.pages:
        item = plan_by_id.get(page.page_plan_id)
        if item is None:
            pages.append(page)
            continue
        canonical_page = page.model_copy(update=canonical_draft_page_content(page, item=item))
        pages.append(canonical_page)
        changed = changed or canonical_page != page
    if not changed:
        return artifact
    return artifact.model_copy(update={"pages": pages})


def normalize_core_body_markdown(body: str) -> str:
    body = normalize_stable_brand_typos(body.strip())
    if body.startswith("---"):
        parts = body.split("---", 2)
        if len(parts) >= 3:
            body = parts[2].strip()
    body = demote_core_body_headings(body)
    body = strip_core_body_system_sections(body)
    return body.strip()


def demote_core_body_headings(body: str) -> str:
    lines: list[str] = []
    for line in body.splitlines():
        match = re.match(r"^(#{1,2})\s+(.+?)\s*$", line)
        if match:
            lines.append(f"### {match.group(2).strip()}")
            continue
        lines.append(line)
    return "\n".join(lines)


def strip_core_body_system_sections(body: str) -> str:
    lines: list[str] = []
    skip_heading_level: int | None = None
    fence_char = ""
    fence_length = 0
    for line in body.splitlines():
        if fence_char:
            if skip_heading_level is None:
                lines.append(line)
            if closing_fence_line(line, fence_char, fence_length):
                fence_char = ""
                fence_length = 0
            continue
        if match := opening_fence_line(line):
            marker = match.group("marker")
            fence_char = marker[0]
            fence_length = len(marker)
            if skip_heading_level is None:
                lines.append(line)
            continue
        heading_match = re.match(r"^\s{0,3}(?P<marks>#{1,6})\s+(?P<title>.+?)\s*$", line)
        if heading_match:
            title = re.sub(r"\s+", " ", heading_match.group("title").strip()).strip("#:： ")
            level = len(heading_match.group("marks"))
            if title.casefold() in SYSTEM_CORE_SECTION_TITLES:
                skip_heading_level = level
                continue
            if skip_heading_level is not None and level <= skip_heading_level:
                skip_heading_level = None
        if skip_heading_level is None:
            lines.append(line)
    return "\n".join(lines).strip()


def core_body_system_heading(body: str) -> str | None:
    for line in body.splitlines():
        match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", line)
        if not match:
            continue
        normalized_title = re.sub(r"\s+", " ", match.group(1).strip()).strip("#:： ")
        if normalized_title.casefold() in SYSTEM_CORE_SECTION_TITLES:
            return normalized_title
    return None


def normalize_stable_brand_typos(text: str) -> str:
    replacements = [
        ("Clade Code", "Claude Code"),
        ("ClaudeCode", "Claude Code"),
        ("Anropinic", "Anthropic"),
        ("Anthopic", "Anthropic"),
        ("ManagedAgents", "Managed Agents"),
        ("Borris Cherny", "Boris Cherny"),
        ("Borris", "Boris"),
    ]
    for wrong, right in replacements:
        text = re.sub(rf"(?<![A-Za-z0-9]){re.escape(wrong)}(?![A-Za-z0-9])", right, text)
    return text


def validate_draft_rendering(artifact: DraftRenderingArtifact, plan: WikiMergePlanArtifact, *, language: str | None = None) -> None:
    plan_by_id = {item.page_plan_id: item for item in plan.items}
    required_ids = {item.page_plan_id for item in plan.items if item.action in {"create", "update"}}
    actual_ids = {page.page_plan_id for page in artifact.pages}
    missing = required_ids - actual_ids
    if missing:
        raise_draft_issue("missing_page_plan_coverage", f"draft_rendering misses page_plan_id(s): {sorted(missing)}", field_path="pages")
    extra = actual_ids - required_ids
    if extra:
        raise_draft_issue("unknown_page_plan_reference", f"draft_rendering contains unexpected page_plan_id(s): {sorted(extra)}", field_path="pages")
    artifact = canonicalize_draft_artifact(artifact, plan)
    for page in artifact.pages:
        plan_item = plan_by_id.get(page.page_plan_id)
        display_title = plan_item.display_title if plan_item is not None else ""
        if not page.canonical_target_path.strip():
            raise_draft_issue("missing_field", f"{page.page_plan_id} canonical_target_path must not be empty", field_path="canonical_target_path")
        if not _update_preservation.draft_page_summary(page):
            raise_draft_issue("missing_field", f"{page.page_plan_id} summary must not be empty", field_path="summary")
        if not _update_preservation.draft_page_core_markdown(page):
            raise_draft_issue("missing_field", f"{page.page_plan_id} body_markdown must not be empty", field_path="body_markdown")
        if not page.change_summary.strip():
            raise_draft_issue("missing_field", f"{page.page_plan_id} change_summary must not be empty", field_path="change_summary")
        if not page.source_coverage_notes.strip():
            raise_draft_issue("missing_field", f"{page.page_plan_id} source_coverage_notes must not be empty", field_path="source_coverage_notes")
        system_heading = core_body_system_heading(_update_preservation.draft_page_core_markdown(page))
        if system_heading:
            raise_draft_issue(
                "forbidden_system_section_in_core",
                (
                    f"{page.page_plan_id} body_markdown contains system-rendered section heading `{system_heading}`; "
                    "keep related pages and open questions outside the free core body."
                ),
                field_path="body_markdown",
            )
        for field_name, body in [
            ("summary", _update_preservation.draft_page_summary(page)),
            ("body_markdown", _update_preservation.draft_page_core_markdown(page)),
            ("open_questions", _update_preservation.draft_page_open_questions(page)),
        ]:
            if "---\n" in body or body.lstrip().startswith("# ") or text_contains_source_graph_link(body):
                raise_draft_issue(
                    "forbidden_page_markdown",
                    f"{page.page_plan_id} {field_name} contains forbidden page-level markdown",
                    field_path=field_name,
                )
            if section_contains_stray_related_links(body, page.canonical_target_path, display_title=display_title):
                raise_draft_issue(
                    "stray_related_links_in_content",
                    (
                        f"{page.page_plan_id} {field_name} contains a related-page block or self wikilink; "
                        "remove body-level related links because the system renders official related pages separately."
                    ),
                    field_path=f"pages.{page.page_plan_id}.{field_name}",
                )
            if language == "zh-CN" and looks_like_untranslated_english(body):
                raise_draft_issue(
                    "zh_cn_untranslated_user_text",
                    f"{page.page_plan_id} {field_name} must be Chinese for zh-CN vault",
                    field_path=field_name,
                )
        if language == "zh-CN" and looks_like_untranslated_english(page.change_summary):
            raise_draft_issue(
                "zh_cn_untranslated_user_text",
                f"{page.page_plan_id} change_summary must be Chinese for zh-CN vault",
                field_path="change_summary",
            )
        if language == "zh-CN" and looks_like_untranslated_english(page.source_coverage_notes):
            raise_draft_issue(
                "zh_cn_untranslated_user_text",
                f"{page.page_plan_id} source_coverage_notes must be Chinese for zh-CN vault",
                field_path="source_coverage_notes",
            )
        if plan_item is not None:
            validate_digestive_quality(page, plan_item)


def validate_digestive_quality(page: DraftPageItem, item: WikiMergePlanItem) -> None:
    summary = _update_preservation.draft_page_summary(page)
    core = _update_preservation.draft_page_core_markdown(page)
    if not is_substantive_digestive_text(core) or normalized_digest_text(summary) == normalized_digest_text(core):
        raise_draft_issue(
            "thin_digestive_content",
            (
                f"{page.page_plan_id} must include concrete digested understanding: viewpoint, example, "
                "use scenario, boundary condition, or value point; it must not be only a source summary."
            ),
            field_path="body_markdown",
        )
    if item.action == "update" and not update_change_summary_is_specific(page.change_summary):
        raise_draft_issue(
            "thin_update_change_summary",
            f"{page.page_plan_id} update change_summary must explain what the new material补充/改变/澄清了旧理解。",
            field_path="change_summary",
        )


def draft_self_talk_issues(artifact: DraftRenderingArtifact) -> list[StructuredIssue]:
    issues: list[StructuredIssue] = []
    for page in artifact.pages:
        for field_name, body in [
            ("summary", _update_preservation.draft_page_summary(page)),
            ("body_markdown", _update_preservation.draft_page_core_markdown(page)),
            ("open_questions", _update_preservation.draft_page_open_questions(page)),
        ]:
            marker = draft_self_talk_marker(body)
            if not marker:
                continue
            issues.append(
                StructuredIssue(
                    issue_code="model_self_talk_leak",
                    field_path=f"pages.{page.page_plan_id}.{field_name}",
                    validator_id="draft_content_quality",
                    message=(
                        f"{page.page_plan_id} {field_name} contains model self-talk marker `{marker}`; "
                        "remove reasoning notes about checking, uncertainty, or future edits, and keep only the final sourced page content."
                    ),
                    repairability="repairable",
                )
            )
    return issues


def draft_self_talk_marker(text: str) -> str:
    compact = re.sub(r"\s+", "", text)
    for marker in DRAFT_SELF_TALK_MARKERS:
        if marker and marker in compact:
            return marker
    if "需要谨慎" in compact and any(marker in compact for marker in ["检查原文", "查看原文", "原文", "我", "记错"]):
        return "需要谨慎"
    if "应该是" in compact and any(marker in compact for marker in ["我", "记错", "检查原文", "原文数据", "但前面说"]):
        return "应该是"
    return ""


def is_substantive_digestive_text(text: str) -> bool:
    normalized = normalized_digest_text(text)
    placeholder_markers = ["暂无", "没有相关", "无相关", "n/a"]
    if not normalized:
        return False
    if any(normalized == marker or (normalized.startswith(marker) and len(normalized) <= 32) for marker in placeholder_markers):
        return False
    return len(normalized) >= 24 or any(marker in text for marker in ["例如", "适用", "边界", "价值", "场景", "反例", "意味着", "可以用来"])


def normalized_digest_text(text: str) -> str:
    return re.sub(r"[\s\-*#`，。；;：:、,.!?！？（）()]+", "", text.strip().lower())


def update_change_summary_is_specific(text: str) -> bool:
    normalized = normalized_digest_text(text)
    if len(normalized) < 12:
        return False
    return any(marker in text for marker in ["补充", "改变", "澄清", "更新", "整合", "新增", "保留", "删除", "修正", "扩展"])


def raise_draft_issue(issue_code: str, message: str, *, field_path: str = "", repairable: bool = True) -> None:
    raise ContractValidationError(
        message,
        issues=[
            StructuredIssue(
                issue_code=issue_code,
                field_path=field_path,
                validator_id="validate_draft_rendering",
                message=message,
                repairability="repairable" if repairable else "non_repairable",
            )
        ],
    )


def section_contains_stray_related_links(text: str, canonical_target_path: str, *, display_title: str = "") -> bool:
    stripped = strip_fenced_code_blocks(text)
    return section_contains_self_wikilink(stripped, canonical_target_path, display_title=display_title) or section_contains_related_link_block(stripped)


def strip_fenced_code_blocks(text: str) -> str:
    kept_lines: list[str] = []
    fence_char = ""
    fence_length = 0
    for line in text.splitlines(keepends=True):
        stripped_newline = line.rstrip("\r\n")
        if fence_char:
            if closing_fence_line(stripped_newline, fence_char, fence_length):
                fence_char = ""
                fence_length = 0
            continue
        if match := opening_fence_line(stripped_newline):
            marker = match.group("marker")
            fence_char = marker[0]
            fence_length = len(marker)
            continue
        kept_lines.append(line)
    return "".join(kept_lines)


def opening_fence_line(line: str) -> re.Match[str] | None:
    return re.match(r"^ {0,3}(?P<marker>`{3,}|~{3,})[^\n]*$", line)


def closing_fence_line(line: str, fence_char: str, fence_length: int) -> bool:
    escaped = re.escape(fence_char)
    return re.match(rf"^ {{0,3}}{escaped}{{{fence_length},}}\s*$", line) is not None


def section_contains_self_wikilink(text: str, canonical_target_path: str, *, display_title: str = "") -> bool:
    canonical = _wiki_markup.normalize_related_candidate_path(canonical_target_path)
    if canonical is None:
        return False
    canonical_aliases = normalized_path_self_aliases(canonical)
    canonical_aliases |= normalized_title_self_aliases(display_title)
    for raw_target in section_link_targets(text):
        link_target = raw_target.strip()
        if link_target.startswith(("#", "//")) or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", link_target):
            continue
        target = _wiki_markup.normalize_related_candidate_path(raw_target)
        if target is not None and normalized_path_self_aliases(target) & canonical_aliases:
            return True
    return False


def section_link_targets(text: str) -> list[str]:
    targets: list[str] = []
    for match in re.finditer(r"\[\[([^\]]+)\]\]", text):
        targets.append(match.group(1).split("|", 1)[0])
    for match in re.finditer(r"\[[^\]\n]+\]\(([^)]+)\)", text):
        targets.append(match.group(1))
    return targets


def normalized_path_self_aliases(path: str) -> set[str]:
    posix = Path(path).as_posix()
    stemless = Path(path).with_suffix("").as_posix()
    return {
        posix,
        stemless,
        Path(posix).name,
        Path(stemless).name,
    }


def normalized_title_self_aliases(title: str) -> set[str]:
    stripped = title.strip()
    if not stripped:
        return set()
    aliases = {stripped}
    if not stripped.endswith(".md"):
        aliases.add(f"{stripped}.md")
    if (title_path := _wiki_markup.normalize_related_candidate_path(stripped)) is not None:
        aliases |= normalized_path_self_aliases(title_path)
    return aliases


def section_contains_related_link_block(text: str) -> bool:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not re.match(
            r"^\s{0,3}(?:[-*+]\s*)?(?:#{1,6}\s*)?(?:\*\*|__)?(?:相关页面|related(?:\s+pages)?)(?:\*\*|__)?\s*(?:[:：]|$)",
            line,
            re.IGNORECASE,
        ):
            continue
        if section_link_targets(line):
            return True
        for follower in lines[index + 1 : index + 6]:
            stripped = follower.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                break
            if section_link_targets(stripped):
                return True
            if not stripped.startswith(("-", "*", "+")):
                break
    return False
