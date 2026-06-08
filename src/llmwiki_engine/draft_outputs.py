from __future__ import annotations

import json
import re
from typing import Any

from . import draft_grounding as _draft_grounding
from . import page_sections as _page_sections
from . import related_pages as _related_pages
from . import section_merge as _section_merge
from . import source_digest_budget as _source_digest_budget
from . import update_preservation as _update_preservation
from . import wiki_markup as _wiki_markup
from .models import (
    DraftPageItem,
    DraftRenderingArtifact,
    GroundingClaim,
    RawLinkCleanupArtifact,
    RelatedCandidateReport,
    RelatedMergeReport,
    SectionMergeChange,
    SourceDigestArtifact,
    UpdateMergeReport,
    UpdatePageMergeReport,
    WikiContextEntry,
    WikiContextSnapshot,
    WikiMergePlanArtifact,
    WikiMergePlanItem,
)
from .system_pages import format_markdown_table


def assemble_knowledge_page(
    *,
    item: WikiMergePlanItem,
    page: DraftPageItem,
    existing_entry: WikiContextEntry,
    raw_path: str,
    raw_hash: str,
    prepared_hash: str,
    operation_id: str,
    log_date: str,
    update_reports: list[UpdatePageMergeReport] | None = None,
    related_reports: list[RelatedCandidateReport] | None = None,
    grounding_claims: list[GroundingClaim] | None = None,
    known_related_paths: set[str] | None = None,
    approved_raw_text: str = "",
) -> str:
    existing_sections = _page_sections.parse_existing_sections(existing_entry.content)
    summary = _update_preservation.draft_page_summary(page) or item.new_understanding
    core = _update_preservation.draft_page_core_markdown(page) or item.knowledge_delta or item.new_understanding
    questions = _update_preservation.draft_page_open_questions(page) or "暂无矛盾与未决问题记录。"
    metadata = existing_entry.metadata
    is_update = item.action == "update" and metadata is not None
    final_title = metadata.title if is_update and metadata.title else item.display_title
    aliases = list(metadata.aliases if metadata is not None else [])
    if is_update and item.display_title and item.display_title != final_title and item.display_title not in aliases:
        aliases.append(item.display_title)
    created = metadata.created if metadata is not None and metadata.created else log_date
    section_changes: list[SectionMergeChange] = []
    if is_update:
        update_absorption_context = "\n\n".join(
            section
            for section in [summary, core, questions]
            if section.strip()
        )
        summary, summary_change = _section_merge.merge_update_section(
            "summary",
            existing_sections.get("summary", ""),
            summary,
            absorption_context=update_absorption_context,
        )
        old_core = _page_sections.existing_core_content_from_sections(existing_sections)
        core, core_change = _section_merge.merge_update_section(
            "core_content",
            old_core,
            core,
            absorption_context=update_absorption_context,
        )
        questions, questions_change = _section_merge.merge_update_section(
            "open_questions",
            existing_sections.get("open_questions", ""),
            questions,
            absorption_context=update_absorption_context,
        )
        section_changes.extend([summary_change, core_change, questions_change])
    related = _related_pages.render_related_pages(
        item,
        existing_entry=existing_entry,
        report_list=related_reports,
        known_paths=known_related_paths,
    )
    if update_reports is not None and is_update:
        update_reports.append(
            UpdatePageMergeReport(
                page_plan_id=item.page_plan_id,
                target_path=item.canonical_target_path,
                old_title=metadata.title,
                final_title=final_title,
                model_title=item.display_title,
                retained_title=final_title == metadata.title,
                merged_page_plan_ids=item.merged_page_plan_ids,
                noop_covered_by_update=item.noop_covered_by_update,
                sections=section_changes,
            )
        )
    if grounding_claims is not None:
        _draft_grounding.collect_grounding_claims(
            item=item,
            page=page,
            existing_entry=existing_entry,
            approved_raw_text=approved_raw_text,
            claims=grounding_claims,
        )
    source_raw_paths = _append_unique(metadata.source_raw_paths if metadata is not None else [], raw_path)
    source_raw_hashes = _append_unique(metadata.source_raw_hashes if metadata is not None else [], raw_hash)
    source_prepared_hashes = _append_unique(metadata.source_prepared_hashes if metadata is not None else [], prepared_hash)
    source_operation_ids = _append_unique(metadata.source_operation_ids if metadata is not None else [], operation_id)
    return (
        "---\n"
        f"llmwiki_type: {item.page_type}\n"
        f"title: {_yaml_scalar(final_title)}\n"
        f"{_yaml_list('aliases', aliases)}"
        f"summary: {_yaml_scalar(summary)}\n"
        f"created: {created}\n"
        f"updated: {log_date}\n"
        f"{_yaml_list('source_raw_paths', source_raw_paths)}"
        f"{_yaml_list('source_raw_hashes', source_raw_hashes)}"
        f"{_yaml_list('source_prepared_hashes', source_prepared_hashes)}"
        f"{_yaml_list('source_operation_ids', source_operation_ids)}"
        f"last_ingest_operation: {_yaml_scalar(operation_id)}\n"
        "---\n\n"
        f"# {final_title}\n\n"
        "## 摘要\n\n"
        f"{summary}\n\n"
        "## 核心内容\n\n"
        f"{core}\n\n"
        "## 相关页面\n\n"
        f"{related}\n\n"
        "## 矛盾与未决问题\n\n"
        f"{questions}\n"
    )


def render_source_page(
    *,
    title: str,
    digest: SourceDigestArtifact,
    operation_id: str,
    linked_pages: list[str],
    no_change_pages: list[str],
    log_date: str,
    raw_hash: str,
    prepared_hash: str,
    cleanup: RawLinkCleanupArtifact,
) -> str:
    links = "\n".join(f"- `{path}`" for path in linked_pages) or "- 暂无派生知识页。"
    no_change = render_source_unwritten_notes(digest, no_change_pages)
    summary = neutralize_markdown_links(digest.summary)
    takeaways = "\n".join(f"- {neutralize_markdown_links(item)}" for item in digest.key_takeaways) or "- 暂无关键收获记录。"
    return (
        "---\n"
        "llmwiki_type: source\n"
        f"title: {_yaml_scalar(title)}\n"
        "aliases: []\n"
        f"summary: {_yaml_scalar(summary)}\n"
        f"created: {log_date}\n"
        f"updated: {log_date}\n"
        "source_raw_paths:\n"
        f"  - {_yaml_scalar(digest.source_raw_path)}\n"
        "source_raw_hashes:\n"
        f"  - {_yaml_scalar(raw_hash)}\n"
        "source_prepared_hashes:\n"
        f"  - {_yaml_scalar(prepared_hash)}\n"
        "source_operation_ids:\n"
        f"  - {_yaml_scalar(operation_id)}\n"
        f"raw_cleanup_pre_sha256: {_yaml_scalar(cleanup.pre_cleanup_sha256)}\n"
        f"raw_cleanup_post_sha256: {_yaml_scalar(cleanup.post_cleanup_sha256)}\n"
        f"raw_cleanup_rule_version: {_yaml_scalar(cleanup.cleanup_rule_version)}\n"
        f"raw_cleanup_artifact_ref: {_yaml_scalar('raw_link_cleanup/raw_link_cleanup.json')}\n"
        f"raw_cleanup_changed: {str(cleanup.changed).lower()}\n"
        f"raw_cleanup_cleaned_link_count: {cleanup.cleaned_link_count}\n"
        f"raw_cleanup_diff_ref: {_yaml_scalar('raw_link_cleanup/cleanup.diff')}\n"
        f"last_ingest_operation: {_yaml_scalar(operation_id)}\n"
        "---\n\n"
        f"# {title}\n\n"
        "## 摘要\n\n"
        f"{summary}\n\n"
        "## 原始材料\n\n"
        f"- `{digest.source_raw_path}`\n\n"
        "## 关键收获\n\n"
        f"{takeaways}\n\n"
        "## 派生知识页\n\n"
        f"{links}\n\n"
        "## 未写入说明\n\n"
        f"{no_change}\n"
    )


def render_source_unwritten_notes(digest: SourceDigestArtifact, no_change_pages: list[str]) -> str:
    sections = [
        (
            "未改动页面：\n" + "\n".join(f"- `{path}`" for path in no_change_pages)
            if no_change_pages
            else "暂无未写入页面。"
        )
    ]
    if digest.budget_deferred_candidates:
        rows = [
            [
                candidate.candidate_id,
                candidate.type,
                neutralize_markdown_links(candidate.suggested_page_title or candidate.name),
                neutralize_markdown_links(candidate.one_sentence_summary),
                neutralize_markdown_links(candidate.wiki_value),
                neutralize_markdown_links(candidate.resolution_hint),
            ]
            for candidate in digest.budget_deferred_candidates
        ]
        sections.extend(
            [
                "### 预算延后候选（未独立建页）",
                "这些候选因 `max_ingest_candidates` 预算限制未进入本轮页面规划；它们保留在 source digest 和预算报告中，后续可单独建页或聚合进总览/对比页。",
                format_markdown_table(["ID", "类型", "建议标题", "摘要", "Wiki 价值", "处理提示"], rows),
            ]
        )
        aggregation_rows = [
            [
                aggregation["suggested_page_type"],
                neutralize_markdown_links(str(aggregation["suggested_title"])),
                neutralize_markdown_links(str(aggregation["coverage_summary"])),
                ", ".join(str(candidate.get("title", candidate.get("candidate_id", ""))) for candidate in aggregation.get("representative_candidates", [])[:4]),
                neutralize_markdown_links(str(aggregation["suggested_action"])),
            ]
            for aggregation in _source_digest_budget.build_deferred_candidate_aggregations(digest.budget_deferred_candidates)
        ]
        if aggregation_rows:
            sections.extend(
                [
                    "### 延后候选聚合建议",
                    "这些聚合建议只记录后续处理路径，不会在本轮增加知识页数量。",
                    format_markdown_table(["建议页类型", "建议标题", "覆盖摘要", "代表候选", "后续动作"], aggregation_rows),
                ]
            )
    return "\n\n".join(sections)


def neutralize_markdown_links(text: str) -> str:
    def wiki_repl(match: re.Match[str]) -> str:
        label = match.group(1).replace("|", " / ").strip()
        return f"`{label}`" if label else ""

    def markdown_repl(match: re.Match[str]) -> str:
        label = match.group(1).strip()
        target = match.group(2).strip()
        if not label:
            return f"`{target}`"
        return f"{label} (`{target}`)" if target else label

    text = re.sub(r"\[\[([^\]]+)\]\]", wiki_repl, text)
    return re.sub(r"\[([^\]]*)\]\(([^)]+)\)", markdown_repl, text)


def build_index_rows(profile: Any, plan: WikiMergePlanArtifact, draft: DraftRenderingArtifact, snapshot: WikiContextSnapshot) -> list[dict[str, str]]:
    rows_by_path: dict[str, dict[str, str]] = {}
    for pool_entry in snapshot.knowledge_metadata_pool:
        metadata = pool_entry.metadata
        if metadata is None or metadata.llmwiki_type.lower() == "source":
            continue
        rows_by_path[metadata.path] = {
            "title": _wiki_markup.clean_display_title(metadata.title),
            "page": _wiki_markup.obsidian_link(metadata.path),
            "type": metadata.llmwiki_type,
            "summary": metadata.summary,
            "updated": metadata.updated,
        }
    page_by_id = {page.page_plan_id: page for page in draft.pages}
    for item in plan.items:
        if item.action not in {"create", "update"}:
            continue
        if item.page_type.lower() == "source":
            continue
        page = page_by_id.get(item.page_plan_id)
        summary = _update_preservation.draft_page_summary(page) if page else item.new_understanding
        title = item.display_title
        if item.action == "update":
            entry = next((entry for entry in snapshot.entries if entry.path == f"wiki/{item.canonical_target_path}"), None)
            if entry is not None and entry.metadata is not None:
                title = _wiki_markup.clean_display_title(entry.metadata.title)
        rows_by_path[item.canonical_target_path] = {
            "title": title,
            "page": _wiki_markup.obsidian_link(item.canonical_target_path),
            "type": item.page_type,
            "summary": summary or item.new_understanding,
            "updated": plan.log_date,
        }
    type_order = list(profile.page_types)
    rows = list(rows_by_path.values())
    rows.sort(key=lambda row: row["page"])
    rows.sort(key=lambda row: row["updated"], reverse=True)
    rows.sort(key=lambda row: type_order.index(row["type"]) if row["type"] in type_order else len(type_order))
    return rows


def render_update_merge_report(report: UpdateMergeReport) -> str:
    if not report.pages:
        return "# Update 合并报告\n\n本次没有 update 页面。\n"
    sections = ["# Update 合并报告", ""]
    for page in report.pages:
        sections.extend(
            [
                f"## {page.final_title or page.target_path}",
                "",
                f"- 页面计划：`{page.page_plan_id}`",
                f"- 目标：`{page.target_path}`",
                f"- 旧标题：{page.old_title or '无'}",
                f"- 模型标题：{page.model_title or '无'}",
                f"- 最终标题：{page.final_title or '无'}",
                f"- 保留旧标题：{'是' if page.retained_title else '否'}",
                f"- 合并计划页：{', '.join(f'`{item}`' for item in page.merged_page_plan_ids) if page.merged_page_plan_ids else '无'}",
                f"- noop 被 update 覆盖：{'是' if page.noop_covered_by_update else '否'}",
                "",
            ]
        )
        rows = [
            [
                change.section_key,
                "\n".join(change.retained) or "无",
                "\n".join(change.added) or "无",
                "\n".join(change.removed) or "无",
                "\n".join(change.preserved_old) or "无",
                "是" if change.needs_manual_resolution else "否",
                change.removal_reason,
            ]
            for change in page.sections
        ]
        sections.append(
            format_markdown_table(["段落", "保留", "新增", "删除", "旧页保留观察", "需人工消化", "原因"], rows)
            if rows
            else "没有记录 section 级变更。"
        )
        sections.append("")
    return "\n".join(sections).rstrip() + "\n"


def render_related_merge_report(report: RelatedMergeReport) -> str:
    rows = [
        [
            item.page_plan_id,
            f"`{item.target_path}`",
            item.display_title,
            item.source,
            item.decision,
            item.reject_reason,
            item.reason,
        ]
        for item in report.candidates
    ]
    body = format_markdown_table(["页面计划", "目标", "标题", "来源", "决策", "过滤原因", "理由"], rows) if rows else "没有 Related 候选。"
    return "# 相关页面合并报告\n\n" + body + "\n"


def _append_unique(existing: list[str], value: str) -> list[str]:
    items: list[str] = []
    for item in [*existing, value]:
        if item and item not in items:
            items.append(item)
    return items


def _yaml_scalar(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _yaml_list(key: str, values: list[str]) -> str:
    if not values:
        return f"{key}: []\n"
    return f"{key}:\n" + "".join(f"  - {_yaml_scalar(value)}\n" for value in values)
