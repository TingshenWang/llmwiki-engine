from __future__ import annotations

from typing import Any

from .events import format_duration
from .merge_plan_risks import merge_plan_create_overlap_risk_items
from .models import CandidateContextsArtifact, WikiContextSnapshot, WikiMergePlanArtifact
from .retrieval import SCORE_BUCKET_EPSILON
from .system_pages import format_markdown_table


def render_merge_planning_shortcut_report(report: dict[str, Any]) -> str:
    rows = [
        ["candidate_count", report.get("candidate_count", 0)],
        ["knowledge_metadata_pool_count", report.get("knowledge_metadata_pool_count", 0)],
        ["candidate_context_hit_count", report.get("candidate_context_hit_count", 0)],
        ["present_target_count", report.get("present_target_count", 0)],
        ["missing_snapshot_target_count", report.get("missing_snapshot_target_count", 0)],
        ["source_page_plan_count", report.get("source_page_plan_count", 0)],
        ["empty_target_page_plan_count", report.get("empty_target_page_plan_count", 0)],
        ["missing_source_candidate_page_plan_count", report.get("missing_source_candidate_page_plan_count", 0)],
        ["unknown_source_candidate_id_count", report.get("unknown_source_candidate_id_count", 0)],
        ["duplicate_candidate_target_count", report.get("duplicate_candidate_target_count", 0)],
    ]
    blockers = report.get("blocking_conditions", [])
    return (
        "# Merge Planning Shortcut Report\n\n"
        f"- Shortcut：`{report.get('shortcut', '')}`\n"
        f"- Used：`{str(bool(report.get('used'))).lower()}`\n"
        f"- Reason：{report.get('reason', '')}\n"
        f"- Blocking conditions：`{', '.join(blockers) if blockers else 'none'}`\n\n"
        "## Guard Counters\n\n"
        f"{format_markdown_table(['检查项', '值'], rows)}\n"
    )


def render_merge_plan_markdown(plan: WikiMergePlanArtifact) -> str:
    rows = []
    for item in plan.items:
        rows.append(
            [
                item.page_plan_id,
                item.model_action or item.action,
                item.action,
                item.display_title,
                f"`{item.canonical_target_path}`",
                f"{item.strongest_overlap.strength} `{item.strongest_overlap.path}`".strip(),
                item.why_not_update,
                item.new_understanding,
                item.apply_eligibility,
                item.blocked_reason,
            ]
        )
    return "# Wiki 合并计划\n\n" + format_markdown_table(
        ["页面计划", "模型动作", "最终动作", "标题", "目标", "最强召回", "为什么不更新旧页", "新增理解", "Apply", "阻断原因"],
        rows,
    ) + "\n"


def render_merge_plan_review_prompt(plan: WikiMergePlanArtifact) -> str:
    decision_rows = [
        [
            item.page_plan_id,
            item.model_action or item.action,
            item.action,
            item.display_title,
            f"`{item.canonical_target_path}`",
            item.strongest_overlap.strength,
            item.blocked_reason,
        ]
        for item in plan.items
    ]
    return (
        "# 合并计划审核\n\n"
        "审查这一步回答：写哪些页面、为什么写、哪些旧页已经被看过。\n\n"
        "## 核心判断\n\n"
        "- create 是否真的不能 update 到已召回的旧页？理由是否具体到范围、来源增量和边界？\n"
        "- update/noop 是否绑定了被召回并读过全文的旧页？\n"
        "- Related 是否少而准，是否只保留最相关的 0-2 条？\n"
        "- 是否存在 needs_human_decision 或 all-create 风险需要先 revise？\n\n"
        "## 关键文件\n\n"
        "- 合并计划：`wiki_merge_planning/wiki_merge_plan.json`\n"
        "- 合并报告：`wiki_merge_planning/merge_decision_report.md`\n"
        "- 召回上下文：`wiki_context_snapshot/candidate_contexts.md`\n"
        "- 可编辑文件：`merge_plan_review/pending_merge_plan.json`（仅 awaiting_review 时存在）\n\n"
        "## 下一步命令\n\n"
        "- 批准：`uv run llmwiki ingest approve \"$VAULT\" \"$OP\" merge_plan_review`\n"
        "- 重新生成/修订：`uv run llmwiki ingest revise \"$VAULT\" \"$OP\" merge_plan_review`\n"
        "- 批准后继续：`uv run llmwiki ingest resume \"$VAULT\" \"$OP\"`\n\n"
        "## 决策概览\n\n"
        + format_markdown_table(["页面计划", "模型动作", "最终动作", "标题", "目标", "最强召回", "阻断原因"], decision_rows)
        + "\n\n"
        "## 完整计划\n\n"
        f"{render_merge_plan_markdown(plan)}"
    )


def render_candidate_contexts_markdown(
    artifact: CandidateContextsArtifact,
    *,
    resolved_cache_path: str = "",
    query_count: int | None = None,
    encoded_page_count: int | None = None,
) -> str:
    sections = [
        "# 候选页召回上下文",
        "",
        f"- 后端：`{artifact.retrieval_backend}`",
        f"- 模型：`{artifact.model}`",
        f"- 本地缓存模式（local files only）：{artifact.local_files_only}",
        f"- 缓存路径（resolved cache path）：`{resolved_cache_path or artifact.cache_dir or '未使用'}`",
        f"- Embedding 加载耗时：{format_duration(artifact.embedding_load_duration_ms)}",
        f"- Embedding 编码耗时：{format_duration(artifact.embedding_encode_duration_ms)}",
        f"- Embedding 总耗时：{format_duration(artifact.embedding_total_duration_ms)}",
        f"- Embedding 页面向量缓存命中：{artifact.embedding_page_vector_cache_hit}",
        f"- Embedding 页面/查询数量：{artifact.embedding_page_count}/{artifact.embedding_query_count}",
        f"- Embedding 输入字符数：{artifact.embedding_text_char_count}",
        f"- TopK：{artifact.top_k}",
        f"- 候选池页面数：{artifact.candidate_pool_size}",
        f"- 查询数（query count）：{artifact.candidate_pool_size if query_count is None else query_count}",
        f"- 编码页面数：{artifact.candidate_pool_size if encoded_page_count is None else encoded_page_count}",
        f"- 不完整 frontmatter 页面数：{artifact.skipped_count}",
        f"- 候选池 Hash：`{artifact.candidate_pool_sha256}`",
        "- 排序说明：先按强度、Score Bucket、依据、页面类型、目录和标题距离排序；Score Bucket 默认宽度为 "
        f"{SCORE_BUCKET_EPSILON:.2f}，所以表格里的原始分数不一定逐行严格递减。",
        "- Sort Key 说明：`bucket` 是分数分桶；`type/dir/title_distance/path` 是同一分数桶内的 tie-break。",
        "- `lexical_expansion` 表示 query 和旧页命中了同一组高信号术语；中文相似度使用 bigram/短语重叠，避免单字重叠把泛相关页面推高。",
    ]
    if artifact.warnings:
        sections.extend(["", "## 警告", "", *[f"- {warning}" for warning in artifact.warnings]])
    for item in artifact.items:
        rows = [
            [
                str(hit.rank),
                hit.strength,
                hit.match_basis,
                f"{hit.score:.4f}",
                str(hit.score_bucket or int(hit.score / SCORE_BUCKET_EPSILON)),
                hit.sort_explanation,
                "`forced`" if hit.forced else "",
                f"`{hit.path}`",
                hit.display_title,
                "`truncated`" if hit.truncated else "",
                hit.excerpt[:180].replace("\n", " "),
            ]
            for hit in item.hits
        ]
        sections.extend(
            [
                "",
                f"## {item.page_plan_id}",
                "",
                f"查询文本: {item.query[:500]}",
                "",
                format_markdown_table(["排名", "强度", "依据", "分数", "Score Bucket", "Sort Key", "强制命中", "路径", "标题", "截断", "片段"], rows)
                if rows
                else "未召回到候选旧页。",
            ]
        )
        if item.unindexable_pages:
            sections.extend(["", "Frontmatter 不完整但已进入低置信候选池：", "", *[f"- `{path}`" for path in item.unindexable_pages[:20]]])
    return "\n".join(sections).rstrip() + "\n"


def render_merge_decision_report(plan: WikiMergePlanArtifact, snapshot: WikiContextSnapshot) -> str:
    context_by_id = {item.page_plan_id: item for item in snapshot.candidate_contexts.items}
    sections = ["# 合并决策报告", ""]
    create_risks = merge_plan_create_overlap_risk_items(plan)
    if create_risks:
        risk_rows = []
        for item in create_risks:
            context = context_by_id.get(item.page_plan_id)
            inspected = item.inspected_context_paths or ([hit.path for hit in context.hits] if context else [])
            risk_rows.append(
                [
                    item.page_plan_id,
                    item.model_action or item.action,
                    item.action,
                    item.strongest_overlap.strength,
                    f"`{item.strongest_overlap.path}`",
                    ", ".join(f"`{path}`" for path in inspected[:5]) if inspected else "无",
                    item.why_not_update or "未提供",
                    item.apply_eligibility,
                    item.blocked_reason,
                ]
            )
        sections.extend(
            [
                "## Create/Update 风险摘要",
                "",
                "这些项目的模型动作或最终动作包含 create，但 TopK 召回中存在 medium/strong 旧页；审核时应优先检查 why_not_update 是否具体说明范围差异、来源增量和为什么不能 update。",
                "",
                format_markdown_table(
                    ["页面计划", "模型动作", "最终动作", "最强召回", "旧页", "看过的旧页", "为什么不更新", "Apply", "阻断原因"],
                    risk_rows,
                ),
                "",
            ]
        )
    for item in plan.items:
        context = context_by_id.get(item.page_plan_id)
        inspected = item.inspected_context_paths or ([hit.path for hit in context.hits] if context else [])
        overlap_rows = []
        if context is not None:
            for hit in context.hits:
                if hit.strength in {"medium", "strong"}:
                    overlap_rows.append(
                        [
                            hit.rank,
                            hit.strength,
                            hit.match_basis,
                            f"{hit.score:.4f}",
                            f"`{hit.path}`",
                            hit.display_title,
                            hit.excerpt[:160].replace("\n", " "),
                        ]
                    )
        related_text = (
            ", ".join(f"`{related.target_path}`" for related in item.related_pages)
            if item.related_pages
            else f"无（{item.related_absence_reason or 'no_candidate'}）"
        )
        action_question = {
            "create": "为什么不 update",
            "update": "为什么 update",
            "noop": "为什么 noop",
            "needs_human_decision": "为什么需要人工决策",
        }.get(item.action, "为什么 create/update/noop")
        action_answer = {
            "create": item.why_not_update or "未提供",
            "update": item.why_create_or_update or item.reason,
            "noop": item.why_create_or_update or item.reason,
            "needs_human_decision": item.blocked_reason or item.why_create_or_update or item.reason,
        }.get(item.action, item.reason)
        sections.extend(
            [
                f"## {item.display_title}",
                "",
                f"- 页面计划：`{item.page_plan_id}`",
                f"- 模型动作：`{item.model_action or item.action}`",
                f"- 最终动作：`{item.action}`",
                f"- 目标：`{item.canonical_target_path}`",
                f"- 最像旧页：`{item.strongest_overlap.path or '无'}` ({item.strongest_overlap.strength}, {item.strongest_overlap.match_basis})",
                f"- 看过的旧页：{', '.join(f'`{path}`' for path in inspected) if inspected else '无'}",
                f"- {action_question}：{action_answer}",
                f"- 为什么 create/update/noop：{item.why_create_or_update or item.reason}",
                f"- Related：{related_text}",
                f"- Finalizer：{item.finalization_reason}",
            ]
        )
        if item.action == "create" and item.strongest_overlap.strength in {"medium", "strong"}:
            sections.extend(
                [
                    "",
                    "### Create 对比审计",
                    "",
                    "- scope_delta：见 why_not_update 中的新旧页面范围差异。",
                    "- source_delta：见 why_not_update 中的新材料增量。",
                    "- why_update_not_enough：见 why_not_update 中为什么整页更新不合适。",
                    "- why_related_link_not_enough：见 why_not_update 中为什么只做 Related 不够。",
                ]
            )
        if overlap_rows:
            sections.extend(
                [
                    "",
                    "### Medium/Strong 召回命中",
                    "",
                    format_markdown_table(["排名", "强度", "依据", "分数", "路径", "标题", "片段"], overlap_rows),
                ]
            )
        if item.blocked_reason:
            sections.append(f"- 阻断原因：{item.blocked_reason}")
        sections.append("")
    return "\n".join(sections).rstrip() + "\n"
