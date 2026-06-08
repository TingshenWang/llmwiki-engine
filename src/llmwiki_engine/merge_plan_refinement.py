from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from . import markdown_utils as _markdown_utils
from . import merge_plan_risks as _merge_plan_risks
from . import related_pages as _related_pages
from . import text_similarity as _text_similarity
from . import wiki_markup as _wiki_markup
from .models import (
    CandidateContextHit,
    CandidateResolutionItem,
    RelatedPageRef,
    SourceBasis,
    WikiContextEntry,
    WikiContextSnapshot,
    WikiMergePlanArtifact,
    WikiMergePlanItem,
)
from .validators import create_reason_needs_repair
from .wiki_context import snapshot_entry


MAX_AUTO_APPROVED_ALL_CREATE_ITEMS = 12
LOCAL_MEDIUM_CREATE_REASON_MARKER = "本地补充结构化 create/update 对比理由"
_OLD_PAGE_ANCHOR = r"(?:旧页|已有页|已有知识页|现有页|现有页面|现有知识页|最像旧页|旧页面|old page|existing page|existing knowledge page)"
_OLD_TITLE_ANCHOR = r"(?:旧页标题|已有页标题|已有知识页标题|现有页标题|现有页面标题|现有知识页标题|old title|existing title)"
_CN_SOURCE_SPECIFIC = r"(?:cloudflare|mem0|redis|qwen|anthropic|平台|产品|实现|教程|官方|项目|案例|来源特定)"
_EN_SOURCE_SPECIFIC = r"(?:cloudflare|mem0|redis|qwen|anthropic|source-specific|product-specific|platform-specific|implementation)"
_CN_SCOPE_SPECIFIC = r"(?:平台|产品|实现|教程|官方|项目|案例|来源特定)"
_EN_SCOPE_SPECIFIC = r"(?:source-specific|product-specific|platform-specific|implementation)"
_CN_GENERIC = r"(?:通用概念|通用页面|通用知识|泛化概念)"
_CN_GENERIC_WITH_SOURCE_NEUTRAL = r"(?:通用概念|通用页面|通用知识|source-neutral|泛化概念)"
_EN_GENERIC = r"(?:source-neutral|generic|general concept|general knowledge)"
_SCOPE_PREDICATE = r"(?:聚焦|侧重|面向|围绕|范围|主要|focus(?:es|ed)? on|center(?:s|ed)? on|centred on|centered on)"


def synthesize_medium_create_why_not_update(
    *,
    item: WikiMergePlanItem,
    resolution_item: CandidateResolutionItem,
    strongest_hit: CandidateContextHit | None,
    snapshot: WikiContextSnapshot,
) -> str:
    old_path = strongest_hit.path if strongest_hit is not None else ""
    old_entry = snapshot_entry(snapshot, f"wiki/{old_path}") if old_path else WikiContextEntry(path="", expected_state="missing")
    old_title = (
        _wiki_markup.clean_display_title(old_entry.metadata.title)
        if old_entry.metadata is not None
        else _wiki_markup.clean_display_title(strongest_hit.display_title if strongest_hit is not None else "已召回旧页")
    )
    old_summary = old_entry.metadata.summary if old_entry.metadata is not None else ""
    old_scope = _markdown_utils.compact_payload_text(old_summary or old_title or old_path, 120)
    new_scope = _markdown_utils.compact_payload_text(
        resolution_item.topic_summary
        or item.new_understanding
        or resolution_item.initial_section_intent
        or resolution_item.display_title,
        140,
    )
    source_delta = _markdown_utils.compact_payload_text(
        resolution_item.why_this_page
        or resolution_item.coverage_notes
        or resolution_item.reason
        or item.knowledge_delta
        or item.why_this_matters,
        140,
    )
    page_kind = _chinese_page_type_label(resolution_item.page_type)
    return (
        "本地补充：scope_delta："
        f"新页《{_wiki_markup.clean_display_title(resolution_item.display_title)}》按 `{resolution_item.candidate_target_path}` 独立沉淀为{page_kind}，"
        f"核心范围是「{new_scope}」；最像旧页《{old_title}》位于 `{old_path}`，旧页范围是「{old_scope}」。"
        "source_delta："
        f"本轮来源增量是「{source_delta}」。"
        "why_update_not_enough："
        "直接 update 旧页会把旧页从原有主题扩成另一个独立知识单元，降低旧页的聚焦度。"
        "why_related_link_not_enough："
        "只做 Related 只能表达关联，不能承载该来源新增的可复用结构、例子和价值点。"
    )


def _chinese_page_type_label(page_type: str) -> str:
    mapping = {
        "concept": "概念页",
        "entity": "实体页",
        "design": "设计页",
        "comparison": "对比页",
        "open_question": "未决问题页",
        "overview": "总览页",
    }
    return mapping.get(page_type.strip().lower(), "知识页")


def merge_update_noop_same_targets(items: list[WikiMergePlanItem]) -> list[WikiMergePlanItem]:
    updates_by_target = {item.canonical_target_path: item for item in items if item.action == "update"}
    noop_by_target: dict[str, list[WikiMergePlanItem]] = {}
    for item in items:
        if item.action == "noop" and item.canonical_target_path in updates_by_target:
            noop_by_target.setdefault(item.canonical_target_path, []).append(item)
    if not noop_by_target:
        return items

    merged: list[WikiMergePlanItem] = []
    skipped_noops: set[str] = set()
    for item in items:
        if item.action == "noop" and item.canonical_target_path in updates_by_target:
            skipped_noops.add(item.page_plan_id)
            continue
        if item.action != "update" or item.canonical_target_path not in noop_by_target:
            merged.append(item)
            continue
        covered_noops = noop_by_target[item.canonical_target_path]
        source_basis = SourceBasis(
            source_candidate_ids=_markdown_utils.dedupe_strings(
                [
                    *item.source_basis.source_candidate_ids,
                    *[
                        candidate_id
                        for noop in covered_noops
                        for candidate_id in noop.source_basis.source_candidate_ids
                    ],
                ]
            ),
            prepared_discovered_candidates=_markdown_utils.dedupe_strings(
                [
                    *item.source_basis.prepared_discovered_candidates,
                    *[
                        candidate
                        for noop in covered_noops
                        for candidate in noop.source_basis.prepared_discovered_candidates
                    ],
                ]
            ),
            source_locator=item.source_basis.source_locator,
        )
        related_pages = [*item.related_pages]
        for noop in covered_noops:
            related_pages.extend(noop.related_pages)
        deduped_related: list[RelatedPageRef] = []
        seen_related: set[str] = set()
        for related in related_pages:
            path = _wiki_markup.normalize_related_candidate_path(related.target_path)
            if path is None or path in seen_related:
                continue
            seen_related.add(path)
            deduped_related.append(related.model_copy(update={"target_path": path}))
            if len(deduped_related) >= _related_pages.FINAL_RELATED_LIMIT:
                break
        noop_ids = [noop.page_plan_id for noop in covered_noops]
        merged_ids = _markdown_utils.dedupe_strings([*item.merged_page_plan_ids, item.page_plan_id, *noop_ids])
        reason = f"同一 canonical target 出现 update + noop；{', '.join(noop_ids)} 已由 update `{item.page_plan_id}` 覆盖。"
        merged.append(
            item.model_copy(
                update={
                    "source_basis": source_basis,
                    "related_pages": deduped_related,
                    "merged_page_plan_ids": merged_ids,
                    "noop_covered_by_update": True,
                    "merge_reason": _markdown_utils.merge_markdown_blocks(item.merge_reason, reason),
                    "finalization_reason": _markdown_utils.merge_markdown_blocks(item.finalization_reason, reason),
                }
            )
        )
    return [item for item in merged if item.page_plan_id not in skipped_noops]


def merge_same_source_duplicate_creates(items: list[WikiMergePlanItem]) -> list[WikiMergePlanItem]:
    result = list(items)
    related_redirects: dict[str, str] = {}
    changed = True
    while changed:
        changed = False
        for left_index in range(len(result)):
            left = result[left_index]
            if left.action != "create":
                continue
            for right_index in range(left_index + 1, len(result)):
                right = result[right_index]
                if right.action != "create":
                    continue
                if not _same_source_duplicate_create(left, right):
                    continue
                canonical, suppressed = sorted([left, right], key=_duplicate_canonical_rank)
                merged = _absorb_duplicate_create(canonical, suppressed)
                suppressed_path = _wiki_markup.normalize_related_candidate_path(suppressed.canonical_target_path)
                canonical_path = _wiki_markup.normalize_related_candidate_path(merged.canonical_target_path)
                if suppressed_path is not None and canonical_path is not None and suppressed_path != canonical_path:
                    related_redirects[suppressed_path] = canonical_path
                keep_index = left_index if canonical is left else right_index
                drop_index = right_index if canonical is left else left_index
                result[keep_index] = merged
                del result[drop_index]
                changed = True
                break
            if changed:
                break
    return _rewrite_related_pages_after_path_redirects(result, related_redirects)


def _rewrite_related_pages_after_path_redirects(
    items: list[WikiMergePlanItem],
    redirects: dict[str, str],
) -> list[WikiMergePlanItem]:
    if not redirects:
        return items
    title_by_path = {
        path: item.display_title
        for item in items
        if (path := _wiki_markup.normalize_related_candidate_path(item.canonical_target_path)) is not None
    }
    rewritten: list[WikiMergePlanItem] = []
    for item in items:
        item_path = _wiki_markup.normalize_related_candidate_path(item.canonical_target_path)
        related_pages: list[RelatedPageRef] = []
        seen_related: set[str] = set()
        for related in item.related_pages:
            path = _wiki_markup.normalize_related_candidate_path(related.target_path)
            if path is None:
                continue
            target_path = _resolve_related_redirect(path, redirects)
            if target_path == item_path or target_path in seen_related:
                continue
            seen_related.add(target_path)
            related_pages.append(
                related.model_copy(
                    update={
                        "target_path": target_path,
                        "display_title": title_by_path.get(target_path, related.display_title),
                    }
                )
            )
            if len(related_pages) >= _related_pages.FINAL_RELATED_LIMIT:
                break
        related_absence_reason = item.related_absence_reason
        if not related_pages and related_absence_reason is None:
            related_absence_reason = "self_link_only" if item.related_pages else "no_candidate"
        rewritten.append(item.model_copy(update={"related_pages": related_pages, "related_absence_reason": related_absence_reason}))
    return rewritten


def _resolve_related_redirect(path: str, redirects: dict[str, str]) -> str:
    current = path
    seen: set[str] = set()
    while current in redirects and current not in seen:
        seen.add(current)
        next_path = redirects[current]
        if next_path == current:
            break
        current = next_path
    return current


def _same_source_duplicate_create(left: WikiMergePlanItem, right: WikiMergePlanItem) -> bool:
    left_title_tokens = _duplicate_tokens(left.display_title)
    right_title_tokens = _duplicate_tokens(right.display_title)
    if len(left_title_tokens | right_title_tokens) < 2:
        return False
    title_overlap = _text_similarity.jaccard(left_title_tokens, right_title_tokens)
    if title_overlap < 0.6 and not _text_similarity.both_agent_workflow_compare(left.display_title, right.display_title):
        return False
    intent_overlap = _text_similarity.jaccard(
        _duplicate_tokens(_duplicate_intent_text(left)),
        _duplicate_tokens(_duplicate_intent_text(right)),
    )
    if intent_overlap < 0.48 and not _text_similarity.both_agent_workflow_compare(left.display_title, right.display_title):
        return False
    if _duplicate_shape_conflict(left, right) and title_overlap < 0.82:
        return False
    return True


def _duplicate_tokens(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", _text_similarity.parenthetical_translation_core(text).lower())
    normalized = normalized.replace("workflow", "workflow").replace("workflows", "workflow")
    normalized = normalized.replace("agentic", "agent").replace("agents", "agent")
    tokens = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", normalized))
    stop = {"concept", "comparison", "design", "entity", "open", "question", "概念", "设计", "实体", "问题", "对比", "区别", "比较", "什么", "如何", "为什么", "页面"}
    return {token for token in tokens if token not in stop}


def _duplicate_intent_text(item: WikiMergePlanItem) -> str:
    return "\n".join(
        [
            item.display_title,
            item.new_understanding,
            item.knowledge_delta,
            item.why_this_matters,
            item.reason,
            " ".join(item.value_points),
            "\n".join(item.section_plans.values()),
        ]
    )


def _duplicate_shape_conflict(left: WikiMergePlanItem, right: WikiMergePlanItem) -> bool:
    pair = {left.page_type, right.page_type}
    if pair <= {"concept", "comparison"}:
        return False
    if pair <= {"concept", "open_question"}:
        return True
    if pair <= {"concept", "design"}:
        left_tokens = _duplicate_tokens(left.display_title)
        right_tokens = _duplicate_tokens(right.display_title)
        return _text_similarity.jaccard(left_tokens, right_tokens) < 0.9
    return len(pair) > 1


def _duplicate_canonical_rank(item: WikiMergePlanItem) -> tuple[int, int, str]:
    title = item.display_title.lower()
    type_rank = {
        "comparison": 0 if any(marker in title for marker in ["vs", "对比", "比较", "区别"]) else 2,
        "design": 1,
        "concept": 2,
        "open_question": 3,
        "entity": 4,
    }.get(item.page_type, 5)
    return (type_rank, -len(_duplicate_intent_text(item)), item.canonical_target_path)


def _absorb_duplicate_create(canonical: WikiMergePlanItem, suppressed: WikiMergePlanItem) -> WikiMergePlanItem:
    source_basis = SourceBasis(
        source_candidate_ids=_markdown_utils.dedupe_strings([*canonical.source_basis.source_candidate_ids, *suppressed.source_basis.source_candidate_ids]),
        prepared_discovered_candidates=_markdown_utils.dedupe_strings(
            [*canonical.source_basis.prepared_discovered_candidates, *suppressed.source_basis.prepared_discovered_candidates]
        ),
        source_locator=canonical.source_basis.source_locator or suppressed.source_basis.source_locator,
    )
    section_plans = dict(canonical.section_plans)
    for key, value in suppressed.section_plans.items():
        if key in section_plans:
            section_plans[key] = _markdown_utils.merge_markdown_blocks(section_plans[key], f"合并自 `{suppressed.page_plan_id}`：{value}")
        else:
            section_plans[key] = f"合并自 `{suppressed.page_plan_id}`：{value}"
    related_pages = [*canonical.related_pages, *suppressed.related_pages]
    merged_related: list[RelatedPageRef] = []
    seen_related: set[str] = set()
    for related in related_pages:
        path = _wiki_markup.normalize_related_candidate_path(related.target_path)
        if path is None or path in seen_related or path == canonical.canonical_target_path:
            continue
        seen_related.add(path)
        merged_related.append(related.model_copy(update={"target_path": path}))
        if len(merged_related) >= _related_pages.FINAL_RELATED_LIMIT:
            break
    absorbed_note = (
        f"同源近重复自动合并：`{suppressed.page_plan_id}`（{suppressed.display_title}）的信息已并入 "
        f"`{canonical.page_plan_id}`；其 section intent、examples/value points 通过 source_basis/section_plans 进入 canonical draft。"
    )
    return canonical.model_copy(
        update={
            "source_basis": source_basis,
            "section_plans": section_plans,
            "related_pages": merged_related,
            "value_points": _markdown_utils.dedupe_strings([*canonical.value_points, *suppressed.value_points]),
            "reuse_scenarios": _markdown_utils.dedupe_strings([*canonical.reuse_scenarios, *suppressed.reuse_scenarios]),
            "merged_page_plan_ids": _markdown_utils.dedupe_strings(
                [*canonical.merged_page_plan_ids, canonical.page_plan_id, suppressed.page_plan_id, *suppressed.merged_page_plan_ids]
            ),
            "merge_reason": _markdown_utils.merge_markdown_blocks(canonical.merge_reason, absorbed_note),
            "finalization_reason": _markdown_utils.merge_markdown_blocks(canonical.finalization_reason, absorbed_note),
            "quality_risks": _markdown_utils.dedupe_strings(
                [
                    *canonical.quality_risks,
                    *suppressed.quality_risks,
                    f"已自动合并 `{suppressed.page_plan_id}`；draft review 需确认被合并候选没有独特信息丢失。",
                ]
            ),
        }
    )


def normalize_model_wiki_target_path(value: str) -> str:
    path = value.strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    if path.startswith("/"):
        path = path[1:]
    if path.startswith("wiki/"):
        path = path.removeprefix("wiki/")
    return path


def merge_plan_all_create_review_reason(
    plan: WikiMergePlanArtifact,
    *,
    max_auto_create_items: int = MAX_AUTO_APPROVED_ALL_CREATE_ITEMS,
) -> str:
    if not plan.items or any(item.action != "create" for item in plan.items):
        return ""
    if len(plan.items) > max_auto_create_items:
        return (
            f"合并计划一次 create {len(plan.items)} 个页面，超过自动通过上限 "
            f"{max_auto_create_items}；请 revise 聚合或延后低优先级页面。"
        )
    strong_risky = [
        item
        for item in _merge_plan_risks.merge_plan_create_overlap_risk_items(plan)
        if item.strongest_overlap.strength == "strong"
    ]
    if strong_risky:
        names = ", ".join(f"{item.page_plan_id}:strong" for item in strong_risky[:8])
        return f"合并计划全部为 create，但存在强召回风险（{names}）；请审核这些页面为什么不应 update 到已有知识页。"
    weak_reason_medium = [
        item
        for item in _merge_plan_risks.merge_plan_create_overlap_risk_items(plan)
        if item.strongest_overlap.strength == "medium"
        and (
            create_reason_needs_repair(item.why_not_update)
            or item.why_not_update.startswith("本地补充：")
            or LOCAL_MEDIUM_CREATE_REASON_MARKER in item.finalization_reason
            or medium_create_generic_old_title_review_reason(item)
        )
    ]
    if not weak_reason_medium:
        return ""
    names = ", ".join(f"{item.page_plan_id}:medium" for item in weak_reason_medium[:8])
    return f"合并计划全部为 create，但存在中等召回风险且 create 理由不充分、仅由本地补充或旧页标题像通用概念页（{names}）；请审核这些页面为什么不应 update 到已有知识页。"


def medium_create_generic_old_title_review_reason(item: WikiMergePlanItem, *, old_display_title: str = "") -> str:
    if item.action != "create" or item.strongest_overlap.strength != "medium" or not item.strongest_overlap.path:
        return ""
    if create_reason_needs_repair(item.why_not_update):
        return ""
    old_path = item.strongest_overlap.path
    old_path_stem = _wiki_markup.clean_display_title(Path(old_path).stem) if old_path else ""
    old_label = " ".join(part for part in [old_display_title.strip(), old_path, old_path_stem] if part)
    new_label = f"{item.display_title} {item.canonical_target_path}"
    if not _merge_titles_share_specific_concept_terms(new_label, old_label):
        return ""
    if not _merge_old_title_looks_source_neutral_generic(old_label):
        return ""
    if not _merge_reason_dismisses_old_as_specific_or_new_as_generic(item.why_not_update):
        return ""
    old_display = old_display_title.strip() or _wiki_markup.clean_display_title(Path(item.strongest_overlap.path).stem)
    return (
        f"召回到中等相关旧页 `{item.strongest_overlap.path}`，旧页标题《{old_display}》像通用概念页，"
        "但模型选择 create 的理由把旧页归为具体平台/产品/实现或把新页归为通用概念；"
        "需要人工确认是否应 update 到已有知识页，或是否真的需要拆成独立页面。"
    )


def _merge_titles_share_specific_concept_terms(new_label: str, old_label: str) -> bool:
    shared = _merge_overlap_concept_terms(new_label) & _merge_overlap_concept_terms(old_label)
    if "memory" in shared:
        return True
    broad_only = {"agent", "ai", "system"}
    return len(shared - broad_only) >= 2


def _merge_overlap_concept_terms(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", text).lower().replace("_", " ").replace("-", " ")
    terms: set[str] = set()
    if re.search(r"\bagents?\b", normalized) or "智能体" in normalized:
        terms.add("agent")
    has_neicun_agent_context = (
        "内存" in normalized
        and (re.search(r"\b(?:ai|agents?)\b", normalized) or "智能体" in normalized)
    )
    if (
        re.search(r"\b(?:memory|memories)\b", normalized)
        or "记忆" in normalized
        or "回忆" in normalized
        or has_neicun_agent_context
    ):
        terms.add("memory")
    if re.search(r"\bcontexts?\b", normalized) or "上下文" in normalized:
        terms.add("context")
    if re.search(r"\bworkflows?\b", normalized) or "工作流" in normalized:
        terms.add("workflow")
    if re.search(r"\bharness(?:es)?\b", normalized):
        terms.add("harness")
    if re.search(r"\brag\b", normalized):
        terms.add("rag")
    if re.search(r"\bevals?\b|\bevaluation\b", normalized) or "评估" in normalized:
        terms.add("evaluation")
    return terms


def _merge_old_title_looks_source_neutral_generic(old_label: str) -> bool:
    normalized = unicodedata.normalize("NFKC", old_label).lower()
    if _merge_label_has_source_specific_token(normalized):
        return False
    terms = _merge_overlap_concept_terms(normalized)
    return bool(terms & {"memory", "context", "workflow", "harness", "rag", "evaluation"})


def _merge_label_has_source_specific_token(normalized: str) -> bool:
    ascii_tokens = {
        "anthropic",
        "cloudflare",
        "claude",
        "github",
        "langchain",
        "mem0",
        "openai",
        "qwen",
        "redis",
        "readme",
        "api",
        "sdk",
    }
    cjk_tokens = {
        "平台",
        "产品",
        "实现",
        "教程",
        "官方",
        "项目",
        "案例",
        "论文",
        "访谈",
        "播客",
    }
    if any(re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", normalized) for token in ascii_tokens):
        return True
    return any(token in normalized for token in cjk_tokens)


def _merge_reason_dismisses_old_as_specific_or_new_as_generic(reason: str) -> bool:
    normalized = unicodedata.normalize("NFKC", reason).lower()
    generic_new_markers = ["通用", "泛化", "generic", "general", "概念页", "独立概念", "通用概念"]
    old_specific_negated = _merge_reason_negates_old_specific_scope(normalized)
    old_explicitly_generic = _merge_reason_says_old_is_source_neutral_generic(normalized)
    old_called_specific = _merge_reason_old_scope_called_specific(normalized) and not old_specific_negated
    new_called_generic = any(
        marker in normalized
        for marker in ["新页", "新页面", "新增页", "新增页面", "新增知识页", "本轮", "new page", "new knowledge page"]
    ) and any(marker in normalized for marker in generic_new_markers) and not old_explicitly_generic
    return old_called_specific or new_called_generic


def _merge_reason_old_scope_called_specific(normalized: str) -> bool:
    if _merge_reason_reasserts_old_specific_after_negation(normalized):
        return True
    patterns = [
        rf"{_OLD_PAGE_ANCHOR}[^。；;，,.\n]{{0,64}}{_CN_SOURCE_SPECIFIC}",
        rf"{_OLD_PAGE_ANCHOR}[^。；;，,.\n]{{0,96}}{_EN_SOURCE_SPECIFIC}",
        rf"{_OLD_PAGE_ANCHOR}[^。；;.\n]{{0,48}}[，,]\s*[^。；;，,.\n]{{0,16}}{_SCOPE_PREDICATE}[^。；;，,.\n]{{0,64}}{_CN_SOURCE_SPECIFIC}",
        rf"{_OLD_PAGE_ANCHOR}[^。；;.\n]{{0,64}}[，,]\s*[^。；;，,.\n]{{0,24}}{_SCOPE_PREDICATE}[^。；;，,.\n]{{0,96}}{_EN_SOURCE_SPECIFIC}",
    ]
    return any(re.search(pattern, normalized) for pattern in patterns)


def _merge_reason_negates_old_specific_scope(normalized: str) -> bool:
    if _merge_reason_reasserts_old_specific_after_negation(normalized):
        return False
    patterns = [
        rf"{_OLD_PAGE_ANCHOR}[^。；;.\n]{{0,24}}(?:不是|并非|并不是|不属于|不应被视为)[^。；;.\n]{{0,18}}{_CN_SCOPE_SPECIFIC}",
        rf"{_OLD_PAGE_ANCHOR}[^。；;.\n]{{0,24}}(?:is not|isn't|should not be treated as)[^。；;.\n]{{0,18}}{_EN_SCOPE_SPECIFIC}",
        rf"{_OLD_TITLE_ANCHOR}[^。；;.\n]{{0,24}}(?:没有|不含|未包含|无)[^。；;.\n]{{0,24}}(?:平台词|产品词|产品/平台词|platform token|product token|source token)",
        rf"{_OLD_TITLE_ANCHOR}[^。；;.\n]{{0,24}}(?:has no|does not contain|doesn't contain|lacks)[^。；;.\n]{{0,24}}(?:platform|product|source)[^。；;.\n]{{0,10}}token",
    ]
    return any(re.search(pattern, normalized) for pattern in patterns)


def _merge_reason_reasserts_old_specific_after_negation(normalized: str) -> bool:
    patterns = [
        rf"{_OLD_PAGE_ANCHOR}[^。；;.\n]{{0,32}}(?:不是|并非|并不是|不只是)[^。；;.\n]{{0,24}}(?:而是|但其实|但仍是|但它是)[^。；;.\n]{{0,32}}{_CN_SOURCE_SPECIFIC}",
        rf"{_OLD_PAGE_ANCHOR}[^。\n]{{0,80}}(?:is not merely|isn't merely|is not just|isn't just|not ordinary)[^。\n]{{0,80}}(?:but|rather|however|it is|it remains)[^。\n]{{0,48}}{_EN_SOURCE_SPECIFIC}",
    ]
    return any(re.search(pattern, normalized) for pattern in patterns)


def _merge_reason_says_old_is_source_neutral_generic(normalized: str) -> bool:
    if _merge_reason_negates_old_generic_scope(normalized):
        return False
    patterns = [
        rf"{_OLD_PAGE_ANCHOR}[^。；;.\n]{{0,24}}(?:本身)?(?:像|是|属于)[^。；;.\n]{{0,12}}{_CN_GENERIC_WITH_SOURCE_NEUTRAL}",
        rf"{_OLD_PAGE_ANCHOR}[^。；;.\n]{{0,24}}(?:is|looks|appears)[^。；;.\n]{{0,12}}{_EN_GENERIC}",
        rf"{_OLD_TITLE_ANCHOR}[^。；;.\n]{{0,24}}(?:没有|不含|未包含|无)[^。；;.\n]{{0,24}}(?:平台词|产品词|产品/平台词|platform token|product token|source token)",
        rf"{_OLD_TITLE_ANCHOR}[^。；;.\n]{{0,24}}(?:has no|does not contain|doesn't contain|lacks)[^。；;.\n]{{0,24}}(?:platform|product|source)[^。；;.\n]{{0,10}}token",
    ]
    return any(re.search(pattern, normalized) for pattern in patterns)


def _merge_reason_negates_old_generic_scope(normalized: str) -> bool:
    patterns = [
        rf"{_OLD_PAGE_ANCHOR}[^。；;，,.\n]{{0,24}}(?:不是|并非|并不是|不属于|不应被视为)[^。；;，,.\n]{{0,18}}{_CN_GENERIC}",
        rf"{_OLD_PAGE_ANCHOR}[^。；;，,.\n]{{0,24}}(?:is not|isn't|not a|not an|should not be treated as)[^。；;，,.\n]{{0,18}}{_EN_GENERIC}",
    ]
    return any(re.search(pattern, normalized) for pattern in patterns)
