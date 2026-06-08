from __future__ import annotations

import re
import unicodedata
from typing import Any, Literal

from . import frontmatter as _frontmatter
from . import markdown_utils as _markdown_utils
from . import open_questions as _open_questions
from . import source_excerpt as _source_excerpt
from . import text_similarity as _text_similarity
from .hash_utils import sha256_bytes
from .models import SourceDigestArtifact, SourceDigestCandidate
from .system_pages import format_markdown_table

SOURCE_DIGEST_BUDGET_GROUP_ORDER = ("concepts", "designs", "comparisons", "open_questions", "entities")
SOURCE_DIGEST_AGGREGATION_MIN_CANDIDATES = 2
SOURCE_DIGEST_AGGREGATION_CLUSTER_SIMILARITY = 0.18
SOURCE_DIGEST_PROMOTED_AGGREGATION_MIN_REPLACEMENT_SIMILARITY = 0.35
DEFERRED_AGGREGATION_GROUP_LABELS = {
    "concepts": "延后概念",
    "designs": "延后设计模式",
    "comparisons": "延后对比",
    "open_questions": "延后未决问题",
    "entities": "延后实体",
}
SOURCE_DIGEST_ANCHOR_ENTITIES: dict[str, dict[str, str]] = {
    "Managed Agents": {
        "summary": "Managed Agents 是源材料显式讨论的托管智能体系统，用于将大脑、会话和双手解耦，并容纳未来不同 harness、sandbox 或其他组件。",
        "why_matters": "它是本材料的中心系统名称，后续材料很可能继续补充其产品、架构和使用边界。",
        "wiki_value": "作为稳定实体锚点，可承接后续关于 Claude Code、harness、sandbox、session 与托管代理能力的更新。",
        "resolution_hint": "deterministic_source_anchor_entity: source title/body repeatedly names Managed Agents; keep as a central reusable entity anchor before page budget.",
    },
    "Claude Code": {
        "summary": "Claude Code 是源材料显式提到的 Anthropic 编程 harness/产品，在本材料中作为 Managed Agents 可适配并广泛使用的 harness 示例出现。",
        "why_matters": "它是后续访谈、产品方法和托管智能体材料之间最容易复用的产品实体锚点。",
        "wiki_value": "让后续 Claude Code 访谈可以 update 既有页面，而不是把架构材料中的 harness 视角遗失到孤立相关页里。",
        "resolution_hint": "deterministic_source_anchor_entity: source body explicitly names Claude Code as an excellent harness; keep as a reusable update target when the model omits it.",
    },
    "Cowork": {
        "summary": "Cowork 是源材料显式提到的 Anthropic 知识工作协作者产品，可作为 Claude Code 之外的产品实体锚点。",
        "why_matters": "它经常与 Claude Code 同源出现，适合承接后续关于非编程知识工作场景的更新。",
        "wiki_value": "提供稳定产品实体页，便于后续比较、团队组织和使用场景材料进行 update 或互链。",
        "resolution_hint": "deterministic_source_anchor_entity: source explicitly names Cowork as a durable product/entity anchor.",
    },
}

SOURCE_ANCHOR_RELATED_LIMIT = 3


def augment_source_digest_anchor_entities(
    digest: SourceDigestArtifact,
    approved_prepared_text: str,
) -> SourceDigestArtifact:
    additions: list[SourceDigestCandidate] = []
    existing_keys = {
        source_digest_candidate_title_key(candidate)
        for candidate in digest.ingest_candidates()
        if source_digest_candidate_title_key(candidate)
    }
    for anchor, metadata in SOURCE_DIGEST_ANCHOR_ENTITIES.items():
        anchor_key = _source_excerpt.normalized_source_match_text(anchor)
        if not anchor_key or anchor_key in existing_keys:
            continue
        signal = source_anchor_signal(approved_prepared_text, anchor)
        if not signal["should_add"]:
            continue
        candidate = SourceDigestCandidate(
            candidate_id=f"auto-ent-{anchor_key}",
            name=anchor,
            type="entity",
            one_sentence_summary=metadata["summary"],
            why_matters=metadata["why_matters"],
            wiki_value=metadata["wiki_value"],
            source_locator=signal["source_locator"],
            suggested_page_title=anchor,
            related_candidates=source_anchor_related_candidates(anchor, digest),
            resolution_hint=(
                f"{metadata['resolution_hint']} occurrence_count={signal['occurrence_count']}; "
                f"signal_reason={signal['reason']}"
            ),
            duplicate_risk="medium",
        )
        additions.append(candidate)
        existing_keys.add(anchor_key)
    if not additions:
        return digest
    return digest.model_copy(update={"entities": [*additions, *digest.entities]})


def source_anchor_signal(text: str, anchor: str) -> dict[str, Any]:
    occurrence_count = source_anchor_occurrence_count(text, anchor)
    if occurrence_count <= 0:
        return {
            "should_add": False,
            "occurrence_count": 0,
            "source_locator": "",
            "reason": "absent",
        }
    frontmatter = _frontmatter.parse_frontmatter(text) or {}
    metadata_text = "\n".join(
        str(frontmatter.get(key) or "")
        for key in ["title", "description", "source", "author"]
    )
    heading_text = "\n".join(line for line in text.splitlines() if line.lstrip().startswith("#"))
    high_signal_text = "\n".join([metadata_text, heading_text])
    high_signal = source_anchor_occurrence_count(high_signal_text, anchor) > 0
    explicit_context = source_anchor_has_explicit_context(text, anchor)
    should_add = high_signal or occurrence_count >= 3 or explicit_context
    reason_parts: list[str] = []
    if high_signal:
        reason_parts.append("metadata_or_heading")
    if occurrence_count >= 3:
        reason_parts.append("repeated")
    if explicit_context:
        reason_parts.append("explicit_context")
    return {
        "should_add": should_add,
        "occurrence_count": occurrence_count,
        "source_locator": source_anchor_first_locator(text, anchor),
        "reason": "+".join(reason_parts) or "weak_mention",
    }


def source_anchor_occurrence_count(text: str, anchor: str) -> int:
    if not text or not anchor:
        return 0
    pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(anchor)}(?![A-Za-z0-9])", re.IGNORECASE)
    return len(pattern.findall(unicodedata.normalize("NFKC", text)))


def source_anchor_has_explicit_context(text: str, anchor: str) -> bool:
    lower = unicodedata.normalize("NFKC", text).lower()
    anchor_lower = anchor.lower()
    context_terms = {
        "managed agents": [
            "meta-harness",
            "托管智能体",
            "managed agents is",
            "managed agents can",
            "managed agents,",
        ],
        "claude code": [
            "excellent harness",
            "广泛使用",
            "head of product",
            "创建了claude code",
            "claude code团队",
            "claude code和cowork",
        ],
        "cowork": [
            "claude code和cowork",
            "head of product",
            "知识工作",
            "not code",
            "非代码",
        ],
    }.get(anchor_lower, [])
    for match in re.finditer(rf"(?<![a-z0-9]){re.escape(anchor_lower)}(?![a-z0-9])", lower):
        window = lower[max(0, match.start() - 120) : min(len(lower), match.end() + 120)]
        if any(term in window for term in context_terms):
            return True
    return False


def source_anchor_first_locator(text: str, anchor: str) -> str:
    pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(anchor)}(?![A-Za-z0-9])", re.IGNORECASE)
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern.search(unicodedata.normalize("NFKC", line)):
            return f"L{line_no}"
    return ""


def source_anchor_related_candidates(anchor: str, digest: SourceDigestArtifact) -> list[str]:
    existing_titles = [candidate.suggested_page_title or candidate.name for candidate in digest.ingest_candidates()]
    desired = {
        "Managed Agents": ["Claude Code", "Harness（适配框架）", "Session（会话）", "大脑与双手解耦"],
        "Claude Code": ["Managed Agents", "Harness（适配框架）", "Cowork"],
        "Cowork": ["Claude Code"],
    }.get(anchor, [])
    available = [title for title in desired if title in existing_titles or title in SOURCE_DIGEST_ANCHOR_ENTITIES]
    return available[:SOURCE_ANCHOR_RELATED_LIMIT]


def cap_source_digest_candidates(digest: SourceDigestArtifact, max_candidates: int) -> tuple[SourceDigestArtifact, dict[str, Any]]:
    groups: dict[str, list[SourceDigestCandidate]] = {
        "entities": list(digest.entities),
        "concepts": list(digest.concepts),
        "designs": list(digest.designs),
        "comparisons": list(digest.comparisons),
        "open_questions": list(digest.open_questions),
    }
    total_before_dedupe = sum(len(items) for items in groups.values())
    groups, deduped_candidates = dedupe_source_digest_groups(groups)
    total = sum(len(items) for items in groups.values())
    budget = max(1, int(max_candidates))
    if total <= budget:
        report = source_digest_budget_report(
            groups,
            {},
            budget=budget,
            total=total,
            applied=False,
            total_before_dedupe=total_before_dedupe,
            deduped_candidates=deduped_candidates,
        )
        capped = digest.model_copy(
            update={
                "entities": groups["entities"],
                "concepts": groups["concepts"],
                "designs": groups["designs"],
                "comparisons": groups["comparisons"],
                "open_questions": groups["open_questions"],
            }
        )
        return capped, report

    selected: dict[str, list[SourceDigestCandidate]] = {name: [] for name in groups}
    indexes = {name: 0 for name in groups}
    selected_count = 0
    while selected_count < budget:
        progressed = False
        for group_name in SOURCE_DIGEST_BUDGET_GROUP_ORDER:
            group = groups[group_name]
            index = indexes[group_name]
            if index >= len(group):
                continue
            selected[group_name].append(group[index])
            indexes[group_name] += 1
            selected_count += 1
            progressed = True
            if selected_count >= budget:
                break
        if not progressed:
            break

    deferred: dict[str, list[SourceDigestCandidate]] = {
        group_name: groups[group_name][indexes[group_name] :]
        for group_name in groups
    }
    selected, selected_aggregations = promote_deferred_aggregations_into_selection(selected, deferred)
    represented_by = {
        candidate_id: aggregation["candidate_id"]
        for aggregation in selected_aggregations
        for candidate_id in aggregation.get("deferred_candidate_ids", [])
    }
    budget_deferred_candidates = list(digest.budget_deferred_candidates)
    for group_name, candidates in deferred.items():
        budget_deferred_candidates.extend(
            deferred_digest_candidate(group_name, candidate, represented_by=represented_by.get(candidate.candidate_id, ""))
            for candidate in candidates
        )
    capped = digest.model_copy(
        update={
            "entities": selected["entities"],
            "concepts": selected["concepts"],
            "designs": selected["designs"],
            "comparisons": selected["comparisons"],
            "open_questions": selected["open_questions"],
            "budget_deferred_candidates": budget_deferred_candidates,
        }
    )
    report = source_digest_budget_report(
        selected,
        deferred,
        budget=budget,
        total=total,
        applied=True,
        total_before_dedupe=total_before_dedupe,
        deduped_candidates=deduped_candidates,
        selected_deferred_aggregations=selected_aggregations,
    )
    return capped, report


def dedupe_source_digest_groups(
    groups: dict[str, list[SourceDigestCandidate]],
) -> tuple[dict[str, list[SourceDigestCandidate]], list[dict[str, Any]]]:
    deduped_groups: dict[str, list[SourceDigestCandidate]] = {group_name: [] for group_name in groups}
    deduped_candidates: list[dict[str, Any]] = []
    for group_name, candidates in groups.items():
        by_key: dict[str, int] = {}
        for candidate in candidates:
            key = source_digest_candidate_dedupe_key(group_name, candidate)
            canonical_index: int | None = by_key.get(key) if key else None
            if canonical_index is None and group_name != "open_questions":
                canonical_index = source_digest_duplicate_candidate_index(
                    group_name,
                    deduped_groups[group_name],
                    candidate,
                )
                if canonical_index is not None and not key:
                    key = source_digest_candidate_similarity_key(group_name, deduped_groups[group_name][canonical_index], candidate)
            if canonical_index is None:
                if key:
                    by_key[key] = len(deduped_groups[group_name])
                deduped_groups[group_name].append(candidate)
                continue
            canonical = deduped_groups[group_name][canonical_index]
            deduped_groups[group_name][canonical_index] = merge_source_digest_duplicate_candidate(
                group_name,
                key,
                canonical,
                candidate,
            )
            deduped_candidates.append(
                {
                    "group": group_name,
                    "dedupe_key": key,
                    "kept_candidate_id": canonical.candidate_id,
                    "merged_candidate_id": candidate.candidate_id,
                    "merged_title": candidate.suggested_page_title or candidate.name,
                    "source_locator": candidate.source_locator,
                }
            )
    return deduped_groups, deduped_candidates


def source_digest_duplicate_candidate_index(
    group_name: str,
    candidates: list[SourceDigestCandidate],
    candidate: SourceDigestCandidate,
) -> int | None:
    for index, existing in enumerate(candidates):
        if source_digest_candidates_semantically_duplicate(group_name, existing, candidate):
            return index
    return None


def source_digest_candidate_dedupe_key(group_name: str, candidate: SourceDigestCandidate) -> str:
    if group_name != "open_questions":
        key = source_digest_candidate_title_key(candidate)
        return f"{group_name}:title:{key}" if len(key) >= 6 else ""
    basis = (
        candidate.open_question_or_tension
        or candidate.suggested_page_title
        or candidate.name
        or candidate.one_sentence_summary
    )
    key = _open_questions.open_question_key(basis)
    return f"open_questions:{key}" if key and len(key) >= 6 else ""


def source_digest_candidates_semantically_duplicate(
    group_name: str,
    left: SourceDigestCandidate,
    right: SourceDigestCandidate,
) -> bool:
    if group_name == "open_questions":
        return source_digest_candidate_dedupe_key(group_name, left) == source_digest_candidate_dedupe_key(group_name, right)
    left_key = source_digest_candidate_title_key(left)
    right_key = source_digest_candidate_title_key(right)
    if left_key and left_key == right_key:
        return True
    title_similarity = source_digest_text_similarity(
        left.suggested_page_title or left.name,
        right.suggested_page_title or right.name,
    )
    intent_similarity = source_digest_text_similarity(
        source_digest_candidate_intent_text(left),
        source_digest_candidate_intent_text(right),
    )
    if group_name == "entities":
        return title_similarity >= 0.82 and intent_similarity >= 0.45
    if group_name == "comparisons" and _text_similarity.both_agent_workflow_compare(left.suggested_page_title or left.name, right.suggested_page_title or right.name):
        return intent_similarity >= 0.35
    shared_title_terms = source_digest_shared_signal_terms(left.suggested_page_title or left.name, right.suggested_page_title or right.name)
    return (title_similarity >= 0.55 and intent_similarity >= 0.42) or (
        title_similarity >= 0.40 and intent_similarity >= 0.55
    ) or (intent_similarity >= 0.56 and bool(shared_title_terms))


def source_digest_candidate_title_key(candidate: SourceDigestCandidate) -> str:
    return source_digest_title_key(candidate.suggested_page_title or candidate.name)


def source_digest_title_key(title: str) -> str:
    core_title = _text_similarity.parenthetical_translation_core(title)
    core_key = _source_excerpt.normalized_source_match_text(core_title)
    if len(core_key) >= 4:
        return core_key
    return _source_excerpt.normalized_source_match_text(title)



def source_digest_candidate_similarity_key(
    group_name: str,
    canonical: SourceDigestCandidate,
    duplicate: SourceDigestCandidate,
) -> str:
    title_similarity = source_digest_text_similarity(
        canonical.suggested_page_title or canonical.name,
        duplicate.suggested_page_title or duplicate.name,
    )
    intent_similarity = source_digest_text_similarity(
        source_digest_candidate_intent_text(canonical),
        source_digest_candidate_intent_text(duplicate),
    )
    return f"{group_name}:similarity:title={title_similarity:.2f}:intent={intent_similarity:.2f}"


def source_digest_candidate_intent_text(candidate: SourceDigestCandidate) -> str:
    return "\n".join(
        [
            candidate.suggested_page_title,
            candidate.name,
            candidate.one_sentence_summary,
            candidate.why_matters,
            candidate.wiki_value,
            candidate.open_question_or_tension,
        ]
    )


def source_digest_text_similarity(left: str, right: str) -> float:
    return _text_similarity.jaccard(source_digest_similarity_terms(left), source_digest_similarity_terms(right))


def source_digest_shared_signal_terms(left: str, right: str) -> set[str]:
    generic = {"ai", "pm", "产品", "管理", "主题", "材料", "知识", "页面"}
    return {
        term
        for term in source_digest_similarity_terms(left) & source_digest_similarity_terms(right)
        if term not in generic and len(term) >= 2
    }


def source_digest_similarity_terms(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", text.lower())
    terms = set(re.findall(r"[a-z0-9]{2,}", normalized))
    for segment in re.findall(r"[\u4e00-\u9fff]{2,}", normalized):
        terms.add(segment)
        max_size = min(4, len(segment))
        for size in range(2, max_size + 1):
            for index in range(0, len(segment) - size + 1):
                terms.add(segment[index : index + size])
    stop = {
        "concept",
        "comparison",
        "design",
        "entity",
        "open",
        "question",
        "概念",
        "设计",
        "实体",
        "问题",
        "对比",
        "比较",
        "页面",
        "来源",
        "摘要",
    }
    return {term for term in terms if term not in stop and len(term) >= 2}


def merge_source_digest_duplicate_candidate(
    group_name: str,
    dedupe_key: str,
    canonical: SourceDigestCandidate,
    duplicate: SourceDigestCandidate,
) -> SourceDigestCandidate:
    duplicate_title = duplicate.suggested_page_title or duplicate.name
    note = (
        f"source_digest_semantic_dedupe: `{duplicate.candidate_id}` ({duplicate_title}) "
        f"按 `{dedupe_key}` 合并进 `{canonical.candidate_id}`，不单独占用本轮页面预算。"
    )
    if duplicate.source_locator:
        note += f" 来源定位：{duplicate.source_locator}。"
    if duplicate.one_sentence_summary:
        note += f" 变体摘要：{duplicate.one_sentence_summary}"
    duplicate_risk = "high" if "high" in {canonical.duplicate_risk, duplicate.duplicate_risk} else (
        "medium" if "medium" in {canonical.duplicate_risk, duplicate.duplicate_risk} else canonical.duplicate_risk
    )
    return canonical.model_copy(
        update={
            "related_candidates": _markdown_utils.dedupe_strings(
                [*canonical.related_candidates, duplicate.candidate_id, *duplicate.related_candidates]
            ),
            "resolution_hint": _markdown_utils.merge_markdown_blocks(canonical.resolution_hint, note),
            "open_question_or_tension": _markdown_utils.merge_markdown_blocks(
                canonical.open_question_or_tension,
                duplicate.open_question_or_tension,
            ),
            "duplicate_risk": duplicate_risk,
        }
    )


def promote_deferred_aggregations_into_selection(
    selected: dict[str, list[SourceDigestCandidate]],
    deferred: dict[str, list[SourceDigestCandidate]],
) -> tuple[dict[str, list[SourceDigestCandidate]], list[dict[str, Any]]]:
    selected = {group_name: list(candidates) for group_name, candidates in selected.items()}
    selected_aggregations: list[dict[str, Any]] = []
    for group_name in SOURCE_DIGEST_BUDGET_GROUP_ORDER:
        deferred_candidates = list(deferred.get(group_name, []))
        clusters = deferred_candidate_topic_clusters(group_name, deferred_candidates)
        if not clusters or not selected.get(group_name):
            continue
        remaining_selected = list(selected[group_name])
        promoted: list[SourceDigestCandidate] = []
        for cluster_index, cluster in enumerate(clusters, start=1):
            if not remaining_selected:
                break
            replacement_index, replacement_score = select_aggregation_replacement(remaining_selected, cluster)
            if replacement_score < SOURCE_DIGEST_PROMOTED_AGGREGATION_MIN_REPLACEMENT_SIMILARITY:
                continue
            replaced = remaining_selected.pop(replacement_index)
            represented = [replaced, *cluster]
            aggregate = deferred_aggregation_digest_candidate(group_name, represented, replaced_candidate_id=replaced.candidate_id)
            promoted.append(aggregate)
            selected_aggregations.append(
                {
                    "group": group_name,
                    "cluster_index": cluster_index,
                    "candidate_id": aggregate.candidate_id,
                    "suggested_page_title": aggregate.suggested_page_title,
                    "suggested_page_type": aggregate.type,
                    "replaced_candidate_id": replaced.candidate_id,
                    "replacement_similarity": round(replacement_score, 4),
                    "represented_candidate_ids": [candidate.candidate_id for candidate in represented],
                    "deferred_candidate_ids": [candidate.candidate_id for candidate in cluster],
                    "cluster_terms": sorted(deferred_cluster_terms(cluster))[:8],
                    "reason": (
                        f"`{group_name}` deferred topic cluster {cluster_index} has {len(cluster)} candidate(s); "
                        "one selected candidate was folded into an aggregation candidate to keep the page budget constant."
                    ),
                }
            )
        selected[group_name] = [*remaining_selected, *promoted]
    return selected, selected_aggregations


def deferred_candidate_topic_clusters(
    group_name: str,
    candidates: list[SourceDigestCandidate],
) -> list[list[SourceDigestCandidate]]:
    clusters: list[list[SourceDigestCandidate]] = []
    for candidate in candidates:
        best_index = -1
        best_score = 0.0
        for index, cluster in enumerate(clusters):
            score = candidate_cluster_similarity(group_name, candidate, cluster)
            if score > best_score:
                best_index = index
                best_score = score
        if best_index >= 0 and best_score >= SOURCE_DIGEST_AGGREGATION_CLUSTER_SIMILARITY:
            clusters[best_index].append(candidate)
        else:
            clusters.append([candidate])
    return [cluster for cluster in clusters if len(cluster) >= SOURCE_DIGEST_AGGREGATION_MIN_CANDIDATES]


def candidate_cluster_similarity(
    group_name: str,
    candidate: SourceDigestCandidate,
    cluster: list[SourceDigestCandidate],
) -> float:
    if not cluster:
        return 0.0
    return min(source_digest_candidate_topic_similarity(group_name, candidate, item) for item in cluster)


def source_digest_candidate_topic_similarity(
    group_name: str,
    left: SourceDigestCandidate,
    right: SourceDigestCandidate,
) -> float:
    if group_name == "open_questions":
        left_key = _open_questions.open_question_key(left.open_question_or_tension or left.suggested_page_title or left.name)
        right_key = _open_questions.open_question_key(right.open_question_or_tension or right.suggested_page_title or right.name)
        if left_key and right_key and left_key == right_key:
            return 1.0
    left_anchors = source_digest_candidate_topic_anchor_terms(left)
    right_anchors = source_digest_candidate_topic_anchor_terms(right)
    shared_anchors = left_anchors & right_anchors
    if not shared_anchors:
        return 0.0
    left_terms = source_digest_non_generic_terms(source_digest_candidate_topic_terms(left))
    right_terms = source_digest_non_generic_terms(source_digest_candidate_topic_terms(right))
    if not left_terms or not right_terms:
        return 0.0
    anchor_score = _text_similarity.jaccard(left_anchors, right_anchors)
    intent_score = _text_similarity.jaccard(left_terms, right_terms)
    if len(shared_anchors) == 1:
        anchor = next(iter(shared_anchors))
        if source_digest_topic_anchor_too_broad(anchor):
            return 0.0
        shared_bonus = 0.22
    else:
        shared_bonus = min(0.50, len(shared_anchors) * 0.20)
    return min(1.0, max(anchor_score, intent_score * 0.5) + shared_bonus)


def source_digest_candidate_topic_terms(candidate: SourceDigestCandidate) -> set[str]:
    return source_digest_similarity_terms(source_digest_candidate_intent_text(candidate))


def source_digest_candidate_topic_anchor_terms(candidate: SourceDigestCandidate) -> set[str]:
    anchor_text = "\n".join(
        [
            candidate.suggested_page_title,
            candidate.name,
            candidate.open_question_or_tension,
        ]
    )
    return source_digest_non_generic_terms(source_digest_similarity_terms(anchor_text))


def source_digest_topic_anchor_too_broad(anchor: str) -> bool:
    broad = {
        "agi",
        "ai",
        "llm",
        "pm",
        "模型",
        "能力",
        "评估",
        "数据",
        "用户",
    }
    return anchor in broad


def source_digest_non_generic_terms(terms: set[str]) -> set[str]:
    generic = {
        "ai",
        "pm",
        "产品",
        "管理",
        "主题",
        "材料",
        "知识",
        "页面",
        "候选",
        "价值",
        "来源",
        "概念",
        "重要",
        "复用",
        "讨论",
        "问题",
        "方式",
        "变化",
        "边界",
        "职责",
        "执行",
        "判断",
        "功能",
        "任务",
        "组织",
    }
    return {term for term in terms if term not in generic and len(term) >= 2}


def deferred_cluster_terms(cluster: list[SourceDigestCandidate]) -> set[str]:
    if not cluster:
        return set()
    shared = source_digest_candidate_topic_terms(cluster[0])
    for candidate in cluster[1:]:
        shared &= source_digest_candidate_topic_terms(candidate)
    if shared:
        return source_digest_non_generic_terms(shared)
    combined: set[str] = set()
    for candidate in cluster:
        combined.update(source_digest_non_generic_terms(source_digest_candidate_topic_terms(candidate)))
    return combined


def select_aggregation_replacement(
    selected: list[SourceDigestCandidate],
    cluster: list[SourceDigestCandidate],
) -> tuple[int, float]:
    if not selected:
        return 0, 0.0
    best_index = len(selected) - 1
    best_score = -1.0
    for index, candidate in enumerate(selected):
        score = max(
            source_digest_candidate_topic_similarity(deferred_candidate_group(candidate), candidate, cluster_candidate)
            for cluster_candidate in cluster
        )
        if score > best_score:
            best_index = index
            best_score = score
    return best_index, max(0.0, best_score)


def deferred_aggregation_digest_candidate(
    group_name: str,
    candidates: list[SourceDigestCandidate],
    *,
    replaced_candidate_id: str,
) -> SourceDigestCandidate:
    aggregation = build_deferred_candidate_aggregations({group_name: candidates})[0]
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    related_candidate_ids = [candidate_id for candidate_id in candidate_ids if candidate_id != replaced_candidate_id]
    candidate_id = f"AGG-{group_name.replace('_', '-')}-{sha256_bytes('|'.join(candidate_ids).encode('utf-8'))[:8]}"
    source_locators = _markdown_utils.dedupe_strings([candidate.source_locator for candidate in candidates if candidate.source_locator])[:6]
    tensions = _markdown_utils.dedupe_strings([candidate.open_question_or_tension for candidate in candidates if candidate.open_question_or_tension])[:4]
    title = str(aggregation["suggested_title"])
    summary = str(aggregation["coverage_summary"])
    wiki_value = str(aggregation.get("wiki_value_summary") or "") or str(aggregation["suggested_action"])
    return SourceDigestCandidate(
        candidate_id=candidate_id,
        name=title,
        type=deferred_aggregation_digest_type(group_name),
        one_sentence_summary=summary,
        why_matters=f"该聚合候选把 {len(candidates)} 个同组候选压缩成一个页面预算槽，避免单篇材料产生过多独立页面。",
        wiki_value=wiki_value,
        source_locator="；".join(source_locators),
        suggested_page_title=title,
        related_candidates=related_candidate_ids,
        resolution_hint=(
            f"source_digest_deferred_aggregation: represented_candidates={', '.join(candidate_ids)}; "
            f"related_deferred_candidates={', '.join(related_candidate_ids) or 'none'}; "
            f"replaced_selected_candidate={replaced_candidate_id}; page budget remains constant."
        ),
        duplicate_risk=source_digest_max_duplicate_risk(candidates),
        open_question_or_tension="；".join(tensions),
    )


def deferred_aggregation_digest_type(group_name: str) -> str:
    if group_name == "comparisons":
        return "comparison"
    if group_name == "open_questions":
        return "open_question"
    return "overview"


def source_digest_max_duplicate_risk(candidates: list[SourceDigestCandidate]) -> Literal["low", "medium", "high"]:
    risks = {candidate.duplicate_risk for candidate in candidates}
    if "high" in risks:
        return "high"
    if "medium" in risks:
        return "medium"
    return "low"


def deferred_digest_candidate(group_name: str, candidate: SourceDigestCandidate, *, represented_by: str = "") -> SourceDigestCandidate:
    data = candidate.model_dump(mode="json")
    note = f"page_budget_deferred: `{group_name}` 超出本次 max_ingest_candidates，保留在 source digest 审计中，后续可单独 ingest 或手动提升。"
    if represented_by:
        note += f" represented_by_aggregation: `{represented_by}` 已在本轮用聚合候选代表该候选的核心价值。"
    data["resolution_hint"] = _markdown_utils.merge_markdown_blocks(
        str(data.get("resolution_hint") or ""),
        note,
    )
    return SourceDigestCandidate.model_validate(data)


def source_digest_budget_report(
    selected: dict[str, list[SourceDigestCandidate]],
    deferred: dict[str, list[SourceDigestCandidate]],
    *,
    budget: int,
    total: int,
    applied: bool,
    total_before_dedupe: int | None = None,
    deduped_candidates: list[dict[str, Any]] | None = None,
    selected_deferred_aggregations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    selected_count = sum(len(items) for items in selected.values())
    deferred_count = sum(len(items) for items in deferred.values())
    deduped_candidates = deduped_candidates or []
    selected_deferred_aggregations = selected_deferred_aggregations or []
    deferred_details = {
        group_name: [source_digest_candidate_budget_detail(group_name, candidate) for candidate in candidates]
        for group_name, candidates in deferred.items()
    }
    followup_batches = [
        {
            "group": group_name,
            "count": len(candidates),
            "candidate_ids": [candidate.candidate_id for candidate in candidates],
            "suggested_action": (
                f"后续如需扩展 `{group_name}`，可单独从这些 deferred candidate 建页，"
                "或把同组候选合并进一个 overview/comparison 页面。"
            ),
        }
        for group_name, candidates in deferred.items()
        if candidates
    ]
    deferred_aggregations = build_deferred_candidate_aggregations(deferred)
    return {
        "schema_version": "source_digest_budget_report.v1",
        "budget": budget,
        "total_formal_candidates_before_dedupe": total_before_dedupe if total_before_dedupe is not None else total,
        "total_formal_candidates_before_budget": total,
        "deduped_count": len(deduped_candidates),
        "dedupe_applied": bool(deduped_candidates),
        "deduped_candidates": deduped_candidates,
        "selected_deferred_aggregations": selected_deferred_aggregations,
        "selected_count": selected_count,
        "deferred_count": deferred_count,
        "applied": applied,
        "group_order": list(SOURCE_DIGEST_BUDGET_GROUP_ORDER),
        "selected": {
            group_name: [candidate.candidate_id for candidate in candidates]
            for group_name, candidates in selected.items()
        },
        "deferred": {
            group_name: [candidate.candidate_id for candidate in candidates]
            for group_name, candidates in deferred.items()
        },
        "deferred_details": deferred_details,
        "followup_batches": followup_batches,
        "deferred_aggregations": deferred_aggregations,
    }


def build_deferred_candidate_aggregations(
    deferred: dict[str, list[SourceDigestCandidate]] | list[SourceDigestCandidate],
) -> list[dict[str, Any]]:
    if isinstance(deferred, list):
        grouped: dict[str, list[SourceDigestCandidate]] = {}
        for candidate in deferred:
            grouped.setdefault(deferred_candidate_group(candidate), []).append(candidate)
    else:
        grouped = {group_name: list(candidates) for group_name, candidates in deferred.items()}
    aggregations: list[dict[str, Any]] = []
    for group_name in SOURCE_DIGEST_BUDGET_GROUP_ORDER:
        candidates = [candidate for candidate in grouped.get(group_name, []) if isinstance(candidate, SourceDigestCandidate)]
        if not candidates:
            continue
        label = DEFERRED_AGGREGATION_GROUP_LABELS.get(group_name, group_name)
        names = [candidate.suggested_page_title or candidate.name for candidate in candidates]
        representative = candidates[:5]
        source_locators = _markdown_utils.dedupe_strings([candidate.source_locator for candidate in candidates if candidate.source_locator])[:6]
        tensions = _markdown_utils.dedupe_strings([candidate.open_question_or_tension for candidate in candidates if candidate.open_question_or_tension])[:4]
        wiki_values = _markdown_utils.dedupe_strings([candidate.wiki_value for candidate in candidates if candidate.wiki_value])[:4]
        aggregations.append(
            {
                "group": group_name,
                "label": label,
                "count": len(candidates),
                "suggested_page_type": deferred_aggregation_page_type(group_name),
                "suggested_title": deferred_aggregation_title(group_name, names),
                "candidate_ids": [candidate.candidate_id for candidate in candidates],
                "representative_candidates": [
                    {
                        "candidate_id": candidate.candidate_id,
                        "title": candidate.suggested_page_title or candidate.name,
                        "summary": candidate.one_sentence_summary,
                    }
                    for candidate in representative
                ],
                "coverage_summary": deferred_aggregation_summary(label, candidates),
                "wiki_value_summary": "；".join(wiki_values) if wiki_values else "",
                "source_locators": source_locators,
                "open_questions_or_tensions": tensions,
                "suggested_action": deferred_aggregation_action(group_name),
            }
        )
    return aggregations


def deferred_candidate_group(candidate: SourceDigestCandidate) -> str:
    type_key = candidate.type.strip().lower()
    mapping = {
        "concept": "concepts",
        "concepts": "concepts",
        "design": "designs",
        "designs": "designs",
        "comparison": "comparisons",
        "comparisons": "comparisons",
        "open_question": "open_questions",
        "open_questions": "open_questions",
        "question": "open_questions",
        "entity": "entities",
        "entities": "entities",
    }
    if type_key in mapping:
        return mapping[type_key]
    for group_name in SOURCE_DIGEST_BUDGET_GROUP_ORDER:
        if f"`{group_name}`" in candidate.resolution_hint or group_name in candidate.resolution_hint:
            return group_name
    return "concepts"


def deferred_aggregation_page_type(group_name: str) -> str:
    return {
        "comparisons": "comparison",
        "open_questions": "open_question_overview",
        "entities": "entity_index",
        "designs": "design_overview",
    }.get(group_name, "concept_overview")


def deferred_aggregation_title(group_name: str, names: list[str]) -> str:
    label = DEFERRED_AGGREGATION_GROUP_LABELS.get(group_name, group_name)
    if not names:
        return f"{label}聚合页"
    if len(names) == 1:
        return f"{names[0]} 后续页"
    return f"{names[0]} 等 {len(names)} 个{label}聚合页"


def deferred_aggregation_summary(label: str, candidates: list[SourceDigestCandidate]) -> str:
    names = [candidate.suggested_page_title or candidate.name for candidate in candidates[:4]]
    suffix = "" if len(candidates) <= 4 else f" 等 {len(candidates)} 项"
    return f"本批次聚合 {label}：{', '.join(names)}{suffix}。"


def deferred_aggregation_action(group_name: str) -> str:
    if group_name == "comparisons":
        return "后续可合并为一个 comparison 页面，集中比较边界、差异和适用场景。"
    if group_name == "open_questions":
        return "后续可合并为一个 open question overview，集中跟踪问题变体和待补来源。"
    if group_name == "entities":
        return "后续可合并为一个 entity index/source companion，避免为低频实体逐个建页。"
    if group_name == "designs":
        return "后续可合并为一个 design overview，保留模式差异和复用场景。"
    return "后续可合并为一个 concept overview，先保留概念簇关系，再决定是否拆独立页。"


def source_digest_candidate_budget_detail(group_name: str, candidate: SourceDigestCandidate) -> dict[str, str]:
    return {
        "group": group_name,
        "candidate_id": candidate.candidate_id,
        "type": candidate.type,
        "name": candidate.name,
        "suggested_page_title": candidate.suggested_page_title,
        "one_sentence_summary": candidate.one_sentence_summary,
        "why_matters": candidate.why_matters,
        "wiki_value": candidate.wiki_value,
        "source_locator": candidate.source_locator,
        "open_question_or_tension": candidate.open_question_or_tension,
        "resolution_hint": candidate.resolution_hint,
    }


def render_source_digest_budget_report(report: dict[str, Any]) -> str:
    rows = []
    selected = report.get("selected", {})
    deferred = report.get("deferred", {})
    for group_name in ["concepts", "designs", "comparisons", "open_questions", "entities"]:
        rows.append(
            [
                group_name,
                ", ".join(selected.get(group_name, [])) or "无",
                ", ".join(deferred.get(group_name, [])) or "无",
            ]
        )
    dedupe_rows = []
    deduped_candidates = report.get("deduped_candidates", [])
    if isinstance(deduped_candidates, list):
        for item in deduped_candidates:
            if not isinstance(item, dict):
                continue
            dedupe_rows.append(
                [
                    str(item.get("group", "")),
                    str(item.get("dedupe_key", "")),
                    str(item.get("kept_candidate_id", "")),
                    str(item.get("merged_candidate_id", "")),
                    str(item.get("merged_title", "")),
                    str(item.get("source_locator", "")),
                ]
            )
    detail_rows = []
    deferred_details = report.get("deferred_details", {})
    if isinstance(deferred_details, dict):
        for group_name in ["concepts", "designs", "comparisons", "open_questions", "entities"]:
            details = deferred_details.get(group_name, [])
            if not isinstance(details, list):
                continue
            for detail in details:
                if not isinstance(detail, dict):
                    continue
                detail_rows.append(
                    [
                        group_name,
                        str(detail.get("candidate_id", "")),
                        str(detail.get("suggested_page_title") or detail.get("name") or ""),
                        str(detail.get("one_sentence_summary", "")),
                        str(detail.get("wiki_value", "")),
                        str(detail.get("source_locator", "")),
                        str(detail.get("resolution_hint", "")),
                    ]
                )
    selected_aggregation_rows = []
    selected_aggregations = report.get("selected_deferred_aggregations", [])
    if isinstance(selected_aggregations, list):
        for aggregation in selected_aggregations:
            if not isinstance(aggregation, dict):
                continue
            selected_aggregation_rows.append(
                [
                    str(aggregation.get("group", "")),
                    str(aggregation.get("candidate_id", "")),
                    str(aggregation.get("suggested_page_type", "")),
                    str(aggregation.get("suggested_page_title", "")),
                    str(aggregation.get("replaced_candidate_id", "")),
                    ", ".join(aggregation.get("represented_candidate_ids", []))
                    if isinstance(aggregation.get("represented_candidate_ids"), list)
                    else "",
                    str(aggregation.get("reason", "")),
                ]
            )
    batch_rows = []
    followup_batches = report.get("followup_batches", [])
    if isinstance(followup_batches, list):
        for batch in followup_batches:
            if not isinstance(batch, dict):
                continue
            batch_rows.append(
                [
                    str(batch.get("group", "")),
                    str(batch.get("count", "")),
                    ", ".join(batch.get("candidate_ids", [])) if isinstance(batch.get("candidate_ids"), list) else "",
                    str(batch.get("suggested_action", "")),
                ]
            )
    aggregation_rows = []
    aggregations = report.get("deferred_aggregations", [])
    if isinstance(aggregations, list):
        for aggregation in aggregations:
            if not isinstance(aggregation, dict):
                continue
            representatives = aggregation.get("representative_candidates", [])
            if isinstance(representatives, list):
                rep_text = ", ".join(
                    str(item.get("title", item.get("candidate_id", "")))
                    for item in representatives[:4]
                    if isinstance(item, dict)
                )
            else:
                rep_text = ""
            aggregation_rows.append(
                [
                    str(aggregation.get("group", "")),
                    str(aggregation.get("suggested_page_type", "")),
                    str(aggregation.get("suggested_title", "")),
                    str(aggregation.get("coverage_summary", "")),
                    rep_text,
                    str(aggregation.get("suggested_action", "")),
                ]
            )
    return (
        "# Source Digest 页面预算报告\n\n"
        f"- 预算：{report.get('budget', 0)}\n"
        f"- 去重前正式候选：{report.get('total_formal_candidates_before_dedupe', report.get('total_formal_candidates_before_budget', 0))}\n"
        f"- 预算前正式候选：{report.get('total_formal_candidates_before_budget', 0)}\n"
        f"- 语义去重合并：{report.get('deduped_count', 0)}\n"
        f"- 进入页面规划：{report.get('selected_count', 0)}\n"
        f"- 延后：{report.get('deferred_count', 0)}\n"
        f"- 是否应用预算：{'是' if report.get('applied') else '否'}\n\n"
        + format_markdown_table(["分组", "进入页面规划", "延后"], rows)
        + "\n\n"
        "## 语义去重候选\n\n"
        + (
            format_markdown_table(["分组", "去重 Key", "保留 ID", "合并 ID", "合并标题", "来源定位"], dedupe_rows)
            if dedupe_rows
            else "_暂无语义去重候选。_"
        )
        + "\n\n"
        "## 延后候选详情\n\n"
        + (
            format_markdown_table(["分组", "ID", "建议标题", "摘要", "Wiki 价值", "来源定位", "处理提示"], detail_rows)
            if detail_rows
            else "_暂无延后候选。_"
        )
        + "\n\n"
        "## 本轮已选聚合候选\n\n"
        + (
            format_markdown_table(["分组", "聚合 ID", "页类型", "标题", "替换候选", "代表候选", "原因"], selected_aggregation_rows)
            if selected_aggregation_rows
            else "_暂无本轮已选聚合候选。_"
        )
        + "\n\n"
        "## 后续处理批次\n\n"
        + (
            format_markdown_table(["分组", "数量", "候选 ID", "建议"], batch_rows)
            if batch_rows
            else "_暂无后续批次。_"
        )
        + "\n\n"
        "## 延后聚合建议\n\n"
        + (
            format_markdown_table(["分组", "建议页类型", "建议标题", "覆盖摘要", "代表候选", "动作"], aggregation_rows)
            if aggregation_rows
            else "_暂无延后聚合建议。_"
        )
        + "\n"
    )
