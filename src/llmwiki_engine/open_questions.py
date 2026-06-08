from __future__ import annotations

import re
import unicodedata
from typing import Any

from .markdown_utils import dedupe_strings
from .models import DraftRenderingArtifact, WikiContextSnapshot, WikiMergePlanArtifact
from .system_pages import format_markdown_table
from .wiki_markup import clean_display_title, obsidian_link


__all__ = (
    "build_open_question_rows_with_report",
    "extract_open_questions",
    "group_open_question_candidates",
    "is_low_signal_open_question",
    "meaningful_open_question_lines",
    "open_question_key",
    "open_question_representative_sort_key",
    "render_index_open_questions_report",
)


_OPEN_QUESTION_SEMANTIC_CLUSTERS = (
    (
        "semantic:agi_pm_role_necessity",
        (("agi",), ("pm", "产品经理"), ("消失", "必要", "取代", "替代", "还有价值", "是否需要", "需要")),
    ),
    (
        "semantic:model_capability_product_function_boundary",
        (("模型能力",), ("产品功能", "产品边界"), ("吞噬", "吞掉", "取代", "替代", "边界")),
    ),
    (
        "semantic:product_judgement_training",
        (("产品品味", "产品判断"), ("训练", "提升", "培养", "系统化")),
    ),
    (
        "semantic:ai_pm_role_evolution",
        (("ai", "agi", "agent"), ("pm", "产品经理"), ("演变", "变化", "转型", "未来")),
    ),
    (
        "semantic:claude_code_product_experience_harness_boundary",
        (("claudecode", "claude code"), ("产品体验", "体验"), ("harness", "安全边界"), ("掩盖", "重要性")),
    ),
    (
        "semantic:rapid_iteration_quality_safety_research_preview",
        (("快速发布", "快速迭代"), ("质量", "安全"), ("研究预览", "用户预期", "长期产品一致性")),
    ),
    (
        "semantic:model_progress_feature_lifecycle",
        (("模型", "模型能力", "模型进步"), ("功能", "产品功能", "ui元素"), ("保留", "移除", "废弃", "过时", "存废", "淘汰")),
    ),
    (
        "semantic:agent_hand_transfer_mechanism",
        (("大脑", "brain"), ("传递", "pass", "handoff"), ("双手", "hand", "hands"), ("机制", "实现细节", "高效")),
    ),
)


def open_question_key(question: str) -> str:
    text = _strip_open_question_marker(question)
    text = re.sub(r"^(待补来源|待补充来源|需要来源|缺少来源)\s*[:：]\s*", "", text)
    text = unicodedata.normalize("NFKC", text).lower()
    normalized = re.sub(r"[\s，。；;：:、,.!?！？（）()【】\[\]\"'“”‘’]+", "", text)
    semantic_key = _semantic_open_question_key(normalized)
    return semantic_key or normalized


def _strip_open_question_marker(question: str) -> str:
    text = re.sub(r"^\s*[-*]\s+", "", question.strip())
    return re.sub(r"^\s*\d+\s*[.)、．]\s*", "", text).strip()


def group_open_question_candidates(candidates: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for candidate in candidates:
        key = open_question_key(candidate["question"])
        merge_key = next(
            (
                existing_key
                for existing_key, grouped in groups.items()
                if _open_question_keys_should_merge(key, existing_key, candidate, grouped)
            ),
            None,
        )
        groups.setdefault(merge_key or key, []).append(candidate)
    return groups


def _open_question_keys_should_merge(
    key: str,
    existing_key: str,
    candidate: dict[str, str],
    grouped: list[dict[str, str]],
) -> bool:
    if key == existing_key:
        return True
    if existing_key.startswith("semantic:") and key.startswith("semantic:"):
        candidate_norm = _open_question_similarity_text(candidate["question"])
        return any(
            candidate.get("path") == existing.get("path")
            and _open_question_token_overlap(candidate_norm, _open_question_similarity_text(existing["question"])) >= 0.60
            for existing in grouped
        )
    if _open_question_key_contains_other(key, existing_key):
        return True
    candidate_norm = _open_question_similarity_text(candidate["question"])
    if not candidate_norm:
        return False
    for existing in grouped:
        existing_norm = _open_question_similarity_text(existing["question"])
        if not existing_norm:
            continue
        same_path = candidate.get("path") == existing.get("path")
        if _open_question_key_contains_other(candidate_norm, existing_norm):
            return True
        if same_path and _open_question_token_overlap(candidate_norm, existing_norm) >= 0.62:
            return True
    return False


def _open_question_key_contains_other(left: str, right: str) -> bool:
    if len(left) < 12 or len(right) < 12:
        return False
    return left in right or right in left


def _open_question_similarity_text(question: str) -> str:
    text = _strip_open_question_marker(question)
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[\s，。；;：:、,.!?！？（）()【】\[\]\"'“”‘’]+", "", text)


def _open_question_token_overlap(left: str, right: str) -> float:
    left_tokens = _open_question_similarity_tokens(left)
    right_tokens = _open_question_similarity_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = left_tokens & right_tokens
    return len(intersection) / min(len(left_tokens), len(right_tokens))


def _open_question_similarity_tokens(normalized: str) -> set[str]:
    text = normalized
    for stop in ["如何", "是否", "能否", "会不会", "为什么", "什么", "哪些", "是否可能", "可能", "应该", "需要"]:
        text = text.replace(stop, "")
    tokens = set(re.findall(r"[a-z][a-z0-9_/-]{1,}", text))
    cjk = "".join(char for char in text if "\u4e00" <= char <= "\u9fff")
    for size in (4, 3):
        for index in range(0, max(0, len(cjk) - size + 1)):
            token = cjk[index : index + size]
            if _open_question_similarity_token_is_noise(token):
                continue
            tokens.add(token)
    return tokens


def _open_question_similarity_token_is_noise(token: str) -> bool:
    if all(char in "的了和与及或是否如何什么为什么可能需要应该能否会不会" for char in token):
        return True
    return token in {"产品", "功能", "用户", "团队", "问题", "未来", "影响", "风险"}


def open_question_representative_sort_key(item: dict[str, str]) -> tuple[int, str, tuple[int, int, int]]:
    question = item["question"]
    non_low_signal = 0 if is_low_signal_open_question(question) else 1
    return (non_low_signal, item["updated"], _open_question_representative_score(question))


def _open_question_representative_score(question: str) -> tuple[int, int, int]:
    stripped = _strip_open_question_marker(question)
    single_question = 1 if stripped.count("？") + stripped.count("?") <= 1 else 0
    has_source_gap = 1 if is_low_signal_open_question(stripped) else 0
    return (single_question, -has_source_gap, -len(stripped))


def _semantic_open_question_key(normalized: str) -> str:
    for key, required_groups in _OPEN_QUESTION_SEMANTIC_CLUSTERS:
        if all(any(term in normalized for term in group) for group in required_groups):
            return key
    return ""


def is_low_signal_open_question(question: str) -> bool:
    normalized = open_question_key(question)
    if len(normalized) < 10:
        return True
    low_signal_markers = ["待补来源", "待补充来源", "需要来源", "缺少来源", "source needed", "citation needed"]
    if any(marker in question.lower() for marker in low_signal_markers):
        return True
    source_gap_markers = ["具体引用", "具体来源", "出处", "引用链接", "原始证据"]
    return any(marker in question for marker in source_gap_markers)


def extract_open_questions(markdown: str) -> list[str]:
    match = re.search(r"(?ms)^##\s+矛盾与未决问题\s*$\n(?P<body>.*?)(?=^##\s+|\Z)", markdown)
    if not match:
        return []
    return meaningful_open_question_lines(match.group("body"))


def meaningful_open_question_lines(text: str) -> list[str]:
    results: list[str] = []
    for raw_line in text.splitlines():
        line = _strip_open_question_marker(raw_line)
        line = line.strip("。；; ")
        if not line:
            continue
        normalized = re.sub(r"\s+", "", line)
        if any(marker in normalized for marker in ["暂无", "没有", "无未决", "无矛盾", "不适用", "N/A", "na"]):
            continue
        if len(normalized) < 4:
            continue
        results.append(line)
    return dedupe_strings(results)


def build_open_question_rows_with_report(
    plan: WikiMergePlanArtifact,
    draft: DraftRenderingArtifact,
    snapshot: WikiContextSnapshot,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    candidates: list[dict[str, str]] = []
    for entry in snapshot.entries:
        metadata = entry.metadata
        if entry.expected_state != "present" or metadata is None or metadata.llmwiki_type.lower() == "source":
            continue
        for question in extract_open_questions(entry.content):
            candidates.append({
                "question": question,
                "page": obsidian_link(metadata.path, clean_display_title(metadata.title)),
                "path": metadata.path,
                "updated": metadata.updated,
                "page_type": metadata.llmwiki_type,
                "source": "existing_wiki",
            })
    plan_by_id = {item.page_plan_id: item for item in plan.items}
    for page in draft.pages:
        item = plan_by_id.get(page.page_plan_id)
        if item is None:
            continue
        for question in meaningful_open_question_lines(page.open_questions.strip()):
            candidates.append({
                "question": question,
                "page": obsidian_link(item.canonical_target_path, item.display_title),
                "path": item.canonical_target_path,
                "updated": plan.log_date,
                "page_type": item.page_type,
                "source": "draft",
            })
    by_key = group_open_question_candidates(candidates)
    rows: list[dict[str, str]] = []
    report_items: list[dict[str, Any]] = []
    for key, grouped in sorted(by_key.items()):
        representative = max(grouped, key=open_question_representative_sort_key)
        low_signal = is_low_signal_open_question(representative["question"])
        repeated_gap = len(grouped) >= 2 and low_signal
        keep = (
            any(item["page_type"] == "open_question" for item in grouped)
            or repeated_gap
            or not low_signal
        )
        pages = dedupe_strings([item["page"] for item in sorted(grouped, key=lambda item: item["updated"], reverse=True)])[:3]
        decision = "kept" if keep else "filtered"
        reason = "open_question_page" if any(item["page_type"] == "open_question" for item in grouped) else ""
        if not reason:
            reason = "repeated_source_gap" if repeated_gap else ("low_signal_or_source_gap" if low_signal else "high_signal")
        report_items.append(
            {
                "normalized_key": key,
                "question": representative["question"],
                "decision": decision,
                "reason": reason,
                "pages": pages,
                "occurrences": len(grouped),
            }
        )
        if not keep:
            continue
        rows.append(
            {
                "question": representative["question"],
                "page": ", ".join(pages),
                "updated": max(item["updated"] for item in grouped),
            }
        )
    rows.sort(key=lambda row: (row["updated"], row["page"], row["question"]), reverse=True)
    return rows, {
        "schema_version": "index_open_questions_report.v1",
        "kept_count": sum(1 for item in report_items if item["decision"] == "kept"),
        "filtered_count": sum(1 for item in report_items if item["decision"] == "filtered"),
        "deduped_count": sum(max(0, item["occurrences"] - 1) for item in report_items if item["decision"] == "kept"),
        "items": report_items,
    }


def render_index_open_questions_report(report: dict[str, Any]) -> str:
    rows = [
        [
            item["decision"],
            item["reason"],
            item["question"],
            ", ".join(item["pages"]),
            str(item["occurrences"]),
        ]
        for item in report.get("items", [])
    ]
    return (
        "# Index 未决问题筛选报告\n\n"
        f"- 保留：{report.get('kept_count', 0)}\n"
        f"- 过滤：{report.get('filtered_count', 0)}\n\n"
        f"- 合并重复：{report.get('deduped_count', 0)}\n\n"
        + (format_markdown_table(["决策", "原因", "问题", "关联页面", "次数"], rows) if rows else "暂无未决问题。")
        + "\n"
    )
