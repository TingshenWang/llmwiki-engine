from __future__ import annotations

import re
import unicodedata
from typing import Any, Literal

from . import source_excerpt as _source_excerpt
from .markdown_utils import compact_payload_text, is_empty_placeholder, merge_markdown_blocks
from .models import DraftPageItem, DraftRenderingArtifact, StructuredIssue, WikiContextSnapshot, WikiMergePlanArtifact
from .page_sections import parse_existing_sections
from .system_pages import format_markdown_table


UPDATE_PRESERVATION_SECTION_KEYS = (
    "summary",
    "core_content",
    "detail",
    "examples",
    "value_points",
    "additional_notes",
)
UPDATE_PRESERVATION_MAX_PHRASES_PER_SECTION = 8
UPDATE_PRESERVATION_CONCEPT_GROUPS = (
    {
        "name": "managed_agents",
        "label": "Managed Agents / 托管智能体",
        "terms": ("Managed Agents", "托管智能体"),
    },
    {
        "name": "harness",
        "label": "harness / 适配框架",
        "terms": ("harness", "Harness", "harnesses", "适配框架", "元harness", "元适配框架"),
    },
    {
        "name": "brain_hands_decoupling",
        "label": "大脑与双手解耦",
        "terms": (
            "大脑与双手",
            "大脑双手",
            "brain and hands",
            "brain hands",
            "brain & hands",
            "separate the model brain from execution hands",
            "model brain from execution hands",
            "model brain and execution hands",
            "model brain / execution hands",
            "推理和规划",
            "工具执行",
            "解耦",
            "路由模型意图",
            "模型意图路由",
        ),
    },
    {
        "name": "session_context",
        "label": "会话/持久上下文",
        "terms": (
            "session object",
            "persistent session",
            "persistent context",
            "durable context",
            "session state",
            "session context",
            "会话对象",
            "持久上下文",
            "上下文对象",
            "会话是持久",
            "执行状态",
        ),
    },
    {
        "name": "safety_boundary",
        "label": "安全边界/权限限制",
        "terms": ("安全边界", "权限", "限制文件", "文件、网络和资源", "文件网络和资源", "无限本机权限"),
    },
    {
        "name": "isolated_execution",
        "label": "隔离执行/容器",
        "terms": (
            "隔离容器",
            "隔离执行环境",
            "container",
            "containers",
            "containerized",
            "sandbox",
            "sandboxed",
            "sandboxes",
            "容器",
            "沙箱",
        ),
    },
    {
        "name": "system_architecture_view",
        "label": "系统架构视角",
        "terms": ("系统架构视角", "产品功能列表", "只写成产品功能列表"),
    },
)


def build_update_preservation_pack(merge_plan: WikiMergePlanArtifact, snapshot: WikiContextSnapshot) -> dict[str, Any]:
    pages: list[dict[str, Any]] = []
    for item in merge_plan.items:
        if item.action != "update":
            continue
        entry = _snapshot_entry(snapshot, f"wiki/{item.canonical_target_path}")
        if entry is None:
            continue
        sections = parse_existing_sections(entry.content)
        section_items: list[dict[str, Any]] = []
        for section_key in UPDATE_PRESERVATION_SECTION_KEYS:
            old_text = sections.get(section_key, "").strip()
            if not old_text:
                continue
            reusable_old_text = update_preservation_non_placeholder_text(old_text)
            phrases = update_preservation_phrases(reusable_old_text)
            concepts = update_preservation_concepts(reusable_old_text)
            if update_preservation_section_is_low_value(section_key, old_text, phrases, concepts):
                continue
            section_items.append(
                {
                    "section_key": section_key,
                    "old_text": compact_payload_text(reusable_old_text, 1800),
                    "old_char_count": len(old_text),
                    "key_phrases": phrases,
                    "min_required_matches": 0 if concepts else update_preservation_required_matches(phrases),
                    "concept_obligations": concepts,
                    "min_required_concept_matches": update_preservation_required_concept_matches(concepts),
                }
            )
        section_items = collapse_update_preservation_sections(section_items)
        if section_items:
            pages.append(
                {
                    "page_plan_id": item.page_plan_id,
                    "target_path": item.canonical_target_path,
                    "display_title": item.display_title,
                    "matched_page": item.matched_page,
                    "sections": section_items,
                }
            )
    return {
        "schema_version": "update_preservation_pack.v1",
        "goal": "For update pages, carry forward old reusable knowledge into the replacement draft or explicitly explain why it changed.",
        "pages": pages,
    }


def _snapshot_entry(snapshot: WikiContextSnapshot, path: str) -> Any | None:
    for entry in snapshot.entries:
        if entry.path == path:
            return entry
    return None


def collapse_update_preservation_sections(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    detail = next((section for section in sections if section.get("section_key") == "detail"), None)
    if detail is None:
        return sections
    detail_concepts = update_preservation_section_concept_names(detail)
    if not detail_concepts:
        return sections
    collapsed: list[dict[str, Any]] = []
    for section in sections:
        section_key = str(section.get("section_key", ""))
        section_concepts = update_preservation_section_concept_names(section)
        if section_key in {"summary", "value_points"} and section_concepts and section_concepts <= detail_concepts:
            continue
        collapsed.append(section)
    return collapsed


def update_preservation_section_concept_names(section: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for concept in section.get("concept_obligations", []):
        if isinstance(concept, dict):
            name = str(concept.get("name") or "").strip()
            if name:
                names.add(name)
    return names


def update_preservation_section_is_low_value(
    section_key: str,
    old_text: str,
    phrases: list[str],
    concepts: list[dict[str, Any]],
) -> bool:
    normalized = _source_excerpt.normalized_source_match_text(old_text)
    meaningful_phrases = [
        phrase
        for phrase in phrases
        if not update_preservation_phrase_is_placeholder(_source_excerpt.normalized_source_match_text(phrase))
    ]
    if not concepts and not meaningful_phrases:
        return True
    if update_preservation_phrase_is_placeholder(normalized):
        return not update_preservation_has_non_placeholder_signal(old_text)
    if section_key == "value_points" and not concepts and len(meaningful_phrases) <= 1:
        return True
    return False


def update_preservation_phrases(text: str, *, limit: int = UPDATE_PRESERVATION_MAX_PHRASES_PER_SECTION) -> list[str]:
    phrases: list[str] = []
    for piece in re.split(r"[\n。；;，,、|：:]+", text):
        for variant in _source_excerpt.source_excerpt_cue_variants(piece):
            normalized = _source_excerpt.normalized_source_match_text(variant)
            if len(normalized) < 4 or len(normalized) > 80:
                continue
            if update_preservation_phrase_is_noise(normalized):
                continue
            if variant not in phrases:
                phrases.append(variant)
    phrases.sort(key=lambda value: (phrase_signal_score(value), len(_source_excerpt.normalized_source_match_text(value))), reverse=True)
    return phrases[:limit]


def update_preservation_phrase_is_noise(normalized: str) -> bool:
    if update_preservation_phrase_is_placeholder(normalized):
        return True
    if normalized in {"暂无", "没有相关", "暂无相关", "无相关", "n/a", "na"}:
        return True
    if normalized.startswith("旧页保留观察"):
        return True
    return False


def update_preservation_has_non_placeholder_signal(text: str) -> bool:
    reusable_text = update_preservation_non_placeholder_text(text)
    if not reusable_text:
        return False
    if update_preservation_concepts(reusable_text):
        return True
    phrases = update_preservation_phrases(reusable_text)
    return any(not update_preservation_phrase_is_placeholder(_source_excerpt.normalized_source_match_text(phrase)) for phrase in phrases)


def update_preservation_non_placeholder_text(text: str) -> str:
    raw_segments = [
        segment.strip()
        for segment in re.split(r"[\n。；;，,、|：:]+", text)
        if _source_excerpt.normalized_source_match_text(segment)
    ]
    placeholder_flags = [
        update_preservation_phrase_is_placeholder(_source_excerpt.normalized_source_match_text(segment))
        for segment in raw_segments
    ]
    if not any(placeholder_flags):
        return text.strip()
    return "\n".join(segment for segment, is_placeholder in zip(raw_segments, placeholder_flags) if not is_placeholder)


def update_preservation_phrase_is_placeholder(normalized: str) -> bool:
    value = normalized.lower()
    if not value or value in {"n/a", "na"}:
        return True
    return any(
        marker in value
        for marker in [
            "待补来源",
            "来源未提供",
            "暂无",
            "没有相关",
            "无相关",
        ]
    )


def update_preservation_ascii_token_spans(text: str) -> tuple[str, list[tuple[str, int, int]]]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    spans = [(match.group(0), match.start(), match.end()) for match in re.finditer(r"[a-z0-9]+", normalized)]
    return normalized, spans


def update_preservation_term_uses_ascii_tokens(term: str) -> bool:
    return bool(re.search(r"[A-Za-z]", term)) and term.isascii()


def update_preservation_ascii_phrase_separator_allowed(separator: str, term_separator: str) -> bool:
    if "&" in term_separator:
        return separator.count("&") == 1 and all(char.isspace() or char == "&" for char in separator)
    return all(char.isspace() or char in "-_/" for char in separator)


def update_preservation_ascii_phrase_matches(text: str, term: str) -> bool:
    normalized_term, term_spans = update_preservation_ascii_token_spans(term)
    term_tokens = [token for token, _, _ in term_spans]
    if not term_tokens:
        return False
    term_separators = [
        normalized_term[term_spans[offset][2] : term_spans[offset + 1][1]]
        for offset in range(len(term_spans) - 1)
    ]
    normalized_text, text_spans = update_preservation_ascii_token_spans(text)
    text_tokens = [token for token, _, _ in text_spans]
    if len(text_spans) < len(term_tokens):
        return False
    window_size = len(term_tokens)
    for index in range(len(text_spans) - window_size + 1):
        if text_tokens[index : index + window_size] != term_tokens:
            continue
        window_spans = text_spans[index : index + window_size]
        separators = [
            normalized_text[window_spans[offset][2] : window_spans[offset + 1][1]]
            for offset in range(len(window_spans) - 1)
        ]
        if all(
            update_preservation_ascii_phrase_separator_allowed(separator, term_separator)
            for separator, term_separator in zip(separators, term_separators)
        ):
            return True
    return False


def update_preservation_term_matches(text: str, term: str) -> bool:
    if not term.strip():
        return False
    if update_preservation_term_uses_ascii_tokens(term):
        return update_preservation_ascii_phrase_matches(text, term)
    normalized_term = _source_excerpt.normalized_source_match_text(term)
    return bool(normalized_term) and normalized_term in _source_excerpt.normalized_source_match_text(text)


def update_preservation_concepts(text: str) -> list[dict[str, Any]]:
    concepts: list[dict[str, Any]] = []
    for group in UPDATE_PRESERVATION_CONCEPT_GROUPS:
        matched_terms = [
            term
            for term in group["terms"]
            if update_preservation_term_matches(text, str(term))
        ]
        if matched_terms:
            concepts.append(
                {
                    "name": group["name"],
                    "label": group["label"],
                    "matched_terms": matched_terms,
                }
            )
    return concepts


def update_preservation_required_concept_matches(concepts: list[dict[str, Any]]) -> int:
    count = len(concepts)
    if count <= 0:
        return 0
    if count <= 2:
        return count
    if count <= 4:
        return 3
    return max(3, (count * 2 + 2) // 3)


def update_preservation_concept_absorption(old: str, new: str) -> tuple[bool, list[str], list[str], int]:
    concepts = update_preservation_concepts(old)
    if not concepts:
        return True, [], [], 0
    matched: list[str] = []
    missing: list[str] = []
    for concept in concepts:
        group = next((item for item in UPDATE_PRESERVATION_CONCEPT_GROUPS if item["name"] == concept["name"]), None)
        terms = tuple(group["terms"] if group is not None else concept.get("matched_terms", []))
        if any(update_preservation_term_matches(new, str(term)) for term in terms):
            matched.append(str(concept["label"]))
        else:
            missing.append(str(concept["label"]))
    required = update_preservation_required_concept_matches(concepts)
    return len(matched) >= required, matched, missing, required


def phrase_signal_score(phrase: str) -> int:
    normalized = _source_excerpt.normalized_source_match_text(phrase)
    score = min(len(normalized), 40)
    if re.search(r"[A-Za-z]", phrase):
        score += 12
    if any(keyword in phrase for keyword in ["harness", "Managed", "安全边界", "会话对象", "隔离", "权限", "架构", "上下文"]):
        score += 10
    if re.search(r"\d|%|倍|收入|用户|增长|下降|裁撤|预算|金额", phrase):
        score += 6
    return score


def update_preservation_required_matches(phrases: list[str]) -> int:
    if not phrases:
        return 0
    return 1 if len(phrases) <= 2 else 2


def update_section_absorption(old: str, new: str) -> tuple[bool, list[str], list[str]]:
    old = old.strip()
    new = new.strip()
    if not old or is_empty_placeholder(old):
        return True, [], []
    if old == new or old in new:
        phrases = update_preservation_phrases(old)
        return True, phrases[: update_preservation_required_matches(phrases)], phrases
    phrases = update_preservation_phrases(old)
    concept_absorbed, matched_concepts, _, _ = update_preservation_concept_absorption(old, new)
    concepts = update_preservation_concepts(old)
    if not phrases:
        if concepts and concept_absorbed and len(matched_concepts) >= max(2, update_preservation_required_concept_matches(concepts)):
            return True, matched_concepts, phrases
        return False, matched_concepts, phrases
    matched = [phrase for phrase in phrases if _source_excerpt.find_source_cue(new, phrase) >= 0]
    required = update_preservation_required_matches(phrases)
    if concepts and not concept_absorbed:
        return False, [*matched, *matched_concepts], phrases
    return len(matched) >= required or bool(matched_concepts), [*matched, *matched_concepts], phrases


def update_preservation_section_absorption(section: dict[str, Any], new_text: str) -> dict[str, Any]:
    old_text = str(section.get("old_text", ""))
    phrases = [str(phrase) for phrase in section.get("key_phrases", []) if str(phrase).strip()]
    if not phrases:
        phrases = update_preservation_phrases(old_text)
    if old_text.strip() and (old_text.strip() == new_text.strip() or old_text.strip() in new_text):
        matched_phrases = phrases[: update_preservation_required_matches(phrases)]
    else:
        matched_phrases = [phrase for phrase in phrases if _source_excerpt.find_source_cue(new_text, phrase) >= 0]
    required_phrases = int(section.get("min_required_matches") or update_preservation_required_matches(phrases))
    concept_obligations = [
        concept
        for concept in section.get("concept_obligations", [])
        if isinstance(concept, dict)
    ]
    if not concept_obligations:
        concept_obligations = update_preservation_concepts(old_text)
    matched_concepts: list[str] = []
    missing_concepts: list[str] = []
    for concept in concept_obligations:
        name = str(concept.get("name", ""))
        label = str(concept.get("label") or name)
        group = next((item for item in UPDATE_PRESERVATION_CONCEPT_GROUPS if item["name"] == name), None)
        terms = tuple(group["terms"] if group is not None else concept.get("matched_terms", []))
        if any(update_preservation_term_matches(new_text, str(term)) for term in terms):
            matched_concepts.append(label)
        else:
            missing_concepts.append(label)
    required_concepts = int(
        section.get("min_required_concept_matches")
        or update_preservation_required_concept_matches(concept_obligations)
    )
    if required_concepts and matched_concepts:
        phrase_absorbed = True
    else:
        phrase_absorbed = len(matched_phrases) >= required_phrases if required_phrases else True
    concept_absorbed = len(matched_concepts) >= required_concepts if required_concepts else True
    absorbed = phrase_absorbed and concept_absorbed
    return {
        "absorbed": absorbed,
        "matched_phrases": matched_phrases,
        "phrases": phrases,
        "required_phrases": required_phrases,
        "matched_concepts": matched_concepts,
        "missing_concepts": missing_concepts,
        "required_concepts": required_concepts,
        "concept_labels": [str(concept.get("label") or concept.get("name", "")) for concept in concept_obligations],
    }


def update_preservation_issues(draft: DraftRenderingArtifact, pack: dict[str, Any]) -> list[StructuredIssue]:
    pages_by_id = {page.page_plan_id: page for page in draft.pages}
    issues: list[StructuredIssue] = []
    for page_pack in pack.get("pages", []):
        if not isinstance(page_pack, dict):
            continue
        page_plan_id = str(page_pack.get("page_plan_id", ""))
        page = pages_by_id.get(page_plan_id)
        if page is None:
            continue
        for section in page_pack.get("sections", []):
            if not isinstance(section, dict):
                continue
            section_key = str(section.get("section_key", ""))
            new_text = draft_page_text_for_preservation_section(page, section_key)
            absorption = update_preservation_section_absorption(section, new_text)
            if absorption["absorbed"]:
                continue
            field_key = draft_field_for_preservation_section(section_key)
            concept_message = ""
            if absorption["concept_labels"]:
                concept_message = (
                    f" Required old concept obligations: {', '.join(absorption['concept_labels'])}. "
                    f"Need {absorption['required_concepts']}; matched concepts: {', '.join(absorption['matched_concepts']) or 'none'}; "
                    f"missing concepts: {', '.join(absorption['missing_concepts']) or 'none'}."
                )
            issues.append(
                StructuredIssue(
                    issue_code="old_knowledge_not_absorbed",
                    field_path=f"pages.{page_plan_id}.{field_key}",
                    validator_id="update_preservation_pack",
                    message=(
                        f"Update draft for `{page_pack.get('target_path', '')}` does not carry forward old `{section_key}` knowledge. "
                        f"Retain or rewrite at least {absorption['required_phrases']} key phrase(s), such as: {', '.join(absorption['phrases'][:4])}. "
                        f"Matched so far: {', '.join([*absorption['matched_phrases'], *absorption['matched_concepts']]) or 'none'}."
                        f"{concept_message}"
                    ),
                    repairability="repairable",
                )
            )
    return issues


def reinforce_update_preservation(
    draft: DraftRenderingArtifact,
    pack: dict[str, Any],
) -> tuple[DraftRenderingArtifact, dict[str, Any]]:
    pages_by_id = {page.page_plan_id: page for page in draft.pages}
    updated_pages: dict[str, DraftPageItem] = {}
    report_pages: list[dict[str, Any]] = []
    for page_pack in pack.get("pages", []):
        if not isinstance(page_pack, dict):
            continue
        page_plan_id = str(page_pack.get("page_plan_id", ""))
        page = pages_by_id.get(page_plan_id)
        if page is None:
            continue
        section_reports: list[dict[str, Any]] = []
        summary = page.summary
        body_markdown = page.body_markdown
        open_questions = page.open_questions
        for section in page_pack.get("sections", []):
            if not isinstance(section, dict):
                continue
            section_key = str(section.get("section_key", ""))
            old_text = str(section.get("old_text", "")).strip()
            if not section_key or not old_text:
                continue
            target_field = draft_field_for_preservation_section(section_key)
            working_page = page.model_copy(
                update={
                    "summary": summary,
                    "body_markdown": body_markdown,
                    "open_questions": open_questions,
                }
            )
            current = draft_page_text_for_preservation_section(working_page, section_key)
            absorption = update_preservation_section_absorption(section, current)
            if absorption["absorbed"]:
                continue
            missing_concepts = list(absorption["missing_concepts"])
            reinforcement = update_preservation_reinforcement_text(section_key, old_text, missing_concepts)
            merged = merge_markdown_blocks(current, reinforcement)
            if target_field == "summary":
                summary = merged
            elif target_field == "open_questions":
                open_questions = merged
            else:
                body_markdown = merged
            section_reports.append(
                {
                    "section_key": section_key,
                    "target_field": target_field,
                    "matched_before": [*absorption["matched_phrases"], *absorption["matched_concepts"]],
                    "missing_concepts_before": missing_concepts,
                    "required_concept_matches": absorption["required_concepts"],
                    "key_phrases": absorption["phrases"][:4],
                    "reinforcement_char_count": len(reinforcement),
                    "reinforcement_preview": compact_payload_text(reinforcement, 240),
                }
            )
        if section_reports:
            updated = page.model_copy(
                update={
                    "summary": summary,
                    "body_markdown": body_markdown,
                    "open_questions": open_questions,
                }
            )
            updated_pages[page_plan_id] = updated
            report_pages.append(
                {
                    "page_plan_id": page_plan_id,
                    "target_path": page_pack.get("target_path", ""),
                    "display_title": page_pack.get("display_title", ""),
                    "sections": section_reports,
                }
            )
    if updated_pages:
        pages = [updated_pages.get(page.page_plan_id, page) for page in draft.pages]
        draft = draft.model_copy(update={"pages": pages})
    report = {
        "schema_version": "update_preservation_reinforcement_report.v1",
        "changed": bool(report_pages),
        "reinforced_page_count": len(report_pages),
        "reinforced_section_count": sum(len(page["sections"]) for page in report_pages),
        "pages": report_pages,
    }
    return draft, report


def update_preservation_reinforcement_text(section_key: str, old_text: str, missing_concepts: list[str]) -> str:
    old_excerpt = compact_payload_text(old_text, 700)
    bridge = "与旧页架构视角相衔接，"
    if missing_concepts:
        concept_text = "、".join(missing_concepts)
        old_excerpt = (
            f"从旧页保留的架构视角看，本段仍需体现：{concept_text}。"
            "这些是旧页已经建立的理解，应与本轮新材料并列保留。"
        )
        bridge = ""
    if section_key == "value_points":
        return f"- {bridge}{old_excerpt}"
    return f"{bridge}{old_excerpt}"


def render_update_preservation_reinforcement_report(report: dict[str, Any]) -> str:
    rows: list[list[Any]] = []
    for page in report.get("pages", []):
        if not isinstance(page, dict):
            continue
        for section in page.get("sections", []):
            if not isinstance(section, dict):
                continue
            rows.append(
                [
                    page.get("page_plan_id", ""),
                    page.get("display_title", ""),
                    section.get("section_key", ""),
                    ", ".join(str(value) for value in section.get("missing_concepts_before", [])),
                    ", ".join(str(value) for value in section.get("key_phrases", [])),
                    section.get("reinforcement_preview", ""),
                ]
            )
    return (
        "# Update Preservation Reinforcement Report\n\n"
        f"- Changed: `{str(bool(report.get('changed'))).lower()}`\n"
        f"- Reinforced pages: `{report.get('reinforced_page_count', 0)}`\n"
        f"- Reinforced sections: `{report.get('reinforced_section_count', 0)}`\n\n"
        + (
            format_markdown_table(["页面计划", "标题", "段落", "补足概念", "关键短语", "补强预览"], rows)
            if rows
            else "_无需本地补强。_"
        )
        + "\n"
    )


def render_update_preservation_pack_markdown(pack: dict[str, Any]) -> str:
    rows: list[list[Any]] = []
    for page in pack.get("pages", []):
        if not isinstance(page, dict):
            continue
        for section in page.get("sections", []):
            if not isinstance(section, dict):
                continue
            rows.append(
                [
                    page.get("page_plan_id", ""),
                    page.get("display_title", ""),
                    section.get("section_key", ""),
                    section.get("min_required_matches", 0),
                    ", ".join(str(phrase) for phrase in section.get("key_phrases", [])[:4]),
                    section.get("min_required_concept_matches", 0),
                    ", ".join(str(concept.get("label", "")) for concept in section.get("concept_obligations", [])[:5] if isinstance(concept, dict)),
                ]
            )
    return (
        "# Update Preservation Pack\n\n"
        "这些 obligations 会传给 draft_rendering，并由本地 validator 检查；如果模型未吸收旧知识，会先触发 repair，最终仍由旧页保留观察兜底。\n\n"
        f"{format_markdown_table(['页面计划', '标题', '段落', '最少短语', '关键短语', '最少概念', '概念义务'], rows) if rows else '_本轮没有 update preservation obligations。_'}\n"
    )


def draft_page_summary(page: DraftPageItem) -> str:
    return page.summary.strip()


def draft_page_core_markdown(page: DraftPageItem) -> str:
    return page.body_markdown.strip()


def draft_page_open_questions(page: DraftPageItem) -> str:
    return page.open_questions.strip()


def draft_page_text_for_preservation_section(page: DraftPageItem, section_key: str) -> str:
    if section_key == "summary":
        return draft_page_summary(page)
    if section_key == "open_questions":
        return draft_page_open_questions(page)
    if draft_field_for_preservation_section(section_key) == "body_markdown":
        return draft_page_core_markdown(page)
    return ""


def draft_field_for_preservation_section(section_key: str) -> Literal["summary", "body_markdown", "open_questions"]:
    if section_key == "summary":
        return "summary"
    if section_key == "open_questions":
        return "open_questions"
    return "body_markdown"
