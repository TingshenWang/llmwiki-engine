from __future__ import annotations

import re
import unicodedata

from . import source_excerpt as _source_excerpt
from . import update_preservation as _update_preservation
from .markdown_utils import dedupe_strings, is_empty_placeholder, merge_markdown_blocks
from .models import SectionMergeChange
from .open_questions import is_low_signal_open_question, meaningful_open_question_lines, open_question_key


__all__ = ("merge_update_section",)


def merge_update_section(
    section_key: str,
    old: str,
    new: str,
    *,
    absorption_context: str | None = None,
) -> tuple[str, SectionMergeChange]:
    old = old.strip()
    new = new.strip()
    if section_key == "open_questions":
        return _merge_update_open_questions_section(old, new)
    retained: list[str] = []
    added: list[str] = []
    removed: list[str] = []
    preserved_old: list[str] = []
    removal_reason = ""
    needs_manual_resolution = False
    absorbed, matched_phrases, _ = _update_preservation.update_section_absorption(old, new) if old and new else (False, [], [])
    context_absorbed = False
    context_matched_phrases: list[str] = []
    if old and new and not absorbed and absorption_context and _update_merge_should_preserve_old_section(section_key, old):
        context_text = absorption_context.strip()
        if context_text and context_text != new:
            context_absorbed, context_matched_phrases, _ = _update_preservation.update_section_absorption(old, context_text)
            absorbed = context_absorbed
    if section_key == "additional_notes":
        absorbed = False
        context_absorbed = False
        matched_phrases = []
        context_matched_phrases = []
    if old and new and absorbed:
        retained.append(old)
    if new and not is_empty_placeholder(new):
        added.append(new)
    if old and not retained and old != new and not is_empty_placeholder(old):
        if section_key == "additional_notes":
            preserved_notes, removed_notes, absorbed_notes = _split_high_signal_old_additional_notes(
                old,
                new,
                absorption_context=absorption_context or "",
            )
            if preserved_notes:
                new = merge_markdown_blocks(new, _preserved_old_additional_notes_block(preserved_notes))
                retained.extend([*absorbed_notes, *preserved_notes])
                preserved_old.extend(preserved_notes)
                removed.extend(removed_notes)
                removal_reason = (
                    "高信号旧补充观察已自动保留为旧页补充观察；无需阻塞审批，建议后续按需整理。"
                    "低信号或已覆盖的旧补充观察不机械保留。"
                )
            elif absorbed_notes:
                retained.extend(absorbed_notes)
                removed.extend(removed_notes)
                removal_reason = "高信号旧补充观察已被新草稿吸收；低信号旧补充观察不机械保留。"
            elif removed_notes:
                removed.extend(removed_notes)
                removal_reason = "旧段落不属于 update preservation 核心义务，且未被新草稿自然吸收；本轮不再机械保留。"
            else:
                removed.append(old)
                removal_reason = "旧段落不属于 update preservation 核心义务，且未被新草稿自然吸收；本轮不再机械保留。"
        elif _update_merge_should_preserve_old_section(section_key, old):
            preserved = _preserved_old_section_block(old)
            new = merge_markdown_blocks(new, preserved)
            retained.append(old)
            preserved_old.append(old)
            needs_manual_resolution = True
            removal_reason = "模型完整重写后未显式吸收该旧段落；系统已临时保留为旧页保留观察，draft review 需消化、改写或确认删除。"
        else:
            removed.append(old)
            removal_reason = "旧段落不属于 update preservation 核心义务，且未被新草稿自然吸收；本轮不再机械保留。"
    elif old and retained and old != new and old not in new:
        if context_absorbed:
            matched = context_matched_phrases[:4]
            removal_reason = (
                f"模型已在新草稿其他章节吸收旧段落关键短语/概念义务：{', '.join(matched)}。"
                if matched
                else "模型已在新草稿其他章节吸收旧段落概念义务。"
            )
        elif matched_phrases:
            removal_reason = f"模型已通过关键短语/概念义务吸收旧段落：{', '.join(matched_phrases[:4])}。"
    return (
        new,
        SectionMergeChange(
            section_key=section_key,
            retained=retained,
            added=added,
            removed=removed,
            preserved_old=preserved_old,
            needs_manual_resolution=needs_manual_resolution,
            removal_reason=removal_reason,
        ),
    )


def _split_high_signal_old_additional_notes(
    old: str,
    new: str,
    *,
    absorption_context: str,
) -> tuple[list[str], list[str], list[str]]:
    preserved: list[str] = []
    removed: list[str] = []
    absorbed: list[str] = []
    for note in _old_additional_note_units(old):
        if not _old_additional_note_is_high_signal_boundary(note):
            removed.append(note)
            continue
        if _old_additional_note_superseded(note, new) or _old_additional_note_superseded(note, absorption_context):
            removed.append(note)
            continue
        if _old_additional_note_absorbed(note, new) or _old_additional_note_absorbed(note, absorption_context):
            absorbed.append(note)
            continue
        preserved.append(note)
    return dedupe_strings(preserved), dedupe_strings(removed), dedupe_strings(absorbed)


def _old_additional_note_units(text: str) -> list[str]:
    units: list[str] = []
    paragraph: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            if paragraph:
                units.append(" ".join(paragraph).strip())
                paragraph = []
            continue
        bullet = re.match(r"^(?:[-*+]|\d+[.)、])\s+(?P<body>.+)$", stripped)
        if bullet:
            if paragraph:
                units.append(" ".join(paragraph).strip())
                paragraph = []
            units.append(bullet.group("body").strip())
            continue
        paragraph.append(stripped)
    if paragraph:
        units.append(" ".join(paragraph).strip())
    normalized_units = [_strip_old_additional_note_label(unit) for unit in units]
    return [unit for unit in dedupe_strings(normalized_units) if unit and not is_empty_placeholder(unit)]


def _strip_old_additional_note_label(text: str) -> str:
    stripped = text.strip()
    while True:
        next_value = re.sub(r"^旧页补充观察[:：]\s*", "", stripped).strip()
        if next_value == stripped:
            break
        stripped = next_value
    return stripped.strip()


def _old_additional_note_is_high_signal_boundary(note: str) -> bool:
    normalized = re.sub(r"\s+", "", unicodedata.normalize("NFKC", note))
    if len(normalized) < 12:
        return False
    if re.search(r"[?？]$", normalized) or normalized.startswith(("如何", "是否", "为什么", "能否", "有没有")):
        return False
    if re.search(r"(?:需要|待|尚需|仍需)?确认是否|待确认|尚需确认|仍需确认|是否存在|待验证|待补来源", normalized):
        return False
    if any(marker in normalized for marker in ["暂无", "没有明确", "可与", "关联阅读", "后续可以继续补充"]):
        return False
    strong_markers = (
        "应视为",
        "不能视为",
        "不可视为",
        "并非绝对真实",
        "不是绝对真实",
        "用户确认",
        "需要确认",
        "必须确认",
        "重要决定",
        "重大决策",
        "不可逆",
        "安全风险",
        "可靠性风险",
        "隐私风险",
        "成本约束",
        "权限边界",
        "隔离边界",
    )
    if any(marker in normalized for marker in strong_markers):
        return True
    risk_or_boundary = any(marker in normalized for marker in ["风险", "边界", "限制", "约束"])
    domain_signal = any(
        marker in normalized
        for marker in ["安全", "可靠性", "准确性", "一致性", "成本", "权限", "隔离", "隐私", "审计", "确认"]
    )
    modal_signal = any(marker in normalized for marker in ["需要", "应该", "应当", "不能", "不应", "必须"])
    return risk_or_boundary and domain_signal and modal_signal


def _old_additional_note_absorbed(note: str, target: str) -> bool:
    if not note.strip() or not target.strip():
        return False
    normalized_note = _source_excerpt.normalized_source_match_text(note)
    normalized_target = _source_excerpt.normalized_source_match_text(target)
    if normalized_note and normalized_note in normalized_target:
        return True
    return _old_additional_note_boundary_paraphrase_absorbed(note, target)


def _old_additional_note_boundary_paraphrase_absorbed(note: str, target: str) -> bool:
    note_norm = re.sub(r"\s+", "", unicodedata.normalize("NFKC", note.lower()))
    target_norm = re.sub(r"\s+", "", unicodedata.normalize("NFKC", target.lower()))
    obligations: list[str] = []
    if "用户确认" in note_norm:
        obligations.append("user_confirmation")
    if "召回记忆" in note_norm and any(marker in note_norm for marker in ["不能只依赖", "不能依赖", "绝对真实", "上下文"]):
        obligations.append("memory_context_boundary")
    if not obligations:
        return False
    for obligation in obligations:
        if obligation == "user_confirmation" and not _old_additional_note_target_has_user_confirmation_boundary(target_norm):
            return False
        if obligation == "memory_context_boundary" and not _old_additional_note_target_has_memory_context_boundary(target_norm):
            return False
    return True


def _old_additional_note_target_has_user_confirmation_boundary(target_norm: str) -> bool:
    decision_markers = ("重要决定", "重要决策", "重大决策")
    positive_markers = ("需要", "仍需", "仍需要", "应由", "必须", "由用户确认")
    negative_markers = ("不需要", "无需", "不必", "不再需要", "免于")
    for clause in _old_additional_note_supersession_clauses(target_norm):
        if "用户确认" not in clause or not any(marker in clause for marker in decision_markers):
            continue
        if any(re.search(rf"{marker}.{{0,8}}用户确认|用户确认.{{0,8}}{marker}", clause) for marker in negative_markers):
            continue
        if any(marker in clause for marker in positive_markers):
            return True
    return False


def _old_additional_note_target_has_memory_context_boundary(target_norm: str) -> bool:
    boundary_markers = ("辅助上下文", "有帮助的上下文", "作为上下文", "只能作为", "不能只依赖", "不能依赖", "不是绝对真实", "非绝对真实")
    for clause in _old_additional_note_supersession_clauses(target_norm):
        if not re.search(r"召回的?记忆|记忆召回", clause):
            continue
        if _old_additional_note_clause_negates_memory_context(clause):
            continue
        if any(marker in clause for marker in boundary_markers):
            return True
    return False


def _old_additional_note_clause_negates_memory_context(clause: str) -> bool:
    negative = ("不是", "并非", "不作为", "不能作为", "不应作为", "不再作为")
    context_terms = ("辅助上下文", "有帮助的上下文", "上下文")
    return any(re.search(rf"{marker}.{{0,8}}{term}", clause) for marker in negative for term in context_terms)


def _old_additional_note_superseded(note: str, target: str) -> bool:
    if not note.strip() or not target.strip():
        return False
    anchors = _old_additional_note_supersession_anchors(note)
    if not anchors:
        return False
    for sentence in _old_additional_note_supersession_sentences(target):
        if not any(anchor in sentence for anchor in anchors):
            continue
        if _old_additional_note_sentence_supersedes_anchor(sentence, anchors):
            return True
    return False


def _old_additional_note_supersession_anchors(note: str) -> list[str]:
    normalized = re.sub(r"\s+", "", unicodedata.normalize("NFKC", note.lower()))
    anchors = [
        "用户确认",
        "不可逆",
        "绝对真实",
        "召回记忆",
        "权限边界",
        "隔离边界",
        "安全风险",
        "可靠性风险",
        "隐私风险",
        "成本约束",
    ]
    return [anchor for anchor in anchors if anchor in normalized]


def _old_additional_note_supersession_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text.lower()))
    return [sentence for sentence in re.split(r"[。！？!?；;\n]+", normalized) if sentence]


def _old_additional_note_supersession_clauses(sentence: str) -> list[str]:
    return [clause for clause in re.split(r"[，,、]|但|不过|然而|而|同时|并且", sentence) if clause]


def _old_additional_note_clause_preserves_anchor(clause: str, anchor: str) -> bool:
    preservation_markers = ("仍", "仍然", "继续", "依然", "还是")
    for marker in preservation_markers:
        if re.search(rf"{marker}.{{0,8}}{re.escape(anchor)}|{re.escape(anchor)}.{{0,8}}{marker}", clause):
            return True
    return False


def _old_additional_note_sentence_supersedes_anchor(sentence: str, anchors: list[str]) -> bool:
    for clause in _old_additional_note_supersession_clauses(sentence):
        for anchor in anchors:
            if anchor not in clause:
                continue
            if _old_additional_note_clause_preserves_anchor(clause, anchor):
                continue
            if _old_additional_note_clause_is_capability_change(clause, anchor):
                continue
            if _old_additional_note_clause_supersedes_anchor(clause, anchor):
                return True
    return False


def _old_additional_note_clause_is_capability_change(clause: str, anchor: str) -> bool:
    capability_terms = r"(?:元数据|字段|日志|api|接口|能力|属性|参数)"
    if re.search(r"(?:已)?改为(?:支持|提供|记录|返回|包含)", clause) and re.search(capability_terms, clause):
        return True
    if re.search(r"(?:已)?改为", clause) and re.search(rf"{re.escape(anchor)}.{{0,8}}{capability_terms}", clause):
        return True
    return False


def _old_additional_note_clause_supersedes_anchor(clause: str, anchor: str) -> bool:
    escaped = re.escape(anchor)
    strong_markers = ("不再需要", "不需要", "无需", "不必", "不再依赖", "不适用", "免于", "deprecated", "废弃", "已废弃")
    if any(re.search(rf"{marker}.{{0,8}}{escaped}|{escaped}.{{0,8}}{marker}", clause) for marker in strong_markers):
        return True
    if re.search(rf"{escaped}.{{0,12}}(?:已)?改为|(?:已)?改为.{{0,12}}{escaped}", clause):
        return True
    if re.search(rf"{escaped}.{{0,12}}替代|替代.{{0,12}}{escaped}", clause):
        return True
    return False


def _preserved_old_additional_notes_block(notes: list[str]) -> str:
    if len(notes) == 1:
        return f"旧页补充观察：{notes[0]}"
    lines = "\n".join(f"- {note}" for note in notes)
    return f"旧页补充观察：\n{lines}"


def _merge_update_open_questions_section(old: str, new: str) -> tuple[str, SectionMergeChange]:
    old_questions = [
        question
        for question in meaningful_open_question_lines(old)
        if not is_low_signal_open_question(question)
    ]
    new_questions = meaningful_open_question_lines(new)
    seen_keys = {open_question_key(question) for question in new_questions}
    retained_old_questions: list[str] = []
    for question in old_questions:
        key = open_question_key(question)
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        retained_old_questions.append(question)
    merged_questions = [*new_questions, *retained_old_questions]
    merged = "\n".join(f"- {question}" for question in merged_questions).strip()
    if not merged:
        merged = new if not is_empty_placeholder(new) else "暂无矛盾与未决问题记录。"
    reason = ""
    if old_questions:
        reason = "旧 open_questions 默认按问题粒度 union/dedupe 保留；占位/低信号问题不机械保留。"
    return (
        merged,
        SectionMergeChange(
            section_key="open_questions",
            retained=retained_old_questions,
            added=new_questions,
            removed=[],
            preserved_old=[],
            needs_manual_resolution=False,
            removal_reason=reason,
        ),
    )


def _update_merge_should_preserve_old_section(section_key: str, old_text: str) -> bool:
    if section_key not in {"summary", "detail", "core_content"}:
        return False
    phrases = _update_preservation.update_preservation_phrases(old_text)
    concepts = _update_preservation.update_preservation_concepts(old_text)
    return not _update_preservation.update_preservation_section_is_low_value(section_key, old_text, phrases, concepts)


def _preserved_old_section_block(old: str) -> str:
    return f"旧页保留观察（来自更新前页面，模型本轮未显式吸收，先保留待审）：\n\n{old.strip()}"
