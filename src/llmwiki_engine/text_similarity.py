from __future__ import annotations

import re
import unicodedata


def parenthetical_translation_core(title: str) -> str:
    text = unicodedata.normalize("NFKC", title).strip()
    if not text:
        return text
    parenthetical_parts = re.findall(r"[（(]([^）)]{1,48})[）)]", text)
    if not parenthetical_parts:
        return text
    stripped = re.sub(r"\s*[（(][^）)]{1,48}[）)]\s*", " ", text)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    if not stripped:
        return text
    stripped_has_latin = bool(re.search(r"[A-Za-z]", stripped))
    stripped_has_cjk = bool(re.search(r"[\u4e00-\u9fff]", stripped))
    paren_text = " ".join(parenthetical_parts)
    paren_has_latin = bool(re.search(r"[A-Za-z]", paren_text))
    paren_has_cjk = bool(re.search(r"[\u4e00-\u9fff]", paren_text))
    if (stripped_has_latin and paren_has_cjk) or (stripped_has_cjk and paren_has_latin):
        return stripped
    return text


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def both_agent_workflow_compare(left_title: str, right_title: str) -> bool:
    left = left_title.lower()
    right = right_title.lower()
    left_has_agent = "agent" in left or "智能体" in left
    left_has_workflow = "workflow" in left or "工作流" in left or "流程" in left
    right_has_agent = "agent" in right or "智能体" in right
    right_has_workflow = "workflow" in right or "工作流" in right or "流程" in right
    if not (left_has_agent and left_has_workflow and right_has_agent and right_has_workflow):
        return False
    combined = f"{left} {right}"
    has_compare = any(marker in combined for marker in ["vs", "对比", "比较", "区别"])
    return has_compare
