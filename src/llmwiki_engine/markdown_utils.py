from __future__ import annotations

import re


def compact_payload_text(text: str, limit: int) -> str:
    stripped = text.strip()
    if limit <= 0:
        return ""
    if len(stripped) <= limit:
        return stripped
    return stripped[: max(0, limit - 3)].rstrip() + "..."


def merge_markdown_blocks(existing: str, addition: str) -> str:
    existing = existing.strip()
    addition = addition.strip()
    if not existing:
        return addition
    if not addition:
        return existing
    return f"{existing}\n\n{addition}"


def is_empty_placeholder(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text)
    return not normalized or any(marker in normalized for marker in ["暂无", "没有相关", "无相关", "N/A"])
