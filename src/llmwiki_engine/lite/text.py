from __future__ import annotations

import re


def title_from_markdown(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip() or fallback
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:80]
    return fallback


def strip_frontmatter(text: str) -> str:
    if not text.startswith("---\n"):
        return text
    end = text.find("\n---", 4)
    if end == -1:
        return text
    return text[end + 4 :].lstrip()


def split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    parts = re.split(r"(?<=[。！？.!?])\s+", normalized)
    return [part.strip() for part in parts if part.strip()]


def summarize(text: str, *, max_sentences: int = 2, max_chars: int = 240) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return ""
    summary = " ".join(sentences[:max_sentences])
    if len(summary) <= max_chars:
        return summary
    return f"{summary[: max_chars - 1].rstrip()}..."


def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", text))
