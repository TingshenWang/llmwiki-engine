from __future__ import annotations

import re
from collections import Counter


TOKEN_RE = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]+")


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


def markdown_sections(text: str) -> list[tuple[str, str]]:
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_title = ""
    current_body: list[str] = []
    for line in lines:
        if line.startswith("## "):
            if current_title or current_body:
                sections.append((current_title, current_body))
            current_title = line[3:].strip()
            current_body = []
        else:
            current_body.append(line)
    if current_title or current_body:
        sections.append((current_title, current_body))
    return [(title, "\n".join(body).strip()) for title, body in sections if "\n".join(body).strip()]


def tokens(text: str) -> set[str]:
    return {match.group(0).lower() for match in TOKEN_RE.finditer(text)}


def top_terms(text: str, limit: int = 8) -> list[str]:
    counts = Counter(match.group(0).lower() for match in TOKEN_RE.finditer(text))
    stop = {
        "the",
        "and",
        "for",
        "with",
        "this",
        "that",
        "一个",
        "这个",
        "以及",
        "可以",
        "需要",
    }
    return [word for word, _ in counts.most_common() if word not in stop][:limit]


def lexical_score(query: str, document: str) -> float:
    query_tokens = tokens(query)
    document_tokens = tokens(document)
    if not query_tokens or not document_tokens:
        return 0.0
    overlap = len(query_tokens & document_tokens)
    return overlap / max(len(query_tokens), 1)


def classify_page_type(title: str, body: str) -> str:
    lowered = f"{title}\n{body}".lower()
    if any(term in lowered for term in [" vs ", "对比", "比较", "comparison"]):
        return "comparison"
    if any(term in lowered for term in ["流程", "架构", "设计", "pipeline", "workflow", "system", "engine"]):
        return "design"
    if any(term in lowered for term in ["问题", "question", "unknown", "tension"]):
        return "open_question"
    return "concept"


def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", text))
