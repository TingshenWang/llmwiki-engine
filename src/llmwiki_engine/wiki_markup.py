from __future__ import annotations

from pathlib import Path

__all__ = (
    "clean_display_title",
    "normalize_related_key",
    "normalize_related_candidate_path",
    "obsidian_alias_link",
    "obsidian_link",
    "obsidian_link_label",
)


def clean_display_title(title: str) -> str:
    stripped = title.strip()
    for prefix in ["Concept_", "Entity_", "Design_", "Comparison_", "Overview_", "Event_", "Memory_", "Idea_", "Open_Question_"]:
        if stripped.lower().startswith(prefix.lower()):
            return stripped[len(prefix) :].strip()
    return stripped


def normalize_related_key(value: str) -> str:
    text = value.strip().lower()
    for prefix in ["concept_", "entity_", "design_", "comparison_", "overview_", "event_", "memory_", "idea_", "open_question_"]:
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def normalize_related_candidate_path(value: str) -> str | None:
    text = value.strip().strip("`").replace("\\", "/")
    if not text:
        return None
    text = text.split("#", 1)[0]
    while text.startswith("./"):
        text = text[2:]
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        return None
    if path.parts and path.parts[0] == "wiki":
        path = Path(*path.parts[1:])
    if not path.parts or path.parts[0] in {"sources", "logs"} or path.as_posix() in {"index.md", "log.md"}:
        return None
    if path.suffix != ".md":
        path = path.with_suffix(".md")
    return path.as_posix()


def obsidian_link(path: str, title: str | None = None) -> str:
    target = Path(path)
    if target.parts and target.parts[0] == "wiki":
        target = Path(*target.parts[1:])
    return f"[[{target.with_suffix('').as_posix()}]]"


def obsidian_alias_link(path: str, title: str) -> str:
    target = Path(path)
    if target.parts and target.parts[0] == "wiki":
        target = Path(*target.parts[1:])
    return f"[[{target.with_suffix('').as_posix()}|{obsidian_link_label(title)}]]"


def obsidian_link_label(value: str) -> str:
    label = " ".join(value.replace("|", "/").replace("]", "").split())
    return label or "Untitled"
