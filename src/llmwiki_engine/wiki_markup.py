from __future__ import annotations

from pathlib import Path

__all__ = (
    "clean_display_title",
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
