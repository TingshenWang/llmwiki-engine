from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .models import WikiKnowledgeEntry


SYSTEM_MARKER = "<!-- llmwiki:system-page:v3 -->"
RELATED_LINK_LIMIT = 3
GRAPH_EXCLUDED_ROOTS = {"raw", "sources", "logs"}
SYSTEM_WIKI_FILES = {"index.md", "log.md"}


@dataclass(frozen=True)
class DailyLogCounts:
    created: int = 0
    updated: int = 0
    noop: int = 0


def initial_index_text() -> str:
    return render_index(entries=[], tension_rows=[], page_type_order=[])


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(_table_cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        padded = [*row, *[""] * max(0, len(headers) - len(row))]
        lines.append("| " + " | ".join(_table_cell(cell) for cell in padded[: len(headers)]) + " |")
    return "\n".join(lines)


def render_index(
    *,
    entries: list[WikiKnowledgeEntry],
    tension_rows: list[dict[str, str]],
    page_type_order: Iterable[str],
) -> str:
    ordered_types = list(page_type_order)
    by_type = {page_type: [] for page_type in ordered_types}
    extra_rows: list[WikiKnowledgeEntry] = []
    for entry in entries:
        if entry.page_type in by_type:
            by_type[entry.page_type].append(entry)
        else:
            extra_rows.append(entry)

    parts = ["# 索引", "", SYSTEM_MARKER, ""]
    wrote_any = False
    for page_type in ordered_types:
        rows = _sorted_entries(by_type[page_type])
        if not rows:
            continue
        wrote_any = True
        parts.extend([f"## {_page_type_heading(page_type)}", ""])
        parts.append(
            markdown_table(
                ["标题", "页面", "摘要", "更新日期"],
                [[entry.title, obsidian_link(entry.path), entry.summary, entry.updated] for entry in rows],
            )
        )
        parts.append("")

    if extra_rows:
        wrote_any = True
        parts.extend(["## 其他", ""])
        parts.append(
            markdown_table(
                ["标题", "页面", "类型", "摘要", "更新日期"],
                [[entry.title, obsidian_link(entry.path), entry.page_type, entry.summary, entry.updated] for entry in _sorted_entries(extra_rows)],
            )
        )
        parts.append("")

    if not wrote_any:
        parts.extend(["## 知识页", "", markdown_table(["标题", "页面", "摘要", "更新日期"], [["暂无知识页记录", "", "", ""]]), ""])

    tension_table_rows = (
        [[row.get("question", ""), row.get("page", ""), row.get("updated", "")] for row in tension_rows]
        if tension_rows
        else [["暂无未决问题记录", "", ""]]
    )
    parts.extend(["## 矛盾与未决问题", "", markdown_table(["问题", "关联页面", "更新日期"], tension_table_rows)])
    return "\n".join(parts).rstrip() + "\n"


def render_daily_log(
    *,
    date: str,
    operation_id: str,
    raw_path: str,
    counts: DailyLogCounts,
    existing_text: str | None,
) -> str:
    rows = [row[:5] for row in _read_table_rows(existing_text or "")]
    row = [
        f"`{operation_id}`",
        f"`{raw_path}`",
        str(counts.created),
        str(counts.updated),
        str(counts.noop),
    ]
    merged = _upsert_by_first_cell(rows, row)
    return "# " + date + "\n\n" + SYSTEM_MARKER + "\n\n" + markdown_table(["操作", "原始材料", "新建数", "实际更新数", "未改动数"], merged) + "\n"


def parse_related_paths(markdown: str) -> list[str]:
    body = _section_body(markdown, {"相关页面", "Related"})
    if not body:
        return []
    paths: list[str] = []
    for match in re.finditer(r"\[\[([^\]]+)\]\]", body):
        target = match.group(1).split("|", 1)[0].strip()
        if target:
            paths.append(target)
    return paths


def normalize_related_path(value: str) -> str | None:
    if value.strip().startswith("/"):
        return None
    text = _clean_target(value)
    if not text:
        return None
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        return None
    parts = path.parts
    if parts and parts[0] == "wiki":
        path = Path(*parts[1:])
    normalized = path.as_posix()
    if not normalized.endswith(".md"):
        normalized += ".md"
    if is_graph_excluded_target(normalized):
        return None
    return normalized


def is_graph_excluded_target(value: str) -> bool:
    text = _clean_target(value)
    if not text:
        return False
    path = Path(text)
    parts = list(path.parts)
    if parts and parts[0] == "wiki":
        parts = parts[1:]
    if not parts:
        return False
    normalized = Path(*parts).as_posix()
    if normalized in SYSTEM_WIKI_FILES:
        return True
    root = parts[0].lower()
    if root in GRAPH_EXCLUDED_ROOTS:
        return True
    return Path(parts[-1]).stem.lower().startswith("source_")


def _clean_target(value: str) -> str:
    text = value.strip().strip("`").strip("<>").replace("\\", "/")
    text = text.split("#", 1)[0].split("?", 1)[0]
    while text.startswith("./"):
        text = text[2:]
    return text.lstrip("/")


def obsidian_link(path: str, alias: str | None = None) -> str:
    target = path[:-3] if path.endswith(".md") else path
    if alias and alias.strip() and alias.strip() != target:
        return f"[[{target}|{alias.strip()}]]"
    return f"[[{target}]]"


def _sorted_entries(entries: list[WikiKnowledgeEntry]) -> list[WikiKnowledgeEntry]:
    return sorted(entries, key=lambda entry: (entry.updated or "", entry.title, entry.path), reverse=True)


def _table_cell(value: object) -> str:
    text = "" if value is None else str(value).strip()
    text = "<br>".join(part.strip() for part in text.splitlines()) if "\n" in text else " ".join(text.split())
    return text.replace("|", r"\|")


def _page_type_heading(page_type: str) -> str:
    headings = {
        "concept": "概念",
        "entity": "实体",
        "design": "设计",
        "comparison": "对比",
        "overview": "总览",
        "event": "事件",
        "memory": "记忆",
        "idea": "想法",
        "open_question": "未决问题页",
    }
    return headings.get(page_type, page_type.replace("_", " ").title())


def _display_title(path: str) -> str:
    stem = Path(path).stem
    for prefix in ["Concept_", "Entity_", "Design_", "Comparison_", "Overview_", "Event_", "Open_Question_"]:
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
    return stem.replace("_", " ")


def _section_body(markdown: str, headings: set[str]) -> str:
    wanted = {heading.lower() for heading in headings}
    lines = markdown.splitlines()
    capture = False
    body: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            title = stripped[3:].strip().lower()
            if capture:
                break
            capture = title in wanted
            continue
        if capture:
            body.append(line)
    return "\n".join(body).strip()


def _read_table_rows(markdown: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue
        cells = _split_table_row(stripped)
        if not cells or _looks_like_table_header(cells) or _looks_like_empty_row(cells):
            continue
        rows.append(cells)
    return rows


def _split_table_row(row: str) -> list[str]:
    text = row.strip().strip("|")
    cells: list[str] = []
    buffer: list[str] = []
    in_link = False
    escaped = False
    index = 0
    while index < len(text):
        pair = text[index : index + 2]
        char = text[index]
        if escaped:
            buffer.append(char)
            escaped = False
            index += 1
            continue
        if char == "\\":
            buffer.append(char)
            escaped = True
            index += 1
            continue
        if pair == "[[":
            in_link = True
            buffer.append(pair)
            index += 2
            continue
        if pair == "]]":
            in_link = False
            buffer.append(pair)
            index += 2
            continue
        if char == "|" and not in_link:
            cells.append("".join(buffer).strip())
            buffer = []
        else:
            buffer.append(char)
        index += 1
    cells.append("".join(buffer).strip())
    return cells


def _looks_like_table_header(cells: list[str]) -> bool:
    first = cells[0].strip().lower()
    if first in {"标题", "日期", "操作", "问题", "page", "date", "operation"}:
        return True
    return all(set(cell.strip()) <= {"-", ":"} and "-" in cell for cell in cells)


def _looks_like_empty_row(cells: list[str]) -> bool:
    return bool(cells and cells[0].startswith("暂无"))


def _upsert_by_first_cell(rows: list[list[str]], new_row: list[str]) -> list[list[str]]:
    result = [row for row in rows if row]
    key = new_row[0]
    for index, row in enumerate(result):
        if row[0] == key:
            result[index] = new_row
            return result
    return [*result, new_row]
