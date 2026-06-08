from __future__ import annotations

from datetime import datetime
from pathlib import Path

SYSTEM_MARKER = "<!-- llmwiki:system-page:v3 -->"
SYSTEM_CONTRACT_ERROR = (
    "system page is not supported by this engine; rerun init or ingest to regenerate it"
)


def ensure_system_pages(vault: Path) -> None:
    wiki = vault / "wiki"
    (wiki / "logs").mkdir(parents=True, exist_ok=True)
    _write_system_page_if_missing(
        wiki / "index.md",
        "# 索引\n\n"
        f"{SYSTEM_MARKER}\n\n"
        "## 矛盾与未决问题\n\n"
        "| 问题 | 关联页面 | 更新日期 |\n"
        "| --- | --- | --- |\n",
    )
    _write_system_page_if_missing(
        wiki / "log.md",
        "# 日志\n\n"
        f"{SYSTEM_MARKER}\n\n"
        "| 日期 | 操作数 | 来源 |\n"
        "| --- | ---: | --- |\n",
    )


def local_date() -> str:
    return datetime.now().date().isoformat()


def markdown_table_cell(value: object) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.strip().splitlines()) if "\n" not in text else "<br>".join(part.strip() for part in text.splitlines())
    text = text.replace("|", r"\|")
    return text


def format_markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(markdown_table_cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        padded = [*row, *[""] * max(0, len(headers) - len(row))]
        lines.append("| " + " | ".join(markdown_table_cell(cell) for cell in padded[: len(headers)]) + " |")
    return "\n".join(lines)


def render_index(
    *,
    knowledge_rows: list[dict[str, str]],
    tension_rows: list[dict[str, str]],
    page_type_order: list[str],
) -> str:
    sections = ["# 索引", "", SYSTEM_MARKER, ""]
    seen_types: set[str] = set()
    for page_type in page_type_order:
        type_rows = [row for row in knowledge_rows if row["type"] == page_type]
        if not type_rows:
            continue
        seen_types.add(page_type)
        sections.extend([f"## {_page_type_heading(page_type)}", ""])
        sections.append(
            format_markdown_table(
                ["标题", "页面", "摘要", "更新日期"],
                [[row.get("title", ""), row["page"], row["summary"], row["updated"]] for row in type_rows],
            )
        )
        sections.append("")
    remaining = [row for row in knowledge_rows if row["type"] not in seen_types]
    if remaining:
        sections.extend(["## 其他", ""])
        sections.append(
            format_markdown_table(
                ["标题", "页面", "类型", "摘要", "更新日期"],
                [[row.get("title", ""), row["page"], row["type"], row["summary"], row["updated"]] for row in remaining],
            )
        )
        sections.append("")
    if not knowledge_rows:
        sections.extend(["## 知识页", ""])
        sections.append(format_markdown_table(["标题", "页面", "摘要", "更新日期"], [["暂无知识页记录", "", "", ""]]))
        sections.append("")
    sections.extend(["## 矛盾与未决问题", ""])
    rows = [[row["question"], row["page"], row["updated"]] for row in tension_rows] if tension_rows else [["暂无未决问题记录", "", ""]]
    sections.append(format_markdown_table(["问题", "关联页面", "更新日期"], rows))
    return "\n".join(sections).rstrip() + "\n"


def render_log_index(*, date: str, operation_id: str, source: str, existing_text: str | None = None) -> str:
    date_link = f"[[logs/{date}]]"
    rows = _table_rows(existing_text or "")
    merged = _upsert_log_index_row(rows, [date_link, "1", f"`{source}`"], source)
    table = format_markdown_table(["日期", "操作数", "来源"], merged or [[date_link, "1", f"`{source}`"]])
    return (
        "# 日志\n\n"
        f"{SYSTEM_MARKER}\n\n"
        f"{table}\n\n"
        f"最新 operation: `{operation_id}`\n"
    )


def render_daily_log(
    *,
    date: str,
    operation_id: str,
    raw_path: str,
    created: int,
    updated: int,
    noop: int,
    needs_human: int,
    existing_text: str | None = None,
) -> str:
    rows = _upsert_rows(
        _table_rows(existing_text or ""),
        [[f"`{operation_id}`", f"`{raw_path}`", str(created), str(updated), str(noop), str(needs_human)]],
    )
    table = format_markdown_table(["操作", "原始材料", "新建数", "实际更新数", "未改动数", "人工决策阻断数"], rows)
    return f"# {date}\n\n{SYSTEM_MARKER}\n\n{table}\n"


def _write_system_page_if_missing(path: Path, text: str) -> None:
    if path.exists():
        assert_current_system_page(path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def assert_current_system_page(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if SYSTEM_MARKER in text:
        return
    raise RuntimeError(f"{SYSTEM_CONTRACT_ERROR}: {path}")


def _page_type_heading(page_type: str) -> str:
    return {
        "concept": "概念",
        "entity": "实体",
        "design": "设计",
        "comparison": "对比",
        "overview": "总览",
        "event": "事件",
        "memory": "记忆",
        "idea": "想法",
        "open_question": "未决问题页",
    }.get(page_type, page_type.replace("_", " ").title())


def _table_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue
        cells = _split_table_row(stripped)
        if not cells or _is_header_or_separator(cells) or _is_empty_state(cells):
            continue
        rows.append(cells)
    return rows


def _split_table_row(line: str) -> list[str]:
    content = line.strip().strip("|")
    cells: list[str] = []
    buf: list[str] = []
    in_wikilink = False
    escaped = False
    i = 0
    while i < len(content):
        char = content[i]
        pair = content[i : i + 2]
        if escaped:
            buf.append(char)
            escaped = False
            i += 1
            continue
        if char == "\\":
            escaped = True
            buf.append(char)
            i += 1
            continue
        if pair == "[[":
            in_wikilink = True
            buf.append(pair)
            i += 2
            continue
        if pair == "]]":
            in_wikilink = False
            buf.append(pair)
            i += 2
            continue
        if char == "|" and not in_wikilink:
            cells.append("".join(buf).strip())
            buf = []
        else:
            buf.append(char)
        i += 1
    cells.append("".join(buf).strip())
    return cells


def _is_header_or_separator(cells: list[str]) -> bool:
    first = cells[0].strip().lower()
    if first in {"标题", "日期", "操作", "operation", "问题", "page", "date"}:
        return True
    return all(set(cell.strip()) <= {"-", ":"} and "-" in cell for cell in cells)


def _is_empty_state(cells: list[str]) -> bool:
    return bool(cells and (cells[0].startswith("暂无") or cells[0].startswith("_No ")))


def _upsert_rows(existing: list[list[str]], new_rows: list[list[str]]) -> list[list[str]]:
    rows = [row for row in existing if row]
    positions = {row[0]: index for index, row in enumerate(rows)}
    for row in new_rows:
        if not row:
            continue
        key = row[0]
        if key in positions:
            rows[positions[key]] = row
        else:
            positions[key] = len(rows)
            rows.append(row)
    return rows


def _upsert_log_index_row(existing: list[list[str]], new_row: list[str], source: str) -> list[list[str]]:
    rows = [row for row in existing if row]
    key = new_row[0]
    for row in rows:
        if row[0] != key:
            continue
        try:
            row[1] = str(int(row[1]) + 1)
        except (IndexError, ValueError):
            row[1] = "1"
        sources = row[2] if len(row) > 2 else ""
        rendered_source = f"`{source}`"
        if rendered_source not in sources:
            row[2] = f"{sources}, {rendered_source}" if sources else rendered_source
        return rows
    rows.append(new_row)
    return rows
