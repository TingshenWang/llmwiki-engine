from __future__ import annotations

import re


SECTION_TITLE_TO_KEY = {
    "摘要": "summary",
    "Summary": "summary",
    "核心内容": "core_content",
    "Core Content": "core_content",
    "详情": "detail",
    "Detail": "detail",
    "Details": "detail",
    "例子": "examples",
    "Examples": "examples",
    "价值点": "value_points",
    "Value Points": "value_points",
    "补充观察": "additional_notes",
    "Additional Notes": "additional_notes",
    "相关页面": "related",
    "Related": "related",
    "Related Pages": "related",
    "矛盾与未决问题": "open_questions",
    "Open Questions": "open_questions",
    "Tensions / Open Questions": "open_questions",
}
ENGLISH_SECTION_TITLE_TO_KEY = {
    title.casefold(): key for title, key in SECTION_TITLE_TO_KEY.items() if title.isascii()
}


def parse_existing_sections(markdown: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current_key: str | None = None
    for line in markdown.splitlines():
        match = re.match(r"^##\s+(.+?)\s*$", line)
        if match:
            section_title = match.group(1).strip()
            current_key = SECTION_TITLE_TO_KEY.get(section_title)
            if current_key is None and section_title.isascii():
                current_key = ENGLISH_SECTION_TITLE_TO_KEY.get(section_title.casefold())
            if current_key is not None:
                sections.setdefault(current_key, [])
            continue
        if current_key is not None:
            sections[current_key].append(line)
    return {key: "\n".join(value).strip() for key, value in sections.items()}


def existing_core_content_from_sections(sections: dict[str, str]) -> str:
    if sections.get("core_content", "").strip():
        return sections["core_content"].strip()
    blocks: list[str] = []
    for key, title in [("detail", ""), ("examples", "例子"), ("value_points", "价值点"), ("additional_notes", "补充观察")]:
        body = sections.get(key, "").strip()
        if not body:
            continue
        blocks.append(f"### {title}\n\n{body}" if title else body)
    return "\n\n".join(blocks).strip()
