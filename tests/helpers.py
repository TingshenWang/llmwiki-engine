from pathlib import Path


def copy_fixture_raw(vault: Path, fixture_raw: Path) -> Path:
    target = vault / "raw" / fixture_raw.name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(fixture_raw.read_bytes())
    return target


def draft_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(f"- {draft_text(item).strip()}" for item in value if draft_text(item).strip())
    return str(value)


def draft_body(
    *,
    detail: object = "",
    examples: object = "",
    value_points: object = "",
    additional_notes: object = "",
) -> str:
    blocks = []
    detail_text = draft_text(detail).strip()
    if detail_text:
        blocks.append(detail_text)
    for title, value in [
        ("例子", examples),
        ("价值点", value_points),
        ("补充观察", additional_notes),
    ]:
        text = draft_text(value).strip()
        if text:
            blocks.append(f"### {title}\n\n{text}")
    return "\n\n".join(blocks)
