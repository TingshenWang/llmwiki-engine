from __future__ import annotations

import re
from typing import Literal

from .hash_utils import sha256_bytes
from .models import RawLinkCleanupArtifact, RawLinkCleanupLink, RawLinkCleanupWarning
from .system_pages import format_markdown_table


WIKILINK_RE = re.compile(r"(?<!!)\[\[([^\]\n]+)\]\]")
MEDIA_EMBED_RE = re.compile(r"!\[\[([^\]\n]+)\]\]")


def cleanup_raw_wikilinks(text: str) -> tuple[str, list[RawLinkCleanupLink], list[RawLinkCleanupWarning], int]:
    lines = text.splitlines(keepends=True)
    cleaned_lines: list[str] = []
    links: list[RawLinkCleanupLink] = []
    warnings: list[RawLinkCleanupWarning] = []
    in_frontmatter = frontmatter_seen = in_fenced_code = False
    fence_marker = ""
    preserved_media_total = 0
    for index, line in enumerate(lines, start=1):
        newline = ""
        body = line
        if body.endswith("\r\n"):
            body = body[:-2]
            newline = "\r\n"
        elif body.endswith("\n"):
            body = body[:-1]
            newline = "\n"
        stripped = body.strip()
        if index == 1 and stripped == "---":
            in_frontmatter = True
            frontmatter_seen = True
            cleaned_lines.append(line)
            continue
        if in_frontmatter and index != 1 and stripped == "---":
            in_frontmatter = False
            cleaned_lines.append(line)
            continue
        fence_match = re.match(r"^(\s*)(```+|~~~+)", body)
        if not in_frontmatter and fence_match:
            marker = fence_match.group(2)[0]
            if not in_fenced_code:
                in_fenced_code = True
                fence_marker = marker
            elif fence_marker == marker:
                in_fenced_code = False
                fence_marker = ""
            cleaned_lines.append(line)
            continue
        context = "frontmatter" if in_frontmatter and frontmatter_seen else "body"
        if in_fenced_code:
            cleaned_lines.append(line)
            continue
        media_matches = list(MEDIA_EMBED_RE.finditer(body))
        for match in media_matches[: max(0, 20 - len(warnings))]:
            warnings.append(
                RawLinkCleanupWarning(warning_type="preserved_media_embed", message="保留 Obsidian 媒体链接；本轮只清理文本 wikilink。", line_number=index, line_excerpt=truncate_excerpt(body))
            )
        preserved_media_count = len(media_matches)
        preserved_media_total += preserved_media_count
        cleaned_body, line_links = cleanup_wikilinks_in_line(body, line_number=index, context=context, start_index=len(links) + 1)
        links.extend(line_links)
        cleaned_lines.append(cleaned_body + newline)
    return "".join(cleaned_lines), links, warnings, preserved_media_total


def cleanup_wikilinks_in_line(
    line: str,
    *,
    line_number: int,
    context: Literal["frontmatter", "body"],
    start_index: int,
) -> tuple[str, list[RawLinkCleanupLink]]:
    pieces: list[str] = []
    links: list[RawLinkCleanupLink] = []
    cursor = 0
    in_code = False
    for match in re.finditer(r"`+", line):
        segment = line[cursor : match.start()]
        pieces.append(_cleanup_wikilink_segment(segment, line, line_number=line_number, context=context, start_index=start_index + len(links), links=links) if not in_code else segment)
        pieces.append(match.group(0))
        in_code = not in_code
        cursor = match.end()
    tail = line[cursor:]
    pieces.append(_cleanup_wikilink_segment(tail, line, line_number=line_number, context=context, start_index=start_index + len(links), links=links) if not in_code else tail)
    return "".join(pieces), links


def _cleanup_wikilink_segment(
    segment: str,
    original_line: str,
    *,
    line_number: int,
    context: Literal["frontmatter", "body"],
    start_index: int,
    links: list[RawLinkCleanupLink],
) -> str:
    def replace(match: re.Match[str]) -> str:
        raw = match.group(1).strip()
        target, label = split_wikilink(raw)
        link_id = f"L{start_index + len(links):03d}"
        links.append(
            RawLinkCleanupLink(link_id=link_id, link_kind="wikilink", label=label, target=target, cleanup_action="unwrap_text", cleanup_context=context, line_number=line_number, original_line_hash=sha256_bytes(original_line.encode("utf-8")), line_excerpt=truncate_excerpt(original_line))
        )
        return label

    return WIKILINK_RE.sub(replace, segment)


def split_wikilink(raw: str) -> tuple[str, str]:
    if "|" in raw:
        target, label = raw.split("|", 1)
        return target.strip(), label.strip() or target.strip()
    return raw.strip(), raw.strip()


def truncate_excerpt(line: str, limit: int = 160) -> str:
    text = " ".join(line.split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1]}..."


def render_raw_link_cleanup_markdown(artifact: RawLinkCleanupArtifact) -> str:
    rows = [
        ["raw_path", f"`{artifact.raw_path}`"],
        ["changed", str(artifact.changed).lower()],
        ["cleanup_rule_version", artifact.cleanup_rule_version],
        ["pre_cleanup_sha256", f"`{artifact.pre_cleanup_sha256}`"],
        ["post_cleanup_sha256", f"`{artifact.post_cleanup_sha256}`"],
        ["cleaned_link_count", str(artifact.cleaned_link_count)],
        ["preserved_media_embed_count", str(artifact.preserved_media_embed_count)],
    ]
    link_rows = [
        [link.link_id, link.cleanup_context, str(link.line_number), link.target, link.label, link.line_excerpt]
        for link in artifact.links
    ]
    warning_rows = [
        [warning.warning_type, str(warning.line_number), warning.message, warning.line_excerpt]
        for warning in artifact.warnings[:20]
    ]
    return (
        "# Raw Obsidian Wikilink 规范化\n\n"
        f"{format_markdown_table(['字段', '值'], rows)}\n\n"
        "## 已清理文本 Wikilink\n\n"
        f"{format_markdown_table(['ID', '位置', '行号', 'Target', 'Label', '行摘录'], link_rows) if link_rows else '_暂无。_'}\n\n"
        "## 保留项 Warning\n\n"
        f"{format_markdown_table(['类型', '行号', '说明', '行摘录'], warning_rows) if warning_rows else '_暂无。_'}\n\n"
        "## Diff\n\n"
        "`cleanup.diff` 是 pre-clean -> post-clean 的 unified diff，仅供人工审计/恢复参考，不会自动 rollback。\n"
    )
