from __future__ import annotations

import re
import unicodedata
from typing import Any


DEFAULT_SOURCE_MAP_MAX_SECTIONS = 56
SOURCE_SECTION_LOCATOR_RE = re.compile(r"\bS(?P<start>\d{3})(?:\s*[-–—~至到]\s*S?(?P<end>\d{3}))?\b", re.IGNORECASE)

SOURCE_EXCERPT_LOW_SIGNAL_NORMALIZED_CUES = {
    "ai", "agi", "anthropic", "claude", "claudecode", "cowork", "pm",
    "产品", "模型", "功能", "团队", "用户", "角色", "设计", "问题", "成功", "未来",
    "什么", "如何", "为什么", "需要", "应该", "可以", "通过", "帮助", "重要", "不同", "类型", "应用", "开发",
}

SOURCE_EXCERPT_SHORT_ASCII_SIGNAL_CUES = {
    "api", "arr", "cli", "eval", "gtm", "mvp", "prd",
}


def markdown_sections_for_source_map(text: str, *, max_sections: int) -> list[dict[str, Any]]:
    lines = text.splitlines(keepends=True)
    offsets: list[int] = []
    cursor = 0
    heading_indices: list[int] = []
    for index, line in enumerate(lines):
        offsets.append(cursor)
        if re.match(r"^\s{0,3}#{1,6}\s+\S", line):
            heading_indices.append(index)
        cursor += len(line)
    if not lines:
        return []
    if not heading_indices or heading_indices[0] != 0:
        heading_indices.insert(0, 0)
    all_heading_indices = sorted(set(heading_indices))
    heading_indices = all_heading_indices[:max_sections]
    sections: list[dict[str, Any]] = []
    for start_index in heading_indices:
        source_position = all_heading_indices.index(start_index)
        end_index = all_heading_indices[source_position + 1] if source_position + 1 < len(all_heading_indices) else len(lines)
        text_block = "".join(lines[start_index:end_index]).strip()
        if not text_block:
            continue
        heading_line = lines[start_index].strip() if start_index < len(lines) else ""
        match = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$", heading_line)
        level = len(match.group(1)) if match else 0
        heading = match.group(2).strip() if match else "Preamble"
        char_start = offsets[start_index] if start_index < len(offsets) else 0
        char_end = offsets[end_index] if end_index < len(offsets) else len(text)
        sections.append(
            {
                "section_id": f"S{len(sections) + 1:03d}",
                "heading": heading,
                "level": level,
                "line_start": start_index + 1,
                "line_end": end_index,
                "char_start": char_start,
                "char_end": char_end,
                "text": text_block,
            }
        )
    return sections


def source_snippets_are_start_fallback(snippets: list[dict[str, Any]]) -> bool:
    return not snippets or all(snippet.get("cue") == "fallback_start" for snippet in snippets)


def source_global_excerpt(text: str, limit: int) -> str:
    headings = "\n".join(line.strip() for line in text.splitlines() if line.lstrip().startswith("#"))
    prefix = text.strip()[: max(0, limit - len(headings) - 4)]
    return _merge_markdown_blocks(prefix, headings)[:limit].strip()


def source_snippets_for_cues(text: str, cues: list[str], *, max_chars: int) -> list[dict[str, Any]]:
    if max_chars <= 0:
        return []
    semantic_terms = source_semantic_match_terms(cues)
    window_candidates: list[tuple[int, int, str, int]] = []
    seen_positions: set[int] = set()
    locator_windows = source_section_locator_windows(text, cues, max_chars=max_chars, semantic_terms=semantic_terms)
    for start, end, cue, score in locator_windows:
        if any(abs(start - existing) < source_excerpt_start_tolerance(cue) for existing in seen_positions):
            continue
        seen_positions.add(start)
        window_candidates.append((start, end, cue, score))
    for cue in source_excerpt_cues(cues):
        position = find_source_cue(text, cue)
        if position < 0:
            continue
        center = max(0, position)
        heading_start = markdown_heading_start_at_position(text, center)
        if heading_start is not None:
            start = heading_start
            line_end = text.find("\n", heading_start)
            section_search_start = line_end + 1 if line_end >= 0 else len(text)
            end = min(source_heading_section_end(text, section_search_start, heading_marker_at_position(text, heading_start)), start + max_chars)
        else:
            before_chars = min(450, max(80, max_chars // 3))
            after_chars = min(650, max(180, max_chars - before_chars))
            start = max(0, center - before_chars)
            end = min(len(text), center + len(cue) + after_chars)
            start = adjust_window_start(text, start)
            end = adjust_window_end(text, end)
        window_text = text[start:end]
        score = source_excerpt_window_score(window_text, cue, semantic_terms)
        if source_excerpt_low_signal_cue(cue) and score < 28:
            continue
        if any(abs(start - existing) < source_excerpt_start_tolerance(cue) for existing in seen_positions):
            continue
        seen_positions.add(start)
        window_candidates.append((start, end, cue, score))
    if locator_windows:
        semantic_window = source_semantic_fallback_window(text, cues, max_chars=max_chars)
        if semantic_window is not None:
            start, end, cue = semantic_window
            if not any(abs(start - existing) < source_excerpt_start_tolerance(cue) for existing in seen_positions):
                seen_positions.add(start)
                window_candidates.append((start, end, cue, 88))
    windows: list[tuple[int, int, str]] = []
    for start, end, cue, _score in sorted(window_candidates, key=lambda item: (-item[3], item[0])):
        if any(
            ranges_overlap(
                start,
                end,
                existing_start,
                existing_end,
                tolerance=source_excerpt_overlap_tolerance(cue, existing_cue),
            )
            for existing_start, existing_end, existing_cue in windows
        ):
            continue
        windows.append((start, end, cue))
        if sum(existing_end - existing_start for existing_start, existing_end, _ in windows) >= max_chars:
            break
    if not windows:
        semantic_window = source_semantic_fallback_window(text, cues, max_chars=max_chars)
        if semantic_window is not None:
            start, end, cue = semantic_window
            return [{"cue": cue, "start": start, "end": end, "text": text[start:end].strip()}]
        heading_window = source_heading_fallback_window(text, cues, max_chars=max_chars)
        if heading_window is not None:
            start, end, cue = heading_window
            return [{"cue": cue, "start": start, "end": end, "text": text[start:end].strip()}]
        fallback = text.strip()[:max_chars]
        return [{"cue": "fallback_start", "start": 0, "end": len(fallback), "text": fallback}] if fallback else []
    snippets: list[dict[str, Any]] = []
    remaining = max_chars
    for start, end, cue in windows:
        if remaining <= 0:
            break
        snippet = text[start:end].strip()
        if len(snippet) > remaining:
            snippet = snippet[:remaining].rstrip()
            end = start + len(snippet)
        snippets.append({"cue": cue, "start": start, "end": end, "text": snippet})
        remaining -= len(snippet)
    return snippets


def source_excerpt_start_tolerance(cue: str) -> int:
    return 1 if cue.startswith(("source_locator:", "fallback_semantic:")) else 120


def source_excerpt_overlap_tolerance(cue: str, existing_cue: str) -> int:
    if cue.startswith("source_locator:") or existing_cue.startswith("source_locator:"):
        return 0
    return 120


def source_section_locator_windows(
    text: str,
    cues: list[str],
    *,
    max_chars: int,
    semantic_terms: list[str] | None = None,
    max_sections: int = DEFAULT_SOURCE_MAP_MAX_SECTIONS,
) -> list[tuple[int, int, str, int]]:
    section_ids = source_section_locator_ids(cues)
    if not section_ids:
        return []
    sections = {
        section["section_id"]: section
        for section in markdown_sections_for_source_map(text, max_sections=max_sections)
    }
    if not sections:
        return []
    per_locator_limit = min(max_chars, max(160, max_chars // max(1, min(len(section_ids), 3))))
    windows: list[tuple[int, int, str, int]] = []
    for section_id in section_ids:
        section = sections.get(section_id)
        if section is None:
            continue
        start = int(section.get("char_start", 0))
        section_end = int(section.get("char_end", start))
        end = min(section_end, start + per_locator_limit)
        if end <= start:
            continue
        heading = str(section.get("heading", "")).strip()
        window_text = text[start:end]
        score = source_section_locator_window_score(window_text, semantic_terms or [])
        windows.append((start, end, f"source_locator:{section_id}:{heading}", score))
    return windows


def source_section_locator_window_score(window_text: str, semantic_terms: list[str]) -> int:
    if not semantic_terms:
        return 96
    normalized_window = normalized_source_match_text(window_text)
    hits = [term for term in semantic_terms if len(term) >= 4 and term in normalized_window]
    if not hits:
        return 64
    return 96 + min(16, sum(min(8, len(term)) for term in hits[:4]))


def source_section_locator_ids(cues: list[str]) -> list[str]:
    section_ids: list[str] = []
    for cue in cues:
        for match in SOURCE_SECTION_LOCATOR_RE.finditer(cue or ""):
            start = int(match.group("start"))
            end_text = match.group("end")
            end = int(end_text) if end_text else start
            if end < start:
                continue
            for number in range(start, min(end, start + 7) + 1):
                section_id = f"S{number:03d}"
                if section_id not in section_ids:
                    section_ids.append(section_id)
    return section_ids


def source_excerpt_window_score(window_text: str, cue: str, semantic_terms: list[str]) -> int:
    normalized_cue = normalized_source_match_text(cue)
    score = min(36, len(normalized_cue))
    if re.search(r"[\u4e00-\u9fff]", cue) and re.search(r"[A-Za-z]", cue):
        score += 6
    if source_excerpt_low_signal_cue(cue):
        score -= 16
    semantic_score, present_terms, _ = source_semantic_block_score(window_text, semantic_terms)
    score += semantic_score
    if len(present_terms) >= 3:
        score += 8
    if markdown_heading_start_at_position(window_text, 0) == 0:
        score += 16
    return score


def source_excerpt_low_signal_cue(cue: str) -> bool:
    normalized_cue = normalized_source_match_text(cue)
    if normalized_cue in SOURCE_EXCERPT_SHORT_ASCII_SIGNAL_CUES:
        return False
    if normalized_cue in SOURCE_EXCERPT_LOW_SIGNAL_NORMALIZED_CUES:
        return True
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9+#.-]{2,14}", cue.strip()):
        return True
    if re.fullmatch(r"s\d{1,4}", normalized_cue):
        return True
    return len(normalized_cue) < 4


def ranges_overlap(start: int, end: int, other_start: int, other_end: int, *, tolerance: int = 0) -> bool:
    return start < other_end + tolerance and other_start < end + tolerance


def markdown_heading_start_at_position(text: str, position: int) -> int | None:
    line_start = text.rfind("\n", 0, position) + 1
    line_end = text.find("\n", position)
    if line_end < 0:
        line_end = len(text)
    line = text[line_start:line_end]
    return line_start if re.match(r"^#{1,6}\s+", line) else None


def heading_marker_at_position(text: str, position: int) -> str:
    match = re.match(r"^(#{1,6})\s+", text[position:])
    return match.group(1) if match else "#"


def source_heading_fallback_window(text: str, cues: list[str], *, max_chars: int) -> tuple[int, int, str] | None:
    cue_variants = source_excerpt_cues(cues)
    if not cue_variants:
        return None
    best: tuple[int, int, str, int] | None = None
    for line_match in re.finditer(r"(?m)^(#{1,6})\s+(.+?)\s*$", text):
        heading_text = line_match.group(2)
        normalized_heading = normalized_source_match_text(heading_text)
        if len(normalized_heading) < 3:
            continue
        for cue in cue_variants:
            normalized_cue = normalized_source_match_text(cue)
            if len(normalized_cue) < 3:
                continue
            score = heading_match_score(normalized_heading, normalized_cue)
            if score <= 0:
                continue
            if best is None or score > best[3]:
                start = line_match.start()
                end = source_heading_section_end(text, line_match.end(), line_match.group(1))
                best = (start, min(end, start + max_chars), f"fallback_heading:{heading_text}", score)
    if best is None:
        return None
    return best[0], best[1], best[2]


def heading_match_score(normalized_heading: str, normalized_cue: str) -> int:
    if normalized_heading in normalized_cue or normalized_cue in normalized_heading:
        return min(len(normalized_heading), len(normalized_cue)) + 20
    heading_terms = meaningful_match_terms(normalized_heading)
    cue_terms = meaningful_match_terms(normalized_cue)
    overlap = heading_terms & cue_terms
    if len(overlap) < 2 and not any(len(term) >= 6 for term in overlap):
        return 0
    if overlap:
        return sum(len(term) for term in overlap)
    return 0


def source_semantic_fallback_window(text: str, cues: list[str], *, max_chars: int) -> tuple[int, int, str] | None:
    terms = source_semantic_match_terms(cues)
    if len(terms) < 2:
        return None
    best: tuple[int, int, str, int, int] | None = None
    for block_start, block_end, block_text in source_semantic_blocks(text):
        score, present_terms, first_position = source_semantic_block_score(block_text, terms)
        if score <= 0:
            continue
        absolute_position = block_start + first_position
        before_chars = min(420, max(100, max_chars // 3))
        start = max(block_start, absolute_position - before_chars)
        end = min(block_end, start + max_chars)
        start = adjust_window_start(text, start)
        end = adjust_window_end(text, end)
        cue = "fallback_semantic:" + ",".join(present_terms[:4])
        candidate = (start, end, cue, score, len(present_terms))
        if best is None or (score, len(present_terms), block_start * -1) > (best[3], best[4], best[0] * -1):
            best = candidate
    if best is None:
        return None
    return best[0], best[1], best[2]


def source_semantic_blocks(text: str) -> list[tuple[int, int, str]]:
    blocks: list[tuple[int, int, str]] = []
    for match in re.finditer(r"(?ms)(?:^|\n{2,})(?P<body>.*?)(?=\n{2,}|\Z)", text):
        body = match.group("body")
        if not body.strip():
            continue
        leading = len(body) - len(body.lstrip())
        trailing = len(body.rstrip())
        start = match.start("body") + leading
        end = match.start("body") + trailing
        block_text = text[start:end]
        if len(normalized_source_match_text(block_text)) < 24:
            continue
        blocks.append((start, end, block_text))
    return blocks


def source_semantic_block_score(block_text: str, terms: list[str]) -> tuple[int, list[str], int]:
    normalized_block, position_map = normalized_source_match_text_with_positions(block_text)
    present: list[str] = []
    first_normalized_position: int | None = None
    for term in terms:
        position = normalized_block.find(term)
        if position < 0:
            continue
        if any(term in existing or existing in term for existing in present):
            continue
        present.append(term)
        first_normalized_position = position if first_normalized_position is None else min(first_normalized_position, position)
    if not present:
        return 0, [], 0
    long_hits = [term for term in present if len(term) >= 4]
    if len(present) < 2 and not long_hits:
        return 0, [], 0
    score = sum(min(12, len(term)) for term in present) + len(present) * 3
    if len(present) >= 2:
        score += 8
    if not long_hits:
        score -= 6
    if score < 18:
        return 0, [], 0
    first_position = 0
    if first_normalized_position is not None and first_normalized_position < len(position_map):
        first_position = position_map[first_normalized_position]
    return score, present, first_position


def source_semantic_match_terms(cues: list[str]) -> list[str]:
    ascii_terms: set[str] = set()
    cjk_terms: set[str] = set()
    english_stopwords = {"and", "are", "approved", "digest", "for", "from", "how", "page", "section", "source", "into", "that", "the", "this", "with", "wiki", "why"}
    cjk_stop_terms = {"来源", "定位", "来源定位", "摘要", "问题", "价值", "页面", "概念", "设计", "部分", "小节", "访谈", "讨论"}
    for cue in source_excerpt_cues(cues):
        normalized = unicodedata.normalize("NFKC", cue).lower()
        for token in re.findall(r"[a-z][a-z0-9+#./-]{2,}", normalized):
            if token in english_stopwords or source_section_locator_token(token):
                continue
            normalized_token = normalized_source_match_text(token)
            if len(normalized_token) >= 3:
                ascii_terms.add(normalized_token)
        for segment in re.findall(r"[\u4e00-\u9fff]{3,}", normalized):
            if segment in cjk_stop_terms:
                continue
            max_size = min(8, len(segment))
            for size in range(max_size, 2, -1):
                for index in range(0, len(segment) - size + 1):
                    term = segment[index : index + size]
                    if term not in cjk_stop_terms:
                        normalized_term = normalized_source_match_text(term)
                        if len(normalized_term) >= 3:
                            cjk_terms.add(normalized_term)
    ascii_sorted = sorted(ascii_terms, key=lambda value: (-len(value), value))
    cjk_sorted = sorted(cjk_terms, key=lambda value: (-len(value), value))
    total_limit = 80
    ascii_limit = 32
    cjk_min_limit = 24
    selected = ascii_sorted[:ascii_limit]
    cjk_limit = max(cjk_min_limit, total_limit - len(selected))
    selected.extend(cjk_sorted[:cjk_limit])
    if len(selected) < total_limit:
        selected.extend(ascii_sorted[ascii_limit : ascii_limit + total_limit - len(selected)])
    return selected[:total_limit]


def source_section_locator_token(token: str) -> bool:
    return bool(re.fullmatch(r"s\d{3}(?:[-–—~至到/]s?\d{3})?", token.strip().lower()))


def meaningful_match_terms(text: str) -> set[str]:
    terms = set(re.findall(r"[\u4e00-\u9fff]{2,}|[a-z0-9]{3,}", text))
    terms -= SOURCE_EXCERPT_LOW_SIGNAL_NORMALIZED_CUES
    if not terms and len(text) >= 4:
        terms.update(text[index : index + 4] for index in range(0, len(text) - 3))
    return terms


def source_heading_section_end(text: str, start: int, marker: str) -> int:
    pattern = re.compile(r"(?m)^(#{1,%d})\s+" % len(marker))
    match = pattern.search(text, start)
    return match.start() if match is not None else len(text)


def source_excerpt_cues(cues: list[str]) -> list[str]:
    normalized: list[str] = []
    for cue in cues:
        for piece in re.split(r"[\n。；;，,、|]+", cue or ""):
            text = piece.strip().strip("`*_ ")
            if len(text) > 160:
                text = text[:160].rstrip()
            for variant in source_excerpt_cue_variants(text):
                if variant not in normalized:
                    normalized.append(variant)
    normalized.sort(key=lambda value: (len(normalized_source_match_text(value)) < 8, -len(normalized_source_match_text(value))))
    return normalized[:40]


def source_excerpt_cue_variants(text: str) -> list[str]:
    variants: list[str] = []

    def add(value: str) -> None:
        value = value.strip().strip("`*_ -")
        if len(normalized_source_match_text(value)) < 3:
            return
        if value not in variants:
            variants.append(value)

    add(text)
    without_parenthetical = re.sub(r"[\(（][^\)）]{2,80}[\)）]", " ", text)
    add(without_parenthetical)
    for match in re.finditer(r"[\(（]([^\)）]{2,80})[\)）]", text):
        add(match.group(1))
    for segment in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9 +#./-]{2,}", text):
        add(segment)
    return variants


def find_source_cue(text: str, cue: str) -> int:
    position = text.find(cue)
    if position >= 0:
        return position
    normalized_cue = normalized_source_match_text(cue)
    if len(normalized_cue) < 4:
        return -1
    normalized_text, position_map = normalized_source_match_text_with_positions(text)
    normalized_position = normalized_text.find(normalized_cue)
    if normalized_position < 0:
        return -1
    return position_map[normalized_position] if normalized_position < len(position_map) else -1


def normalized_source_match_text(text: str) -> str:
    return normalized_source_match_text_with_positions(text)[0]


def normalized_source_match_text_with_positions(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    positions: list[int] = []
    for index, char in enumerate(text):
        normalized = unicodedata.normalize("NFKC", char).lower()
        for normalized_char in normalized:
            if normalized_char.isspace():
                continue
            if unicodedata.category(normalized_char).startswith("P"):
                continue
            chars.append(normalized_char)
            positions.append(index)
    return "".join(chars), positions


def adjust_window_start(text: str, start: int) -> int:
    newline = text.rfind("\n", 0, start)
    return newline + 1 if newline >= 0 and start - newline < 160 else start


def adjust_window_end(text: str, end: int) -> int:
    newline = text.find("\n", end)
    return newline if newline >= 0 and newline - end < 160 else end


def _merge_markdown_blocks(existing: str, addition: str) -> str:
    existing = existing.strip()
    addition = addition.strip()
    if not existing:
        return addition
    if not addition:
        return existing
    return f"{existing}\n\n{addition}"
