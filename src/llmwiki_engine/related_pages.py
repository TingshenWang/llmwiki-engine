from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

from . import draft_validation as _draft_validation
from . import wiki_markup as _wiki_markup
from .models import RelatedCandidateReport, WikiContextEntry, WikiMergePlanItem
from .validators import looks_like_untranslated_english


FINAL_RELATED_LIMIT = 3


def chinese_related_reason(model_reason: str, fallback: str) -> str:
    return fallback if not model_reason.strip() or looks_like_untranslated_english(model_reason) else model_reason.strip()


def public_related_reason(model_reason: str, fallback: str) -> str:
    reason = chinese_related_reason(model_reason, fallback)
    if related_reason_has_internal_reference(reason):
        return fallback
    return reason


def related_reason_has_internal_reference(reason: str) -> bool:
    normalized = unicodedata.normalize("NFKC", reason)
    return bool(
        re.search(
            r"\b(?:PP-[A-Za-z0-9_-]+|CAND[A-Za-z0-9_-]*|AGG-[A-Za-z0-9_-]+|auto-[A-Za-z0-9_-]+|"
            r"(?:ENT|CON|DES|CMP|OQ|[ECDO])-?\d+[A-Za-z0-9_-]*|CMP\d{1,3})\b",
            normalized,
            re.IGNORECASE,
        )
        or re.search(
            r"\b(?:source[\s_-]?digest|page[\s_-]?plan|candidate|artifact|prepared[\s_-]?discovered)\b",
            normalized,
            re.IGNORECASE,
        )
        or any(marker in normalized for marker in ["来源摘要把", "相关候选", "页面计划", "候选 id", "候选ID", "候选页面", "候选编号", "候选条目"])
    )


def related_public_fallback(source: str, title: str) -> str:
    clean_title = _wiki_markup.clean_display_title(title)
    if source == "wiki_context":
        return f"`{clean_title}` 可作为当前主题的背景补充。"
    if source == "existing_wiki":
        return f"`{clean_title}` 是已有相关页面，保留作背景补充。"
    return f"`{clean_title}` 与本页同属本次材料中的互补主题，可帮助补足上下游理解。"


def normalize_related_path(value: str) -> str | None:
    text = value.strip().strip("`").replace("\\", "/")
    if not text.endswith(".md"):
        return None
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        return None
    if path.parts and path.parts[0] == "wiki":
        path = Path(*path.parts[1:])
    if not path.parts or path.parts[0] in {"sources", "logs"} or path.as_posix() in {"index.md", "log.md"}:
        return None
    return path.as_posix()


def render_related_pages(
    item: WikiMergePlanItem,
    *,
    existing_entry: WikiContextEntry | None = None,
    report_list: list[RelatedCandidateReport] | None = None,
    known_paths: set[str] | None = None,
) -> str:
    candidates: list[dict[str, Any]] = []
    for order, related in enumerate(item.related_pages):
        candidates.append(
            {
                "target_path": _strip_wiki_prefix(related.target_path),
                "display_title": related.display_title,
                "reason": public_related_reason(
                    related.reason,
                    related_public_fallback(related.source, related.display_title),
                ),
                "source": related.source,
                "priority": _related_candidate_priority(related.source),
                "order": order,
            }
        )
    if existing_entry is not None and existing_entry.expected_state == "present":
        candidates.extend(parse_existing_related_candidates(existing_entry.content))

    candidates.sort(key=lambda candidate: (int(candidate.get("priority", 50)), int(candidate.get("order", 0))))
    rows: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        path = _wiki_markup.normalize_related_candidate_path(candidate["target_path"])
        title = candidate["display_title"].strip() or _wiki_markup.clean_display_title(Path(candidate["target_path"]).stem)
        reason = _draft_validation.normalize_stable_brand_typos(
            public_related_reason(candidate["reason"], related_public_fallback(str(candidate.get("source") or ""), title))
        )
        reject_reason = ""
        if path is None:
            reject_reason = "unknown_path"
        elif path == item.canonical_target_path:
            reject_reason = "self_link"
        elif path in seen:
            reject_reason = "duplicate"
        elif known_paths is not None and path not in known_paths:
            reject_reason = "unknown_path"
        elif len(rows) >= FINAL_RELATED_LIMIT:
            reject_reason = "cap_cutoff"
        if report_list is not None:
            report_list.append(
                RelatedCandidateReport(
                    page_plan_id=item.page_plan_id,
                    target_path=path or candidate["target_path"],
                    display_title=title,
                    reason=reason,
                    source=candidate["source"],
                    decision="cutoff" if reject_reason == "cap_cutoff" else ("filtered" if reject_reason else "kept"),
                    reject_reason=reject_reason,
                )
            )
        if reject_reason or path is None:
            continue
        seen.add(path)
        rows.append(f"- {_wiki_markup.obsidian_alias_link(path, title)}：{reason}")
    if not rows:
        if report_list is not None:
            report_list.append(
                RelatedCandidateReport(
                    page_plan_id=item.page_plan_id,
                    target_path=item.canonical_target_path,
                    display_title=item.display_title,
                    reason=item.related_absence_reason or "low_confidence",
                    source="absence_reason",
                    decision="filtered",
                    reject_reason=item.related_absence_reason or "low_confidence",
                )
            )
        return "- 暂无相关页面记录。"
    return "\n".join(rows)


def parse_existing_related_candidates(markdown: str) -> list[dict[str, Any]]:
    match = re.search(r"(?ms)^##\s+相关页面\s*$\n(?P<body>.*?)(?=^##\s+|\Z)", markdown)
    if not match:
        return []
    candidates: list[dict[str, Any]] = []
    for line in match.group("body").splitlines():
        for link in re.finditer(r"\[\[([^\]]+)\]\]", line):
            target, _, alias = link.group(1).partition("|")
            path = _wiki_markup.normalize_related_candidate_path(target)
            order = len(candidates)
            candidates.append(
                {
                    "target_path": path or target.strip(),
                    "display_title": alias.strip() or _wiki_markup.clean_display_title(Path(target).stem),
                    "reason": "旧 Related 作为候选重新参与排序。",
                    "source": "existing_related",
                    "priority": 0,
                    "order": order,
                }
            )
    return candidates


def _related_candidate_priority(source: str) -> int:
    return {
        "existing_related": 0,
        "exact_or_alias": 1,
        "source_digest": 2,
        "wiki_context": 3,
    }.get(source, 9)


def _strip_wiki_prefix(value: str) -> str:
    path = Path(value.strip().replace("\\", "/"))
    if path.parts and path.parts[0] == "wiki":
        return Path(*path.parts[1:]).as_posix()
    return path.as_posix()
