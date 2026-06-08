from __future__ import annotations

import re
from typing import Any

from . import markdown_utils as _markdown_utils
from . import source_excerpt as _source_excerpt
from .models import SourceDigestCandidate, VaultConfig, WeakOrNoiseItem
from .system_pages import format_markdown_table

SOURCE_DIGEST_FULL_SOURCE_CHAR_LIMIT = 24_000
SOURCE_DIGEST_SOURCE_MAP_TOTAL_LIMIT = 22_000
SOURCE_DIGEST_SOURCE_MAP_GLOBAL_EXCERPT_LIMIT = 2_400
SOURCE_DIGEST_SOURCE_MAP_MIN_SECTION_EXCERPT_LIMIT = 320
SOURCE_DIGEST_SOURCE_MAP_MAX_SECTION_EXCERPT_LIMIT = 900
SOURCE_DIGEST_SOURCE_MAP_MAX_SECTIONS = _source_excerpt.DEFAULT_SOURCE_MAP_MAX_SECTIONS
SOURCE_DIGEST_SOURCE_MAP_MAX_CAPTIONS = 24
PAPER_CAPTION_RE = re.compile(r"(?im)^\s*(?:table|figure)\s+\d+\s*:")


def build_source_digest_source_map(
    approved_prepared_text: str,
    *,
    approved_prepared_ref: str,
    full_source_limit: int = SOURCE_DIGEST_FULL_SOURCE_CHAR_LIMIT,
    total_limit: int = SOURCE_DIGEST_SOURCE_MAP_TOTAL_LIMIT,
    global_limit: int = SOURCE_DIGEST_SOURCE_MAP_GLOBAL_EXCERPT_LIMIT,
    min_section_limit: int = SOURCE_DIGEST_SOURCE_MAP_MIN_SECTION_EXCERPT_LIMIT,
    max_section_limit: int = SOURCE_DIGEST_SOURCE_MAP_MAX_SECTION_EXCERPT_LIMIT,
    max_sections: int = SOURCE_DIGEST_SOURCE_MAP_MAX_SECTIONS,
    max_captions: int = SOURCE_DIGEST_SOURCE_MAP_MAX_CAPTIONS,
) -> dict[str, Any]:
    include_full_source = len(approved_prepared_text) <= full_source_limit
    sections = _source_excerpt.markdown_sections_for_source_map(approved_prepared_text, max_sections=max_sections)
    section_budget_total = max(0, total_limit - global_limit)
    section_limit = max_section_limit
    if sections:
        section_limit = max(min_section_limit, min(max_section_limit, section_budget_total // max(1, len(sections))))
    source_map_sections: list[dict[str, Any]] = []
    included_section_chars = 0
    for section in sections:
        excerpt = "" if include_full_source else _markdown_utils.compact_payload_text(section["text"], section_limit)
        included_section_chars += len(excerpt)
        source_map_sections.append(
            {
                "section_id": section["section_id"],
                "heading": section["heading"],
                "level": section["level"],
                "line_start": section["line_start"],
                "line_end": section["line_end"],
                "char_start": section["char_start"],
                "char_end": section["char_end"],
                "original_char_count": len(section["text"]),
                "excerpt": excerpt,
                "excerpt_char_count": len(excerpt),
                "truncated": len(section["text"].strip()) > len(excerpt),
            }
        )
    global_excerpt = "" if include_full_source else _source_excerpt.source_global_excerpt(approved_prepared_text, global_limit)
    captions = [] if include_full_source else source_digest_caption_snippets(approved_prepared_text, max_captions=max_captions)
    included_chars = len(global_excerpt) + included_section_chars + sum(len(item["text"]) for item in captions)
    heading_count = sum(
        1
        for line in approved_prepared_text.splitlines()
        if re.match(r"^\s{0,3}#{1,6}\s+\S", line)
    )
    return {
        "schema_version": "source_digest_source_map.v1",
        "approved_prepared_ref": approved_prepared_ref,
        "full_source_in_payload": include_full_source,
        "full_source_limit": full_source_limit,
        "original_char_count": len(approved_prepared_text),
        "included_char_count": included_chars,
        "total_limit": total_limit,
        "global_excerpt_limit": global_limit,
        "section_excerpt_limit": section_limit,
        "max_sections": max_sections,
        "omitted_section_count": max(0, heading_count - len(source_map_sections)),
        "global_excerpt": global_excerpt,
        "outline": [
            {
                "heading": section["heading"],
                "level": section["level"],
                "line_start": section["line_start"],
                "section_id": section["section_id"],
            }
            for section in sections
        ],
        "captions": captions,
        "sections": source_map_sections,
    }


def source_digest_caption_snippets(text: str, *, max_captions: int) -> list[dict[str, Any]]:
    captions: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if not PAPER_CAPTION_RE.match(stripped):
            continue
        captions.append(
            {
                "line": line_number,
                "text": _markdown_utils.compact_payload_text(stripped, 280),
            }
        )
        if len(captions) >= max_captions:
            break
    return captions


def source_digest_language_contract(vault_config: VaultConfig) -> dict[str, Any]:
    return {
        "vault_language": vault_config.wiki_language,
        "hard_requirement": (
            "All source_digest user-visible prose fields must be written in Chinese for zh-CN vaults. "
            "Do not answer source_digest in English and do not rely on a later repair pass to translate."
        ),
        "fields_must_be_chinese": [
            "summary",
            "key_takeaways",
            "one_sentence_summary",
            "why_matters",
            "wiki_value",
            "open_question_or_tension",
            "resolution_hint",
            "weak_or_noise_items.why_matters",
        ],
        "stable_terms_may_remain_english": [
            "Claude Code",
            "Cowork",
            "PM",
            "RAG",
            "Workflow",
            "Agent",
            "API",
            "Evals",
            "MCP",
        ],
        "allowed_english_boundary": (
            "Keep stable product names and technical terms in English when they are canonical names, "
            "but surround them with Chinese explanation instead of writing full English sentences."
        ),
        "bad_example": "Cat Wu discusses how the team achieves extremely fast product development cycles.",
        "good_example": "Cat Wu 讨论 Anthropic 团队如何缩短产品开发周期，并说明 AI 时代 PM 角色与产品品味的重要性。",
    }


def render_source_digest_source_map_markdown(source_map: dict[str, Any]) -> str:
    rows = [
        ["full_source_in_payload", str(source_map.get("full_source_in_payload", False)).lower()],
        ["original_char_count", source_map.get("original_char_count", 0)],
        ["included_char_count", source_map.get("included_char_count", 0)],
        ["section_excerpt_limit", source_map.get("section_excerpt_limit", 0)],
        ["section_count", len(source_map.get("sections", []))],
        ["omitted_section_count", source_map.get("omitted_section_count", 0)],
        ["caption_count", len(source_map.get("captions", []))],
    ]
    section_rows = [
        [
            section.get("section_id", ""),
            section.get("line_start", ""),
            "#" * int(section.get("level", 0) or 0),
            section.get("heading", ""),
            section.get("original_char_count", 0),
            section.get("excerpt_char_count", 0),
            str(section.get("truncated", False)).lower(),
        ]
        for section in source_map.get("sections", [])
    ]
    return (
        "# Source Digest Source Map\n\n"
        f"- Approved prepared ref: `{source_map.get('approved_prepared_ref', '')}`\n\n"
        "## Payload Budget\n\n"
        f"{format_markdown_table(['字段', '值'], rows)}\n\n"
        "## Sections\n\n"
        f"{format_markdown_table(['ID', 'Line', 'Level', 'Heading', 'Original chars', 'Excerpt chars', 'Truncated'], section_rows)}\n"
    )


def project_source_digest_source_map_for_payload(source_map: dict[str, Any], *, full_source_map_ref: str) -> dict[str, Any]:
    return {
        "schema_version": "source_digest_source_map_payload.v1",
        "full_source_map_ref": full_source_map_ref,
        "approved_prepared_ref": source_map.get("approved_prepared_ref", ""),
        "full_source_in_payload": source_map.get("full_source_in_payload", False),
        "original_char_count": source_map.get("original_char_count", 0),
        "included_char_count": source_map.get("included_char_count", 0),
        "section_excerpt_limit": source_map.get("section_excerpt_limit", 0),
        "global_excerpt": source_map.get("global_excerpt", ""),
        "captions": source_map.get("captions", []),
        "sections": [
            {
                "id": section.get("section_id", ""),
                "heading": section.get("heading", ""),
                "level": section.get("level", 0),
                "line_start": section.get("line_start", 0),
                "original_char_count": section.get("original_char_count", 0),
                "excerpt": section.get("excerpt", ""),
            }
            for section in source_map.get("sections", [])
        ],
    }


def build_source_kind_hints(text: str, raw_rel: str) -> dict[str, Any]:
    lowered_path = raw_rel.lower()
    lowered_text = text.lower()
    readme_path = lowered_path.endswith("readme.md")
    toc_link_count = len(re.findall(r"\]\((?:\./)?(?:docs|chapter|chapters|extra-chapter|co-creation-projects)/", text, flags=re.IGNORECASE))
    markdown_link_count = len(re.findall(r"\[[^\]\n]+\]\([^)]+\)", text))
    badge_count = len(re.findall(r"shields\.io|badge|trendshift|github stars|github forks", lowered_text))
    download_marker_count = len(re.findall(r"下载|download|releases/latest|pdf", lowered_text))
    github_url_count = len(re.findall(r"https?://(?:www\.)?github\.com/|github\.com[:/]", lowered_text))
    heading_decoration = r"(?:[^\w\u4e00-\u9fff#\n]+)?\s*"
    contributor_heading_count = len(
        re.findall(
            rf"(?im)^\s{{0,3}}#{{1,6}}\s*{heading_decoration}(?:核心贡献者|贡献者|致谢|contributors?|acknowledg)",
            text,
        )
    )
    tutorial_heading_count = len(
        re.findall(
            rf"(?im)^\s{{0,3}}#{{1,6}}\s*{heading_decoration}(?:内容导航|目录|学习路线|快速开始|如何学习|课程|教程|chapters?|curriculum)",
            text,
        )
    )
    contributor_section_present = contributor_heading_count > 0
    badge_or_download_heavy = badge_count >= 3 or download_marker_count >= 3
    github_url_present = github_url_count > 0
    index_heading_present = tutorial_heading_count > 0
    tutorial_index = (toc_link_count >= 6 and (readme_path or index_heading_present)) or (
        index_heading_present and (readme_path or toc_link_count >= 3 or markdown_link_count >= 8)
    )
    navigation_heavy = (toc_link_count >= 8 and (readme_path or index_heading_present)) or (
        markdown_link_count >= 24 and (readme_path or index_heading_present or contributor_section_present or badge_or_download_heavy)
    )
    repository_readme = readme_path or (
        github_url_present
        and (badge_count > 0 or contributor_section_present)
        and (tutorial_index or markdown_link_count >= 12 or download_marker_count > 0)
    )
    flags = [
        name
        for name, enabled in [
            ("github_url_present", github_url_present),
            ("repository_readme", repository_readme),
            ("tutorial_index", tutorial_index),
            ("navigation_heavy", navigation_heavy),
            ("contributor_section_present", contributor_section_present),
            ("badge_or_download_heavy", badge_or_download_heavy),
        ]
        if enabled
    ]
    return {
        "schema_version": "source_kind_hints.v1",
        "source_raw_path": raw_rel,
        "flags": flags,
        "github_url_present": github_url_present,
        "repository_readme": repository_readme,
        "tutorial_index": tutorial_index,
        "navigation_heavy": navigation_heavy,
        "contributor_section_present": contributor_section_present,
        "badge_or_download_heavy": badge_or_download_heavy,
        "counts": {
            "toc_link_count": toc_link_count,
            "markdown_link_count": markdown_link_count,
            "badge_count": badge_count,
            "download_marker_count": download_marker_count,
            "github_url_count": github_url_count,
            "contributor_heading_count": contributor_heading_count,
            "tutorial_heading_count": tutorial_heading_count,
        },
        "guidance": [
            "README/index sources are entry pages; avoid turning badges, downloads, contributor lists, and TOC-only rows into formal pages.",
            "Keep formal candidates for durable project/framework entities, distinctive concepts, reusable designs, and comparisons with substantive source context.",
            "Move contributor/acknowledgement people to weak_or_noise_items unless the body gives reusable context beyond a name in a list.",
        ],
    }


def render_source_kind_hints_markdown(hints: dict[str, Any]) -> str:
    rows = [
        [name, str(bool(hints.get(name, False))).lower()]
        for name in [
            "repository_readme",
            "github_url_present",
            "tutorial_index",
            "navigation_heavy",
            "contributor_section_present",
            "badge_or_download_heavy",
        ]
    ]
    count_rows = [[key, value] for key, value in hints.get("counts", {}).items()]
    guidance_rows = [[item] for item in hints.get("guidance", [])]
    return (
        "# Source Kind Hints\n\n"
        f"- source：`{hints.get('source_raw_path', '')}`\n"
        f"- flags：{', '.join(f'`{item}`' for item in hints.get('flags', [])) or '无'}\n\n"
        "## Flags\n\n"
        f"{format_markdown_table(['flag', 'enabled'], rows)}\n\n"
        "## Counts\n\n"
        f"{format_markdown_table(['count', 'value'], count_rows)}\n\n"
        "## Guidance\n\n"
        f"{format_markdown_table(['rule'], guidance_rows)}\n"
    )


def build_source_digest_payload(
    *,
    raw_rel: str,
    approved_prepared_text: str,
    approved_prepared_ref: str,
    source_map_payload: dict[str, Any],
    source_kind_hints: dict[str, Any],
    profile: dict[str, Any],
    vault_config: VaultConfig,
) -> dict[str, Any]:
    return {
        "source_raw_path": raw_rel,
        "approved_prepared_markdown": approved_prepared_text if source_map_payload["full_source_in_payload"] else "",
        "approved_prepared_ref": approved_prepared_ref,
        "source_digest_source_map": source_map_payload,
        "source_kind_hints": source_kind_hints,
        "profile": profile,
        "language_contract": source_digest_language_contract(vault_config),
        "contract": {
            "goal": "Create a complete source digest of wiki-worthy candidates from this one raw file.",
            "candidate_fields": list(SourceDigestCandidate.model_fields),
            "weak_or_noise_fields": list(WeakOrNoiseItem.model_fields),
            "candidate_page_budget": vault_config.max_ingest_candidates,
            "rules": [
                "Return each candidate group as an array of candidate objects, never as bare fields.",
                "Every candidate object must include candidate_id, name, type, one_sentence_summary, why_matters, and wiki_value.",
                "For entities, concepts, designs, comparisons, and open_questions, suggested_page_title must be non-empty.",
                "Do not decide create, update, duplicate, or cross-reference actions in source_digest.",
                "Keep formal ingest candidates within candidate_page_budget; choose durable reusable themes over every subtopic.",
                "Prefer wiki-worthy candidates over every minor mention.",
                "Use source_locator as a lightweight review locator, not a strict evidence chain.",
                "Put weak or noisy mentions in weak_or_noise_items instead of creating pages for them.",
                "weak_or_noise_items are review-only and are not ingested as wiki pages.",
                "weak_or_noise_items may leave suggested_page_title empty and may use suggested_action='ignore'.",
                "weak_or_noise_items must use why_matters to explain why the mention was filtered.",
                "The vault language is zh-CN: write summary, key_takeaways, candidate summaries, why_matters, and wiki_value in Chinese.",
                "Stable domain terms such as Claude Code, RAG, PM, Workflow, Agent may stay in English, but explain them in Chinese when needed.",
                "Do not return whole English paragraphs for user-visible fields; zh-CN validation will fail instead of silently translating.",
                "If approved_prepared_markdown is empty, use source_digest_source_map sections, outline, captions, and approved_prepared_ref instead of assuming source content is absent.",
                "For long source-map payloads, choose durable candidates visible across the outline and section excerpts; do not create candidates from bibliography or appendix-only noise.",
                "Use section headings, line_start, and source_map section_id values as source_locator review handles when exact full source text is not in the payload.",
                "If source_kind_hints suggests a repository README, tutorial index, or navigation-heavy source, treat the file as an entry page rather than a chapter-by-chapter source.",
                "For README/index sources, do not create formal candidates for badges, status counters, install/download links, release links, table-of-contents rows, or chapter headings that only navigate elsewhere.",
                "For README/index sources, omit contributor/acknowledgement people or place them in weak_or_noise_items unless the person is central to the material and the body provides substantive reusable context beyond a contributor list.",
                "For README/index sources, prefer at most a few durable candidates: the core project/framework/entity, distinctive concepts, reusable designs, and comparisons with source-backed explanations.",
            ],
        },
    }
