from __future__ import annotations

import json
from typing import Any

from . import markdown_utils as _markdown_utils
from . import source_digest_budget as _source_digest_budget
from . import source_excerpt as _source_excerpt
from . import source_refs as _source_refs
from .models import (
    CandidateContextHit,
    CandidateContextsArtifact,
    CandidateResolutionArtifact,
    SourceDigestArtifact,
    SourceDigestCandidate,
    WikiContextSnapshot,
)
from .system_pages import format_markdown_table


CANDIDATE_RESOLUTION_FULL_SOURCE_CHAR_LIMIT = 20_000
CANDIDATE_RESOLUTION_GLOBAL_EXCERPT_LIMIT = 1_600
CANDIDATE_RESOLUTION_PER_CANDIDATE_EXCERPT_LIMIT = 500
MERGE_PLANNING_FULL_SOURCE_CHAR_LIMIT = 16_000
MERGE_PLANNING_SOURCE_GLOBAL_EXCERPT_LIMIT = 1_600
MERGE_PLANNING_SOURCE_PER_PAGE_EXCERPT_LIMIT = 700
MERGE_PLANNING_CONTEXT_HIT_EXCERPT_LIMIT = 240
MERGE_PLANNING_WEAK_CONTEXT_HIT_EXCERPT_LIMIT = 120
MERGE_PLANNING_WEAK_CONTEXT_EXCERPT_MAX_RANK = 2
MERGE_PLANNING_CONTEXT_QUERY_LIMIT = 420
MERGE_PLANNING_ENTRY_EXCERPT_LIMIT = 900


def build_candidate_resolution_source_pack(
    approved_prepared_text: str,
    digest: SourceDigestArtifact,
    *,
    full_source_limit: int = CANDIDATE_RESOLUTION_FULL_SOURCE_CHAR_LIMIT,
    global_limit: int = CANDIDATE_RESOLUTION_GLOBAL_EXCERPT_LIMIT,
    per_candidate_limit: int = CANDIDATE_RESOLUTION_PER_CANDIDATE_EXCERPT_LIMIT,
) -> dict[str, Any]:
    include_full_source = len(approved_prepared_text) <= full_source_limit
    global_excerpt = "" if include_full_source else _source_excerpt.source_global_excerpt(approved_prepared_text, global_limit)
    items: list[dict[str, Any]] = []
    included_chars = len(global_excerpt)
    for group_name in _source_digest_budget.SOURCE_DIGEST_BUDGET_GROUP_ORDER:
        if group_name in {"budget_deferred_candidates", "weak_or_noise_items"}:
            continue
        for candidate in getattr(digest, group_name):
            cues = [
                candidate.name,
                candidate.suggested_page_title,
                candidate.one_sentence_summary,
                candidate.why_matters,
                candidate.wiki_value,
                candidate.source_locator,
                candidate.open_question_or_tension,
            ]
            snippets = (
                []
                if include_full_source
                else _source_excerpt.source_snippets_for_cues(
                    approved_prepared_text,
                    cues,
                    max_chars=per_candidate_limit,
                )
            )
            included_chars += sum(len(snippet["text"]) for snippet in snippets)
            items.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "group": group_name,
                    "name": candidate.name,
                    "suggested_page_title": candidate.suggested_page_title,
                    "source_locator": candidate.source_locator,
                    "snippets": snippets,
                }
            )
    return {
        "schema_version": "candidate_resolution_source_excerpt_pack.v1",
        "source_raw_path": digest.source_raw_path,
        "approved_prepared_ref": "prepared_raw_review/approved_prepared.md",
        "original_char_count": len(approved_prepared_text),
        "included_char_count": included_chars,
        "full_source_in_payload": include_full_source,
        "full_source_limit": full_source_limit,
        "global_excerpt_limit": global_limit,
        "per_candidate_excerpt_limit": per_candidate_limit,
        "candidate_count": len(items),
        "global_excerpt": global_excerpt,
        "items": items,
    }


def render_candidate_resolution_source_pack_markdown(pack: dict[str, Any]) -> str:
    rows = [
        [
            item.get("candidate_id", ""),
            item.get("group", ""),
            item.get("name", ""),
            len(item.get("snippets", [])) if isinstance(item.get("snippets", []), list) else 0,
            item.get("source_locator", ""),
        ]
        for item in pack.get("items", [])
        if isinstance(item, dict)
    ]
    return (
        "# Candidate Resolution Source Excerpt Pack\n\n"
        f"- payload 是否包含完整 source：`{str(pack.get('full_source_in_payload', False)).lower()}`\n"
        f"- 原始 source 字符：{pack.get('original_char_count', 0)}\n"
        f"- 入模摘录字符：{pack.get('included_char_count', 0)}\n"
        f"- 完整 source 引用：`{pack.get('approved_prepared_ref', '')}`\n\n"
        "## Candidate Snippets\n\n"
        f"{format_markdown_table(['Candidate', 'Group', 'Name', 'Snippets', 'Locator'], rows) if rows else '_无 candidate snippets。_'}\n"
    )


def build_merge_planning_context_pack(
    *,
    approved_prepared_text: str,
    digest: SourceDigestArtifact,
    resolution: CandidateResolutionArtifact,
    snapshot: WikiContextSnapshot,
    candidate_contexts: CandidateContextsArtifact,
    snapshot_ref: str,
) -> dict[str, Any]:
    candidate_by_id = _source_refs.source_digest_candidate_lookup(digest)
    include_full_source = len(approved_prepared_text) <= MERGE_PLANNING_FULL_SOURCE_CHAR_LIMIT
    source_pack = build_merge_planning_source_pack(
        approved_prepared_text,
        resolution,
        candidate_by_id,
        include_full_source=include_full_source,
    )
    candidate_contexts_projection = compact_candidate_contexts_for_merge_planning(candidate_contexts)
    relevant_paths = merge_planning_relevant_wiki_paths(resolution, candidate_contexts)
    snapshot_projection = compact_snapshot_for_merge_planning(snapshot, relevant_paths, snapshot_ref)
    original_counts = {
        "approved_prepared_markdown_chars": len(approved_prepared_text),
        "approved_digest_json_chars": json_char_count(digest.model_dump(mode="json")),
        "candidate_resolution_json_chars": json_char_count(resolution.model_dump(mode="json")),
        "wiki_context_snapshot_json_chars": json_char_count(snapshot.model_dump(mode="json")),
        "candidate_contexts_json_chars": json_char_count(candidate_contexts.model_dump(mode="json")),
        "snapshot_entries_chars": sum(len(entry.content) for entry in snapshot.entries),
        "snapshot_entry_count": len(snapshot.entries),
        "knowledge_metadata_pool_count": len(snapshot.knowledge_metadata_pool),
    }
    projected_counts = {
        "approved_prepared_markdown_chars": len(approved_prepared_text) if include_full_source else 0,
        "source_excerpt_chars": source_pack["included_char_count"],
        "wiki_context_projection_json_chars": json_char_count(snapshot_projection),
        "candidate_contexts_projection_json_chars": json_char_count(candidate_contexts_projection),
        "projected_entry_chars": snapshot_projection["included_entry_content_chars"],
        "projected_entry_count": len(snapshot_projection["entries"]),
        "projected_metadata_count": len(snapshot_projection["knowledge_metadata_pool"]),
    }
    return {
        "schema_version": "merge_planning_context_pack.v1",
        "approved_prepared_ref": "prepared_raw_review/approved_prepared.md",
        "approved_digest_ref": "source_digest_review/approved_digest.json",
        "candidate_resolution_ref": "candidate_resolution/candidate_resolution.json",
        "wiki_context_snapshot_ref": snapshot_ref,
        "candidate_contexts_ref": snapshot.candidate_contexts_ref,
        "full_source_in_payload": include_full_source,
        "source_excerpt_pack": source_pack,
        "wiki_context_projection": snapshot_projection,
        "candidate_contexts_projection": candidate_contexts_projection,
        "original_counts": original_counts,
        "projected_counts": projected_counts,
    }


def build_merge_planning_source_pack(
    approved_prepared_text: str,
    resolution: CandidateResolutionArtifact,
    candidate_by_id: dict[str, SourceDigestCandidate],
    *,
    include_full_source: bool,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    included_chars = 0
    global_excerpt = "" if include_full_source else _source_excerpt.source_global_excerpt(approved_prepared_text, MERGE_PLANNING_SOURCE_GLOBAL_EXCERPT_LIMIT)
    included_chars += len(global_excerpt)
    for item in resolution.items:
        candidate_cues: list[str] = []
        source_locators = [item.source_basis.source_locator]
        candidate_refs = _source_refs.source_basis_candidate_refs(item.source_basis)
        for candidate_id in candidate_refs:
            candidate = candidate_by_id.get(candidate_id)
            if candidate is None:
                continue
            source_locators.append(candidate.source_locator)
            candidate_cues.extend(
                [
                    candidate.name,
                    candidate.suggested_page_title,
                    candidate.one_sentence_summary,
                    candidate.why_matters,
                    candidate.wiki_value,
                    candidate.source_locator,
                    candidate.open_question_or_tension,
                ]
            )
        cues = [
            item.display_title,
            item.topic_summary,
            item.why_this_page,
            item.initial_section_intent,
            item.coverage_notes,
            item.reason,
            item.source_basis.source_locator,
            *candidate_cues,
            *item.source_basis.prepared_discovered_candidates,
        ]
        snippets = (
            []
            if include_full_source
            else _source_excerpt.source_snippets_for_cues(
                approved_prepared_text,
                cues,
                max_chars=MERGE_PLANNING_SOURCE_PER_PAGE_EXCERPT_LIMIT,
            )
        )
        included_chars += sum(len(snippet["text"]) for snippet in snippets)
        items.append(
            {
                "page_plan_id": item.page_plan_id,
                "display_title": item.display_title,
                "candidate_target_path": item.candidate_target_path,
                "source_candidate_ids": item.source_basis.source_candidate_ids,
                "prepared_discovered_candidates": item.source_basis.prepared_discovered_candidates,
                "source_candidate_refs": candidate_refs,
                "source_locators": [locator for locator in source_locators if locator],
                "snippets": snippets,
            }
        )
    return {
        "schema_version": "merge_planning_source_excerpt_pack.v1",
        "original_char_count": len(approved_prepared_text),
        "included_char_count": included_chars,
        "full_source_in_payload": include_full_source,
        "full_source_limit": MERGE_PLANNING_FULL_SOURCE_CHAR_LIMIT,
        "global_excerpt_limit": MERGE_PLANNING_SOURCE_GLOBAL_EXCERPT_LIMIT,
        "per_page_excerpt_limit": MERGE_PLANNING_SOURCE_PER_PAGE_EXCERPT_LIMIT,
        "global_excerpt": global_excerpt,
        "items": items,
    }


def compact_candidate_contexts_for_merge_planning(candidate_contexts: CandidateContextsArtifact) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for item in candidate_contexts.items:
        hits: list[dict[str, Any]] = []
        for hit in item.hits:
            excerpt_limit = merge_planning_hit_excerpt_limit(hit)
            excerpt = _markdown_utils.compact_payload_text(hit.excerpt, excerpt_limit)
            hits.append(
                {
                    "page_plan_id": hit.page_plan_id,
                    "rank": hit.rank,
                    "path": hit.path,
                    "display_title": hit.display_title,
                    "score": hit.score,
                    "score_bucket": hit.score_bucket,
                    "strength": hit.strength,
                    "match_basis": hit.match_basis,
                    "sort_explanation": hit.sort_explanation,
                    "forced": hit.forced,
                    "page_sha256": hit.page_sha256,
                    "excerpt_limit": excerpt_limit,
                    "excerpt": excerpt,
                    "truncated": hit.truncated or len(hit.excerpt) > len(excerpt),
                }
            )
        items.append(
            {
                "page_plan_id": item.page_plan_id,
                "query": _markdown_utils.compact_payload_text(item.query, MERGE_PLANNING_CONTEXT_QUERY_LIMIT),
                "hits": hits,
                "unindexable_pages": item.unindexable_pages,
            }
        )
    return {
        "schema_version": "candidate_contexts_projection.v1",
        "source_schema_version": candidate_contexts.schema_version,
        "retrieval_backend": candidate_contexts.retrieval_backend,
        "model": candidate_contexts.model,
        "model_revision": candidate_contexts.model_revision,
        "top_k": candidate_contexts.top_k,
        "candidate_pool_size": candidate_contexts.candidate_pool_size,
        "candidate_pool_sha256": candidate_contexts.candidate_pool_sha256,
        "skipped_count": candidate_contexts.skipped_count,
        "warnings": candidate_contexts.warnings,
        "query_limit": MERGE_PLANNING_CONTEXT_QUERY_LIMIT,
        "hit_excerpt_limit": MERGE_PLANNING_CONTEXT_HIT_EXCERPT_LIMIT,
        "weak_hit_excerpt_limit": MERGE_PLANNING_WEAK_CONTEXT_HIT_EXCERPT_LIMIT,
        "weak_hit_excerpt_max_rank": MERGE_PLANNING_WEAK_CONTEXT_EXCERPT_MAX_RANK,
        "hit_excerpt_role": "match_preview",
        "content_evidence_ref": "wiki_context_projection.entries",
        "items": items,
    }


def merge_planning_hit_excerpt_limit(hit: CandidateContextHit) -> int:
    if hit.strength == "weak" and not hit.forced:
        if hit.rank > MERGE_PLANNING_WEAK_CONTEXT_EXCERPT_MAX_RANK:
            return 0
        return MERGE_PLANNING_WEAK_CONTEXT_HIT_EXCERPT_LIMIT
    return MERGE_PLANNING_CONTEXT_HIT_EXCERPT_LIMIT


def merge_planning_relevant_wiki_paths(
    resolution: CandidateResolutionArtifact,
    candidate_contexts: CandidateContextsArtifact,
) -> set[str]:
    paths = {f"wiki/{item.candidate_target_path}" for item in resolution.items if item.candidate_target_path}
    for context_item in candidate_contexts.items:
        for hit in context_item.hits:
            paths.add(f"wiki/{hit.path}")
    return paths


def compact_snapshot_for_merge_planning(
    snapshot: WikiContextSnapshot,
    relevant_paths: set[str],
    snapshot_ref: str,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for entry in snapshot.entries:
        if entry.path not in relevant_paths:
            continue
        content_excerpt = _source_excerpt.source_global_excerpt(
            entry.content,
            MERGE_PLANNING_ENTRY_EXCERPT_LIMIT,
        )
        entries.append(
            {
                "path": entry.path,
                "expected_state": entry.expected_state,
                "preimage_sha256": entry.preimage_sha256,
                "metadata": entry.metadata.model_dump(mode="json") if entry.metadata else None,
                "content_excerpt": content_excerpt,
                "content_truncated": len(entry.content.strip()) > len(content_excerpt),
            }
        )
    metadata_paths = {path.removeprefix("wiki/") for path in relevant_paths}
    metadata_pool = [
        {
            "path": pool_entry.path,
            "rel_path": pool_entry.rel_path,
            "preimage_sha256": pool_entry.preimage_sha256,
            "metadata": pool_entry.metadata.model_dump(mode="json") if pool_entry.metadata else None,
            "display_title": pool_entry.display_title,
            "summary": pool_entry.summary,
            "aliases": pool_entry.aliases,
            "llmwiki_type": pool_entry.llmwiki_type,
            "indexable": pool_entry.indexable,
            "unindexable_reason": pool_entry.unindexable_reason,
        }
        for pool_entry in snapshot.knowledge_metadata_pool
        if pool_entry.path in metadata_paths
    ]
    included_chars = sum(len(entry["content_excerpt"]) for entry in entries)
    return {
        "schema_version": "wiki_context_snapshot_projection.v1",
        "source_schema_version": snapshot.schema_version,
        "full_snapshot_ref": snapshot_ref,
        "log_date": snapshot.log_date,
        "source_target_path": snapshot.source_target_path,
        "candidate_contexts_ref": snapshot.candidate_contexts_ref,
        "candidate_pool_sha256": snapshot.candidate_pool_sha256,
        "entry_excerpt_limit": MERGE_PLANNING_ENTRY_EXCERPT_LIMIT,
        "full_entry_count": len(snapshot.entries),
        "included_entry_count": len(entries),
        "included_entry_content_chars": included_chars,
        "omitted_entry_count": max(0, len(snapshot.entries) - len(entries)),
        "full_metadata_pool_count": len(snapshot.knowledge_metadata_pool),
        "included_metadata_pool_count": len(metadata_pool),
        "knowledge_metadata_pool": metadata_pool,
        "entries": entries,
    }


def json_char_count(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False))


def merge_planning_payload_pack_summary(pack: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in pack.items()
        if key
        not in {
            "source_excerpt_pack",
            "wiki_context_projection",
            "candidate_contexts_projection",
        }
    }


def render_merge_planning_context_pack_markdown(pack: dict[str, Any]) -> str:
    source_pack = pack.get("source_excerpt_pack", {})
    original = pack.get("original_counts", {})
    projected = pack.get("projected_counts", {})
    rows = [
        ["approved_prepared_markdown", original.get("approved_prepared_markdown_chars", 0), projected.get("approved_prepared_markdown_chars", 0)],
        ["source_excerpt_pack", 0, projected.get("source_excerpt_chars", 0)],
        ["wiki_context_snapshot", original.get("wiki_context_snapshot_json_chars", 0), projected.get("wiki_context_projection_json_chars", 0)],
        ["candidate_contexts", original.get("candidate_contexts_json_chars", 0), projected.get("candidate_contexts_projection_json_chars", 0)],
        ["snapshot entries", original.get("snapshot_entries_chars", 0), projected.get("projected_entry_chars", 0)],
    ]
    context_rows = [
        [
            item.get("page_plan_id", ""),
            item.get("display_title", ""),
            item.get("candidate_target_path", ""),
            len(item.get("snippets", [])) if isinstance(item.get("snippets", []), list) else 0,
        ]
        for item in source_pack.get("items", [])
        if isinstance(item, dict)
    ]
    return (
        "# Merge Planning Context Pack\n\n"
        f"- payload 是否包含完整 source：`{str(pack.get('full_source_in_payload', False)).lower()}`\n"
        f"- 完整 source 引用：`{pack.get('approved_prepared_ref', '')}`\n"
        f"- 完整 snapshot 引用：`{pack.get('wiki_context_snapshot_ref', '')}`\n\n"
        "## Payload 字符预算\n\n"
        f"{format_markdown_table(['对象', '原始字符', '投影字符'], rows)}\n\n"
        "## Source 页面摘录\n\n"
        f"{format_markdown_table(['页面计划', '标题', '目标', '片段数'], context_rows)}\n"
    )
