from __future__ import annotations

from typing import Any

from . import source_digest_budget as _source_digest_budget
from . import source_excerpt as _source_excerpt
from . import source_refs as _source_refs
from .models import (
    SourceDigestArtifact,
    SourceDigestCandidate,
    WikiContextSnapshot,
    WikiMergePlanArtifact,
    WikiMergePlanItem,
)
from .system_pages import format_markdown_table


DRAFT_RENDERING_FULL_SOURCE_CHAR_LIMIT = 24_000
DRAFT_RENDERING_EXCERPT_TOTAL_CHAR_LIMIT = 24_000
DRAFT_RENDERING_EXCERPT_PER_PAGE_LIMIT = 1_600
DRAFT_RENDERING_GLOBAL_EXCERPT_LIMIT = 2_400
DRAFT_RENDERING_EXCERPT_MAX_SOURCE_RATIO = 0.55
DRAFT_RENDERING_EXCERPT_MIN_PAGE_CHARS = 600
DRAFT_RENDERING_CONTEXT_ENTRY_EXCERPT_LIMIT = 1_200


def build_draft_source_excerpt_pack(
    approved_prepared_text: str,
    digest: SourceDigestArtifact,
    merge_plan: WikiMergePlanArtifact,
    *,
    full_source_limit: int = DRAFT_RENDERING_FULL_SOURCE_CHAR_LIMIT,
    total_limit: int = DRAFT_RENDERING_EXCERPT_TOTAL_CHAR_LIMIT,
    per_page_limit: int = DRAFT_RENDERING_EXCERPT_PER_PAGE_LIMIT,
    global_limit: int = DRAFT_RENDERING_GLOBAL_EXCERPT_LIMIT,
    force_excerpt: bool = False,
) -> dict[str, Any]:
    include_full_source = len(approved_prepared_text) <= full_source_limit and not force_excerpt
    candidates = _source_refs.source_digest_candidate_lookup(digest)
    draftable_items = [item for item in merge_plan.items if item.action in {"create", "update"}]
    effective_total_limit = total_limit
    if not include_full_source:
        minimum_page_budget = global_limit + len(draftable_items) * DRAFT_RENDERING_EXCERPT_MIN_PAGE_CHARS
        ratio_budget = int(len(approved_prepared_text) * DRAFT_RENDERING_EXCERPT_MAX_SOURCE_RATIO)
        effective_total_limit = min(total_limit, max(minimum_page_budget, ratio_budget))
    items = []
    used_chars = 0
    global_excerpt = _source_excerpt.source_global_excerpt(approved_prepared_text, global_limit)
    used_chars += len(global_excerpt)
    for index, item in enumerate(draftable_items):
        direct_candidate_cues = []
        expanded_candidate_cues = []
        source_locators = [item.source_basis.source_locator]
        source_candidate_refs = _source_refs.source_basis_candidate_refs(item.source_basis)
        expanded_candidate_ids = _source_refs.source_digest_candidate_id_closure(source_candidate_refs, candidates)
        direct_candidate_ids = set(source_candidate_refs)
        for candidate_id in expanded_candidate_ids:
            candidate = candidates.get(candidate_id)
            if candidate is not None:
                source_locators.append(candidate.source_locator)
                target_cues = direct_candidate_cues if candidate_id in direct_candidate_ids else expanded_candidate_cues
                target_cues.extend(
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
        base_cues = [
            item.display_title,
            item.new_understanding,
            item.why_create_or_update,
            item.why_not_update,
            item.prior_knowledge_state,
            item.knowledge_delta,
            item.why_this_matters,
            *item.reuse_scenarios,
            *item.value_points,
            item.source_basis.source_locator,
            *item.section_plans.values(),
        ]
        remaining_budget = max(0, effective_total_limit - used_chars)
        remaining_items = max(1, len(draftable_items) - index)
        if include_full_source:
            page_limit = per_page_limit
        else:
            page_limit = min(per_page_limit, max(DRAFT_RENDERING_EXCERPT_MIN_PAGE_CHARS, remaining_budget // remaining_items))
        primary_cues = [*base_cues, *direct_candidate_cues, *item.source_basis.prepared_discovered_candidates]
        snippets = _source_excerpt.source_snippets_for_cues(approved_prepared_text, primary_cues, max_chars=page_limit)
        if _source_excerpt.source_snippets_are_start_fallback(snippets) and expanded_candidate_cues:
            snippets = _source_excerpt.source_snippets_for_cues(
                approved_prepared_text,
                [*primary_cues, *expanded_candidate_cues],
                max_chars=page_limit,
            )
        used_chars += sum(len(snippet["text"]) for snippet in snippets)
        items.append(
            {
                "page_plan_id": item.page_plan_id,
                "display_title": item.display_title,
                "target_path": item.canonical_target_path,
                "source_candidate_ids": item.source_basis.source_candidate_ids,
                "prepared_discovered_candidates": item.source_basis.prepared_discovered_candidates,
                "source_candidate_refs": source_candidate_refs,
                "expanded_source_candidate_ids": expanded_candidate_ids,
                "source_locators": [locator for locator in source_locators if locator],
                "snippets": snippets,
            }
        )
    included_chars = len(global_excerpt) + sum(
        len(snippet["text"])
        for item in items
        for snippet in item["snippets"]
    )
    return {
        "schema_version": "draft_source_excerpt_pack.v1",
        "source_raw_path": digest.source_raw_path,
        "original_char_count": len(approved_prepared_text),
        "included_char_count": included_chars,
        "full_source_in_payload": include_full_source,
        "force_excerpt": force_excerpt,
        "full_source_limit": full_source_limit,
        "configured_total_excerpt_limit": total_limit,
        "total_excerpt_limit": effective_total_limit,
        "per_page_excerpt_limit": per_page_limit,
        "global_excerpt_limit": global_limit,
        "max_source_ratio": DRAFT_RENDERING_EXCERPT_MAX_SOURCE_RATIO,
        "min_page_excerpt_chars": DRAFT_RENDERING_EXCERPT_MIN_PAGE_CHARS,
        "approved_prepared_ref": "prepared_raw_review/approved_prepared.md",
        "truncated_for_payload": not include_full_source,
        "global_excerpt": global_excerpt,
        "items": items,
    }


def render_draft_source_excerpt_pack_markdown(pack: dict[str, Any]) -> str:
    rows = []
    details: list[str] = []
    for item in pack.get("items", []):
        if not isinstance(item, dict):
            continue
        snippets = item.get("snippets", [])
        rows.append(
            [
                str(item.get("page_plan_id", "")),
                str(item.get("display_title", "")),
                str(item.get("target_path", "")),
                str(len(snippets) if isinstance(snippets, list) else 0),
                ", ".join(str(snippet.get("cue", "")) for snippet in snippets[:3] if isinstance(snippet, dict)) if isinstance(snippets, list) else "",
            ]
        )
        details.append(f"## {item.get('display_title', item.get('page_plan_id', ''))}\n")
        details.append(f"- 页面计划: `{item.get('page_plan_id', '')}`\n")
        locators = item.get("source_locators", [])
        if isinstance(locators, list) and locators:
            details.append(f"- 来源定位: {', '.join(str(locator) for locator in locators)}\n")
        if isinstance(snippets, list):
            for snippet_index, snippet in enumerate(snippets, start=1):
                if not isinstance(snippet, dict):
                    continue
                details.append(f"\n### Snippet {snippet_index}: {snippet.get('cue', '')}\n\n")
                details.append(_blockquote_markdown(str(snippet.get("text", ""))) + "\n")
    return (
        "# Draft Rendering Source Excerpt Pack\n\n"
        f"- 原始字符数：{pack.get('original_char_count', 0)}\n"
        f"- 纳入字符数：{pack.get('included_char_count', 0)}\n"
        f"- payload 是否包含完整 source：`{str(pack.get('full_source_in_payload', False)).lower()}`\n"
        f"- 是否强制使用 excerpt：`{str(pack.get('force_excerpt', False)).lower()}`\n"
        f"- 完整 source 阈值：{pack.get('full_source_limit', 0)}\n"
        f"- 配置总 excerpt 阈值：{pack.get('configured_total_excerpt_limit', pack.get('total_excerpt_limit', 0))}\n"
        f"- 生效总 excerpt 阈值：{pack.get('total_excerpt_limit', 0)}\n"
        f"- 最低单页 excerpt：{pack.get('min_page_excerpt_chars', 0)}\n"
        f"- 最大 source 比例：{pack.get('max_source_ratio', '')}\n\n"
        "## 页面摘录索引\n\n"
        f"{format_markdown_table(['页面计划', '标题', '目标', '片段数', '主要 cue'], rows)}\n\n"
        "## 全局摘录\n\n"
        f"{_blockquote_markdown(str(pack.get('global_excerpt', '')))}\n\n"
        "## 分页摘录\n\n"
        + "\n".join(details).rstrip()
        + "\n"
    )


def build_draft_rendering_payload(
    *,
    digest: SourceDigestArtifact,
    merge_plan: WikiMergePlanArtifact,
    snapshot: WikiContextSnapshot,
    source_excerpt_pack: dict[str, Any],
    update_preservation_pack: dict[str, Any],
    approved_prepared_text: str,
    profile_payload: dict[str, Any],
    language_contract: dict[str, Any],
    grounding_risk_rules: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    approved_prepared_payload = approved_prepared_text if source_excerpt_pack["full_source_in_payload"] else ""
    draftable_count = len([item for item in merge_plan.items if item.action in {"create", "update"}])
    projected_digest = project_source_digest_for_merge_plan(digest, merge_plan)
    projected_merge_plan = project_merge_plan_for_draft_rendering(merge_plan)
    required_items = [item for item in merge_plan.items if item.action in {"create", "update"}]
    snapshot_ref = merge_plan.context_snapshot_ref or "wiki_context_snapshot/wiki_context_snapshot.json"
    relevant_snapshot_paths, content_snapshot_paths = draft_rendering_relevant_wiki_paths(merge_plan)
    projected_snapshot = compact_snapshot_for_draft_rendering(
        snapshot,
        relevant_snapshot_paths,
        snapshot_ref,
        content_paths=content_snapshot_paths,
    )
    return {
        "approved_prepared_markdown": approved_prepared_payload,
        "approved_prepared_ref": source_excerpt_pack["approved_prepared_ref"],
        "source_excerpt_pack": source_excerpt_pack,
        "update_preservation_pack": update_preservation_pack,
        "approved_digest": projected_digest.model_dump(mode="json"),
        "approved_digest_ref": "source_digest_review/approved_digest.json",
        "approved_merge_plan": projected_merge_plan,
        "approved_merge_plan_ref": "merge_plan_review/approved_merge_plan.json",
        "wiki_context_snapshot": projected_snapshot,
        "wiki_context_snapshot_ref": snapshot_ref,
        "profile": profile_payload,
        "language_contract": language_contract,
        "required_page_plan_ids": [item.page_plan_id for item in required_items],
        "required_target_paths": [item.canonical_target_path for item in required_items],
        "contract": {
            "goal": "Generate a short summary, free-form core wiki body, and open questions for each create/update page from approved source excerpts/full source and frozen wiki context.",
            "batch_note": (
                f"This payload covers {draftable_count} create/update page(s). "
                "Return exactly those draftable pages and no pages from other batches."
            ),
            "preferred_page_shape": {
                "summary": "Required short Chinese summary for frontmatter/index.",
                "body_markdown": "Required core wiki content. The model owns headings, order, examples, boundaries, and explanatory structure.",
                "open_questions": "Optional contradictions, uncertainties, or 待补来源 questions.",
            },
            "rules": [
                "Return summary, body_markdown, and open_questions for each page.",
                "Return content only; do not include frontmatter, level-1 headings, source wikilinks, or full markdown pages.",
                "The output pages array must contain exactly required_page_plan_ids, one page per id, with no omissions, duplicates, or extra ids.",
                "body_markdown is the core output area. Write detailed, concrete Chinese wiki prose there with self-chosen Markdown subheadings; do not merely write a few vague sentences.",
                "Do not force content into fixed sections such as examples/value_points/additional_notes. If examples, boundaries, tradeoffs, mechanisms, or observations are useful, place them naturally inside body_markdown under headings you choose.",
                "Use open_questions only for real contradictions, uncertainties, or 待补来源 questions. If none are useful, return an empty string.",
                "Do not put `相关页面`/`Related Pages` blocks or self wikilinks inside body_markdown; the system renders official related pages separately.",
                "approved_digest, approved_merge_plan, and wiki_context_snapshot are compact projections for this draft batch; full reviewed artifacts are fixed by their *_ref fields for local audit and validators.",
                "For updates, read existing page excerpts from wiki_context_snapshot and update_preservation_pack, then produce a complete replacement core body that absorbs still-useful old knowledge naturally.",
                "Use source_excerpt_pack as the primary source support. If approved_prepared_markdown is empty, the full approved source is intentionally omitted from this model payload and remains available only to downstream validators through approved_prepared_ref.",
                "For updates, satisfy update_preservation_pack in the first draft: carry forward concept obligations "
                "and reusable key phrases into body_markdown, rewritten naturally with the new source "
                "rather than appended as a dump. change_summary may summarize retention but does not satisfy the obligation.",
                "Do not produce pages that are only source summaries; every body_markdown must include concrete digested understanding such as viewpoint, mechanism, example, use scenario, boundary condition, tradeoff, or value point.",
                "For updates, change_summary must explain what the new source adds, changes, clarifies, retains, or removes from the old understanding.",
                "If the merge plan has merged_page_plan_ids, absorb the unique section intent/examples/value points from suppressed candidates into the canonical page.",
                "Write all user-visible content in Chinese except stable domain terms with Chinese explanation when needed.",
                "For zh-CN vaults, translate or paraphrase English raw examples into Chinese; do not paste whole English sentences into body_markdown, open_questions, change_summary, or source_coverage_notes.",
                "Stable English product/protocol terms such as Claude Code, Managed Agents, harness, sandbox, session, MCP, Eval, TTFT, CLI, API, and Cowork may remain in English, but surrounding prose must be Chinese.",
                "Ground examples, value points, and reuse scenarios in source content.",
                "Across body_markdown/open_questions, prefer source-backed or clearly illustrative examples; when a concrete value such as `张三`, `Alice`, `user-123`, a preference, date, plan, metric, credential, or ID is not from the source/wiki, avoid presenting it as an observed user fact and use placeholders such as `<user_id>`, `<memory_text>`, `<memory_query>`, `某个用户`, or `用户偏好 X` when that preserves the meaning.",
                "When body_markdown includes examples, keep concrete user facts, user ids, preferences, dates, plans, metrics, credentials, and command arguments source-aware; for generic explanation, prefer abstract placeholders such as `某个用户`, `用户偏好 X`, `user_id`, `memory` or describe the pattern without quoted literals.",
                "The user has already approved this material for ingest; do not suppress content merely because it belongs to medical, legal, financial, security, account, password, payment, or privacy domains.",
                "Grounding should protect source fidelity, not make domain-risk judgments for the user. Keep domain-specific claims when they reflect the approved source or inspected wiki context.",
                "New named-entity relationships, releases, acquisitions, identity claims, causal facts, or concrete private facts should stay source-aware. Absence of support may be reported as a warning, but only contradiction with the approved source should create review.",
                "For CLI/API/code examples, Chinese surrounding explanation is fine, but command/API literal arguments are an explicit exception to the zh-CN translation rule: they must either copy exact source literals or use placeholders such as `<memory_text>`, `<user_id>`, or `<memory_query>`; do not translate a source literal into a new concrete preference, user id, query, path, or command argument.",
                "When the source only states a recommendation or best practice, do not invent causal outcomes with terms such as `导致`, `造成`, `影响到`, or `用户会...`; either state the source-backed boundary without a new consequence, or move the consequence to open_questions as 待补来源.",
                *grounding_risk_rules,
                "Do not write implementation details, examples, or claims as facts unless they are supported by source_excerpt_pack, approved_prepared_markdown, or inspected wiki context.",
                "If a useful detail is plausible but unsupported, put it under open_questions as 待补来源 instead of writing it as fact.",
                "source_coverage_notes must briefly say which source/wiki context supports the page and what was intentionally left uncertain.",
            ],
            "grounding_risk_rules": list(grounding_risk_rules),
        },
    }


def project_merge_plan_for_draft_rendering(merge_plan: WikiMergePlanArtifact) -> dict[str, Any]:
    draftable_items = [item for item in merge_plan.items if item.action in {"create", "update"}]
    projected_items = [_project_merge_plan_item_for_draft_rendering(item) for item in draftable_items]
    return {
        "schema_version": "draft_merge_plan_projection.v1",
        "source_schema_version": merge_plan.schema_version,
        "full_merge_plan_ref": "merge_plan_review/approved_merge_plan.json",
        "context_snapshot_ref": merge_plan.context_snapshot_ref,
        "log_date": merge_plan.log_date,
        "draftable_item_count": len(draftable_items),
        "omitted_non_draft_item_count": max(0, len(merge_plan.items) - len(draftable_items)),
        "items": projected_items,
    }


def draft_rendering_relevant_wiki_paths(merge_plan: WikiMergePlanArtifact) -> tuple[set[str], set[str]]:
    metadata_paths: set[str] = set()
    content_paths: set[str] = set()
    for item in merge_plan.items:
        if item.action not in {"create", "update"}:
            continue
        target_paths = [
            item.canonical_target_path,
            item.matched_page or "",
        ]
        context_paths = [related.target_path for related in item.related_pages]
        if should_include_draft_inspected_context(item):
            context_paths.extend([item.strongest_overlap.path, *item.inspected_context_paths])
        for raw_path in [*target_paths, *context_paths]:
            normalized = _normalize_wiki_snapshot_path(raw_path)
            if normalized:
                metadata_paths.add(normalized)
        if item.action == "update":
            for raw_path in target_paths:
                normalized = _normalize_wiki_snapshot_path(raw_path)
                if normalized:
                    content_paths.add(normalized)
        elif item.strongest_overlap.strength == "strong":
            normalized = _normalize_wiki_snapshot_path(item.strongest_overlap.path)
            if normalized:
                content_paths.add(normalized)
    return metadata_paths, content_paths


def should_include_draft_inspected_context(item: WikiMergePlanItem) -> bool:
    if item.action == "update":
        return True
    return item.strongest_overlap.strength in {"medium", "strong"}


def compact_snapshot_for_draft_rendering(
    snapshot: WikiContextSnapshot,
    relevant_paths: set[str],
    snapshot_ref: str,
    *,
    content_paths: set[str] | None = None,
) -> dict[str, Any]:
    content_paths = content_paths or set()
    entries: list[dict[str, Any]] = []
    content_entry_count = 0
    for entry in snapshot.entries:
        if entry.path not in relevant_paths:
            continue
        include_content = entry.path in content_paths
        content_excerpt = _source_excerpt.source_global_excerpt(entry.content, DRAFT_RENDERING_CONTEXT_ENTRY_EXCERPT_LIMIT) if include_content else ""
        if content_excerpt:
            content_entry_count += 1
        entries.append(
            {
                "path": entry.path,
                "expected_state": entry.expected_state,
                "preimage_sha256": entry.preimage_sha256,
                "metadata": entry.metadata.model_dump(mode="json") if entry.metadata else None,
                "content_excerpt": content_excerpt,
                "content_truncated": include_content and len(entry.content.strip()) > len(content_excerpt),
                "content_role": "draft_context" if include_content else "metadata_only",
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
    return {
        "schema_version": "wiki_context_snapshot_projection.v1",
        "source_schema_version": snapshot.schema_version,
        "full_snapshot_ref": snapshot_ref,
        "log_date": snapshot.log_date,
        "source_target_path": snapshot.source_target_path,
        "candidate_contexts_ref": snapshot.candidate_contexts_ref,
        "candidate_pool_sha256": snapshot.candidate_pool_sha256,
        "entry_excerpt_limit": DRAFT_RENDERING_CONTEXT_ENTRY_EXCERPT_LIMIT,
        "full_entry_count": len(snapshot.entries),
        "included_entry_count": len(entries),
        "included_content_entry_count": content_entry_count,
        "included_entry_content_chars": sum(len(entry["content_excerpt"]) for entry in entries),
        "omitted_entry_count": max(0, len(snapshot.entries) - len(entries)),
        "full_metadata_pool_count": len(snapshot.knowledge_metadata_pool),
        "included_metadata_pool_count": len(metadata_pool),
        "knowledge_metadata_pool": metadata_pool,
        "entries": entries,
    }


def project_source_digest_for_merge_plan(
    digest: SourceDigestArtifact,
    merge_plan: WikiMergePlanArtifact,
) -> SourceDigestArtifact:
    needed_ids = _source_digest_candidate_ids_for_merge_plan(digest, merge_plan)
    projected_groups: dict[str, list[SourceDigestCandidate]] = {}
    for group_name in _source_digest_budget.SOURCE_DIGEST_BUDGET_GROUP_ORDER:
        candidates = list(getattr(digest, group_name))
        projected_groups[group_name] = [candidate for candidate in candidates if candidate.candidate_id in needed_ids]
    projected_deferred = [
        candidate
        for candidate in digest.budget_deferred_candidates
        if candidate.candidate_id in needed_ids
    ]
    return digest.model_copy(
        update={
            **projected_groups,
            "budget_deferred_candidates": projected_deferred,
            "weak_or_noise_items": [],
        }
    )


def _project_merge_plan_item_for_draft_rendering(item: WikiMergePlanItem) -> dict[str, Any]:
    strongest_overlap = _compact_optional_dict(
        {
            "strength": item.strongest_overlap.strength,
            "match_basis": item.strongest_overlap.match_basis,
            "path": item.strongest_overlap.path,
            "score": item.strongest_overlap.score,
            "reason": item.strongest_overlap.reason,
        }
    )
    related_pages = [
        _compact_optional_dict(
            {
                "target_path": related.target_path,
                "display_title": related.display_title,
                "source": related.source,
                "reason": related.reason,
            }
        )
        for related in item.related_pages
    ]
    return _compact_optional_dict(
        {
            "page_plan_id": item.page_plan_id,
            "source_basis": _compact_optional_dict(item.source_basis.model_dump(mode="json")),
            "action": item.action,
            "canonical_target_path": item.canonical_target_path,
            "display_title": item.display_title,
            "page_type": item.page_type,
            "matched_page": item.matched_page,
            "inspected_context_paths": item.inspected_context_paths,
            "strongest_overlap": strongest_overlap,
            "why_not_update": item.why_not_update,
            "why_create_or_update": item.why_create_or_update,
            "prior_knowledge_state": item.prior_knowledge_state,
            "new_understanding": item.new_understanding,
            "changed_view": item.changed_view,
            "knowledge_delta": item.knowledge_delta,
            "why_this_matters": item.why_this_matters,
            "reuse_scenarios": item.reuse_scenarios,
            "value_points": item.value_points,
            "section_plans": item.section_plans,
            "related_pages": related_pages,
            "related_absence_reason": item.related_absence_reason,
            "related_unresolved": item.related_unresolved,
            "unresolved_related": item.unresolved_related,
            "conflicts": item.conflicts,
            "uncertainties": item.uncertainties,
            "quality_risks": item.quality_risks,
            "reason": item.reason,
            "merged_page_plan_ids": item.merged_page_plan_ids,
            "merge_reason": item.merge_reason,
        }
    )


def _source_digest_candidate_ids_for_merge_plan(
    digest: SourceDigestArtifact,
    merge_plan: WikiMergePlanArtifact,
) -> set[str]:
    candidate_by_id = _source_refs.source_digest_candidate_lookup(digest)
    needed: set[str] = set()
    for item in merge_plan.items:
        if item.action not in {"create", "update"}:
            continue
        needed.update(
            _source_refs.source_digest_candidate_id_closure(
                _source_refs.source_basis_candidate_refs(item.source_basis),
                candidate_by_id,
            )
        )
    return needed


def _blockquote_markdown(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return ">"
    return "\n".join(f"> {line}" if line else ">" for line in lines)


def _normalize_wiki_snapshot_path(path: str) -> str:
    cleaned = path.strip().lstrip("/")
    if not cleaned:
        return ""
    if cleaned.startswith("wiki/"):
        return cleaned
    return f"wiki/{cleaned}"


def _compact_optional_dict(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if item is not None and item != "" and item != [] and item != {}
    }
