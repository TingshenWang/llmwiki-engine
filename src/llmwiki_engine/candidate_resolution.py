from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from . import markdown_utils as _markdown_utils
from . import wiki_markup as _wiki_markup
from .hash_utils import sha256_bytes
from .models import (
    CandidateResolutionArtifact,
    CandidateResolutionItem,
    SourceBasis,
    SourceDigestArtifact,
    SourceDigestCandidate,
    StructuredIssue,
)
from .profiles import page_output_path
from .system_pages import format_markdown_table
from .validators import ValidationError as ContractValidationError


def backfill_missing_candidate_resolution_items(
    artifact: CandidateResolutionArtifact,
    digest: SourceDigestArtifact,
    profile: Any,
) -> CandidateResolutionArtifact:
    covered_ids: set[str] = set()
    for item in artifact.items:
        covered_ids.update(item.source_basis.source_candidate_ids)
    additions: list[CandidateResolutionItem] = []
    notes = list(artifact.missed_candidate_risks)
    for group_name, candidates in [
        ("entities", digest.entities),
        ("concepts", digest.concepts),
        ("designs", digest.designs),
        ("comparisons", digest.comparisons),
        ("open_questions", digest.open_questions),
    ]:
        for candidate in candidates:
            if candidate.candidate_id in covered_ids:
                continue
            page_type = page_type_for_digest_candidate(group_name, candidate, profile)
            display_title = candidate.suggested_page_title.strip() or candidate.name.strip() or candidate.candidate_id
            additions.append(
                CandidateResolutionItem(
                    source_basis=SourceBasis(
                        source_candidate_ids=[candidate.candidate_id],
                        source_locator=candidate.source_locator,
                    ),
                    page_type=page_type,
                    display_title=display_title,
                    topic_summary=candidate.one_sentence_summary,
                    why_this_page=candidate.wiki_value or candidate.why_matters or "该候选来自 source_digest，模型在页面规划中遗漏，系统补齐为最小页面计划。",
                    initial_section_intent="系统补齐的最小页面计划；后续 merge planning / draft rendering 需要重新对照全文消化。",
                    coverage_notes=f"模型遗漏 approved_digest candidate `{candidate.candidate_id}`，系统已补齐。",
                    reason="candidate_resolution coverage backfill",
                )
            )
            notes.append(f"candidate_resolution model missed `{candidate.candidate_id}`; deterministic backfill added `{display_title}`.")
    if not additions:
        return artifact
    return CandidateResolutionArtifact(items=[*artifact.items, *additions], missed_candidate_risks=notes)


def page_type_for_digest_candidate(group_name: str, candidate: SourceDigestCandidate, profile: Any) -> str:
    normalized = candidate.type.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "question": "open_question",
        "open_questions": "open_question",
        "open_question": "open_question",
        "concept_overview": "overview",
        "design_overview": "overview",
        "entity_index": "overview",
        "open_question_overview": "open_question",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in profile.page_types and normalized != profile.source_page_type:
        return normalized
    preferred = {
        "entities": "entity",
        "concepts": "concept",
        "designs": "design",
        "comparisons": "comparison",
        "open_questions": "open_question",
    }.get(group_name)
    if preferred in profile.page_types:
        return preferred
    return profile.default_page_type


def finalize_candidate_resolution(
    vault: Path,
    profile: Any,
    artifact: CandidateResolutionArtifact,
    digest: SourceDigestArtifact | None = None,
) -> CandidateResolutionArtifact:
    items: list[CandidateResolutionItem] = []
    seen_paths: dict[str, int] = {}
    selected_candidate_ids = {candidate.candidate_id for candidate in digest.ingest_candidates()} if digest is not None else set()
    weak_or_noise_ids = {candidate.candidate_id for candidate in digest.weak_or_noise_items} if digest is not None else set()
    sanitized_items: list[CandidateResolutionItem] = []
    sanitation_notes = list(artifact.missed_candidate_risks)
    for index, item in enumerate(artifact.items):
        source_ids = list(item.source_basis.source_candidate_ids)
        leaked_ids = [
            candidate_id
            for candidate_id in source_ids
            if candidate_id in weak_or_noise_ids or candidate_id.strip().lower().startswith(("noise", "weak", "ignore"))
        ]
        if leaked_ids:
            cleaned_ids = [candidate_id for candidate_id in source_ids if candidate_id not in leaked_ids]
            if not cleaned_ids:
                sanitation_notes.append(
                    f"candidate_resolution item `{item.display_title or item.page_plan_id or index}` dropped because it only referenced weak/noise candidates: {sorted(leaked_ids)}."
                )
                continue
            item = item.model_copy(
                update={
                    "source_basis": item.source_basis.model_copy(update={"source_candidate_ids": cleaned_ids}),
                    "coverage_notes": _markdown_utils.merge_markdown_blocks(
                        item.coverage_notes,
                        f"系统清理 weak/noise candidate 引用：{', '.join(f'`{candidate_id}`' for candidate_id in leaked_ids)}。",
                    ),
                }
            )
            sanitation_notes.append(
                f"candidate_resolution item `{item.display_title or item.page_plan_id or index}` removed weak/noise candidate refs: {sorted(leaked_ids)}."
            )
        if digest is not None:
            unknown_source_ids = [
                candidate_id
                for candidate_id in item.source_basis.source_candidate_ids
                if candidate_id not in selected_candidate_ids
            ]
            if unknown_source_ids:
                prepared_discovered = list(item.source_basis.prepared_discovered_candidates)
                cleaned_ids = [
                    candidate_id
                    for candidate_id in item.source_basis.source_candidate_ids
                    if candidate_id in selected_candidate_ids
                ]
                if prepared_discovered:
                    for candidate_id in unknown_source_ids:
                        if candidate_id not in prepared_discovered:
                            prepared_discovered.append(candidate_id)
                    unknown_note = "系统将非 source_digest candidate id 移入 prepared_discovered_candidates："
                    sanitation_note = "moved unknown candidate refs to prepared_discovered_candidates"
                else:
                    prepared_discovered = list(unknown_source_ids)
                    unknown_note = "系统将非 source_digest candidate id 移入 prepared_discovered_candidates："
                    sanitation_note = "moved unknown-only candidate refs to prepared_discovered_candidates"
                item = item.model_copy(
                    update={
                        "source_basis": item.source_basis.model_copy(
                            update={
                                "source_candidate_ids": cleaned_ids,
                                "prepared_discovered_candidates": prepared_discovered,
                            }
                        ),
                        "coverage_notes": _markdown_utils.merge_markdown_blocks(
                            item.coverage_notes,
                            unknown_note + f"{', '.join(f'`{candidate_id}`' for candidate_id in unknown_source_ids)}。",
                        ),
                    }
                )
                sanitation_notes.append(
                    f"candidate_resolution item `{item.display_title or item.page_plan_id or index}` {sanitation_note}: {sorted(unknown_source_ids)}."
                )
        sanitized_items.append(item)
    artifact = artifact.model_copy(update={"items": sanitized_items, "missed_candidate_risks": sanitation_notes})
    issues: list[StructuredIssue] = []
    for index, item in enumerate(artifact.items):
        if item.page_type not in profile.page_types:
            issues.append(
                StructuredIssue(
                    issue_code="unknown_page_type",
                    field_path=f"items.{index}.page_type",
                    validator_id="finalize_candidate_resolution",
                    message=(
                        f"candidate_resolution uses unsupported page_type `{item.page_type}`; "
                        "remove weak/noise formal items or choose a valid profile page_type."
                    ),
                    repairability="repairable",
                )
            )
        if item.reason.strip().lower() in {"ignore", "ignored", "noise", "weak", "弱相关", "噪声"}:
            issues.append(
                StructuredIssue(
                    issue_code="ignore_as_formal_item",
                    field_path=f"items.{index}.reason",
                    validator_id="finalize_candidate_resolution",
                    message="formal page plans must not use ignore/noise as the reason; remove this item.",
                    repairability="repairable",
                )
            )
    if issues:
        raise ContractValidationError(
            "candidate_resolution contains weak/noise formal item(s); repair by deleting those formal items.",
            issues=issues,
        )
    for item in artifact.items:
        page_type = item.page_type
        display_title = _wiki_markup.clean_display_title(item.display_title) or item.display_title.strip()
        source_fingerprint = source_basis_fingerprint(item.source_basis)
        page_plan_id = stable_page_plan_id(page_type, display_title, source_fingerprint)
        stem = unicode_safe_stem(display_title)
        target = page_output_path(vault / "wiki", profile, page_type, stem)
        rel_target = target.relative_to(vault / "wiki").as_posix()
        if rel_target in seen_paths:
            seen_paths[rel_target] += 1
            path = Path(rel_target)
            suffix = sha256_bytes(f"{page_type}:{display_title}:{page_plan_id}".encode("utf-8"))[:8]
            rel_target = path.with_name(f"{path.stem}_{suffix}{path.suffix}").as_posix()
        else:
            seen_paths[rel_target] = 1
        items.append(
            item.model_copy(
                update={
                    "page_plan_id": page_plan_id,
                    "page_type": page_type,
                    "display_title": display_title,
                    "path_stem": stem,
                    "candidate_target_path": rel_target,
                }
            )
        )
    return CandidateResolutionArtifact(items=items, missed_candidate_risks=artifact.missed_candidate_risks)


def source_basis_fingerprint(source_basis: SourceBasis) -> str:
    payload = json.dumps(source_basis.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
    return sha256_bytes(payload.encode("utf-8"))[:12]


def stable_page_plan_id(page_type: str, display_title: str, source_fingerprint: str) -> str:
    base = f"{page_type}:{_wiki_markup.normalize_related_key(display_title)}:{source_fingerprint}"
    return f"PP-{sha256_bytes(base.encode('utf-8'))[:12]}"


def unicode_safe_stem(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value)
    normalized = re.sub(r"\s+", " ", normalized.strip())
    bad = '\\/:*?"<>|#^[]'
    cleaned = "".join("_" if char in bad else char for char in normalized).strip(" .")
    return cleaned or "untitled"


def render_candidate_resolution_markdown(artifact: CandidateResolutionArtifact) -> str:
    rows = [
        [
            item.page_plan_id,
            item.page_type,
            item.display_title,
            f"`{item.candidate_target_path}`",
            ", ".join(item.source_basis.source_candidate_ids),
            ", ".join(item.source_basis.prepared_discovered_candidates),
            item.why_this_page,
        ]
        for item in artifact.items
    ]
    return "# 候选页面规划\n\n" + format_markdown_table(
        ["页面计划", "类型", "标题", "目标", "来源候选", "Prepared 发现候选", "为什么写"],
        rows,
    ) + "\n"
