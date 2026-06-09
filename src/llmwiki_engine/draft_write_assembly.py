from __future__ import annotations

from pathlib import Path

from . import diff_utils as _diff_utils
from . import draft_grounding as _draft_grounding
from . import draft_outputs as _draft_outputs
from . import errors as _errors
from . import open_questions as _open_questions
from .hash_utils import sha256_file
from .io import write_json
from .models import (
    DraftRenderingArtifact,
    DraftWriteManifest,
    DraftWriteTarget,
    GroundingClaim,
    ProfileSpec,
    RawLinkCleanupArtifact,
    RelatedCandidateReport,
    RelatedMergeReport,
    SourceDigestArtifact,
    UpdateMergeReport,
    UpdatePageMergeReport,
    WikiContextSnapshot,
    WikiMergePlanArtifact,
)
from .rendering import source_title_for_raw
from .system_pages import assert_current_system_page, render_daily_log, render_index, render_log_index
from .wiki_context import snapshot_entry


def assemble_draft_write_outputs(
    *,
    vault: Path,
    run_dir: Path,
    step_root: Path,
    profile: ProfileSpec,
    operation_id: str,
    digest: SourceDigestArtifact,
    merge_plan: WikiMergePlanArtifact,
    snapshot: WikiContextSnapshot,
    draft_artifact: DraftRenderingArtifact,
    approved_prepared_text: str,
    raw_hash: str,
    prepared_hash: str,
    cleanup: RawLinkCleanupArtifact,
    root_model_input_sidecars: list[Path],
) -> list[Path]:
    draft_artifact_path = step_root / "draft_rendering.json"
    write_json(draft_artifact_path, draft_artifact)

    draft_root = step_root / "draft_pages"
    outputs: list[Path] = [*root_model_input_sidecars, *_existing_step_sidecars(step_root)]
    target_manifest: list[DraftWriteTarget] = []
    action_by_id = {item.page_plan_id: item for item in merge_plan.items}
    update_report_pages: list[UpdatePageMergeReport] = []
    related_report_candidates: list[RelatedCandidateReport] = []
    grounding_claims: list[GroundingClaim] = []
    known_related_paths = _known_related_paths(snapshot, merge_plan)
    knowledge_changed_paths: list[str] = []
    no_change_pages = [item.canonical_target_path for item in merge_plan.items if item.action == "noop"]

    for page in draft_artifact.pages:
        plan_item = action_by_id[page.page_plan_id]
        entry = snapshot_entry(snapshot, f"wiki/{page.canonical_target_path}")
        target = draft_root / page.canonical_target_path
        target.parent.mkdir(parents=True, exist_ok=True)
        markdown = _draft_outputs.assemble_knowledge_page(
            item=plan_item,
            page=page,
            existing_entry=entry,
            raw_path=digest.source_raw_path,
            raw_hash=raw_hash,
            prepared_hash=prepared_hash,
            operation_id=operation_id,
            log_date=snapshot.log_date,
            update_reports=update_report_pages,
            related_reports=related_report_candidates,
            grounding_claims=grounding_claims,
            known_related_paths=known_related_paths,
            approved_raw_text=approved_prepared_text,
        )
        target.write_text(markdown, encoding="utf-8")
        outputs.append(target)
        knowledge_changed_paths.append(page.canonical_target_path)
        target_manifest.append(
            DraftWriteTarget(
                action=page.action,
                target_path=f"wiki/{page.canonical_target_path}",
                draft_path=target.relative_to(run_dir).as_posix(),
                expected_state=entry.expected_state,
                preimage_sha256=entry.preimage_sha256,
                page_plan_id=page.page_plan_id,
            )
        )
        if page.action != "update":
            continue
        diff_path = step_root / "diffs" / f"{page.page_plan_id}.diff"
        diff_path.parent.mkdir(parents=True, exist_ok=True)
        diff_path.write_text(
            _diff_utils.render_update_diff(entry.content, markdown, f"old/{page.canonical_target_path}", f"new/{page.canonical_target_path}"),
            encoding="utf-8",
        )
        diff_json = step_root / "diffs" / f"{page.page_plan_id}.json"
        write_json(
            diff_json,
            {
                "old_snapshot_ref": f"wiki_context_snapshot/wiki_context_snapshot.json#{entry.path}",
                "old_rendered_markdown_path": entry.path,
                "new_draft_ref": target.relative_to(run_dir).as_posix(),
                "new_rendered_markdown_path": target.relative_to(run_dir).as_posix(),
                "unified_diff": diff_path.relative_to(run_dir).as_posix(),
                "change_summary": page.change_summary,
                "preimage_sha256": page.preimage_sha256,
                "draft_sha256": sha256_file(target),
            },
        )
        outputs.extend([diff_path, diff_json])

    source_page = draft_root / snapshot.source_target_path
    source_page.parent.mkdir(parents=True, exist_ok=True)
    source_page.write_text(
        _draft_outputs.render_source_page(
            title=source_title_for_raw(digest.source_raw_path),
            digest=digest,
            operation_id=operation_id,
            linked_pages=knowledge_changed_paths,
            no_change_pages=no_change_pages,
            log_date=snapshot.log_date,
            raw_hash=raw_hash,
            prepared_hash=prepared_hash,
            cleanup=cleanup,
        ),
        encoding="utf-8",
    )
    outputs.append(source_page)
    source_entry = snapshot_entry(snapshot, f"wiki/{snapshot.source_target_path}")
    target_manifest.append(
        DraftWriteTarget(
            action="source",
            target_path=f"wiki/{snapshot.source_target_path}",
            draft_path=source_page.relative_to(run_dir).as_posix(),
            expected_state=source_entry.expected_state,
            preimage_sha256=source_entry.preimage_sha256,
        )
    )

    log_date = snapshot.log_date
    if knowledge_changed_paths:
        _assert_system_page_can_be_overwritten(vault, "wiki/index.md")
        index = draft_root / "index.md"
        open_question_rows, open_question_report = _open_questions.build_open_question_rows_with_report(merge_plan, draft_artifact, snapshot)
        index.write_text(
            render_index(
                knowledge_rows=_draft_outputs.build_index_rows(profile, merge_plan, draft_artifact, snapshot),
                tension_rows=open_question_rows,
                page_type_order=list(profile.page_types),
            ),
            encoding="utf-8",
        )
        open_question_report_path = step_root / "index_open_questions_report.json"
        open_question_report_md = step_root / "index_open_questions_report.md"
        write_json(open_question_report_path, open_question_report)
        open_question_report_md.write_text(_open_questions.render_index_open_questions_report(open_question_report), encoding="utf-8")
        outputs.extend([index, open_question_report_path, open_question_report_md])
        index_entry = snapshot_entry(snapshot, "wiki/index.md")
        target_manifest.append(
            DraftWriteTarget(
                action="index",
                target_path="wiki/index.md",
                draft_path=index.relative_to(run_dir).as_posix(),
                expected_state=index_entry.expected_state,
                preimage_sha256=index_entry.preimage_sha256,
            )
        )

    _assert_system_page_can_be_overwritten(vault, "wiki/log.md")
    log_entry = snapshot_entry(snapshot, "wiki/log.md")
    log_index = draft_root / "log.md"
    log_index.write_text(
        render_log_index(
            date=log_date,
            operation_id=operation_id,
            source=digest.source_raw_path,
            existing_text=log_entry.content if log_entry.expected_state == "present" else None,
        ),
        encoding="utf-8",
    )
    outputs.append(log_index)
    target_manifest.append(
        DraftWriteTarget(
            action="global_log",
            target_path="wiki/log.md",
            draft_path=log_index.relative_to(run_dir).as_posix(),
            expected_state=log_entry.expected_state,
            preimage_sha256=log_entry.preimage_sha256,
        )
    )

    _assert_system_page_can_be_overwritten(vault, f"wiki/logs/{log_date}.md")
    daily_entry = snapshot_entry(snapshot, f"wiki/logs/{log_date}.md")
    daily_log = draft_root / "logs" / f"{log_date}.md"
    daily_log.parent.mkdir(parents=True, exist_ok=True)
    daily_log.write_text(
        render_daily_log(
            date=log_date,
            operation_id=operation_id,
            raw_path=digest.source_raw_path,
            created=sum(1 for item in merge_plan.items if item.action == "create"),
            updated=sum(1 for item in merge_plan.items if item.action == "update"),
            noop=sum(1 for item in merge_plan.items if item.action == "noop"),
            needs_human=sum(1 for item in merge_plan.items if item.action == "needs_human_decision"),
            existing_text=daily_entry.content if daily_entry.expected_state == "present" else None,
        ),
        encoding="utf-8",
    )
    outputs.append(daily_log)
    target_manifest.append(
        DraftWriteTarget(
            action="daily_log",
            target_path=f"wiki/logs/{log_date}.md",
            draft_path=daily_log.relative_to(run_dir).as_posix(),
            expected_state=daily_entry.expected_state,
            preimage_sha256=daily_entry.preimage_sha256,
        )
    )

    update_report = UpdateMergeReport(pages=update_report_pages)
    update_report_path = step_root / "update_merge_report.json"
    update_report_md = step_root / "update_merge_report.md"
    write_json(update_report_path, update_report)
    update_report_md.write_text(_draft_outputs.render_update_merge_report(update_report), encoding="utf-8")
    related_report = RelatedMergeReport(candidates=related_report_candidates)
    related_report_path = step_root / "related_merge_report.json"
    related_report_md = step_root / "related_merge_report.md"
    write_json(related_report_path, related_report)
    related_report_md.write_text(_draft_outputs.render_related_merge_report(related_report), encoding="utf-8")
    grounding_review = _draft_grounding.draft_grounding_review_from_claims(grounding_claims)
    grounding_review_path = step_root / "draft_grounding_review.json"
    grounding_review_md = step_root / "draft_grounding_review.md"
    write_json(grounding_review_path, grounding_review)
    grounding_review_md.write_text(_draft_grounding.render_draft_grounding_review(grounding_review), encoding="utf-8")
    outputs.extend([update_report_path, update_report_md, related_report_path, related_report_md, grounding_review_path, grounding_review_md])

    write_manifest = DraftWriteManifest(
        targets=target_manifest,
        has_updates=any(item.action == "update" for item in merge_plan.items),
        has_noops=any(item.action == "noop" for item in merge_plan.items),
        source_only_noop=all(item.action == "noop" for item in merge_plan.items),
        requires_grounding_review=grounding_review.requires_review,
    )
    write_manifest_path = step_root / "draft_write_manifest.json"
    write_json(write_manifest_path, write_manifest)
    outputs.extend([draft_artifact_path, write_manifest_path])
    return outputs


def _existing_step_sidecars(step_root: Path) -> list[Path]:
    return [
        path
        for path in [
            step_root / "update_preservation_reinforcement_report.json",
            step_root / "update_preservation_reinforcement_report.md",
            step_root / "grounding_paraphrase_rewrite_report.json",
            step_root / "grounding_paraphrase_rewrite_report.md",
            step_root / "draft_rendering_batch_report.json",
            step_root / "draft_rendering_batch_report.md",
        ]
        if path.exists()
    ]


def _known_related_paths(snapshot: WikiContextSnapshot, merge_plan: WikiMergePlanArtifact) -> set[str]:
    known = {
        entry.path
        for entry in snapshot.knowledge_metadata_pool
        if not entry.path.startswith("sources/")
        and not entry.path.startswith("logs/")
        and entry.path not in {"index.md", "log.md"}
    }
    known.update(
        item.canonical_target_path
        for item in merge_plan.items
        if item.action in {"create", "update", "noop"} and not item.canonical_target_path.startswith(("sources/", "logs/"))
    )
    return known


def _assert_system_page_can_be_overwritten(vault: Path, target_path: str) -> None:
    path = vault / target_path
    if not path.exists():
        return
    try:
        assert_current_system_page(path)
    except RuntimeError as exc:
        raise _errors.PipelineError(f"{exc}: {target_path}") from exc
