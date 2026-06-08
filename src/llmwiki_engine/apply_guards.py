from __future__ import annotations

from pathlib import Path

from . import errors as _errors
from .models import SourceDuplicateGuardArtifact
from .profiles import safe_filename
from .rendering import source_title_for_raw
from .source_records import frontmatter_list, normalize_vault_path, scan_source_pages
from .system_pages import format_markdown_table


REQUIRED_DRAFT_RENDERING_SIDECARS = (
    "draft_rendering/draft_rendering.json", "draft_rendering/draft_write_manifest.json",
    "draft_rendering/provider_result.json", "draft_rendering/update_merge_report.json",
    "draft_rendering/update_merge_report.md", "draft_rendering/related_merge_report.json",
    "draft_rendering/related_merge_report.md", "draft_rendering/draft_grounding_review.json",
    "draft_rendering/draft_grounding_review.md",
)


def require_draft_rendering_sidecars(run_dir: Path) -> None:
    missing = [rel_path for rel_path in REQUIRED_DRAFT_RENDERING_SIDECARS if not (run_dir / rel_path).is_file()]
    if missing:
        preview = ", ".join(f"`{path}`" for path in missing[:6])
        suffix = "" if len(missing) <= 6 else f", ... and {len(missing) - 6} more"
        raise _errors.PipelineError(
            "Draft rendering sidecar artifacts are missing; resume from draft_rendering or earlier before approve/apply: "
            f"{preview}{suffix}"
        )


def build_source_duplicate_guard_artifact(
    vault: Path,
    *,
    source_raw_path: str,
    source_raw_hash: str,
    source_prepared_hash: str,
    operation_id: str,
) -> SourceDuplicateGuardArtifact:
    normalized_raw_path = normalize_vault_path(source_raw_path)
    source_title = source_title_for_raw(normalized_raw_path)
    source_target_path = f"sources/{safe_filename(source_title)}.md"
    base = {
        "source_raw_path": normalized_raw_path, "source_raw_hash": source_raw_hash,
        "source_prepared_hash": source_prepared_hash, "source_target_path": source_target_path,
    }
    for source_page, frontmatter in scan_source_pages(vault):
        operation_ids = frontmatter_list(frontmatter, "source_operation_ids")
        if operation_id in operation_ids:
            continue
        raw_paths = [normalize_vault_path(value) for value in frontmatter_list(frontmatter, "source_raw_paths")]
        raw_hashes = frontmatter_list(frontmatter, "source_raw_hashes")
        prepared_hashes = frontmatter_list(frontmatter, "source_prepared_hashes")
        if normalized_raw_path in raw_paths and source_raw_hash in raw_hashes:
            return SourceDuplicateGuardArtifact(
                **base,
                status="source_duplicate",
                matched_source_page=source_page,
                matched_raw_path=normalized_raw_path,
                matched_raw_hash=source_raw_hash,
                matched_prepared_hash=source_prepared_hash if source_prepared_hash in prepared_hashes else None,
                reason="same raw path and raw hash already recorded in source page frontmatter",
            )
        if source_raw_hash in raw_hashes:
            return SourceDuplicateGuardArtifact(
                **base,
                status="source_duplicate",
                matched_source_page=source_page,
                matched_raw_hash=source_raw_hash,
                matched_prepared_hash=source_prepared_hash if source_prepared_hash in prepared_hashes else None,
                reason="same raw hash already recorded in source page frontmatter",
            )
        if normalized_raw_path in raw_paths and source_raw_hash not in raw_hashes:
            return SourceDuplicateGuardArtifact(
                **base,
                status="source_revision_detected",
                matched_source_page=source_page,
                matched_raw_path=normalized_raw_path,
                reason="same raw path exists with a different content hash",
            )
    target = vault / "wiki" / source_target_path
    if target.exists():
        return SourceDuplicateGuardArtifact(
            **base,
            status="source_duplicate",
            matched_source_page=f"wiki/{source_target_path}",
            reason="source target page already exists",
        )
    return SourceDuplicateGuardArtifact(
        **base,
        status="clear",
        reason="no matching source path, hash, or source target page found",
    )


def render_source_duplicate_guard_markdown(artifact: SourceDuplicateGuardArtifact) -> str:
    return "\n".join(
        [
            "# 来源重复检查",
            "",
            format_markdown_table(
                ["字段", "值"],
                [
                    ["status", artifact.status],
                    ["source_raw_path", f"`{artifact.source_raw_path}`"],
                    ["source_raw_hash", f"`{artifact.source_raw_hash}`"],
                    ["source_prepared_hash", f"`{artifact.source_prepared_hash}`"],
                    ["source_target_path", f"`{artifact.source_target_path}`"],
                    ["matched_source_page", f"`{artifact.matched_source_page}`" if artifact.matched_source_page else ""],
                    ["reason", artifact.reason],
                ],
            ),
        ]
    ).rstrip() + "\n"
