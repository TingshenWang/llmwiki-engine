from __future__ import annotations

from pathlib import Path

from .hash_utils import artifact_ref
from .models import ArtifactRef


def ref(
    run_dir: Path,
    path: Path,
    producer_step: str,
    kind: str,
    schema_version: str | None = None,
    required_for_resume: bool = True,
) -> ArtifactRef:
    return artifact_ref(
        base=run_dir,
        path=path,
        kind=kind,
        producer_step=producer_step,
        schema_version=schema_version,
        required_for_resume=required_for_resume,
    )


def structured_model_output_refs(run_dir: Path, step_root: Path, step_name: str) -> list[ArtifactRef]:
    refs: list[ArtifactRef] = []
    provider_result = step_root / "provider_result.json"
    if provider_result.exists():
        refs.append(ref(run_dir, provider_result, step_name, "provider_result", "provider_result.v1"))
    report_json = step_root / "structured_repair_report.json"
    if report_json.exists():
        refs.append(ref(run_dir, report_json, step_name, "json", "structured_repair_report.v1"))
    report_md = step_root / "structured_repair_report.md"
    if report_md.exists():
        refs.append(ref(run_dir, report_md, step_name, "markdown"))
    provider_results = step_root / "provider_results"
    if provider_results.exists():
        for path in sorted(provider_results.glob("attempt-*.json")):
            refs.append(ref(run_dir, path, step_name, "provider_result", "provider_result.v1", required_for_resume=False))
    repair_prompts = step_root / "repair_prompts"
    if repair_prompts.exists():
        for path in sorted(repair_prompts.glob("attempt-*.json")):
            refs.append(ref(run_dir, path, step_name, "json", required_for_resume=False))
    return refs


def draft_rendering_model_batch_refs(run_dir: Path, step_root: Path, step_name: str) -> list[ArtifactRef]:
    batch_root = step_root / "model_batches"
    if not batch_root.exists():
        return []
    refs: list[ArtifactRef] = []
    for path in sorted(batch_root.rglob("*")):
        if not path.is_file():
            continue
        required = not any(part in {"provider_results", "repair_prompts"} for part in path.relative_to(step_root).parts)
        schema = None
        if path.name == "provider_result.json" or (path.parent.name == "provider_results" and path.name.startswith("attempt-")):
            schema = "provider_result.v1"
        elif path.name == "structured_repair_report.json":
            schema = "structured_repair_report.v1"
        elif path.name == "draft_source_excerpt_pack.json":
            schema = "draft_source_excerpt_pack.v1"
        elif path.name == "update_preservation_pack.json":
            schema = "update_preservation_pack.v1"
        elif path.name == "update_preservation_reinforcement_report.json":
            schema = "update_preservation_reinforcement_report.v1"
        elif path.name == "grounding_paraphrase_rewrite_report.json":
            schema = "grounding_paraphrase_rewrite_report.v1"
        refs.append(ref(run_dir, path, step_name, artifact_kind_for_path(path), schema, required_for_resume=required))
    return refs


def artifact_kind_for_path(path: Path) -> str:
    return {
        ".md": "markdown",
        ".json": "json",
        ".jsonl": "jsonl",
        ".diff": "diff",
    }.get(path.suffix, "text")


def draft_rendering_ref(run_dir: Path, path: Path, step_name: str) -> ArtifactRef:
    schemas = {
        "draft_rendering.json": "draft_rendering.v3",
        "draft_source_excerpt_pack.json": "draft_source_excerpt_pack.v1",
        "update_preservation_pack.json": "update_preservation_pack.v1",
        "update_preservation_reinforcement_report.json": "update_preservation_reinforcement_report.v1",
        "grounding_paraphrase_rewrite_report.json": "grounding_paraphrase_rewrite_report.v1",
        "draft_write_manifest.json": "draft_write_manifest.v1",
        "update_merge_report.json": "update_merge_report.v1",
        "related_merge_report.json": "related_merge_report.v1",
        "draft_grounding_review.json": "draft_grounding_review.v1",
        "index_open_questions_report.json": "index_open_questions_report.v1",
        "draft_rendering_batch_report.json": "draft_rendering_batch_report.v1",
    }
    return ref(run_dir, path, step_name, artifact_kind_for_path(path), schemas.get(path.name))


def replace_artifact_ref(refs: list[ArtifactRef], new_ref: ArtifactRef) -> list[ArtifactRef]:
    replaced = False
    next_refs: list[ArtifactRef] = []
    for existing in refs:
        if existing.relative_path == new_ref.relative_path:
            next_refs.append(new_ref)
            replaced = True
        else:
            next_refs.append(existing)
    if not replaced:
        next_refs.append(new_ref)
    return next_refs
