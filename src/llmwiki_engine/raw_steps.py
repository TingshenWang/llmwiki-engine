from __future__ import annotations

from . import artifact_refs as _artifact_refs
from . import diff_utils as _diff_utils
from . import errors as _errors
from . import redaction as _redaction
from .hash_utils import sha256_file
from .io import read_model, write_json
from .manifest import complete_step, raw_ref
from .models import (
    RawBinding,
    RawLinkCleanupArtifact,
    RawPreparationArtifact,
    RawPreparePolicy,
    ReviewDecision,
    StructuredIssue,
)
from .raw_cleanup import cleanup_raw_wikilinks, render_raw_link_cleanup_markdown
from .step_runtime import StepRunContext, complete_review_step, structured_call
from .steps import require_step_output_dir
from .validators import ValidationError as ContractValidationError
from .validators import validate_raw_preparation
from .workspace import relative_to_vault


RAW_PREPARE_CONTRACT = {
    "goal": "Create a higher-quality canonical prepared raw for downstream knowledge compilation.",
    "rules": [
        "Do not add facts that are not supported by the original raw.",
        "Remove or relocate non-content noise such as navigation fragments, boilerplate, self-promotion, and obvious formatting artifacts.",
        "Correct obvious wording or formatting errors only when the surrounding context makes the correction clear.",
        "Return prepared_markdown as clean Markdown suitable for source_digest and downstream knowledge digestion.",
    ],
}


def run_raw_link_cleanup(ctx: StepRunContext) -> None:
    step_name = "raw_link_cleanup"
    step_root = require_step_output_dir(ctx.run_dir, step_name)
    raw_rel = relative_to_vault(ctx.vault, ctx.raw_path)
    original_text = ctx.raw_path.read_text(encoding="utf-8")
    pre_hash = sha256_file(ctx.raw_path)
    cleaned_text, links, warnings, preserved_media_count = cleanup_raw_wikilinks(original_text)
    changed = cleaned_text != original_text
    if changed:
        if sha256_file(ctx.raw_path) != pre_hash:
            raise _errors.PipelineError("raw changed during raw_link_cleanup; rerun ingest")
        tmp = ctx.raw_path.with_name(f".{ctx.raw_path.name}.tmp")
        tmp.write_text(cleaned_text, encoding="utf-8")
        tmp.replace(ctx.raw_path)
    post_hash, post_size = raw_ref(ctx.raw_path)
    ctx.manifest.raw_bindings = [RawBinding(relative_path=raw_rel, sha256=post_hash, size_bytes=post_size)]
    artifact = RawLinkCleanupArtifact(
        raw_path=raw_rel,
        changed=changed,
        pre_cleanup_sha256=pre_hash,
        post_cleanup_sha256=post_hash,
        cleaned_link_count=len(links),
        preserved_media_embed_count=preserved_media_count,
        links=links,
        warnings=warnings,
    )
    out = step_root / "raw_link_cleanup.json"
    write_json(out, artifact)
    report = step_root / "raw_link_cleanup.md"
    report.write_text(render_raw_link_cleanup_markdown(artifact), encoding="utf-8")
    diff_path = step_root / "cleanup.diff"
    diff_path.write_text(_diff_utils.render_update_diff(original_text, cleaned_text, f"pre/{raw_rel}", f"post/{raw_rel}"), encoding="utf-8")
    complete_step(
        ctx.manifest,
        step_name,
        outputs=[
            _artifact_refs.ref(ctx.run_dir, out, step_name, "json", "raw_link_cleanup.v1"),
            _artifact_refs.ref(ctx.run_dir, report, step_name, "markdown"),
            _artifact_refs.ref(ctx.run_dir, diff_path, step_name, "diff"),
        ],
    )


def _write_raw_prepare_outputs(ctx: StepRunContext, preparation: RawPreparationArtifact, *, include_model_outputs: bool) -> None:
    step_name = "raw_prepare"
    step_root = require_step_output_dir(ctx.run_dir, step_name)
    validate_raw_preparation(preparation)
    out = step_root / "raw_preparation.json"
    write_json(out, preparation)
    prepared = step_root / "prepared.md"
    prepared.parent.mkdir(parents=True, exist_ok=True)
    prepared.write_text(preparation.prepared_markdown.rstrip() + "\n", encoding="utf-8")
    outputs = [
        _artifact_refs.ref(ctx.run_dir, out, step_name, "json", "raw_preparation.v1"),
        _artifact_refs.ref(ctx.run_dir, prepared, step_name, "markdown"),
    ]
    if include_model_outputs:
        outputs.extend(_artifact_refs.structured_model_output_refs(ctx.run_dir, step_root, step_name))
    complete_step(ctx.manifest, step_name, outputs=outputs)


def run_raw_prepare(ctx: StepRunContext) -> None:
    step_name = "raw_prepare"
    raw_rel = relative_to_vault(ctx.vault, ctx.raw_path)
    cleanup_path = require_step_output_dir(ctx.run_dir, "raw_link_cleanup") / "raw_link_cleanup.json"
    cleanup = read_model(cleanup_path, RawLinkCleanupArtifact)
    input_raw_sha256 = sha256_file(ctx.raw_path)
    cleanup_ref = cleanup_path.relative_to(ctx.run_dir).as_posix()
    raw_prepare_policy = ctx.manifest.vault_config_snapshot.raw_prepare_policy
    if raw_prepare_policy == RawPreparePolicy.skip:
        if ctx.raw_path.suffix.lower() not in {".md", ".markdown", ".mdown"}:
            raise _errors.PipelineError("--prepare skip requires Markdown raw; use --prepare auto or --prepare force for non-Markdown raw.")
        raw_text = ctx.raw_path.read_text(encoding="utf-8")
        if not raw_text.strip():
            raise _errors.PipelineError("--prepare skip requires non-empty raw Markdown.")
        preparation = RawPreparationArtifact(
            source_raw_path=raw_rel,
            input_raw_sha256=input_raw_sha256,
            raw_link_cleanup_ref=cleanup_ref,
            prepared_markdown=raw_text.rstrip() + "\n",
            operations_applied=["user_skip_markdown_passthrough"],
            omission_policy="none",
        )
        _write_raw_prepare_outputs(ctx, preparation, include_model_outputs=False)
        return

    payload = {
        "source_raw_path": raw_rel,
        "source_raw_sha256": input_raw_sha256,
        "raw_prepare_policy": raw_prepare_policy.value,
        "raw_markdown": ctx.raw_path.read_text(encoding="utf-8"),
        "raw_link_cleanup_ref": cleanup_ref,
        "raw_link_cleanup": {
            "changed": cleanup.changed,
            "cleaned_link_count": cleanup.cleaned_link_count,
            "preserved_media_embed_count": cleanup.preserved_media_embed_count,
            "cleanup_rule_version": cleanup.cleanup_rule_version,
        },
        "contract": RAW_PREPARE_CONTRACT,
    }

    def validate_raw_prepare_model(model: RawPreparationArtifact) -> None:
        candidate = model.model_copy(
            update={
                "input_raw_sha256": input_raw_sha256,
                "raw_link_cleanup_ref": cleanup_ref,
            }
        )
        validate_raw_preparation(candidate)
        if candidate.source_raw_path != raw_rel:
            raise ContractValidationError(
                f"raw_prepare source path mismatch: {candidate.source_raw_path} != {raw_rel}",
                issues=[
                    StructuredIssue(
                        issue_code="source_path_mismatch",
                        field_path="source_raw_path",
                        validator_id="validate_raw_prepare_model",
                        message=f"raw_prepare source path mismatch: {candidate.source_raw_path} != {raw_rel}",
                        repairability="repairable",
                    )
                ],
            )

    preparation, _ = structured_call(ctx.run_dir, ctx.execution_context, step_name).run(
        step_name,
        payload,
        RawPreparationArtifact,
        validator=validate_raw_prepare_model,
    )
    preparation = _redaction.redact_model(ctx.execution_context.redactor, preparation, RawPreparationArtifact)
    preparation = preparation.model_copy(
        update={
            "input_raw_sha256": input_raw_sha256,
            "raw_link_cleanup_ref": cleanup_ref,
        }
    )
    if preparation.source_raw_path != raw_rel:
        raise _errors.PipelineError(f"raw_prepare source path mismatch: {preparation.source_raw_path} != {raw_rel}")
    _write_raw_prepare_outputs(ctx, preparation, include_model_outputs=True)


def run_prepared_raw_review(ctx: StepRunContext) -> None:
    step_name = "prepared_raw_review"
    step_root = require_step_output_dir(ctx.run_dir, step_name)
    prepared = require_step_output_dir(ctx.run_dir, "raw_prepare") / "prepared.md"
    approved = step_root / "approved_prepared.md"
    approved.write_text(prepared.read_text(encoding="utf-8"), encoding="utf-8")
    prompt = step_root / "review_prompt.md"
    prompt.write_text(
        "# Prepared Raw 审核\n\n"
        "当前运行自动批准 prepared raw；下游步骤会继续基于 Approved Raw 校验。\n",
        encoding="utf-8",
    )
    feedback = step_root / "review_feedback.jsonl"
    feedback.write_text("", encoding="utf-8")
    decision = ReviewDecision(
        review_step=step_name,
        decision="approved",
        auto_approved=True,
        notes="当前运行自动批准；交互式审核尚未接入。",
    )
    decision_path = step_root / "review_decision.json"
    write_json(decision_path, decision)
    complete_review_step(
        ctx.manifest,
        step_name,
        outputs=[
            _artifact_refs.ref(ctx.run_dir, prompt, step_name, "markdown"),
            _artifact_refs.ref(ctx.run_dir, feedback, step_name, "jsonl"),
            _artifact_refs.ref(ctx.run_dir, decision_path, step_name, "json", "review_decision.v2"),
            _artifact_refs.ref(ctx.run_dir, approved, step_name, "markdown"),
        ],
        review_decision_ref=decision_path.relative_to(ctx.run_dir).as_posix(),
    )
