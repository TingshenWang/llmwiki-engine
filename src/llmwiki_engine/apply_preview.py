from __future__ import annotations

import json
from pathlib import Path

from . import draft_reviewing as _draft_reviewing
from . import errors as _errors
from .hash_utils import sha256_bytes, sha256_file
from .io import read_model
from .models import ApplyPreview, ApplyTarget, DraftWriteManifest
from .steps import require_step_output_dir


def build_apply_preview(vault: Path, run_dir: Path) -> ApplyPreview:
    operation_id = run_dir.name
    manifest_path = require_step_output_dir(run_dir, "draft_review") / "approved_write_manifest.json"
    draft_manifest = read_model(manifest_path, DraftWriteManifest)
    target_paths = [item.target_path for item in draft_manifest.targets]
    if len(target_paths) != len(set(target_paths)):
        raise _errors.PipelineError("draft_write_manifest contains duplicate target_path values")
    targets = [
        ApplyTarget(
            action=item.action,
            target_path=item.target_path,
            draft_path=item.draft_path,
            expected_state=item.expected_state,
            preimage_sha256=item.preimage_sha256,
            current_sha256=sha256_file(current) if (current := vault / item.target_path).exists() else None,
            will_write=True,
            approved_draft_ref=manifest_path.relative_to(run_dir).as_posix(),
            page_plan_id=item.page_plan_id,
        )
        for item in draft_manifest.targets
    ]
    write_set_payload = {
        "approved_manifest_sha256": sha256_file(manifest_path),
        "targets": [
            {
                "target_path": target.target_path,
                "draft_path": target.draft_path,
                "preimage_sha256": target.preimage_sha256,
                "expected_state": target.expected_state,
                "draft_sha256": sha256_file(run_dir / target.draft_path),
            }
            for target in targets
        ],
    }
    write_set_sha = sha256_bytes(json.dumps(write_set_payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return ApplyPreview(
        operation_id=operation_id,
        operation_applyable=bool(targets),
        requires_draft_review=_draft_reviewing.draft_review_requires_manual(run_dir, draft_manifest),
        has_updates=draft_manifest.has_updates,
        has_noops=draft_manifest.has_noops,
        blocked_reasons=[],
        write_set_sha256=write_set_sha,
        targets=targets,
        source_targets=[target.target_path for target in targets if target.action == "source"],
        log_targets=[target.target_path for target in targets if target.action in {"global_log", "daily_log"}],
        index_targets=[target.target_path for target in targets if target.action == "index"],
    )
