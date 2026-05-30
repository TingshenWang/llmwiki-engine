from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .hash_utils import artifact_ref, sha256_file
from .io import append_jsonl, read_model
from .manifest import read_manifest, write_manifest
from .models import AppliedReceipt, ApplyPreview, ArtifactRef, ArtifactVisibility, OperationStatus, utc_now
from .verify import require_verified
from .workspace import RunStore, run_lock


class ApplyError(RuntimeError):
    pass


def apply_operation(vault: Path, operation_id: str, *, commit: bool = False) -> list[Path]:
    store = RunStore(vault)
    run_dir = store.run_dir(operation_id)
    with run_lock(vault, operation_id):
        manifest = read_manifest(store.manifest_path(operation_id))
        if manifest.status == OperationStatus.applied:
            raise ApplyError("Operation has already been applied.")
        if manifest.status != OperationStatus.drafted:
            raise ApplyError(f"Operation is not apply-ready: {manifest.status}")
        if _receipt_exists(store.applied_log, operation_id):
            raise ApplyError("Applied receipt already exists for this operation.")
        require_verified(vault, manifest)
        preview = read_model(run_dir / "apply_preview" / "apply_preview.json", ApplyPreview)
        _verify_preimages(vault, preview)
        written: list[Path] = []
        for target in preview.targets:
            draft = run_dir / target.draft_path
            output = vault / target.target_path
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(draft.read_bytes())
            written.append(output)
        manifest.status = OperationStatus.applied
        manifest.updated_at = utc_now()
        write_manifest(store.manifest_path(operation_id), manifest)
        receipt = AppliedReceipt(
            operation_id=operation_id,
            raw_bindings=manifest.raw_bindings,
            prepared_raw=_optional_receipt_ref(vault, run_dir / "raw_prepare" / "prepared.md", "markdown", "raw_prepare"),
            raw_preparation=_optional_receipt_ref(vault, run_dir / "raw_prepare" / "raw_preparation.json", "json", "raw_prepare"),
            written_pages=[
                artifact_ref(
                    base=vault,
                    path=path,
                    kind="markdown",
                    producer_step="apply",
                    required_for_resume=False,
                    visibility=ArtifactVisibility.wiki_output,
                )
                for path in written
            ],
            profile=manifest.profile,
            profile_version=manifest.profile_version,
            engine_version=manifest.engine_version,
        )
        append_jsonl(store.applied_log, [receipt])
        if commit:
            commit_changes(vault, operation_id)
        return written


def _optional_receipt_ref(vault: Path, path: Path, kind: str, producer_step: str) -> ArtifactRef | None:
    if not path.exists():
        return None
    return artifact_ref(
        base=vault,
        path=path,
        kind=kind,
        producer_step=producer_step,
        required_for_resume=False,
        visibility=ArtifactVisibility.run_cache,
    )


def _verify_preimages(vault: Path, preview: ApplyPreview) -> None:
    for target in preview.targets:
        path = vault / target.target_path
        if target.preimage_missing:
            if path.exists():
                raise ApplyError(f"Target appeared after preview: {target.target_path}")
            continue
        if not path.exists():
            raise ApplyError(f"Target disappeared after preview: {target.target_path}")
        if sha256_file(path) != target.preimage_sha256:
            raise ApplyError(f"Target changed after preview: {target.target_path}")


def _receipt_exists(path: Path, operation_id: str) -> bool:
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if read_json_line(line).get("operation_id") == operation_id:
            return True
    return False


def read_json_line(line: str) -> dict:
    return json.loads(line)


def commit_changes(vault: Path, operation_id: str) -> None:
    if not (vault / ".git").exists():
        raise ApplyError("Cannot commit because vault is not a Git repository.")
    subprocess.run(["git", "add", "wiki", ".llmwiki/config.yaml", ".llmwiki/profiles", ".llmwiki/applied"], cwd=vault, check=True)
    subprocess.run(["git", "commit", "-m", f"apply ingest {operation_id}"], cwd=vault, check=True)
