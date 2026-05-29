from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .io import read_model, write_json
from .models import OperationManifest, utc_now


class ApplyError(RuntimeError):
    pass


def apply_operation(vault: Path, operation_id: str, *, commit: bool = False) -> list[Path]:
    stage_dir = vault / "stage" / "ingest" / operation_id
    draft_root = stage_dir / "draft_pages"
    if not draft_root.exists():
        raise ApplyError(f"Draft root not found: {draft_root}")
    manifest = read_model(stage_dir / "manifest.json", OperationManifest)
    if manifest.status not in {"drafted", "applied"}:
        raise ApplyError(f"Operation is not apply-ready: {manifest.status}")
    snapshot = snapshot_existing_pages(vault, draft_root, stage_dir)
    preview = build_apply_preview(vault, draft_root)
    write_json(stage_dir / "apply_preview.json", {"operation_id": operation_id, "writes": [str(path) for path in preview]})
    written: list[Path] = []
    for draft in draft_root.rglob("*.md"):
        target = vault / "wiki" / draft.relative_to(draft_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(draft, target)
        written.append(target)
    manifest.status = "applied"
    manifest.updated_at = utc_now()
    write_json(stage_dir / "manifest.json", manifest)
    append_markdown_audit(vault, operation_id, written, snapshot)
    if commit:
        commit_changes(vault, operation_id)
    return written


def snapshot_existing_pages(vault: Path, draft_root: Path, stage_dir: Path) -> list[Path]:
    snapshot_root = stage_dir / "pre_apply_snapshot"
    copied: list[Path] = []
    for draft in draft_root.rglob("*.md"):
        target = vault / "wiki" / draft.relative_to(draft_root)
        if target.exists():
            out = snapshot_root / target.relative_to(vault / "wiki")
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(target, out)
            copied.append(out)
    write_json(stage_dir / "pre_apply_snapshot.json", {"files": [str(path) for path in copied]})
    return copied


def build_apply_preview(vault: Path, draft_root: Path) -> list[Path]:
    return [vault / "wiki" / draft.relative_to(draft_root) for draft in draft_root.rglob("*.md")]


def append_markdown_audit(vault: Path, operation_id: str, written: list[Path], snapshot: list[Path]) -> None:
    path = vault / "logs" / "audit.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"## {operation_id}",
        "",
        f"- status: applied",
        f"- written: {len(written)}",
        f"- snapshot_files: {len(snapshot)}",
        "",
    ]
    for item in written:
        lines.append(f"- {item.relative_to(vault)}")
    lines.append("")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def commit_changes(vault: Path, operation_id: str) -> None:
    if not (vault / ".git").exists():
        raise ApplyError("Cannot commit because vault is not a Git repository.")
    subprocess.run(["git", "add", "wiki", "logs", "stage"], cwd=vault, check=True)
    subprocess.run(["git", "commit", "-m", f"apply ingest {operation_id}"], cwd=vault, check=True)

