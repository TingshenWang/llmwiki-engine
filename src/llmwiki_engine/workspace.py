from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .models import utc_now


class WorkspaceError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunStore:
    vault: Path

    @property
    def llmwiki(self) -> Path:
        return self.vault / ".llmwiki"

    @property
    def runs_root(self) -> Path:
        return self.llmwiki / "runs" / "ingest"

    @property
    def applied_log(self) -> Path:
        return self.llmwiki / "applied" / "operations.jsonl"

    @property
    def config_path(self) -> Path:
        return self.llmwiki / "config.yaml"

    def run_dir(self, operation_id: str) -> Path:
        return self.runs_root / operation_id

    def manifest_path(self, operation_id: str) -> Path:
        return self.run_dir(operation_id) / "manifest.json"

    def lock_path(self, operation_id: str) -> Path:
        return self.run_dir(operation_id) / ".lock"

def ensure_v2_layout(vault: Path) -> None:
    if (vault / "stage" / "ingest").exists() and not (vault / ".llmwiki").exists():
        raise WorkspaceError("Legacy stage/ingest layout detected. Re-run init and create a new operation.")
    store = RunStore(vault)
    (store.runs_root).mkdir(parents=True, exist_ok=True)
    (store.llmwiki / "profiles").mkdir(parents=True, exist_ok=True)
    (store.llmwiki / "applied").mkdir(parents=True, exist_ok=True)
    if not store.applied_log.exists():
        store.applied_log.write_text("", encoding="utf-8")
    ensure_gitignore(vault)


def ensure_gitignore(vault: Path) -> None:
    path = vault / ".gitignore"
    line = ".llmwiki/runs/"
    existing = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    if line not in existing:
        existing.append(line)
        path.write_text("\n".join(existing).strip() + "\n", encoding="utf-8")


@contextmanager
def run_lock(vault: Path, operation_id: str) -> Iterator[None]:
    store = RunStore(vault)
    lock = store.lock_path(operation_id)
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = lock.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise WorkspaceError(f"Run lock exists: {lock}") from exc
    with fd:
        fd.write(utc_now())
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def relative_to_vault(vault: Path, path: Path) -> str:
    return path.resolve().relative_to(vault.resolve()).as_posix()


def resolve_raw_path(vault: Path, raw_file: Path) -> tuple[Path, str]:
    raw_root = (vault / "raw").resolve()
    candidate = raw_file if raw_file.is_absolute() else vault / raw_file
    resolved = candidate.resolve()
    try:
        rel = resolved.relative_to(raw_root)
    except ValueError as exc:
        raise WorkspaceError("Raw input must be inside the vault raw/ directory.") from exc
    raw_rel = Path("raw") / rel
    if ".." in raw_rel.parts:
        raise WorkspaceError("Raw path traversal is not allowed.")
    return resolved, raw_rel.as_posix()
