from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import subprocess
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

    def run_dir(self, operation_id: str) -> Path:
        return self.runs_root / require_operation_id(operation_id)

    def manifest_path(self, operation_id: str) -> Path:
        return self.run_dir(operation_id) / "manifest.json"

    def lock_path(self, operation_id: str) -> Path:
        return self.run_dir(operation_id) / ".lock"

    @property
    def apply_lock_path(self) -> Path:
        return self.llmwiki / "apply.lock"


def require_operation_id(operation_id: str) -> str:
    value = operation_id.strip()
    if not value:
        raise WorkspaceError("Operation id is empty. Check that your OP/OP2 shell variable is set.")
    path = Path(value)
    if path.is_absolute() or len(path.parts) != 1 or value in {".", ".."}:
        raise WorkspaceError(f"Invalid operation id: {operation_id}")
    return value

def ensure_workspace_layout(vault: Path) -> None:
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
    line = ".llmwiki/"
    existing = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    existing = [item for item in existing if item != ".llmwiki/runs/"]
    if line not in existing:
        existing.append(line)
    path.write_text("\n".join(existing).strip() + "\n", encoding="utf-8")


@dataclass(frozen=True)
class LlmwikiGitState:
    is_git_repo: bool
    tracked: list[str]
    staged: list[str]
    root: Path | None = None


def llmwiki_git_state(vault: Path) -> LlmwikiGitState:
    root = _git_root(vault)
    if root is None:
        return LlmwikiGitState(is_git_repo=False, tracked=[], staged=[])
    rel = _relative_git_path(root, vault / ".llmwiki")
    tracked = _git_paths(root, ["git", "ls-files", "-z", "--", rel])
    staged = _git_paths(root, ["git", "diff", "--cached", "--name-only", "-z", "--", rel])
    return LlmwikiGitState(is_git_repo=True, tracked=tracked, staged=staged, root=root)


def _git_paths(vault: Path, args: list[str]) -> list[str]:
    result = subprocess.run(args, cwd=vault, check=False, capture_output=True)
    if result.returncode != 0:
        raise WorkspaceError(result.stderr.decode("utf-8", errors="replace").strip() or "git command failed")
    return [item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def _git_root(vault: Path) -> Path | None:
    result = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=vault, check=False, capture_output=True)
    if result.returncode != 0:
        return None
    return Path(result.stdout.decode("utf-8").strip()).resolve()


def _relative_git_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise WorkspaceError(f"Vault path is outside Git root: {path}") from exc


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


@contextmanager
def apply_lock(vault: Path) -> Iterator[None]:
    store = RunStore(vault)
    lock = store.apply_lock_path
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = lock.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise WorkspaceError(f"Apply lock exists: {lock}") from exc
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
    if not resolved.is_file():
        raise WorkspaceError(f"Raw input file does not exist: {raw_rel.as_posix()}")
    return resolved, raw_rel.as_posix()
