from __future__ import annotations

from pathlib import Path

from . import errors as _errors
from .hash_utils import sha256_file
from .models import WikiContextEntry, WikiContextSnapshot
from .retrieval import build_knowledge_pool, candidate_pool_sha256


def snapshot_entry(snapshot: WikiContextSnapshot, path: str) -> WikiContextEntry:
    for entry in snapshot.entries:
        if entry.path == path:
            return entry
    raise _errors.PipelineError(f"snapshot missing path: {path}")


def wiki_context_drift_messages(vault: Path, snapshot: WikiContextSnapshot) -> list[str]:
    messages: list[str] = []
    if snapshot.candidate_pool_sha256:
        current_pool_hash = candidate_pool_sha256(build_knowledge_pool(vault))
        if current_pool_hash != snapshot.candidate_pool_sha256:
            messages.append("wiki knowledge candidate pool changed after planning")
    for entry in snapshot.entries:
        path = vault / entry.path
        if entry.expected_state == "missing":
            if path.exists():
                messages.append(f"wiki context appeared after planning: {entry.path}")
            continue
        if not path.exists():
            messages.append(f"wiki context disappeared after planning: {entry.path}")
            continue
        if entry.preimage_sha256 is None:
            messages.append(f"wiki context snapshot is missing sha256: {entry.path}")
            continue
        if sha256_file(path) != entry.preimage_sha256:
            messages.append(f"wiki context changed after planning: {entry.path}")
    return messages
