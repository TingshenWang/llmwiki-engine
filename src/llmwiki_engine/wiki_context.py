from __future__ import annotations

from pathlib import Path

from . import errors as _errors
from .hash_utils import sha256_file
from .models import CandidateResolutionArtifact, EmbeddingRetrievalConfig, WikiContextEntry, WikiContextSnapshot
from .retrieval import (
    RetrievalError,
    build_candidate_contexts,
    build_knowledge_pool,
    candidate_pool_sha256,
    metadata_from_text,
)


def build_wiki_context_snapshot(
    vault: Path,
    resolution: CandidateResolutionArtifact,
    *,
    log_date: str,
    source_target_path: str,
    retrieval_config: EmbeddingRetrievalConfig | None = None,
    force_exact_backend: bool = False,
) -> WikiContextSnapshot:
    retrieval_config = retrieval_config or EmbeddingRetrievalConfig(backend="exact")
    knowledge_pool = build_knowledge_pool(vault)
    try:
        candidate_contexts = build_candidate_contexts(
            resolution=resolution,
            knowledge_pool=knowledge_pool,
            config=retrieval_config,
            vault=vault,
            force_exact_backend=force_exact_backend,
        )
    except RetrievalError as exc:
        raise _errors.PipelineError(str(exc)) from exc
    paths = {
        "wiki/index.md",
        "wiki/log.md",
        f"wiki/logs/{log_date}.md",
        f"wiki/{source_target_path}",
    }
    for item in resolution.items:
        paths.add(f"wiki/{item.candidate_target_path}")
    for context_item in candidate_contexts.items:
        for hit in context_item.hits:
            paths.add(f"wiki/{hit.path}")
    entries: list[WikiContextEntry] = []
    for rel in sorted(paths):
        path = vault / rel
        if path.exists():
            text = path.read_text(encoding="utf-8")
            entries.append(
                WikiContextEntry(
                    path=rel,
                    expected_state="present",
                    preimage_sha256=sha256_file(path),
                    content=text,
                    metadata=metadata_from_text(text, rel),
                )
            )
        else:
            entries.append(WikiContextEntry(path=rel, expected_state="missing", preimage_sha256=None, content=""))
    return WikiContextSnapshot(
        log_date=log_date,
        source_target_path=source_target_path,
        candidate_pool_sha256=candidate_pool_sha256(knowledge_pool),
        knowledge_metadata_pool=knowledge_pool,
        candidate_contexts=candidate_contexts,
        entries=entries,
    )


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
