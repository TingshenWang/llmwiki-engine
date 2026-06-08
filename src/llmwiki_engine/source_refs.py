from __future__ import annotations

from . import markdown_utils as _markdown_utils
from .models import SourceBasis, SourceDigestArtifact, SourceDigestCandidate


def source_basis_candidate_refs(source_basis: SourceBasis) -> list[str]:
    return _markdown_utils.dedupe_strings(
        [
            str(ref).strip()
            for ref in [*source_basis.source_candidate_ids, *source_basis.prepared_discovered_candidates]
            if str(ref).strip()
        ]
    )


def source_digest_candidate_lookup(digest: SourceDigestArtifact) -> dict[str, SourceDigestCandidate]:
    candidates = {candidate.candidate_id: candidate for candidate in digest.ingest_candidates()}
    for candidate in digest.budget_deferred_candidates:
        candidates.setdefault(candidate.candidate_id, candidate)
    return candidates


def source_digest_candidate_id_closure(
    candidate_ids: list[str],
    candidate_by_id: dict[str, SourceDigestCandidate],
) -> list[str]:
    needed: list[str] = []
    seen: set[str] = set()
    queue = [candidate_id for candidate_id in candidate_ids if candidate_id]
    while queue:
        candidate_id = queue.pop(0)
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        needed.append(candidate_id)
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            continue
        for related_id in candidate.related_candidates:
            if related_id and related_id not in seen:
                queue.append(related_id)
    return needed


def first_source_basis_candidate(
    source_basis: SourceBasis,
    candidates: dict[str, SourceDigestCandidate],
) -> SourceDigestCandidate | None:
    for candidate_id in source_basis_candidate_refs(source_basis):
        candidate = candidates.get(candidate_id)
        if candidate is not None:
            return candidate
    return None
