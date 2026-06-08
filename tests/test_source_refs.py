import llmwiki_engine.source_refs as source_refs
from llmwiki_engine.models import SourceBasis, SourceDigestArtifact, SourceDigestCandidate


def _candidate(candidate_id: str, *, related: list[str] | None = None, name: str | None = None) -> SourceDigestCandidate:
    return SourceDigestCandidate(
        candidate_id=candidate_id,
        name=name or candidate_id,
        type="concept",
        one_sentence_summary=f"{candidate_id} summary",
        why_matters=f"{candidate_id} matters",
        wiki_value=f"{candidate_id} value",
        related_candidates=related or [],
    )


def test_source_basis_candidate_refs_keeps_order_and_dedupes() -> None:
    basis = SourceBasis(
        source_candidate_ids=[" C1 ", "C2", "C1", ""],
        prepared_discovered_candidates=["C2", " C3 "],
    )

    assert source_refs.source_basis_candidate_refs(basis) == ["C1", "C2", "C3"]


def test_source_digest_candidate_lookup_keeps_ingest_candidate_over_deferred_duplicate() -> None:
    digest = SourceDigestArtifact(
        source_raw_path="raw/source.md",
        summary="summary",
        concepts=[_candidate("C1", name="ingest")],
        budget_deferred_candidates=[_candidate("C1", name="deferred"), _candidate("C2")],
    )

    lookup = source_refs.source_digest_candidate_lookup(digest)

    assert lookup["C1"].name == "ingest"
    assert lookup["C2"].name == "C2"


def test_source_digest_candidate_id_closure_expands_related_without_looping() -> None:
    candidates = {
        "C1": _candidate("C1", related=["C2", "C3"]),
        "C2": _candidate("C2", related=["C3", "C1"]),
        "C3": _candidate("C3"),
    }

    assert source_refs.source_digest_candidate_id_closure(["C1", "missing"], candidates) == ["C1", "missing", "C2", "C3"]


def test_first_source_basis_candidate_uses_direct_refs_only() -> None:
    candidates = {
        "C1": _candidate("C1", related=["C2"]),
        "C2": _candidate("C2"),
    }
    basis = SourceBasis(source_candidate_ids=["missing", "C1"], prepared_discovered_candidates=["C2"])

    assert source_refs.first_source_basis_candidate(basis, candidates) == candidates["C1"]
