from __future__ import annotations

import hashlib
import math
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from .hash_utils import sha256_file
from .models import (
    CandidateContextHit,
    CandidateContextItem,
    CandidateContextsArtifact,
    CandidateResolutionArtifact,
    CandidateResolutionItem,
    EmbeddingRetrievalConfig,
    WikiKnowledgePoolEntry,
    WikiPageMetadata,
)


WIKI_EXCLUDED_ROOTS = {"sources", "logs"}
SYSTEM_WIKI_FILES = {"index.md", "log.md"}
TYPE_PREFIXES = ("concept_", "entity_", "design_", "comparison_", "overview_", "event_", "memory_", "idea_", "open_question_")


class RetrievalError(RuntimeError):
    pass


def build_knowledge_pool(vault: Path) -> list[WikiKnowledgePoolEntry]:
    wiki = vault / "wiki"
    if not wiki.exists():
        return []
    entries: list[WikiKnowledgePoolEntry] = []
    for path in sorted(wiki.rglob("*.md")):
        rel_to_wiki = path.relative_to(wiki).as_posix()
        if rel_to_wiki in SYSTEM_WIKI_FILES:
            continue
        parts = Path(rel_to_wiki).parts
        if parts and parts[0] in WIKI_EXCLUDED_ROOTS:
            continue
        text = path.read_text(encoding="utf-8")
        metadata = metadata_from_text(text, f"wiki/{rel_to_wiki}")
        if metadata is not None and metadata.llmwiki_type.lower() == "source":
            continue
        heading = first_heading(text)
        display_title = clean_title((metadata.title if metadata is not None else "") or heading or path.stem)
        summary = metadata.summary if metadata is not None else first_paragraph(text)
        aliases = metadata.aliases if metadata is not None else []
        indexable = metadata is not None
        entries.append(
            WikiKnowledgePoolEntry(
                path=rel_to_wiki,
                rel_path=f"wiki/{rel_to_wiki}",
                preimage_sha256=sha256_file(path),
                metadata=metadata,
                display_title=display_title,
                summary=summary,
                aliases=aliases,
                llmwiki_type=metadata.llmwiki_type if metadata is not None else "unknown",
                indexable=indexable,
                unindexable_reason="" if indexable else "missing_or_incomplete_frontmatter",
            )
        )
    return entries


def build_candidate_contexts(
    *,
    resolution: CandidateResolutionArtifact,
    knowledge_pool: list[WikiKnowledgePoolEntry],
    config: EmbeddingRetrievalConfig,
    vault: Path,
    force_exact_backend: bool = False,
) -> CandidateContextsArtifact:
    backend = "exact" if force_exact_backend or config.backend == "exact" or not config.enabled else config.backend
    warnings: list[str] = []
    top_k = max(1, config.top_k)
    candidate_pool_sha = candidate_pool_sha256(knowledge_pool)
    items: list[CandidateContextItem] = []
    embeddings = None
    model_revision = ""
    if backend == "sentence_transformers" and knowledge_pool:
        try:
            embeddings, model_revision = SentenceTransformerRanker(config).rank_inputs(resolution.items, knowledge_pool, vault)
        except Exception as exc:
            raise RetrievalError(f"embedding retrieval unavailable: {exc}") from exc
    for item in resolution.items:
        query = query_for_item(item)
        ranked = rank_candidates(
            item=item,
            query=query,
            knowledge_pool=knowledge_pool,
            vault=vault,
            config=config,
            embedding_scores=embeddings.get(item.page_plan_id, {}) if embeddings else None,
        )
        hits = ranked[:top_k]
        for rank, hit in enumerate(hits, start=1):
            hit.rank = rank
        unindexable = [entry.path for entry in knowledge_pool if not entry.indexable]
        items.append(CandidateContextItem(page_plan_id=item.page_plan_id, query=query, hits=hits, unindexable_pages=unindexable[:50]))
    return CandidateContextsArtifact(
        retrieval_backend=backend,
        model=config.model if backend == "sentence_transformers" else "exact-lexical",
        model_revision=model_revision,
        cache_dir=config.cache_dir,
        top_k=top_k,
        candidate_pool_size=len(knowledge_pool),
        candidate_pool_sha256=candidate_pool_sha,
        skipped_count=sum(1 for entry in knowledge_pool if not entry.indexable),
        warnings=warnings,
        items=items,
    )


def rank_candidates(
    *,
    item: CandidateResolutionItem,
    query: str,
    knowledge_pool: list[WikiKnowledgePoolEntry],
    vault: Path,
    config: EmbeddingRetrievalConfig,
    embedding_scores: dict[str, float] | None = None,
) -> list[CandidateContextHit]:
    hits: list[CandidateContextHit] = []
    for entry in knowledge_pool:
        signal = deterministic_signal(item, entry)
        lexical = lexical_score(query, candidate_text(vault, entry))
        embedding = embedding_scores.get(entry.path, 0.0) if embedding_scores else 0.0
        score = max(signal[1], lexical, embedding)
        basis = signal[0] if signal[0] else ("embedding" if embedding_scores is not None and embedding >= lexical else "lexical")
        strength = strength_for_score(score, basis, config)
        text = (vault / "wiki" / entry.path).read_text(encoding="utf-8")
        excerpt, truncated = excerpt_text(text, config.max_excerpt_chars)
        hits.append(
            CandidateContextHit(
                page_plan_id=item.page_plan_id,
                rank=0,
                path=entry.path,
                display_title=entry.display_title,
                score=round(score, 4),
                strength=strength,
                match_basis=basis,
                forced=basis.startswith("exact") or basis.startswith("normalized"),
                page_sha256=entry.preimage_sha256,
                excerpt=excerpt,
                truncated=truncated,
            )
        )
    hits.sort(key=lambda hit: (-strength_rank(hit.strength), -hit.score, hit.path))
    return hits


def deterministic_signal(item: CandidateResolutionItem, entry: WikiKnowledgePoolEntry) -> tuple[str, float]:
    target = normalize_path(item.candidate_target_path)
    entry_path = normalize_path(entry.path)
    item_title = normalize_key(item.display_title)
    entry_titles = [normalize_key(entry.display_title), *(normalize_key(alias) for alias in entry.aliases)]
    if target and target == entry_path:
        return "exact_path", 1.0
    if item_title and item_title in entry_titles:
        return "exact_title_or_alias", 0.98
    item_stem = normalize_key(Path(target).stem if target else item.path_stem or item.display_title)
    entry_stem = normalize_key(Path(entry_path).stem)
    if item_stem and item_stem == entry_stem:
        return "normalized_path_stem", 0.82
    if item_title and any(item_title in title or title in item_title for title in entry_titles if title):
        return "normalized_title", 0.72
    return "", 0.0


def strength_for_score(score: float, basis: str, config: EmbeddingRetrievalConfig) -> str:
    if basis in {"exact_path", "exact_title_or_alias"}:
        return "strong"
    if basis in {"normalized_path_stem"}:
        return "strong"
    if basis.startswith("normalized"):
        return "medium"
    if score >= config.strong_score:
        return "strong"
    if score >= config.medium_score:
        return "medium"
    return "weak"


def strength_rank(value: str) -> int:
    return {"strong": 3, "medium": 2, "weak": 1}.get(value, 0)


def candidate_pool_sha256(pool: list[WikiKnowledgePoolEntry]) -> str:
    payload = [
        {"path": entry.path, "sha256": entry.preimage_sha256, "indexable": entry.indexable}
        for entry in sorted(pool, key=lambda item: item.path)
    ]
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def query_for_item(item: CandidateResolutionItem) -> str:
    values = [
        item.display_title,
        item.path_stem,
        item.topic_summary,
        item.why_this_page,
        item.initial_section_intent,
        item.coverage_notes,
        item.reason,
        " ".join(item.source_basis.source_candidate_ids),
        " ".join(item.source_basis.prepared_discovered_candidates),
    ]
    return "\n".join(value for value in values if value and value.strip())


def candidate_text(vault: Path, entry: WikiKnowledgePoolEntry) -> str:
    path = vault / "wiki" / entry.path
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    values = [entry.display_title, " ".join(entry.aliases), entry.summary, first_heading(text), first_paragraph(text)]
    return "\n".join(value for value in values if value)


def lexical_score(query: str, text: str) -> float:
    query_tokens = set(tokens(query))
    text_tokens = set(tokens(text))
    if not query_tokens or not text_tokens:
        return 0.0
    overlap = len(query_tokens & text_tokens) / max(1, len(query_tokens))
    char_overlap = len(set(normalize_key(query)) & set(normalize_key(text))) / max(1, len(set(normalize_key(query))))
    return min(0.95, max(overlap, char_overlap * 0.6))


def tokens(text: str) -> list[str]:
    normalized = normalize_key(text)
    words = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", normalized)
    chars = [char for char in normalized if "\u4e00" <= char <= "\u9fff"]
    return [*words, *chars]


def normalize_key(value: str) -> str:
    text = re.sub(r"\s+", " ", value.strip().lower())
    for prefix in TYPE_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix) :]
    return text


def normalize_path(value: str) -> str:
    path = value.strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path.removeprefix("wiki/")


def metadata_from_text(text: str, rel_path: str) -> WikiPageMetadata | None:
    if not text.startswith("---\n"):
        return None
    parts = text.split("---\n", 2)
    if len(parts) < 3:
        return None
    data = yaml.safe_load(parts[1]) or {}
    if not isinstance(data, dict):
        return None
    llmwiki_type = data.get("llmwiki_type")
    title = data.get("title")
    summary = data.get("summary")
    updated = data.get("updated")
    if isinstance(created := data.get("created", ""), (datetime, date)):
        created = created.isoformat()
    if isinstance(updated, (datetime, date)):
        updated = updated.isoformat()
    if not all(isinstance(value, str) and value.strip() for value in [llmwiki_type, title, summary, updated]):
        return None
    aliases_raw = data.get("aliases", [])
    aliases = [item for item in aliases_raw if isinstance(item, str)] if isinstance(aliases_raw, list) else []
    return WikiPageMetadata(
        path=rel_path.removeprefix("wiki/"),
        llmwiki_type=llmwiki_type,
        title=title,
        summary=summary,
        created=created if isinstance(created, str) else "",
        updated=updated,
        aliases=aliases,
        source_raw_paths=frontmatter_list(data, "source_raw_paths"),
        source_raw_hashes=frontmatter_list(data, "source_raw_hashes"),
        source_prepared_hashes=frontmatter_list(data, "source_prepared_hashes"),
        source_operation_ids=frontmatter_list(data, "source_operation_ids"),
    )


def frontmatter_list(frontmatter: dict[str, Any], key: str) -> list[str]:
    value = frontmatter.get(key, [])
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def clean_title(value: str) -> str:
    return re.sub(r"^(Concept|Entity|Design|Comparison|Open Question)_", "", value).strip() or "Untitled"


def first_heading(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def first_paragraph(text: str) -> str:
    body = text.split("---\n", 2)[-1] if text.startswith("---\n") else text
    lines: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            if lines:
                break
            continue
        lines.append(stripped)
        if len(" ".join(lines)) > 240:
            break
    return " ".join(lines)[:300]


def excerpt_text(text: str, max_chars: int) -> tuple[str, bool]:
    body = text.split("---\n", 2)[-1] if text.startswith("---\n") else text
    compact = re.sub(r"\n{3,}", "\n\n", body).strip()
    if len(compact) <= max_chars:
        return compact, False
    return compact[:max_chars].rstrip() + "\n...", True


class SentenceTransformerRanker:
    def __init__(self, config: EmbeddingRetrievalConfig) -> None:
        self.config = config

    def rank_inputs(
        self,
        items: list[CandidateResolutionItem],
        knowledge_pool: list[WikiKnowledgePoolEntry],
        vault: Path,
    ) -> tuple[dict[str, dict[str, float]], str]:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except Exception as exc:
            raise RetrievalError("sentence-transformers is not installed; install the embedding extra or set embedding_retrieval.backend=exact for tests") from exc
        model = SentenceTransformer(
            self.config.model,
            device=self.config.device,
            cache_folder=str(resolve_cache_dir(vault, self.config.cache_dir)),
        )
        candidate_texts = [candidate_text(vault, entry) for entry in knowledge_pool]
        query_texts = [query_for_item(item) for item in items]
        candidate_vectors = model.encode(candidate_texts, normalize_embeddings=True)
        query_vectors = model.encode(query_texts, normalize_embeddings=True)
        scores: dict[str, dict[str, float]] = {}
        for item, query_vector in zip(items, query_vectors, strict=True):
            item_scores: dict[str, float] = {}
            for entry, candidate_vector in zip(knowledge_pool, candidate_vectors, strict=True):
                item_scores[entry.path] = float(dot(query_vector, candidate_vector))
            scores[item.page_plan_id] = item_scores
        revision = getattr(model, "model_card_data", None)
        return scores, str(revision) if revision is not None else ""


def dot(left: Any, right: Any) -> float:
    try:
        return float(left @ right)
    except Exception:
        return float(sum(a * b for a, b in zip(left, right, strict=True)) / max(1e-9, math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))))


def resolve_cache_dir(vault: Path, cache_dir: str) -> Path:
    path = Path(cache_dir).expanduser()
    if not path.is_absolute():
        path = vault / path
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()
