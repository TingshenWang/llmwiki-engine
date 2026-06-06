from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import warnings
from datetime import date, datetime
from pathlib import Path
from time import perf_counter
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
SCORE_BUCKET_EPSILON = 0.01
LEXICAL_EXPANSION_MAX_WEAK_SCORE = 0.61
LEXICAL_EXPANSION_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "agent_harness_runtime",
        (
            "harness",
            "meta-harness",
            "元适配",
            "模型调用框架",
            "agent runtime",
            "agent运行时",
            "代理运行时",
            "managed agents",
            "托管代理",
            "托管智能体",
            "工具编排",
            "tool orchestration",
        ),
    ),
    (
        "durable_context_session",
        (
            "durable context",
            "persistent context",
            "context object",
            "session object",
            "持久上下文",
            "上下文对象",
            "会话对象",
            "会话作为",
        ),
    ),
    (
        "brain_hand_decoupling",
        (
            "brain and hands",
            "brain-hand",
            "大脑与双手",
            "大脑和双手",
            "解耦大脑",
            "解耦架构",
        ),
    ),
    (
        "pets_cattle_operations",
        (
            "pets vs cattle",
            "pet vs cattle",
            "宠物与牲畜",
            "宠物 vs 牛",
            "一次性环境",
            "可替换实例",
        ),
    ),
)


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
    model_revision = "unknown"
    embedding_metrics: dict[str, int] = {}
    if backend == "sentence_transformers" and knowledge_pool:
        try:
            embeddings, model_revision, embedding_metrics = SentenceTransformerRanker(config).rank_inputs(resolution.items, knowledge_pool, vault)
        except RetrievalError:
            raise
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
        local_files_only=config.local_files_only if backend == "sentence_transformers" else False,
        cache_dir=config.cache_dir,
        embedding_load_duration_ms=embedding_metrics.get("load_duration_ms", 0),
        embedding_encode_duration_ms=embedding_metrics.get("encode_duration_ms", 0),
        embedding_total_duration_ms=embedding_metrics.get("total_duration_ms", 0),
        embedding_page_vector_cache_hit=bool(embedding_metrics.get("page_vector_cache_hit", 0)),
        embedding_page_count=embedding_metrics.get("page_count", 0),
        embedding_query_count=embedding_metrics.get("query_count", 0),
        embedding_text_char_count=embedding_metrics.get("text_char_count", 0),
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
        text_for_entry = candidate_text(vault, entry)
        lexical = lexical_score(query, text_for_entry)
        lexical_expansion = lexical_expansion_score(query, text_for_entry)
        embedding = embedding_scores.get(entry.path, 0.0) if embedding_scores else 0.0
        scored_bases = [
            (signal[0], signal[1]),
            ("lexical_expansion", lexical_expansion),
            ("lexical", lexical),
        ]
        if embedding_scores is not None:
            scored_bases.append(("embedding", embedding))
        basis, score = max(scored_bases, key=lambda value: (value[1], basis_rank(value[0])))
        if not basis:
            basis = "lexical"
        strength = strength_for_score(score, basis, config)
        text = (vault / "wiki" / entry.path).read_text(encoding="utf-8")
        excerpt, truncated = excerpt_text(text, config.max_excerpt_chars)
        score_bucket = retrieval_score_bucket(score)
        hits.append(
            CandidateContextHit(
                page_plan_id=item.page_plan_id,
                rank=0,
                path=entry.path,
                display_title=entry.display_title,
                score=round(score, 4),
                score_bucket=score_bucket,
                strength=strength,
                match_basis=basis,
                forced=basis.startswith("exact") or basis.startswith("normalized"),
                page_sha256=entry.preimage_sha256,
                excerpt=excerpt,
                truncated=truncated,
            )
        )
    entry_by_path = {entry.path: entry for entry in knowledge_pool}
    for hit in hits:
        hit.sort_explanation = retrieval_sort_explanation(hit, item, entry_by_path)
    hits.sort(key=lambda hit: retrieval_sort_key(hit, item, entry_by_path))
    return hits


def retrieval_sort_key(
    hit: CandidateContextHit,
    item: CandidateResolutionItem,
    entry_by_path: dict[str, WikiKnowledgePoolEntry],
) -> tuple[int, int, int, int, int, int, str]:
    return retrieval_sort_components(hit, item, entry_by_path)["key"]


def retrieval_sort_components(
    hit: CandidateContextHit,
    item: CandidateResolutionItem,
    entry_by_path: dict[str, WikiKnowledgePoolEntry],
) -> dict[str, Any]:
    entry = entry_by_path.get(hit.path)
    score_bucket = retrieval_score_bucket(hit.score)
    same_type = entry is not None and entry.llmwiki_type == item.page_type
    same_directory = same_target_directory(item.candidate_target_path, hit.path)
    distance = title_distance(item, entry)
    key = (
        -strength_rank(hit.strength),
        -score_bucket,
        -basis_rank(hit.match_basis),
        0 if same_type else 1,
        0 if same_directory else 1,
        distance,
        hit.path,
    )
    return {
        "key": key,
        "score_bucket": score_bucket,
        "strength_rank": strength_rank(hit.strength),
        "basis_rank": basis_rank(hit.match_basis),
        "same_type": same_type,
        "same_directory": same_directory,
        "title_distance": distance,
    }


def retrieval_sort_explanation(
    hit: CandidateContextHit,
    item: CandidateResolutionItem,
    entry_by_path: dict[str, WikiKnowledgePoolEntry],
) -> str:
    components = retrieval_sort_components(hit, item, entry_by_path)
    return (
        f"bucket={components['score_bucket']}; "
        f"strength_rank={components['strength_rank']}; "
        f"basis_rank={components['basis_rank']}; "
        f"type={'same' if components['same_type'] else 'different'}; "
        f"dir={'same' if components['same_directory'] else 'different'}; "
        f"title_distance={components['title_distance']}; "
        f"path={hit.path}"
    )


def retrieval_score_bucket(score: float) -> int:
    return math.floor(score / SCORE_BUCKET_EPSILON)


def basis_rank(value: str) -> int:
    return {
        "exact_path": 5,
        "exact_title_or_alias": 4,
        "normalized_path_stem": 3,
        "normalized_title": 2,
        "lexical_expansion": 2,
        "embedding": 1,
        "lexical": 1,
    }.get(value, 0)


def same_target_directory(candidate_target_path: str, hit_path: str) -> bool:
    candidate_parts = Path(candidate_target_path).parts
    hit_parts = Path(hit_path).parts
    return bool(candidate_parts and hit_parts and candidate_parts[0] == hit_parts[0])


def title_distance(item: CandidateResolutionItem, entry: WikiKnowledgePoolEntry | None) -> int:
    if entry is None:
        return 3
    item_title = normalize_key(item.display_title)
    titles = [normalize_key(entry.display_title), *(normalize_key(alias) for alias in entry.aliases)]
    if item_title and item_title in titles:
        return 0
    if item_title and any(item_title in title or title in item_title for title in titles if title):
        return 1
    return 2


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
    values = [
        entry.display_title,
        " ".join(entry.aliases),
        entry.summary,
        first_heading(text),
        headings_text(text),
        salient_body_text(text),
    ]
    return "\n".join(value for value in values if value)


def embedding_candidate_text(vault: Path, entry: WikiKnowledgePoolEntry, max_chars: int) -> str:
    path = vault / "wiki" / entry.path
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    values = [
        entry.display_title,
        " ".join(entry.aliases),
        entry.summary,
        headings_text(text),
        salient_body_text(text, max_chars=max(120, max_chars // 2)),
    ]
    return trim_embedding_text("\n".join(value for value in values if value), max_chars)


def trim_embedding_text(text: str, max_chars: int) -> str:
    compact = re.sub(r"\n{3,}", "\n\n", text.strip())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rstrip()


def embedding_page_vector_cache_path(
    vault: Path,
    config: EmbeddingRetrievalConfig,
    revision: str,
    knowledge_pool: list[WikiKnowledgePoolEntry],
) -> Path:
    cache_dir = resolve_cache_dir(vault, config.cache_dir) / "vector-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha256(
        json.dumps(
            {
                "schema_version": "embedding_page_vector_cache.v1",
                "model": config.model,
                "revision": revision,
                "candidate_pool_sha256": candidate_pool_sha256(knowledge_pool),
                "max_embedding_page_chars": config.max_embedding_page_chars,
                "input_version": "embedding_candidate_text.v1",
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return cache_dir / f"{cache_key}.json"


def read_embedding_page_vector_cache(path: Path, entry_paths: list[str]) -> list[list[float]] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("schema_version") != "embedding_page_vector_cache.v1":
        return None
    if payload.get("entry_paths") != entry_paths:
        return None
    vectors = payload.get("vectors")
    if not isinstance(vectors, list) or len(vectors) != len(entry_paths):
        return None
    return vectors


def write_embedding_page_vector_cache(path: Path, *, entry_paths: list[str], vectors: Any) -> None:
    try:
        vector_payload = vectors.tolist() if hasattr(vectors, "tolist") else [list(vector) for vector in vectors]
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "schema_version": "embedding_page_vector_cache.v1",
                    "entry_paths": entry_paths,
                    "vectors": vector_payload,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        tmp.replace(path)
    except Exception:
        return


def lexical_score(query: str, text: str) -> float:
    query_tokens = set(tokens(query))
    text_tokens = set(tokens(text))
    if not query_tokens or not text_tokens:
        return 0.0
    overlap = len(query_tokens & text_tokens) / max(1, len(query_tokens))
    bigram_overlap = overlap_ratio(cjk_bigrams(query), cjk_bigrams(text))
    char_overlap = overlap_ratio(cjk_chars(query), cjk_chars(text))
    return min(0.95, max(overlap, bigram_overlap * 0.55, char_overlap * 0.2))


def overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left)


def cjk_chars(text: str) -> set[str]:
    return {char for char in normalize_key(text) if "\u4e00" <= char <= "\u9fff"}


def cjk_bigrams(text: str) -> set[str]:
    bigrams: set[str] = set()
    for segment in re.findall(r"[\u4e00-\u9fff]{2,}", normalize_key(text)):
        bigrams.update(segment[index : index + 2] for index in range(len(segment) - 1))
    return bigrams


def lexical_expansion_score(query: str, text: str) -> float:
    query_norm = normalize_key(query)
    text_norm = normalize_key(text)
    if not query_norm or not text_norm:
        return 0.0
    matched_groups = 0
    matched_terms = 0
    for _, terms in LEXICAL_EXPANSION_GROUPS:
        query_hits = [term for term in terms if lexical_term_present(query_norm, term)]
        text_hits = [term for term in terms if lexical_term_present(text_norm, term)]
        if query_hits and text_hits:
            matched_groups += 1
            matched_terms += min(len(query_hits), len(text_hits))
    if not matched_groups:
        return 0.0
    score = 0.53 + matched_groups * 0.055 + min(matched_terms, 4) * 0.0125
    return min(LEXICAL_EXPANSION_MAX_WEAK_SCORE, score)


def lexical_term_present(normalized_text: str, term: str) -> bool:
    normalized_term = normalize_key(term)
    if not normalized_term:
        return False
    if normalized_term in normalized_text:
        return True
    term_tokens = tokens(normalized_term)
    if len(term_tokens) <= 1:
        return False
    text_tokens = set(tokens(normalized_text))
    return set(term_tokens).issubset(text_tokens)


def tokens(text: str) -> list[str]:
    normalized = normalize_key(text)
    words = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", normalized)
    return words


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


def headings_text(text: str) -> str:
    headings: list[str] = []
    for line in text.splitlines():
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            if heading:
                headings.append(heading)
    return "\n".join(headings[:12])


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


def salient_body_text(text: str, max_chars: int = 1800) -> str:
    body = text.split("---\n", 2)[-1] if text.startswith("---\n") else text
    lines: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("---"):
            continue
        lines.append(stripped)
        if len("\n".join(lines)) >= max_chars:
            break
    return "\n".join(lines)[:max_chars]


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
    ) -> tuple[dict[str, dict[str, float]], str, dict[str, int]]:
        configure_huggingface_quiet_mode()
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except Exception as exc:
            raise RetrievalError("sentence-transformers is not installed; install the embedding extra or set embedding_retrieval.backend=exact for tests") from exc
        total_start = perf_counter()
        cache_dir = resolve_cache_dir(vault, self.config.cache_dir)
        try:
            load_start = perf_counter()
            model = SentenceTransformer(
                self.config.model,
                device=self.config.device,
                cache_folder=str(cache_dir),
                local_files_only=self.config.local_files_only,
                token=False,
            )
            load_duration_ms = round((perf_counter() - load_start) * 1000)
        except Exception as exc:
            if self.config.local_files_only:
                raise RetrievalError(
                    "embedding model is not available in the local cache; "
                    f"model={self.config.model}, cache_dir={cache_dir}. "
                    "Populate the cache first or set embedding_retrieval.local_files_only=false explicitly for a download-enabled run."
                ) from exc
            raise
        revision = model_revision(model)
        candidate_texts = [
            embedding_candidate_text(vault, entry, self.config.max_embedding_page_chars)
            for entry in knowledge_pool
        ]
        query_texts = [
            trim_embedding_text(query_for_item(item), self.config.max_embedding_query_chars)
            for item in items
        ]
        encode_start = perf_counter()
        page_cache_hit = False
        cache_path = embedding_page_vector_cache_path(vault, self.config, revision, knowledge_pool)
        candidate_vectors = None
        if self.config.page_vector_cache:
            candidate_vectors = read_embedding_page_vector_cache(cache_path, [entry.path for entry in knowledge_pool])
            page_cache_hit = candidate_vectors is not None
        if candidate_vectors is None:
            candidate_vectors = model.encode(candidate_texts, normalize_embeddings=True, show_progress_bar=False)
            if self.config.page_vector_cache:
                write_embedding_page_vector_cache(
                    cache_path,
                    entry_paths=[entry.path for entry in knowledge_pool],
                    vectors=candidate_vectors,
                )
        query_vectors = model.encode(query_texts, normalize_embeddings=True, show_progress_bar=False)
        encode_duration_ms = round((perf_counter() - encode_start) * 1000)
        scores: dict[str, dict[str, float]] = {}
        for item, query_vector in zip(items, query_vectors, strict=True):
            item_scores: dict[str, float] = {}
            for entry, candidate_vector in zip(knowledge_pool, candidate_vectors, strict=True):
                item_scores[entry.path] = float(dot(query_vector, candidate_vector))
            scores[item.page_plan_id] = item_scores
        return scores, revision, {
            "load_duration_ms": load_duration_ms,
            "encode_duration_ms": encode_duration_ms,
            "total_duration_ms": round((perf_counter() - total_start) * 1000),
            "page_vector_cache_hit": 1 if page_cache_hit else 0,
            "page_count": len(candidate_texts),
            "query_count": len(query_texts),
            "text_char_count": sum(len(text) for text in [*candidate_texts, *query_texts]),
        }


def configure_huggingface_quiet_mode() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    warnings.filterwarnings("ignore", message=".*unauthenticated requests.*HF Hub.*")
    warnings.filterwarnings("ignore", message=".*You are sending unauthenticated requests.*")
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)


def dot(left: Any, right: Any) -> float:
    try:
        return float(left @ right)
    except Exception:
        return float(sum(a * b for a, b in zip(left, right, strict=True)) / max(1e-9, math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))))


def model_revision(model: Any) -> str:
    candidates: list[Any] = []
    try:
        first_module = model._first_module()
        candidates.extend(
            [
                getattr(getattr(getattr(first_module, "auto_model", None), "config", None), "_commit_hash", None),
                getattr(getattr(getattr(first_module, "tokenizer", None), "init_kwargs", {}), "_commit_hash", None),
            ]
        )
    except Exception:
        pass
    candidates.extend(
        [
            getattr(model, "revision", None),
            getattr(model, "_model_revision", None),
        ]
    )
    for value in candidates:
        if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{7,40}", value):
            return value
    return "unknown"


def resolve_cache_dir(vault: Path, cache_dir: str) -> Path:
    path = Path(cache_dir).expanduser()
    if not path.is_absolute():
        path = vault / path
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()
