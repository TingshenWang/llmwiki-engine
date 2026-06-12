from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .io import now_utc, read_json, read_text, safe_filename, sha256_text, stable_json_hash, write_json
from .models import CandidateContext, CandidateContextHit, CandidateContexts, CandidatePage, CandidatePages, WikiKnowledgeEntry
from .text import strip_frontmatter


CACHE_SCHEMA_VERSION = "lite_page_embedding_cache.v1"
INPUT_VERSION = "page_card_v1"
DEFAULT_QWEN_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"


@dataclass(frozen=True)
class EmbeddingConfig:
    enabled: bool = True
    backend: str = "sentence_transformers"
    cache_dir: str = ".llmwiki/cache/embeddings"
    top_k_pages: int = 5
    dimensions: int = 1024
    input_version: str = INPUT_VERSION
    max_page_chars: int = 6000
    max_query_chars: int = 4000
    max_excerpt_chars: int = 800
    batch_size: int = 8
    normalize_embeddings: bool = True
    query_prompt_name: str = "query"


def load_embedding_config(config: dict[str, object]) -> EmbeddingConfig:
    raw = config.get("embedding")
    if not isinstance(raw, dict):
        raw = {}
    backend = str(raw.get("backend", "sentence_transformers"))
    return EmbeddingConfig(
        enabled=bool(raw.get("enabled", True)),
        backend=backend,
        cache_dir=str(raw.get("cache_dir", ".llmwiki/cache/embeddings")),
        top_k_pages=int(raw.get("top_k_pages", 5)),
        dimensions=int(raw.get("dimensions", 1024)),
        input_version=str(raw.get("input_version", INPUT_VERSION)),
        max_page_chars=int(raw.get("max_page_chars", 6000)),
        max_query_chars=int(raw.get("max_query_chars", 4000)),
        max_excerpt_chars=int(raw.get("max_excerpt_chars", 800)),
        batch_size=int(raw.get("batch_size", 8)),
        normalize_embeddings=bool(raw.get("normalize_embeddings", True)),
        query_prompt_name=str(raw.get("query_prompt_name", "query")),
    )


def sync_page_embedding_cache(
    vault: Path,
    entries: list[WikiKnowledgeEntry],
    config: EmbeddingConfig,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    pages_dir = _pages_cache_dir(vault, config)
    pages_dir.mkdir(parents=True, exist_ok=True)
    expected_files: set[Path] = set()
    records: dict[str, dict[str, Any]] = {}
    hits = 0
    misses = 0
    stale = 0
    refreshed = 0
    text_char_count = 0
    pending: list[tuple[WikiKnowledgeEntry, Path, str, str]] = []
    hit_paths: list[str] = []
    created_paths: list[str] = []
    updated_paths: list[str] = []
    for entry in entries:
        cache_path = _record_path(pages_dir, entry.path)
        expected_files.add(cache_path)
        card = page_card(entry, vault, config.max_page_chars)
        text_char_count += len(card)
        input_sha = sha256_text(card)
        cached = _read_valid_record(cache_path, entry, config, input_sha)
        if cached is not None:
            hits += 1
            hit_paths.append(entry.path)
            cached["cache_hit"] = True
            records[entry.path] = cached
            continue
        if cache_path.exists():
            stale += 1
            updated_paths.append(entry.path)
        else:
            misses += 1
            created_paths.append(entry.path)
        pending.append((entry, cache_path, input_sha, card))
    if pending:
        vectors = embed_texts([item[3] for item in pending], config, is_query=False)
        for (entry, cache_path, input_sha, _card), vector in zip(pending, vectors, strict=True):
            record = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "path": entry.path,
                "title": entry.title,
                "page_type": entry.page_type,
                "content_sha256": entry.sha256,
                "input_sha256": input_sha,
                **embedding_contract(config),
                "vector": vector,
                "updated_at": now_utc(),
                "cache_hit": False,
            }
            write_json(cache_path, record)
            refreshed += 1
            records[entry.path] = record
    pruned = 0
    pruned_paths: list[str] = []
    for path in pages_dir.glob("*.json"):
        if path not in expected_files:
            try:
                payload = read_json(path)
                if isinstance(payload, dict) and isinstance(payload.get("path"), str):
                    pruned_paths.append(payload["path"])
                else:
                    pruned_paths.append(path.name)
            except Exception:
                pruned_paths.append(path.name)
            path.unlink()
            pruned += 1
    write_json(
        pages_dir.parent / "index.json",
        {
            "schema_version": CACHE_SCHEMA_VERSION,
            **embedding_contract(config),
            "page_paths": [entry.path for entry in entries],
            "page_count": len(entries),
            "updated_at": now_utc(),
        },
    )
    metrics: dict[str, Any] = {
        "backend": config.backend,
        "model": DEFAULT_QWEN_EMBEDDING_MODEL,
        "input_version": config.input_version,
        "dimensions": config.dimensions,
        "max_page_chars": config.max_page_chars,
        "page_count": len(entries),
        "cache_hit": hits,
        "cache_miss": misses,
        "cache_stale": stale,
        "cache_refreshed": refreshed,
        "cache_pruned": pruned,
        "cache_hit_paths": hit_paths,
        "cache_created_paths": created_paths,
        "cache_updated_paths": updated_paths,
        "cache_refreshed_paths": [*created_paths, *updated_paths],
        "cache_pruned_paths": pruned_paths,
        "text_char_count": text_char_count,
        "current_state_only": True,
    }
    return records, metrics


def build_candidate_contexts(
    candidate_pages: CandidatePages,
    entries: list[WikiKnowledgeEntry],
    page_records: dict[str, dict[str, Any]],
    config: EmbeddingConfig,
) -> CandidateContexts:
    require_real_embedding_backend(config)
    top_k = max(1, config.top_k_pages)
    items: list[CandidateContext] = []
    queries = [candidate_query(page, config.max_query_chars) for page in candidate_pages.pages]
    query_vectors = embed_texts(queries, config, is_query=True) if queries else []
    for page, query, query_vector in zip(candidate_pages.pages, queries, query_vectors, strict=True):
        hits: list[CandidateContextHit] = []
        for entry in entries:
            record = page_records.get(entry.path)
            vector = record.get("vector", []) if isinstance(record, dict) else []
            if not isinstance(record, dict) or not isinstance(vector, list) or not vector:
                raise RuntimeError(f"候选召回必须使用 embedding cache，但页面缺少有效向量：{entry.path}")
            embedding = cosine(query_vector, vector) if isinstance(vector, list) else 0.0
            excerpt, truncated = excerpt_text(entry.text_excerpt, config.max_excerpt_chars)
            hits.append(
                CandidateContextHit(
                    path=entry.path,
                    title=entry.title,
                    rank=0,
                    page_type=entry.page_type,
                    page_sha256=entry.sha256,
                    score=round(embedding, 4),
                    match_basis="embedding",
                    reason=_hit_reason(embedding),
                    excerpt=excerpt,
                    truncated=truncated,
                    embedding_cache_hit=bool(record.get("cache_hit")) if isinstance(record, dict) else False,
                )
            )
        hits.sort(key=lambda hit: (-hit.score, hit.path))
        ranked = [hit.model_copy(update={"rank": index}) for index, hit in enumerate(hits[:top_k], start=1)]
        items.append(CandidateContext(candidate_page_id=page.candidate_page_id, query=query, hits=ranked))
    return CandidateContexts(
        retrieval_backend=config.backend,
        model=DEFAULT_QWEN_EMBEDDING_MODEL,
        input_version=config.input_version,
        top_k=top_k,
        knowledge_pool_size=len(entries),
        candidate_page_count=len(candidate_pages.pages),
        candidate_pool_hash=stable_json_hash([entry.model_dump(mode="json") for entry in entries]),
        embedding_metrics={
            "backend": config.backend,
            "model": DEFAULT_QWEN_EMBEDDING_MODEL,
            "query_count": len(candidate_pages.pages),
            "page_count": len(entries),
            "top_k": top_k,
            "max_query_chars": config.max_query_chars,
        },
        items=items,
    )


def require_real_embedding_backend(config: EmbeddingConfig) -> None:
    if not config.enabled:
        raise RuntimeError("候选召回必须启用 embedding。")
    if config.backend != "sentence_transformers":
        raise RuntimeError("候选召回必须使用真实 embedding 后端：请配置 sentence_transformers / Qwen。")


def page_card(entry: WikiKnowledgeEntry, vault: Path, max_chars: int) -> str:
    path = vault / "wiki" / entry.path
    text = read_text(path) if path.exists() else entry.text_excerpt
    values = [
        f"title: {entry.title}",
        f"type: {entry.page_type}",
        f"summary: {entry.summary}",
        f"path: {entry.path}",
        strip_frontmatter(text),
    ]
    return trim_text("\n\n".join(value for value in values if value.strip()), max_chars)


def candidate_query(page: CandidatePage, max_chars: int) -> str:
    values = [
        f"title: {page.title}",
        f"type: {page.proposed_page_type}",
        f"summary: {page.summary}",
        f"path_hint: {page.proposed_path_hint}",
        page.body_markdown,
        "\n".join(page.evidence_notes),
    ]
    return trim_text("\n\n".join(value for value in values if value.strip()), max_chars)


def embed_texts(texts: list[str], config: EmbeddingConfig, *, is_query: bool) -> list[list[float]]:
    if config.backend != "sentence_transformers":
        raise RuntimeError(f"不支持的 embedding 后端：{config.backend}")
    model = _sentence_transformer_model(DEFAULT_QWEN_EMBEDDING_MODEL)
    kwargs: dict[str, object] = {
        "batch_size": max(1, config.batch_size),
        "normalize_embeddings": config.normalize_embeddings,
        "show_progress_bar": False,
    }
    if config.dimensions > 0:
        kwargs["truncate_dim"] = config.dimensions
    if is_query and config.query_prompt_name:
        kwargs["prompt_name"] = config.query_prompt_name
    try:
        encoded = model.encode(texts, **kwargs)
    except TypeError:
        kwargs.pop("truncate_dim", None)
        encoded = model.encode(texts, **kwargs)
    return [_coerce_vector(row, config) for row in encoded]


@lru_cache(maxsize=4)
def _sentence_transformer_model(model_name: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence_transformers 后端需要安装 sentence-transformers 包。") from exc
    return SentenceTransformer(model_name)


def _coerce_vector(row: Any, config: EmbeddingConfig) -> list[float]:
    values = row.tolist() if hasattr(row, "tolist") else list(row)
    vector = [float(value) for value in values]
    if config.dimensions > 0 and len(vector) > config.dimensions:
        vector = vector[: config.dimensions]
    if config.normalize_embeddings:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
    return [round(value, 8) for value in vector]


def cosine(left: list[float], right: list[Any]) -> float:
    if not left or not right:
        return 0.0
    total = 0.0
    for a, b in zip(left, right, strict=False):
        try:
            total += float(a) * float(b)
        except (TypeError, ValueError):
            continue
    return max(0.0, min(1.0, total))


def trim_text(text: str, max_chars: int) -> str:
    compact = re.sub(r"\n{3,}", "\n\n", text.strip())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rstrip()


def excerpt_text(text: str, max_chars: int) -> tuple[str, bool]:
    compact = trim_text(strip_frontmatter(text), max_chars)
    return compact, len(strip_frontmatter(text).strip()) > len(compact)


def _hit_reason(score: float) -> str:
    return f"候选页向量与该 wiki 页面接近；分数={score:.4f}"


def _pages_cache_dir(vault: Path, config: EmbeddingConfig) -> Path:
    root = Path(config.cache_dir).expanduser()
    if not root.is_absolute():
        root = vault / root
    namespace = safe_filename(f"{config.backend}_{DEFAULT_QWEN_EMBEDDING_MODEL}_{config.input_version}_{config.dimensions}")
    return root.resolve() / namespace / "pages"


def _record_path(pages_dir: Path, page_path: str) -> Path:
    key = sha256_text(page_path)[:16]
    name = safe_filename(page_path.replace("/", "__"))
    return pages_dir / f"{key}_{name}.json"


def _read_valid_record(cache_path: Path, entry: WikiKnowledgeEntry, config: EmbeddingConfig, input_sha: str) -> dict[str, Any] | None:
    if not cache_path.exists():
        return None
    try:
        payload = read_json(cache_path)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    checks = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "path": entry.path,
        "content_sha256": entry.sha256,
        "input_sha256": input_sha,
        **embedding_contract(config),
    }
    for key, value in checks.items():
        if payload.get(key) != value:
            return None
    vector = payload.get("vector")
    if not isinstance(vector, list) or len(vector) != config.dimensions:
        return None
    return payload


def embedding_contract(config: EmbeddingConfig) -> dict[str, int | str | bool]:
    return {
        "backend": config.backend,
        "model": DEFAULT_QWEN_EMBEDDING_MODEL,
        "dimensions": config.dimensions,
        "input_version": config.input_version,
        "max_page_chars": config.max_page_chars,
        "max_query_chars": config.max_query_chars,
        "normalize_embeddings": config.normalize_embeddings,
        "query_prompt_name": config.query_prompt_name,
    }
