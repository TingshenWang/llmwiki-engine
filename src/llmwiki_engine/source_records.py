from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .hash_utils import sha256_file
from .models import RawIngestCandidate, RawIngestCandidateReport


RAW_INGEST_TEXT_SUFFIXES = {".md", ".markdown", ".mdown", ".txt"}


@dataclass(frozen=True)
class _SourceRawCoverageRecord:
    source_page: str
    raw_paths: tuple[str, ...]
    raw_hashes: tuple[str, ...]
    operation_ids: tuple[str, ...]


def parse_frontmatter(text: str) -> dict[str, Any] | None:
    if not text.startswith("---\n"):
        return None
    parts = text.split("---\n", 2)
    if len(parts) < 3:
        return None
    data = yaml.safe_load(parts[1]) or {}
    return data if isinstance(data, dict) else None


def scan_source_pages(vault: Path) -> list[tuple[str, dict[str, Any]]]:
    source_root = vault / "wiki" / "sources"
    if not source_root.exists():
        return []
    found: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(source_root.rglob("*.md")):
        frontmatter = parse_frontmatter(path.read_text(encoding="utf-8"))
        if frontmatter is not None:
            found.append((path.relative_to(vault).as_posix(), frontmatter))
    return found


def scan_raw_ingest_candidates(
    vault: Path,
    *,
    include_processed: bool = False,
    limit: int | None = None,
) -> RawIngestCandidateReport:
    if limit is not None and limit < 0:
        raise ValueError("limit must be >= 0")
    vault = vault.expanduser().resolve()
    raw_root = vault / "raw"
    if not raw_root.exists():
        raise ValueError(f"Raw directory not found: {raw_root}")
    records_by_path, records_by_hash = _source_raw_coverage_index(vault)
    raw_files = _iter_raw_ingest_files(raw_root)
    duplicate_url_first_paths = _duplicate_raw_url_first_paths(raw_files)
    items: list[RawIngestCandidate] = []
    for raw_path in raw_files:
        rel_path = normalize_vault_path(raw_path.relative_to(vault).as_posix())
        raw_hash = sha256_file(raw_path)
        path_records = records_by_path.get(rel_path, [])
        hash_records = records_by_hash.get(raw_hash, [])
        duplicate_url_first_path = duplicate_url_first_paths.get(raw_path)
        matched_records: list[_SourceRawCoverageRecord] = []
        if any(raw_hash in record.raw_hashes for record in path_records):
            status = "processed"
            matched_by = "path_and_hash"
            matched_records = [record for record in path_records if raw_hash in record.raw_hashes]
            reason = "same raw path and content hash are already recorded in source frontmatter"
        elif path_records and not any(record.raw_hashes for record in path_records):
            status = "processed"
            matched_by = "path"
            matched_records = path_records
            reason = "same raw path is recorded in source frontmatter; content hash is unavailable"
        elif path_records:
            status = "changed"
            matched_by = "path"
            matched_records = path_records
            reason = "same raw path is recorded, but the current content hash is different"
        elif hash_records:
            status = "duplicate_hash"
            matched_by = "hash"
            matched_records = hash_records
            reason = "same content hash is already recorded under another raw path"
        elif duplicate_url_first_path is not None:
            status = "duplicate_url"
            matched_by = "url"
            reason = (
                "same imported URL is already present under another raw path: "
                f"{normalize_vault_path(duplicate_url_first_path.relative_to(vault).as_posix())}"
            )
        else:
            status = "unprocessed"
            matched_by = "none"
            reason = "no matching source raw path or content hash found"
        stat = raw_path.stat()
        items.append(
            RawIngestCandidate(
                raw_path=rel_path,
                status=status,
                raw_sha256=raw_hash,
                size_bytes=stat.st_size,
                mtime=datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                matched_by=matched_by,
                source_pages=sorted({record.source_page for record in matched_records}),
                operation_ids=sorted({operation_id for record in matched_records for operation_id in record.operation_ids}),
                reason=reason,
            )
        )
    status_rank = {"unprocessed": 0, "changed": 1, "duplicate_hash": 2, "duplicate_url": 3, "processed": 4}
    items.sort(key=lambda item: (status_rank[item.status], item.raw_path))
    visible_items = items if include_processed else [item for item in items if item.status != "processed"]
    status_counts = {status: sum(1 for item in items if item.status == status) for status in status_rank}
    if limit is not None:
        visible_items = visible_items[:limit]
    return RawIngestCandidateReport(
        vault=vault.as_posix(),
        raw_root=raw_root.as_posix(),
        include_processed=include_processed,
        limit=limit,
        total_raw_files=len(items),
        candidate_count=len(visible_items),
        processed_count=status_counts["processed"],
        changed_count=status_counts["changed"],
        duplicate_hash_count=status_counts["duplicate_hash"],
        duplicate_url_count=status_counts["duplicate_url"],
        unprocessed_count=status_counts["unprocessed"],
        items=visible_items,
    )


def frontmatter_list(frontmatter: dict[str, Any], key: str) -> list[str]:
    value = frontmatter.get(key, [])
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def normalize_vault_path(path: str) -> str:
    return unicodedata.normalize("NFC", path.strip()).replace("\\", "/")


def _duplicate_raw_url_first_paths(raw_files: list[Path]) -> dict[Path, Path]:
    first_by_url: dict[str, Path] = {}
    duplicates: dict[Path, Path] = {}
    for raw_path in raw_files:
        matched_first: Path | None = None
        for url in _raw_import_urls(raw_path):
            first = first_by_url.get(url)
            if first is not None and first != raw_path:
                matched_first = first
                break
        if matched_first is not None:
            duplicates[raw_path] = matched_first
            continue
        for url in _raw_import_urls(raw_path):
            first_by_url.setdefault(url, raw_path)
    return duplicates


def _raw_import_urls(raw_path: Path) -> list[str]:
    urls: list[str] = []
    try:
        with raw_path.open("r", encoding="utf-8", errors="ignore") as handle:
            prefix = handle.read(8192)
    except OSError:
        return urls
    for line in prefix.splitlines()[:24]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip().lower() not in {"imported from", "fetched url", "final url"}:
            continue
        url = value.strip()
        if url and url not in urls:
            urls.append(url)
    return urls


def _iter_raw_ingest_files(raw_root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in raw_root.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(raw_root).parts
        if any(part.startswith(".") for part in relative_parts):
            continue
        if relative_parts and relative_parts[0] == "log":
            continue
        if path.suffix.lower() not in RAW_INGEST_TEXT_SUFFIXES:
            continue
        paths.append(path)
    return sorted(paths)


def _source_raw_coverage_index(vault: Path) -> tuple[dict[str, list[_SourceRawCoverageRecord]], dict[str, list[_SourceRawCoverageRecord]]]:
    records_by_path: dict[str, list[_SourceRawCoverageRecord]] = {}
    records_by_hash: dict[str, list[_SourceRawCoverageRecord]] = {}
    for source_page, frontmatter in scan_source_pages(vault):
        raw_paths = tuple(_frontmatter_raw_paths(frontmatter))
        raw_hashes = tuple(value.strip() for value in frontmatter_list(frontmatter, "source_raw_hashes") if value.strip())
        operation_ids = tuple(value.strip() for value in frontmatter_list(frontmatter, "source_operation_ids") if value.strip())
        if not raw_paths and not raw_hashes:
            continue
        record = _SourceRawCoverageRecord(
            source_page=source_page,
            raw_paths=raw_paths,
            raw_hashes=raw_hashes,
            operation_ids=operation_ids,
        )
        for raw_path in raw_paths:
            records_by_path.setdefault(raw_path, []).append(record)
        for raw_hash in raw_hashes:
            records_by_hash.setdefault(raw_hash, []).append(record)
    return records_by_path, records_by_hash


def _frontmatter_raw_paths(frontmatter: dict[str, Any]) -> list[str]:
    raw_paths: list[str] = []
    for value in frontmatter_list(frontmatter, "source_raw_paths"):
        normalized = normalize_vault_path(value)
        if normalized:
            raw_paths.append(normalized)
    return raw_paths
