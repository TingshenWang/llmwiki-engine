from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, replace
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

import httpx

from .hash_utils import sha256_file
from .profiles import safe_filename
from .models import utc_now


RAW_IMPORT_TEXT_SUFFIXES = {".md", ".markdown", ".mdown", ".txt"}
ARXIV_ID_RE = re.compile(r"^(?P<id>(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?)$", re.IGNORECASE)
ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_QUERY_FIELD_RE = re.compile(r"\b(?:all|ti|au|abs|co|jr|cat|rn|id):", re.IGNORECASE)
ARXIV_SEARCH_MAX_WINDOW = 50
ARXIV_SEARCH_MIN_RERANK_WINDOW = 20
ARXIV_SEARCH_RERANK_MULTIPLIER = 8
QUERY_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


class RawUrlImportError(RuntimeError):
    pass


@dataclass(frozen=True)
class RawUrlImportResult:
    vault: str
    url: str
    fetch_url: str
    final_url: str
    title: str
    raw_path: str
    absolute_path: str
    content_type: str
    format: str
    imported_at: str
    sha256: str
    size_bytes: int
    overwritten: bool = False
    status: str = "imported"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ArxivSearchEntry:
    arxiv_id: str
    title: str
    abs_url: str
    html_url: str
    pdf_url: str
    published: str = ""
    updated: str = ""
    authors: tuple[str, ...] = ()
    summary: str = ""
    relevance_score: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ArxivRawImportItem:
    arxiv_id: str
    title: str
    abs_url: str
    html_url: str
    status: str
    raw_path: str = ""
    sha256: str = ""
    size_bytes: int = 0
    relevance_score: int = 0
    error: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ArxivRawImportReport:
    vault: str
    query: str
    search_query: str
    sort_by: str
    sort_order: str
    candidate_window: int
    min_relevance_score: int
    skipped_count: int
    limit: int
    dry_run: bool
    fetched_count: int
    imported_count: int
    existing_count: int
    failed_count: int
    items: tuple[ArxivRawImportItem, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def import_raw_url(
    vault: Path,
    url: str,
    *,
    title: str | None = None,
    output_name: str | None = None,
    overwrite: bool = False,
    dedupe_url: bool = True,
    prefer_arxiv_html: bool = True,
    timeout: float = 30.0,
    max_bytes: int = 5_000_000,
    client: httpx.Client | None = None,
) -> RawUrlImportResult:
    """Fetch a public URL and save it as Markdown under vault/raw."""
    if max_bytes < 1024:
        raise ValueError("max_bytes must be >= 1024")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("url must be an absolute http(s) URL")
    fetch_url = normalize_arxiv_html_url(url) if prefer_arxiv_html else url

    vault = vault.expanduser().resolve()
    raw_root = vault / "raw"
    if not raw_root.exists():
        raise ValueError(f"Raw directory not found: {raw_root}")
    if dedupe_url and not overwrite:
        existing = _find_existing_url_import(raw_root, [url, fetch_url])
        if existing is not None:
            return _existing_url_result(vault, url, fetch_url, existing)

    response = _fetch_url(fetch_url, timeout=timeout, max_bytes=max_bytes, client=client)
    if dedupe_url and not overwrite:
        existing = _find_existing_url_import(raw_root, [url, fetch_url, str(response.url)])
        if existing is not None:
            return _existing_url_result(vault, url, fetch_url, existing, final_url=str(response.url))

    content_type = _content_type(response)
    fetched_text = response.text
    inferred_title = (title or "").strip()
    source_format = _source_format(url, content_type, fetched_text)
    if source_format == "html":
        html_title = _extract_html_title(fetched_text)
        inferred_title = inferred_title or html_title or _title_from_url(url)
        body = _html_to_markdown(fetched_text)
    elif source_format == "text":
        inferred_title = inferred_title or _title_from_markdown(fetched_text) or _title_from_url(url)
        body = _normalize_text(fetched_text)
    else:
        raise RawUrlImportError(
            f"Unsupported content type {content_type!r}; only HTML, Markdown, and plain text URLs are supported."
        )

    imported_at = utc_now()
    target = _raw_output_path(raw_root, output_name=output_name, title=inferred_title, url=url, overwrite=overwrite)
    overwritten = target.exists() and overwrite
    markdown = _render_imported_markdown(
        title=inferred_title,
        source_url=url,
        fetch_url=fetch_url,
        final_url=str(response.url),
        imported_at=imported_at,
        content_type=content_type,
        body=body,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(markdown, encoding="utf-8")
    stat = target.stat()
    return RawUrlImportResult(
        vault=vault.as_posix(),
        url=url,
        fetch_url=fetch_url,
        final_url=str(response.url),
        title=inferred_title,
        raw_path=target.relative_to(vault).as_posix(),
        absolute_path=target.as_posix(),
        content_type=content_type,
        format=source_format,
        imported_at=imported_at,
        sha256=sha256_file(target),
        size_bytes=stat.st_size,
        overwritten=overwritten,
        status="overwritten" if overwritten else "imported",
    )


def import_arxiv_search(
    vault: Path,
    query: str,
    *,
    limit: int = 1,
    dry_run: bool = False,
    overwrite: bool = False,
    dedupe_url: bool = True,
    sort_by: str = "relevance",
    sort_order: str = "descending",
    min_relevance_score: int = 1,
    timeout: float = 30.0,
    max_bytes: int = 5_000_000,
    client: httpx.Client | None = None,
) -> ArxivRawImportReport:
    if limit < 1:
        raise ValueError("limit must be >= 1")
    if min_relevance_score < 0:
        raise ValueError("min_relevance_score must be >= 0")
    vault = vault.expanduser().resolve()
    entries, search_query, candidate_window = search_arxiv(
        query,
        limit=limit,
        sort_by=sort_by,
        sort_order=sort_order,
        timeout=timeout,
        client=client,
    )
    fielded_query = _is_arxiv_fielded_query(query)
    items: list[ArxivRawImportItem] = []
    for entry in entries:
        low_relevance = (not fielded_query) and entry.relevance_score < min_relevance_score
        if dry_run:
            items.append(
                ArxivRawImportItem(
                    arxiv_id=entry.arxiv_id,
                    title=entry.title,
                    abs_url=entry.abs_url,
                    html_url=entry.html_url,
                    status="low_relevance" if low_relevance else "found",
                    relevance_score=entry.relevance_score,
                    error=(
                        f"relevance_score {entry.relevance_score} below min_relevance_score {min_relevance_score}"
                        if low_relevance
                        else ""
                    ),
                )
            )
            continue
        if low_relevance:
            items.append(
                ArxivRawImportItem(
                    arxiv_id=entry.arxiv_id,
                    title=entry.title,
                    abs_url=entry.abs_url,
                    html_url=entry.html_url,
                    status="low_relevance",
                    relevance_score=entry.relevance_score,
                    error=f"relevance_score {entry.relevance_score} below min_relevance_score {min_relevance_score}",
                )
            )
            continue
        try:
            result = import_raw_url(
                vault,
                entry.abs_url,
                title=entry.title,
                overwrite=overwrite,
                dedupe_url=dedupe_url,
                prefer_arxiv_html=True,
                timeout=timeout,
                max_bytes=max_bytes,
                client=client,
            )
            items.append(
                ArxivRawImportItem(
                    arxiv_id=entry.arxiv_id,
                    title=entry.title,
                    abs_url=entry.abs_url,
                    html_url=entry.html_url,
                    status=result.status,
                    raw_path=result.raw_path,
                    sha256=result.sha256,
                    size_bytes=result.size_bytes,
                    relevance_score=entry.relevance_score,
                )
            )
        except (RawUrlImportError, ValueError) as exc:
            items.append(
                ArxivRawImportItem(
                    arxiv_id=entry.arxiv_id,
                    title=entry.title,
                    abs_url=entry.abs_url,
                    html_url=entry.html_url,
                    status="failed",
                    relevance_score=entry.relevance_score,
                    error=str(exc),
                )
            )
    return ArxivRawImportReport(
        vault=vault.as_posix(),
        query=query,
        search_query=search_query,
        sort_by=sort_by,
        sort_order=sort_order,
        candidate_window=candidate_window,
        min_relevance_score=min_relevance_score,
        skipped_count=sum(1 for item in items if item.status == "low_relevance"),
        limit=limit,
        dry_run=dry_run,
        fetched_count=len(entries),
        imported_count=sum(1 for item in items if item.status in {"imported", "overwritten"}),
        existing_count=sum(1 for item in items if item.status == "existing_url"),
        failed_count=sum(1 for item in items if item.status == "failed"),
        items=tuple(items),
    )


def search_arxiv(
    query: str,
    *,
    limit: int = 5,
    sort_by: str = "relevance",
    sort_order: str = "descending",
    timeout: float = 30.0,
    client: httpx.Client | None = None,
) -> tuple[list[ArxivSearchEntry], str, int]:
    if limit < 1:
        raise ValueError("limit must be >= 1")
    if sort_by not in {"relevance", "lastUpdatedDate", "submittedDate"}:
        raise ValueError("sort_by must be one of: relevance, lastUpdatedDate, submittedDate")
    if sort_order not in {"ascending", "descending"}:
        raise ValueError("sort_order must be one of: ascending, descending")
    search_query = _arxiv_search_query(query)
    fielded_query = _is_arxiv_fielded_query(query)
    candidate_window = limit if fielded_query else min(
        ARXIV_SEARCH_MAX_WINDOW,
        max(limit * ARXIV_SEARCH_RERANK_MULTIPLIER, ARXIV_SEARCH_MIN_RERANK_WINDOW),
    )
    headers = {"User-Agent": "llmwiki-engine/0.1 raw-import-arxiv"}
    params = {
        "search_query": search_query,
        "start": "0",
        "max_results": str(candidate_window),
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }
    try:
        if client is not None:
            response = client.get(ARXIV_API_URL, params=params, headers=headers, timeout=timeout)
        else:
            with httpx.Client(headers=headers, timeout=timeout) as owned_client:
                response = owned_client.get(ARXIV_API_URL, params=params)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RawUrlImportError(f"arXiv search failed with HTTP {exc.response.status_code}") from exc
    except httpx.HTTPError as exc:
        raise RawUrlImportError(f"arXiv search failed: {exc}") from exc
    entries = _parse_arxiv_feed(response.text)
    if not fielded_query:
        entries = _rerank_arxiv_entries(entries, query)
    return entries[:limit], search_query, candidate_window


def _arxiv_search_query(query: str) -> str:
    query = " ".join(query.split())
    if not query:
        raise ValueError("query must not be empty")
    if _is_arxiv_fielded_query(query):
        return query
    terms = re.findall(r"[A-Za-z0-9_.-]+", query)
    if not terms:
        raise ValueError("query must contain searchable terms")
    return " AND ".join(f"all:{term}" for term in terms[:8])


def _is_arxiv_fielded_query(query: str) -> bool:
    return bool(ARXIV_QUERY_FIELD_RE.search(query))


def _query_terms(query: str) -> list[str]:
    terms: list[str] = []
    for term in re.findall(r"[A-Za-z0-9_.-]+", query.lower()):
        if len(term) < 2 or term in QUERY_STOP_WORDS:
            continue
        if term not in terms:
            terms.append(term)
    return terms[:8]


def _rerank_arxiv_entries(entries: list[ArxivSearchEntry], query: str) -> list[ArxivSearchEntry]:
    terms = _query_terms(query)
    if not terms:
        return entries
    indexed = [(index, replace(entry, relevance_score=_arxiv_entry_query_score(entry, terms))) for index, entry in enumerate(entries)]
    indexed.sort(key=lambda pair: (-pair[1].relevance_score, pair[0]))
    return [entry for _, entry in indexed]


def _arxiv_entry_query_score(entry: ArxivSearchEntry, terms: list[str]) -> int:
    title = entry.title.lower()
    summary = entry.summary.lower()
    score = 0
    for term in terms:
        variants = {term}
        if not term.endswith("s"):
            variants.add(f"{term}s")
        if any(_contains_term(title, variant) for variant in variants):
            score += 5
        if any(_contains_term(summary, variant) for variant in variants):
            score += 2
    if all(any(_contains_term(title + " " + summary, variant) for variant in ({term, f"{term}s"} if not term.endswith("s") else {term})) for term in terms):
        score += 6
    return score


def _contains_term(text: str, term: str) -> bool:
    return bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", text))


def _parse_arxiv_feed(text: str) -> list[ArxivSearchEntry]:
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise RawUrlImportError(f"arXiv search returned invalid Atom XML: {exc}") from exc
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries: list[ArxivSearchEntry] = []
    for entry in root.findall("atom:entry", ns):
        abs_url = _xml_text(entry, "atom:id", ns)
        title = " ".join(_xml_text(entry, "atom:title", ns).split())
        summary = " ".join(_xml_text(entry, "atom:summary", ns).split())
        arxiv_id = _arxiv_id_from_abs_url(abs_url)
        pdf_url = ""
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href", "")
                break
        authors = tuple(
            " ".join(_xml_text(author, "atom:name", ns).split())
            for author in entry.findall("atom:author", ns)
            if _xml_text(author, "atom:name", ns).strip()
        )
        if not arxiv_id or not abs_url or not title:
            continue
        entries.append(
            ArxivSearchEntry(
                arxiv_id=arxiv_id,
                title=title,
                abs_url=abs_url,
                html_url=normalize_arxiv_html_url(abs_url),
                pdf_url=pdf_url,
                published=_xml_text(entry, "atom:published", ns),
                updated=_xml_text(entry, "atom:updated", ns),
                authors=authors,
                summary=summary,
            )
        )
    return entries


def _xml_text(element: ET.Element, path: str, namespaces: dict[str, str]) -> str:
    found = element.find(path, namespaces)
    return found.text.strip() if found is not None and found.text else ""


def _arxiv_id_from_abs_url(abs_url: str) -> str:
    parsed = urlparse(abs_url)
    if (parsed.hostname or "").lower() not in {"arxiv.org", "www.arxiv.org"}:
        return ""
    path = parsed.path.strip("/")
    if not path.startswith("abs/"):
        return ""
    arxiv_id = path.removeprefix("abs/")
    return arxiv_id if ARXIV_ID_RE.match(arxiv_id) else ""


def normalize_arxiv_html_url(url: str) -> str:
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    if hostname not in {"arxiv.org", "www.arxiv.org"}:
        return url
    path = parsed.path.strip("/")
    if not path:
        return url
    parts = path.split("/")
    if len(parts) < 2 or parts[0] not in {"abs", "pdf", "html"}:
        return url
    arxiv_id = "/".join(parts[1:])
    if parts[0] == "pdf" and arxiv_id.endswith(".pdf"):
        arxiv_id = arxiv_id[:-4]
    if not ARXIV_ID_RE.match(arxiv_id):
        return url
    return f"https://arxiv.org/html/{arxiv_id}"


def _find_existing_url_import(raw_root: Path, urls: list[str]) -> Path | None:
    normalized_urls = {url.strip() for url in urls if url.strip()}
    for path in sorted(raw_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in RAW_IMPORT_TEXT_SUFFIXES:
            continue
        relative_parts = path.relative_to(raw_root).parts
        if any(part.startswith(".") for part in relative_parts) or (relative_parts and relative_parts[0] == "log"):
            continue
        metadata = _raw_import_metadata(path)
        imported_urls = {metadata.get("imported from", ""), metadata.get("fetched url", ""), metadata.get("final url", "")}
        if normalized_urls.intersection(imported_urls):
            return path
    return None


def _existing_url_result(
    vault: Path,
    url: str,
    fetch_url: str,
    path: Path,
    *,
    final_url: str | None = None,
) -> RawUrlImportResult:
    metadata = _raw_import_metadata(path)
    stat = path.stat()
    title = _title_from_markdown(_read_text_prefix(path)) or path.stem
    return RawUrlImportResult(
        vault=vault.as_posix(),
        url=url,
        fetch_url=fetch_url,
        final_url=final_url or metadata.get("final url") or url,
        title=title,
        raw_path=path.relative_to(vault).as_posix(),
        absolute_path=path.as_posix(),
        content_type=metadata.get("content type", "unknown"),
        format="existing",
        imported_at=metadata.get("imported at", ""),
        sha256=sha256_file(path),
        size_bytes=stat.st_size,
        overwritten=False,
        status="existing_url",
    )


def _raw_import_metadata(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in _read_text_prefix(path).splitlines()[:24]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized_key = key.strip().lower()
        if normalized_key in {"imported from", "fetched url", "final url", "imported at", "content type"}:
            metadata[normalized_key] = value.strip()
    return metadata


def _read_text_prefix(path: Path, limit: int = 8192) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return handle.read(limit)


def _fetch_url(url: str, *, timeout: float, max_bytes: int, client: httpx.Client | None) -> httpx.Response:
    headers = {
        "User-Agent": "llmwiki-engine/0.1 raw-import-url",
        "Accept": "text/html,text/markdown,text/plain;q=0.9,*/*;q=0.1",
    }
    try:
        if client is not None:
            with client.stream("GET", url, headers=headers, follow_redirects=True, timeout=timeout) as response:
                return _read_limited_response(url, response, max_bytes=max_bytes)
        else:
            with httpx.Client(headers=headers, follow_redirects=True, timeout=timeout) as owned_client:
                with owned_client.stream("GET", url) as response:
                    return _read_limited_response(url, response, max_bytes=max_bytes)
    except httpx.HTTPStatusError as exc:
        raise RawUrlImportError(f"Fetch failed with HTTP {exc.response.status_code} for {url}") from exc
    except httpx.HTTPError as exc:
        raise RawUrlImportError(f"Fetch failed for {url}: {exc}") from exc


def _read_limited_response(url: str, response: httpx.Response, *, max_bytes: int) -> httpx.Response:
    response.raise_for_status()
    is_encoded = bool(response.headers.get("content-encoding"))
    content_length = response.headers.get("content-length")
    if content_length and not is_encoded:
        try:
            declared_bytes = int(content_length)
        except ValueError:
            declared_bytes = 0
        if declared_bytes > max_bytes:
            raise RawUrlImportError(f"Fetched content is too large: {declared_bytes} bytes > {max_bytes} max_bytes")
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_bytes():
        total += len(chunk)
        if total > max_bytes:
            raise RawUrlImportError(f"Fetched content is too large: {total} bytes > {max_bytes} max_bytes")
        chunks.append(chunk)
    headers = httpx.Headers(response.headers)
    for transport_header in ("content-encoding", "content-length", "transfer-encoding"):
        if transport_header in headers:
            del headers[transport_header]
    return httpx.Response(
        response.status_code,
        headers=headers,
        content=b"".join(chunks),
        request=response.request,
        extensions=response.extensions,
    )


def _content_type(response: httpx.Response) -> str:
    value = response.headers.get("content-type", "")
    return value.split(";", 1)[0].strip().lower()


def _source_format(url: str, content_type: str, text: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if content_type in {"text/html", "application/xhtml+xml"} or suffix in {".html", ".htm"}:
        return "html"
    if content_type in {"text/markdown", "text/x-markdown", "text/plain"}:
        return "text"
    if suffix in RAW_IMPORT_TEXT_SUFFIXES:
        return "text"
    if not content_type and _looks_like_html(text):
        return "html"
    if content_type.startswith("text/") and not _looks_like_html(text):
        return "text"
    return "unsupported"


def _raw_output_path(raw_root: Path, *, output_name: str | None, title: str, url: str, overwrite: bool) -> Path:
    if output_name:
        relative = Path(output_name)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError("--output must be a relative path inside raw/")
        if not relative.suffix:
            relative = relative.with_suffix(".md")
        if relative.suffix.lower() not in RAW_IMPORT_TEXT_SUFFIXES:
            raise ValueError(f"--output suffix must be one of: {', '.join(sorted(RAW_IMPORT_TEXT_SUFFIXES))}")
        candidate = raw_root / relative
    else:
        stem = safe_filename(title or _title_from_url(url))
        candidate = raw_root / f"{stem}.md"
    candidate = candidate.resolve()
    raw_root = raw_root.resolve()
    try:
        candidate.relative_to(raw_root)
    except ValueError as exc:
        raise ValueError("--output must stay inside raw/") from exc
    if overwrite or not candidate.exists():
        return candidate
    return _next_available_path(candidate)


def _next_available_path(path: Path) -> Path:
    for index in range(2, 10_000):
        candidate = path.with_name(f"{path.stem}-{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RawUrlImportError(f"Could not find an available filename for {path.name}")


def _render_imported_markdown(
    *,
    title: str,
    source_url: str,
    fetch_url: str,
    final_url: str,
    imported_at: str,
    content_type: str,
    body: str,
) -> str:
    metadata = [
        f"# {title}",
        "",
        f"Imported from: {source_url}",
        f"Fetched URL: {fetch_url}",
        f"Final URL: {final_url}",
        f"Imported at: {imported_at}",
        f"Content type: {content_type or 'unknown'}",
        "",
        "---",
        "",
    ]
    return "\n".join(metadata) + body.strip() + "\n"


def _title_from_url(url: str) -> str:
    parsed = urlparse(url)
    tail = Path(parsed.path).stem or parsed.netloc
    title = unescape(tail.replace("-", " ").replace("_", " ")).strip()
    return title or parsed.netloc or "imported-url"


def _title_from_markdown(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip() or None
    return None


def _extract_html_title(text: str) -> str | None:
    parser = _TitleParser()
    parser.feed(text)
    return parser.title


def _html_to_markdown(text: str) -> str:
    for target_tag in ["article", "main"]:
        if re.search(rf"<{target_tag}\b", text, flags=re.IGNORECASE):
            parser = _HTMLMarkdownParser(target_tag=target_tag)
            parser.feed(text)
            markdown = _cleanup_html_markdown_noise(parser.markdown())
            if markdown:
                return markdown
    parser = _HTMLMarkdownParser()
    parser.feed(text)
    return _cleanup_html_markdown_noise(parser.markdown())


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _looks_like_html(text: str) -> bool:
    sample = text[:2048].lower()
    return "<html" in sample or "<body" in sample or "<article" in sample


def _cleanup_html_markdown_noise(text: str) -> str:
    cleaned: list[str] = []
    visible_count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            visible_count += 1
        if stripped == r"\setcctype":
            continue
        if visible_count <= 4 and stripped.lower() == "by":
            continue
        if stripped.startswith("† †"):
            continue
        if stripped in {"•", "- •"}:
            continue
        cleaned.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned)).strip()


class _TitleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._in_title = False
        self._parts: list[str] = []

    @property
    def title(self) -> str | None:
        title = " ".join(" ".join(self._parts).split()).strip()
        return title or None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._parts.append(data)


class _HTMLMarkdownParser(HTMLParser):
    SKIP_TAGS = {"head", "script", "style", "noscript", "svg", "canvas"}
    BLOCK_TAGS = {
        "article",
        "aside",
        "blockquote",
        "body",
        "div",
        "footer",
        "header",
        "main",
        "nav",
        "ol",
        "p",
        "section",
        "table",
        "tbody",
        "td",
        "th",
        "thead",
        "tr",
        "ul",
    }
    HEADING_TAGS = {"h1": "#", "h2": "##", "h3": "###", "h4": "####", "h5": "#####", "h6": "######"}

    def __init__(self, *, target_tag: str | None = None) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0
        self._pre_depth = 0
        self._target_tag = target_tag
        self._target_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if self._target_tag is not None:
            if tag == self._target_tag:
                self._target_depth += 1
            elif self._target_depth == 0:
                return
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "pre":
            self._new_block()
            self._append("```")
            self._newline()
            self._pre_depth += 1
            return
        if tag in self.HEADING_TAGS:
            self._new_block()
            self._append(f"{self.HEADING_TAGS[tag]} ")
            return
        if tag == "li":
            self._new_block()
            self._append("- ")
            return
        if tag == "br":
            self._newline()
            return
        if tag in self.BLOCK_TAGS:
            self._new_block()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._target_tag is not None and self._target_depth == 0:
            return
        if tag in self.SKIP_TAGS:
            if self._skip_depth:
                self._skip_depth -= 1
            if tag == self._target_tag and self._target_depth:
                self._target_depth -= 1
            return
        if self._skip_depth:
            return
        if tag == "pre":
            if self._pre_depth:
                self._pre_depth -= 1
            self._newline()
            self._append("```")
            self._new_block()
            return
        if tag in self.HEADING_TAGS or tag in self.BLOCK_TAGS or tag == "li":
            self._new_block()
        if tag == self._target_tag and self._target_depth:
            self._target_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._target_tag is not None and self._target_depth == 0:
            return
        if self._skip_depth:
            return
        if self._pre_depth:
            self._append(data.replace("\r\n", "\n").replace("\r", "\n"))
            return
        text = " ".join(data.split())
        if text:
            self._append(text)

    def markdown(self) -> str:
        text = "".join(self._parts)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _append(self, text: str) -> None:
        if not text:
            return
        if self._parts and not self._parts[-1].endswith((" ", "\n", "-", "`")) and not text.startswith((" ", "\n", ".", ",", ":", ";", "!", "?")):
            self._parts.append(" ")
        self._parts.append(text)

    def _newline(self) -> None:
        if not self._parts or self._parts[-1].endswith("\n"):
            return
        self._parts.append("\n")

    def _new_block(self) -> None:
        text = "".join(self._parts)
        if not text:
            return
        if text.endswith("\n\n"):
            return
        if text.endswith("\n"):
            self._parts.append("\n")
        else:
            self._parts.append("\n\n")
