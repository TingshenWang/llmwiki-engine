from __future__ import annotations

import re
from dataclasses import asdict, dataclass
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
