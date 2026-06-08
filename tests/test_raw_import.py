import gzip
from pathlib import Path

import httpx
import pytest

from llmwiki_engine.hash_utils import sha256_file
from llmwiki_engine.pipeline import init_vault
from llmwiki_engine.raw_import import RawUrlImportError, import_raw_url, normalize_arxiv_html_url
from llmwiki_engine.source_records import scan_raw_ingest_candidates


def test_normalize_arxiv_html_url_prefers_html_for_abs_and_pdf_urls() -> None:
    assert normalize_arxiv_html_url("https://arxiv.org/abs/2507.21504") == "https://arxiv.org/html/2507.21504"
    assert normalize_arxiv_html_url("https://arxiv.org/pdf/2507.21504v2.pdf") == "https://arxiv.org/html/2507.21504v2"
    assert normalize_arxiv_html_url("https://arxiv.org/html/2507.21504") == "https://arxiv.org/html/2507.21504"
    assert normalize_arxiv_html_url("https://example.com/pdf/2507.21504.pdf") == "https://example.com/pdf/2507.21504.pdf"


def test_import_raw_url_converts_html_to_markdown_and_records_source(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")

    html = """
    <html>
      <head><title>Test Article</title><script>hidden()</script></head>
      <body>
        <nav>Navigation noise</nav>
        <article>
          <p>\\setcctype</p>
          <p>by</p>
          <p>† † journalyear: 2025 † † copyright: cc</p>
          <h1>Main Idea</h1>
          <p>Alpha <strong>beta</strong> insight.</p>
          <ul><li>First point</li><li>Second point</li></ul>
          <ul><li>•</li></ul>
        </article>
        <footer>Footer noise</footer>
      </body>
    </html>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://arxiv.org/html/2507.21504"
        return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, text=html, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = import_raw_url(vault, "https://arxiv.org/abs/2507.21504", client=client)

    imported = Path(result.absolute_path)
    text = imported.read_text(encoding="utf-8")
    assert result.raw_path == "raw/Test Article.md"
    assert result.title == "Test Article"
    assert result.url == "https://arxiv.org/abs/2507.21504"
    assert result.fetch_url == "https://arxiv.org/html/2507.21504"
    assert result.format == "html"
    assert result.sha256 == sha256_file(imported)
    assert "# Test Article" in text
    assert "Imported from: https://arxiv.org/abs/2507.21504" in text
    assert "Fetched URL: https://arxiv.org/html/2507.21504" in text
    assert "# Main Idea" in text
    assert "Alpha beta insight." in text
    assert "- First point" in text
    assert "hidden()" not in text
    assert "Navigation noise" not in text
    assert "Footer noise" not in text
    assert "\\setcctype" not in text
    assert "journalyear" not in text
    assert "- •" not in text

    report = scan_raw_ingest_candidates(vault)
    assert [item.raw_path for item in report.items] == [result.raw_path]
    assert report.items[0].status == "unprocessed"


def test_import_raw_url_avoids_overwrite_with_suffix(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    bodies = ["# Same\n\nfirst body\n", "# Same\n\nsecond body\n"]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/markdown"},
            text=bodies.pop(0),
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        first = import_raw_url(vault, "https://example.com/same.md", dedupe_url=False, client=client)
        second = import_raw_url(vault, "https://example.com/same.md", dedupe_url=False, client=client)

    assert first.raw_path == "raw/Same.md"
    assert second.raw_path == "raw/Same-2.md"
    assert "first body" in Path(first.absolute_path).read_text(encoding="utf-8")
    assert "second body" in Path(second.absolute_path).read_text(encoding="utf-8")
    assert not first.overwritten
    assert not second.overwritten


def test_import_raw_url_reuses_existing_imported_url_without_fetch(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")

    def first_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/markdown"},
            text="# Stable Source\n\nfirst body\n",
            request=request,
        )

    def failing_handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("duplicate URL import should not fetch again")

    with httpx.Client(transport=httpx.MockTransport(first_handler)) as client:
        first = import_raw_url(vault, "https://example.com/stable.md", client=client)
    with httpx.Client(transport=httpx.MockTransport(failing_handler)) as client:
        second = import_raw_url(vault, "https://example.com/stable.md", client=client)

    assert second.status == "existing_url"
    assert second.raw_path == first.raw_path
    assert second.sha256 == first.sha256
    assert second.format == "existing"
    assert sorted((vault / "raw").glob("Stable Source*.md")) == [Path(first.absolute_path)]


def test_import_raw_url_rejects_unsupported_binary_content(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF-1.7", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RawUrlImportError, match="Unsupported content type"):
            import_raw_url(vault, "https://example.com/paper.pdf", client=client)


def test_import_raw_url_aborts_stream_when_max_bytes_exceeded(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")

    class ChunkStream(httpx.SyncByteStream):
        def __init__(self, chunks: list[bytes]):
            self.chunks = chunks
            self.yielded = 0

        def __iter__(self):
            for chunk in self.chunks:
                self.yielded += 1
                yield chunk

    stream = ChunkStream([b"a" * 1024, b"b" * 1024, b"c" * 1024, b"d" * 1024])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            stream=stream,
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RawUrlImportError, match="Fetched content is too large"):
            import_raw_url(vault, "https://example.com/large.txt", client=client, max_bytes=2500)

    assert stream.yielded == 3
    assert list((vault / "raw").glob("*.md")) == []


def test_import_raw_url_handles_gzip_stream_without_double_decompression(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    html = b"""
    <html>
      <head><title>Compressed Article</title></head>
      <body><article><h1>Compressed Body</h1><p>Decoded text survives.</p></article></body>
    </html>
    """
    compressed = gzip.compress(html)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-type": "text/html; charset=utf-8",
                "content-encoding": "gzip",
                "content-length": str(len(compressed)),
            },
            content=compressed,
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = import_raw_url(vault, "https://example.com/compressed", client=client)

    text = Path(result.absolute_path).read_text(encoding="utf-8")
    assert result.content_type == "text/html"
    assert result.format == "html"
    assert result.raw_path == "raw/Compressed Article.md"
    assert "# Compressed Body" in text
    assert "Decoded text survives." in text
