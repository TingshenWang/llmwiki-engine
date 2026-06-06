import httpx
import pytest
from pathlib import Path

from llmwiki_engine.hash_utils import sha256_file
from llmwiki_engine.pipeline import init_vault, scan_raw_ingest_candidates
from llmwiki_engine.raw_import import RawUrlImportError, import_arxiv_search, import_raw_url, normalize_arxiv_html_url


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


def test_import_arxiv_search_imports_top_result_via_html(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    feed = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>https://arxiv.org/abs/2601.00001</id>
        <title>A Plan Reuse Mechanism for LLM-Driven Agent</title>
        <summary>This paper discusses agents and planning.</summary>
      </entry>
      <entry>
        <id>https://arxiv.org/abs/2507.21504</id>
        <updated>2025-07-30T00:00:00Z</updated>
        <published>2025-07-29T00:00:00Z</published>
        <title> Evaluation and Benchmarking of LLM Agents: A Survey </title>
        <summary>Agent evaluation survey for LLM agents.</summary>
        <author><name>Mahmoud Mohammadi</name></author>
        <link title="pdf" href="https://arxiv.org/pdf/2507.21504" type="application/pdf"/>
      </entry>
    </feed>
    """
    html = """
    <html><head><title>Ignored HTML Title</title></head>
      <body><article><h1>Paper Body</h1><p>Agent evaluation content.</p></article></body>
    </html>
    """
    seen_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        if request.url.host == "export.arxiv.org":
            assert "all%3ALLM" in str(request.url) or "all:LLM" in str(request.url)
            assert request.url.params["sortBy"] == "relevance"
            assert request.url.params["sortOrder"] == "descending"
            assert request.url.params["max_results"] == "20"
            return httpx.Response(200, headers={"content-type": "application/atom+xml"}, text=feed, request=request)
        if request.url.host == "arxiv.org" and request.url.path == "/html/2507.21504":
            return httpx.Response(200, headers={"content-type": "text/html"}, text=html, request=request)
        raise AssertionError(f"unexpected request: {request.url}")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = import_arxiv_search(vault, "LLM agents", limit=1, client=client)

    assert report.fetched_count == 1
    assert report.imported_count == 1
    assert report.failed_count == 0
    assert report.search_query == "all:LLM AND all:agents"
    assert report.sort_by == "relevance"
    assert report.sort_order == "descending"
    assert report.candidate_window == 20
    item = report.items[0]
    assert item.status == "imported"
    assert item.relevance_score > 0
    assert item.arxiv_id == "2507.21504"
    assert item.html_url == "https://arxiv.org/html/2507.21504"
    imported = vault / item.raw_path
    text = imported.read_text(encoding="utf-8")
    assert "# Evaluation and Benchmarking of LLM Agents: A Survey" in text
    assert "Imported from: https://arxiv.org/abs/2507.21504" in text
    assert "Fetched URL: https://arxiv.org/html/2507.21504" in text
    assert "# Paper Body" in text
    assert seen_paths == ["/api/query", "/html/2507.21504"]


def test_import_arxiv_search_dry_run_does_not_fetch_html(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    feed = """<feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>https://arxiv.org/abs/2507.21504</id>
        <title>Evaluation Survey</title>
      </entry>
    </feed>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "export.arxiv.org"
        assert request.url.params["max_results"] == "1"
        return httpx.Response(200, headers={"content-type": "application/atom+xml"}, text=feed, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = import_arxiv_search(vault, "cat:cs.AI", limit=1, dry_run=True, client=client)

    assert report.dry_run is True
    assert report.imported_count == 0
    assert report.candidate_window == 1
    assert report.items[0].status == "found"
    assert list((vault / "raw").glob("*.md")) == []


def test_import_arxiv_search_skips_low_relevance_natural_language_result(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault, profile_name="project_basic")
    feed = """<feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>https://arxiv.org/abs/2601.00001</id>
        <title>Compiler Scheduling in Embedded Systems</title>
        <summary>Scheduling heuristics for embedded compilers.</summary>
      </entry>
    </feed>
    """
    seen_hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_hosts.append(request.url.host or "")
        assert request.url.host == "export.arxiv.org"
        return httpx.Response(200, headers={"content-type": "application/atom+xml"}, text=feed, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = import_arxiv_search(vault, "LLM agent evaluation", limit=1, client=client)

    assert report.imported_count == 0
    assert report.skipped_count == 1
    item = report.items[0]
    assert item.status == "low_relevance"
    assert item.relevance_score == 0
    assert "below min_relevance_score" in item.error
    assert seen_hosts == ["export.arxiv.org"]
    assert list((vault / "raw").glob("*.md")) == []
