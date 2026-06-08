import json
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel

from llmwiki_engine.models import RawPreparationArtifact, SourceDigestArtifact
from llmwiki_engine.providers import OpenAICompatibleProvider, ProviderError, ProviderRegistry
from llmwiki_engine.redaction import Redactor
from llmwiki_engine.structured import StructuredModelCall, StructuredOutputError


FIXTURE = Path(__file__).parent / "fixtures" / "simple_project" / "mock"


class JsonLikeArtifact(BaseModel):
    value_points: str
    quote: str


def test_mock_provider_returns_source_digest_fixture() -> None:
    provider = ProviderRegistry().create("mock:fixture", fixture_dir=FIXTURE)
    model, result = StructuredModelCall(provider).run("source_digest", {}, SourceDigestArtifact)
    assert result.schema_valid
    assert model.concepts[0].candidate_id == "CAND001"


def test_raw_prepare_fixture_contract() -> None:
    provider = ProviderRegistry().create("mock:fixture", fixture_dir=FIXTURE)
    model, result = StructuredModelCall(provider).run("raw_prepare", {}, RawPreparationArtifact)
    assert result.schema_valid
    assert model.prepared_markdown.strip()
    assert model.risk_level == "low"


def test_mock_provider_numbered_json_takes_precedence_over_plain_fixture(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "mock"
    fixture_dir.mkdir()
    (fixture_dir / "source_digest.json").write_text('{"summary": "plain"}', encoding="utf-8")
    (fixture_dir / "source_digest.1.json").write_text('{"summary": "first"}', encoding="utf-8")
    provider = ProviderRegistry().create("mock:fixture", fixture_dir=fixture_dir)

    assert provider.generate_raw("source_digest", {}, SourceDigestArtifact) == '{"summary": "first"}'
    assert provider.generate_raw("source_digest", {}, SourceDigestArtifact) == '{"summary": "plain"}'


def test_mock_provider_repair_payload_uses_normal_call_count_contract(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "mock"
    fixture_dir.mkdir()
    (fixture_dir / "draft_rendering.json").write_text('{"status": "plain"}', encoding="utf-8")
    (fixture_dir / "draft_rendering.1.json").write_text('{"status": "first"}', encoding="utf-8")
    (fixture_dir / "draft_rendering.repair.json").write_text('{"status": "stale-repair"}', encoding="utf-8")
    provider = ProviderRegistry().create("mock:fixture", fixture_dir=fixture_dir)

    assert provider.generate_raw("draft_rendering", {"repair_contract": {"mode": "page_scoped_repair"}}, JsonLikeArtifact) == '{"status": "first"}'
    assert provider.generate_raw("draft_rendering", {"repair_contract": {"mode": "page_scoped_repair"}}, JsonLikeArtifact) == '{"status": "plain"}'


def test_source_digest_bad_format_is_blocked(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "mock"
    fixture_dir.mkdir()
    (fixture_dir / "source_digest.json").write_text('{"summary": "bad"}', encoding="utf-8")
    provider = ProviderRegistry().create("mock:fixture", fixture_dir=fixture_dir)
    with pytest.raises(StructuredOutputError):
        StructuredModelCall(provider).run("source_digest", {}, SourceDigestArtifact)


def test_registry_lists_current_provider_types() -> None:
    assert ProviderRegistry().names() == ["mock", "openai_compatible"]


def test_openai_compatible_provider_uses_authorization_header() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}}]})

    provider = OpenAICompatibleProvider(
        "model-test",
        "https://example.test/v1/chat/completions",
        "secret-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert provider.check_live() == '{"ok": true}'
    assert seen["authorization"] == "Bearer secret-key"
    assert seen["body"]["temperature"] == 0
    assert seen["body"]["max_tokens"] == 512
    assert seen["body"]["response_format"] == {"type": "json_object"}


def test_openai_compatible_live_check_uses_twenty_second_timeout() -> None:
    seen: dict[str, object] = {}
    calls = 0

    class FakeResponse:
        text = ""

        def raise_for_status(self) -> None:
            if calls == 1:
                request = httpx.Request("POST", "https://example.test/v1/chat/completions")
                response = httpx.Response(503, text="temporary", request=request)
                raise httpx.HTTPStatusError("temporary", request=request, response=response)
            return None

        def json(self) -> dict[str, object]:
            return {"choices": [{"message": {"content": '{"ok": true}'}}]}

    class FakeClient:
        def post(self, endpoint, *, json, headers, timeout):
            nonlocal calls
            calls += 1
            seen["endpoint"] = endpoint
            seen["body"] = json
            seen["headers"] = headers
            seen["timeout"] = timeout
            return FakeResponse()

    provider = OpenAICompatibleProvider(
        "model-test",
        "https://example.test/v1/chat/completions",
        "secret-key",
        http_client=FakeClient(),
    )

    with pytest.raises(ProviderError, match="HTTP 503"):
        provider.check_live()
    assert calls == 1
    assert seen["timeout"] == 20.0


def test_openai_compatible_generate_raw_defaults_to_five_minute_timeout() -> None:
    seen: dict[str, object] = {}

    class FakeResponse:
        text = ""

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"choices": [{"message": {"content": '{"source_raw_path":"raw/sample.md","summary":"ok"}'}}]}

    class FakeClient:
        def post(self, endpoint, *, json, headers, timeout):
            seen["timeout"] = timeout
            seen["body"] = json
            return FakeResponse()

    provider = OpenAICompatibleProvider(
        "model-test",
        "https://example.test/v1/chat/completions",
        "secret-key",
        http_client=FakeClient(),
    )

    assert provider.generate_raw("source_digest", {}, SourceDigestArtifact) == '{"source_raw_path":"raw/sample.md","summary":"ok"}'
    assert seen["timeout"] == 300.0
    assert seen["body"]["response_format"] == {"type": "json_object"}
    system_prompt = seen["body"]["messages"][0]["content"]
    assert "valid JSON object" in system_prompt
    assert "Arrays must contain JSON objects" in system_prompt


def test_openai_compatible_generate_raw_requires_json_mode() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        seen.append(body)
        return httpx.Response(
            400,
            json={"error": {"message": "response_format is not supported"}},
            request=request,
        )

    provider = OpenAICompatibleProvider(
        "model-test",
        "https://example.test/v1/chat/completions",
        "secret-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(ProviderError, match="response_format is not supported") as exc:
        provider.generate_raw("source_digest", {}, SourceDigestArtifact)
    assert exc.value.status_code == 400
    assert exc.value.attempt_count == 1
    assert len(seen) == 1
    assert provider.last_http_attempt_count == 1
    assert seen[0]["response_format"] == {"type": "json_object"}


def test_openai_compatible_generate_raw_resets_http_attempt_count_between_calls() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, text="temporary overload", request=request)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}}]}, request=request)

    provider = OpenAICompatibleProvider(
        "model-test",
        "https://example.test/v1/chat/completions",
        "secret-key",
        max_retries=1,
        retry_backoff_seconds=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert provider.generate_raw("source_digest", {}, SourceDigestArtifact) == '{"ok": true}'
    assert provider.last_http_attempt_count == 2
    assert provider.generate_raw("source_digest", {}, SourceDigestArtifact) == '{"ok": true}'
    assert provider.last_http_attempt_count == 1


def test_structured_model_call_persists_http_attempt_count_for_provider_retry(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("connection reset", request=request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"value_points": "ok", "quote": "fine"}'}}]},
            request=request,
        )

    provider = OpenAICompatibleProvider(
        "model-test",
        "https://example.test/v1/chat/completions",
        "secret-key",
        max_retries=1,
        retry_backoff_seconds=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    _model, result = StructuredModelCall(
        provider,
        output_dir=tmp_path,
        result_filename="provider_result.json",
    ).run("draft_rendering", {}, JsonLikeArtifact)

    attempt = json.loads((tmp_path / "provider_results" / "attempt-1.json").read_text(encoding="utf-8"))
    assert result.http_attempt_count == 2
    assert attempt["http_attempt_count"] == 2


def test_openai_compatible_generate_raw_keeps_json_mode_after_transient_retry_exhaustion() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        seen.append(body)
        return httpx.Response(503, text="response_format is temporarily unavailable", request=request)

    provider = OpenAICompatibleProvider(
        "model-test",
        "https://example.test/v1/chat/completions",
        "secret-key",
        max_retries=1,
        retry_backoff_seconds=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(ProviderError, match="after 2 attempts"):
        provider.generate_raw("source_digest", {}, SourceDigestArtifact)
    assert len(seen) == 2
    assert provider.last_http_attempt_count == 2
    assert all("response_format" in body for body in seen)


def test_openai_compatible_generate_raw_retries_transient_http_error() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadError("[Errno 54] Connection reset by peer", request=request)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}}]}, request=request)

    provider = OpenAICompatibleProvider(
        "model-test",
        "https://example.test/v1/chat/completions",
        "secret-key",
        retry_backoff_seconds=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert provider.generate_raw("source_digest", {}, SourceDigestArtifact) == '{"ok": true}'
    assert calls == 2


def test_openai_compatible_generate_raw_retries_retryable_status() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, text="temporary overload", request=request)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}}]}, request=request)

    provider = OpenAICompatibleProvider(
        "model-test",
        "https://example.test/v1/chat/completions",
        "secret-key",
        retry_backoff_seconds=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert provider.generate_raw("source_digest", {}, SourceDigestArtifact) == '{"ok": true}'
    assert calls == 2


def test_openai_compatible_generate_raw_honors_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    sleeps: list[float] = []
    monkeypatch.setattr("llmwiki_engine.providers.time.sleep", lambda delay: sleeps.append(delay))

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, text="slow down", headers={"Retry-After": "0.25"}, request=request)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}}]}, request=request)

    provider = OpenAICompatibleProvider(
        "model-test",
        "https://example.test/v1/chat/completions",
        "secret-key",
        retry_backoff_seconds=999,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert provider.generate_raw("source_digest", {}, SourceDigestArtifact) == '{"ok": true}'
    assert calls == 2
    assert sleeps == [0.25]


def test_openai_compatible_generate_raw_reports_retry_exhaustion() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadError("[Errno 54] Connection reset by peer", request=request)

    provider = OpenAICompatibleProvider(
        "model-test",
        "https://example.test/v1/chat/completions",
        "secret-key",
        max_retries=1,
        retry_backoff_seconds=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(ProviderError, match="after 2 attempts"):
        provider.generate_raw("source_digest", {}, SourceDigestArtifact)
    assert calls == 2


def test_openai_compatible_generate_raw_does_not_retry_bad_request() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"error": {"message": "bad request"}}, request=request)

    provider = OpenAICompatibleProvider(
        "model-test",
        "https://example.test/v1/chat/completions",
        "secret-key",
        max_retries=2,
        retry_backoff_seconds=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(ProviderError, match="HTTP 400"):
        provider.generate_raw("source_digest", {}, SourceDigestArtifact)
    assert calls == 1


@pytest.mark.parametrize(
    "exc",
    [
        httpx.InvalidURL("bad url"),
        httpx.UnsupportedProtocol("unsupported protocol"),
        httpx.LocalProtocolError("local protocol error"),
    ],
)
def test_openai_compatible_generate_raw_does_not_retry_permanent_transport_errors(exc: Exception) -> None:
    calls = 0

    class FakeClient:
        def post(self, endpoint, *, json, headers, timeout):
            nonlocal calls
            calls += 1
            raise exc

    provider = OpenAICompatibleProvider(
        "model-test",
        "https://example.test/v1/chat/completions",
        "secret-key",
        max_retries=2,
        retry_backoff_seconds=0,
        http_client=FakeClient(),
    )

    with pytest.raises(ProviderError):
        provider.generate_raw("source_digest", {}, SourceDigestArtifact)
    assert calls == 1


def test_structured_model_call_repairs_json_like_output_locally(tmp_path: Path) -> None:
    class JsonLikeProvider:
        name = "json_like"

        def generate_raw(self, task, payload, output_model):
            return """
            Here is the JSON:
            {
              "value_points": [
                "第一点",
                "第二点"
              ].join("\\n"),
              "quote": "例如"用户喜欢甜食"。"
            }
            """

    model, result = StructuredModelCall(
        JsonLikeProvider(),
        output_dir=tmp_path,
        result_filename="provider_result.json",
    ).run("draft_rendering", {}, JsonLikeArtifact)

    assert model.value_points == "第一点\n第二点"
    assert model.quote == '例如"用户喜欢甜食"。'
    assert result.parse_success is True
    assert result.json_repair_applied is True
    report = json.loads((tmp_path / "structured_repair_report.json").read_text(encoding="utf-8"))
    assert report["repair_count"] == 0
    attempt = json.loads((tmp_path / "provider_results" / "attempt-1.json").read_text(encoding="utf-8"))
    assert attempt["json_repair_applied"] is True


def test_structured_model_call_repairs_control_chars_and_missing_string_quote_locally(tmp_path: Path) -> None:
    class BrokenStringProvider:
        name = "broken_string"

        def generate_raw(self, task, payload, output_model):
            return '{\n  "value_points": "第一点\n第二点",\n  "quote": "问题是什么？}\n}\n'

    model, result = StructuredModelCall(
        BrokenStringProvider(),
        output_dir=tmp_path,
        result_filename="provider_result.json",
    ).run("draft_rendering", {}, JsonLikeArtifact)

    assert model.value_points == "第一点\n第二点"
    assert model.quote == "问题是什么？"
    assert result.parse_success is True
    assert result.json_repair_applied is True
    report = json.loads((tmp_path / "structured_repair_report.json").read_text(encoding="utf-8"))
    assert report["repair_count"] == 0


def test_structured_model_call_rejects_missing_required_and_extra_field(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "mock"
    fixture_dir.mkdir()
    data = json.loads((FIXTURE / "source_digest.json").read_text(encoding="utf-8"))
    data["designs"][0].pop("why_matters")
    data["designs"][0]["unexpected_reason_field"] = "This field is not part of the current contract."
    (fixture_dir / "source_digest.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    provider = ProviderRegistry().create("mock:fixture", fixture_dir=fixture_dir)

    with pytest.raises(StructuredOutputError, match="invalid structured output"):
        StructuredModelCall(
            provider,
            output_dir=tmp_path,
            result_filename="provider_result.json",
            max_repair_attempts=0,
        ).run("source_digest", {}, SourceDigestArtifact)

    report = json.loads((tmp_path / "structured_repair_report.json").read_text(encoding="utf-8"))
    assert report["repair_count"] == 0
    assert report["final_outcome"] == "failed"
    issue_paths = [issue["field_path"] for issue in report["attempts"][0]["issues"]]
    assert "designs.0.why_matters" in issue_paths
    assert "designs.0.unexpected_reason_field" in issue_paths
    attempt = json.loads((tmp_path / "provider_results" / "attempt-1.json").read_text(encoding="utf-8"))
    assert attempt["schema_valid"] is False
    assert attempt["json_repair_applied"] is False


def test_structured_model_call_accepts_current_field_names(tmp_path: Path) -> None:
    provider = ProviderRegistry().create("mock:fixture", fixture_dir=FIXTURE)

    model, result = StructuredModelCall(
        provider,
        output_dir=tmp_path,
        result_filename="provider_result.json",
    ).run("source_digest", {}, SourceDigestArtifact)

    assert model.designs[0].why_matters
    assert result.schema_valid is True
    assert result.json_repair_applied is False
    report = json.loads((tmp_path / "structured_repair_report.json").read_text(encoding="utf-8"))
    assert report["repair_count"] == 0


def test_structured_model_call_redacts_provider_result(tmp_path: Path) -> None:
    class SecretEchoProvider:
        name = "secret_echo"

        def generate_raw(self, task, payload, output_model):
            return '{"source_raw_path": "raw/sample.md", "summary": "ok", "leak": "sk-redact-me"}'

    with pytest.raises(StructuredOutputError):
        StructuredModelCall(
            SecretEchoProvider(),
            output_dir=tmp_path,
            result_filename="provider_result.json",
            redactor=Redactor(("sk-redact-me",)),
        ).run("source_digest", {}, SourceDigestArtifact)

    persisted = (tmp_path / "provider_result.json").read_text(encoding="utf-8")
    assert "sk-redact-me" not in persisted
    assert "[REDACTED]" in persisted
