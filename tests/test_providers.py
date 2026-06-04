import json
from pathlib import Path

import httpx
import pytest

from llmwiki_engine.models import RawPreparationArtifact, SourceDigestArtifact
from llmwiki_engine.providers import OpenAICompatibleProvider, ProviderRegistry
from llmwiki_engine.redaction import Redactor
from llmwiki_engine.structured import StructuredModelCall, StructuredOutputError


FIXTURE = Path(__file__).parent / "fixtures" / "simple_project" / "mock"


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


def test_source_digest_bad_format_is_blocked(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "mock"
    fixture_dir.mkdir()
    (fixture_dir / "source_digest.json").write_text('{"summary": "bad"}', encoding="utf-8")
    provider = ProviderRegistry().create("mock:fixture", fixture_dir=fixture_dir)
    with pytest.raises(StructuredOutputError):
        StructuredModelCall(provider).run("source_digest", {}, SourceDigestArtifact)


def test_registry_lists_planned_provider_types() -> None:
    assert ProviderRegistry().names() == ["human", "mock", "openai_compatible"]


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

    class FakeResponse:
        text = ""

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"choices": [{"message": {"content": '{"ok": true}'}}]}

    class FakeClient:
        def post(self, endpoint, *, json, headers, timeout):
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

    assert provider.check_live() == '{"ok": true}'
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


def test_openai_compatible_generate_raw_falls_back_when_json_mode_is_unsupported() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        seen.append(body)
        if len(seen) == 1:
            return httpx.Response(
                400,
                json={"error": {"message": "response_format is not supported"}},
                request=request,
            )
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}}]}, request=request)

    provider = OpenAICompatibleProvider(
        "model-test",
        "https://example.test/v1/chat/completions",
        "secret-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert provider.generate_raw("source_digest", {}, SourceDigestArtifact) == '{"ok": true}'
    assert seen[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in seen[1]


def test_openai_compatible_live_check_can_skip_json_mode() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}}]})

    provider = OpenAICompatibleProvider(
        "model-test",
        "https://example.test/v1/chat/completions",
        "secret-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert provider.check_live(use_json_mode=False) == '{"ok": true}'
    assert "response_format" not in seen["body"]


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
