import json
from pathlib import Path

import httpx
import pytest

from llmwiki_engine.models import ClaimsArtifact, RawPreparationArtifact
from llmwiki_engine.providers import OpenAICompatibleProvider, ProviderRegistry
from llmwiki_engine.redaction import Redactor
from llmwiki_engine.structured import StructuredModelCall, StructuredOutputError


FIXTURE = Path(__file__).parent / "fixtures" / "simple_project" / "mock"


def test_mock_provider_returns_claim_fixture() -> None:
    provider = ProviderRegistry().create("mock:fixture", fixture_dir=FIXTURE)
    model, result = StructuredModelCall(provider).run("claim_extraction", {}, ClaimsArtifact)
    assert result.schema_valid
    assert model.claims[0].source_window_id == "W001"


def test_raw_prepare_fixture_contract() -> None:
    provider = ProviderRegistry().create("mock:fixture", fixture_dir=FIXTURE)
    model, result = StructuredModelCall(provider).run("raw_prepare", {}, RawPreparationArtifact)
    assert result.schema_valid
    assert model.prepared_markdown.strip()
    assert model.risk_level == "low"


def test_claim_extraction_bad_format_is_blocked(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "mock"
    fixture_dir.mkdir()
    (fixture_dir / "claim_extraction.json").write_text('{"claims": "bad"}', encoding="utf-8")
    provider = ProviderRegistry().create("mock:fixture", fixture_dir=fixture_dir)
    with pytest.raises(StructuredOutputError):
        StructuredModelCall(provider).run("claim_extraction", {}, ClaimsArtifact)


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
            return '{"claims": [], "leak": "sk-redact-me"}'

    with pytest.raises(StructuredOutputError):
        StructuredModelCall(
            SecretEchoProvider(),
            output_dir=tmp_path,
            result_filename="provider_result.json",
            redactor=Redactor(("sk-redact-me",)),
        ).run("claim_extraction", {}, ClaimsArtifact)

    persisted = (tmp_path / "provider_result.json").read_text(encoding="utf-8")
    assert "sk-redact-me" not in persisted
    assert "[REDACTED]" in persisted
