from pathlib import Path

import pytest

from llmwiki_engine.models import ClaimsArtifact, RawPreparationArtifact, ReviewResult
from llmwiki_engine.providers import ProviderRegistry
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


def test_critic_bad_format_is_blocked(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "mock"
    fixture_dir.mkdir()
    (fixture_dir / "critic_review.json").write_text('{"verdict": "pass"}', encoding="utf-8")
    provider = ProviderRegistry().create("mock:fixture", fixture_dir=fixture_dir)
    with pytest.raises(StructuredOutputError):
        StructuredModelCall(provider).run("critic_review", {}, ReviewResult)


def test_registry_lists_planned_provider_types() -> None:
    assert {"mock", "openai", "ollama", "local_http", "human"}.issubset(set(ProviderRegistry().names()))


def test_openai_provider_uses_requested_api_key_env() -> None:
    provider = ProviderRegistry().create("openai:gpt-test", api_key_env="LLMWIKI_TEST_OPENAI_KEY")
    assert provider.name == "openai"
    assert provider.api_key_env == "LLMWIKI_TEST_OPENAI_KEY"
