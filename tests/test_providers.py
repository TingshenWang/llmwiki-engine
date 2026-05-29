from pathlib import Path

import pytest

from llmwiki_engine.models import ReviewResult, SemanticAggregationArtifact
from llmwiki_engine.providers import ProviderRegistry
from llmwiki_engine.structured import StructuredModelCall, StructuredOutputError


FIXTURE = Path(__file__).parent / "fixtures" / "simple_project" / "mock"


def test_mock_provider_returns_schema_valid_result() -> None:
    provider = ProviderRegistry().create("mock:fixture", fixture_dir=FIXTURE)
    model, result = StructuredModelCall(provider).run("semantic_aggregation", {}, SemanticAggregationArtifact)
    assert result.schema_valid
    assert model.aggregations[0].aggregation_id == "A001"


def test_critic_bad_format_is_blocked(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "mock"
    fixture_dir.mkdir()
    (fixture_dir / "critic_review.json").write_text('{"verdict": "pass"}', encoding="utf-8")
    provider = ProviderRegistry().create("mock:fixture", fixture_dir=fixture_dir)
    with pytest.raises(StructuredOutputError):
        StructuredModelCall(provider).run("critic_review", {}, ReviewResult)


def test_registry_lists_planned_provider_types() -> None:
    assert {"mock", "openai", "ollama", "local_http", "human"}.issubset(set(ProviderRegistry().names()))

