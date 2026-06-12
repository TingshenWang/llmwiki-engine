from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from llmwiki_engine.cli import app
from llmwiki_engine.lite.io import read_json
from llmwiki_engine.lite.pipeline import init_vault, run_ingest
from llmwiki_engine.lite.profile import load_profile
from llmwiki_engine.lite.prompts import source_digest_prompt
from llmwiki_engine.lite.models import SourceDigest
from llmwiki_engine.lite.providers import (
    ProviderRegistry,
    ProviderSpec,
    _parse_chat_completion_json,
    _response_was_truncated,
    build_chat_payload,
    load_provider_registry,
)

from test_lite_pipeline import write_raw


def test_openai_compatible_payload_uses_json_schema_and_sanitizes_key(tmp_path: Path) -> None:
    vault = init_vault(tmp_path / "vault")
    profile = load_profile(vault)
    spec = ProviderSpec(
        spec="openai_compatible:test-model",
        endpoint="https://example.test/v1/chat/completions",
        api_key="secret-value",
    )
    request = source_digest_prompt(raw_path="raw/a.md", raw_sha256="abc", raw_text="# A\n\nBody", profile=profile)

    payload = build_chat_payload(spec, request)
    context = spec.sanitized_context()

    assert payload["model"] == "test-model"
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["name"] == "llmwiki_lite_source_digest"
    assert "SourceDigestCandidate" in json.dumps(payload["response_format"]["json_schema"]["schema"])
    assert context["has_api_key"] is True
    assert "secret-value" not in json.dumps(context)


def test_deepseek_endpoint_uses_json_object_mode(tmp_path: Path) -> None:
    vault = init_vault(tmp_path / "vault")
    profile = load_profile(vault)
    spec = ProviderSpec(
        spec="openai_compatible:deepseek-v4-flash",
        endpoint="https://api.deepseek.com/v1/chat/completions",
        api_key="redacted-test-key",
    )
    request = source_digest_prompt(raw_path="raw/a.md", raw_sha256="abc", raw_text="# A\n\nBody", profile=profile)

    payload = build_chat_payload(spec, request)

    assert spec.effective_json_mode() == "json_object"
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["max_tokens"] == 262144
    assert "json_output_example" in payload["messages"][1]["content"]
    assert spec.max_retries == 1


def test_response_was_truncated_detects_length_finish_reason() -> None:
    assert _response_was_truncated({"choices": [{"finish_reason": "length"}]}) is True
    assert _response_was_truncated({"choices": [{"finish_reason": "stop"}]}) is False


def test_openai_compatible_retries_truncated_response(tmp_path: Path, monkeypatch) -> None:
    vault = init_vault(tmp_path / "vault")
    profile = load_profile(vault)
    raw_sha = "abc123"
    request = source_digest_prompt(raw_path="raw/a.md", raw_sha256=raw_sha, raw_text="# A\n\nBody", profile=profile)
    spec = ProviderSpec(
        spec="openai_compatible:test-model",
        endpoint="https://example.test/v1/chat/completions",
        api_key="test-key",
        max_retries=1,
        max_tokens=8,
        retry_backoff_seconds=0,
    )
    valid_digest = {
        "source_raw_path": "raw/a.md",
        "raw_sha256": raw_sha,
        "summary": "A valid digest after retry.",
        "key_takeaways": ["Retry recovered from truncation."],
        "entities": [],
        "concepts": [],
        "designs": [],
        "comparisons": [],
        "open_questions": [],
        "budget_deferred_candidates": [],
        "weak_or_noise_items": [],
    }
    responses = [
        {"choices": [{"finish_reason": "length", "message": {"role": "assistant", "content": '{"source_raw_path": "raw/a.md"'}}]},
        {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(valid_digest)}}]},
    ]
    posted_payloads: list[dict[str, object]] = []

    class DummyResponse:
        status_code = 200

        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload
            self.text = json.dumps(payload)

        def json(self) -> dict[str, object]:
            return self._payload

        def raise_for_status(self) -> None:
            return None

    class DummyClient:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        def __enter__(self) -> "DummyClient":
            return self

        def __exit__(self, exc_type, exc, traceback) -> bool:
            return False

        def post(self, *args, **kwargs) -> DummyResponse:
            posted_payloads.append(kwargs["json"])
            return DummyResponse(responses.pop(0))

    monkeypatch.setattr("llmwiki_engine.lite.providers.httpx.Client", DummyClient)

    result = ProviderRegistry({"source_digest": spec}).call_structured("source_digest", request, SourceDigest)

    assert result.model_calls == 2
    assert posted_payloads[0]["max_tokens"] == 8
    assert posted_payloads[1]["max_tokens"] == 16
    assert result.output.summary == "A valid digest after retry."
    assert result.provider_result["provider"]["retry_count"] == 1
    assert result.provider_result["provider"]["retry_reason"] == "truncated_response"
    assert responses == []


def test_truncated_retry_caps_at_256k(tmp_path: Path, monkeypatch) -> None:
    vault = init_vault(tmp_path / "vault")
    profile = load_profile(vault)
    raw_sha = "abc123"
    request = source_digest_prompt(raw_path="raw/a.md", raw_sha256=raw_sha, raw_text="# A\n\nBody", profile=profile)
    spec = ProviderSpec(
        spec="openai_compatible:test-model",
        endpoint="https://example.test/v1/chat/completions",
        api_key="test-key",
        max_retries=1,
        max_tokens=200000,
        retry_backoff_seconds=0,
    )
    valid_digest = {
        "source_raw_path": "raw/a.md",
        "raw_sha256": raw_sha,
        "summary": "A valid digest after retry.",
        "key_takeaways": ["Retry recovered from truncation."],
        "entities": [],
        "concepts": [],
        "designs": [],
        "comparisons": [],
        "open_questions": [],
        "budget_deferred_candidates": [],
        "weak_or_noise_items": [],
    }
    responses = [
        {"choices": [{"finish_reason": "length", "message": {"role": "assistant", "content": '{"source_raw_path": "raw/a.md"'}}]},
        {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(valid_digest)}}]},
    ]
    posted_payloads: list[dict[str, object]] = []

    class DummyResponse:
        status_code = 200

        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload
            self.text = json.dumps(payload)

        def json(self) -> dict[str, object]:
            return self._payload

        def raise_for_status(self) -> None:
            return None

    class DummyClient:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        def __enter__(self) -> "DummyClient":
            return self

        def __exit__(self, exc_type, exc, traceback) -> bool:
            return False

        def post(self, *args, **kwargs) -> DummyResponse:
            posted_payloads.append(kwargs["json"])
            return DummyResponse(responses.pop(0))

    monkeypatch.setattr("llmwiki_engine.lite.providers.httpx.Client", DummyClient)

    result = ProviderRegistry({"source_digest": spec}).call_structured("source_digest", request, SourceDigest)

    assert result.model_calls == 2
    assert posted_payloads[0]["max_tokens"] == 200000
    assert posted_payloads[1]["max_tokens"] == 262144


def test_parse_chat_completion_json_ignores_text_after_first_object() -> None:
    parsed = _parse_chat_completion_json({"choices": [{"message": {"content": '{"ok": true}\\nextra note'}}]})

    assert parsed == {"ok": True}


def test_parse_chat_completion_json_skips_bad_brace_before_object() -> None:
    parsed = _parse_chat_completion_json({"choices": [{"message": {"content": "not json {oops}\\n{\"ok\": true}"}}]})

    assert parsed == {"ok": True}


def test_mock_fixture_provider_writes_prompt_and_provider_result_artifacts(tmp_path: Path) -> None:
    vault = init_vault(tmp_path / "vault")
    raw = write_raw(vault)
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    raw_sha = _sha256(raw)
    (fixture_dir / "source_digest.json").write_text(
        json.dumps(
            {
                "source_raw_path": "raw/project_note.md",
                "raw_sha256": raw_sha,
                "summary": "模型 fixture 生成的 digest。",
                "key_takeaways": ["保留 raw 作为证据。"],
                "concepts": [
                    {
                        "candidate_id": "CAND-FIXTURE",
                        "kind": "concept",
                        "name": "测试 Provider",
                        "suggested_page_title": "测试 Provider",
                        "summary": "用于验证 provider prompt/schema 分支。",
                        "source_basis": "fixture 来源依据。",
                        "source_refs": [{"raw_path": "raw/project_note.md", "raw_sha256": raw_sha, "locator": "whole_file"}],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (vault / ".llmwiki" / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "providers": {
                    "default": {"spec": "local:heuristic"},
                    "source_digest": {"spec": "mock:fixture", "fixture_dir": fixture_dir.as_posix()},
                }
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    manifest = run_ingest(vault, raw, slug="mock-provider", emit_progress=False, allow_test_providers=True)

    run_dir = vault / ".llmwiki" / "runs" / "ingest" / manifest.operation_id
    source_step = next(step for step in manifest.steps if step.name == "source_digest")
    assert source_step.model_calls == 1
    assert (run_dir / "source_digest" / "model_calls" / "source_digest.prompt.json").exists()
    provider_result = read_json(run_dir / "source_digest" / "model_calls" / "source_digest.provider_result.json")
    assert provider_result["provider"]["spec"] == "mock:fixture"
    receipt = read_json(vault / manifest.receipt_path)
    assert receipt["provider_contexts"]["source_digest"]["spec"] == "mock:fixture"


def test_load_provider_registry_merges_vault_yaml(tmp_path: Path) -> None:
    vault = init_vault(tmp_path / "vault")
    (vault / ".llmwiki" / "config.yaml").write_text(
        yaml.safe_dump({"providers": {"default": {"spec": "local:heuristic"}, "merge_plan": "local:heuristic"}}, sort_keys=False),
        encoding="utf-8",
    )

    registry = load_provider_registry(vault)

    assert registry.provider_for("source_digest").spec == "local:heuristic"
    assert registry.provider_for("merge_plan").spec == "local:heuristic"


def test_global_provider_config_applies_when_vault_has_no_provider_yaml(tmp_path: Path) -> None:
    home_config = Path.home() / ".llmwiki" / "config.yaml"
    home_config.parent.mkdir(parents=True, exist_ok=True)
    home_config.write_text(
        yaml.safe_dump(
            {
                "providers": {
                    "default": {
                        "spec": "openai_compatible:deepseek-v4-flash",
                        "endpoint": "https://api.deepseek.com/v1/chat/completions",
                        "api_key": "redacted-test-key",
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    vault = init_vault(tmp_path / "vault")

    registry = load_provider_registry(vault)

    assert not (vault / ".llmwiki" / "config.yaml").exists()
    assert registry.provider_for("source_digest").spec == "openai_compatible:deepseek-v4-flash"
    assert registry.provider_for("source_digest").endpoint == "https://api.deepseek.com/v1/chat/completions"


def test_providers_check_live_rejects_local_and_mock_providers(tmp_path: Path) -> None:
    vault = init_vault(tmp_path / "vault")
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    (vault / ".llmwiki" / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "providers": {
                    "default": {"spec": "local:heuristic"},
                    "source_digest": {"spec": "mock:fixture", "fixture_dir": fixture_dir.as_posix()},
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    registry = load_provider_registry(vault)
    reports = {report["name"]: report for report in registry.check(live=True)}

    assert reports["default"]["ok"] is False
    assert reports["default"]["message"] == "not a real model provider"
    assert reports["source_digest"]["ok"] is False
    assert reports["source_digest"]["message"] == "mock provider is not live"

    result = CliRunner().invoke(app, ["providers", "check", str(vault), "--live"])
    assert result.exit_code == 2
    assert "不是真实模型 provider" in result.output
    assert "mock 不是 live provider" in result.output


def test_run_ingest_requires_real_model_provider_by_default(tmp_path: Path) -> None:
    vault = init_vault(tmp_path / "vault")
    raw = write_raw(vault)

    from llmwiki_engine.lite.pipeline import PipelineError

    try:
        run_ingest(vault, raw, slug="requires-real", emit_progress=False)
    except PipelineError as exc:
        assert "必须使用真实模型 provider" in str(exc)
    else:
        raise AssertionError("run_ingest should reject local heuristic provider by default")


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
