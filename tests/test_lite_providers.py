from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import yaml
from typer.testing import CliRunner

from llmwiki_engine.cli import app
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
    assert "SourceContentUnit" in json.dumps(payload["response_format"]["json_schema"]["schema"])
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
    assert spec.timeout_seconds == 180.0
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
        "content_units": [],
        "weak_or_noise_items": [],
    }
    responses = [
        {
            "choices": [{"finish_reason": "length", "message": {"role": "assistant", "content": json.dumps({"source_raw_path": "raw/a.md"})}}],
            "usage": {
                "prompt_tokens": 10,
                "prompt_cache_hit_tokens": 2,
                "prompt_cache_miss_tokens": 8,
                "completion_tokens": 1,
                "total_tokens": 11,
            },
        },
        {
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(valid_digest)}}],
            "usage": {
                "prompt_tokens": 20,
                "prompt_cache_hit_tokens": 10,
                "prompt_cache_miss_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 25,
                "completion_tokens_details": {"reasoning_tokens": 3},
            },
        },
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
    assert result.api_calls[0]["status"] == "paused"
    assert result.api_calls[0]["finish_reason"] == "length"
    assert result.api_calls[0]["prompt_tokens"] == 10
    assert result.api_calls[0]["prompt_cache_hit_tokens"] == 2
    assert result.api_calls[0]["error"] == "truncated_response"
    assert result.api_calls[1]["status"] == "success"
    assert result.api_calls[1]["finish_reason"] == "stop"
    assert result.api_calls[1]["completion_tokens"] == 5
    assert result.api_calls[1]["reasoning_tokens"] == 3
    assert result.provider_result["api_calls"] == result.api_calls
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
        "content_units": [],
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


def test_openai_compatible_retries_timeout_and_records_paused_call(tmp_path: Path, monkeypatch) -> None:
    vault = init_vault(tmp_path / "vault")
    profile = load_profile(vault)
    raw_sha = "abc123"
    request = source_digest_prompt(raw_path="raw/a.md", raw_sha256=raw_sha, raw_text="# A\n\nBody", profile=profile)
    spec = ProviderSpec(
        spec="openai_compatible:test-model",
        endpoint="https://example.test/v1/chat/completions",
        api_key="test-key",
        max_retries=1,
        retry_backoff_seconds=0,
    )
    valid_digest = {
        "source_raw_path": "raw/a.md",
        "raw_sha256": raw_sha,
        "summary": "A valid digest after timeout retry.",
        "key_takeaways": ["Timeout recovered at the same step request."],
        "content_units": [],
        "weak_or_noise_items": [],
    }
    posted_payloads: list[dict[str, object]] = []
    client_timeouts: list[float] = []

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
            client_timeouts.append(timeout)

        def __enter__(self) -> "DummyClient":
            return self

        def __exit__(self, exc_type, exc, traceback) -> bool:
            return False

        def post(self, *args, **kwargs) -> DummyResponse:
            posted_payloads.append(kwargs["json"])
            if len(posted_payloads) == 1:
                raise httpx.ReadTimeout("read timed out")
            return DummyResponse({"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(valid_digest)}}]})

    monkeypatch.setattr("llmwiki_engine.lite.providers.httpx.Client", DummyClient)

    result = ProviderRegistry({"source_digest": spec}).call_structured("source_digest", request, SourceDigest)

    assert result.model_calls == 2
    assert client_timeouts == [180.0, 180.0]
    assert len(posted_payloads) == 2
    assert result.output.summary == "A valid digest after timeout retry."
    assert result.provider_result["provider"]["retry_count"] == 1
    assert result.provider_result["provider"]["retry_reason"] == "provider_timeout"
    assert result.api_calls[0]["status"] == "paused"
    assert result.api_calls[0]["error"].startswith("provider_timeout")
    assert result.api_calls[0]["response_status_code"] == 0
    assert result.api_calls[1]["status"] == "success"


def test_openai_compatible_wall_timeout_retries_slow_post(tmp_path: Path, monkeypatch) -> None:
    vault = init_vault(tmp_path / "vault")
    profile = load_profile(vault)
    raw_sha = "abc123"
    request = source_digest_prompt(raw_path="raw/a.md", raw_sha256=raw_sha, raw_text="# A\n\nBody", profile=profile)
    spec = ProviderSpec(
        spec="openai_compatible:test-model",
        endpoint="https://example.test/v1/chat/completions",
        api_key="test-key",
        timeout_seconds=0.01,
        max_retries=1,
        retry_backoff_seconds=0,
    )
    valid_digest = {
        "source_raw_path": "raw/a.md",
        "raw_sha256": raw_sha,
        "summary": "A valid digest after wall timeout retry.",
        "key_takeaways": ["Wall timeout recovered at the same step request."],
        "content_units": [],
        "weak_or_noise_items": [],
    }
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
            if len(posted_payloads) == 1:
                time.sleep(0.05)
            return DummyResponse({"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(valid_digest)}}]})

    monkeypatch.setattr("llmwiki_engine.lite.providers.httpx.Client", DummyClient)

    result = ProviderRegistry({"source_digest": spec}).call_structured("source_digest", request, SourceDigest)

    assert result.model_calls == 2
    assert result.output.summary == "A valid digest after wall timeout retry."
    assert result.provider_result["provider"]["retry_reason"] == "provider_timeout"
    assert result.api_calls[0]["status"] == "paused"
    assert "exceeded timeout_seconds=0.01" in result.api_calls[0]["error"]
    assert result.api_calls[1]["status"] == "success"


def test_parse_chat_completion_json_ignores_text_after_first_object() -> None:
    parsed = _parse_chat_completion_json({"choices": [{"message": {"content": '{"ok": true}\\nextra note'}}]})

    assert parsed == {"ok": True}


def test_parse_chat_completion_json_skips_bad_brace_before_object() -> None:
    parsed = _parse_chat_completion_json({"choices": [{"message": {"content": "not json {oops}\\n{\"ok\": true}"}}]})

    assert parsed == {"ok": True}


def test_load_provider_registry_merges_vault_yaml(tmp_path: Path) -> None:
    vault = init_vault(tmp_path / "vault")
    (vault / ".llmwiki" / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "providers": {
                    "default": {
                        "spec": "openai_compatible:deepseek-v4-flash",
                        "endpoint": "https://api.deepseek.com/v1/chat/completions",
                        "api_key": "redacted-test-key",
                    },
                    "merge_plan": {
                        "spec": "openai_compatible:merge-model",
                        "endpoint": "https://example.test/v1/chat/completions",
                        "api_key": "redacted-test-key",
                    },
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    registry = load_provider_registry(vault)

    assert registry.provider_for("source_digest").spec == "openai_compatible:deepseek-v4-flash"
    assert registry.provider_for("merge_plan").spec == "openai_compatible:merge-model"


def test_global_provider_config_applies_when_vault_has_no_provider_yaml(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", home.as_posix())
    home_config = home / ".llmwiki" / "config.yaml"
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


def test_providers_check_live_rejects_local_and_unsupported_providers(tmp_path: Path) -> None:
    vault = init_vault(tmp_path / "vault")
    (vault / ".llmwiki" / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "providers": {
                    "default": {"spec": "local:heuristic"},
                    "source_digest": {"spec": "unsupported:test-model"},
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
    assert reports["source_digest"]["message"] == "unsupported provider spec"

    result = CliRunner().invoke(app, ["providers", "check", str(vault), "--live"])
    assert result.exit_code == 2
    assert "不是真实模型 provider" in result.output
    assert "不支持的 provider" in result.output


def test_providers_check_live_calls_chat_completion(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", home.as_posix())
    vault = init_vault(tmp_path / "vault")
    (vault / ".llmwiki" / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "providers": {
                    "default": {
                        "spec": "openai_compatible:test-model",
                        "endpoint": "https://example.test/v1/chat/completions",
                        "api_key": "test-key",
                        "max_tokens": 262144,
                        "max_retries": 3,
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    posted_payloads: list[dict[str, object]] = []

    class DummyResponse:
        status_code = 200
        text = "{}"

        def json(self) -> dict[str, object]:
            return {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": '{"ok": true}'}}]}

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
            return DummyResponse()

    monkeypatch.setattr("llmwiki_engine.lite.providers.httpx.Client", DummyClient)

    result = CliRunner().invoke(app, ["providers", "check", str(vault), "--live"])

    assert result.exit_code == 0, result.output
    assert "真实调用通过" in result.output
    assert len(posted_payloads) == 1
    assert posted_payloads[0]["max_tokens"] == 512
    assert posted_payloads[0]["response_format"]["type"] == "json_schema"
    assert posted_payloads[0]["response_format"]["json_schema"]["name"] == "llmwiki_lite_provider_live_check"
    assert "message" not in posted_payloads[0]["response_format"]["json_schema"]["schema"].get("properties", {})
    assert "只返回 ok=true" in posted_payloads[0]["messages"][1]["content"]

    report = load_provider_registry(vault).check(live=True)[0]
    assert report["context"]["max_tokens"] == 262144
    assert report["context"]["live_max_tokens"] == 512
    assert report["context"]["live_model_calls"] == 1


def test_providers_check_live_reports_call_failure(monkeypatch) -> None:
    spec = ProviderSpec(
        spec="openai_compatible:test-model",
        endpoint="https://example.test/v1/chat/completions",
        api_key="test-key",
        retry_backoff_seconds=0,
    )

    class FailingClient:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        def __enter__(self) -> "FailingClient":
            return self

        def __exit__(self, exc_type, exc, traceback) -> bool:
            return False

        def post(self, *args, **kwargs):
            raise RuntimeError("network down")

    monkeypatch.setattr("llmwiki_engine.lite.providers.httpx.Client", FailingClient)
    reports = ProviderRegistry({"default": spec}).check(live=True)

    assert reports[0]["ok"] is False
    assert "live check failed" in reports[0]["message"]
    assert "network down" in reports[0]["message"]


def test_run_ingest_requires_real_model_provider_by_default(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", home.as_posix())
    vault = init_vault(tmp_path / "vault")
    raw = write_raw(vault)

    from llmwiki_engine.lite.pipeline import PipelineError

    try:
        run_ingest(vault, raw, slug="requires-real", emit_progress=False)
    except PipelineError as exc:
        assert "必须使用真实模型 provider" in str(exc)
    else:
        raise AssertionError("run_ingest should reject missing real model provider by default")
