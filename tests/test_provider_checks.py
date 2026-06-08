import json
import subprocess
from pathlib import Path

import httpx
import pytest

from llmwiki_engine.io import read_yaml, write_yaml
from llmwiki_engine.pipeline import init_vault
from llmwiki_engine.provider_checks import check_providers


def _write_default_openai_config(vault: Path, *, api_key: str = "sk-live-secret") -> None:
    config_path = vault / ".llmwiki" / "config.yaml"
    config = read_yaml(config_path)
    config["providers"] = {
        "default": {
            "spec": "openai_compatible:test-model",
            "endpoint": "https://example.test/v1/chat/completions",
            "api_key": api_key,
        }
    }
    write_yaml(config_path, config)


def test_providers_check_live_uses_fake_http_without_printing_key(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault)
    config_path = vault / ".llmwiki" / "config.yaml"
    config = read_yaml(config_path)
    config["providers"] = {
        "raw_prepare": {
            "spec": "openai_compatible:test-model",
            "endpoint": "https://example.test/v1/chat/completions",
            "api_key": "sk-live-secret-a",
        },
        "source_digest": {
            "spec": "openai_compatible:test-model",
            "endpoint": "https://example.test/v1/chat/completions",
            "api_key": "sk-live-secret-b",
        },
    }
    write_yaml(config_path, config)
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}}]})

    result = check_providers(
        vault,
        live=True,
        http_client_factory=lambda: httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert result.ok
    assert len(seen) == 2
    assert all(body["max_tokens"] == 512 for body in seen)
    assert all(body["response_format"] == {"type": "json_object"} for body in seen)
    rendered = json.dumps(result, default=lambda item: item.__dict__, ensure_ascii=False)
    assert "sk-live-secret-a" not in rendered
    assert "sk-live-secret-b" not in rendered
    assert {row.credential_label for row in result.rows if row.credential_label} == {"credential #1", "credential #2"}


def test_providers_check_live_deduplicates_same_credential(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault)
    config_path = vault / ".llmwiki" / "config.yaml"
    config = read_yaml(config_path)
    config["providers"] = {
        "default": {
            "spec": "openai_compatible:test-model",
            "endpoint": "https://example.test/v1/chat/completions",
            "api_key": "sk-shared-secret",
        }
    }
    write_yaml(config_path, config)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}}]})

    result = check_providers(
        vault,
        live=True,
        http_client_factory=lambda: httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert result.ok
    assert len(seen) == 1
    assert {row.credential_label for row in result.rows if row.credential_label} == {"credential #1"}
    rendered = json.dumps(result, default=lambda item: item.__dict__, ensure_ascii=False)
    assert "sk-shared-secret" not in rendered
    assert "fingerprint" not in rendered.lower()


def test_providers_check_live_does_not_call_http_for_mock_providers(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault)
    config_path = vault / ".llmwiki" / "config.yaml"
    config = read_yaml(config_path)
    config["providers"] = {
        "default": {
            "spec": "mock:fixture",
            "fixture_dir": str(Path(__file__).parent / "fixtures" / "simple_project" / "mock"),
        }
    }
    write_yaml(config_path, config)
    factory_calls = 0

    def http_client_factory() -> httpx.Client:
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError("mock live check must not create an HTTP client")

    result = check_providers(vault, live=True, http_client_factory=http_client_factory)

    assert result.ok
    assert factory_calls == 0


def test_providers_check_warns_for_mock_without_fixture_dir(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault)

    result = check_providers(vault)

    assert result.ok
    assert any("configure fixture_dir" in warning for warning in result.warnings)
    assert any("--mock-fixture-dir" in warning for warning in result.warnings)
    assert all("--fixture-dir" not in warning for warning in result.warnings)


def test_providers_check_reports_tracked_llmwiki_in_git_repo(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault)
    subprocess.run(["git", "init"], cwd=vault, check=True, capture_output=True)
    subprocess.run(["git", "add", "-f", ".llmwiki/config.yaml"], cwd=vault, check=True, capture_output=True)

    result = check_providers(vault)

    assert not result.ok
    assert any(".llmwiki/" in error for error in result.errors)


def test_providers_check_reports_tracked_llmwiki_in_parent_git_repo(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    vault = repo / "vault"
    init_vault(vault)
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "add", "-f", "vault/.llmwiki/config.yaml"], cwd=repo, check=True, capture_output=True)

    result = check_providers(vault)

    assert not result.ok
    assert any("vault/.llmwiki/config.yaml" in error for error in result.errors)


@pytest.mark.parametrize(
    "exception_factory",
    [
        lambda request: httpx.ConnectError("boom sk-live-secret", request=request),
        lambda request: httpx.ReadTimeout("timeout sk-live-secret", request=request),
    ],
)
def test_providers_check_live_does_not_fallback_for_network_errors_and_redacts(
    tmp_path: Path, exception_factory
) -> None:
    vault = tmp_path / "vault"
    init_vault(vault)
    _write_default_openai_config(vault)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        raise exception_factory(request)

    result = check_providers(
        vault,
        live=True,
        http_client_factory=lambda: httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert not result.ok
    assert len(seen) == 1
    rendered = json.dumps(result, default=lambda item: item.__dict__, ensure_ascii=False)
    assert "sk-live-secret" not in rendered
    assert "[REDACTED]" in rendered


@pytest.mark.parametrize(
    "content",
    [
        '{"ok": true}',
        '  {\n  "ok": true\n}\n',
        '{"ok": true, "extra": "value"}',
    ],
)
def test_providers_check_live_accepts_json_object_variants(tmp_path: Path, content: str) -> None:
    vault = tmp_path / "vault"
    init_vault(vault)
    _write_default_openai_config(vault)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    result = check_providers(
        vault,
        live=True,
        http_client_factory=lambda: httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert result.ok


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("not-json", "Expecting value"),
        ({"choices": [{"message": {"content": ""}}]}, "invalid JSON"),
        ({"choices": [{"message": {"content": "nope"}}]}, "invalid JSON"),
        ({"choices": [{"message": {"content": '{"ok": false}'}}]}, "ok: true"),
        ({"choices": [{"message": {"content": '[{"ok": true}]'}}]}, "root must be an object"),
    ],
)
def test_providers_check_live_reports_invalid_responses(tmp_path: Path, body, message: str) -> None:
    vault = tmp_path / "vault"
    init_vault(vault)
    _write_default_openai_config(vault)

    def handler(request: httpx.Request) -> httpx.Response:
        if isinstance(body, str):
            return httpx.Response(200, text=body)
        return httpx.Response(200, json=body)

    result = check_providers(
        vault,
        live=True,
        http_client_factory=lambda: httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert not result.ok
    assert any(message in error for error in result.errors)


def test_providers_check_live_falls_back_when_json_mode_is_unsupported(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault)
    _write_default_openai_config(vault)
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        seen.append(body)
        if len(seen) == 1:
            return httpx.Response(400, json={"error": {"message": "response_format is not supported"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}}]})

    result = check_providers(
        vault,
        live=True,
        http_client_factory=lambda: httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert result.ok
    assert len(seen) == 2
    assert seen[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in seen[1]
    assert any("prompt-only JSON live check" in warning for warning in result.warnings)


def test_providers_check_live_reports_fallback_failure_redacted(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault)
    _write_default_openai_config(vault)
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content.decode("utf-8")))
        if len(seen) == 1:
            return httpx.Response(422, json={"error": {"message": "json_object response_format unsupported"}})
        raise httpx.ConnectError("fallback boom sk-live-secret", request=request)

    result = check_providers(
        vault,
        live=True,
        http_client_factory=lambda: httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert not result.ok
    assert len(seen) == 2
    rendered = json.dumps(result, default=lambda item: item.__dict__, ensure_ascii=False)
    assert "sk-live-secret" not in rendered
    assert "[REDACTED]" in rendered


@pytest.mark.parametrize("status_code", [401, 403, 429, 500])
def test_providers_check_live_does_not_fallback_for_non_json_mode_errors(
    tmp_path: Path, status_code: int
) -> None:
    vault = tmp_path / "vault"
    init_vault(vault)
    _write_default_openai_config(vault)
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            status_code,
            json={"error": {"message": "response_format is not supported sk-live-secret"}},
        )

    result = check_providers(
        vault,
        live=True,
        http_client_factory=lambda: httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert not result.ok
    assert len(seen) == 1
    rendered = json.dumps(result, default=lambda item: item.__dict__, ensure_ascii=False)
    assert "sk-live-secret" not in rendered


def test_providers_check_live_does_not_fallback_for_plain_response_format_400(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault)
    _write_default_openai_config(vault)
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(400, json={"error": {"message": "response_format field is malformed"}})

    result = check_providers(
        vault,
        live=True,
        http_client_factory=lambda: httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert not result.ok
    assert len(seen) == 1
