import json
import subprocess
from pathlib import Path

import httpx
import pytest

from llmwiki_engine.io import read_yaml, write_yaml
from llmwiki_engine.pipeline import init_vault
from llmwiki_engine.provider_checks import check_providers


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
        "claim_extraction": {
            "spec": "openai_compatible:test-model",
            "endpoint": "https://example.test/v1/chat/completions",
            "api_key": "sk-live-secret-b",
        },
        "page_planning": "human",
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


def test_providers_check_live_redacts_http_failure(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault)
    config_path = vault / ".llmwiki" / "config.yaml"
    config = read_yaml(config_path)
    config["providers"] = {
        "default": {
            "spec": "openai_compatible:test-model",
            "endpoint": "https://example.test/v1/chat/completions",
            "api_key": "sk-live-secret",
        }
    }
    write_yaml(config_path, config)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom sk-live-secret", request=request)

    result = check_providers(
        vault,
        live=True,
        http_client_factory=lambda: httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert not result.ok
    rendered = json.dumps(result, default=lambda item: item.__dict__, ensure_ascii=False)
    assert "sk-live-secret" not in rendered
    assert "[REDACTED]" in rendered


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("not-json", "Expecting value"),
        ({"choices": [{"message": {"content": "nope"}}]}, "unexpected content"),
    ],
)
def test_providers_check_live_reports_invalid_responses(tmp_path: Path, body, message: str) -> None:
    vault = tmp_path / "vault"
    init_vault(vault)
    config_path = vault / ".llmwiki" / "config.yaml"
    config = read_yaml(config_path)
    config["providers"] = {
        "default": {
            "spec": "openai_compatible:test-model",
            "endpoint": "https://example.test/v1/chat/completions",
            "api_key": "sk-live-secret",
        }
    }
    write_yaml(config_path, config)

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
