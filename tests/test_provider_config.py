from pathlib import Path

import pytest

from llmwiki_engine.io import read_yaml, write_yaml
from llmwiki_engine.pipeline import init_vault
from llmwiki_engine.provider_config import ProviderConfigError, build_provider_execution_context


def test_global_provider_default_and_vault_whole_step_override(tmp_path: Path) -> None:
    home = Path.home()
    global_fixture = home / ".llmwiki" / "global_mock"
    global_fixture.mkdir(parents=True)
    write_yaml(
        home / ".llmwiki" / "config.yaml",
        {
            "providers": {
                "default": {
                    "spec": "mock:fixture",
                    "fixture_dir": "global_mock",
                }
            }
        },
    )
    vault = tmp_path / "vault"
    init_vault(vault)
    vault_fixture = vault / "vault_mock"
    vault_fixture.mkdir()
    config_path = vault / ".llmwiki" / "config.yaml"
    config = read_yaml(config_path)
    config["providers"] = {
        "page_planning": {
            "spec": "mock:fixture",
            "fixture_dir": "vault_mock",
        }
    }
    write_yaml(config_path, config)

    context = build_provider_execution_context(
        vault=vault,
        manifest_contexts=[],
        fixture_dir=None,
        source="initial_run",
        from_step=None,
        tasks=["raw_prepare", "page_planning"],
    )

    assert context.record is not None
    assert context.record.providers["raw_prepare"].fixture_dir == global_fixture.resolve().as_posix()
    assert context.record.providers["page_planning"].fixture_dir == vault_fixture.resolve().as_posix()


def test_global_config_rejects_vault_only_fields(tmp_path: Path) -> None:
    (Path.home() / ".llmwiki").mkdir()
    write_yaml(Path.home() / ".llmwiki" / "config.yaml", {"profile": "project_basic"})
    vault = tmp_path / "vault"
    init_vault(vault)
    with pytest.raises(ProviderConfigError, match="Global config only supports providers"):
        build_provider_execution_context(
            vault=vault,
            manifest_contexts=[],
            fixture_dir=tmp_path,
            source="initial_run",
            from_step=None,
            tasks=["raw_prepare"],
        )


def test_openai_compatible_context_omits_api_key(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault)
    secret = "sk-test-secret"
    config_path = vault / ".llmwiki" / "config.yaml"
    config = read_yaml(config_path)
    config["providers"] = {
        "default": {
            "spec": "openai_compatible:test-model",
            "endpoint": "https://example.test/v1/chat/completions",
            "api_key": secret,
        }
    }
    write_yaml(config_path, config)

    context = build_provider_execution_context(
        vault=vault,
        manifest_contexts=[],
        fixture_dir=None,
        source="initial_run",
        from_step=None,
        tasks=["raw_prepare"],
    )

    assert context.credentials_by_task["raw_prepare"] == secret
    assert context.record is not None
    assert "affected_steps" not in context.record.model_dump()
    assert secret not in context.record.model_dump_json()


def test_endpoint_guard_rejects_userinfo_and_secret_query(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault)
    config_path = vault / ".llmwiki" / "config.yaml"
    config = read_yaml(config_path)
    config["providers"] = {
        "default": {
            "spec": "openai_compatible:test-model",
            "endpoint": "https://user:pass@example.test/v1/chat/completions",
            "api_key": "sk-test",
        }
    }
    write_yaml(config_path, config)
    with pytest.raises(ProviderConfigError, match="username or password"):
        build_provider_execution_context(
            vault=vault,
            manifest_contexts=[],
            fixture_dir=None,
            source="initial_run",
            from_step=None,
            tasks=["raw_prepare"],
        )

    config["providers"]["default"]["endpoint"] = "https://example.test/v1/chat/completions?TOKEN=value"
    write_yaml(config_path, config)
    with pytest.raises(ProviderConfigError, match="secret query parameter"):
        build_provider_execution_context(
            vault=vault,
            manifest_contexts=[],
            fixture_dir=None,
            source="initial_run",
            from_step=None,
            tasks=["raw_prepare"],
        )
