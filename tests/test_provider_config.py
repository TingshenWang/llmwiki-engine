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
    global_config_path = Path.home() / ".llmwiki" / "config.yaml"
    write_yaml(global_config_path, {"profile": "project_basic"})
    vault = tmp_path / "vault"
    init_vault(vault)
    with pytest.raises(ProviderConfigError) as exc:
        build_provider_execution_context(
            vault=vault,
            manifest_contexts=[],
            fixture_dir=tmp_path,
            source="initial_run",
            from_step=None,
            tasks=["raw_prepare"],
        )
    message = str(exc.value)
    assert "Global config only supports providers" in message
    assert str(global_config_path) in message


@pytest.mark.parametrize(
    ("source", "expected_label"),
    [
        ("global", "global config"),
        ("vault", "vault config"),
    ],
)
def test_config_yaml_parse_error_includes_source(tmp_path: Path, source: str, expected_label: str) -> None:
    vault = tmp_path / "vault"
    init_vault(vault)
    if source == "global":
        config_path = Path.home() / ".llmwiki" / "config.yaml"
        config_path.parent.mkdir(parents=True)
    else:
        config_path = vault / ".llmwiki" / "config.yaml"
    config_path.write_text("providers:\n  default: [\n", encoding="utf-8")

    with pytest.raises(ProviderConfigError) as exc:
        build_provider_execution_context(
            vault=vault,
            manifest_contexts=[],
            fixture_dir=tmp_path,
            source="initial_run",
            from_step=None,
            tasks=["raw_prepare"],
        )
    message = str(exc.value)
    assert "Config YAML parse failed" in message
    assert expected_label in message
    assert str(config_path) in message
    assert "Traceback" not in message


@pytest.mark.parametrize(
    ("source", "expected_label"),
    [
        ("global", "global config"),
        ("vault", "vault config"),
    ],
)
def test_config_root_must_be_mapping_error_includes_source(tmp_path: Path, source: str, expected_label: str) -> None:
    vault = tmp_path / "vault"
    init_vault(vault)
    if source == "global":
        config_path = Path.home() / ".llmwiki" / "config.yaml"
        config_path.parent.mkdir(parents=True)
    else:
        config_path = vault / ".llmwiki" / "config.yaml"
    config_path.write_text("- providers\n", encoding="utf-8")

    with pytest.raises(ProviderConfigError) as exc:
        build_provider_execution_context(
            vault=vault,
            manifest_contexts=[],
            fixture_dir=tmp_path,
            source="initial_run",
            from_step=None,
            tasks=["raw_prepare"],
        )
    message = str(exc.value)
    assert "Config must be a mapping" in message
    assert expected_label in message
    assert str(config_path) in message


def test_vault_unknown_provider_key_error_includes_source(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault)
    config_path = vault / ".llmwiki" / "config.yaml"
    config = read_yaml(config_path)
    config["providers"] = {"unknown": "human"}
    write_yaml(config_path, config)

    with pytest.raises(ProviderConfigError) as exc:
        build_provider_execution_context(
            vault=vault,
            manifest_contexts=[],
            fixture_dir=tmp_path,
            source="initial_run",
            from_step=None,
            tasks=["raw_prepare"],
        )
    message = str(exc.value)
    assert "Unknown provider key(s)" in message
    assert "unknown" in message
    assert str(config_path) in message


def test_provider_unknown_field_error_includes_source_and_key(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault)
    config_path = vault / ".llmwiki" / "config.yaml"
    config = read_yaml(config_path)
    config["providers"] = {
        "page_planning": {
            "spec": "mock:fixture",
            "unexpected": "value",
        }
    }
    write_yaml(config_path, config)

    with pytest.raises(ProviderConfigError) as exc:
        build_provider_execution_context(
            vault=vault,
            manifest_contexts=[],
            fixture_dir=tmp_path,
            source="initial_run",
            from_step=None,
            tasks=["page_planning"],
        )
    message = str(exc.value)
    assert "Unsupported provider config field(s) for page_planning" in message
    assert "unexpected" in message
    assert str(config_path) in message
    assert "provider key: page_planning" in message


def test_openai_missing_required_field_error_includes_source_and_key(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    init_vault(vault)
    config_path = vault / ".llmwiki" / "config.yaml"
    config = read_yaml(config_path)
    config["providers"] = {
        "default": {
            "spec": "openai_compatible:test-model",
            "endpoint": "https://example.test/v1/chat/completions",
        }
    }
    write_yaml(config_path, config)

    with pytest.raises(ProviderConfigError) as exc:
        build_provider_execution_context(
            vault=vault,
            manifest_contexts=[],
            fixture_dir=None,
            source="initial_run",
            from_step=None,
            tasks=["raw_prepare"],
        )
    message = str(exc.value)
    assert "openai_compatible provider for raw_prepare requires api_key" in message
    assert str(config_path) in message
    assert "provider key: default" in message


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
