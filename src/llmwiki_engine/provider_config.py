from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qsl, urlsplit

import yaml

from .io import read_yaml
from .models import ProviderContextRecord, ProviderRuntimeSpec
from .providers import (
    DEFAULT_OPENAI_COMPATIBLE_MAX_RETRIES,
    DEFAULT_OPENAI_COMPATIBLE_RETRY_BACKOFF_SECONDS,
    Provider,
    ProviderRegistry,
)
from .redaction import Redactor
from .steps import PROVIDER_CONFIG_KEYS


class ProviderConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderConfigEntry:
    value: Any
    base_dir: Path
    source_label: str
    source_path: Path | None
    provider_key: str


@dataclass(frozen=True)
class ProviderExecutionContext:
    record: ProviderContextRecord | None
    credentials_by_task: dict[str, str]
    redactor: Redactor

    def runtime_for_task(self, task: str) -> ProviderRuntimeSpec:
        if self.record is None or task not in self.record.providers:
            raise ProviderConfigError(f"No provider execution context found for task: {task}")
        return self.record.providers[task]

    def provider_for_task(self, task: str, *, http_client: Any | None = None) -> Provider:
        runtime = self.runtime_for_task(task)
        fixture_dir = Path(runtime.fixture_dir) if runtime.fixture_dir else None
        return ProviderRegistry().create(
            runtime.spec,
            fixture_dir=fixture_dir,
            endpoint=runtime.endpoint,
            api_key=self.credentials_by_task.get(task),
            max_retries=runtime.max_retries,
            retry_backoff_seconds=runtime.retry_backoff_seconds,
            http_client=http_client,
        )


SECRET_QUERY_KEYS = {"api_key", "key", "token", "access_token", "authorization"}
PROVIDER_RUNTIME_KEYS = {"spec", "endpoint", "api_key", "fixture_dir", "max_retries", "retry_backoff_seconds"}


def global_config_path() -> Path:
    return Path.home() / ".llmwiki" / "config.yaml"


def load_provider_entries(vault: Path) -> dict[str, ProviderConfigEntry]:
    entries: dict[str, ProviderConfigEntry] = {}
    global_path = global_config_path()
    if global_path.exists():
        global_config = _read_config(global_path, "global config")
        unknown_global = set(global_config) - {"providers"}
        if unknown_global:
            names = ", ".join(sorted(unknown_global))
            raise ProviderConfigError(
                f"Global config only supports providers; unsupported top-level field(s): {names} "
                f"(source: global config {global_path})"
            )
        entries.update(
            _provider_entries_from_config(
                global_config,
                base_dir=global_path.parent,
                source_label="global config",
                source_path=global_path,
            )
        )

    vault_path = vault / ".llmwiki" / "config.yaml"
    vault_config = _read_config(vault_path, "vault config")
    entries.update(
        _provider_entries_from_config(
            vault_config,
            base_dir=vault,
            source_label="vault config",
            source_path=vault_path,
        )
    )
    return entries


def _read_config(path: Path, source_label: str) -> dict[str, Any]:
    try:
        data = read_yaml(path)
    except FileNotFoundError as exc:
        raise ProviderConfigError(f"Config file not found (source: {source_label} {path})") from exc
    except yaml.YAMLError as exc:
        location = _yaml_error_location(exc)
        raise ProviderConfigError(f"Config YAML parse failed{location} (source: {source_label} {path})") from exc
    if not isinstance(data, dict):
        raise ProviderConfigError(f"Config must be a mapping (source: {source_label} {path})")
    return data


def _yaml_error_location(exc: yaml.YAMLError) -> str:
    mark = getattr(exc, "problem_mark", None)
    if mark is None:
        return ""
    return f" at line {mark.line + 1}, column {mark.column + 1}"


def _provider_entries_from_config(
    config: dict[str, Any],
    *,
    base_dir: Path,
    source_label: str,
    source_path: Path,
) -> dict[str, ProviderConfigEntry]:
    providers = config.get("providers", {})
    if providers is None:
        providers = {}
    if not isinstance(providers, dict):
        raise ProviderConfigError(f"providers must be a mapping (source: {source_label} {source_path}).")
    unknown = set(providers) - set(PROVIDER_CONFIG_KEYS)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ProviderConfigError(f"Unknown provider key(s): {names} (source: {source_label} {source_path}).")
    return {
        key: ProviderConfigEntry(
            value=value,
            base_dir=base_dir,
            source_label=source_label,
            source_path=source_path,
            provider_key=key,
        )
        for key, value in providers.items()
    }


def provider_entry_for_task(entries: dict[str, ProviderConfigEntry], task: str) -> ProviderConfigEntry:
    if task in entries:
        return entries[task]
    if "default" in entries:
        return entries["default"]
    return ProviderConfigEntry(
        value="mock:fixture",
        base_dir=Path.cwd(),
        source_label="built-in default",
        source_path=None,
        provider_key="default",
    )


def build_provider_execution_context(
    *,
    vault: Path,
    manifest_contexts: list[ProviderContextRecord],
    fixture_dir: Path | None,
    mock_fixture_dir: Path | None = None,
    source: Literal["initial_run", "resume_current_config"],
    from_step: str | None,
    tasks: list[str],
    require_mock_fixture: bool = True,
) -> ProviderExecutionContext:
    entries = load_provider_entries(vault)
    secrets: list[str] = _api_keys_from_entries(entries)
    providers: dict[str, ProviderRuntimeSpec] = {}
    credentials_by_task: dict[str, str] = {}
    resolved_mock_fixture_dir = mock_fixture_dir.resolve() if mock_fixture_dir is not None else None
    if resolved_mock_fixture_dir is not None and not resolved_mock_fixture_dir.is_dir():
        raise ProviderConfigError(f"mock fixture_dir does not exist: {resolved_mock_fixture_dir}")
    for task in tasks:
        if resolved_mock_fixture_dir is not None:
            runtime = ProviderRuntimeSpec(spec="mock:fixture", fixture_dir=resolved_mock_fixture_dir.as_posix())
            credential = None
        else:
            entry = provider_entry_for_task(entries, task)
            runtime, credential = provider_runtime_spec_for_task(
                task,
                entry,
                fixture_dir=fixture_dir,
                require_mock_fixture=require_mock_fixture,
            )
        providers[task] = runtime
        if credential is not None:
            credentials_by_task[task] = credential
            if credential not in secrets:
                secrets.append(credential)

    record = None
    if tasks:
        record = ProviderContextRecord(
            record_id=f"provider-context-{len(manifest_contexts) + 1:03d}",
            source=source,
            from_step=from_step,
            providers=providers,
        )
    return ProviderExecutionContext(record=record, credentials_by_task=credentials_by_task, redactor=Redactor(tuple(secrets)))


def provider_runtime_spec_for_task(
    task: str,
    entry: ProviderConfigEntry,
    *,
    fixture_dir: Path | None,
    require_mock_fixture: bool,
) -> tuple[ProviderRuntimeSpec, str | None]:
    value = entry.value
    endpoint = None
    api_key = None
    config_fixture_dir = None
    max_retries = None
    retry_backoff_seconds = None
    if isinstance(value, dict):
        unknown = set(value) - PROVIDER_RUNTIME_KEYS
        if unknown:
            names = ", ".join(sorted(unknown))
            raise _provider_error(entry, task, f"Unsupported provider config field(s) for {task}: {names}")
        spec = value.get("spec")
        endpoint = value.get("endpoint")
        api_key = value.get("api_key")
        config_fixture_dir = value.get("fixture_dir")
        max_retries = value.get("max_retries")
        retry_backoff_seconds = value.get("retry_backoff_seconds")
    else:
        spec = value
    if not isinstance(spec, str) or not spec:
        raise _provider_error(entry, task, f"Invalid provider config for task: {task}")
    provider_name = spec.partition(":")[0]
    _, separator, model = spec.partition(":")
    if provider_name not in {"mock", "human", "openai_compatible"}:
        raise _provider_error(entry, task, f"Unknown provider spec for {task}: {provider_name}")

    _validate_optional_string(entry, task, endpoint, f"Invalid provider endpoint for task: {task}")
    _validate_optional_string(entry, task, api_key, f"Invalid provider api_key for task: {task}")
    _validate_optional_string(entry, task, config_fixture_dir, f"Invalid provider fixture_dir for task: {task}")
    _validate_optional_nonnegative_int(entry, task, max_retries, f"Invalid provider max_retries for task: {task}")
    _validate_optional_nonnegative_number(
        entry,
        task,
        retry_backoff_seconds,
        f"Invalid provider retry_backoff_seconds for task: {task}",
    )

    if provider_name == "openai_compatible":
        if not separator or not model:
            raise _provider_error(entry, task, f"openai_compatible provider for {task} requires model in spec.")
        if not isinstance(value, dict):
            raise _provider_error(entry, task, f"openai_compatible provider for {task} must use mapping config.")
        if config_fixture_dir is not None:
            raise _provider_error(entry, task, f"openai_compatible provider for {task} does not support fixture_dir.")
        if not endpoint:
            raise _provider_error(entry, task, f"openai_compatible provider for {task} requires endpoint.")
        if not api_key:
            raise _provider_error(entry, task, f"openai_compatible provider for {task} requires api_key.")
        validate_endpoint(endpoint, task, entry=entry)
        effective_max_retries = max_retries if max_retries is not None else DEFAULT_OPENAI_COMPATIBLE_MAX_RETRIES
        effective_retry_backoff_seconds = (
            float(retry_backoff_seconds)
            if retry_backoff_seconds is not None
            else DEFAULT_OPENAI_COMPATIBLE_RETRY_BACKOFF_SECONDS
        )
        return (
            ProviderRuntimeSpec(
                spec=spec,
                endpoint=endpoint,
                max_retries=effective_max_retries,
                retry_backoff_seconds=effective_retry_backoff_seconds,
            ),
            api_key,
        )

    if provider_name == "human":
        if separator:
            raise _provider_error(entry, task, f"human provider for {task} must use spec: human.")
        if (
            endpoint is not None
            or api_key is not None
            or config_fixture_dir is not None
            or max_retries is not None
            or retry_backoff_seconds is not None
        ):
            raise _provider_error(
                entry,
                task,
                f"human provider for {task} does not support endpoint, api_key, fixture_dir, max_retries, or retry_backoff_seconds.",
            )
        return ProviderRuntimeSpec(spec=spec), None

    if endpoint is not None or api_key is not None:
        raise _provider_error(entry, task, f"mock provider for {task} does not support endpoint or api_key.")
    if max_retries is not None or retry_backoff_seconds is not None:
        raise _provider_error(entry, task, f"mock provider for {task} does not support max_retries or retry_backoff_seconds.")
    resolved_fixture_dir = None
    if fixture_dir is not None:
        resolved_fixture_dir = fixture_dir.resolve()
    elif config_fixture_dir:
        resolved_fixture_dir = _resolve_config_path(entry.base_dir, config_fixture_dir)
    if resolved_fixture_dir is None and require_mock_fixture:
        raise _provider_error(entry, task, f"mock provider for {task} requires fixture_dir")
    return (
        ProviderRuntimeSpec(
            spec=spec,
            fixture_dir=resolved_fixture_dir.as_posix() if resolved_fixture_dir is not None else None,
        ),
        None,
    )


def validate_endpoint(endpoint: str, task: str, *, entry: ProviderConfigEntry | None = None) -> None:
    parts = urlsplit(endpoint)
    if parts.username or parts.password:
        message = f"Provider endpoint for {task} must not include username or password."
        if entry is not None:
            raise _provider_error(entry, task, message)
        raise ProviderConfigError(message)
    for key, _ in parse_qsl(parts.query, keep_blank_values=True):
        if key.lower() in SECRET_QUERY_KEYS:
            message = f"Provider endpoint for {task} must not include secret query parameter: {key}"
            if entry is not None:
                raise _provider_error(entry, task, message)
            raise ProviderConfigError(message)


def _validate_optional_string(entry: ProviderConfigEntry, task: str, value: Any, message: str) -> None:
    if value is not None and not isinstance(value, str):
        raise _provider_error(entry, task, message)


def _validate_optional_nonnegative_int(entry: ProviderConfigEntry, task: str, value: Any, message: str) -> None:
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise _provider_error(entry, task, message)


def _validate_optional_nonnegative_number(entry: ProviderConfigEntry, task: str, value: Any, message: str) -> None:
    if value is None:
        return
    if not isinstance(value, int | float) or isinstance(value, bool) or value < 0:
        raise _provider_error(entry, task, message)


def _provider_error(entry: ProviderConfigEntry, task: str, message: str) -> ProviderConfigError:
    source = f"{entry.source_label}"
    if entry.source_path is not None:
        source = f"{source} {entry.source_path}"
    return ProviderConfigError(f"{message} (source: {source}, provider key: {entry.provider_key}, task: {task})")


def _resolve_config_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _api_keys_from_entries(entries: dict[str, ProviderConfigEntry]) -> list[str]:
    secrets: list[str] = []
    for entry in entries.values():
        value = entry.value
        if isinstance(value, dict):
            api_key = value.get("api_key")
            if isinstance(api_key, str) and api_key and api_key not in secrets:
                secrets.append(api_key)
    return secrets
