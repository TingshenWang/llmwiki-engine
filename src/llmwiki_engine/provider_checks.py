from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import httpx

from .provider_config import ProviderConfigError, build_provider_execution_context
from .providers import ProviderError
from .steps import MODEL_BACKED_STEPS
from .workspace import WorkspaceError, llmwiki_git_state


@dataclass
class ProviderCheckRow:
    task: str
    spec: str
    endpoint: str | None = None
    fixture_dir: str | None = None
    credential_label: str | None = None


@dataclass
class ProviderCheckResult:
    ok: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    rows: list[ProviderCheckRow] = field(default_factory=list)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def error(self, message: str) -> None:
        self.ok = False
        self.errors.append(message)


HttpClientFactory = Callable[[], httpx.Client]


def check_providers(
    vault: Path,
    *,
    live: bool = False,
    http_client_factory: HttpClientFactory | None = None,
) -> ProviderCheckResult:
    result = ProviderCheckResult()
    try:
        execution_context = build_provider_execution_context(
            vault=vault,
            manifest_contexts=[],
            source="initial_run",
            from_step=None,
            tasks=list(MODEL_BACKED_STEPS),
            require_mock_fixture=False,
        )
    except ProviderConfigError as exc:
        result.error(str(exc))
        return result

    try:
        state = llmwiki_git_state(vault)
    except WorkspaceError as exc:
        result.error(str(exc))
        return result
    if not state.is_git_repo:
        result.warn("vault is not a Git repository; skipped .llmwiki Git tracked/staged check.")
    elif state.tracked or state.staged:
        paths = ", ".join(sorted(set(state.tracked + state.staged)))
        result.error(f".llmwiki/ must not be tracked or staged by Git: {paths}")

    record = execution_context.record
    if record is None:
        return result

    credential_labels: dict[tuple[str, str | None, str | None], str] = {}
    credential_count = 0
    for task in MODEL_BACKED_STEPS:
        runtime = record.providers[task]
        credential = execution_context.credentials_by_task.get(task)
        credential_label = None
        if credential is not None:
            key = (runtime.spec, runtime.endpoint, credential)
            if key not in credential_labels:
                credential_count += 1
                credential_labels[key] = f"credential #{credential_count}"
            credential_label = credential_labels[key]
        result.rows.append(
            ProviderCheckRow(
                task=task,
                spec=runtime.spec,
                endpoint=runtime.endpoint,
                fixture_dir=runtime.fixture_dir,
                credential_label=credential_label,
            )
        )

        provider_name = runtime.spec.partition(":")[0]
        if provider_name == "mock":
            if runtime.fixture_dir is None:
                result.warn(
                    f"mock provider for {task} has no fixture_dir; configure fixture_dir or use "
                    "--mock-fixture-dir to force all model-backed steps to mock."
                )
            elif not Path(runtime.fixture_dir).is_dir():
                result.warn(f"mock fixture_dir for {task} does not exist: {runtime.fixture_dir}")
    if live and result.ok:
        _run_live_checks(result, execution_context, credential_labels, http_client_factory)
    return result


def _run_live_checks(
    result: ProviderCheckResult,
    execution_context,
    credential_labels: dict[tuple[str, str | None, str | None], str],
    http_client_factory: HttpClientFactory | None,
) -> None:
    record = execution_context.record
    if record is None:
        return
    steps_by_key: dict[tuple[str, str | None, str | None], list[str]] = {}
    for task, runtime in record.providers.items():
        credential = execution_context.credentials_by_task.get(task)
        key = (runtime.spec, runtime.endpoint, credential)
        steps_by_key.setdefault(key, []).append(task)

    for key, steps in steps_by_key.items():
        spec, endpoint, credential = key
        provider_name = spec.partition(":")[0]
        if provider_name != "openai_compatible":
            continue
        label = credential_labels.get(key, "credential #1")
        client = http_client_factory() if http_client_factory is not None else None
        try:
            provider = execution_context.provider_for_task(steps[0], http_client=client)
            check_live = getattr(provider, "check_live", None)
            if check_live is None:
                result.error(f"{spec} {label} live check is not supported for steps {', '.join(steps)}.")
                continue
            fallback_used = False
            try:
                raw = check_live()
            except ProviderError as exc:
                if not _is_json_mode_unsupported(exc):
                    raise
                raw = check_live(use_json_mode=False)
                fallback_used = True
            _validate_live_probe_content(raw)
            if fallback_used:
                result.warn(
                    f"{spec} {label} does not support JSON mode; "
                    f"used prompt-only JSON live check for steps {', '.join(steps)}."
                )
        except (ProviderError, httpx.HTTPError) as exc:
            message = execution_context.redactor.redact_text(str(exc))
            result.error(f"{spec} {label} live check failed for steps {', '.join(steps)}: {message}")
        finally:
            if client is not None:
                client.close()


def _is_json_mode_unsupported(exc: ProviderError) -> bool:
    if exc.status_code not in {400, 422}:
        return False
    message = str(exc).lower()
    json_mode_terms = ("response_format", "json_object", "json mode")
    unsupported_terms = (
        "unsupported",
        "not support",
        "does not support",
        "unrecognized",
        "unknown parameter",
        "invalid parameter",
        "not allowed",
    )
    return any(term in message for term in json_mode_terms) and any(term in message for term in unsupported_terms)


def _validate_live_probe_content(raw: str) -> None:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderError(f"live check returned invalid JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ProviderError("live check JSON root must be an object.")
    if data.get("ok") is not True:
        raise ProviderError("live check JSON must contain ok: true.")
