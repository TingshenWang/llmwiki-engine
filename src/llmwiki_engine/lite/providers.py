from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar
from urllib.parse import urlparse

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, Field


MODEL_BACKED_STEPS = ["source_digest", "candidate_pages", "merge_plan", "composition_plan", "final_pages"]
MAX_OUTPUT_TOKENS = 262144

T = TypeVar("T", bound=BaseModel)


class ProviderConfigError(ValueError):
    pass


class ProviderCallError(RuntimeError):
    pass


class ProviderLiveCheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    message: str


class ProviderSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec: str = "unconfigured"
    endpoint: str | None = None
    api_key: str | None = None
    timeout_seconds: float = 300.0
    max_retries: int = 1
    retry_backoff_seconds: float = 1.0
    temperature: float = 0.0
    max_tokens: int | None = MAX_OUTPUT_TOKENS
    json_mode: Literal["json_schema", "json_object"] = "json_schema"
    json_schema_strict: bool = False

    @property
    def kind(self) -> str:
        return self.spec.split(":", 1)[0]

    @property
    def model_name(self) -> str | None:
        if ":" not in self.spec:
            return None
        return self.spec.split(":", 1)[1]

    @property
    def is_openai_compatible(self) -> bool:
        return self.kind == "openai_compatible"

    def resolved_api_key(self) -> str | None:
        return self.api_key.strip() if self.api_key and self.api_key.strip() else None

    def effective_json_mode(self) -> Literal["json_schema", "json_object"]:
        if self.json_mode == "json_schema" and self.endpoint and "api.deepseek.com" in self.endpoint:
            return "json_object"
        return self.json_mode

    def sanitized_context(self) -> dict[str, Any]:
        context = {
            "spec": self.spec,
            "endpoint": self.endpoint,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "retry_backoff_seconds": self.retry_backoff_seconds,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "json_mode": self.json_mode,
            "effective_json_mode": self.effective_json_mode(),
            "json_schema_strict": self.json_schema_strict,
            "has_api_key": bool(self.resolved_api_key()),
        }
        return context


class PromptRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: str
    schema_name: str
    system_prompt: str
    user_payload: dict[str, Any]
    response_schema: dict[str, Any]
    json_output_example: dict[str, Any] = Field(default_factory=dict)


@dataclass
class ProviderCallResult:
    output: BaseModel
    prompt_artifact: dict[str, Any]
    provider_result: dict[str, Any]
    sanitized_context: dict[str, Any]
    model_calls: int = 1


class ProviderRegistry:
    def __init__(self, providers: dict[str, ProviderSpec]):
        self.providers = providers

    def provider_for(self, step: str) -> ProviderSpec:
        return self.providers.get(step) or self.providers.get("default") or ProviderSpec()

    def sanitized_contexts(self, steps: list[str] | None = None) -> dict[str, dict[str, Any]]:
        step_names = steps or MODEL_BACKED_STEPS
        return {step: self.provider_for(step).sanitized_context() for step in step_names}

    def call_structured(self, step: str, request: PromptRequest, output_model: type[T]) -> ProviderCallResult:
        spec = self.provider_for(step)
        if _is_non_model_provider(spec):
            raise ProviderConfigError(f"步骤 {step} 没有配置真实模型 provider，不能发起模型请求。")
        if spec.is_openai_compatible:
            return self._call_openai_compatible(step, request, output_model, spec)
        raise ProviderConfigError(f"步骤 {step} 使用了不支持的 provider spec：{spec.spec}")

    def check(self, *, live: bool = False) -> list[dict[str, Any]]:
        reports = []
        for name, spec in sorted(self.providers.items()):
            report = {"name": name, "ok": True, "context": spec.sanitized_context(), "message": "ok"}
            issue = _real_provider_issue(spec)
            if issue is not None:
                report.update({"ok": False, "message": issue})
            elif live:
                try:
                    result = self._call_live_check(name, spec)
                except (ProviderConfigError, ProviderCallError, ValueError) as exc:
                    report.update({"ok": False, "message": f"live check failed: {exc}"})
                else:
                    report.update(
                        {
                            "message": "live ok",
                            "context": {
                                **spec.sanitized_context(),
                                "live": True,
                                "live_model_calls": result.model_calls,
                                "live_max_tokens": _live_check_max_tokens(spec),
                            },
                            "model_calls": result.model_calls,
                            "provider_message": result.output.message,
                        }
                    )
            reports.append(report)
        return reports

    def require_real_model_providers(self, steps: list[str] | None = None) -> None:
        step_names = steps or MODEL_BACKED_STEPS
        issues = []
        for step in step_names:
            spec = self.provider_for(step)
            issue = _real_provider_issue(spec)
            if issue is not None:
                issues.append(f"{step}: {issue} ({spec.spec})")
        if issues:
            raise ProviderConfigError(
                "Lite Ingest 必须使用真实模型 provider，请在 ~/.llmwiki/config.yaml 或 vault/.llmwiki/config.yaml 配置 api_key 和完整 chat completions endpoint。"
                + " 问题："
                + "；".join(issues)
            )

    def _call_openai_compatible(self, step: str, request: PromptRequest, output_model: type[T], spec: ProviderSpec) -> ProviderCallResult:
        if not spec.endpoint:
            raise ProviderConfigError(f"{step} 的 openai_compatible provider 缺少 endpoint。")
        if not _is_chat_completions_endpoint(spec.endpoint):
            raise ProviderConfigError(f"{step} 的 endpoint 必须是完整 chat completions URL，例如 https://api.deepseek.com/v1/chat/completions。")
        model = spec.model_name
        if not model:
            raise ProviderConfigError(f"{step} 的 openai_compatible provider spec 缺少 model。")
        headers = {"Content-Type": "application/json"}
        api_key = spec.resolved_api_key()
        if not api_key:
            raise ProviderConfigError(f"{step} 的 openai_compatible provider 缺少 api_key。")
        headers["Authorization"] = f"Bearer {api_key}"
        last_error: Exception | None = None
        calls_made = 0
        retry_reason: str | None = None
        for attempt in range(spec.max_retries + 1):
            attempt_spec = _spec_for_retry_attempt(spec, attempt, retry_reason)
            payload = build_chat_payload(attempt_spec, request)
            current_retry_reason: str | None = None
            try:
                with httpx.Client(timeout=spec.timeout_seconds) as client:
                    response = client.post(spec.endpoint, headers=headers, json=payload)
                    calls_made += 1
                    fallback_used = False
                    if _should_fallback_to_json_object(response, attempt_spec):
                        fallback_spec = attempt_spec.model_copy(update={"json_mode": "json_object"})
                        fallback_payload = build_chat_payload(fallback_spec, request)
                        response = client.post(spec.endpoint, headers=headers, json=fallback_payload)
                        calls_made += 1
                        fallback_used = True
                response.raise_for_status()
                raw_response = response.json()
                if _response_was_truncated(raw_response):
                    current_retry_reason = "truncated_response"
                    raise ProviderCallError("chat completion 响应被 max_tokens 截断。")
                parsed = _parse_chat_completion_json(raw_response)
                output = output_model.model_validate(parsed)
                provider_context = attempt_spec.sanitized_context() | {"retry_count": attempt}
                if retry_reason:
                    provider_context["retry_reason"] = retry_reason
                if fallback_used:
                    provider_context = {**provider_context, "response_format_fallback": "json_object"}
                return ProviderCallResult(
                    output=output,
                    prompt_artifact=_prompt_artifact(request, attempt_spec),
                    provider_result={
                        "provider": provider_context,
                        "raw_response": _compact_response(raw_response),
                        "parsed": output.model_dump(mode="json"),
                    },
                    sanitized_context=provider_context,
                    model_calls=calls_made,
                )
            except Exception as exc:  # noqa: BLE001 - retry surface should preserve provider failure text.
                last_error = exc
                retry_reason = current_retry_reason or "provider_or_schema_error"
                if attempt < spec.max_retries:
                    time.sleep(spec.retry_backoff_seconds * (attempt + 1))
        raise ProviderCallError(f"{step} provider 调用失败：{last_error}") from last_error

    def _call_live_check(self, name: str, spec: ProviderSpec) -> ProviderCallResult:
        smoke_spec = spec.model_copy(
            update={
                "max_tokens": _live_check_max_tokens(spec),
                "max_retries": 0,
                "retry_backoff_seconds": 0,
            }
        )
        request = PromptRequest(
            step="provider_live_check",
            schema_name="llmwiki_lite_provider_live_check",
            system_prompt=(
                "你是 llmwiki-engine Lite 的模型服务连通性检查。"
                "只返回符合 schema 的 JSON object，不要输出 Markdown。"
            ),
            user_payload={
                "provider_name": name,
                "instruction": "请返回 ok=true，并用中文简短说明模型服务可用。",
            },
            response_schema=ProviderLiveCheckResult.model_json_schema(),
            json_output_example={"ok": True, "message": "模型服务可用。"},
        )
        if spec.is_openai_compatible:
            result = self._call_openai_compatible("provider_live_check", request, ProviderLiveCheckResult, smoke_spec)
        else:
            raise ProviderConfigError(f"不支持的 provider spec：{spec.spec}")
        output = result.output
        if not isinstance(output, ProviderLiveCheckResult) or not output.ok:
            message = output.message if isinstance(output, ProviderLiveCheckResult) else "模型服务返回了无效 live check 结果。"
            raise ProviderCallError(message)
        return result


def load_provider_registry(vault: Path) -> ProviderRegistry:
    merged: dict[str, Any] = {"default": {"spec": "unconfigured"}}
    for path in [Path.home() / ".llmwiki" / "config.yaml", vault.expanduser().resolve() / ".llmwiki" / "config.yaml"]:
        if not path.exists():
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        providers = data.get("providers", {}) if isinstance(data, dict) else {}
        if not isinstance(providers, dict):
            raise ProviderConfigError(f"{path} 中的 providers 必须是 mapping。")
        for name, raw_spec in providers.items():
            if isinstance(raw_spec, dict):
                merged[str(name)] = raw_spec
            else:
                raise ProviderConfigError(f"{path} 中的 provider {name} 必须是 mapping。")
    return ProviderRegistry({name: ProviderSpec.model_validate(value) for name, value in merged.items()})


def _real_provider_issue(spec: ProviderSpec) -> str | None:
    if _is_non_model_provider(spec):
        return "not a real model provider"
    if not spec.is_openai_compatible:
        return "unsupported provider spec"
    if not spec.endpoint:
        return "missing endpoint"
    if not _is_chat_completions_endpoint(spec.endpoint):
        return "endpoint must be chat completions URL"
    if not spec.model_name:
        return "missing model"
    if not spec.resolved_api_key():
        return "missing API key"
    return None


def _is_non_model_provider(spec: ProviderSpec) -> bool:
    return spec.spec in {"unconfigured", "local:heuristic"}


def _live_check_max_tokens(spec: ProviderSpec) -> int:
    if spec.max_tokens is None:
        return 128
    return max(1, min(spec.max_tokens, 128))


def _is_chat_completions_endpoint(endpoint: str) -> bool:
    path = urlparse(endpoint).path.rstrip("/")
    return path.endswith("/chat/completions")


def build_chat_payload(spec: ProviderSpec, request: PromptRequest) -> dict[str, Any]:
    user_content = json.dumps(
        {
            "task": request.step,
            "input": request.user_payload,
            "response_schema": request.response_schema,
            "json_output_example": request.json_output_example,
        },
        ensure_ascii=False,
    )
    payload: dict[str, Any] = {
        "model": spec.model_name,
        "messages": [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": spec.temperature,
    }
    if spec.max_tokens is not None:
        payload["max_tokens"] = spec.max_tokens
    if spec.effective_json_mode() == "json_schema":
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": request.schema_name,
                "schema": request.response_schema,
                "strict": spec.json_schema_strict,
            },
        }
    else:
        payload["response_format"] = {"type": "json_object"}
    return payload


def _prompt_artifact(request: PromptRequest, spec: ProviderSpec) -> dict[str, Any]:
    return {
        "step": request.step,
        "schema_name": request.schema_name,
        "system_prompt": request.system_prompt,
        "user_payload": request.user_payload,
        "response_schema": request.response_schema,
        "provider": spec.sanitized_context(),
    }


def _parse_chat_completion_json(raw_response: dict[str, Any]) -> Any:
    try:
        content = raw_response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderCallError("chat completion 响应缺少 choices[0].message.content。") from exc
    if isinstance(content, list):
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    if not isinstance(content, str):
        raise ProviderCallError("chat completion content 不是文本。")
    return _extract_json_object(content)


def _should_fallback_to_json_object(response: httpx.Response, spec: ProviderSpec) -> bool:
    if spec.json_mode != "json_schema" or response.status_code != 400:
        return False
    text = response.text.lower()
    return "response_format" in text and ("unavailable" in text or "not support" in text or "unsupported" in text)


def _response_was_truncated(raw_response: dict[str, Any]) -> bool:
    choices = raw_response.get("choices")
    if not isinstance(choices, list):
        return False
    for choice in choices:
        if isinstance(choice, dict) and str(choice.get("finish_reason", "")).lower() in {"length", "max_tokens"}:
            return True
    return False


def _spec_for_retry_attempt(spec: ProviderSpec, attempt: int, retry_reason: str | None) -> ProviderSpec:
    if attempt <= 0 or retry_reason != "truncated_response" or spec.max_tokens is None:
        return spec
    retry_tokens = max(spec.max_tokens + 1, spec.max_tokens * (2**attempt))
    return spec.model_copy(update={"max_tokens": min(MAX_OUTPUT_TOKENS, retry_tokens)})


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    decoder = json.JSONDecoder()
    last_error: json.JSONDecodeError | None = None
    for match in re.finditer(r"\{", stripped):
        try:
            parsed, _ = decoder.raw_decode(stripped[match.start() :])
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(parsed, dict):
            return parsed
    if last_error is not None:
        raise ProviderCallError(f"模型响应没有可解析的 JSON object：{last_error}") from last_error
    raise ProviderCallError("模型响应没有 JSON object。")


def _compact_response(raw_response: dict[str, Any]) -> dict[str, Any]:
    compact = dict(raw_response)
    if "choices" in compact:
        compact["choices"] = [
            {
                "index": choice.get("index"),
                "finish_reason": choice.get("finish_reason"),
                "message": {"role": choice.get("message", {}).get("role"), "content": choice.get("message", {}).get("content")},
            }
            for choice in compact.get("choices", [])
            if isinstance(choice, dict)
        ]
    return compact
