from __future__ import annotations

import json
import queue
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, TypeVar
from urllib.parse import urlparse

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, Field

from .token_usage import api_call_record


MODEL_BACKED_STEPS = [
    "source_digest",
    "digest_coverage_judge",
    "candidate_pages_warmup",
    "candidate_pages",
    "merge_plan",
    "composition_plan",
    "final_pages",
    "final_coverage_judge",
]
MAX_OUTPUT_TOKENS = 262144

T = TypeVar("T", bound=BaseModel)


class ProviderConfigError(ValueError):
    pass


class ProviderCallError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        api_calls: list[dict[str, Any]] | None = None,
        prompt_artifact: dict[str, Any] | None = None,
        provider_result: dict[str, Any] | None = None,
        sanitized_context: dict[str, Any] | None = None,
        model_calls: int = 0,
    ) -> None:
        super().__init__(message)
        self.api_calls = api_calls or []
        self.prompt_artifact = prompt_artifact or {}
        self.provider_result = provider_result or {}
        self.sanitized_context = sanitized_context or {}
        self.model_calls = model_calls


class ProviderLiveCheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool


class ProviderSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec: str = "unconfigured"
    endpoint: str | None = None
    api_key: str | None = None
    timeout_seconds: float = 180.0
    max_retries: int = 1
    retry_backoff_seconds: float = 1.0
    temperature: float = 0.0
    max_tokens: int | None = MAX_OUTPUT_TOKENS
    json_mode: Literal["json_schema", "json_object"] = "json_schema"
    json_schema_strict: bool = False
    stream_reasoning: bool = False

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
            "stream_reasoning": self.stream_reasoning,
            "has_api_key": bool(self.resolved_api_key()),
        }
        return context


class PromptRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: str
    schema_name: str
    system_prompt: str
    cache_prefix_payload: dict[str, Any] | None = None
    user_payload: dict[str, Any]
    response_schema: dict[str, Any]
    json_output_example: dict[str, Any] = Field(default_factory=dict)


@dataclass
class ProviderCallResult:
    output: BaseModel
    prompt_artifact: dict[str, Any]
    provider_result: dict[str, Any]
    sanitized_context: dict[str, Any]
    api_calls: list[dict[str, Any]]
    model_calls: int = 1
    reasoning_content: str | None = None


class ProviderRegistry:
    def __init__(self, providers: dict[str, ProviderSpec]):
        self.providers = providers

    def provider_for(self, step: str) -> ProviderSpec:
        return self.providers.get(step) or self.providers.get("default") or ProviderSpec()

    def sanitized_contexts(self, steps: list[str] | None = None) -> dict[str, dict[str, Any]]:
        step_names = steps or MODEL_BACKED_STEPS
        return {step: self.provider_for(step).sanitized_context() for step in step_names}

    def call_structured(self, step: str, request: PromptRequest, output_model: type[T], *, reasoning_callback: Callable[[str], None] | None = None) -> ProviderCallResult:
        spec = self.provider_for(step)
        if _is_non_model_provider(spec):
            raise ProviderConfigError(f"步骤 {step} 没有配置真实模型 provider，不能发起模型请求。")
        if spec.is_openai_compatible:
            return self._call_openai_compatible(step, request, output_model, spec, reasoning_callback=reasoning_callback)
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
                            "provider_message": "OK",
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

    def _call_openai_compatible(self, step: str, request: PromptRequest, output_model: type[T], spec: ProviderSpec, *, reasoning_callback: Callable[[str], None] | None = None) -> ProviderCallResult:
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
        api_calls: list[dict[str, Any]] = []
        retry_reason: str | None = None
        last_attempt_spec = spec
        use_streaming = reasoning_callback is not None or spec.stream_reasoning

        def post_once(client: httpx.Client, payload: dict[str, Any], attempt: int) -> tuple[httpx.Response, dict[str, Any], float]:
            nonlocal calls_made
            calls_made += 1
            call_index = calls_made
            started = time.perf_counter()
            try:
                response = _post_with_wall_timeout(client, spec.endpoint, headers, payload, spec.timeout_seconds)
            except Exception as exc:  # noqa: BLE001 - record provider failures before retrying.
                duration_ms = (time.perf_counter() - started) * 1000
                api_calls.append(
                    api_call_record(
                        step=step,
                        model=model,
                        attempt=attempt,
                        call_index=call_index,
                        status="paused",
                        duration_ms=duration_ms,
                        error=_provider_exception_message(exc),
                    )
                )
                raise
            duration_ms = (time.perf_counter() - started) * 1000
            response_record = api_call_record(
                step=step,
                model=model,
                attempt=attempt,
                call_index=call_index,
                status="paused",
                duration_ms=duration_ms,
                response_status_code=response.status_code,
            )
            api_calls.append(response_record)
            return response, response_record, duration_ms

        def stream_once(client: httpx.Client, payload: dict[str, Any], attempt: int) -> tuple[dict[str, Any] | None, dict[str, Any], float, int, str]:
            nonlocal calls_made
            calls_made += 1
            call_index = calls_made
            started = time.perf_counter()
            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            finish_reason: str | None = None
            usage: dict[str, Any] | None = None
            status_code = 200
            error_text = ""
            try:
                with client.stream("POST", spec.endpoint or "", headers=headers, json=payload) as response:  # type: ignore[arg-type]
                    status_code = response.status_code
                    if status_code != 200:
                        error_text = response.read().decode("utf-8", errors="replace")
                        duration_ms = (time.perf_counter() - started) * 1000
                        record = api_call_record(
                            step=step,
                            model=model,
                            attempt=attempt,
                            call_index=call_index,
                            status="paused",
                            duration_ms=duration_ms,
                            response_status_code=status_code,
                        )
                        api_calls.append(record)
                        return None, record, duration_ms, status_code, error_text
                    for line in response.iter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        choices = chunk.get("choices") or []
                        if choices and isinstance(choices[0], dict):
                            delta = choices[0].get("delta") or {}
                            rc = delta.get("reasoning_content")
                            if rc:
                                reasoning_parts.append(rc)
                                if reasoning_callback:
                                    reasoning_callback(rc)
                            ct = delta.get("content")
                            if ct:
                                content_parts.append(ct)
                            fr = choices[0].get("finish_reason")
                            if fr:
                                finish_reason = fr
                        if chunk.get("usage"):
                            usage = chunk["usage"]
            except Exception as exc:  # noqa: BLE001 - record provider failures before retrying.
                duration_ms = (time.perf_counter() - started) * 1000
                api_calls.append(
                    api_call_record(
                        step=step,
                        model=model,
                        attempt=attempt,
                        call_index=call_index,
                        status="paused",
                        duration_ms=duration_ms,
                        error=_provider_exception_message(exc),
                    )
                )
                raise
            duration_ms = (time.perf_counter() - started) * 1000
            raw_response: dict[str, Any] = {
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": finish_reason or "stop",
                        "message": {
                            "role": "assistant",
                            "content": "".join(content_parts),
                        },
                    }
                ],
                "usage": usage or {},
            }
            reasoning_text = "".join(reasoning_parts)
            if reasoning_text:
                raw_response["choices"][0]["message"]["reasoning_content"] = reasoning_text
            record = api_call_record(
                step=step,
                model=model,
                attempt=attempt,
                call_index=call_index,
                status="paused",
                duration_ms=duration_ms,
                response_status_code=status_code,
            )
            api_calls.append(record)
            return raw_response, record, duration_ms, status_code, error_text

        for attempt in range(spec.max_retries + 1):
            attempt_spec = _spec_for_retry_attempt(spec, attempt, retry_reason)
            last_attempt_spec = attempt_spec
            payload = build_chat_payload(attempt_spec, request, stream=use_streaming)
            current_retry_reason: str | None = None
            try:
                with httpx.Client(timeout=spec.timeout_seconds) as client:
                    fallback_used = False
                    if use_streaming:
                        stream_raw, response_record, duration_ms, status_code, error_text = stream_once(client, payload, attempt)
                        raw_response: dict[str, Any] | None = stream_raw
                        if status_code == 400 and _is_json_schema_unavailable_text(error_text, attempt_spec):
                            response_record["error"] = "json_schema_response_format_fallback"
                            fallback_spec = attempt_spec.model_copy(update={"json_mode": "json_object"})
                            fallback_payload = build_chat_payload(fallback_spec, request)
                            response, response_record, duration_ms = post_once(client, fallback_payload, attempt)
                            fallback_used = True
                            response.raise_for_status()
                            raw_response = response.json()
                            status_code = response.status_code
                        elif raw_response is None:
                            raise httpx.HTTPStatusError(
                                f"streaming 请求失败：{status_code}",
                                request=httpx.Request("POST", spec.endpoint or ""),
                                response=httpx.Response(status_code, text=error_text),
                            )
                    else:
                        response, response_record, duration_ms = post_once(client, payload, attempt)
                        status_code = response.status_code
                        if _should_fallback_to_json_object(response, attempt_spec):
                            response_record["error"] = "json_schema_response_format_fallback"
                            fallback_spec = attempt_spec.model_copy(update={"json_mode": "json_object"})
                            fallback_payload = build_chat_payload(fallback_spec, request)
                            response, response_record, duration_ms = post_once(client, fallback_payload, attempt)
                            fallback_used = True
                            status_code = response.status_code
                        response.raise_for_status()
                        raw_response = response.json()
                response_record.update(
                    api_call_record(
                        step=step,
                        model=model,
                        attempt=attempt,
                        call_index=calls_made,
                        status="paused",
                        duration_ms=duration_ms,
                        raw_response=raw_response,
                        response_status_code=status_code,
                    )
                )
                if _response_was_truncated(raw_response):
                    current_retry_reason = "truncated_response"
                    response_record["error"] = current_retry_reason
                    raise ProviderCallError("chat completion 响应被 max_tokens 截断。")
                parsed = _parse_chat_completion_json(raw_response)
                output = output_model.model_validate(parsed)
                response_record["status"] = "success"
                provider_context = attempt_spec.sanitized_context() | {"retry_count": attempt}
                if retry_reason:
                    provider_context["retry_reason"] = retry_reason
                if fallback_used:
                    provider_context = {**provider_context, "response_format_fallback": "json_object"}
                reasoning_content = _extract_reasoning_content(raw_response)
                return ProviderCallResult(
                    output=output,
                    prompt_artifact=_prompt_artifact(request, attempt_spec),
                    provider_result={
                        "provider": provider_context,
                        "raw_response": _compact_response(raw_response),
                        "api_calls": api_calls,
                        "parsed": output.model_dump(mode="json"),
                        "reasoning_content": reasoning_content,
                    },
                    sanitized_context=provider_context,
                    api_calls=api_calls,
                    model_calls=calls_made,
                    reasoning_content=reasoning_content,
                )
            except Exception as exc:  # noqa: BLE001 - retry surface should preserve provider failure text.
                if api_calls and api_calls[-1].get("status") != "success" and not api_calls[-1].get("error"):
                    api_calls[-1]["error"] = _provider_exception_message(exc)
                last_error = exc
                retry_reason = current_retry_reason or _provider_retry_reason(exc)
                if attempt < spec.max_retries:
                    time.sleep(spec.retry_backoff_seconds * (attempt + 1))
        failure_context = last_attempt_spec.sanitized_context() | {"retry_count": spec.max_retries}
        if retry_reason:
            failure_context["retry_reason"] = retry_reason
        failure_result = {
            "provider": failure_context,
            "api_calls": api_calls,
            "error": _provider_exception_message(last_error) if last_error else "unknown provider error",
        }
        raise ProviderCallError(
            f"{step} provider 调用失败：{last_error}",
            api_calls=api_calls,
            prompt_artifact=_prompt_artifact(request, last_attempt_spec),
            provider_result=failure_result,
            sanitized_context=failure_context,
            model_calls=calls_made,
        ) from last_error

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
                "只返回符合 schema 的 JSON object，不要输出 Markdown，不要解释。"
            ),
            user_payload={
                "provider_name": name,
                "instruction": "只返回 ok=true。",
            },
            response_schema=ProviderLiveCheckResult.model_json_schema(),
            json_output_example={"ok": True},
        )
        if spec.is_openai_compatible:
            result = self._call_openai_compatible("provider_live_check", request, ProviderLiveCheckResult, smoke_spec)
        else:
            raise ProviderConfigError(f"不支持的 provider spec：{spec.spec}")
        output = result.output
        if not isinstance(output, ProviderLiveCheckResult) or not output.ok:
            raise ProviderCallError("模型服务返回了无效 live check 结果。")
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


def _post_with_wall_timeout(client: httpx.Client, endpoint: str, headers: dict[str, str], payload: dict[str, Any], timeout_seconds: float) -> httpx.Response:
    if timeout_seconds <= 0:
        return client.post(endpoint, headers=headers, json=payload)

    result_queue: queue.Queue[tuple[bool, httpx.Response | BaseException]] = queue.Queue(maxsize=1)

    def run_post() -> None:
        try:
            result_queue.put((True, client.post(endpoint, headers=headers, json=payload)))
        except BaseException as exc:  # noqa: BLE001 - preserve provider exception type for retry classification.
            result_queue.put((False, exc))

    thread = threading.Thread(target=run_post, daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        raise httpx.TimeoutException(f"provider call exceeded timeout_seconds={timeout_seconds:g}")
    ok, value = result_queue.get_nowait()
    if ok:
        return value  # type: ignore[return-value]
    raise value


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
        return 512
    return max(1, min(spec.max_tokens, 512))


def _is_chat_completions_endpoint(endpoint: str) -> bool:
    path = urlparse(endpoint).path.rstrip("/")
    return path.endswith("/chat/completions")


def build_chat_payload(spec: ProviderSpec, request: PromptRequest, *, stream: bool = False) -> dict[str, Any]:
    messages = [{"role": "system", "content": request.system_prompt}]
    suffix_content = json.dumps(
        {
            "task": request.step,
            "input": request.user_payload,
            "response_schema": request.response_schema,
            "json_output_example": request.json_output_example,
        },
        ensure_ascii=False,
    )
    if request.cache_prefix_payload is not None:
        messages.append({"role": "user", "content": json.dumps({"cache_prefix": request.cache_prefix_payload}, ensure_ascii=False)})
        messages.append({"role": "user", "content": suffix_content})
    else:
        messages.append({"role": "user", "content": suffix_content})
    payload: dict[str, Any] = {
        "model": spec.model_name,
        "messages": messages,
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
    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    return payload


def _prompt_artifact(request: PromptRequest, spec: ProviderSpec) -> dict[str, Any]:
    artifact = {
        "step": request.step,
        "schema_name": request.schema_name,
        "system_prompt": request.system_prompt,
        "user_payload": request.user_payload,
        "response_schema": request.response_schema,
        "provider": spec.sanitized_context(),
    }
    if request.cache_prefix_payload is not None:
        artifact["cache_prefix_payload"] = request.cache_prefix_payload
    return artifact


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


def _provider_retry_reason(exc: Exception) -> str:
    if isinstance(exc, httpx.TimeoutException):
        return "provider_timeout"
    if isinstance(exc, httpx.HTTPStatusError):
        return "provider_http_error"
    if isinstance(exc, httpx.TransportError):
        return "provider_transport_error"
    return "provider_or_schema_error"


def _provider_exception_message(exc: Exception | None) -> str:
    if exc is None:
        return ""
    reason = _provider_retry_reason(exc)
    message = str(exc).strip()
    if message:
        return f"{reason}: {message}"
    return reason


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
                "message": {
                    "role": choice.get("message", {}).get("role"),
                    "content": choice.get("message", {}).get("content"),
                    "reasoning_content": choice.get("message", {}).get("reasoning_content"),
                },
            }
            for choice in compact.get("choices", [])
            if isinstance(choice, dict)
        ]
    return compact


def _extract_reasoning_content(raw_response: dict[str, Any]) -> str | None:
    try:
        rc = raw_response["choices"][0]["message"]["reasoning_content"]
        if isinstance(rc, str) and rc:
            return rc
    except (KeyError, IndexError, TypeError):
        pass
    return None


def _is_json_schema_unavailable_text(text: str, spec: ProviderSpec) -> bool:
    if spec.json_mode != "json_schema":
        return False
    lowered = text.lower()
    return "response_format" in lowered and ("unavailable" in lowered or "not support" in lowered or "unsupported" in lowered)
