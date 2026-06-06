from __future__ import annotations

import json
import re
from email.utils import parsedate_to_datetime
from random import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import httpx
from pydantic import BaseModel


DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS = 300.0
DEFAULT_OPENAI_COMPATIBLE_MAX_RETRIES = 2
DEFAULT_OPENAI_COMPATIBLE_RETRY_BACKOFF_SECONDS = 1.0
OPENAI_COMPATIBLE_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
OPENAI_COMPATIBLE_MAX_RETRY_DELAY_SECONDS = 30.0
RETRY_ATTEMPT_SUFFIX_RE = re.compile(r" \(after \d+ attempts?\)$")


class ProviderError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None, attempt_count: int = 1):
        super().__init__(message)
        self.status_code = status_code
        self.attempt_count = max(1, int(attempt_count))


class Provider(Protocol):
    name: str

    def generate_raw(self, task: str, payload: dict[str, Any], output_model: type[BaseModel]) -> str:
        ...


class MockProvider:
    name = "mock"

    def __init__(self, fixture_dir: Path):
        self.fixture_dir = fixture_dir
        self.call_counts: dict[str, int] = {}

    def generate_raw(self, task: str, payload: dict[str, Any], output_model: type[BaseModel]) -> str:
        self.call_counts[task] = self.call_counts.get(task, 0) + 1
        sequence_path = self.fixture_dir / f"{task}.sequence.jsonl"
        if sequence_path.exists():
            rows = [line for line in sequence_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            index = min(self.call_counts[task], len(rows)) - 1
            item = json.loads(rows[index])
            if isinstance(item, dict):
                if "raw" in item:
                    return str(item["raw"])
                if "file" in item:
                    return (self.fixture_dir / str(item["file"])).read_text(encoding="utf-8")
            return json.dumps(item, ensure_ascii=False)
        numbered_path = self.fixture_dir / f"{task}.{self.call_counts[task]}.json"
        if numbered_path.exists():
            return numbered_path.read_text(encoding="utf-8")
        numbered_raw_path = self.fixture_dir / f"{task}.{self.call_counts[task]}.raw"
        if numbered_raw_path.exists():
            return numbered_raw_path.read_text(encoding="utf-8")
        if isinstance(payload, dict) and payload.get("repair_contract"):
            repair_path = self.fixture_dir / f"{task}.repair.json"
            if repair_path.exists():
                return repair_path.read_text(encoding="utf-8")
        path = self.fixture_dir / f"{task}.json"
        if not path.exists():
            raise ProviderError(f"Mock fixture missing: {path}")
        return path.read_text(encoding="utf-8")


class OpenAICompatibleProvider:
    name = "openai_compatible"

    def __init__(
        self,
        model: str,
        endpoint: str,
        api_key: str,
        *,
        timeout: float = DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_OPENAI_COMPATIBLE_MAX_RETRIES,
        retry_backoff_seconds: float = DEFAULT_OPENAI_COMPATIBLE_RETRY_BACKOFF_SECONDS,
        http_client: httpx.Client | None = None,
    ):
        self.model = model
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self.http_client = http_client

    def generate_raw(self, task: str, payload: dict[str, Any], output_model: type[BaseModel]) -> str:
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return only one complete valid JSON object matching the requested schema. "
                        "Do not use markdown. Arrays must contain JSON objects with braces and commas. "
                        "Include every required field."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "task": task,
                            "payload": payload,
                            "schema": output_model.model_json_schema(),
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        try:
            return _extract_openai_compatible_content(self._post_chat(body))
        except ProviderError as exc:
            if not _is_json_mode_unsupported(exc):
                raise
            body_without_json_mode = dict(body)
            body_without_json_mode.pop("response_format", None)
            consumed_retry_budget = max(0, exc.attempt_count - 1)
            fallback_retries = max(0, self.max_retries - consumed_retry_budget)
            try:
                return _extract_openai_compatible_content(
                    self._post_chat(body_without_json_mode, max_retries=fallback_retries)
                )
            except ProviderError as fallback_exc:
                combined_attempts = exc.attempt_count + fallback_exc.attempt_count
                message = _retry_exhausted_message(_strip_retry_attempt_suffix(str(fallback_exc)), combined_attempts)
                raise ProviderError(
                    message,
                    status_code=fallback_exc.status_code,
                    attempt_count=combined_attempts,
                ) from fallback_exc

    def check_live(self, *, use_json_mode: bool = True) -> str:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Return only JSON."},
                {"role": "user", "content": 'Return exactly {"ok": true}'},
            ],
            "temperature": 0,
            "max_tokens": 512,
        }
        if use_json_mode:
            body["response_format"] = {"type": "json_object"}
        return _extract_openai_compatible_content(self._post_chat(body, timeout=20.0, max_retries=0))

    def _post_chat(
        self,
        body: dict[str, Any],
        *,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        request_timeout = timeout if timeout is not None else self.timeout
        retry_count = self.max_retries if max_retries is None else max(0, int(max_retries))
        total_attempts = retry_count + 1
        for attempt in range(1, total_attempts + 1):
            try:
                if self.http_client is not None:
                    response = self.http_client.post(self.endpoint, json=body, headers=headers, timeout=request_timeout)
                else:
                    with httpx.Client(timeout=request_timeout) as client:
                        response = client.post(self.endpoint, json=body, headers=headers)
                response.raise_for_status()
                break
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                retryable = status_code in OPENAI_COMPATIBLE_RETRYABLE_STATUS_CODES
                if retryable and attempt < total_attempts:
                    self._sleep_before_retry(attempt, response=exc.response)
                    continue
                message = str(exc)
                if exc.response.text:
                    message = f"HTTP {status_code}: {exc.response.text}"
                raise ProviderError(
                    _retry_exhausted_message(message, attempt if retryable else 1),
                    status_code=status_code,
                    attempt_count=attempt,
                ) from exc
            except httpx.InvalidURL as exc:
                raise ProviderError(str(exc), attempt_count=attempt) from exc
            except httpx.HTTPError as exc:
                retryable = _is_transient_httpx_error(exc)
                if retryable and attempt < total_attempts:
                    self._sleep_before_retry(attempt)
                    continue
                raise ProviderError(
                    _retry_exhausted_message(str(exc), attempt if retryable else 1),
                    attempt_count=attempt,
                ) from exc
        else:  # pragma: no cover - loop always breaks or raises
            raise ProviderError("OpenAI-compatible request failed without response.")
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(str(exc)) from exc
        if not isinstance(data, dict):
            raise ProviderError("OpenAI-compatible response root must be a JSON object.")
        return data

    def _sleep_before_retry(self, attempt: int, *, response: httpx.Response | None = None) -> None:
        delay = _retry_delay_seconds(
            attempt,
            base_seconds=self.retry_backoff_seconds,
            response=response,
        )
        if delay > 0:
            time.sleep(delay)


class HumanProvider:
    name = "human"

    def generate_raw(self, task: str, payload: dict[str, Any], output_model: type[BaseModel]) -> str:
        raise ProviderError("HumanProvider is a placeholder; provide artifact files manually and resume.")


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, type | object] = {
            "mock": MockProvider,
            "openai_compatible": OpenAICompatibleProvider,
            "human": HumanProvider,
        }

    def names(self) -> list[str]:
        return sorted(self._providers)

    def create(
        self,
        spec: str,
        *,
        fixture_dir: Path | None = None,
        endpoint: str | None = None,
        api_key: str | None = None,
        max_retries: int | None = None,
        retry_backoff_seconds: float | None = None,
        http_client: httpx.Client | None = None,
    ) -> Provider:
        provider_name, _, model = spec.partition(":")
        if provider_name == "mock":
            if fixture_dir is None:
                raise ProviderError("mock provider requires fixture_dir")
            if not fixture_dir.is_dir():
                raise ProviderError(f"mock fixture_dir does not exist: {fixture_dir}")
            return MockProvider(fixture_dir)
        if provider_name == "openai_compatible":
            if not model:
                raise ProviderError("openai_compatible provider requires model in spec")
            if endpoint is None:
                raise ProviderError("openai_compatible provider requires endpoint")
            if api_key is None:
                raise ProviderError("openai_compatible provider requires api_key")
            kwargs: dict[str, Any] = {"http_client": http_client}
            if max_retries is not None:
                kwargs["max_retries"] = max_retries
            if retry_backoff_seconds is not None:
                kwargs["retry_backoff_seconds"] = retry_backoff_seconds
            return OpenAICompatibleProvider(model, endpoint, api_key, **kwargs)
        if provider_name == "human":
            return HumanProvider()
        raise ProviderError(f"Unknown provider spec: {spec}")


def _extract_openai_compatible_content(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProviderError("OpenAI-compatible response missing choices.")
    first = choices[0]
    if not isinstance(first, dict):
        raise ProviderError("OpenAI-compatible choice must be an object.")
    message = first.get("message")
    if not isinstance(message, dict):
        raise ProviderError("OpenAI-compatible choice missing message.")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append(item["text"])
        if chunks:
            return "\n".join(chunks)
    raise ProviderError("OpenAI-compatible message content must be text.")


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


def _is_transient_httpx_error(exc: httpx.HTTPError) -> bool:
    return isinstance(
        exc,
        (
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.ReadError,
            httpx.WriteError,
            httpx.RemoteProtocolError,
            httpx.PoolTimeout,
        ),
    )


def _retry_exhausted_message(message: str, attempts: int) -> str:
    if attempts <= 1:
        return message
    return f"{message} (after {attempts} attempts)"


def _strip_retry_attempt_suffix(message: str) -> str:
    return RETRY_ATTEMPT_SUFFIX_RE.sub("", message)


def _retry_delay_seconds(
    attempt: int,
    *,
    base_seconds: float,
    response: httpx.Response | None = None,
) -> float:
    retry_after = _retry_after_seconds(response)
    if retry_after is not None:
        return min(max(0.0, retry_after), OPENAI_COMPATIBLE_MAX_RETRY_DELAY_SECONDS)
    if base_seconds <= 0:
        return 0.0
    exponential = base_seconds * (2 ** max(0, attempt - 1))
    jitter = exponential * 0.1 * random()
    return min(exponential + jitter, OPENAI_COMPATIBLE_MAX_RETRY_DELAY_SECONDS)


def _retry_after_seconds(response: httpx.Response | None) -> float | None:
    if response is None:
        return None
    value = response.headers.get("Retry-After")
    if not value:
        return None
    stripped = value.strip()
    try:
        return float(stripped)
    except ValueError:
        pass
    try:
        parsed = parsedate_to_datetime(stripped)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (parsed - datetime.now(timezone.utc)).total_seconds()


def timed_call(provider: Provider, task: str, payload: dict[str, Any], output_model: type[BaseModel]) -> tuple[str, int]:
    started = time.perf_counter()
    raw = provider.generate_raw(task, payload, output_model)
    return raw, int((time.perf_counter() - started) * 1000)
