from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Protocol

import httpx
from pydantic import BaseModel


DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS = 300.0


class ProviderError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class Provider(Protocol):
    name: str

    def generate_raw(self, task: str, payload: dict[str, Any], output_model: type[BaseModel]) -> str:
        ...


class MockProvider:
    name = "mock"

    def __init__(self, fixture_dir: Path):
        self.fixture_dir = fixture_dir

    def generate_raw(self, task: str, payload: dict[str, Any], output_model: type[BaseModel]) -> str:
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
        http_client: httpx.Client | None = None,
    ):
        self.model = model
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = timeout
        self.http_client = http_client

    def generate_raw(self, task: str, payload: dict[str, Any], output_model: type[BaseModel]) -> str:
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only JSON matching the requested schema.",
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
        }
        return _extract_openai_compatible_content(self._post_chat(body))

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
        return _extract_openai_compatible_content(self._post_chat(body, timeout=20.0))

    def _post_chat(self, body: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        request_timeout = timeout if timeout is not None else self.timeout
        try:
            if self.http_client is not None:
                response = self.http_client.post(self.endpoint, json=body, headers=headers, timeout=request_timeout)
            else:
                with httpx.Client(timeout=request_timeout) as client:
                    response = client.post(self.endpoint, json=body, headers=headers)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            message = str(exc)
            if exc.response.text:
                message = f"HTTP {exc.response.status_code}: {exc.response.text}"
            raise ProviderError(message, status_code=exc.response.status_code) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(str(exc)) from exc
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(str(exc)) from exc
        if not isinstance(data, dict):
            raise ProviderError("OpenAI-compatible response root must be a JSON object.")
        return data


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
            return OpenAICompatibleProvider(model, endpoint, api_key, http_client=http_client)
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


def timed_call(provider: Provider, task: str, payload: dict[str, Any], output_model: type[BaseModel]) -> tuple[str, int]:
    started = time.perf_counter()
    raw = provider.generate_raw(task, payload, output_model)
    return raw, int((time.perf_counter() - started) * 1000)
