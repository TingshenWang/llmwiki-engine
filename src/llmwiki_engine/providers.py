from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from .models import ProviderConfig, ProviderResult


class ProviderError(RuntimeError):
    pass


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


class OpenAIProvider:
    name = "openai"

    def __init__(self, model: str, api_key_env: str = "OPENAI_API_KEY"):
        self.model = model
        self.api_key_env = api_key_env

    def generate_raw(self, task: str, payload: dict[str, Any], output_model: type[BaseModel]) -> str:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise ProviderError(f"Missing API key env var: {self.api_key_env}")
        body = {
            "model": self.model,
            "input": [
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
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        return _extract_text_from_response(_http_json_text(request))


class OllamaProvider:
    name = "ollama"

    def __init__(self, model: str, endpoint: str = "http://localhost:11434/api/generate"):
        self.model = model
        self.endpoint = endpoint

    def generate_raw(self, task: str, payload: dict[str, Any], output_model: type[BaseModel]) -> str:
        body = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "prompt": json.dumps(
                {
                    "instruction": "Return only JSON matching schema.",
                    "task": task,
                    "payload": payload,
                    "schema": output_model.model_json_schema(),
                },
                ensure_ascii=False,
            ),
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        data = json.loads(_http_json_text(request))
        return data.get("response", "")


class LocalHTTPProvider:
    name = "local_http"

    def __init__(self, endpoint: str, model: str | None = None):
        self.endpoint = endpoint
        self.model = model

    def generate_raw(self, task: str, payload: dict[str, Any], output_model: type[BaseModel]) -> str:
        body = {
            "model": self.model,
            "task": task,
            "payload": payload,
            "schema": output_model.model_json_schema(),
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return _extract_text_from_response(_http_json_text(request))


class HumanProvider:
    name = "human"

    def generate_raw(self, task: str, payload: dict[str, Any], output_model: type[BaseModel]) -> str:
        raise ProviderError("HumanProvider is a placeholder; provide artifact files manually and resume.")


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, type | object] = {
            "mock": MockProvider,
            "openai": OpenAIProvider,
            "ollama": OllamaProvider,
            "local_http": LocalHTTPProvider,
            "human": HumanProvider,
        }

    def names(self) -> list[str]:
        return sorted(self._providers)

    def create(self, spec: str, *, fixture_dir: Path | None = None, endpoint: str | None = None) -> Provider:
        provider_name, _, model = spec.partition(":")
        if provider_name == "mock":
            if fixture_dir is None:
                raise ProviderError("mock provider requires fixture_dir")
            return MockProvider(fixture_dir)
        if provider_name == "openai":
            return OpenAIProvider(model or "gpt-4.1-mini")
        if provider_name == "ollama":
            return OllamaProvider(model or "llama3", endpoint or "http://localhost:11434/api/generate")
        if provider_name == "local_http":
            if endpoint is None:
                raise ProviderError("local_http provider requires endpoint")
            return LocalHTTPProvider(endpoint, model or None)
        if provider_name == "human":
            return HumanProvider()
        raise ProviderError(f"Unknown provider spec: {spec}")


def provider_config_from_spec(task: str, spec: str, fixture_dir: Path | None = None) -> ProviderConfig:
    provider, _, model = spec.partition(":")
    return ProviderConfig(task=task, provider=provider, model=model or None, fixture_dir=fixture_dir)


def _http_json_text(request: urllib.request.Request) -> str:
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise ProviderError(str(exc)) from exc


def _extract_text_from_response(raw: str) -> str:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(data, dict):
        for key in ("output_text", "response", "text", "result"):
            value = data.get(key)
            if isinstance(value, str):
                return value
        output = data.get("output")
        if isinstance(output, list):
            chunks: list[str] = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                for content in item.get("content", []):
                    if isinstance(content, dict) and isinstance(content.get("text"), str):
                        chunks.append(content["text"])
            if chunks:
                return "\n".join(chunks)
    return raw


def timed_call(provider: Provider, task: str, payload: dict[str, Any], output_model: type[BaseModel]) -> tuple[str, int]:
    started = time.perf_counter()
    raw = provider.generate_raw(task, payload, output_model)
    return raw, int((time.perf_counter() - started) * 1000)
