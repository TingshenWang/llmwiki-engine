from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .io import write_json
from .models import ProviderResult
from .providers import Provider, ProviderError, timed_call

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    pass


class StructuredModelCall:
    def __init__(
        self,
        provider: Provider,
        *,
        output_dir: Path | None = None,
        result_filename: str | None = None,
        max_repair_attempts: int = 1,
    ):
        self.provider = provider
        self.output_dir = output_dir
        self.result_filename = result_filename
        self.max_repair_attempts = max_repair_attempts

    def run(self, task: str, payload: dict[str, Any], output_model: type[T]) -> tuple[T, ProviderResult]:
        errors: list[str] = []
        latency_ms = 0
        try:
            raw, latency_ms = timed_call(self.provider, task, payload, output_model)
        except ProviderError as exc:
            result = ProviderResult(
                task=task,
                provider=self.provider.name,
                raw_output="",
                latency_ms=latency_ms,
                errors=[str(exc)],
            )
            self._persist(task, result)
            raise StructuredOutputError(str(exc)) from exc

        parsed: dict[str, Any] | None = None
        model: T | None = None
        try:
            parsed = _parse_json(raw)
            model = output_model.model_validate(parsed)
        except (ValueError, ValidationError) as exc:
            errors.append(str(exc))

        result = ProviderResult(
            task=task,
            provider=self.provider.name,
            raw_output=raw,
            parsed_output=parsed,
            parse_success=parsed is not None,
            schema_valid=model is not None,
            latency_ms=latency_ms,
            errors=errors,
        )
        self._persist(task, result)
        if model is None:
            raise StructuredOutputError(f"{task} returned invalid structured output: {'; '.join(errors)}")
        return model, result

    def _persist(self, task: str, result: ProviderResult) -> None:
        if self.output_dir is None:
            return
        filename = self.result_filename or f"{task}.provider_result.json"
        write_json(self.output_dir / filename, result)


def _parse_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Structured output root must be a JSON object.")
    return data
