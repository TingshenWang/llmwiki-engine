from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel


@dataclass(frozen=True)
class Redactor:
    secrets: tuple[str, ...] = ()

    def redact_text(self, value: str) -> str:
        text = value
        for secret in self.secrets:
            if secret:
                text = text.replace(secret, "[REDACTED]")
        return text

    def redact(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.redact_text(value)
        if isinstance(value, list):
            return [self.redact(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.redact(item) for item in value)
        if isinstance(value, dict):
            return {key: self.redact(item) for key, item in value.items()}
        return value


NO_REDACTION = Redactor()


TModel = TypeVar("TModel", bound=BaseModel)


def redact_model(redactor: Redactor, model: TModel, model_type: type[TModel]) -> TModel:
    data = redactor.redact(model.model_dump(mode="json"))
    return model_type.model_validate(data)
