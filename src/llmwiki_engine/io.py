from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, TypeVar

import yaml
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def write_json(path: Path, model_or_data: BaseModel | dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(model_or_data, BaseModel):
        text = model_or_data.model_dump_json(indent=2)
    else:
        text = json.dumps(model_or_data, ensure_ascii=False, indent=2)
    path.write_text(text + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_model(path: Path, model_type: type[T]) -> T:
    return model_type.model_validate(read_json(path))


def append_jsonl(path: Path, rows: Iterable[BaseModel | dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            data = row.model_dump(mode="json") if isinstance(row, BaseModel) else row
            handle.write(json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {} if data is None else data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")

