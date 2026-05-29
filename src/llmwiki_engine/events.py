from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from rich.console import Console

from .io import append_jsonl
from .models import EventRecord


class EventLogger:
    def __init__(self, operation_id: str, path: Path, console: Console | None = None):
        self.operation_id = operation_id
        self.path = path
        self.console = console or Console()

    def emit(self, step: str, event: str, *, status: str | None = None, message: str | None = None, **data: object) -> None:
        record = EventRecord(
            operation_id=self.operation_id,
            step=step,
            event=event,
            status=status,
            message=message,
            data={key: value for key, value in data.items() if value is not None},
        )
        append_jsonl(self.path, [record])
        if event == "started":
            self.console.print(f"[cyan]{self.operation_id}[/] {step} started")
        elif event == "completed":
            suffix = f" {message}" if message else ""
            self.console.print(f"[green]OK[/] {step} completed{suffix}")
        elif event == "failed":
            suffix = f": {message}" if message else ""
            self.console.print(f"[red]FAIL[/] {step} failed{suffix}")

    @contextmanager
    def step(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        self.emit(name, "started", status="running")
        try:
            yield
        except Exception as exc:
            self.emit(name, "failed", status="failed", message=str(exc), duration_ms=int((time.perf_counter() - started) * 1000))
            raise
        self.emit(name, "completed", status="completed", duration_ms=int((time.perf_counter() - started) * 1000))
