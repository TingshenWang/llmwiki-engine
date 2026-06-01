from __future__ import annotations

from pathlib import Path

from rich.console import Console

from .io import append_jsonl
from .models import EventRecord
from .redaction import NO_REDACTION, Redactor


class EventLogger:
    def __init__(
        self,
        operation_id: str,
        path: Path,
        console: Console | None = None,
        *,
        redactor: Redactor = NO_REDACTION,
    ):
        self.operation_id = operation_id
        self.path = path
        self.console = console or Console()
        self.redactor = redactor

    def emit(self, step: str, event: str, *, status: str | None = None, message: str | None = None, **data: object) -> None:
        redacted_message = self.redactor.redact_text(message) if message is not None else None
        record = EventRecord(
            operation_id=self.operation_id,
            step=step,
            event=event,
            status=status,
            message=redacted_message,
            data={key: self.redactor.redact(value) for key, value in data.items() if value is not None},
        )
        append_jsonl(self.path, [record])
        if event == "started":
            self.console.print(f"[cyan]{self.operation_id}[/] {step} started")
        elif event == "completed":
            suffix = f" {redacted_message}" if redacted_message else ""
            self.console.print(f"[green]OK[/] {step} completed{suffix}")
        elif event == "failed":
            suffix = f": {redacted_message}" if redacted_message else ""
            self.console.print(f"[red]FAIL[/] {step} failed{suffix}")
