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

    def emit(
        self,
        step: str,
        event: str,
        *,
        status: str | None = None,
        message: str | None = None,
        duration_ms: int | None = None,
        **data: object,
    ) -> None:
        redacted_message = self.redactor.redact_text(message) if message is not None else None
        record = EventRecord(
            operation_id=self.operation_id,
            step=step,
            event=event,
            status=status,
            duration_ms=duration_ms,
            message=redacted_message,
            data={key: self.redactor.redact(value) for key, value in data.items() if value is not None},
        )
        append_jsonl(self.path, [record])
        if event == "started":
            provider = data.get("provider_spec")
            if data.get("model_backed"):
                suffix = f" ({provider})" if provider else ""
                self.console.print(f"[cyan]{self.operation_id}[/] 模型处理中: {step}{suffix} ...")
            else:
                self.console.print(f"[cyan]{self.operation_id}[/] 本地处理中: {step} ...")
        elif event == "completed":
            suffix = f" {redacted_message}" if redacted_message else ""
            self.console.print(f"[green]完成[/] {step}，用时 {format_duration(duration_ms)}{suffix}")
        elif event == "failed":
            suffix = f": {redacted_message}" if redacted_message else ""
            self.console.print(f"[red]失败[/] {step}，用时 {format_duration(duration_ms)}{suffix}")


def format_duration(duration_ms: int | None) -> str:
    if duration_ms is None:
        return "-"
    seconds = round(duration_ms / 1000)
    minutes, second = divmod(seconds, 60)
    hours, minute = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minute}m {second}s"
    if minute:
        return f"{minute}m {second}s"
    return f"{second}s"
