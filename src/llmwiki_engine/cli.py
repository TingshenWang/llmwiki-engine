from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .lite import pipeline
from .lite.labels import count_label, step_label
from .lite.providers import ProviderConfigError, load_provider_registry
from .lite.token_usage import format_duration_seconds, format_percent, format_price_cny, summarize_step_counts


app = typer.Typer(help="LLM-Wiki Lite 中文知识编译引擎。")
ingest_app = typer.Typer(help="运行和检查自动化 Lite ingest。")
providers_app = typer.Typer(help="检查 Lite provider 配置。")
app.add_typer(ingest_app, name="ingest")
app.add_typer(providers_app, name="providers")

console = Console()


@app.command()
def init(vault: Path, profile: str = typer.Option("project_basic", "--profile")) -> None:
    """初始化 Lite vault。"""
    path = pipeline.init_vault(vault, profile_name=profile)
    console.print(f"[green]已初始化[/] {path}")


@ingest_app.command("run")
def ingest_run(
    vault: Path,
    raw: Path,
    profile: Optional[str] = typer.Option(None, "--profile"),
    slug: Optional[str] = typer.Option(None, "--slug"),
    json_output: bool = typer.Option(False, "--json", help="输出机器可读 JSON。"),
) -> None:
    """运行全自动 Lite ingest 并写入 wiki transaction。"""
    try:
        run_console = Console(file=io.StringIO()) if json_output else console
        manifest = pipeline.run_ingest(vault, raw, profile_name=profile, slug=slug, console=run_console, emit_progress=not json_output)
    except (ValueError, pipeline.PipelineError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if json_output:
        typer.echo(manifest.model_dump_json(indent=2))
        return
    _print_manifest(manifest)


@ingest_app.command("status")
def ingest_status(
    vault: Path,
    operation_id: Optional[str] = typer.Argument(None),
    verify: bool = typer.Option(False, "--verify"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """显示 operation 状态。"""
    try:
        manifest = pipeline.status(vault, operation_id)
        verify_report = pipeline.verify_operation(vault, manifest.operation_id) if verify else None
    except (ValueError, pipeline.PipelineError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if json_output:
        payload = manifest.model_dump(mode="json")
        if verify_report is not None:
            payload["verify"] = verify_report.model_dump(mode="json")
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        raise typer.Exit(0 if verify_report is None or verify_report.ok else 2)
    _print_manifest(manifest)
    if verify_report is not None:
        console.print("[green]校验：通过[/]" if verify_report.ok else "[red]校验：失败[/]")
        for issue in verify_report.issues:
            console.print(f"- {_severity_label(issue.severity)} `{issue.code}`：{issue.message}")
        raise typer.Exit(0 if verify_report.ok else 2)


@ingest_app.command("inspect")
def ingest_inspect(
    vault: Path,
    operation_id: Optional[str] = typer.Argument(None),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """显示 operation 详情。"""
    payload = pipeline.inspect_operation(vault, operation_id)
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    console.print_json(json.dumps(payload, ensure_ascii=False))


@ingest_app.command("raw-candidates")
def raw_candidates(
    vault: Path,
    include_processed: bool = typer.Option(False, "--all"),
    limit: Optional[int] = typer.Option(None, "--limit", min=0),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """列出可 ingest 的 raw 文件。"""
    report = pipeline.scan_raw_candidates(vault, include_processed=include_processed, limit=limit)
    if json_output:
        typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
        return
    table = Table(title="原始材料候选")
    table.add_column("原始材料")
    table.add_column("已处理")
    table.add_column("大小")
    for item in report["items"]:
        table.add_row(str(item["raw_path"]), _bool_label(bool(item["processed"])), str(item["size_bytes"]))
    console.print(table)


@providers_app.command("check")
def providers_check(vault: Path, live: bool = typer.Option(False, "--live")) -> None:
    """检查 Lite provider 配置，不泄露密钥。"""
    try:
        registry = load_provider_registry(vault)
        reports = registry.check(live=live)
    except ProviderConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc
    table = Table(title="模型服务配置")
    table.add_column("名称")
    table.add_column("可用")
    table.add_column("spec")
    table.add_column("信息")
    for report in reports:
        context = report["context"]
        table.add_row(str(report["name"]), "是" if report["ok"] else "否", str(context["spec"]), _provider_message(str(report["message"])))
    console.print(table)
    if not all(report["ok"] for report in reports):
        raise typer.Exit(2)


def _print_manifest(manifest: object) -> None:
    table = Table(title=f"操作 {manifest.operation_id}")
    table.add_column("步骤")
    table.add_column("状态")
    table.add_column("耗时")
    table.add_column("计数")
    for step in manifest.steps:
        counts = " ".join(f"{count_label(key)}={_format_count_value(key, value)}" for key, value in step.counts.items())
        duration = "" if step.duration_seconds is None else f"{step.duration_seconds:.2f}s"
        table.add_row(step_label(step.name), _status_label(step.status), duration, counts)
    console.print(f"状态：[bold]{_status_label(manifest.status)}[/]")
    if manifest.receipt_path:
        console.print(f"回执：{manifest.receipt_path}")
    console.print(table)
    _print_operation_token_summary(manifest)


def _print_operation_token_summary(manifest: object) -> None:
    duration_seconds = sum(float(step.duration_seconds or 0) for step in manifest.steps)
    summary = summarize_step_counts([step.counts for step in manifest.steps], duration_seconds=duration_seconds)
    if not summary.get("api_call_count"):
        return
    table = Table(title="本次 Ingest 汇总")
    table.add_column("总耗时", justify="right")
    table.add_column("总输入token", justify="right")
    table.add_column("总缓存token", justify="right")
    table.add_column("总输出token", justify="right")
    table.add_column("总缓存命中率", justify="right")
    table.add_column("总价", justify="right")
    table.add_row(
        format_duration_seconds(summary["duration_seconds"]),
        str(summary["prompt_tokens"]),
        str(summary["prompt_cache_hit_tokens"]),
        str(summary["completion_tokens"]),
        format_percent(summary["cache_hit_rate_percent"]),
        format_price_cny(summary["price_cny"]),
    )
    console.print(table)


def _format_count_value(key: str, value: object) -> object:
    if key == "price_cny":
        return format_price_cny(value)
    if key == "cache_hit_rate_percent":
        return format_percent(value)
    return value


def _status_label(status: str) -> str:
    return {
        "created": "已创建",
        "running": "运行中",
        "failed": "失败",
        "written": "已写入",
        "source_recorded": "已记录来源",
        "completed": "完成",
    }.get(status, status)


def _severity_label(severity: str) -> str:
    return {"error": "错误", "warning": "警告"}.get(severity, severity)


def _bool_label(value: bool) -> str:
    return "是" if value else "否"


def _provider_message(message: str) -> str:
    return {
        "ok": "正常",
        "live ok": "真实调用通过",
        "missing endpoint": "缺少 endpoint",
        "missing API key": "缺少 API key",
        "not a real model provider": "不是真实模型 provider",
        "unsupported provider spec": "不支持的 provider",
        "endpoint must be chat completions URL": "endpoint 必须是 chat completions URL",
        "missing model": "缺少 model",
    }.get(message, message)


if __name__ == "__main__":
    app()
