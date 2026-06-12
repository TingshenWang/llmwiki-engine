from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .lite import pipeline
from .lite.providers import ProviderConfigError, load_provider_registry


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
        counts = " ".join(f"{_count_label(key)}={value}" for key, value in step.counts.items())
        duration = "" if step.duration_seconds is None else f"{step.duration_seconds:.2f}s"
        table.add_row(_step_label(step.name), _status_label(step.status), duration, counts)
    console.print(f"状态：[bold]{_status_label(manifest.status)}[/]")
    if manifest.receipt_path:
        console.print(f"回执：{manifest.receipt_path}")
    console.print(table)


def _step_label(name: str) -> str:
    return {
        "raw_binding": "绑定 raw",
        "source_digest": "来源消化",
        "wiki_snapshot": "Wiki 快照",
        "candidate_pages": "候选知识页",
        "candidate_contexts": "候选召回",
        "merge_plan": "合并计划",
        "composition_plan": "写作编排",
        "final_pages": "最终页面",
        "validation": "校验",
        "knowledge_write": "写入知识页",
        "source_record_write": "写入来源页",
        "index_log_write": "写入索引日志",
        "embedding_cache_refresh": "刷新向量缓存",
        "receipt": "写入回执",
    }.get(name, name)


def _status_label(status: str) -> str:
    return {
        "created": "已创建",
        "running": "运行中",
        "failed": "失败",
        "validated": "已校验",
        "written": "已写入",
        "source_recorded": "已记录来源",
        "pending": "等待中",
        "completed": "完成",
    }.get(status, status)


def _severity_label(severity: str) -> str:
    return {"error": "错误", "warning": "警告"}.get(severity, severity)


def _bool_label(value: bool) -> str:
    return "是" if value else "否"


def _count_label(key: str) -> str:
    return {
        "raw_size_bytes": "raw 大小",
        "candidate_count": "候选数",
        "weak_noise_count": "弱/噪声数",
        "deferred_count": "延后数",
        "candidate_page_count": "候选页数",
        "covered_digest_candidate_count": "覆盖候选数",
        "knowledge_pool_size": "知识池",
        "query_count": "查询数",
        "top_k": "召回数量",
        "retrieval_backend": "召回后端",
        "final_page_count": "最终页数",
        "final_target_count": "最终目标数",
        "update_target_count": "更新目标数",
        "related_link_count": "相关链接数",
        "knowledge_written_count": "知识页写入",
        "source_record_count": "来源页",
        "system_written_count": "系统页",
        "written_target_count": "写入目标",
        "receipt_count": "回执",
        "cache_hit": "缓存命中",
        "cache_refreshed": "缓存刷新",
        "cache_pruned": "缓存清理",
        "parallel_request_count": "并发请求",
        "parallel_max_workers": "最大并发",
        "diff_count": "diff 数",
        "error_count": "错误数",
        "warning_count": "警告数",
        "create_count": "新建数",
        "update_count": "更新数",
        "noop_count": "不改动数",
        "split_count": "拆分数",
        "merge_count": "合并数",
        "related_kept_count": "相关保留数",
        "related_filtered_count": "相关过滤数",
    }.get(key, key)


def _provider_message(message: str) -> str:
    return {
        "ok": "正常",
        "missing endpoint": "缺少 endpoint",
        "missing API key": "缺少 API key",
        "missing fixture_dir": "缺少 fixture_dir",
        "not a real model provider": "不是真实模型 provider",
        "mock provider is not live": "mock 不是 live provider",
        "unsupported provider spec": "不支持的 provider",
        "endpoint must be chat completions URL": "endpoint 必须是 chat completions URL",
        "missing model": "缺少 model",
    }.get(message, message)


if __name__ == "__main__":
    app()
