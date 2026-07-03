"""Reasoning 思维链保存与流式显示的测试。"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from pydantic import BaseModel
from rich.console import Console
from typer.testing import CliRunner

from llmwiki_engine.cli import app
from llmwiki_engine.lite.io import read_json
from llmwiki_engine.lite.models import OperationManifest
from llmwiki_engine.lite.pipeline import (
    PipelineError,
    _call_provider_artifact_soft,
    _call_provider_artifacts_parallel_soft,
    _run_step,
    init_vault,
)
from llmwiki_engine.lite.providers import (
    ProviderCallResult,
    ProviderRegistry,
    ProviderSpec,
    PromptRequest,
    _compact_response,
    build_chat_payload,
)

from test_lite_pipeline import write_raw


# ---------------------------------------------------------------------------
# Helper models and classes
# ---------------------------------------------------------------------------


class SimpleOutput(BaseModel):
    ok: bool = True


def _make_spec(*, endpoint: str = "https://example.test/v1/chat/completions", **kwargs) -> ProviderSpec:
    return ProviderSpec(
        spec="openai_compatible:test-model",
        endpoint=endpoint,
        api_key="test-key",
        max_retries=0,
        retry_backoff_seconds=0,
        **kwargs,
    )


def _make_request() -> PromptRequest:
    return PromptRequest(
        step="test_step",
        schema_name="test_schema",
        system_prompt="test",
        user_payload={"input": "test"},
        response_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
    )


class _DummyResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=httpx.Request("POST", "https://example.test"),
                response=httpx.Response(self.status_code, text=self.text),
            )


class _DummyClient:
    """非流式 mock client，支持 post()。"""

    def __init__(self, timeout: float, response: _DummyResponse) -> None:
        self.timeout = timeout
        self._response = response
        self.posted_payloads: list[dict] = []

    def __enter__(self) -> "_DummyClient":
        return self

    def __exit__(self, *args) -> bool:
        return False

    def post(self, url, *, headers=None, json=None, **kwargs) -> _DummyResponse:
        self.posted_payloads.append(json or {})
        return self._response


class _DummyStreamResponse:
    """流式响应 mock。"""

    def __init__(self, status_code: int, lines: list[str] | None = None, error_text: str = "") -> None:
        self.status_code = status_code
        self._lines = lines or []
        self._error_text = error_text

    def iter_lines(self):
        for line in self._lines:
            yield line

    def read(self) -> bytes:
        return self._error_text.encode("utf-8")


class _DummyStreamCM:
    def __init__(self, response: _DummyStreamResponse) -> None:
        self._response = response

    def __enter__(self) -> _DummyStreamResponse:
        return self._response

    def __exit__(self, *args) -> bool:
        return False


class _DummyStreamClient:
    """流式 mock client，支持 stream() 和 post()（用于 fallback）。"""

    def __init__(self, timeout: float, stream_response: _DummyStreamResponse, post_response: _DummyResponse | None = None) -> None:
        self.timeout = timeout
        self._stream_response = stream_response
        self._post_response = post_response
        self.stream_payloads: list[dict] = []
        self.post_payloads: list[dict] = []

    def __enter__(self) -> "_DummyStreamClient":
        return self

    def __exit__(self, *args) -> bool:
        return False

    def stream(self, method, url, *, headers=None, json=None, **kwargs) -> _DummyStreamCM:
        self.stream_payloads.append(json or {})
        return _DummyStreamCM(self._stream_response)

    def post(self, url, *, headers=None, json=None, **kwargs) -> _DummyResponse:
        self.post_payloads.append(json or {})
        if self._post_response is None:
            raise RuntimeError("post() called but no post_response configured")
        return self._post_response


def _make_reasoning_sse_lines(reasoning_chunks: list[str], content_json: dict, *, finish_reason: str = "stop") -> list[str]:
    """构造 DeepSeek 流式 SSE 行序列。"""
    lines: list[str] = []
    for chunk in reasoning_chunks:
        lines.append("data: " + json.dumps({"choices": [{"delta": {"reasoning_content": chunk}}]}, ensure_ascii=False))
    lines.append("data: " + json.dumps({"choices": [{"delta": {"content": json.dumps(content_json, ensure_ascii=False)}}]}, ensure_ascii=False))
    lines.append("data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": finish_reason}]}, ensure_ascii=False))
    lines.append("data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "completion_tokens_details": {"reasoning_tokens": 8}}}, ensure_ascii=False))
    lines.append("data: [DONE]")
    return lines


class _FakeRegistry:
    """pipeline 测试用的 FakeRegistry，支持 reasoning_callback 和 reasoning_content。"""

    def __init__(self, *, reasoning_content: str | None = None, output_model: type[BaseModel] | None = None) -> None:
        self.spec = ProviderSpec(
            spec="openai_compatible:deepseek-reasoner",
            endpoint="https://api.deepseek.com/v1/chat/completions",
            api_key="test-key",
        )
        self.reasoning_content = reasoning_content
        self._output_model = output_model or SimpleOutput
        self.requests: list = []
        self.kwargs_list: list[dict] = []

    def provider_for(self, step: str) -> ProviderSpec:
        return self.spec

    def call_structured(self, step: str, request, output_model, **kwargs) -> ProviderCallResult:
        self.requests.append(request)
        self.kwargs_list.append(kwargs)
        return ProviderCallResult(
            output=output_model(ok=True),
            prompt_artifact={"step": step},
            provider_result={"parsed": {"ok": True}, "reasoning_content": self.reasoning_content} if self.reasoning_content else {"parsed": {"ok": True}},
            sanitized_context=self.spec.sanitized_context(),
            api_calls=[
                {
                    "step": step,
                    "request_key": "",
                    "model": "deepseek-reasoner",
                    "attempt": 0,
                    "call_index": len(self.requests),
                    "status": "success",
                    "finish_reason": "stop",
                    "duration_ms": 1.0,
                    "response_status_code": 200,
                    "error": "",
                    "prompt_tokens": 10,
                    "prompt_cache_hit_tokens": 0,
                    "prompt_cache_miss_tokens": 10,
                    "completion_tokens": 5,
                    "reasoning_tokens": 8 if self.reasoning_content else 0,
                    "total_tokens": 15,
                    "cache_hit_rate_percent": 0.0,
                    "price_cny": 0.00002,
                }
            ],
            reasoning_content=self.reasoning_content,
        )


# ---------------------------------------------------------------------------
# providers.py tests
# ---------------------------------------------------------------------------


def test_reasoning_content_extracted_from_non_streaming_response(tmp_path: Path, monkeypatch) -> None:
    """非流式响应含 reasoning_content 时，ProviderCallResult.reasoning_content 非空。"""
    spec = _make_spec()
    request = _make_request()
    response = _DummyResponse({
        "choices": [{
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": json.dumps({"ok": True}),
                "reasoning_content": "这是一段推理过程。",
            },
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    })
    client = _DummyClient(timeout=180.0, response=response)
    monkeypatch.setattr("llmwiki_engine.lite.providers.httpx.Client", lambda **kw: client)

    result = ProviderRegistry({"test_step": spec}).call_structured("test_step", request, SimpleOutput)

    assert result.reasoning_content == "这是一段推理过程。"
    assert result.output.ok is True
    assert result.provider_result["reasoning_content"] == "这是一段推理过程。"


def test_reasoning_content_none_for_non_reasoning_model(tmp_path: Path, monkeypatch) -> None:
    """非推理模型响应无 reasoning_content 时，reasoning_content 为 None。"""
    spec = _make_spec()
    request = _make_request()
    response = _DummyResponse({
        "choices": [{
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": json.dumps({"ok": True}),
            },
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    })
    client = _DummyClient(timeout=180.0, response=response)
    monkeypatch.setattr("llmwiki_engine.lite.providers.httpx.Client", lambda **kw: client)

    result = ProviderRegistry({"test_step": spec}).call_structured("test_step", request, SimpleOutput)

    assert result.reasoning_content is None
    assert result.output.ok is True


def test_build_chat_payload_stream_flag() -> None:
    """stream=True 时 payload 含 stream 和 stream_options；stream=False 时不含。"""
    spec = _make_spec()
    request = _make_request()

    payload_stream = build_chat_payload(spec, request, stream=True)
    assert payload_stream["stream"] is True
    assert payload_stream["stream_options"] == {"include_usage": True}

    payload_normal = build_chat_payload(spec, request, stream=False)
    assert "stream" not in payload_normal
    assert "stream_options" not in payload_normal


def test_compact_response_preserves_reasoning_content() -> None:
    """_compact_response 保留 message 中的 reasoning_content 字段。"""
    raw = {
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": '{"ok": true}',
                "reasoning_content": "推理文本",
            },
        }],
        "usage": {"prompt_tokens": 10},
    }
    compact = _compact_response(raw)
    msg = compact["choices"][0]["message"]
    assert msg["reasoning_content"] == "推理文本"
    assert msg["content"] == '{"ok": true}'
    assert msg["role"] == "assistant"


def test_streaming_reasoning_invokes_callback(tmp_path: Path, monkeypatch) -> None:
    """流式模式下 reasoning_callback 被按序调用，reasoning_content 完整，content 可解析。"""
    spec = _make_spec()
    request = _make_request()
    reasoning_chunks = ["思考第一步。", "思考第二步。"]
    sse_lines = _make_reasoning_sse_lines(reasoning_chunks, {"ok": True})
    stream_response = _DummyStreamResponse(200, sse_lines)
    client = _DummyStreamClient(timeout=180.0, stream_response=stream_response)
    monkeypatch.setattr("llmwiki_engine.lite.providers.httpx.Client", lambda **kw: client)

    received_chunks: list[str] = []
    result = ProviderRegistry({"test_step": spec}).call_structured(
        "test_step", request, SimpleOutput, reasoning_callback=received_chunks.append
    )

    assert received_chunks == reasoning_chunks
    assert result.reasoning_content == "思考第一步。思考第二步。"
    assert result.output.ok is True
    assert client.stream_payloads[0]["stream"] is True


def test_streaming_falls_back_on_400_json_schema_unsupported(tmp_path: Path, monkeypatch) -> None:
    """流式 400（response_format unavailable）时 fallback 到非流式 json_object。"""
    spec = _make_spec()  # json_schema mode by default
    request = _make_request()
    stream_response = _DummyStreamResponse(400, error_text='{"error":{"message":"response_format unavailable"}}')
    post_response = _DummyResponse({
        "choices": [{
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": json.dumps({"ok": True})},
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    })
    client = _DummyStreamClient(timeout=180.0, stream_response=stream_response, post_response=post_response)
    monkeypatch.setattr("llmwiki_engine.lite.providers.httpx.Client", lambda **kw: client)

    result = ProviderRegistry({"test_step": spec}).call_structured("test_step", request, SimpleOutput, reasoning_callback=lambda _: None)

    assert result.output.ok is True
    assert len(client.stream_payloads) == 1
    assert len(client.post_payloads) == 1
    assert client.post_payloads[0]["response_format"] == {"type": "json_object"}
    assert result.provider_result["provider"]["response_format_fallback"] == "json_object"


def test_call_structured_passes_reasoning_callback(tmp_path: Path, monkeypatch) -> None:
    """传入 reasoning_callback 时走流式路径（payload 含 stream:true）；不传时走非流式。"""
    spec = _make_spec()
    request = _make_request()
    sse_lines = _make_reasoning_sse_lines(["推理"], {"ok": True})
    stream_response = _DummyStreamResponse(200, sse_lines)
    post_response = _DummyResponse({
        "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps({"ok": True})}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    })
    client = _DummyStreamClient(timeout=180.0, stream_response=stream_response, post_response=post_response)
    monkeypatch.setattr("llmwiki_engine.lite.providers.httpx.Client", lambda **kw: client)

    # With callback → streaming
    client.stream_payloads.clear()
    client.post_payloads.clear()
    ProviderRegistry({"test_step": spec}).call_structured("test_step", request, SimpleOutput, reasoning_callback=lambda _: None)
    assert len(client.stream_payloads) == 1
    assert client.stream_payloads[0]["stream"] is True

    # Without callback → non-streaming
    client.stream_payloads.clear()
    client.post_payloads.clear()
    ProviderRegistry({"test_step": spec}).call_structured("test_step", request, SimpleOutput)
    assert len(client.post_payloads) == 1
    assert "stream" not in client.post_payloads[0]


# ---------------------------------------------------------------------------
# pipeline.py tests
# ---------------------------------------------------------------------------


def _make_state(registry: _FakeRegistry, *, show_reasoning: bool = False, console: Console | None = None) -> dict:
    return {
        "provider_registry": registry,
        "provider_contexts": {},
        "show_reasoning": show_reasoning,
        "console": console,
    }


def test_reasoning_md_artifact_saved_non_streaming(tmp_path: Path) -> None:
    """非流式模式下 reasoning_content 非空时保存 .reasoning.md artifact。"""
    registry = _FakeRegistry(reasoning_content="这是推理内容。")
    state = _make_state(registry)
    out_dir = tmp_path / "step_out"
    out_dir.mkdir()

    output, artifacts, model_calls, api_calls, error = _call_provider_artifact_soft(
        state, out_dir, "test_step", _make_request(), SimpleOutput
    )

    assert error is None
    reasoning_path = out_dir / "model_calls" / "test_step.reasoning.md"
    assert reasoning_path.exists()
    assert reasoning_path.read_text(encoding="utf-8") == "这是推理内容。"
    assert reasoning_path in artifacts


def test_reasoning_content_in_provider_result_json(tmp_path: Path) -> None:
    """provider_result.json 含 reasoning_content 字段。"""
    registry = _FakeRegistry(reasoning_content="推理审计文本。")
    state = _make_state(registry)
    out_dir = tmp_path / "step_out"
    out_dir.mkdir()

    _call_provider_artifact_soft(state, out_dir, "test_step", _make_request(), SimpleOutput)

    result_path = out_dir / "model_calls" / "test_step.provider_result.json"
    provider_result = read_json(result_path)
    assert provider_result["reasoning_content"] == "推理审计文本。"


def test_no_reasoning_artifact_when_empty(tmp_path: Path) -> None:
    """reasoning_content 为 None 时不生成 .reasoning.md 文件。"""
    registry = _FakeRegistry(reasoning_content=None)
    state = _make_state(registry)
    out_dir = tmp_path / "step_out"
    out_dir.mkdir()

    output, artifacts, model_calls, api_calls, error = _call_provider_artifact_soft(
        state, out_dir, "test_step", _make_request(), SimpleOutput
    )

    assert error is None
    reasoning_path = out_dir / "model_calls" / "test_step.reasoning.md"
    assert not reasoning_path.exists()
    assert not any(p.name.endswith(".reasoning.md") for p in artifacts)


def test_show_reasoning_disables_spinner(tmp_path: Path) -> None:
    """show_reasoning=True 时 _run_step 不使用 spinner（打印"开始"行）。"""
    output_buf = io.StringIO()
    console = Console(file=output_buf, force_terminal=True)
    manifest = OperationManifest(
        operation_id="test",
        status="running",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        vault=str(tmp_path),
        raw_path="raw/test.md",
        profile_name="test",
        engine_version="test",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    from llmwiki_engine.lite.pipeline import StepOutput

    def step_fn() -> StepOutput:
        return StepOutput([], {})

    _run_step(run_dir, manifest, "source_digest", step_fn, console, emit_progress=True, show_reasoning=True)

    text = output_buf.getvalue()
    assert "开始" in text
    assert "完成" in text


def test_parallel_reasoning_artifacts_saved(tmp_path: Path) -> None:
    """并行步骤每个 key 的 .reasoning.md 都生成。"""
    registry = _FakeRegistry(reasoning_content="并行推理内容。")
    state = _make_state(registry)
    out_dir = tmp_path / "parallel_out"
    out_dir.mkdir()

    requests = [("key_一", _make_request()), ("key_二", _make_request())]
    results, artifacts, model_calls, api_calls, errors = _call_provider_artifacts_parallel_soft(
        state, out_dir, "final_pages", requests, SimpleOutput
    )

    assert errors == {}
    model_dir = out_dir / "model_calls"
    for key, _ in requests:
        from llmwiki_engine.lite.io import safe_filename

        stem = f"final_pages_{safe_filename(key)}"
        reasoning_path = model_dir / f"{stem}.reasoning.md"
        assert reasoning_path.exists(), f"Missing reasoning artifact for key={key}"
        assert reasoning_path.read_text(encoding="utf-8") == "并行推理内容。"


def test_parallel_reasoning_displayed_on_completion(tmp_path: Path) -> None:
    """show_reasoning=True + 并行步骤时，每个 key 完成后打印 reasoning 块。"""
    registry = _FakeRegistry(reasoning_content="并行推理展示文本。")
    output_buf = io.StringIO()
    console = Console(file=output_buf)
    state = _make_state(registry, show_reasoning=True, console=console)
    out_dir = tmp_path / "parallel_out"
    out_dir.mkdir()

    requests = [("keyA", _make_request()), ("keyB", _make_request())]
    _call_provider_artifacts_parallel_soft(state, out_dir, "final_pages", requests, SimpleOutput)

    text = output_buf.getvalue()
    assert "▌" in text
    assert "keyA" in text
    assert "keyB" in text
    assert "并行推理展示文本。" in text


# ---------------------------------------------------------------------------
# cli.py tests
# ---------------------------------------------------------------------------


def test_ingest_run_reasoning_flag_exists() -> None:
    """--help 输出含 --reasoning 选项。"""
    result = CliRunner().invoke(app, ["ingest", "run", "--help"])
    assert result.exit_code == 0
    assert "--reasoning" in result.output


def test_reasoning_flag_passed_to_pipeline(tmp_path: Path, monkeypatch) -> None:
    """--reasoning flag 传递 show_reasoning=True 给 pipeline.run_ingest。"""
    vault = init_vault(tmp_path / "vault")
    raw = write_raw(vault)
    captured: dict = {}

    def fake_run_ingest(*args, **kwargs) -> OperationManifest:
        captured.update(kwargs)
        return OperationManifest(
            operation_id="test",
            status="written",
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
            vault=str(vault),
            raw_path="raw/project_note.md",
            profile_name="project_basic",
            engine_version="test",
        )

    monkeypatch.setattr("llmwiki_engine.lite.pipeline.run_ingest", fake_run_ingest)
    result = CliRunner().invoke(app, ["ingest", "run", str(vault), str(raw), "--reasoning"])

    assert result.exit_code == 0, result.output
    assert captured.get("show_reasoning") is True


def test_json_mode_with_reasoning_ignored(tmp_path: Path, monkeypatch) -> None:
    """--json --reasoning 组合不报错，show_reasoning 被 --json 覆盖为 False。"""
    vault = init_vault(tmp_path / "vault")
    raw = write_raw(vault)
    captured: dict = {}

    def fake_run_ingest(*args, **kwargs) -> OperationManifest:
        captured.update(kwargs)
        return OperationManifest(
            operation_id="test",
            status="written",
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
            vault=str(vault),
            raw_path="raw/project_note.md",
            profile_name="project_basic",
            engine_version="test",
        )

    monkeypatch.setattr("llmwiki_engine.lite.pipeline.run_ingest", fake_run_ingest)
    result = CliRunner().invoke(app, ["ingest", "run", str(vault), str(raw), "--json", "--reasoning"])

    assert result.exit_code == 0, result.output
    assert captured.get("show_reasoning") is False
