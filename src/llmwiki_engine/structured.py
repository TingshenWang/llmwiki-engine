from __future__ import annotations

import json
import re
from pathlib import Path
from time import monotonic
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, ValidationError

from .io import write_json
from .models import ProviderResult, StructuredAttemptRef, StructuredIssue, StructuredRepairReport
from .providers import Provider, ProviderError, timed_call
from .redaction import NO_REDACTION, Redactor

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
        redactor: Redactor = NO_REDACTION,
        max_repair_attempts: int = 2,
    ):
        self.provider = provider
        self.output_dir = output_dir
        self.result_filename = result_filename
        self.redactor = redactor
        self.max_repair_attempts = max_repair_attempts

    def run(
        self,
        task: str,
        payload: dict[str, Any],
        output_model: type[T],
        *,
        validator: Callable[[T], None] | None = None,
        accept_after_repair_issue_codes: set[str] | None = None,
        repair_payload_builder: Callable[[str, dict[str, Any], str, list[StructuredIssue], type[T]], dict[str, Any] | None]
        | None = None,
    ) -> tuple[T, ProviderResult]:
        started = monotonic()
        attempts: list[StructuredAttemptRef] = []
        final_result: ProviderResult | None = None
        final_model: T | None = None
        last_issues: list[StructuredIssue] = []
        provider_error: Exception | None = None
        max_attempts = self.max_repair_attempts + 1
        attempt_payload = payload

        for attempt_index in range(1, max_attempts + 1):
            issues: list[StructuredIssue] = []
            errors: list[str] = []
            repair_prompt_ref = self._persist_repair_prompt(attempt_index, attempt_payload) if attempt_index > 1 else None
            latency_ms = 0
            raw = ""
            parsed: dict[str, Any] | None = None
            json_repair_applied = False
            model: T | None = None
            http_attempt_count = 1
            try:
                raw, latency_ms = timed_call(self.provider, task, attempt_payload, output_model)
                http_attempt_count = _provider_http_attempt_count(self.provider)
                try:
                    parsed, json_repair_applied = parse_structured_json_object(raw)
                    model = output_model.model_validate(parsed)
                    if validator is not None:
                        validator(model)
                except ValidationError as exc:
                    issues = _pydantic_issues(exc)
                    errors = [self.redactor.redact_text(str(exc))]
                    model = None
                except ValueError as exc:
                    issues = [_issue("invalid_json", str(exc), repairable=True)]
                    errors = [self.redactor.redact_text(str(exc))]
                    model = None
                except Exception as exc:
                    failed_model = model
                    issue_list = getattr(exc, "issues", None)
                    if isinstance(issue_list, list) and all(isinstance(item, StructuredIssue) for item in issue_list):
                        issues = issue_list
                    else:
                        issues = [_issue("business_validation_failed", str(exc), repairable=False)]
                    errors = [self.redactor.redact_text(str(exc))]
                    if (
                        failed_model is not None
                        and attempt_index >= max_attempts
                        and _issues_are_accepted_after_repair(issues, accept_after_repair_issue_codes)
                    ):
                        model = failed_model
                    else:
                        model = None
            except ProviderError as exc:
                provider_error = exc
                http_attempt_count = max(exc.attempt_count, _provider_http_attempt_count(self.provider))
                error = self.redactor.redact_text(str(exc))
                issues = [_issue(_provider_issue_code(exc), error, repairable=False)]
                errors = [error]

            result = ProviderResult(
                task=task,
                provider=self.provider.name,
                raw_output=self.redactor.redact_text(raw),
                parsed_output=self.redactor.redact(parsed),
                parse_success=parsed is not None,
                json_repair_applied=json_repair_applied,
                schema_valid=model is not None,
                repair_attempted=attempt_index > 1,
                latency_ms=latency_ms,
                payload_char_count=_payload_char_count(attempt_payload),
                http_attempt_count=http_attempt_count,
                errors=errors,
            )
            attempt_ref = self._persist_attempt(task, attempt_index, result, issues, repair_prompt_ref=repair_prompt_ref)
            attempts.append(attempt_ref)
            final_result = result
            last_issues = issues
            if model is not None:
                final_model = model
                break
            if not issues or not all(issue.repairability == "repairable" for issue in issues):
                break
            if attempt_index >= max_attempts:
                break
            custom_repair_payload = (
                repair_payload_builder(task, payload, raw, issues, output_model)
                if repair_payload_builder is not None
                else None
            )
            attempt_payload = custom_repair_payload or _repair_payload(task, payload, raw, issues, output_model)

        if final_result is None:
            raise StructuredOutputError(f"{task} returned no provider result")
        self._persist(task, final_result)
        duration_ms = round((monotonic() - started) * 1000)
        self._persist_report(
            StructuredRepairReport(
                task=task,
                provider=self.provider.name,
                final_outcome="success" if final_model is not None else "failed",
                repair_attempted=len(attempts) > 1,
                max_repair_attempts=self.max_repair_attempts,
                attempt_count=len(attempts),
                repair_count=max(0, len(attempts) - 1),
                duration_ms=duration_ms,
                attempts=attempts,
                final_provider_result_ref=self.result_filename or f"{task}.provider_result.json",
                non_repairable_issues=[issue for issue in last_issues if issue.repairability != "repairable"],
            ),
        )
        if final_model is None:
            message = "; ".join(issue.message for issue in last_issues) or "; ".join(final_result.errors)
            if provider_error is not None:
                raise StructuredOutputError(message) from provider_error
            raise StructuredOutputError(f"{task} returned invalid structured output: {message}")
        return final_model, final_result

    def _persist(self, task: str, result: ProviderResult) -> None:
        if self.output_dir is None:
            return
        filename = self.result_filename or f"{task}.provider_result.json"
        write_json(self.output_dir / filename, result)

    def _persist_attempt(
        self,
        task: str,
        attempt_index: int,
        result: ProviderResult,
        issues: list[StructuredIssue],
        *,
        repair_prompt_ref: str | None = None,
    ) -> StructuredAttemptRef:
        ref = f"provider_results/attempt-{attempt_index}.json"
        if self.output_dir is not None:
            path = self.output_dir / ref
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path, result)
        return StructuredAttemptRef(
            attempt=attempt_index,
            provider_result_ref=ref,
            issues=issues,
            repair_prompt_ref=repair_prompt_ref,
            parse_success=result.parse_success,
            schema_valid=result.schema_valid,
            duration_ms=result.latency_ms,
        )

    def _persist_repair_prompt(self, attempt_index: int, payload: dict[str, Any]) -> str | None:
        if self.output_dir is None:
            return None
        ref = f"repair_prompts/attempt-{attempt_index}.json"
        write_json(self.output_dir / ref, self.redactor.redact(payload))
        return ref

    def _persist_report(self, report: StructuredRepairReport) -> None:
        if self.output_dir is None:
            return
        write_json(self.output_dir / "structured_repair_report.json", report)
        (self.output_dir / "structured_repair_report.md").write_text(render_structured_repair_report_markdown(report), encoding="utf-8")


def render_structured_repair_report_markdown(report: StructuredRepairReport) -> str:
    lines = [
        "# 结构化输出返工报告",
        "",
        f"- 任务：`{report.task}`",
        f"- Provider：`{report.provider}`",
        f"- 结果：`{report.final_outcome}`",
        f"- 尝试次数：{report.attempt_count}",
        f"- 返工次数：{report.repair_count}",
        "",
        "## 尝试记录",
        "",
    ]
    for attempt in report.attempts:
        issue_text = "; ".join(f"{issue.issue_code}: {issue.message}" for issue in attempt.issues) or "无"
        prompt_text = f"；返工 prompt：`{attempt.repair_prompt_ref}`" if attempt.repair_prompt_ref else ""
        lines.append(f"- 第 {attempt.attempt} 次：`{attempt.provider_result_ref}`{prompt_text}；问题：{issue_text}")
    return "\n".join(lines).rstrip() + "\n"


def _parse_json(raw: str) -> tuple[dict[str, Any], bool]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        data = json.loads(text)
        repaired = False
    except json.JSONDecodeError as strict_error:
        repaired_text = _repair_json_like_output(text)
        if repaired_text == text:
            raise strict_error
        try:
            data = json.loads(repaired_text)
        except json.JSONDecodeError as repair_error:
            raise strict_error from repair_error
        repaired = True
    if not isinstance(data, dict):
        raise ValueError("Structured output root must be a JSON object.")
    return data, repaired


def parse_structured_json_object(raw: str) -> tuple[dict[str, Any], bool]:
    return _parse_json(raw)


def _repair_json_like_output(text: str) -> str:
    text = _extract_outer_json_object(text.strip())
    text = _replace_string_array_join(text)
    text = _escape_inner_string_quotes(text)
    text = _remove_trailing_commas(text)
    return text


def _extract_outer_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1].strip()
    return text


ARRAY_JOIN_RE = re.compile(
    r'(?P<prefix>"[A-Za-z_][A-Za-z0-9_]*"\s*:\s*)\[(?P<body>(?:\s*"(?:(?:\\.)|[^"\\])*"\s*,?)+)\]\s*\.join\(\s*"(?P<sep>(?:\\.|[^"\\])*)"\s*\)',
    re.DOTALL,
)


def _replace_string_array_join(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        body = match.group("body")
        sep_raw = match.group("sep")
        try:
            items = json.loads(f"[{body}]")
            sep = json.loads(f'"{sep_raw}"')
        except Exception:
            return match.group(0)
        if not all(isinstance(item, str) for item in items):
            return match.group(0)
        return f"{match.group('prefix')}{json.dumps(sep.join(items), ensure_ascii=False)}"

    return ARRAY_JOIN_RE.sub(replace, text)


def _escape_inner_string_quotes(text: str) -> str:
    output: list[str] = []
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            output.append(char)
            escaped = False
            continue
        if char == "\\" and in_string:
            output.append(char)
            escaped = True
            continue
        if char == '"':
            if not in_string:
                in_string = True
                output.append(char)
                continue
            if _quote_looks_like_string_boundary(text, index):
                in_string = False
                output.append(char)
            else:
                output.append('\\"')
            continue
        if in_string:
            if char == "\n":
                output.append("\\n")
                continue
            if char == "\r":
                output.append("\\r")
                continue
            if char == "\t":
                output.append("\\t")
                continue
            boundary_repair = _missing_string_quote_boundary_repair(text, index) if char in {"}", "]"} else ""
            if boundary_repair == "replace":
                output.append('"')
                in_string = False
                continue
            if boundary_repair == "insert":
                output.append('"')
                output.append(char)
                in_string = False
                continue
        output.append(char)
    return "".join(output)


def _missing_string_quote_boundary_repair(text: str, boundary_index: int) -> str:
    cursor = boundary_index + 1
    saw_line_break = False
    while cursor < len(text) and text[cursor].isspace():
        if text[cursor] in {"\n", "\r"}:
            saw_line_break = True
        cursor += 1
    if cursor >= len(text):
        return "insert"
    if not saw_line_break:
        return ""
    if text[cursor] in {"}", "]"}:
        return "replace"
    if text[cursor] == ",":
        return "insert"
    return ""


def _quote_looks_like_string_boundary(text: str, quote_index: int) -> bool:
    cursor = quote_index + 1
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    if cursor >= len(text):
        return True
    return text[cursor] in {":", ",", "}", "]"}


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text)


def _payload_char_count(payload: dict[str, Any]) -> int:
    try:
        return len(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))
    except TypeError:
        return len(str(payload))


def _provider_http_attempt_count(provider: Provider) -> int:
    try:
        return max(1, int(getattr(provider, "last_http_attempt_count", 1)))
    except (TypeError, ValueError):
        return 1


def _issue(code: str, message: str, *, repairable: bool) -> StructuredIssue:
    return StructuredIssue(
        issue_code=code,
        message=message,
        repairability="repairable" if repairable else "non_repairable",
    )


def _pydantic_issues(exc: ValidationError) -> list[StructuredIssue]:
    issues: list[StructuredIssue] = []
    repairable_types = {
        "missing",
        "extra_forbidden",
        "literal_error",
        "string_type",
        "list_type",
        "dict_type",
        "model_type",
    }
    for error in exc.errors():
        error_type = str(error.get("type", "schema_error"))
        location = ".".join(str(part) for part in error.get("loc", []))
        code = {
            "missing": "missing_field",
            "extra_forbidden": "extra_field",
            "literal_error": "invalid_literal",
        }.get(error_type, "schema_validation_failed")
        issues.append(
            StructuredIssue(
                issue_code=code,
                field_path=location,
                validator_id="pydantic",
                message=str(error.get("msg", error_type)),
                repairability="repairable" if error_type in repairable_types else "non_repairable",
            )
        )
    return issues or [_issue("schema_validation_failed", str(exc), repairable=True)]


def _provider_issue_code(exc: ProviderError) -> str:
    if exc.status_code in {401, 403}:
        return "provider_auth"
    if exc.status_code == 429:
        return "provider_rate_limit"
    return "provider_network"


def _issues_are_accepted_after_repair(issues: list[StructuredIssue], accepted_codes: set[str] | None) -> bool:
    if not issues or not accepted_codes:
        return False
    return all(issue.issue_code in accepted_codes for issue in issues)


def _repair_payload(
    task: str,
    original_payload: dict[str, Any],
    raw: str,
    issues: list[StructuredIssue],
    output_model: type[BaseModel],
) -> dict[str, Any]:
    return {
        **original_payload,
        "repair_contract": {
            "goal": "Rewrite the previous model output as one complete valid JSON object.",
            "task": task,
            "rules": [
                "Return only a complete JSON object matching the schema.",
                "Do not return markdown fences or commentary.",
                "Preserve valid content when possible, but fix every listed issue.",
                "If user-visible content is required to be Chinese, rewrite it in Chinese while retaining stable domain terms.",
                "For zh-CN outputs, do not paste whole English sentences from the source; translate or paraphrase them into Chinese, while keeping stable product/protocol terms in English.",
            ],
            "issues": [issue.model_dump(mode="json") for issue in issues],
            "previous_output_excerpt": raw[:4000],
            "schema": output_model.model_json_schema(),
        },
    }
