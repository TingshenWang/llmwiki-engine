from __future__ import annotations

from typing import Any, Iterable


DEEPSEEK_RMB_PRICING_PER_1M = {
    "deepseek-v4-flash": {
        "prompt_cache_hit_tokens": 0.02,
        "prompt_cache_miss_tokens": 1.0,
        "completion_tokens": 2.0,
    },
    "deepseek-v4-pro": {
        "prompt_cache_hit_tokens": 0.025,
        "prompt_cache_miss_tokens": 3.0,
        "completion_tokens": 6.0,
    },
}

DEEPSEEK_RMB_PRICING_ALIASES = {
    "deepseek-chat": "deepseek-v4-flash",
    "deepseek-reasoner": "deepseek-v4-flash",
    "sensenova-6.7-flash-lite": "deepseek-v4-flash",
}

TOKEN_USAGE_KEYS = [
    "prompt_tokens",
    "prompt_cache_hit_tokens",
    "prompt_cache_miss_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "total_tokens",
]


def extract_token_usage(raw_response: dict[str, Any] | None) -> dict[str, int]:
    if not isinstance(raw_response, dict):
        return _empty_usage()
    usage = raw_response.get("usage")
    if not isinstance(usage, dict):
        return _empty_usage()
    result = _empty_usage()
    for key in ["prompt_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "completion_tokens", "total_tokens"]:
        result[key] = _safe_int(usage.get(key))
    prompt_details = usage.get("prompt_tokens_details")
    if isinstance(prompt_details, dict) and not result["prompt_cache_hit_tokens"]:
        result["prompt_cache_hit_tokens"] = _safe_int(prompt_details.get("cached_tokens"))
    completion_details = usage.get("completion_tokens_details")
    if isinstance(completion_details, dict):
        result["reasoning_tokens"] = _safe_int(completion_details.get("reasoning_tokens"))
    if result["prompt_tokens"] and not result["prompt_cache_miss_tokens"]:
        result["prompt_cache_miss_tokens"] = max(0, result["prompt_tokens"] - result["prompt_cache_hit_tokens"])
    if not result["total_tokens"]:
        result["total_tokens"] = result["prompt_tokens"] + result["completion_tokens"]
    return result


def first_finish_reason(raw_response: dict[str, Any] | None) -> str:
    if not isinstance(raw_response, dict):
        return ""
    choices = raw_response.get("choices")
    if not isinstance(choices, list):
        return ""
    for choice in choices:
        if isinstance(choice, dict):
            return str(choice.get("finish_reason") or "")
    return ""


def api_call_record(
    *,
    step: str,
    model: str,
    attempt: int,
    call_index: int,
    status: str,
    duration_ms: float,
    raw_response: dict[str, Any] | None = None,
    response_status_code: int | None = None,
    error: str = "",
    request_key: str = "",
) -> dict[str, int | float | str]:
    usage = extract_token_usage(raw_response)
    record: dict[str, int | float | str] = {
        "step": step,
        "request_key": request_key,
        "model": model,
        "attempt": attempt,
        "call_index": call_index,
        "status": status,
        "finish_reason": first_finish_reason(raw_response),
        "duration_ms": round(duration_ms, 2),
        "response_status_code": response_status_code or 0,
        "error": error,
        **usage,
    }
    record["cache_hit_rate_percent"] = cache_hit_rate_percent(usage)
    record["price_cny"] = price_cny(model, usage)
    return record


def tag_api_calls(api_calls: Iterable[dict[str, Any]], request_key: str) -> list[dict[str, Any]]:
    tagged = []
    for call in api_calls:
        item = dict(call)
        item["request_key"] = request_key
        tagged.append(item)
    return tagged


def summarize_api_calls(api_calls: Iterable[dict[str, Any]]) -> dict[str, int | float]:
    records = list(api_calls)
    totals: dict[str, int | float] = {
        "api_call_count": len(records),
        "api_success_count": sum(1 for item in records if item.get("status") == "success"),
        "api_paused_count": sum(1 for item in records if item.get("status") != "success"),
    }
    for key in TOKEN_USAGE_KEYS:
        totals[key] = sum(_safe_int(item.get(key)) for item in records)
    totals["cache_hit_rate_percent"] = cache_hit_rate_percent(totals)
    totals["price_cny"] = round(sum(_safe_float(item.get("price_cny")) for item in records), 8)
    return totals


def summarize_step_counts(step_counts: Iterable[dict[str, Any]], *, duration_seconds: float = 0.0) -> dict[str, int | float]:
    records = list(step_counts)
    totals: dict[str, int | float] = {
        "duration_seconds": round(duration_seconds, 2),
        "api_call_count": sum(_safe_int(item.get("api_call_count")) for item in records),
    }
    for key in TOKEN_USAGE_KEYS:
        totals[key] = sum(_safe_int(item.get(key)) for item in records)
    totals["cache_hit_rate_percent"] = cache_hit_rate_percent(totals)
    totals["price_cny"] = round(sum(_safe_float(item.get("price_cny")) for item in records), 8)
    return totals


def cache_hit_rate_percent(usage: dict[str, Any]) -> float:
    prompt_tokens = _safe_int(usage.get("prompt_tokens"))
    if prompt_tokens <= 0:
        return 0.0
    return round(_safe_int(usage.get("prompt_cache_hit_tokens")) / prompt_tokens * 100, 2)


def price_cny(model: str, usage: dict[str, Any]) -> float:
    pricing = DEEPSEEK_RMB_PRICING_PER_1M.get(_pricing_model_key(model))
    if pricing is None:
        return 0.0
    price = (
        _safe_int(usage.get("prompt_cache_hit_tokens")) / 1_000_000 * pricing["prompt_cache_hit_tokens"]
        + _safe_int(usage.get("prompt_cache_miss_tokens")) / 1_000_000 * pricing["prompt_cache_miss_tokens"]
        + _safe_int(usage.get("completion_tokens")) / 1_000_000 * pricing["completion_tokens"]
    )
    return round(price, 8)


def format_price_cny(value: Any) -> str:
    price = _safe_float(value)
    decimals = 4 if price >= 0.01 else 6
    return f"¥{price:.{decimals}f}"


def format_percent(value: Any) -> str:
    return f"{_safe_float(value):.2f}%"


def format_duration_ms(value: Any) -> str:
    duration_ms = _safe_float(value)
    if duration_ms >= 1000:
        return f"{duration_ms / 1000:.2f}s"
    return f"{duration_ms:.0f}ms"


def format_duration_seconds(value: Any) -> str:
    seconds = _safe_float(value)
    if seconds >= 60:
        minutes = int(seconds // 60)
        remainder = seconds - minutes * 60
        return f"{minutes}分{remainder:.0f}秒"
    return f"{seconds:.2f}s"


def _pricing_model_key(model: str) -> str:
    normalized = model.strip().lower()
    normalized = normalized.removesuffix("[1m]")
    if normalized in DEEPSEEK_RMB_PRICING_ALIASES:
        return DEEPSEEK_RMB_PRICING_ALIASES[normalized]
    return normalized


def _empty_usage() -> dict[str, int]:
    return {key: 0 for key in TOKEN_USAGE_KEYS}


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
