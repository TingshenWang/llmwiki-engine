from __future__ import annotations

from llmwiki_engine.lite.token_usage import extract_token_usage, price_cny, summarize_api_calls


def test_extract_deepseek_usage_and_price_cny() -> None:
    raw_response = {
        "usage": {
            "prompt_tokens": 1000,
            "prompt_cache_hit_tokens": 400,
            "prompt_cache_miss_tokens": 600,
            "completion_tokens": 200,
            "total_tokens": 1200,
            "completion_tokens_details": {"reasoning_tokens": 50},
        }
    }

    usage = extract_token_usage(raw_response)

    assert usage["prompt_tokens"] == 1000
    assert usage["prompt_cache_hit_tokens"] == 400
    assert usage["prompt_cache_miss_tokens"] == 600
    assert usage["completion_tokens"] == 200
    assert usage["reasoning_tokens"] == 50
    assert price_cny("deepseek-v4-flash", usage) == 0.001008
    assert price_cny("deepseek-v4-pro", usage) == 0.00301


def test_sensenova_usage_uses_deepseek_flash_pricing_alias() -> None:
    raw_response = {
        "usage": {
            "prompt_tokens": 1000,
            "prompt_tokens_details": {"cached_tokens": 250},
            "completion_tokens": 200,
            "total_tokens": 1200,
        }
    }

    usage = extract_token_usage(raw_response)

    assert usage["prompt_cache_hit_tokens"] == 250
    assert usage["prompt_cache_miss_tokens"] == 750
    assert price_cny("sensenova-6.7-flash-lite", usage) == price_cny("deepseek-v4-flash", usage)
    assert price_cny("sensenova-6.7-flash-lite", usage) == 0.001155


def test_summarize_api_calls_uses_cache_hit_rate_and_total_price() -> None:
    summary = summarize_api_calls(
        [
            {
                "status": "success",
                "prompt_tokens": 100,
                "prompt_cache_hit_tokens": 25,
                "prompt_cache_miss_tokens": 75,
                "completion_tokens": 20,
                "reasoning_tokens": 5,
                "total_tokens": 120,
                "price_cny": 0.000115,
            },
            {
                "status": "paused",
                "prompt_tokens": 300,
                "prompt_cache_hit_tokens": 175,
                "prompt_cache_miss_tokens": 125,
                "completion_tokens": 80,
                "reasoning_tokens": 10,
                "total_tokens": 380,
                "price_cny": 0.000285,
            },
        ]
    )

    assert summary["api_call_count"] == 2
    assert summary["api_success_count"] == 1
    assert summary["api_paused_count"] == 1
    assert summary["prompt_tokens"] == 400
    assert summary["prompt_cache_hit_tokens"] == 200
    assert summary["completion_tokens"] == 100
    assert summary["reasoning_tokens"] == 15
    assert summary["cache_hit_rate_percent"] == 50.0
    assert summary["price_cny"] == 0.0004
