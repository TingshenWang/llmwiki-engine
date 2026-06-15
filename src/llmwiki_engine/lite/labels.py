from __future__ import annotations


STEP_LABELS = {
    "raw_binding": "绑定 raw",
    "source_digest": "来源消化",
    "candidate_merge": "候选合并",
    "candidate_pages_warmup": "候选页缓存预热",
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
}

COUNT_LABELS = {
    "raw_size_bytes": "raw大小",
    "candidate_count": "候选数",
    "candidate_unit_count": "候选单元数",
    "warmup_count": "预热数",
    "skipped_candidate_count": "跳过候选数",
    "weak_noise_count": "弱/噪声数",
    "deferred_count": "延后数",
    "knowledge_pool_size": "知识池",
    "candidate_page_count": "候选页数",
    "covered_digest_candidate_count": "覆盖候选数",
    "query_count": "查询数",
    "top_k": "TopK",
    "create_count": "新建数",
    "update_count": "更新数",
    "noop_count": "不改动数",
    "merge_decision_count": "合并决策数",
    "related_kept_count": "相关保留数",
    "related_filtered_count": "相关过滤数",
    "related_link_count": "相关链接数",
    "final_target_count": "最终目标数",
    "final_page_count": "最终页数",
    "update_target_count": "更新目标数",
    "diff_count": "diff数",
    "parallel_request_count": "并发请求",
    "parallel_max_workers": "最大并发",
    "error_count": "错误数",
    "warning_count": "警告数",
    "artifact_count": "产物数",
    "model_calls": "模型调用",
    "repair_count": "修复数",
    "semantic_retry_count": "语义重试数",
    "api_call_count": "API调用",
    "api_success_count": "API成功",
    "api_paused_count": "API暂停",
    "prompt_tokens": "输入token",
    "prompt_cache_hit_tokens": "命中缓存token",
    "prompt_cache_miss_tokens": "未命中缓存token",
    "completion_tokens": "输出token",
    "reasoning_tokens": "思考token",
    "total_tokens": "总token",
    "cache_hit_rate_percent": "缓存命中率",
    "price_cny": "价格",
    "knowledge_written_count": "知识页写入数",
    "source_record_count": "来源页数",
    "system_written_count": "系统页数",
    "written_target_count": "写入目标数",
    "receipt_count": "回执数",
    "cache_hit": "缓存命中",
    "cache_refreshed": "缓存刷新",
    "cache_pruned": "缓存清理",
    "retrieval_backend": "召回后端",
}


def step_label(name: str) -> str:
    return STEP_LABELS.get(name, name)


def count_label(key: str) -> str:
    return COUNT_LABELS.get(key, key)
