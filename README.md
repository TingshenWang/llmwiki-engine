# llmwiki-engine Lite

Fresh Lite branch for a fully automatic LLM-Wiki ingest core.

The first implementation target is a deterministic local pipeline:

```text
raw_file
-> source_digest
-> wiki_snapshot
-> candidate_pages
-> candidate_contexts
-> merge_plan
-> composition_plan
-> final_pages
-> validation
-> knowledge_write
-> source_record_write
-> index_log_write
-> embedding_cache_refresh
-> receipt
```

There is no manual review, no manual apply, and no raw prepare step in this
branch. The `raw/` file remains the durable evidence layer; generated wiki
pages are a compiled navigation and understanding layer.

`wiki_snapshot` freezes the current knowledge pool and synchronizes the
current-state page embedding cache. `candidate_contexts` then retrieves the
top related old pages for each generated candidate page. The cache is
path-current: the same wiki path has one latest vector record, and stale
records are pruned during cache sync.

```bash
llmwiki init /path/to/vault
llmwiki ingest run /path/to/vault raw/example.md
llmwiki ingest status /path/to/vault --verify
```

## Provider Config

New vaults default to deterministic local heuristics in `.llmwiki/config.yaml`:

```yaml
providers:
  default:
    spec: local:heuristic
```

To use a real OpenAI-compatible chat completions endpoint for model-backed
steps, configure the default provider or a step-specific provider:

```yaml
providers:
  default:
    spec: openai_compatible:gpt-4.1-mini
    endpoint: https://api.openai.com/v1/chat/completions
    api_key_env: OPENAI_API_KEY
    json_mode: json_schema
    json_schema_strict: false
    temperature: 0
    max_tokens: 262144
    max_retries: 2
  merge_plan:
    spec: local:heuristic
```

For DeepSeek, Lite automatically uses `json_object` mode because DeepSeek's
current JSON Output guide documents `response_format: {"type": "json_object"}`
rather than JSON Schema response format:

```yaml
providers:
  default:
    spec: openai_compatible:deepseek-v4-flash
    endpoint: https://api.deepseek.com/v1/chat/completions
    api_key_env: DEEPSEEK_API_KEY
    max_tokens: 262144
```

Supported model-backed step keys are `source_digest`, `candidate_pages`,
`merge_plan`, `composition_plan`, and `final_pages`. Provider prompts and
response schemas are written under each step's `model_calls/` directory; API
keys are not written to artifacts or receipts.

## Local Qwen Embeddings

New vaults default to local Qwen embeddings through Sentence Transformers.
`candidate_contexts` requires this real embedding backend for top-k retrieval;
it does not fall back to lexical, title, exact, or hashing recall.

```json
{
  "embedding": {
    "enabled": true,
    "backend": "sentence_transformers",
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "dimensions": 1024,
    "top_k_pages": 5,
    "max_page_chars": 6000,
    "max_query_chars": 4000,
    "batch_size": 4,
    "query_prompt_name": "query"
  }
}
```

The hashing backend remains available only for low-level deterministic tests.
Ingest candidate recall rejects hashing so the fifth step always uses
embedding vectors.
